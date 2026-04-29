# Omni T2A Dataloader — deep-dive notes

Working notes from a code-reading / debugging session on the T2A (text-to-audio)
training data loader. Covers: the bridge → koba-v2 migration, how koba v2 shards
a Lance table, what `shuffle_ranges` controls, a worked example of
"content-drift" in W&B run `52emg8ek`, and a debug case study where a reverted
`list<binary>` unwrap silently stalled every training job that used whisperx
datasets.

Primary files referenced:

- `projects/kuma/kuma/projects/omni/audio/configs/data/omni_t2a_packing_koba_v2.py`
- `lib/koba/koba/v2/core/sampler.py` (`WeightedMultiSourceSampler`)
- `lib/koba/koba/pipelines/default_t2a.py`
- `lib/koba/koba/processor/audio_ops.py`
- `lib/koba/koba/processor/omni_audio_ops.py`


## 1. Where T2A training gets its data today

All currently-live T2A experiment configs (the `_formal_koba_v2_*` family in
`projects/kuma/kuma/projects/omni/audio/configs/t2a.py`) run a single dataset
path:

```
LanceReader → V1PipelineAdapter (decode → normalize → tokenize) → PackingDataset
```

`OmniT2APackingConfigKobaV2` is the only Python-visible dataset class in
`configs/data/` after the T2A-v2 cleanup. Earlier incarnations were:

1. **V1 `OmniT2ADatasetConfig`** (koba v1 `AllModalityDatasetWithMultithreading`) —
   original path; had a known memory leak from the multithreading feeder.
2. **Bridge `OmniT2ABridgeDatasetConfig`** — "Option B" pragmatic stopgap that
   reused the already-proven `AudioBatchingDatasetV2` from the Ray3 T2A stack and
   bolted on a thin adapter that rebuilt each flat batch into the omni packed
   format (`[TEXT(caption), AUDIO(audio_tensor)]`), then ran
   `pack_sequence()` on a background prefetch thread. Its reason to exist was
   *parity with Ray3 T2A reference runs* (same duration buckets, peak
   normalization, clamping) while avoiding the V1 memory leak.
3. **Koba v2 native `OmniT2APackingConfigKobaV2`** — final replacement. Does the
   whole decode→tokenize→pack in a single pass over each Lance row, reuses the
   same koba v2 `PackingDataset` as the Omni T2I V2 path, and has no memory
   leak. Supersedes both predecessors.

Both V1 and the bridge were deleted once all active experiment configs moved to
the koba-v2 native path; the bridge's duration-bucket / peak-norm behavior is
reproduced by `OmniT2APackingConfigKobaV2`'s built-in parameters.


## 2. How koba v2 shards a Lance table

`OmniT2APackingConfigKobaV2` hands the per-modality sampling job to
`WeightedMultiSourceSampler` in `lib/koba/koba/v2/core/sampler.py`. Understanding
the shape of the shards is essential for understanding the `shuffle_ranges`
knob.

Setup:

- `total_rows = N` rows in the Lance table
- `dp_world_size = W` data-parallel workers (dp_ranks × num_workers)
- `batch_size = B` rows per batch

The sampler carves the table into **interleaved** shards, not contiguous slabs.
Define **stride** `= W × B`. For rank `r`, range index `i` starts at row

```
start(i, r) = r * B + i * (W * B)
```

so each worker's sequence of ranges is an arithmetic progression of
`B`-row-wide contiguous blocks, separated by a full stride. Example with
`W = 16, B = 32`:

| Worker | Its ranges (row spans) |
|---|---|
| worker 0 | `[0, 32)` → `[512, 544)` → `[1024, 1056)` → `[1536, 1568)` → … |
| worker 1 | `[32, 64)` → `[544, 576)` → `[1056, 1088)` → `[1568, 1600)` → … |
| worker 2 | `[64, 96)` → `[576, 608)` → `[1088, 1120)` → … |
| … | … |
| worker 15 | `[480, 512)` → `[992, 1024)` → `[1504, 1536)` → … |

Every 512-row stride is tiled exactly by one batch from each of the 16 workers.
If the table is ~1 M rows, each worker has ~1953 ranges per epoch.

Two consequences worth keeping in mind:

- **Reads are contiguous:** each range is a 32-row `scanner.take(range)`, which
  Lance serves as a sequential scan. I/O is cheap.
- **Workers are tightly interleaved across the table:** no worker owns a
  contiguous slab. This matters for shuffling semantics (see next section).


## 3. What `shuffle_ranges` actually decides

`shuffle_ranges` is a `PackingConfig` flag threaded through to
`WeightedMultiSourceSampler.Config.shuffle_ranges` (default `True` for
production training). It only controls **the order in which each worker visits
its own list of ~1953 ranges** — nothing about which rows a worker gets.

### `shuffle_ranges=False` (natural order)

Worker `r` visits range 0, then 1, then 2, … in ascending order. Worker 0's
trajectory:

```
t=0: [0, 32)     t=1: [512, 544)     t=2: [1024, 1056)   …   t=k: [512k, 512k+32)
```

All 16 workers advance in lockstep: at step `k`, their row-block starts are
`[0, 32, 64, …, 480] + 512·k`, which **all land within the same 512-row
stride**. Visualize it as a ruler of width 512 sweeping the table left-to-right
as step count grows — the model, at any single step, is sampling from one
narrow region of the table at a time.

### `shuffle_ranges=True`

Each worker independently permutes its ~1953 range indices using
`seed + rank + 1000` (so every rank gets a different permutation — see
`sampler.py:185-187`). Worker 0 might go `range_1057 → range_42 → range_1893 → …`
while worker 1 goes through a completely different permutation.

At any given step the 16 workers are **scattered across 16 random strides of
the table**. The group's reads span the whole table from step 1.

### What stays the same either way

- Each range is still a 32-row contiguous block → Lance reads remain sequential.
- The sharding boundaries don't change → no duplication, no change in total data
  volume.
- Per-epoch coverage is identical — only the within-epoch order differs.
- Memory overhead of the shuffle is minimal: one `int32` permutation array per
  worker per dataset (~8 KB for ~1953 ranges).

So it's a near-free knob that trades "deterministic sweep through the table"
for "stationary sampling across the table." Shuffle implementation lives in
`sampler.py:324-330` and the epoch-end reshuffle is in `sampler.py:384-388`.


## 4. Why natural order produces correlated content dips

Lance tables like `whisperx__multilingual_v1_compacted.lance` are built by
compaction of upstream shards in a particular order. That order is almost never
globally random — consecutive fragments tend to share:

- **Source** (audiobook batch X for N fragments, then podcast batch Y for M
  fragments, …)
- **Duration distribution** (long read-aloud vs short conversational)
- **Language / speaker** if upstream shards were organized that way
- **Encoding / quality tier** when the same ingestion run produced a block

With `shuffle_ranges=False`, because all 16 workers are simultaneously in the
same 512-row stride, **they all see the same local content regime at once**:

- Early steps (stride 0–1000): maybe all workers see 30-s audiobook clips →
  each packed sequence fits only 2–3 samples → `num_samples` dips low across
  all ranks together.
- Mid steps (stride 1000–5000): maybe all workers hit conversational 5-s clips
  → each pack fits 20+ samples → `num_samples` spikes high across all ranks.
- Later: next region's mix → another synchronized swing.

The time series of `num_samples` (and anything that depends on it —
throughput, per-sample loss distribution, gradient variance) tracks the table's
internal layout as the ruler sweeps through it.

With `shuffle_ranges=True`, at step 1 some workers are already in the audiobook
region and others in the conversational region. The per-step mixture across
workers approximates the table's global mixture from the start, and metrics
stay roughly stationary over step count rather than tracking the table's
physical layout.


## 5. Case study: W&B run `52emg8ek` — the step 28K–32K dip

[Run URL](https://wandb.ai/luma-ai/omni-t2a/runs/52emg8ek)

Between step ~28K and ~32K, `num_samples` per step dropped significantly. The
drop coincided with a rise in average audio duration per sample.

Mechanism, with `W = 16, B = 32` and `max_num_tokens = 16_000` (the run's
packed-sequence budget):

1. At step `k`, all workers are reading from stride
   `[512k, 512k + 512)` of the Lance table.
2. For `k ≈ 28_000–32_000`, the corresponding row ranges are roughly
   `[14.3 M, 16.4 M)`.
3. That row region of `whisperx__multilingual_v1_compacted.lance` happens to
   contain a contiguous block of longer clips (an audiobook batch, a long-form
   podcast source, or similar — the exact source doesn't matter, only that the
   fragment ordering kept them contiguous at compaction time).
4. Longer clips consume more of each packed sequence's 16 K token budget, so
   fewer samples fit per pack → `num_samples` drops.
5. Because all 16 workers are in the same stride, the drop is **correlated
   across ranks** and shows up as a coordinated trough in the averaged
   metric — not noise, not a data-pipeline bug, not model divergence.
6. Once the sweep exits that region (step `k > 32K`), durations revert to
   whatever the next stride holds → `num_samples` recovers.

That shape — monotonic, correlated across workers, synchronized with step
count, traceable to a specific row band — is the signature of the problem
`shuffle_ranges=True` was introduced to fix.


## 6. Note on the term "content-drift"

"Content-drift" is not a standard ML term; it's a shorthand I've been using in
these discussions. Worth a few clarifying notes so future-me doesn't mistake it
for something it isn't.

### Why "drift" feels natural

In ML, **drift** generally describes a distribution that changes over time
(data drift, concept drift, covariate shift). From the optimization loop's
perspective — which only sees "what's in this batch right now" — the fact that
the **per-step empirical distribution** is changing with step count is
indistinguishable from a drifting data source. The underlying dataset is
static, but the view the model trains on at time `t` does differ from the view
at time `t'`.

### Why "content-" qualifier

It specifies *what* is drifting: the content characteristics (audio duration,
language, speaker, source encoding) rather than labels or a concept boundary.
It also signals that the cause is the **sampling order over a fixed dataset**,
not changes to the data itself — which would be the usual meaning of "data
drift."

### Where the term is loose

- Strictly, "drift" in statistics implies non-stationarity. Here the
  distribution is stationary in the long run (the table is what it is); if you
  integrate over a full epoch you get the global distribution regardless of
  `shuffle_ranges`.
- So this is better described as **local non-stationarity induced by
  structured, correlated sampling** — not true drift.
- More precise names would be "sampler-induced batch homogeneity" or
  "correlated-content sweep," but those are mouthfuls.
- Internally, this phenomenon is sometimes just called a "content dip" or
  "distribution wave" when looking at metric plots.

### Canonical phrasing for write-ups

> "Non-shuffled sampler ordering causes correlated content across workers,
> producing step-correlated dips in `num_samples`."

or, terser:

> "The sampler swept into a long-clip region of the table; `shuffle_ranges=True`
> decorrelates worker trajectories so the per-step distribution tracks the
> global mixture instead of the table's physical layout."


## 7. Debug diary: a silent dataloader stall from reverted `list<binary>` handling

Included because it's instructive: it shows how a small, benign-looking
cleanup commit can silently take down every T2A job that uses whisperx data,
and how to recognize the symptom quickly.

### Symptom

Two T2A Flyte jobs (`ii6b0m5j` and `jv9mfs6c`) launched in quick succession
both hung with the following signature:

- Flyte status: **Running** (pods healthy, heartbeat active).
- `flytecli wtf`: no traceback, no failure status in Flyte DB; only an
  informational `SetPodTemplateRestartPolicy` k8s event (benign).
- `flytecli analyze-last-lines`: all 16 ranks at **exactly the same** last log
  line: `I: next_batch.wait_for_data`, at the same timestamp — perfect
  inter-rank consistency.
- `flytecli dataloader-errors --dataloader-rank 0`: "No dataloader worker
  traces found."
- W&B: run created, `running` state, but zero training steps ever logged.

This shape — healthy infra, zero crashes, all ranks synchronized on
`next_batch.wait_for_data` — is diagnostic of **all samples being silently
dropped upstream of the packer**.

### Root cause

Commit `2c26846d05` "[omni] Revert audio_ops.py to match main" removed ~20
lines from `AudioDecoder.forward()`:

```python
if isinstance(audio_bytes, list):
    if len(audio_bytes) == 0:
        return None
    if len(audio_bytes) > 1:
        raise ValueError(...)
    audio_bytes = audio_bytes[0]
```

Whisperx-annotated Lance tables (e.g. `internal-audio-v2-english`,
`internal-audio-v2`, anything under
`s3://.../audio/pretrain/podcast_10m/asr/whisperx__*.lance`) store
`audio_bytes` as a single-element `list<binary>` column. Without the unwrap,
control flow becomes:

```python
audio_bytes = sample[self.config.audio_key]     # list[bytes]
...
decoder = torchcodec.decoders.AudioDecoder(
    source=io.BytesIO(audio_bytes),              # TypeError: list → BytesIO
    ...
)
```

The `TypeError` is caught by the broad `except Exception: return None` at the
bottom of `forward()`. Every sample returns `None`. The packing dataset
forever awaits enough valid samples to fill a packed sequence, and the
training loop hangs in its very first `next(data_iterator)` call — before any
step is logged.

### Why it evaded detection in the earlier `gayckhf8` run

`gayckhf8` launched 2026-04-15 with the list-handling still present (added in
`af3c4c96fe` on 2026-04-09). The revert happened 2026-04-20, so any run after
that against a whisperx table hits this bug.

### Fix

Restore the list-unwrap block in `AudioDecoder.forward()` (committed as
`1bf30b85e0`). Runs launched post-fix (W&B `xicx8wqh` in this thread) produced
training-loss / grad-norm metrics within ~3 min, confirming the dataloader
was unblocked.

### Takeaways for future debugging

- "Healthy Flyte status + all ranks on `next_batch.wait_for_data` + zero
  training steps" → always suspect the upstream pipeline is silently
  dropping samples.
- `analyze-last-lines` showing **all ranks at the exact same message at the
  exact same time** is a much stronger signal than any single rank's log.
- A bare `except Exception: return None` in a processor is a classic
  silent-drop hazard. Either log the exception at `WARNING`, or let specific
  exceptions propagate so the dataloader crashes loudly instead of stalling.
- Schema drift across Lance datasets (scalar vs. `list<binary>` for
  `audio_bytes`, different transcript column names, etc.) deserves a handful
  of schema-compatibility tests up front.


## 8. `attention_mode` and the per-token stream masks

Two orthogonal axes on every `TokenizedSequenceElement` decide what each token
"is" inside a packed sample. They are independent and routinely confused; both
are produced by the data pipeline (`OmniElementVAEAudio` /
`OmniElementVAEImage` / `OmniElementText` / `OmniElementVit`) and consumed by
the trainer at attention time.

- **`attention_mode`** — set per `SequenceElement` (one string for the whole
  element), governs the **attention visibility pattern** for every token in
  that element.
- **`text_token_mask` / `vae_token_mask` / `clean_vae_token_mask` /
  `noisy_vae_token_mask` / `txt_loss_mask`** — set per token, govern **which
  projection / adapter weights process the token at each layer** (the
  "stream") and where the text-side CE loss applies.

Stream routing decides what the token feeds through; `attention_mode` decides
who can see it. They run on different subsystems and don't constrain each
other — a token can be on the text stream and inside a `noise` block, or on
the VAE stream and inside a `full` block, with no contradiction.

### 8.1 The three attention modes

Defined and assembled in
`lib/ursa/ursa/models/omni/inference/sequence_packing.py` by
`prepare_attention_mask_per_sample(split_lens, attn_modes, device)`. The
function takes the per-element `(length, mode)` pairs of one packed sample
and returns the full `(sample_len, sample_len)` additive mask (`0.0` where
attention is allowed, `-inf` where masked). The three modes:

- **`causal`** — standard autoregressive mask within the segment (lower
  triangular self-block) + full visibility to all prior tokens (the row
  slice `[csum:csum+s, :csum]` is set to 1). Used for text spans
  (`OmniElementText.attention_mode = "causal"`,
  `lib/koba_shared/koba_shared/processor/omni_text_ops.py:45`) and ViT image
  spans (`OmniElementVit`,
  `lib/koba_shared/koba_shared/processor/omni_vit_ops.py:50,61` — note ViT
  switches between `"causal"` and `"full"` per its own logic).
- **`full`** — fully bidirectional self-block + full visibility to all prior
  tokens. Used for **clean** VAE elements: `CLEAN_VAE_AUDIO` /
  `CLEAN_VAE_IMAGE` →
  `attention_mode = "full"` (`omni_audio_ops.py:190`, `omni_vae_ops.py:97`).
- **`noise`** — fully bidirectional self-block + full visibility to all
  prior tokens, **AND** the entire column slice covering this segment is
  zeroed for every row outside the segment, so downstream tokens cannot
  attend to anything in this segment. Used for **noisy** VAE elements:
  `NOISY_VAE_AUDIO` / `NOISY_VAE_IMAGE` →
  `attention_mode = "noise"` (`omni_audio_ops.py:204`, `omni_vae_ops.py:111`).
  This is the diffusion-isolation rule: noisy latents must not leak into the
  LM stream.

The actual algorithm is two passes, simplified:

```python
# pass 1: per-segment self/prior block
for s, mode in zip(split_lens, attn_modes):
    if mode == "causal":
        mask[csum:csum+s, csum:csum+s] = tril(ones(s, s))   # lower triangular self
    else:
        mask[csum:csum+s, csum:csum+s] = ones(s, s)         # bidirectional self
    mask[csum:csum+s, :csum] = 1                            # see all prior tokens
    csum += s

# pass 2: noise isolation overrides outbound visibility
for s, mode in zip(split_lens, attn_modes):
    if mode == "noise":
        mask[:, csum:csum+s] = 0                            # nobody sees noisy tokens
        mask[csum:csum+s, csum:csum+s] = 1                  # but the segment sees itself
    csum += s
```

For a noisy element of length `L` placed at positions `[csum, csum+L)`:

- It **reads** all tokens before `csum` (pass-1 line that sets
  `[csum:csum+L, :csum] = 1`).
- Its self-block `[csum, csum+L) × [csum, csum+L)` is fully bidirectional.
- Its **outbound** column `[:, csum:csum+L]` is zeroed in pass 2 for every
  row that is not inside the segment itself — so any token after `csum+L`
  cannot attend to it. Registers and audio_pad tokens in the same noisy
  element CAN attend to each other (the self-block is restored in the same
  pass).

Pass 1 also applies to the noise segment first (giving it `ones(s, s)` self
and a row of 1s into prior tokens), and pass 2 then clears the outbound
column. That ordering is what makes registers see the prompt while still
being invisible to anything downstream.

Special case in `pack_sequence`: when `pad_to_length` is set (flex-attention
path), the trailing `[pad_length]` block is always appended with mode
`"causal"` (`sequence_packing.py:547,552`) so the padding doesn't leak
visibility back into real content.

### 8.2 The stream masks (per-token routing)

Set per token by the per-element processors in
`lib/koba_shared/koba_shared/processor/omni_audio_ops.py` (`OmniElementVAEAudio`,
lines 113–204) and `omni_vae_ops.py` (`OmniElementVAEImage`, lines ~50–111).
For a noisy audio element with `M` register tokens and `N` audio_pad tokens
(`num_tokens = M + N + 2`):

| Position | Token (current default string)             | `text_token_mask` | `vae_token_mask` | `clean_vae_token_mask` | `noisy_vae_token_mask` | `txt_loss_mask` |
| -------- | ------------------------------------------ | ----------------- | ---------------- | ---------------------- | ---------------------- | --------------- |
| 0        | `<\|vision_start\|>` (audio_start)          | 1                 | 0                | 0                      | 0                      | 0               |
| 1..M     | `<\|endoftext\|>` (register)                | 1                 | 0                | 0                      | 0                      | 0               |
| M+1..M+N | `<\|image_pad\|>` (audio_pad / frame slot)  | 0                 | 1                | 0                      | 1                      | 0               |
| M+N+1    | `<\|vision_end\|>` (audio_end)              | 1                 | 0                | 0                      | 0                      | 0               |

For a clean audio element (no registers, `num_tokens = N + 2`):

| Position | Token                                       | `text_token_mask` | `vae_token_mask` | `clean_vae_token_mask` | `noisy_vae_token_mask` | `txt_loss_mask` |
| -------- | ------------------------------------------- | ----------------- | ---------------- | ---------------------- | ---------------------- | --------------- |
| 0        | `<\|vision_start\|>`                         | 1                 | 0                | 0                      | 0                      | 0               |
| 1..N     | `<\|image_pad\|>`                            | 0                 | 1                | 1                      | 0                      | 0               |
| N+1      | `<\|vision_end\|>`                           | 1                 | 0                | 0                      | 0                      | 0               |

What each mask drives at runtime:

- **`text_token_mask`** — token feeds through the LM's embedding table
  (`get_input_embeddings()`), no VAE projection. Boundaries (start/end) and
  registers are text-stream because they're meaningful as token ids; the
  encoder/decoder doesn't synthesize their representations.
- **`vae_token_mask`** — token's input embedding is *replaced* by the audio
  (or image) VAE encoder output at the corresponding frame slot, and the
  noise prediction head writes back into the same positions on the noisy
  branch. The boolean is the union of clean+noisy; the per-branch masks
  narrow which prediction head reads from / writes to that slot.
- **`clean_vae_token_mask`** — slot is conditioning context for diffusion
  (clean latents from the encoder; no loss applied). Set 1 inside a CLEAN
  VAE span, 0 elsewhere.
- **`noisy_vae_token_mask`** — slot is a noisy latent the diffusion head
  must denoise; the diffusion loss is computed at these positions. Set 1
  inside a NOISY VAE span, 0 elsewhere.
- **`txt_loss_mask`** — text-side cross-entropy applies only where this is 1.
  It is **zero across the entire VAE span on both branches** — audio/image
  tokens never get text CE loss (`omni_audio_ops.py:164–166`,
  matching the image side). Set on the assistant-text positions in
  `OmniElementText`.

Note the loose duck-typing convention in `OmniElementVAEAudio`: the start /
end markers have `vae_token_mask=1` initially (the whole span is set to ones
at line 159–161), then positions 0 and `-1` are flipped to 0 explicitly
(lines 162–163). Registers are toggled in a follow-up block (lines 173–178):
`vae_token_mask[1:1+M] = 0`, `text_token_mask[1:1+M] = 1`. The end result is
the table above; the construction order is "set everything as VAE, then carve
out the text-stream tokens."

`padding_mask` is set later by `OmniQwen3Tokenizer` for every element in the
same pass, so it does not appear in the per-element processor output.

### 8.3 Span structure with registers (for reference)

`OmniElementVAEAudio.forward` (`omni_audio_ops.py:143–149`) builds the
text_str by concatenation:

```python
tokens = [audio_start_token]                                       # 1
if NOISY_VAE_AUDIO:
    tokens += [audio_register_token] * audio_register_token_amount # M (only on noisy)
tokens += [audio_pad_token] * num_audio_tokens                     # N
tokens += [audio_end_token]                                        # 1
```

So the per-branch span layouts are:

```
NOISY:  <audio_start> <register>×M <audio_pad>×N <audio_end>     attention_mode = "noise"
CLEAN:  <audio_start>               <audio_pad>×N <audio_end>     attention_mode = "full"
```

`num_audio_tokens = ceil(num_frames / compression_factor)` where
`compression_factor` is the encoder's hop length (512 for MMAudio at 16 kHz,
960 for Hunyuan DAC). For a 5 s clip @16 kHz with compression_factor=512:
`num_frames = 80000`, `num_audio_tokens = 157`. Total noisy element length
with 16 registers would be `16 + 157 + 2 = 175`.

#### Token strings — current defaults vs proposed POC swap

The audio defaults in `OmniElementVAEAudio.Config` (lines 64–68) currently
reuse vision-side Qwen3 specials. The class docstring marks this as
temporary: *"Qwen3 has no dedicated audio special tokens, so
boundary/register/pad tokens reuse the vision-side analogs (all single-token
specials in Qwen3). Flip these once the tokenizer gains dedicated audio
tokens."*

| Role             | Current default (image reuse)               | After POC tokenizer swap                        |
| ---------------- | ------------------------------------------- | ----------------------------------------------- |
| Span open        | `<\|vision_start\|>` (id 151652)             | `<\|audio_start\|>` (id 151669)                  |
| Span close       | `<\|vision_end\|>`   (id 151653)             | `<\|audio_end\|>`   (id 151670)                  |
| Frame placeholder| `<\|image_pad\|>`    (id 151655)             | `<\|audio_pad\|>`   (id 151676)                  |
| Register slot    | `<\|endoftext\|>`    (id 151643)             | unchanged (amount=0; never emitted today)       |

The 36 audio specials at ids 151669–151704 in the merged
Qwen3-VL-2B-Audio-POC tokenizer are described in the `qwen3_audio_graft_poc`
notes (§1.2). The validation contract in
`projects/kuma/kuma/projects/omni/audio/data/tokenizer_validation.py` (lines
31–34) requires every audio-role token string to tokenize to **exactly one**
id — so the POC tokenizer is the prerequisite for flipping defaults; the
current Qwen3-0.6B tokenizer doesn't have these strings.

The structure (start / registers / pads / end) does not change with the
swap; only the four token id constants do.

### 8.4 Why registers are noisy, not their own mode

Registers ride the noisy element's `attention_mode = "noise"` because they
sit inside that element. There is no "register" attention mode. Functionally:

- They are **inbound-readable** within the noisy span (other registers,
  audio_pads, boundary tokens all see them via the bidirectional self-block)
  and from prior context (text prompt, conditioning latents — anything
  whose token index is `< csum`, via the `[:, :csum]` row in pass 1).
- They are **outbound-invisible** to any token after the element ends
  (pass-2 column zero-out).

That outbound invisibility is the key property: registers accumulate
diffusion-conditioned summary signal that downstream tokens never read
directly. Whatever escapes the noisy block does so only through the residual
stream of *prior-context* tokens that registers have attended to — i.e.
through the LM's own representations of the conditioning text, never through
the register tokens themselves.

The register **id** is `<|endoftext|>` (the same id repeated `M` times) by
deliberate choice. The image stack uses `VIS_REG_TOKEN = "<|endoftext|>"`
literally everywhere this token is referenced
(`lib/ursa/ursa/models/omni/inference/tokenization.py:25`,
`lib/ursa/ursa/models/omni/inference/constants.py:7`, and the eight
per-task inference processors:
`projects/kuma/kuma/projects/omni/bagel/inference/processor/{t2i,i2t,vlm,image_edit,multiview,siso,interleaved,storyboard}_processor.py`).
Registers are not content tokens; their initial embeddings should be
identical and the model differentiates them by position + attention. Using a
content-bearing id would seed the registers with that content's prior, which
is the opposite of what we want — they exist to be a learnable summary, not
to inject lexical information.

### 8.5 Where the image-token reuse is wired (single source of truth)

There is exactly one place in the omni-t2a stack that maps audio span tokens
onto image / vision tokens — the defaults of `OmniElementVAEAudio.Config` in
`lib/koba_shared/koba_shared/processor/omni_audio_ops.py:64–68`:

```python
audio_start_token: str = VISION_START_TOKEN          # "<|vision_start|>"  id 151652
audio_end_token:   str = VISION_END_TOKEN            # "<|vision_end|>"    id 151653
audio_register_token: str = QWEN3_PAD_TOKEN          # "<|endoftext|>"     id 151643  (unused: amount=0)
audio_pad_token:   str = QWEN3_IMAGE_PAD_TOKEN       # "<|image_pad|>"     id 151655
```

These flow into the pipeline via:

1. `OmniElementVAEAudio` — the per-element processor that builds the span
   text_str and freezes `num_audio_tokens` from the audio tensor's length.
2. `default_t2a_pipeline_processors` (`lib/koba/koba/pipelines/default_t2a.py:110`)
   — the chain composer that instantiates `OmniElementVAEAudio.Config(...)`.
3. `tokenizer_validation.py:31–34` — the validation contract that asserts
   every audio-role token string tokenizes to a single id; this is the gate
   that would fail if the strings are flipped before the tokenizer has the
   matching specials.

No code in `projects/kuma/kuma/projects/omni/audio/` overrides these
defaults, and there are no other hardcoded `image_pad` / `vision_start`
references anywhere in that subtree. Flipping the defaults is a single
config-override edit at the call site in `default_t2a.py`, plus mirroring
the same three id substitutions in
`projects/kuma/kuma/projects/omni/audio/inference/processor/t2a_processor.py`
(which builds the audio span by hand for inference).

### 8.6 Image-side production reference

For comparison: the image stack actively uses 16 register tokens in
production, with the same `<|endoftext|>` id. From
`projects/kuma/kuma/projects/omni/bagel/`:

- T2I:        `vision_register_token_amount=16` (`inference/processor/t2i_processor.py:192`, `datasets/registry_utils_8b.py` 4 sites)
- I2T:        `vision_register_token_amount=16` (`inference/processor/i2t_processor.py:161`)
- VLM:        `vision_register_token_amount=16` (`inference/processor/vlm_processor.py:156`)
- image_edit: `vision_register_token_amount=16` (`inference/processor/image_edit_processor.py:148`)
- (other tasks pass `vision_register_token=VIS_REG_TOKEN` but inherit
  `vision_register_token_amount=0`.)

Audio side currently inherits the framework default
`audio_register_token_amount=0`, so registers are never emitted on audio
spans today. If/when we enable them, mirroring the image convention
(`audio_register_token=<|endoftext|>`, `amount=16`) is the path of least
surprise. The POC tokenizer doesn't add a dedicated audio register and
inventing one wouldn't gain anything, because all `M` register slots
collapse to the same id by design.

### 8.7 Should vision and audio share `<|endoftext|>` as the register id?

The current design uses the same id (151643, `<|endoftext|>`) for vision and
audio register slots. This is reasonable as a POC / single-modality choice
and starts to look slightly worse in true mixed-modality settings. It is not
a bug, but it is also not the design we'd pick if we were starting fresh.

**Where it's clearly fine.**

- Registers are differentiated by *position + attention*, not by id. All `M`
  register slots inside a single noisy block already share the same id —
  that's by design (see §8.4). Whether the next modality's block reuses the
  same id or a different one doesn't change what the model has to learn from
  positional context anyway.
- Attention isolation eliminates cross-modality leakage. Image and audio
  noisy blocks are each `attention_mode="noise"`, so their outbound columns
  are zeroed (§8.1 pass 2). A register in an image block can't be read by an
  audio register, and vice versa. Sharing the id doesn't break the
  isolation.
- The pad-token contention concern is essentially zero. `<|endoftext|>` is
  also `QWEN3_PAD_TOKEN`, but in any reasonable training setup pads are
  attention-masked, so the pad usage contributes no gradient to the
  embedding row. The register usage dominates, which means id 151643's row
  effectively trains as "I am a register summary slot."

**Where it gets worse.**

- *Mixed-modality gradient contention (mild).* When a single packed sample
  contains both an image noise block (16 registers) and an audio noise block
  (`M` registers), id 151643's embedding row receives gradient from two
  different summary roles in the same backward pass. The row settles at a
  compromise that has to support both "summarize image latents conditioned
  on prior text" and "summarize audio latents conditioned on prior text."
  The model can recover via positional context (surrounding `image_pad`
  vs `audio_pad`-stream tokens are different), but you spend capacity to
  do so.
- *No room for modality-specialized priors.* A dedicated `<|audio_register|>`
  could specialize toward audio-summary statistics (rhythm, prosody,
  spectral energy) while `<|vision_register|>` specializes toward
  image-summary statistics (composition, color, spatial layout). Sharing
  forces a generic prior. With unrelated modalities (audio is acoustically
  far from images) this is a worse fit than with closely related ones
  (image + video).
- *Conceptual hygiene / readability.* A reader scanning a tokenized sequence
  and seeing `<|vision_start|> <|endoftext|>×16 <|image_pad|>×N <|vision_end|>`
  has to know the convention to interpret `<|endoftext|>` as "register"
  rather than "EOS / pad." The semantics are role-determined entirely by
  surrounding context. A dedicated `<|register|>` would self-document.

**Where it would actively hurt.**

- *If a future training recipe ever has the assistant generate a register
  block as part of its output.* The shared id would create a real ambiguity
  — the model can't be both "predicting end of document" and "predicting
  register slot 0" with the same logit. Today this doesn't happen
  (registers live inside noisy VAE elements that the LM doesn't generate
  token-by-token), but it's a footgun for future designs.

**Recommendation, tiered by use case.**

| Use case                                                     | Recommendation                              |
| ------------------------------------------------------------ | ------------------------------------------- |
| Single-modality T2A, registers off (today)                   | Keep shared. Zero practical cost.           |
| Single-modality T2A with registers enabled (`amount=16`)     | Keep shared. Image-side interference doesn't apply. |
| Mixed-modality omni (image + audio in one pack, both with registers enabled) | Introduce dedicated register tokens. The Qwen3-VL-2B-Audio-POC tokenizer has 230+ unused slots in `[151705, 151935]` (per `qwen3_audio_graft_poc.md` §1.2). Adding `<|audio_register|>` and `<|vision_register|>` is a tokenizer-config-only change with near-zero risk. |

The original choice of `<|endoftext|>` was likely driven by (a) it was
pad-shaped — no content prior in the base model, fresh embedding row to
specialize, (b) it required no tokenizer changes, and (c) registers were
originally an image-only feature, so the cross-modality concern didn't exist
when the choice was made. None of those reasons survive contact with the
audio extension; they're artifacts of the historical sequencing, not
principled justifications. Worth flipping the next time we touch the
tokenizer.


## 9. What governs a token's behavior — the full picture

Sections 6–8 cover individual axes; this section assembles the complete set.
Every token in a packed sample has its forward / backward behavior
determined by the combination of the fields below. They are mostly
independent, set at different granularities (per token, per element, per
sample), and produced at different stages of the data pipeline.

### 9.1 Stream routing — which forward-stream weights process the token

Three top-level boolean masks select which "processor" (sub-network with its
own embedding table, RoPE, and projection weights) handles the token at
every layer. Code at
[lib/ursa/ursa/models/omni/model/model.py:560-672](lib/ursa/ursa/models/omni/model/model.py#L560-L672):

```python
text_position_ids   = position_ids[text_token_mask]       # → text_processor   (understanding / LM stream)
visual_position_ids = position_ids[vae_token_mask]        # → visual_processor (image generation stream)
video_position_ids  = position_ids[video_vae_token_mask]  # → video_processor  (video generation stream)
```

A token's mask boolean is what slices the position-ids tensor (and the
feature tensor) into the right processor's input. The masks are mutually
exclusive — every token has exactly one of them set to 1.

**Audio piggybacks on `vae_token_mask`.** The audio_pad slots set
`vae_token_mask=1`, just like image_pad. There is no `audio_vae_token_mask`
as a separate top-level mask today. Within the visual_processor, the
per-element field `x_vae_by_modality` (set by `OmniElementVAEAudio` to
`og_element.modality`, e.g. `"t2a"`) is what tells the trainer to dispatch
those features to the audio VAE encoder (DAC / MMAudio) instead of the
image VAE. So routing is two-stage:

```
mask (text / vae / video_vae)  →  picks the stream
x_vae_by_modality              →  picks the modality-specific encoder within that stream
```

The same is true for the inverse direction (decoder / loss head): the noise
prediction head reads `x_vae_by_modality` to dispatch to the audio
denoiser.

**Tokens within one element can have different stream masks.** This is the
key mechanism that lets a single noisy VAE element split its token budget
between text-stream boundary/register tokens and VAE-stream pad slots — see
§8.2 for the per-position mask table.

### 9.2 Modality dispatch within the generation stream

Two element-level fields refine the dispatch inside the generation stream:

- **`x_vae_by_modality`** — string tag (`"t2a"`, `"t2i"`, `"i2i"`, etc.).
  Selects the modality-specific encoder/decoder within the visual_processor.
- **`vae_latent_shapes`** — list of `(H, W)` tuples, one per VAE element.
  For audio it is `None` (audio is 1-D, no spatial structure); for image it
  is the patch grid; the trainer uses this to reshape the flat VAE token
  sequence back into an encoder-shaped tensor.

These don't affect attention; they only affect which weights and which
shape the VAE-stream slots are processed under.

### 9.3 Attention mode — visibility pattern (cross-ref §8.1)

Set per `SequenceElement` (one of `causal` / `full` / `noise`); applies to
every token in the element. To restate precisely:

- **`causal`**: lower-triangular self-block + visibility to all priors.
- **`full`**: bidirectional self-block + visibility to all priors. Used
  for clean VAE elements.
- **`noise`**: same self/prior visibility as `full`, *plus* the outbound
  column is zeroed in pass 2 → tokens after the segment ends cannot attend
  to anything inside it. Used for noisy VAE elements.

So the *only* difference between `full` and `noise` is whether downstream
elements can read into this segment. Both are bidirectional internally and
both can read prior context.

The mode applies regardless of stream — a text-stream token (e.g. a register
or boundary marker inside a noisy element) is governed by `attention_mode =
"noise"` even though it's processed by the LM weights. Stream and attention
mode are orthogonal.

### 9.4 Loss masks — what contributes training signal

These don't affect the forward pass; they decide which positions contribute
to which loss during backward.

- **`txt_loss_mask`** — text cross-entropy applies only where this is 1.
  Critically, **the entire VAE span has `txt_loss_mask=0`** on both
  branches, including registers and start/end markers
  ([omni_audio_ops.py:164-166](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L164-L166)).
  Registers go through the LM weights forward but contribute zero CE loss
  directly; their gradient comes entirely from being attended to by tokens
  that *do* have non-zero loss.
- **`noisy_vae_token_mask`** — doubles as the diffusion-loss target mask.
  Diffusion / MSE loss is computed only at positions where this is 1. This
  is why registers exist *only* on the noisy branch but have
  `noisy_vae_token_mask=0` themselves: they participate in the
  diffusion-conditioned attention block without being a denoising target.
- **`clean_vae_token_mask`** — mirror of the above for the clean branch.
  Clean VAE positions are conditioning context (no loss).

Loss masks and stream masks together produce a useful classification:

| Token role               | `text_token_mask` | `vae_token_mask` | `txt_loss_mask` | `noisy_vae_token_mask` |
| ------------------------ | ----------------- | ---------------- | --------------- | ---------------------- |
| Assistant text (target)  | 1                 | 0                | 1               | 0                      |
| Prompt / user text       | 1                 | 0                | 0               | 0                      |
| Audio register (noisy)   | 1                 | 0                | 0               | 0                      |
| Audio start / end markers| 1                 | 0                | 0               | 0                      |
| Audio pad (noisy)        | 0                 | 1                | 0               | 1                      |
| Audio pad (clean)        | 0                 | 1                | 0               | 0                      |

The "register" row is the unusual one: text-stream forward pass, no direct
loss contribution. Everything else falls along the diagonal.

### 9.5 3D position_ids and M-RoPE

`position_ids` has shape `(S, 3)` carrying `(T, H, W)` axis values for
M-RoPE. The three rows index different RoPE-pair lanes per
`mrope_section=[24, 20, 20]` (§4.3 of `qwen3_audio_graft_poc.md` for the
interleaved layout):

- **Text and audio tokens** — all three axes get the same sequential value.
  Audio is 1-D; there is no spatial structure, so H and W carry no audio
  signal.
- **Image tokens** — H and W carry the patch grid coordinates within the
  image; T carries the element-level temporal index.
- **Video tokens** — T carries the frame index; H, W carry per-frame patch
  coordinates.

Two tokens with identical (`text_token_mask`, `attention_mode`) but
different position-axis structure receive different rotations from M-RoPE.
This is a fourth axis governing what a token "is" at the attention layer,
visible only when image / video tokens are present.

### 9.6 Packing-level isolation

Cross-sample bookkeeping that prevents tokens in different samples (within
the same packed sequence) from interacting:

- **`sample_lens`** — list of token counts per sample inside a packed
  sequence. FlexAttention uses it to isolate samples; tokens across two
  different samples in the same pack don't attend to each other, regardless
  of their per-element `attention_mode`.
- **`padding_mask`** — set by `OmniQwen3Tokenizer` per element; the
  sequence-tail padding when `pad_to_length` is in effect (flex-attention
  path) is forced to `attention_mode="causal"`
  ([sequence_packing.py:547,552](lib/ursa/ursa/models/omni/inference/sequence_packing.py#L547))
  so that padding doesn't leak visibility back into real content.

These are sample / sequence-level concerns rather than per-token, but a
token's effective visibility is the **intersection** of its element-level
`attention_mode` with these sample-level boundaries.

### 9.7 Element ordering in `sequence_plan`

`attention_mode` decides a token's visibility *given* its position. But
which tokens count as "prior" depends on the **order of elements in
`sequence_plan`** — that's data-pipeline-level, not token-level.

For T2A: `[TEXT(prompt), NOISY_VAE_AUDIO(audio)]`. The audio register can
attend to the prompt because the prompt comes first; the prompt cannot see
the audio register because the audio span is `noise` and zeros out its
outbound column. Reorder them and the conditioning would be inverted —
the prompt would read register summaries that haven't been computed yet
relative to the prompt's own forward pass.

This ordering is a consequence of how each task-specific
`OmniAudioSeqBuilder.handle_<task>` (or its image-side analog) emits the
element list. It is data-pipeline state, not model state.

### 9.8 Summary table

| Decides …                                              | Field(s)                                                                        | Granularity         | Set by                                                                |
| ------------------------------------------------------ | ------------------------------------------------------------------------------- | ------------------- | --------------------------------------------------------------------- |
| Which forward-stream weights process the token         | `text_token_mask` / `vae_token_mask` / `video_vae_token_mask`                    | per token           | `OmniElement{Text,VAEAudio,VAEImage,Vit}`                              |
| Which modality-specific encoder within the VAE stream  | `x_vae_by_modality`, `vae_latent_shapes`                                         | per element         | `OmniElementVAEAudio` / `OmniElementVAEImage`                          |
| Attention visibility pattern                           | `attention_mode`                                                                 | per element         | per-element processor (`"causal"` / `"full"` / `"noise"`)              |
| Text CE loss positions                                 | `txt_loss_mask`                                                                  | per token           | `OmniElementText` (1 on assistant text spans)                          |
| Diffusion loss positions + denoising target            | `noisy_vae_token_mask`                                                           | per token           | `OmniElementVAEAudio` / `OmniElementVAEImage` (noisy branch only)      |
| Conditioning-context VAE positions                     | `clean_vae_token_mask`                                                           | per token           | `OmniElementVAEAudio` / `OmniElementVAEImage` (clean branch only)      |
| RoPE rotation along T / H / W axes                     | `position_ids` (S × 3)                                                           | per token           | `OmniPositionIDMRoPE` / `OmniPositionIDStableV0`                       |
| Cross-sample isolation in a pack                       | `sample_lens`, `padding_mask`                                                    | sample / sequence   | `OmniQwen3Tokenizer`, `pack_sequence`                                  |
| What counts as "prior" for any of the above            | order of elements in `sequence_plan`                                             | element-list        | `OmniAudioSeqBuilder.handle_<task>` (or image-side analog)             |

### 9.9 Walkthrough: the life of one register token

Concretely, what happens at one register position `p` in a NOISY_VAE_AUDIO
element with `M=16`, `N=157`, in a packed sample with one prior text element
of length 50:

1. **Stream selection.** `text_token_mask[p] = 1` → goes through
   `text_processor`. `vae_token_mask[p] = 0`. The LM embedding table looks
   up id 151643 (`<|endoftext|>`) and outputs the register's initial vector.
2. **Position encoding.** `position_ids[p]` = `(T, H, W)` all set to the
   element-relative sequential index (audio is 1-D). M-RoPE rotates Q/K
   accordingly. RoPE pair indices owned by axis 0 (the T axis) drive the
   rotation; H/W lanes rotate by the same scalar (no audio-specific signal).
3. **Attention.** Element's `attention_mode = "noise"`. In pass 1 the
   register row gets full visibility into priors (positions `[0, 50)` —
   the text element) and bidirectional self-block with the rest of the
   noisy element (positions `[50, 50+M+N+2)`). In pass 2 the register's
   outbound column (its column-index slice) is zeroed for every row outside
   the segment.
4. **Loss masks at this position.** `txt_loss_mask[p] = 0` → no CE
   contribution. `noisy_vae_token_mask[p] = 0` → no diffusion contribution.
   The register has no direct supervision.
5. **Gradient flow.** The register's representation is updated only by
   gradients flowing back through attention from positions that *do* have
   non-zero loss — i.e. from the noisy audio_pad slots
   (`noisy_vae_token_mask = 1`) inside the same element. Those slots
   attend bidirectionally to all 16 register slots and read summary signal
   from them; gradient on the diffusion loss flows through that attention
   into each register's representation.
6. **Across-sample isolation.** Other samples in the same FlexAttention
   pack are `sample_lens`-bounded out, so their registers (if any) don't
   interfere.

The end product, after training: the register row's embedding for id
151643 has specialized into a "summary slot conditioned on prior text"
prior, distinct from `<|endoftext|>`'s nominal role as an EOS / pad token.
The model differentiates the 16 register slots by their position in the
M-RoPE schedule.


## 10. Plan: distinguish vision vs audio in the VAE token masks

The current mask scheme — `vae_token_mask`, `clean_vae_token_mask`,
`noisy_vae_token_mask` — predates the audio path. Audio piggybacks on
these by setting them on its own pad slots and relying on the per-element
string `x_vae_by_modality` to dispatch downstream (§9.1, §9.2). It works,
but every consumer that wants to act on "audio specifically" or "vision
specifically" has to read modality out of element-level metadata and
intersect it with a per-token mask. This section plans the rename / extension
that makes the modality split first-class at the mask level.

### 10.1 Goals

1. **Per-modality VAE token sets become directly addressable as masks.**
   Loss heads, stream routers, and inference-time dispatchers should be
   able to write `position_ids[noisy_audio_vae_token_mask]` without
   consulting `x_vae_by_modality`.
2. **Keep a general `vae_token_mask` union** for code paths that don't
   care about modality (e.g. attention masking, packing bookkeeping).
3. **Treat image and video tokens as one "vision" group, for now.** The
   model already has a separate `video_vae_token_mask` / `video_processor`
   path at the trainer surface; we keep that intact at the model layer,
   but at the data-pipeline layer we do not split image vs video. If a
   future need arises, adding `image_vae_token_mask` is an additive
   change — no rename, no consumer churn.
4. **No checkpoint impact.** Masks are data-pipeline / trainer-surface
   state; model weights are unaffected.

### 10.2 Current state and friction points

Defined on `TokenizedSequenceElement`
([lib/koba_shared/koba_shared/processor/tokenized_types.py:26-30](lib/koba_shared/koba_shared/processor/tokenized_types.py#L26-L30)):

```python
vae_token_mask: torch.Tensor | None = None         # union: any VAE slot (image OR audio today)
clean_vae_token_mask: torch.Tensor | None = None   # clean branch (any modality)
noisy_vae_token_mask: torch.Tensor | None = None   # noisy branch (any modality)
```

Plus the trainer-side `video_vae_token_mask` (separate from the above;
image and audio share `vae_token_mask`, video has its own mask and its
own `video_processor` path).

Friction points today:

- **`vae_token_mask` is overloaded.** In image-only sequences it means
  "image VAE slot"; in audio-only sequences "audio VAE slot"; in mixed
  sequences (none today, but A2A / mixed-modality omni training will get
  there) it means both, with the modality split hidden in
  `x_vae_by_modality`.
- **Loss heads have to AND with a modality predicate.** The audio
  diffusion loss can't be `loss(x[noisy_vae_token_mask])` — it needs
  `loss(x[noisy_vae_token_mask & is_audio_modality(x_vae_by_modality, split_lens)])`,
  with the predicate broadcast from per-element strings to per-token
  booleans.
- **Granularity mismatch.** `x_vae_by_modality` is per-element (one
  string per `SequenceElement`); masks are per-token. Every consumer that
  wants to attribute a token to a modality has to bridge the two via
  `split_lens`.
- **Precedent.** `video_vae_token_mask` is already a separate top-level
  mask. Audio should follow the same pattern.

### 10.3 Design rationale

Two design questions sit underneath the rename. Calling them out
explicitly because either could be answered "the cheap way" and produce a
plausible-looking but worse design.

#### 10.3.1 Are leaf masks redundant given `x_vae_by_modality`?

**Both schemes carry the same information** — leaf masks
(`{clean,noisy}_{vision,audio}_vae_token_mask`) can be reconstructed
from `(x_vae_by_modality, vae_token_mask, clean_/noisy_vae_token_mask, split_lens)`
by broadcasting the per-element modality string to per-token booleans
and intersecting. So the question is not "is the new scheme more
expressive" — it isn't — it's "does the indirection cost matter."

Where the indirection costs show up:

| Aspect                                  | Today: `x_vae_by_modality` + branch masks                                              | Proposed: leaf masks (`{clean,noisy}_{vision,audio}_vae_token_mask`) |
| --------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Information content                     | Sufficient                                                                              | Sufficient — identical                                          |
| Modality classification site            | **Scattered.** Every consumer that wants "is audio" needs an `is_audio()` table        | **Centralized.** Per-element processor sets the leaf mask once  |
| Granularity at consumer                 | Per-element string + per-token bool → broadcast required                                | Per-token bool, no conversion                                   |
| Call site readability                   | `noisy_vae_token_mask & is_audio_modality(x_vae_by_modality, split_lens)`               | `noisy_audio_vae_token_mask`                                    |
| GPU vectorization                       | String lookup + range broadcast → harder to fuse                                        | Pure bool ANDs, trivially fused                                 |
| Maintenance per new modality            | Touch every classifier site (loss head, sampler, stream router, …)                      | Touch one per-element processor                                 |
| Failure mode (forgotten classifier)     | **Silent attribution bug** — token routed to wrong loss / encoder, no error             | Doesn't apply                                                   |
| Failure mode (granularity mismatch)     | Crash if shapes mismatch; silent if shapes happen to align                              | Doesn't apply                                                   |
| Memory cost                             | One string list per element                                                              | Four bool tensors per element (~tens of KB per packed sample)   |

The information-content row is the only neutral one. Every other row
favors the leaf-mask scheme. The decisive ones in practice:

- **Silent attribution bugs.** If a new audio task adds modality string
  `"a2a"` and the loss head's `is_audio()` classifier doesn't learn it,
  audio tokens get routed into the image diffusion loss with no error —
  the tensors have compatible shapes, they just contain the wrong data.
  This class of bug compounds as the modality vocabulary grows.
- **Maintenance locality.** With leaf masks, "what counts as audio" is
  decided **once**, in the per-element processor that already knows its
  own modality. Without leaf masks, it's decided **N times** at each
  consumer.

So the answer to "is the existing scheme sufficient": yes, mathematically.
The answer to "should we keep using only it": no — the indirection is
the problem we're solving, and adding leaf masks moves classification to
the producer where it belongs.

There is one mild caveat. `x_vae_by_modality` is set on VAE elements
only, so it tells you nothing about boundary tokens or registers — those
are `text_token_mask=1` / `vae_token_mask=0` and are by construction not
audio VAE tokens. The proposed leaf masks have the same scope (VAE slots
only), so this isn't a regression. A separate concept "is this token part
of an audio *element* including its boundaries and registers" would be
`audio_element_token_mask`, which we don't have a use case for today.

#### 10.3.2 How many new masks?

If we accept that leaf masks earn their keep, the next question is which
combinations to materialize. The full Cartesian product is nine (union +
two modality rollups + two branch rollups + four leaves). The minimum
that supports every concrete consumer is five (union + four leaves);
anything in the middle just shifts work between materialization and
on-the-fly OR.

| Mask                                | 9-mask scheme | 5-mask scheme                             |
| ----------------------------------- | ------------- | ----------------------------------------- |
| `vae_token_mask` (union)            | Materialized  | Materialized                              |
| `clean_vision_vae_token_mask` (leaf) | Materialized  | Materialized                              |
| `noisy_vision_vae_token_mask` (leaf) | Materialized  | Materialized                              |
| `clean_audio_vae_token_mask` (leaf)  | Materialized  | Materialized                              |
| `noisy_audio_vae_token_mask` (leaf)  | Materialized  | Materialized                              |
| `vision_vae_token_mask` (modality)  | Materialized  | Computed: `clean_vision \| noisy_vision`  |
| `audio_vae_token_mask` (modality)   | Materialized  | Computed: `clean_audio \| noisy_audio`    |
| `clean_vae_token_mask` (branch)     | Materialized  | Computed: `clean_vision \| clean_audio`   |
| `noisy_vae_token_mask` (branch)     | Materialized  | Computed: `noisy_vision \| noisy_audio`   |
| **Fields per element**              | **9**         | **5**                                     |
| **Consistency invariants to assert** | 4 OR-decompositions across rollups + leaves | 1 (`vae_token_mask = OR of 4 leaves`) |
| Hot-path cost (leaf queries)        | Free          | Free                                      |
| Hot-path cost (axis queries)        | Free          | One vectorized OR per query, ~free        |
| Conceptual surface                  | 9 names       | 5 names                                   |
| Maintenance per per-element processor change | 9 fields to populate consistently | 5 fields                          |
| Risk of inconsistency drift         | Higher (multiple redundant fields can diverge under partial edits) | Lower |

The 9-mask scheme buys nothing at the leaf-query level (which is the use
case the rename is *for*) and pays for it with 4 extra fields, a
combinatorial assertion surface, and more places where a partial edit
can leave the masks inconsistent. The axis queries it does accelerate
("any noisy slot", "any audio slot") cost a single vectorized OR over a
boolean tensor of length ~8k — essentially free on GPU, amortized to
once per packed sample anyway. **Decision: 5 masks.**

If a future profiling pass shows axis queries on the hot path enough
that the OR is non-trivial, materializing them is an additive change.

### 10.4 Proposed scheme

Five masks on `TokenizedSequenceElement`:

| Mask                                | Set when                                                              |
| ----------------------------------- | --------------------------------------------------------------------- |
| `vae_token_mask`                    | Any VAE slot (union — kept; semantics unchanged)                      |
| `clean_vision_vae_token_mask`       | Clean branch + image or video VAE slot                                |
| `noisy_vision_vae_token_mask`       | Noisy branch + image or video VAE slot                                |
| `clean_audio_vae_token_mask`        | Clean branch + audio VAE slot                                         |
| `noisy_audio_vae_token_mask`        | Noisy branch + audio VAE slot                                         |

The OR-decomposition (single invariant to assert):

```
vae_token_mask = clean_vision_vae_token_mask | noisy_vision_vae_token_mask
              | clean_audio_vae_token_mask  | noisy_audio_vae_token_mask
```

Axis rollups computed lazily where needed:

```python
noisy_vae_token_mask  = noisy_vision_vae_token_mask | noisy_audio_vae_token_mask
clean_vae_token_mask  = clean_vision_vae_token_mask | clean_audio_vae_token_mask
vision_vae_token_mask = clean_vision_vae_token_mask | noisy_vision_vae_token_mask
audio_vae_token_mask  = clean_audio_vae_token_mask  | noisy_audio_vae_token_mask
```

Naming convention is `<branch>_<modality>_vae_token_mask` because (a) it
preserves the existing `<branch>_vae_token_mask` prefix (less visual
churn for readers), and (b) "noisy" / "clean" is the more salient axis
at the loss layer (which is the primary consumer).

`vision` collapses image + video at the data-pipeline level. The trainer
surface keeps `video_vae_token_mask` (which gates `video_processor`
dispatch); a video token sets both `*_vision_vae_token_mask` and
`video_vae_token_mask`. We deliberately do not introduce
`image_vae_token_mask` or `*_image_vae_token_mask` until a real consumer
needs the image-vs-video split.

### 10.5 What changes downstream

**1. Stream routing in
[model.py:560-672](lib/ursa/ursa/models/omni/model/model.py#L560-L672)
gains an audio path.**

```python
# Before
text_position_ids   = position_ids[text_token_mask]                         # → text_processor
visual_position_ids = position_ids[vae_token_mask]                          # → visual_processor (image AND audio)
video_position_ids  = position_ids[video_vae_token_mask]                    # → video_processor

# After
text_position_ids   = position_ids[text_token_mask]                                                          # → text_processor
visual_position_ids = position_ids[clean_vision_vae_token_mask | noisy_vision_vae_token_mask]                # → visual_processor (image only)
video_position_ids  = position_ids[video_vae_token_mask]                                                     # → video_processor
audio_position_ids  = position_ids[clean_audio_vae_token_mask  | noisy_audio_vae_token_mask]                 # → audio_processor (NEW)
```

The `clean | noisy` ORs are vectorized booleans on a length-`S` tensor,
once per packed sample. If profiling shows them on the hot path, cache
the result inline; for now leave inline.

**2. Loss heads switch to leaf masks.**

```python
# Before
diffusion_loss_image = mse(model_out[noisy_vae_token_mask & is_image], targets_image)
diffusion_loss_audio = mse(model_out[noisy_vae_token_mask & is_audio], targets_audio)

# After
diffusion_loss_image = mse(model_out[noisy_vision_vae_token_mask], targets_image)
diffusion_loss_audio = mse(model_out[noisy_audio_vae_token_mask],  targets_audio)
```

This is the change the rename is for: modality is in the mask name, the
classifier table goes away.

**3. Per-element processors set the leaf masks instead of the branch
mask.**

In `OmniElementVAEAudio`
([lib/koba_shared/koba_shared/processor/omni_audio_ops.py:182-204](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L182-L204)):

```python
# Before
if tok_element.type == SequenceType.CLEAN_VAE_AUDIO:
    tok_element.clean_vae_token_mask = ...
elif tok_element.type == SequenceType.NOISY_VAE_AUDIO:
    tok_element.noisy_vae_token_mask = ...
tok_element.vae_token_mask = ...

# After
if tok_element.type == SequenceType.CLEAN_VAE_AUDIO:
    tok_element.clean_audio_vae_token_mask = ...
elif tok_element.type == SequenceType.NOISY_VAE_AUDIO:
    tok_element.noisy_audio_vae_token_mask = ...
tok_element.vae_token_mask = ...   # union — unchanged
```

Mirror in `OmniElementVAEImage` with `*_vision_vae_token_mask`. Video
processors set both `*_vision_vae_token_mask` and `video_vae_token_mask`.

A small helper centralizes the mask construction so the three processors
don't drift:

```python
def _set_vae_leaf_mask(tok_element, branch: Literal["clean", "noisy"], modality: Literal["vision", "audio"]):
    # one place that knows: "leaf mask = vae_token_mask intersected with this branch and modality"
    ...
```

### 10.6 Migration plan

**Phase 0 — additive (no breaking changes).**

- Add the four leaf fields to `TokenizedSequenceElement` (default `None`).
  Keep `clean_vae_token_mask` / `noisy_vae_token_mask` for one cycle as
  optional rollups so existing consumers don't break.
- Update `OmniElementVAEAudio`, `OmniElementVAEImage`, and the video VAE
  element processor to populate the new leaf fields **in addition to**
  the existing branch-only ones.
- Add a debug-mode consistency assertion:
  `vae_token_mask == OR of the four leaves`. Run on a sample batch in CI
  and on the first batch of each training job.

After phase 0 nothing breaks; consumers can opt in to the new masks
incrementally.

**Phase 1 — migrate consumers.**

- Trainer `model.py`: switch visual path to
  `clean_vision | noisy_vision`; add audio path on
  `clean_audio | noisy_audio`.
- Loss heads: switch `noisy_vae_token_mask & is_modality(...)` →
  `noisy_<modality>_vae_token_mask`.
- Inference (`generate_modality_disaggregated.py`,
  `generate_modality_legacy.py`, `tdm_sampler.py`): same pattern.
- Tests (`test_t2a_data_roundtrip.py`,
  `test_audio_position_ids_offset.py`, `dummy_dataset.py`): update
  fixtures to set the leaf masks.

This is the bulk of the diff (~30 files per the grep). Lands in stages,
file by file, because phase 0 keeps the old fields populated.

**Phase 2 — drop the branch-only rollups (`clean_vae_token_mask`,
`noisy_vae_token_mask`).**

After phase 1, audit remaining uses. Anything genuinely modality-agnostic
gets rewritten to use the union or a lazy OR; anything that should have
been modality-aware gets fixed. Once the field count is 0, remove the
field. This is when the conceptual surface area actually shrinks from
"old + new" back down to 5.

**Phase 3 — collapse `x_vae_by_modality` if no longer needed.**

After phase 2, `x_vae_by_modality`'s only remaining roles are:

- `vae_latent_shapes` indexing (per-element shape lookups for the
  visual_processor)
- Sampler-time per-modality branching (e.g.
  [tdm_sampler.py:145](lib/ursa/ursa/models/omni/inference/tdm_sampler.py#L145))
- Telemetry / logging
- Picking timestep shift mappings (per
  [docs/src/onboarding/omni/omni-data.md:50](docs/src/onboarding/omni/omni-data.md#L50))

If those settle into a single per-element string field, leave it. If
they fragment, consider replacing it with a richer per-element struct
(one field per role). Optional cleanup; not on the critical path.

### 10.7 Risks and open questions

- **Mask-set consistency.** Four leaves + one union have one OR
  invariant. Centralize construction in `_set_vae_leaf_mask` so the
  three element processors can't drift.
- **Memory cost.** Four extra bool tensors per element ≤ 50 KB per packed
  sample at `max_num_tokens=8000`. Negligible.
- **Checkpoint compatibility.** None — masks are data-pipeline state.
- **Inference-side mirror.** The hand-built audio span construction in
  [t2a_processor.py](projects/kuma/kuma/projects/omni/audio/inference/processor/t2a_processor.py)
  must populate the new leaf masks. If it doesn't, inference reads a
  zero-init `noisy_audio_vae_token_mask` and silently routes audio
  features through the wrong path. Same dual-edit hazard as the POC
  tokenizer swap (§8.3).
- **Image-vs-video deferral.** `image_vae_token_mask` / `*_image_*_mask`
  are not introduced. If a future image-only consumer needs the split,
  add as an additive change — no rename, no consumer churn.
- **Lazy OR cost on hot paths.** The 5-mask scheme replaces three
  potentially-materialized rollups (`vision_*`, `audio_*`, `clean_*`,
  `noisy_*`) with on-the-fly ORs. If a later profile shows any of them
  on the hot path, that rollup is an additive change to materialize.

### 10.8 Rollout sequencing relative to the audio §5 plans

Sequencing relative to `qwen3_audio_graft_poc.md` §5:

| Plan item                               | Relation to §10        | Notes                                                                          |
| --------------------------------------- | ---------------------- | ------------------------------------------------------------------------------ |
| §5.1 swap omni-t2a backbone to VL-2B-Audio-POC | Independent     | Backbone reads whatever masks the data pipeline produces                        |
| §5.3 A2T data processing module         | After §10 phase 0–1    | A2T builder needs the audio-vs-text mask vocabulary                            |
| §5.4 CE + diffusion loss mixing         | After §10 phase 0–1    | Loss switch is exactly the case the rename targets                             |
| §5.5 update hardcoded special-token IDs | Independent (coordinate) | Both touch `OmniElementVAEAudio`; coordinate into one PR if convenient         |

Dependency edges:

```
§10 (phase 0–1)  ──>  §5.3 (A2T builder)  ──>  §5.4 (loss switch)
§10 (phase 0–1)  ──────────────────────────>  §5.4
§5.5 (token id swap)  ⊥  §10
§5.1 (backbone swap)  ⊥  §10
```

Phases 2–3 of §10 are clean-up — they happen after phase 1 lands and
gate nothing in §5.


## 11. Variable inventory, naming consistency, and known-problematic names

§10 commits to a 5-mask scheme and lays out the migration. This section is
the companion: a **complete inventory** of the variables that govern token
behavior today, **cross-directory consistency analysis**, and a list of
**three names that are problematic on their own merits** (independent of
the §10 rename) — the missing `aut_token_mask`, the over-specific
`video_vae_token_mask`, and the misnamed `clean_vae_img_mask`. Some content
overlaps §10 by design; both sections will be reviewed together and merged
later.

Every claim below is grounded in code:
[`lib/koba_shared/koba_shared/processor/tokenized_types.py`](lib/koba_shared/koba_shared/processor/tokenized_types.py),
[`omni_vae_ops.py`](lib/koba_shared/koba_shared/processor/omni_vae_ops.py),
[`omni_audio_ops.py`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py),
[`omni_vit_ops.py`](lib/koba_shared/koba_shared/processor/omni_vit_ops.py),
[`omni_text_ops.py`](lib/koba_shared/koba_shared/processor/omni_text_ops.py),
[`lib/ursa/ursa/models/omni/model/model.py`](lib/ursa/ursa/models/omni/model/model.py),
[`sequence_packing.py`](lib/ursa/ursa/models/omni/inference/sequence_packing.py).

### 11.1 Per-token boolean masks (declared on `TokenizedSequenceElement`)

All of these are `torch.Tensor | None` of shape `(seq_len,)`, dtype `bool`,
declared on
[`TokenizedSequenceElement`](lib/koba_shared/koba_shared/processor/tokenized_types.py#L10-L67).
Granularity is **per token**.

| Field                    | Comment in `tokenized_types.py`                                          | Set by                                                    |
| ------------------------ | ------------------------------------------------------------------------ | --------------------------------------------------------- |
| `text_token_mask`        | "binary mask of which tokens are pure text tokens (not in VAE or ViT)"   | `OmniElement{Text,VAEAudio,VAEImage,Vit}`                  |
| `vae_token_mask`         | "binary mask of which tokens are in the VAE branch"                      | `OmniElementVAE{Audio,Image}`                              |
| `clean_vae_token_mask`   | "binary mask of which tokens are in clean VAE"                           | `OmniElementVAE{Audio,Image}` on the clean branch          |
| `noisy_vae_token_mask`   | "binary mask of which tokens are in noisy VAE"                           | `OmniElementVAE{Audio,Image}` on the noisy branch          |
| `vit_token_mask`         | "binary mask of which tokens are in the ViT branch"                      | `OmniElementVit`                                           |
| `txt_loss_mask`          | "binary mask of which tokens receive text supervision"                   | `OmniElementText` (1 on assistant text spans)              |
| `padding_mask`           | "binary mask of which tokens are padding"                                | `OmniQwen3Tokenizer` / `pack_sequence` for the tail block  |

Notes verified against code:

- `text_token_mask`, `vit_token_mask` and `vae_token_mask` are **mutually
  exclusive at the trainer surface** — the model dispatches each token to
  exactly one of `text_processor`, `visual_processor`, `video_processor`
  by mask boolean. See
  [`model.py:560–672`](lib/ursa/ursa/models/omni/model/model.py#L560-L672).
- The boundary tokens (`<|vision_start|>` / `<|vision_end|>`) and any
  registers are explicitly carved out of `vae_token_mask` (set to 0) and
  carved into `text_token_mask` (set to 1) inside `OmniElementVAE{Audio,Image}`
  — see [`omni_vae_ops.py:66-85`](lib/koba_shared/koba_shared/processor/omni_vae_ops.py#L66-L85)
  for the image side and the same pattern on the audio side at
  [`omni_audio_ops.py:159-178`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L159-L178).

### 11.2 Per-token mask declared at the trainer surface (not on `TokenizedSequenceElement`)

| Field                      | Where declared                                                                                  | Granularity | Notes                                                                          |
| -------------------------- | ----------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------ |
| `video_vae_token_mask`     | `Bool[Tensor, " S"] \| None` arg of `model.forward` ([`model.py:496`](lib/ursa/ursa/models/omni/model/model.py#L496), `:1303`) | per token   | Selects video tokens for `video_processor` dispatch. **Not declared on `TokenizedSequenceElement`** — it is constructed by upstream packing code. |

Verified: zero matches for `video_vae_token_mask` in `lib/koba`,
`lib/koba_shared`, and `projects/kuma/kuma/projects/omni/audio` (per the
grep run in §10's inconsistency check). Three matches in `omni/bagel` are
all wiring it through call sites, not setting it. So this mask is a
**model-only** field today, fed in by the packing layer's video-aware
path.

### 11.3 Per-element fields on `TokenizedSequenceElement`

These are element-level fields (one value per `SequenceElement`, not per
token).

| Field                  | Type                                                                            | Set by / value range                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `type`                 | `SequenceType` enum                                                              | `TEXT=0`, `NOISY_VAE_IMAGE=1`, `CLEAN_VAE_IMAGE=2`, `VIT_IMAGE=3`, `PACKED=4`, `TEXT_INCOMPLETE=5`, `NOISY_VAE_AUDIO=6`, `CLEAN_VAE_AUDIO=7`, `VIT_AUDIO=8` ([`koba_shared/common/types.py:23-36`](lib/koba_shared/koba_shared/common/types.py#L23-L36)) |
| `modality`             | `str`                                                                            | Task tag, e.g. `"t2a"`, `"t2i"`, `"i2i"`, `"image_edit"`                                                                      |
| `attention_mode`       | `Literal["causal", "full", "noise"] \| None`                                     | Set by per-element processor ([`tokenized_types.py:64`](lib/koba_shared/koba_shared/processor/tokenized_types.py#L64))         |
| `num_tokens`           | `int \| None`                                                                    | Element's token count (set in per-element processors)                                                                         |
| `text_str`             | `str \| list[str] \| None`                                                       | Concatenated string form of the element's tokens                                                                              |
| `input_ids`            | `Tensor \| None`                                                                 | `[seq_len,]` text-token ids after tokenization                                                                                 |
| `x_vae`                | `Tensor \| list[Tensor] \| None`                                                 | Source data for the VAE encoder (waveform / pixel data)                                                                       |
| `x_vit`                | `list[Tensor] \| Tensor \| None`                                                 | Source images for ViT                                                                                                         |
| `vae_latent_shapes`    | `list[tuple[int, int]] \| list[tuple[int]] \| None`                              | `(H, W)` for image; `(L,)` or `None` for audio (1-D)                                                                          |
| `vit_latent_shapes`    | `list[tuple[int, int]] \| None`                                                  | ViT latent grid                                                                                                               |
| `x_vae_by_modality`    | `str \| list[str] \| None`                                                       | Modality tag for VAE-element dispatch (encoder, timestep schedule, sampler branching)                                          |
| `clean_vae_img_mask`   | **`bool \| torch.Tensor \| None`** (note: `bool` for unpacked, list when packed) | `1` if element is CLEAN_VAE_*, `0` if NOISY_VAE_* — see §11.6.3 for the misnomer                                              |
| `position_ids`         | `Tensor \| None`                                                                 | `(seq_len, 3)` 3-axis M-RoPE indices. **Note:** the dataclass comment says `[3, seq_len]` but the model.py uses `(S, 3)` — see §11.6.4 |

### 11.4 Sample / sequence-level fields (in the packed dict)

These appear at packing time (not on `TokenizedSequenceElement` itself),
produced by `pack_sequence` in
[`sequence_packing.py`](lib/ursa/ursa/models/omni/inference/sequence_packing.py).

| Field                       | Granularity                       | Meaning                                                                                                                  |
| --------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `sample_lens`               | per-sample (one int per sample)   | Token counts per sample inside a packed sequence — FlexAttention block boundaries between samples                        |
| `split_lens`                | per-element (one int per element) | Token counts per element ([`TokenizedSequencePlan.split_lens`](lib/koba_shared/koba_shared/processor/tokenized_types.py#L78)) |
| `attention_modes`           | per-element list (one str per element) | Flattened `attention_mode` strings across one packed sample                                                            |
| `attn_modes`                | per-token list (one str per token) | Per-token broadcast of `attention_modes`, used inside `prepare_attention_mask_per_sample`                                |
| `x_vae_by_modality_tensor`  | per-element int tensor             | Int-encoded `x_vae_by_modality` via `MODALITY_TO_INDEX` ([`tensorize.py:119-124`](lib/koba/koba/v2/core/tensorize.py#L119-L124)) |
| `position_ids`              | `(seq_len, 3)`                     | Concatenated per-token M-RoPE positions across the pack                                                                  |
| `nested_attention_mask`     | `(seq_len, seq_len)` additive       | Built by `prepare_attention_mask_per_sample` from `(split_lens, attn_modes)`                                              |

### 11.5 Cross-directory consistency analysis

Counts of name occurrences, repeated here for completeness (file-level grep
from §10's earlier verification, restricted to non-binary, non-cache files):

| Field                     | ursa | koba | koba_shared | omni/bagel | omni/audio |
| ------------------------- | ---: | ---: | ----------: | ---------: | ---------: |
| `text_token_mask`         |   15 |    1 |          12 |         25 |         12 |
| `vae_token_mask`          |   14 |    6 |          10 |         22 |         10 |
| `vit_token_mask`          |   12 |    3 |           6 |         12 |          7 |
| `video_vae_token_mask`    |    2 |    0 |           0 |          3 |          0 |
| `clean_vae_token_mask`    |   10 |    2 |           8 |          4 |          2 |
| `noisy_vae_token_mask`    |    8 |    1 |           8 |         13 |          2 |
| `clean_vae_img_mask`      |    9 |    1 |           6 |         15 |          1 |
| `txt_loss_mask`           |    8 |    4 |          10 |          6 |          3 |
| `padding_mask`            |   26 |    1 |           6 |          4 |          2 |
| `attention_mode`          |   26 |    2 |          10 |          8 |          3 |
| `x_vae_by_modality`       |    3 |    1 |           6 |         15 |          5 |
| `vae_latent_shapes`       |   11 |    6 |           7 |         20 |          2 |

**Conclusions:**

- The canonical names are used **consistently as themselves** wherever they
  appear. No place uses a different identifier for the same field.
- Asymmetric distributions are by design (and verified against producer
  files, not just counts):
  - `video_vae_token_mask` is sparse because video paths only flow through
    ursa's trainer (model.py) and bagel's video tasks; koba_shared and the
    audio dir don't construct video VAE elements.
  - `clean_vae_img_mask` is heavy on the image stack because it's set by
    `OmniElementVAEImage` (image-side producer) and read by
    `tdm_sampler.py` (image-side consumer). The single audio occurrence
    is in `omni_audio_ops.py:181` — see §11.6.3 for why audio sets it
    despite the misleading name.
  - `x_vae_by_modality` is dense in `omni/bagel` because every per-task
    inference processor sets it (`t2i_processor.py`, `i2t_processor.py`,
    etc.).

### 11.6 Three known-problematic names

These are independent of §10's rename — each is a name-quality problem on
its own merits.

#### 11.6.1 The missing `aut_token_mask` (audio understanding token mask)

**The asymmetry.** The understanding stream has a per-token mask for image
tokens (`vit_token_mask`, set by `OmniElementVit`) but no analog for
audio. Today the omni stack has no semantic audio encoder wired into the
understanding stream; audio only enters via the VAE generation stream
(`x_vae`, `vae_token_mask`).

**Why this becomes a real gap.** The §5 audio-graft plan
(`qwen3_audio_graft_poc.md`) attaches Qwen3-ASR-1.7B's audio encoder as a
sibling of the ViT. Inference splices its output into the LM via
`<|audio_pad|>` row substitution — i.e. specific positions in the
sequence get their `inputs_embeds` overwritten by the audio encoder's
output. That's exactly the role `vit_token_mask` plays for image tokens
today: tells the trainer "at these positions, the value is supplied by an
external encoder, not by an embedding table lookup."

The natural pair:

```
vit_token_mask  ←→  aut_token_mask
                    (or audio_understanding_token_mask if we want the long form)
```

Producer would be a new `OmniElementAut` (mirroring `OmniElementVit` at
[`omni_vit_ops.py`](lib/koba_shared/koba_shared/processor/omni_vit_ops.py)).
Consumer would be a new dispatch path in
[`model.py:560-672`](lib/ursa/ursa/models/omni/model/model.py#L560-L672)
that slices `position_ids[aut_token_mask]` into a hypothetical
`audio_understanding_processor`. (The model architecture itself is the
heaviest lift; the mask field is the easy part.)

**Status today.** `grep -rn aut_token_mask` returns zero hits across the
repo. So this is a clean addition, not a rename. Reasonable to add as a
phase-0 field in §10 (default `None`), even before the audio graft lands,
to avoid the dual-edit hazard later.

#### 11.6.2 `video_vae_token_mask` — model-internal, not a data-pipeline citizen

**Where it lives.** Top-level `model.forward` arg at
[`model.py:496`](lib/ursa/ursa/models/omni/model/model.py#L496) and `:1303`.
**Not declared on `TokenizedSequenceElement`** (verified — absent from
[`tokenized_types.py`](lib/koba_shared/koba_shared/processor/tokenized_types.py)).

**What it does.** Selects video tokens for the separate `video_processor`
dispatch:

```python
# model.py:669-672
if (... and video_vae_token_mask is not None):
    video_tokens_embed = self.video_processor.embedding(video_tokens)
    video_position_ids = position_ids[video_vae_token_mask]
    video_rope_cos, video_rope_sin = self.video_processor.rotary_embedding(...)
```

**Why the name is awkward.** The existing per-token mask scheme has two
levels: a per-modality dispatch mask (`text_token_mask`, `vit_token_mask`,
`vae_token_mask`) and per-branch / per-modality sub-masks. The image and
audio paths share the same `vae_token_mask` for dispatch and split inside
the VAE stream via `x_vae_by_modality`; video uses a *different* dispatch
mask `video_vae_token_mask`. So the naming convention is ambiguous about
what's a stream-dispatch mask vs what's a sub-classification mask.

**Under §10's "image + video = vision" simplification.** The data-pipeline
side does not need to expose a separate video mask — image and video both
set `*_vision_vae_token_mask` (and `vae_token_mask`). The model surface
might still want to dispatch video to its own processor (architectural
choice, separate from the data layer), in which case `video_vae_token_mask`
would be **derived** at the model boundary from the data-pipeline's masks
plus a per-element video flag (similar to how `x_vae_by_modality` is used
today for audio).

**Recommendation.**

- Document `video_vae_token_mask` as a **model-internal** mask, not a
  data-pipeline field. (It already is one — but the name suggests it
  belongs to the same family as `vae_token_mask`, which it doesn't.)
- Consider deriving it at the model boundary from
  `vision_vae_token_mask & is_video_element_broadcast` so the data
  pipeline has only one vision mask family.
- Out of scope for §10 phase 1; a candidate for §10 phase 3 cleanup
  alongside `x_vae_by_modality`.

#### 11.6.3 `clean_vae_img_mask` — misnamed, overlapping with the per-token branch masks

**What the field is.** From
[`tokenized_types.py:24`](lib/koba_shared/koba_shared/processor/tokenized_types.py#L24):

```python
# length N binary mask of which images in the vaes are clean
clean_vae_img_mask: bool | torch.Tensor | None = None
```

So it's a **per-VAE-element** scalar bool (`bool` on a single element;
becomes `Tensor[N_vae_elements]` after packing). The consumer indexing
pattern confirms this:

```python
# tdm_sampler.py:181
if not kwargs_cond["clean_vae_img_mask"][index]:    # index is per-element
    sigma_shift = ...
```

**Three problems with the name.**

1. **Not a per-token mask.** The `_mask` suffix in this codebase otherwise
   means "per-token bool tensor" by convention (`vae_token_mask`,
   `clean_vae_token_mask`, `noisy_vae_token_mask`, `vit_token_mask`,
   `text_token_mask`, `txt_loss_mask`, `padding_mask` all are per-token).
   `clean_vae_img_mask` is **per-VAE-element**, breaking the convention.

2. **Not image-specific.** The audio-side processor at
   [`omni_audio_ops.py:181`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L181)
   sets `clean_vae_img_mask = 1` for `CLEAN_VAE_AUDIO` and `0` for
   `NOISY_VAE_AUDIO`. So the `_img_` infix is wrong — audio sets it too.
   Verified by reading both producers.

3. **Asymmetric without justification.** No `noisy_vae_img_mask` exists,
   because this single 0/1 field encodes both branches inversely
   (`clean=1` ⇔ `noisy=0`). That's fine as a representation, but it's
   inconsistent with the per-token family which has both
   `clean_vae_token_mask` and `noisy_vae_token_mask` declared explicitly.

**Information overlap.** The per-element `clean_vae_img_mask` is
**reconstructible** from the per-token `clean_vae_token_mask` plus
`split_lens`:

```python
clean_vae_img_mask[i] = clean_vae_token_mask[start_i:end_i].any()  # per element
```

The reason it exists separately is that the sampler needs **per-element
indexing** (`kwargs_cond["clean_vae_img_mask"][index]`), and computing the
above on every step is wasteful. So it's a denormalization for sampler
convenience.

**Recommendation.** Rename in a parallel cleanup (independent of §10
phase 1):

```
clean_vae_img_mask  →  is_clean_vae_element        (per-element bool flag, semantically accurate)
```

Or, if we want strict parallelism with §10's leaf-mask names:

```
clean_vae_img_mask  →  clean_vae_element_flag      (and add noisy_vae_element_flag for symmetry)
```

The cleanup is independent of §10 because the field is per-element and the
§10 rename is about per-token masks. They could ship together but don't
have to.

#### 11.6.4 Bonus: `position_ids` shape comment is wrong

In [`tokenized_types.py:62`](lib/koba_shared/koba_shared/processor/tokenized_types.py#L62)
the comment says `[3, seq_len]`:

```python
# [3, seq_len] position ids
position_ids: torch.Tensor | None = None
```

But the trainer at
[`model.py:562`](lib/ursa/ursa/models/omni/model/model.py#L562) and
the docstring at line 334 (`shape (sequence_length, 3) - 3D RoPE
positions`) treat it as `(S, 3)`. The slicing
`position_ids[text_token_mask][:, 0]` only makes sense if axis 0 is the
sequence axis and axis 1 is the (T, H, W) axis. So the dataclass comment
is stale or wrong. Trivial fix — flag for the same parallel cleanup as
§11.6.3.

### 11.7 Apparent-but-not-real inconsistencies

Cases where greps surface a similar-looking name that turns out to be
unrelated to the omni stack:

| Pattern              | Where it appears                                                      | Why it's not a drift                                                     |
| -------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `text_mask`          | `lib/ursa/ursa/loss/*_test.py`, `lib/ursa/ursa/models/mmdit_test.py`   | Different model family (mmdit / older diffusion-loss code path), different convention |
| `is_image`           | `lib/ursa/ursa/infer_job.py` (~10 sites), `samplers`, `video_tasks/filters.py` | Boolean flag derived from `video.shape[0] == 1` for branching in samplers — orthogonal to per-token modality classification |
| `is_audio`           | (none in omni stack)                                                  | No false positive                                                         |
| `clean_mask` (7)     | unrelated test fixtures and modules                                   | False positive                                                           |
| `vis_mask` (3)       | unrelated module surfaces                                             | False positive                                                           |

### 11.8 Granularity / abbreviation pairs that take a beat to read

Three name pairs exist where the two members are semantically different but
the names don't telegraph the granularity hop:

| Pair                                                | Granularity                                                                   | Notes                                                                |
| --------------------------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `attention_mode` / `attention_modes` / `attn_modes` | per-element / per-element list / per-token list                               | Three different things. The abbreviation `attn` only appears on the per-token form by convention. |
| `sample_lens` / `split_lens`                        | per-sample inside a pack / per-element inside a sample                        | Both consistent within their scope, but the pair takes a beat to disentangle. |
| `vae_latent_shapes` / `video_vae_latent_shapes`     | per-element list `(H, W)` for image+audio / parallel field for video          | Same idea as `video_vae_token_mask` — video has a parallel field family at the model surface. |

These are stable across the codebase and not worth renaming on their own,
but worth flagging for first-time readers of this stack.

### 11.9 Implications for §10

Three small additions to fold into §10 when we merge the sections:

1. **Add `aut_token_mask` to phase 0.** Pure addition; no consumer reads
   it on the audio side today, but reserves the name and avoids a
   dual-edit hazard once the §5 audio graft lands. Pairs with
   `vit_token_mask` symmetrically.

2. **Document `video_vae_token_mask` as model-internal.** §10 currently
   keeps it as-is at the trainer surface. After phase 1, derive it at the
   model boundary from `vision_vae_token_mask & is_video_element` instead
   of having the data pipeline expose a parallel mask family. Phase 3 (or
   later) cleanup.

3. **Flag `clean_vae_img_mask` as a parallel cleanup, not part of §10.**
   The §10 rename is about per-token masks; `clean_vae_img_mask` is a
   misnamed per-element flag. Independent rename PR (preferred name:
   `is_clean_vae_element`). Coordinate timing if convenient (both touch
   per-element processors), but don't gate one on the other.

Plus the trivial fix:

4. **Correct the `position_ids` shape comment** from `[3, seq_len]` to
   `(seq_len, 3)` in `tokenized_types.py`. Trivially small; bundle into
   any of the above PRs.

