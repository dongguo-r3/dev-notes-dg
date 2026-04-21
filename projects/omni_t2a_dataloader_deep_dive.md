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
