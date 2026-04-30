# Omni T2A SequenceElement lifecycle — concrete walkthrough

> **Code-version anchor.** All line-number references in this document
> are pinned to lumaverse commit
> [`7fa0eb17a3c03f5386c0975f55ef7b6454405fd4`](https://github.com/lumalabs/lumaverse/tree/7fa0eb17a3c03f5386c0975f55ef7b6454405fd4)
> (branch `dongguo/omni-t2a-v2`, committed 2026-04-28).
> Line numbers may drift on later commits; if a referenced line doesn't
> match the file you're looking at, check out this commit to verify, or
> use the file path + symbol name (without line number) and re-locate
> with `grep`.

A stage-by-stage trace of one omni-T2A data sample, from raw Lance row to
the trainer's `model.forward` call. Companion to the broader
[`projects/omni_t2a_dataloader_deep_dive.md`](../projects/omni_t2a_dataloader_deep_dive.md)
(see §11.0 for the `SequenceElement` mental model, §8 for the
attention-mode / stream-mask story, and §10 / §12 for the mask-rename
design discussion). This document is the concrete answer to the question:
**where in the data pipeline does a `SequenceElement` come into
existence, where does its `num_tokens` get computed, and how does it
disappear at the trainer boundary?**

## Concrete example

One T2A sample with:

- Transcript: `"Once upon a time in a quiet village by the sea..."`
- Audio: 5 seconds at 16 kHz → `audio_tensor` of shape `(80000,)`
- `compression_factor = 512` (MMAudio default)
- `audio_register_token_amount = 0` (current default — registers off for audio)

By the end of the lifecycle, this sample becomes **2 elements totaling
~189 tokens** (≈30 text tokens + 159 audio tokens) ready for packing.

## Stage 0: raw row from Lance table

```python
sample = {
    "audio_bytes": <raw bytes>,
    "raw_transcript": "Once upon a time in a quiet village by the sea...",
    "duration": 5.0,
}
```

No `SequenceElement`s yet. The pipeline is a chain of processors that
each mutate `sample` in place.

> *Side note:* you might expect a `conversation_modality` field on the
> raw row. It is **not** in the Lance source — it is *injected* by an
> early-pipeline processor (next stage). Discussion entry
> [§1: `conversation_modality` provenance](#1-conversation_modality-provenance)
> covers where the value comes from and why.

## Stage 1: audio decode, normalize, and metadata injection

Producers: [audio_ops.py](../../lumaverse/lib/koba/koba/processor/audio_ops.py),
[audio_batching_ops.py](../../lumaverse/lib/koba/koba/processor/audio_batching_ops.py),
[omni_interleaved_packed_ops_refactor.py — `AddDummyConversationModality`](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py#L57-L66).

Three small per-sample mutations happen here, all before any
`SequenceElement` exists:

- `AudioDecoder.forward(sample)` decodes `audio_bytes` → `audio_tensor`
  of shape `(80000,)` (5 s × 16 kHz).
- `AudioToX.forward(sample)` normalizes (peak / RMS) and writes back to
  `sample["audio_tensor"]`. Shape unchanged.
- Optional bucketed-loader step: `_PadAudioToCeiling.forward(sample)`
  pads `audio_tensor` up to a bucket-uniform length (per
  [omni_t2a_packing_koba_v2.py:51-99](../../lumaverse/projects/kuma/kuma/projects/omni/audio/data/omni_t2a_packing_koba_v2.py#L51-L99)).
  Skipped here since we're using the non-bucketed path.
- `AddDummyConversationModality.forward(sample)` injects
  `sample["conversation_modality"] = "t2a"` (the value comes from the
  pipeline config, not the row). This is the field `OmniAudioSeqBuilder`
  reads next stage to dispatch to its T2A handler.
- Optional: `AppendDurationToTranscript.forward(sample)` may append
  `". 5.0 seconds long."` to the transcript when
  `audio_length_probability > 0`.

Still no `SequenceElement`s. Audio data lives at `sample["audio_tensor"]`;
transcript at `sample["raw_transcript"]`; the injected modality tag at
`sample["conversation_modality"]`.

## Stage 2: build the initial `sequence_plan`

Producer: [omni_audio_packed_ops.py:78-103](../../lumaverse/lib/koba/koba/processor/omni_audio_packed_ops.py#L78-L103).

`OmniAudioSeqBuilder.handle_t2a(sample)` produces the first
`SequenceElement` list:

```python
sample["sequence_plan"] = [
    SequenceElement(
        type=SequenceType.TEXT,
        text_str="Generate the following transcript:\nOnce upon a time in a quiet village by the sea...",
        loss=False,
        modality="t2a",
    ),
    SequenceElement(
        type=SequenceType.NOISY_VAE_AUDIO,
        media=Media(media_type="audio", data=audio_tensor),   # shape (80000,)
        loss=True,
        modality="t2a",
    ),
]
```

**This is the first time `SequenceElement`s exist.** The element list
captures the task's structure: prompt-text-then-audio, with loss only on
the audio. Each element carries its `type`, `modality`, `loss` flag, and
either `text_str` (for TEXT) or `media` (for NOISY_VAE_AUDIO). **No
`num_tokens` yet** — that is a `TokenizedSequenceElement` field,
materialized later.

> *Side note:* `type` and `modality` look similar but answer different
> questions. `type` (a `SequenceType` enum) drives processor dispatch
> and attention-mode selection; `modality` (a string) drives task-level
> behavior at sampling and loss attribution. See discussion entry
> [§2: `type` vs `modality` field design](#2-type-vs-modality-field-design)
> for the breakdown.

## Stage 3: pair each `SequenceElement` with a `TokenizedSequenceElement`

Producer: [omni_interleaved_packed_ops_refactor.py:511-528](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py#L511-L528).

`OmniAddTokenizedSequenceElement.forward(sample)` creates an empty
`TokenizedSequenceElement` for each `SequenceElement`, copying *only*
`type` and `modality`:

```python
sample["tokenized_sequence_plan"] = [
    TokenizedSequenceElement(type=TEXT,             modality="t2a"),  # everything else None
    TokenizedSequenceElement(type=NOISY_VAE_AUDIO,  modality="t2a"),  # everything else None
]
```

The pairing is positional: `tokenized_sequence_plan[i]` corresponds to
`sequence_plan[i]`. Subsequent processors (Stages 5–8) mutate the
tokenized side, leaving the `SequenceElement` side mostly untouched.

> *Side note:* this stage introduces a structural asymmetry — the SE
> list now carries the *content* (raw text strings, source media tensors)
> while the TSE list carries only *structural identity* (`type`,
> `modality`); every other TSE field is `None` until later stages fill
> it in. The SE / TSE separation is deliberate and is the basis for the
> "data builder vocabulary vs trainer vocabulary" mental model.
> Discussion entry
> [§3: `SequenceElement` vs `TokenizedSequenceElement`](#3-sequenceelement-vs-tokenizedsequenceelement)
> walks through three related questions: the SE-as-input / TSE-as-output
> framing, `sample` as universal accumulator, and the per-field
> asymmetry at end of this stage.

## Stage 4: optional CFG dropout

Producer: [omni_interleaved_packed_ops_refactor.py](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py).

`OmniCFGDropout.forward(sample)` — with probability `cfg_dropout_prob`
(default 0.1), drops conditioning by collapsing the sample to
`[empty_text] + [noisy_VAE_elements]` (see the actual transformation
in [omni_interleaved_packed_ops_refactor.py:574-606](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py#L574-L606)).
The dropout fires only for samples whose `conversation_modality` is in
`cfg_dropout_modalities` (default = a 29-entry allowlist; T2A overrides
to `["t2a"]` to avoid relying on the broad shared-lib default).

> *Side note:* the format assumption (drop-all is meaningful only for
> "pure-generation" task shapes), the multi-class CFG ecosystem
> (`OmniCFGDropout`, `OmniCFGDropoutLast`, `OmniCFGDropoutMixed`), and
> the per-pipeline class wiring are all covered in the dedicated
> "Special topic: CFG dropout" section below. T2A uses simple
> `OmniCFGDropout` because T2A is genuinely pure-generation
> (`[text] + [noisy_audio]`).

## Stage 5: per-element VAE processor — **`num_tokens` materializes here**

Producer: [omni_audio_ops.py:113-205](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L113-L205).

`OmniElementVAEAudio.forward(sample)` iterates `(og_element, tok_element)`
pairs and acts on the audio element:

```python
audio_tensor = og_element.media.data           # shape (80000,)
num_frames = audio_tensor.shape[-1]            # 80000
num_audio_tokens = math.ceil(80000 / 512)      # 157

# Build text_str (the wrapper string, single-token special tokens):
tokens = ["<|vision_start|>"]                  # 1 token
tokens += ["<|endoftext|>"] * 0                # M = 0 registers (today)
tokens += ["<|image_pad|>"] * 157              # 157 audio_pad slots
tokens += ["<|vision_end|>"]                   # 1 token
# Total: 159 tokens

tok_element.x_vae = audio_tensor.clone()
tok_element.x_vae_by_modality = "t2a"
tok_element.text_str = "".join(tokens)
tok_element.num_tokens = 157 + 0 + 2           # 159  ← MATERIALIZED HERE

# Per-token masks (all length 159):
tok_element.vae_token_mask        = ones(159);  vae_token_mask[0] = 0; vae_token_mask[-1] = 0
tok_element.text_token_mask       = zeros(159); text_token_mask[0] = 1; text_token_mask[-1] = 1
tok_element.txt_loss_mask         = zeros(159)

# Branch-specific (this is the noisy branch):
tok_element.clean_vae_img_mask    = 0          # per-element scalar (see deep-dive §11.6.3)
tok_element.noisy_vae_token_mask  = ones(159); noisy_vae_token_mask[0] = 0; noisy_vae_token_mask[-1] = 0
tok_element.clean_vae_token_mask  = zeros(159)
tok_element.attention_mode        = "noise"
```

Two things in that code worth pinning down explicitly:

- **The dispatch is on `tok_element.type`, not on `modality`.** The
  outer `if` at [omni_audio_ops.py:120-122](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L120-L122)
  filters elements whose `type ∈ {CLEAN_VAE_AUDIO, NOISY_VAE_AUDIO}`.
  Any future modality (e.g., `"a2a"`, `"i2a"`) producing a `NOISY_VAE_AUDIO`
  element would also flow through this processor; conversely, a
  hypothetical `"t2a"` element with a different `type` would not.
  `modality` is read at line 138 as a passthrough tag for
  `x_vae_by_modality` but doesn't gate dispatch.
- **`clean_vae_img_mask` is a per-element scalar bool (`1` for
  CLEAN_VAE_*, `0` for NOISY_VAE_*), not a per-token tensor.** The
  `_mask` suffix is a misnomer in the codebase's convention; this is a
  per-element flag the sampler uses to make per-element timestep-shift
  decisions. The audio side sets it (despite the misleading `_img_`
  infix) to mirror the image side's expectations — see deep-dive
  §11.6.3 for why this name is on the cleanup list.

Concretely for our 5-second / 16 kHz / `compression_factor=512` example:

| Quantity                    | Value                                |
| --------------------------- | ------------------------------------ |
| `audio_tensor.shape[-1]`    | 80000                                |
| `num_audio_tokens`          | `ceil(80000 / 512)` = 157            |
| `num_tokens`                | 157 + 0 (no registers) + 2 = **159** |
| `vae_token_mask.sum()`      | 157 (boundaries excluded)            |
| `text_token_mask.sum()`     | 2 (start + end)                      |
| `noisy_vae_token_mask.sum()`| 157                                  |
| `attention_mode`            | `"noise"`                            |

**`num_tokens` is computed at the per-element processor**, derived from
the element's media (audio_tensor's last-dim length) and the encoder's
known compression schedule (512 samples per audio token for MMAudio at
16 kHz). The 1-to-1 `SequenceElement ⇔ num_tokens` correspondence holds,
but the value is materialized here, not on the element at construction
time.

## Stage 6: per-element text processor

Producer: [omni_text_ops.py](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_text_ops.py).

`OmniElementText.forward(sample)` acts on the TEXT element, sets
`attention_mode = "causal"`, ensures `tok_element.text_str` is set, and
prepares fields for the tokenizer step that comes next.

## Stage 7: tokenize all text strings

Producer: `OmniQwen3Tokenizer` in
[omni_text_ops.py](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_text_ops.py).

`OmniQwen3Tokenizer.forward(sample)` runs the Qwen3 tokenizer over each
element's `text_str`:

- **TEXT element**: `"Generate the following transcript:\n..."` →
  tokenized to (say) 30 ids. `tok_element.input_ids = tensor([...30 ids...])`.
  Sets `text_token_mask=1` for the assistant span (or whole element
  depending on the loss flag). `num_tokens = 30`. (Exact count varies
  by transcript length; using 30 as a placeholder.)
- **NOISY_VAE_AUDIO element**:
  `"<|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>"`
  → tokenized to **exactly 159** ids (each special is a single token).
  `input_ids = tensor([151652, 151655, 151655, ..., 151655, 151653])`.
  The pre-set per-token masks from Stage 5 are preserved.

The contract enforced by `tokenizer_validation.py` (see deep-dive §8.5)
— every audio-role token string must tokenize to exactly one id — is
what makes the count come out to 159 instead of something else. If
`<|image_pad|>` ever tokenized to multiple BPE pieces, this stage would
silently misalign every per-token mask in the audio element.

After Stage 7, the sample has:

```python
sample["tokenized_sequence_plan"] = [
    TokenizedSequenceElement(type=TEXT,    num_tokens=30,  input_ids=..., text_token_mask=ones(30),  attention_mode="causal", ...),
    TokenizedSequenceElement(type=NOISY_VAE_AUDIO, num_tokens=159, input_ids=..., x_vae=..., text_token_mask=..., vae_token_mask=..., noisy_vae_token_mask=..., attention_mode="noise", ...),
]
```

Total: **189 tokens** in this sample.

## Stage 8: position IDs

Producer: `OmniPositionIDMRoPE` in
[position_ids_dev.py](../../lumaverse/lib/koba/koba/processor/position_ids_dev.py)
or `OmniPositionIDStableV0` in
[position_ids_stable_v0.py](../../lumaverse/lib/koba/koba/processor/position_ids_stable_v0.py).

The position-ID processor computes `position_ids` of shape
`(num_tokens, 3)` per element, with `(T, H, W)` axis values. For T2A:

- TEXT element positions 0..29: `(0, 0, 0), (1, 0, 0), ..., (29, 0, 0)`
  — sequential text axis only.
- NOISY_VAE_AUDIO element positions 30..188:
  `(30, 30, 30), (31, 31, 31), ..., (188, 188, 188)` — audio is 1-D, so
  all three axes get the same sequential value (per deep-dive §9.5).

(Exact wiring depends on the position-id processor variant; the
principle is "text axis is sequential through the whole sample, H/W
axes are flat for non-image content.")

## Stage 9: per-sample tokenized plan ready

The sample is now a fully-materialized `TokenizedSequencePlan`:

```python
sample = {
    "tokenized_sequence_plan": TokenizedSequencePlan(
        sequence_elements=[TSE_text, TSE_noisy_audio],   # 2 elements
        split_lens=[30, 159],                             # one per element
        ...
    ),
    # ... other fields
}
```

`split_lens` for this sample is `[30, 159]` — one entry per
`SequenceElement`. `sum(split_lens) = 189 = total token count`.

## Stage 10: pack across multiple samples

Producer: `pack_sequence` in
[sequence_packing.py](../../lumaverse/lib/ursa/ursa/models/omni/inference/sequence_packing.py).

The packer receives N samples (each with its own
`tokenized_sequence_plan`) and concatenates them. Suppose the pack has 3
T2A samples with element counts `[(30, 159), (28, 162), (32, 155)]`:

```python
packed = {
    "sample_lens":   [189, 190, 187],                              # one per sample, total = 566
    "split_lens":    [30, 159, 28, 162, 32, 155],                  # one per element, total = 566
    "attn_modes":    ["causal", "noise", "causal", "noise", "causal", "noise"],
    "input_ids":     tensor of shape (566,)                        # concatenated
    "position_ids":  tensor of shape (566, 3)                      # concatenated
    "text_token_mask":      bool tensor of shape (566,)
    "vae_token_mask":       bool tensor of shape (566,)
    "noisy_vae_token_mask": bool tensor of shape (566,)
    "txt_loss_mask":        bool tensor of shape (566,)
    "padding_mask":         bool tensor of shape (566,)
    "x_vae":                [audio_tensor_0, audio_tensor_1, audio_tensor_2]   # one per VAE element
    "vae_latent_shapes":    [None, None, None]                     # 1-D audio, no spatial
    "x_vae_by_modality":    ["t2a", "t2a", "t2a"]                  # one per VAE element
    "clean_vae_img_mask":   tensor([0, 0, 0])                      # per-VAE-element
    ...
}
```

The structural triplet `(sample_lens, split_lens, attn_modes)` plus all
the per-token masks are now ready for the trainer.

#### Three things worth pinning down explicitly at this stage

**1. `sample_lens` and `split_lens` have different lengths and don't
need pairwise alignment.** They live at different granularities:

| Field        | Length                                              | Indexed by         |
| ------------ | --------------------------------------------------- | ------------------ |
| `sample_lens` | N (number of samples in the pack)                   | per sample         |
| `split_lens` | sum over samples of (number of elements in sample) | per element across all samples |
| `attn_modes` | same as `split_lens` — paired 1-to-1 with elements  | per element        |

Code that wants to map a single `split_lens[i]` back to its source
sample CAN do so via a prefix-sum walk over `sample_lens`, but **no
production consumer needs this mapping**. The two lists feed orthogonal
parts of the sparse-mask construction (see Stage 11). The only
"alignment" between them is at construction time inside
`_connect_flattened_sequence_plans`
([sequence_packing.py:351-394](../../lumaverse/lib/ursa/ursa/models/omni/inference/sequence_packing.py#L351-L394)),
where each sample's `split_lens` are appended in sample order. After
that, they're consumed independently.

**2. Audio encoding (raw waveform → DAC latents) does NOT happen at
this stage.** Stage 10 packs raw audio waveforms (set by Stage 5 as
`tok_element.x_vae = audio_tensor.clone()`) into the per-batch list
`batch["x_vae"]`. The DAC encoder runs at Stage 11 inside the trainer
step, not here:

```python
# trainer.py:207-214
x_audio = [
    x for x, m in zip(batch["x_vae"], batch["x_vae_by_modality"])
    if m in AUDIO_MODALITIES
]
z0 = self._encode_audio(x_audio)        # ← VAE encoding happens here
```

So `batch["x_vae"]` after pack contains **input to the VAE encoder**,
not encoded latents. The encoded form `z0` is bound to a different
variable inside the trainer and the original `batch["x_vae"]` is freed
right after. (See deep-dive §11.6 / discussion §4 below for why this
naming is misleading.)

**3. The sparse attention mask is NOT built here either.** Stage 10
materializes the structural triplet `(sample_lens, split_lens,
attn_modes)`, which is the **input** to mask construction. The mask
itself is built at Stage 11 inside `model.forward` per batch — see the
next stage's discussion of `create_sparse_mask` for the per-token
broadcast of the triplet into `document_id` / `full_and_noise_seq_id`
/ `noise_seq_id`. The reason for this split: sparse-mask construction
needs the per-batch concatenated lengths (which only exist after pack)
AND runs at runtime on each forward call; staging it inside the data
pipeline would either re-do it every step (wasteful) or freeze a stale
mask (broken).

## Stage 11: trainer consumes the packed dict

Consumer: `OmniModel.forward` at
[model.py:470](../../lumaverse/lib/ursa/ursa/models/omni/model/model.py#L470).

`OmniModel.forward(...)` slices the packed sequence:

```python
text_position_ids = position_ids[text_token_mask][:, 0]   # length 30+28+32 + 6 boundaries (start/end of each audio elem) = 96
visual_position_ids = position_ids[vae_token_mask]        # length 157+160+153 = 470
```

The boundary tokens of each audio element (the 2 × 3 = 6
`<|vision_start|>` / `<|vision_end|>` positions) join the text path; the
audio_pad positions join the visual path.

#### Sparse-mask construction: two variable sets at two layers

There are two parallel sets of "sparse-attention-mask-shaping
variables" that show up in this pipeline, and they live at different
layers:

| Layer                          | Variables                                                | Provided by                                               | Granularity                            |
| ------------------------------ | -------------------------------------------------------- | --------------------------------------------------------- | -------------------------------------- |
| **Input to `create_sparse_mask`** | `sample_lens`, `split_lens`, `attn_modes`                | `_connect_flattened_sequence_plans` at Stage 10           | per-sample / per-element / per-element |
| **Internal to `create_sparse_mask`** (per-token broadcasts) | `document_id`, `full_and_noise_seq_id`, `noise_seq_id` | Built inside [flex_attn.py:24-59](../../lumaverse/lib/ursa/ursa/models/omni/model/flex_attn.py#L24-L59) | all per-token (length = total tokens)  |

The internal layer is **derived from the input layer at runtime**, by
broadcasting per-element / per-sample values across token positions.
The exact build:

```python
# flex_attn.py — broadcast the input triplet to per-token tensors
# document_id ← sample_lens
document_id = torch.cat(
    [torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]
).to(device)

# full_and_noise_seq_id, noise_seq_id ← (split_lens, attn_modes)
for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
    value       = i if mode in ["full", "noise"] else -1
    value_noise = i if mode == "noise"           else -1
    full_and_noise_tmp.extend([value]       * length)
    noise_tmp         .extend([value_noise] * length)
```

Then the four mask predicates (`causal_mask`, `full_and_noise_mask`,
`remove_noise_mask`, `sample_mask`) consume the per-token id tensors.
For the T2A pack of 3 samples [(30, 159), (28, 162), (32, 155)] from
the example above:

- `document_id`: length 566. Tokens of sample 0 get id 1, sample 1 get
  id 2, sample 2 get id 3.
- `full_and_noise_seq_id`: length 566. For the 6-element pack with
  `attn_modes` `["causal", "noise", "causal", "noise", "causal",
  "noise"]`, elements 1, 3, 5 get their indices; elements 0, 2, 4 get
  `-1`.
- `noise_seq_id`: identical to `full_and_noise_seq_id` for this T2A
  pack since no `"full"` elements exist.

The mask is then
`(causal_mask OR full_and_noise_mask) AND remove_noise_mask AND sample_mask`,
with the visibility behavior described in deep-dive §8.1 / §9.7.

So the two variable sets are **input** vs **per-token broadcast of the
input** — different shapes, different layers, but encoding the same
structural information. The input layer is what the data pipeline
produces; the broadcast layer is what FlexAttention's mask predicates
need.

## Summary table

| Stage | What happens                                                                  | Where `SequenceElement` lives                                                  | Token-count status                                          |
| :---: | :---------------------------------------------------------------------------- | :----------------------------------------------------------------------------- | :---------------------------------------------------------- |
| 0     | Lance row read                                                                | Doesn't exist yet                                                              | N/A                                                         |
| 1     | Audio decode + normalize                                                      | Doesn't exist yet                                                              | N/A                                                         |
| 2     | `OmniAudioSeqBuilder.handle_t2a` builds initial `sequence_plan`               | **Created** — 2 elements (TEXT, NOISY_VAE_AUDIO)                               | Not computed yet                                            |
| 3     | `OmniAddTokenizedSequenceElement` pairs with empty `TokenizedSequenceElement` | Same 2 elements; tokenized counterparts attached                               | Not computed yet                                            |
| 4     | `OmniCFGDropout` (optional)                                                    | Same 2 elements; text might be nulled                                          | Not computed yet                                            |
| 5     | `OmniElementVAEAudio` processes the VAE element                                | Same 2 elements; `tok_element.x_vae`, `text_str`, masks, `attention_mode` set  | **`num_tokens=159` materialized** for the audio element     |
| 6     | `OmniElementText` processes the TEXT element                                   | Same 2 elements                                                                | (text `num_tokens` from tokenizer in next stage)            |
| 7     | `OmniQwen3Tokenizer` tokenizes all `text_str`                                   | Same 2 elements                                                                | `num_tokens=30` materialized for the text element           |
| 8     | `OmniPositionID*` computes `position_ids`                                       | Same 2 elements                                                                | All set                                                     |
| 9     | Per-sample `TokenizedSequencePlan` finalized                                    | 2 elements, fully materialized                                                 | `split_lens = [30, 159]`, `sum = 189`                       |
| 10    | `pack_sequence` packs N samples                                                  | Element list flattens; identity preserved via `split_lens`                     | Pack totals: `sum(split_lens) = sum(sample_lens)` ≈ N × 189 |
| 11    | Trainer consumes packed dict                                                    | Element list now invisible at the model surface — only `split_lens`/`attn_modes` survive | Total seq length per pack                          |

## What this exercise reveals

Three things worth noting:

1. **The `SequenceElement` is born at Stage 2, lives unchanged in
   `type` and `modality` through every subsequent stage, and quietly
   disappears at the trainer boundary** — at Stage 11 the model only
   sees the flattened structural triplet (`sample_lens`, `split_lens`,
   `attn_modes`) plus per-token tensors. The element identity is never
   serialized to GPU; it lives only as the *implicit organizing
   structure* behind those flat tensors.
2. **`num_tokens` is computed twice along the way, by two different
   processors.** For the audio element, Stage 5
   (`OmniElementVAEAudio`) computes 157 audio tokens + 2 boundaries
   from the audio_tensor's frame count. For the text element, Stage 7
   (`OmniQwen3Tokenizer`) computes 30 from the actual tokenizer output.
   The Stage-5 audio count is *predicted* from the encoder's
   compression schedule and then *verified* by the tokenizer in Stage
   7 (because the audio element's `text_str` is constructed to tokenize
   to exactly that many ids). If the prediction and the tokenizer
   disagree, the per-token masks set in Stage 5 silently misalign with
   the actual tokens — the `tokenizer_validation.py` contract
   (deep-dive §8.5) is what prevents this drift.
3. **The audio element is one element with 159 tokens, not four
   elements (start + 0 registers + 157 pads + end).** Stages 5–7 all
   treat the boundaries, registers (when present), and pad slots as a
   single unit. This is the design choice deep-dive §10.4 documented
   and §11.0 emphasized: per-element processors emit one indivisible
   block per `SequenceType`, even though the per-token masks within it
   differ.

## Design notes & clarifications

This section pairs questions that arose while reviewing the walkthrough
with code-grounded answers. The 11 stages above stay lean; technical
details and design rationale that would otherwise weigh down the linear
flow live here. Each entry is self-contained and answers a specific
question; cross-references in the main stages point here by section
number. Read top-to-bottom for the full design context, or jump
directly to the entry referenced from the stage you're looking at.

### 1. `conversation_modality` provenance

**Question.** Is `sample["conversation_modality"] = "t2a"` read from the
Lance source row, or is it hardcoded somewhere?

**Answer: hardcoded in the pipeline config, not read from the table.**

The flow:

1. **Pipeline-level constant.** `default_t2a_pipeline_params()` at
   [default_t2a.py:148-149](../../lumaverse/lib/koba/koba/pipelines/default_t2a.py#L148-L149)
   defines:

   ```python
   conversation_modality_key="conversation_modality",
   conversation_modality_value="t2a",
   ```

2. **Processor injection.** `AddDummyConversationModality.forward(sample)`
   at [omni_interleaved_packed_ops_refactor.py:57-66](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py#L57-L66)
   writes the value into the sample dict:

   ```python
   sample[self.config.conversation_modality_key] = self.config.conversation_modality_value
   #     ↑ "conversation_modality"                  ↑ "t2a"
   ```

The `Dummy` in the class name is the giveaway — the field is *injected*
into the sample, not read from it. Every T2A pipeline run tags every
sample with `"t2a"`; T2I runs use `"t2i"` (default in the same Config
class); T2V uses `"t2v"`; etc.

This design has two implications:

- The Lance schema doesn't need a `conversation_modality` column. The
  same Lance table can be read by different pipelines, each tagging with
  its own task identifier.
- `OmniAudioSeqBuilder` (Stage 2) reads `conversation_modality` to
  dispatch to the right `handle_<task>` method
  ([omni_audio_packed_ops.py:108-128](../../lumaverse/lib/koba/koba/processor/omni_audio_packed_ops.py#L108-L128)).
  So the value injected here is the input that selects which sequence
  plan gets built.

### 2. `type` vs `modality` field design

**Question.** Both fields appear on every `SequenceElement` and look
similar — what's the difference, and why are they both needed?

**Answer: they answer different questions and drive different downstream
decisions.**

| Field      | Type             | What it answers                                                                                                | Drives                                                                                                                                                                              |
| ---------- | ---------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`     | `SequenceType` enum | **What is the structural role of this element?** TEXT / NOISY_VAE_IMAGE / CLEAN_VAE_IMAGE / NOISY_VAE_AUDIO / VIT_IMAGE / etc. — encodes (data-modality + branch) jointly. | Which per-element processor handles it (TEXT → `OmniElementText`; NOISY_VAE_AUDIO → `OmniElementVAEAudio` noisy branch / "noise" attention mode), per-token mask layout, attention mode |
| `modality` | `str`            | **What task does this element belong to?** `"t2a"`, `"t2i"`, `"i2i"`, `"image_edit"`, `"a2a"` (future), etc.    | Sampler-time per-task branching, timestep shift schedule (per the omni-data docs), encoder-internal dispatch via `x_vae_by_modality`                                               |

Concrete example showing they're orthogonal — same `type` can appear
under different `modality`'s, and same `modality` can have multiple
`type`'s in one sample:

| `type`              | `modality` | Meaning                                                                  |
| ------------------- | ---------- | ------------------------------------------------------------------------ |
| `NOISY_VAE_AUDIO`   | `"t2a"`    | Audio output of a text→audio task (current omni-t2a)                      |
| `NOISY_VAE_AUDIO`   | `"a2a"`    | Audio output of an audio→audio task (future, after deep-dive §5.3 lands)  |
| `NOISY_VAE_IMAGE`   | `"t2i"`    | Image output of a text→image task                                         |
| `NOISY_VAE_IMAGE`   | `"i2i"`    | Image output of an image-edit task                                        |
| `TEXT`              | `"t2a"`    | The text prompt of a text→audio sample                                    |
| `TEXT`              | `"t2i"`    | The text prompt of a text→image sample (same `type`, different task)      |


In the T2A walkthrough's Stage 2, both elements (TEXT and
NOISY_VAE_AUDIO) share `modality="t2a"` — they belong to the same
task. Their `type` differs because they play different structural roles
within that task.

`type` is consumed by per-element processors and by the structural
triplet `(sample_lens, split_lens, attn_modes)`; `modality` is consumed
by sampler / loss code that needs task-level branching. Same element
can appear in multiple tasks with the same `type` but different
`modality`.

#### Where `modality` is actually consumed (and where it isn't)

A more complete map of what `modality` drives, especially relevant for
understanding why the TEXT element of a T2A sample has
`modality="t2a"` even though the field's per-element granularity might
suggest more:

| Use of `modality`                                                   | Where it fires                                                                                                                                              | Today's status                                          |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Per-VAE-element encoder dispatch (audio VAE vs image VAE)            | `OmniElementVAEAudio` writes `tok_element.x_vae_by_modality = og_element.modality` ([omni_audio_ops.py:138](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L138)); trainer's `_encode_audio` filters `batch["x_vae"]` by it ([trainer.py:207-211](../../lumaverse/projects/kuma/kuma/projects/omni/audio/trainer.py#L207-L211)) | Active                                                  |
| Per-sample task identifier propagated up                              | `flatten_sequence_plan` reads `sequence_plan[0].modality` ([sequence_packing.py:282](../../lumaverse/lib/ursa/ursa/models/omni/inference/sequence_packing.py#L282)); becomes `batch["modalities"]` (per-sample list) at pack time ([sequence_packing.py:352](../../lumaverse/lib/ursa/ursa/models/omni/inference/sequence_packing.py#L352)) | Active                                                  |
| Sampler-time per-task branching (timestep schedules, CFG variants)   | Inference samplers read per-sample modality (e.g., [tdm_sampler.py:145](../../lumaverse/lib/ursa/ursa/models/omni/inference/tdm_sampler.py#L145))             | Active                                                  |
| Loss-path gating (audio MSE branch runs iff audio elements present)   | Indirectly via encoder dispatch → `has_audio = len(z0_seq) > 0` ([bagel_t2a.py:205](../../lumaverse/projects/kuma/kuma/projects/omni/audio/losses/bagel_t2a.py#L205)) | Active                                                  |
| CFG dropout per-task allowlist (`cfg_dropout_modalities`)             | `OmniCFGDropout.forward` filters by `sample["conversation_modality"]`, not by per-element `modality` — but the pipeline-level value comes from the same source | Active (see CFG special-topic section)                   |
| Per-token loss masking                                                | **Not used.** Per-token masks (`txt_loss_mask`, `noisy_vae_token_mask`) are set by per-element processors based on `type` + `loss` flag, not `modality`        | (intentional — `modality` is per-element, not per-token) |
| Per-modality loss weighting in combined losses                         | Loss classes; T2A uses scalar `diffusion_loss_weight` only                                                                                                  | Not used today; would matter for mixed-task training     |

#### What "task-level branching" looks like in code

The phrase covers four concrete things:

1. **Encoder selection.** The trainer's audio-encoder filter is
   modality-driven: only entries whose `x_vae_by_modality` is in
   `AUDIO_MODALITIES` get sent to DAC ([trainer.py:209-211](../../lumaverse/projects/kuma/kuma/projects/omni/audio/trainer.py#L209-L211)).
2. **Loss-class selection.** The trainer instantiates a per-task loss
   class (`BagelT2ALoss` for T2A, `BagelT2ILoss` for T2I, etc.). Inside
   each, modality dictates which branch fires — for T2A,
   `if audio_out is not None and has_audio:` gates the only loss path.
3. **Inference-time per-task decisions.** Samplers like
   `tdm_sampler.py:145` switch behavior on per-sample modality (timestep
   shift schedules, CFG variant, etc.).
4. **CFG dropout per-task gating.** `OmniCFGDropout` consults the
   sample-level `conversation_modality` (= same value as per-element
   modality, by current design) to decide whether to apply dropout
   (covered in the CFG special-topic section).

#### Specific to the T2A case

For the **TEXT element** of a T2A sample, `modality="t2a"` is read in
exactly one place at the data-pipeline level:
[`flatten_sequence_plan`](../../lumaverse/lib/ursa/ursa/models/omni/inference/sequence_packing.py#L282)
reads `sequence_plan[0].modality` to set the per-sample modality. The
TEXT element is element 0 of the sample, so its modality donates the
sample's task tag. That tag then surfaces at the batch level as
`batch["modalities"][i] = "t2a"` for sample i — consumed by the
sampler / loss class downstream.

For the **NOISY_VAE_AUDIO element** of the same sample, `modality="t2a"`
plays a different and more active role: `OmniElementVAEAudio.forward`
copies it to `tok_element.x_vae_by_modality`, which the trainer then
filters by to send the right tensors to the audio encoder
([trainer.py:209-211](../../lumaverse/projects/kuma/kuma/projects/omni/audio/trainer.py#L209-L211)).

So the user's observation that "modality on TEXT seems mostly
annotation" is right at the **data-pipeline-stage level** — for the
TEXT element, the value is just a per-sample tag waiting to be
read by `flatten_sequence_plan`. It becomes an active dispatch input
only at the trainer / sampler stage.

#### Per-sample uniformity invariant

Today, **all `SequenceElement`s within one sample share the same
`modality`** value. The invariant is enforced by construction:

1. `AddDummyConversationModality.forward(sample)` writes a single
   per-sample value (`sample["conversation_modality"]`) at Stage 1.
2. The SeqBuilder reads that single value and stamps it on every
   `SequenceElement` it constructs (Stage 2).
3. Even CFG dropout (which inserts new TEXT elements) preserves the
   existing modality.

This is why the `flatten_sequence_plan` shortcut "use element 0's
modality as the sample's modality" works correctly — within any sample
today, `modality` is uniform. The field is **per-element** at the
schema level, not because elements vary in modality, but because:

- Per-element passthrough convenience (each element carries its own
  modality, no need to look up the sample's modality from elsewhere).
- Architectural extensibility — heterogeneous-modality samples could
  exist in the future (e.g., a sample that mixes "t2i" and "i2t"
  elements within one multi-turn conversation), and the field is
  already shaped to support that.
- Decoupling the SequenceElement list from `sample["conversation_modality"]`,
  so that once the element list is built, it's self-contained.

So the rule of thumb: **`modality` is per-sample-uniform today, kept
per-element for forward compatibility; `type` is per-element-heterogeneous
always.**

### 3. `SequenceElement` vs `TokenizedSequenceElement`

Three closely related questions about the SE / TSE pair, answered
together because they share the same underlying design.

#### Q3a. Is SE the input-side placeholder and TSE the processed-side placeholder?

**Yes, mostly correct — with one wrinkle.** The split is the "two-stage"
interpretation:

- **`SequenceElement` (SE)** — the *semantic plan* abstraction. The
  data builder (`OmniAudioSeqBuilder.handle_t2a`) emits a list of these.
  Each carries `type`, `modality`, `loss`, plus a *source* (`text_str`
  for TEXT elements, `media` for VAE/ViT elements). It is the data
  builder's vocabulary.
- **`TokenizedSequenceElement` (TSE)** — the *processed-tensor*
  abstraction. The per-element processors (`OmniElementVAEAudio`,
  `OmniElementText`, ...) populate these with `num_tokens`, `input_ids`,
  all the per-token masks, `attention_mode`, etc. It is the trainer's
  vocabulary.

**The wrinkle:** per-element processors *also* mutate the SE side for
some derived fields. Specifically, `OmniElementVAEImage.forward` at
[omni_vae_ops.py:60](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_vae_ops.py#L60)
and the `OmniElementVAEAudio` analog
([omni_audio_ops.py:150](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L150))
write back to `og_element.text_str` after building the wrapper string:

```python
og_element.text_str = "".join(tokens)         # SE side gets the wrapper text
tok_element.text_str = "".join(tokens)        # TSE side gets the same
```

> *Side note:* what `OmniElementVAE*` writes to `text_str` is **only the
> per-element span wrapper** (`<\|vision_start\|>` + placeholders +
> `<\|vision_end\|>`), not any kind of prompt template. Task instructions
> (e.g., "Generate the following transcript:\n…") are constructed
> earlier by the SeqBuilder at Stage 2; chat-format role keywords (if
> any) live in the SeqBuilder's `text_str` for TEXT elements; per-text
> wrappers (`<\|im_start\|>` / `<\|im_end\|>`) are added later by the
> tokenizer at Stage 7. The four-layer decomposition is in the special
> topic "Prompt templating: a layered system" below.

So strictly, SE isn't *immutable* / *input-only* after Stage 2 — its
`text_str` field gets written by the per-element processor. But for
*information flow*, the SE → TSE distinction is exactly the right mental
model.

A cleaner restatement of the rule: **SE carries data needed to *build*
the per-token tensors; TSE carries the per-token tensors themselves.**
Some fields (like `text_str` for VAE elements) appear on both sides
because they're constructed by the per-element processor in Stage 5
and then needed downstream by the tokenizer step in Stage 7.

#### Q3b. Is `sample` (the dict) the universal accumulator carrying both lists?

**Yes, exactly right.** Every processor in the chain mutates a single
`sample: dict` in place. By Stage 9 the dict has accumulated:

```python
sample = {
    # Stage 0 — original Lance row fields:
    "audio_bytes":     <bytes>,                  # may be popped after decode
    "raw_transcript":  "Once upon a time...",
    "duration":        5.0,
    # Stage 1 — injected by AddDummyConversationModality:
    "conversation_modality": "t2a",
    # Stage 1 — added by AudioDecoder + AudioToX:
    "audio_tensor":    Tensor(80000),
    # Stage 2 — added by OmniAudioSeqBuilder:
    "sequence_plan":            [SE_text, SE_noisy_audio],
    # Stage 3 — added by OmniAddTokenizedSequenceElement:
    "tokenized_sequence_plan":  [TSE_text, TSE_noisy_audio],
    # Stages 5–8 — TSEs inside tokenized_sequence_plan get mutated
    # (masks / ids / positions filled in)
    # ...
}
```

So `sample` is **the entire state of one training row** at any point in
the pipeline. Processors are conceptually pure functions of
`sample → sample`, conventionally written as in-place mutations.

One thing that *isn't* on `sample` until later: the structural triplet
`(sample_lens, split_lens, attn_modes)` and the concatenated per-token
tensors. Those are produced by `pack_sequence` (Stage 10) when N samples
(each with its own `sample` dict) are flattened into one packed dict.

A sharper version: **before `pack_sequence`, `sample` carries everything
for one row; after `pack_sequence`, the packed dict carries everything
for the batch.** Once flattened, the per-sample dicts are no longer used
directly.

#### Q3c. At end of Stage 3, what does each list actually contain?

**Asymmetric:** the SE list carries content; the TSE list is mostly
empty placeholder.

`OmniAddTokenizedSequenceElement.forward` at
[omni_interleaved_packed_ops_refactor.py:511-528](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py#L511-L528):

```python
def forward(self, sample: dict):
    sequence_plan: list[SequenceElement] = sample["sequence_plan"]
    tokenized_sequence_plan: list[TokenizedSequenceElement] = []
    for og_element in sequence_plan:
        tok_element = TokenizedSequenceElement(
            type=og_element.type,           # only these two fields are set
            modality=og_element.modality,
        )
        tokenized_sequence_plan.append(tok_element)
    sample["tokenized_sequence_plan"] = tokenized_sequence_plan
    return sample
```

Only `type` and `modality` are copied. Every other field on
`TokenizedSequenceElement` stays at its dataclass default — `None`.

So at the end of Stage 3 the two lists are markedly asymmetric:

| Field                           | `sequence_plan[i]` (SE)               | `tokenized_sequence_plan[i]` (TSE)      |
| ------------------------------- | ------------------------------------- | --------------------------------------- |
| `type`                          | `SequenceType.TEXT` / `NOISY_VAE_AUDIO` | Same (copied)                          |
| `modality`                      | `"t2a"`                               | Same (copied)                          |
| `loss`                          | `False` / `True`                       | (no such field on TSE)                  |
| `text_str` (for TEXT element)   | `"Generate the following transcript:\n..."` | **`None`**                       |
| `media` (for NOISY_VAE_AUDIO)   | `Media(media_type="audio", data=Tensor)` | (no such field on TSE)              |
| `num_tokens`                    | (no such field on SE)                  | **`None`**                              |
| `x_vae`                         | (no such field on SE)                  | **`None`**                              |
| All per-token masks             | (no such fields on SE)                 | **all `None`**                          |
| `input_ids`                     | (no such field on SE)                  | **`None`**                              |
| `attention_mode`                | (no such field on SE)                  | **`None`**                              |

Sharper restatement: **at end of Stage 3, the SE list carries the
*content* (raw text strings, source media tensors), and the TSE list
carries only *structural identity* (type, modality) — everything else
on TSE is `None`, waiting to be filled in by later stages.**

The fill-in then happens stage by stage:

| Stage | Processor                | TSE fields populated                                                       |
| :---: | ------------------------ | -------------------------------------------------------------------------- |
| 5     | `OmniElementVAEAudio`    | `x_vae`, `text_str`, `num_tokens`, all per-token masks, `attention_mode`   |
| 6     | `OmniElementText`        | `text_str`, `attention_mode`                                                |
| 7     | `OmniQwen3Tokenizer`     | `input_ids`, `padding_mask`, finalizes `num_tokens` for text elements       |
| 8     | `OmniPositionID*`        | `position_ids`                                                              |

By Stage 9, the TSE list is fully populated; the SE list has been read
but is no longer the source of truth — the trainer at Stage 11 only
consumes TSE-derived per-token tensors.

## Special topic: CFG dropout in the omni data pipeline

CFG (classifier-free guidance) dropout is the most counter-intuitive
piece of the omni data pipeline. It shows up at Stage 4 of the
walkthrough as a single-line `OmniCFGDropout.forward(sample)` call, but
the actual design space is broader and the per-pipeline routing carries
more weight than the per-class defaults. This section consolidates a
multi-round design discussion into one reference. It is grounded in
[`lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py`](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py)
plus the per-task pipelines under
[`lib/koba/koba/pipelines/`](../../lumaverse/lib/koba/koba/pipelines/).

If you only remember three things from this section:

1. There are three CFG-dropout classes with **distinct semantics**, not
   just configuration knobs. Picking the wrong one for a task is a
   silent correctness bug, not a runtime error.
2. The pipeline-level **choice of class** determines correctness; the
   class-level `cfg_dropout_modalities` list is a near-vestigial runtime
   filter that almost never excludes anything in practice.
3. **Drop-all CFG is conceptually wrong for editing tasks**: applying it
   to an editing sample replaces the editing task with an
   unconditional-generation task. Editing tasks need a partial-conditioning
   variant.

### What CFG dropout is supposed to do

Classifier-free guidance trains the model to produce two predictions
side by side:

- A **conditional** prediction: `p(target | full_conditioning)`.
- An **unconditional** prediction: `p(target | empty_conditioning)`.

At inference, CFG combines them as `pred = uncond + cfg_scale * (cond - uncond)`,
amplifying the conditional signal. To train this, ~10% of training
samples (`cfg_dropout_prob` default) are randomly transformed to drop
their conditioning so the model sees both views.

The "drop conditioning" step is what each CFG-dropout class implements,
and the design choice of *what counts as conditioning* — and therefore
what gets dropped — is where the three classes diverge.

### The three CFG-dropout classes

All three live in
[`omni_interleaved_packed_ops_refactor.py`](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py).

| Class                        | Code                                                                                                                                                  | What it preserves                                       | What it drops                                                                                | Right for which task shape                                              |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `OmniCFGDropout` (drop-all)  | [`omni_interleaved_packed_ops_refactor.py:531-606`](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py#L531-L606)        | Only `NOISY_VAE_*` elements                              | All `TEXT`, `VIT_IMAGE`, `CLEAN_VAE_*`, prior-turn elements; replaced by single empty TEXT     | Pure generation: `[text] + [noisy_target]` (T2I, T2A, SISO)             |
| `OmniCFGDropoutLast` (drop-trailing-text-only) | [`:609+`](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py#L609)                            | All non-text conditioning (CLEAN_VAE, VIT, prior turns)  | Picks one noisy target, truncates after it; blanks ALL consecutive TEXT/TEXT_INCOMPLETE elements immediately before it | Editing / multi-turn: keep image inputs, drop only the instruction text |
| `OmniCFGDropoutMixed` (probabilistic blend) | [`:740-840`](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py#L740)                            | Depends on which branch fires                            | With prob `drop_all_ratio × cfg_dropout_prob` runs `OmniCFGDropout`; else runs `OmniCFGDropoutLast` | Editing tasks that benefit from BOTH unconditional branches at inference  |

A worked example for `image_edit` with original sequence
`[VIT_IMAGE(input), TEXT(edit_cmd), NOISY_VAE_IMAGE(target)]`:

| Variant fires       | Resulting sequence                                                | Trained `p(target | …)`                          |
| ------------------- | ----------------------------------------------------------------- | ------------------------------------------------ |
| `OmniCFGDropout`    | `[TEXT(""), NOISY_VAE_IMAGE(target)]` — input image gone           | `p(target | empty_conditioning)` — fully unconditional |
| `OmniCFGDropoutLast`| `[VIT_IMAGE(input), TEXT(""), NOISY_VAE_IMAGE(target)]`            | `p(target | input_image, empty_text)` — text-only dropped |
| `OmniCFGDropoutMixed` | one of the above two, at `drop_all_ratio` ratio                | both, in fixed proportion                        |
| (no fire, 90% case) | unchanged: full editing sample                                    | `p(target | input_image, edit_cmd)` — the conditional |

### Pipeline-by-pipeline class assignment

This is the layer that actually determines runtime behavior. Each
pipeline picks one CFG-dropout class explicitly, and **only T2A and the
bagel registry configs override `cfg_dropout_modalities`**; every other
pipeline inherits the class's broad 29-entry default list.

| Pipeline                                                                                                                                | CFG class               | Overrides `cfg_dropout_modalities`?                                                          | Comment                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`default_t2i.py:75`](../../lumaverse/lib/koba/koba/pipelines/default_t2i.py#L75)                                                       | `OmniCFGDropout`        | No (uses default)                                                                            | `t2i` is in the default list — drop-all fires correctly for T2I samples                                                                                  |
| [`default_siso.py:58`](../../lumaverse/lib/koba/koba/pipelines/default_siso.py#L58)                                                     | `OmniCFGDropout`        | No                                                                                            | SISO has `[text] + [noisy_image]` shape — drop-all is correct                                                                                            |
| [`default_t2a.py:103-108`](../../lumaverse/lib/koba/koba/pipelines/default_t2a.py#L103-L108)                                            | `OmniCFGDropout`        | **Yes — `["t2a"]`**                                                                          | T2A is the only pipeline that explicitly tightens. Comment in source: *"T2A-only pipeline — restrict dropout to t2a samples so we don't rely on the shared-lib default modality list."* |
| [`default_i2t.py:47`](../../lumaverse/lib/koba/koba/pipelines/default_i2t.py#L47)                                                       | `OmniCFGDropout`        | No                                                                                            | Image-to-text task                                                                                                                                       |
| [`t2i_grounding.py:186`](../../lumaverse/lib/koba/koba/pipelines/t2i_grounding.py#L186)                                                 | `OmniCFGDropoutMixed`   | No                                                                                            | T2I with grounding map — Mixed gives both unconditional branches                                                                                          |
| [`image_edit_supermax.py:10`](../../lumaverse/lib/koba/koba/pipelines/image_edit_supermax.py#L10)                                       | `OmniCFGDropoutMixed`   | No                                                                                            | Editing task — Mixed routes correctly                                                                                                                    |
| [`storyboard.py:82`](../../lumaverse/lib/koba/koba/pipelines/storyboard.py#L82)                                                         | `OmniCFGDropoutMixed`   | No                                                                                            | Multi-turn task — Mixed routes correctly                                                                                                                 |
| [`dataset_config_registry_vl_drop_combine.py:108`](../../lumaverse/projects/kuma/kuma/projects/omni/bagel/datasets/dataset_config_registry_vl_drop_combine.py#L108) | (any) | **Yes — `[]` (empty list)** | Empty list → CFG dropout **never fires**. Used by some bagel VL configs to disable CFG entirely. |
| [`dataset_config_registry_vl_cfg.py:114`](../../lumaverse/projects/kuma/kuma/projects/omni/bagel/datasets/dataset_config_registry_vl_cfg.py#L114) | (any) | **Yes — `[]`** | Same pattern, also disables                                                                                                                              |

The implicit invariant the codebase relies on: **each pipeline processes
one modality at a time** (set via
`AddDummyConversationModality.conversation_modality_value`), so the
class-level modality list never has to filter cross-modality samples.

### Why drop-all CFG is conceptually wrong for editing tasks

This is the subtlest part of the design and the one that took the
longest to clarify. The argument:

For pure-generation tasks (T2I, T2A, SISO), the unconditional branch
`p(target | empty_conditioning)` is meaningful — it asks "what does the
model produce from nothing?" — and CFG correctly amplifies the conditional
signal at inference.

For editing tasks (`image_edit`, `multiref`, `storyboard_inter`,
`character_edit`, `multiview`, `upres`, `style_transfer`, ...), the
*task itself* is impossible without the input conditioning. Asking
"what does the model produce from nothing?" is well-defined as a
*different* task — generic image generation in the style of edit
outputs — but it isn't a useful unconditional reference for editing.
At inference, the CFG difference `cond - uncond` would amplify *both*
the input image's effect and the edit instruction's effect together,
when usually you want to amplify only the instruction's effect while
preserving the input image's identity.

This leads to the cleanest reframe of the issue:

> **When `OmniCFGDropout` (drop-all) fires on an editing sample, the
> editing task is replaced with an unconditional-generation task on
> that step.** The 90% conditional path trains the editing skill; the
> 10% drop-all path trains a generic-generation skill that has no useful
> deployment for editing.

The fix is `OmniCFGDropoutLast` (preserve the input image, drop only
the trailing text), or `OmniCFGDropoutMixed` (probabilistic blend of
drop-all and drop-text). Editing pipelines wire `OmniCFGDropoutMixed`
explicitly — and that's why they don't suffer the failure mode despite
their modality strings appearing in `OmniCFGDropout`'s default list.

A useful mental rule:

| Task shape                                        | Conceptually right CFG variant                                                              |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `[text] + [noisy_target]` (pure generation)       | `OmniCFGDropout` (drop-all)                                                                  |
| `[non-text-conditioning..., text, noisy_target]` (editing, multi-modal) | `OmniCFGDropoutLast` or `OmniCFGDropoutMixed`                                              |
| Multi-turn `[..., text_n, noisy_n, ...]`          | `OmniCFGDropoutLast` or `OmniCFGDropoutMixed` (truncates + blanks the right text span)       |
| Tasks where CFG should be off entirely            | Set `cfg_dropout_modalities=[]` on whichever class is wired (the bagel pattern)              |

### Counter-intuitive things and known smells

Several things in this design are easy to misread on first encounter.

**1. The class-level default `cfg_dropout_modalities` list is mostly a
copy-paste smell, not a routing decision.** Both `OmniCFGDropout`
([:535](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py#L535))
and `OmniCFGDropoutMixed`
([:750](../../lumaverse/lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py#L750))
default to nearly the same 29-entry list:

```text
"t2i", "interleaved", "siso", "scv", "style_transfer", "style_transfer_icl",
"image_edit", "group", "multiref", "grounded_siso", "grounded_miso",
"multiview", "storyboard_inter", "storyboard_intra", "scv_plus", "t2i_hq",
"t2i_ep", "camera", "image_edit_tree", "image_edit_supermax", "brand_kit",
"t2i_grounding", "character_edit", "multiref_manga", "t2i_hq_human", "upres",
"t2i_hq_high_res", "multiref_high_res", "image_edit_supermax_high_res",
("storyboard_finetune" only in Mixed)
```

Roughly **8 entries are pure-generation tasks correct for drop-all**;
the other ~21 are editing-shape tasks correct only for partial-conditioning
CFG. The reason this is not a runtime bug: each pipeline is single-modality
(see invariant above) and each editing pipeline wires `OmniCFGDropoutMixed`
explicitly, so the wrong-class-for-that-modality combination is never
constructed in practice.

The discoverability cost is real, though: a new-pipeline author who
copies `default_t2i.py` and reuses its class choice for a multi-modality
batch can silently miscode editing samples.

**2. Most pipelines don't override `cfg_dropout_modalities`.** Despite the
problem above, only T2A and the bagel registry configs explicitly tighten
the list. Pipelines like `default_t2i.py`, `default_siso.py`,
`image_edit_supermax.py`, `storyboard.py`, `t2i_grounding.py` all inherit
the broad default. The runtime behavior is still correct because each
pipeline handles a single modality.

**3. The runtime safety check almost never filters anything.** The check is
`if modality not in self.cfg_dropout_modalities: return sample`. Since
each pipeline is single-modality and that modality is always in the
default list, the check passes 100% of the time in practice. It's
documentation by convention, not enforcement.

**4. `OmniCFGDropoutMixed` has unintuitive semantics from its name.** "Mixed"
might suggest "mixes conditional and unconditional"; it actually means
"probabilistic blend of `OmniCFGDropout` and `OmniCFGDropoutLast`". A
better name would be `OmniCFGDropoutProbabilisticBlend` or
`OmniCFGDropoutDualUnconditional`, but the current name is what's wired
across the codebase.

**5. Editing tasks `image_edit_*` etc. ARE in `OmniCFGDropout`'s default
list, but no editing pipeline actually wires `OmniCFGDropout`.** The
appearance of editing modality strings in that list does not mean
"drop-all CFG works for these tasks." It means "if a hypothetical pipeline
wired `OmniCFGDropout` for one of these tasks, the runtime check would
let drop-all fire" — which would be a silent correctness bug.

**6. Empty list = CFG disabled.** `cfg_dropout_modalities=[]` is the
explicit "off switch" used by some bagel VL configs. Easy to misread as
"unset" / "default to broad list." It is not — empty means *no modality
qualifies*, so the runtime check returns the sample unchanged every time.

### T2A specifics

T2A is the cleanest case in the codebase:

- T2A has `[text] + [noisy_audio]` shape — a pure-generation task by the
  classification above.
- Therefore `OmniCFGDropout` (drop-all) is the conceptually correct choice.
- T2A explicitly tightens `cfg_dropout_modalities=["t2a"]`, which:
  - Avoids relying on the broad shared-lib default that includes editing
    modalities (a legitimate concern even though it doesn't bite in
    single-modality pipelines today).
  - Makes the source self-documenting — a reader sees that this pipeline
    handles only T2A and applies drop-all only to T2A samples.

**A future `a2a` (audio-edit) or `i2a` (image-to-audio) task would NOT
use `OmniCFGDropout`.** Mirroring the image-side pattern:

| Hypothetical future audio task | CFG dropout class                  | Why                                                                  |
| ------------------------------- | ---------------------------------- | -------------------------------------------------------------------- |
| `t2a` (current)                 | `OmniCFGDropout` with `["t2a"]`     | Pure generation — drop-all is right                                   |
| `a2a` (audio-edit, future)      | `OmniCFGDropoutMixed` with `["a2a"]`| Editing — needs partial-conditioning CFG                              |
| `i2a` (image-to-audio, future)  | `OmniCFGDropoutMixed` with `["i2a"]`| Has image conditioning to preserve — needs partial-conditioning CFG   |
| Mixed-modality audio batches    | `OmniCFGDropoutMixed` with explicit list of all modalities served | Forces explicit per-pipeline declaration, avoids inherited-default bugs |

### Cleanup directions (open issues, not on critical path)

Three options of increasing aggressiveness, from the design discussion:

1. **Trim the class-level defaults to semantically correct subsets.**
   `OmniCFGDropout`'s default list keeps only pure-generation modalities
   (`t2i`, `t2i_hq`, `t2i_ep`, `t2i_hq_human`, `t2i_hq_high_res`, `siso`,
   `scv`, `scv_plus` — about 8 entries). `OmniCFGDropoutMixed`'s default
   list keeps the editing-shape modalities. No behavior change for current
   pipelines; clarifies intent for future readers.
2. **Empty the defaults and force per-pipeline explicit override.** Every
   pipeline that uses any CFG-dropout class must declare its modalities,
   like T2A already does. Removes the discoverability smell entirely.
   Bigger migration but cleanest end state.
3. **Add a runtime guard.** When `OmniCFGDropout.forward` runs on a
   sample whose `sequence_plan` contains non-trivial conditioning
   (CLEAN_VAE, VIT, multi-turn), warn or raise. Catches misuse loudly
   when a pipeline mistakenly wires the wrong class. Lightest touch but
   only catches misuse rather than fixing the design.

Practical recommendation: option 1 is the lowest-risk, highest-clarity
change. It documents intent at the class level without altering runtime
behavior in any current pipeline. Option 2 is the principled end state
once option 1 lands and consumers have migrated.

## Special topic: special tokens and text-element wrapping

Special tokens in the omni data pipeline come from **two distinct
families** that get applied to **different element types** at
**different processing stages**, and the naming conventions inherited
from upstream Qwen3 don't match how the codebase actually uses them.
This section consolidates the design behind `<|im_start|>` /
`<|im_end|>` element wrapping and its relationship to the data-specific
span markers (`<|vision_start|>`, `<|image_pad|>`, etc.), including the
subtle CE-loss interactions at the wrap positions.

If you only remember three things from this section:

1. There are **two parallel wrapping systems**, each fired by a
   different processor at a different stage. They never collide on the
   same element — TEXT-family elements get one system, VAE/ViT
   elements get the other.
2. The `<|im_start|>` / `<|im_end|>` wrap is **unconditional and
   per-text-element**, not per-sample. There is no config flag to
   disable it, and no pretrain-vs-SFT mode switch — the "template" is
   determined entirely by the SeqBuilder's `text_str` content and the
   element layout.
3. The names `<|im_start|>` / `<|im_end|>` are inherited from Qwen3's
   chat-format convention but used by omni for a **different purpose**
   (per-element wrapping). The codebase variable name
   `QWEN3_BEGIN_OF_SENTENCE_TOKEN` adds a third interpretation. Reading
   the codebase requires holding all three in mind.

### Two classes of special tokens

The two systems differ in everything except being "special tokens":

| Class                                    | Token examples (string / id)                                                                                       | Set by                                                                                                                                                                                                                | Set at stage  | Granularity                | Purpose                                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----------: | -------------------------- | ------------------------------------------------------------------------------------------------ |
| **Data-specific span markers**           | `<\|vision_start\|>` (151652), `<\|vision_end\|>` (151653), `<\|image_pad\|>` (151655); audio aliases reuse the same ids today | `OmniElementVAEAudio` ([omni_audio_ops.py:143-149](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L143-L149)), `OmniElementVAEImage` ([omni_vae_ops.py:53-65](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_vae_ops.py#L53-L65)), `OmniElementVit` ([omni_vit_ops.py:33-49](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_vit_ops.py#L33-L49)) | **Stage 5**   | Per-VAE/ViT element        | Mark span boundaries and per-frame placeholders for media slots                                  |
| **Element wrapper markers**              | `<\|im_start\|>` (151644), `<\|im_end\|>` (151645)                                                                  | `OmniQwen3Tokenizer` ([omni_text_ops.py:184-211](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_text_ops.py#L184-L211))                                                                                  | **Stage 7**   | Per-TEXT-family element    | Mark per-element text-span boundaries; reused from Qwen3's chat-format ids but used as **opaque element wrappers**, not as chat-message turn boundaries |

The two are fired by completely different processors at different
stages and never overlap on the same element — one is for media-bearing
elements, the other is for text-bearing elements.

### The element wrapper: unconditional, per-text-element

[omni_text_ops.py:190-211](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_text_ops.py#L190-L211)
shows the wrap is added **unconditionally** for every `SequenceType.TEXT`
element:

```python
elif tok_element.type == SequenceType.TEXT:
    text_ids, padding_mask = self.encode_text(tok_element.text_str, ...)
    text_ids = (
        [self.config.im_start_id] + text_ids + [self.config.im_end_id]
    )
    padding_mask = [1] + padding_mask + [1]
```

No `if`-guard, no config flag, no modality-driven branching. Every
TEXT element gets `[<|im_start|>, ...content..., <|im_end|>]`. A sample
with multiple TEXT elements gets multiple pairs — one per text element,
not one per sample.

`SequenceType.TEXT_INCOMPLETE` (inference-only NTP path) gets a
slightly different asymmetric wrap:

```python
text_ids = ([self.config.im_start_id]
            + self.config.assistant_newline_id    # = [77091, 198] = "assistant\n"
            + text_ids)
```

So the inference format is `<|im_start|>assistant\n<content>` — start
marker plus an "assistant\n" role prefix, no closing `<|im_end|>`
(because the model is generating from this prefix and the closer will
be sampled at runtime). This is the **only** place the codebase actively
uses Qwen3-chat-format role keywords (the literal string "assistant"),
and it only fires at inference.

### Per-element-type behavior (full summary)

| `SequenceType`                                | Wrap added by `OmniQwen3Tokenizer` (Stage 7)            | Span markers from Stage 5                                            | Final ids in the element                                                                          |
| --------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `TEXT`                                        | `[im_start_id]` + content + `[im_end_id]` (both ends)    | none                                                                  | `[<\|im_start\|>, t₁, t₂, …, tₙ, <\|im_end\|>]`                                                   |
| `TEXT_INCOMPLETE` (inference NTP only)        | `[im_start_id, "assistant", "\n"]` + content (start-only) | none                                                                  | `[<\|im_start\|>, "assistant", "\n", t₁, t₂, …]` (open-ended)                                     |
| `NOISY_VAE_AUDIO` / `CLEAN_VAE_AUDIO`         | none — raw encode of existing `text_str`                  | `<\|vision_start\|>` / `<\|vision_end\|>` (today aliased for audio)   | `[<\|vision_start\|>, (registers,) <\|image_pad\|>×N, <\|vision_end\|>]`                          |
| `NOISY_VAE_IMAGE` / `CLEAN_VAE_IMAGE`         | none                                                      | `<\|vision_start\|>` / `<\|vision_end\|>`                              | `[<\|vision_start\|>, (registers,) <\|image_pad\|>×N, <\|vision_end\|>]`                          |
| `VIT_IMAGE` (Qwen path)                       | none                                                      | `<\|vision_start\|>` / `<\|vision_end\|>`                              | `[<\|vision_start\|>, <\|image_pad\|>×N, <\|vision_end\|>]`                                       |
| `VIT_IMAGE` (legacy non-Qwen path)            | none                                                      | none                                                                  | `[<\|image_pad\|>×N]` (no boundaries — being phased out)                                            |
| `VIT_AUDIO` (future, after audio-graft §5.1) | none                                                      | (TBD; audio-understanding analog)                                     | (TBD)                                                                                              |
| `PACKED`                                       | none (already-packed plan, no per-element re-wrap)        | n/a                                                                   | n/a                                                                                                |

The two wrap families are **disjoint by element type**:
TEXT-family → `<|im_start|>` / `<|im_end|>` from the tokenizer; VAE/ViT
family → `<|vision_start|>` / `<|vision_end|>` (or audio analogs) from
the per-element processor. Reading a tokenized packed sample, you can
identify each element's boundaries by which marker pair encloses it.

### Prompt templating: a layered system, not a single layer

The phrase "prompt template" naturally suggests a single layer where
you write a Jinja-style string with `{user}` / `{assistant}` slots and
the tokenizer fills them in. The omni data pipeline does not work this
way. Templating is **distributed across four stages**, each owning a
different piece of the final id sequence. None of the four is "the"
prompt template — they compose to produce one.

#### The four templating layers

| Layer | Stage | Owner                                                   | What it produces                                                                                                                          |
| :---: | :---: | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | 2     | per-task SeqBuilder Config + handler (e.g. [`OmniAudioSeqBuilder.handle_t2a`](../../lumaverse/lib/koba/koba/processor/omni_audio_packed_ops.py#L78-L103)) | **Task-instruction template** — the natural-language instruction string that gets embedded in a TEXT element's `text_str`. Per-task hardcoded format. |
| 2     | 2     | per-task SeqBuilder forward                              | **Element-layout template** — the *structure* of the sample: how many `SequenceElement`s, what `type` each has, what order they appear in. |
| 3     | 5     | `OmniElement{VAEAudio,VAEImage,Vit}` ([omni_audio_ops.py:143-149](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L143-L149), [omni_vae_ops.py:53-65](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_vae_ops.py#L53-L65), [omni_vit_ops.py:33-49](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_vit_ops.py#L33-L49)) | **Per-VAE/ViT span-marker template** — the wrapper string for media elements: `<\|vision_start\|>` + content placeholders + `<\|vision_end\|>`.  |
| 4     | 7     | `OmniQwen3Tokenizer` ([omni_text_ops.py:208-211](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_text_ops.py#L208-L211)) | **Per-text-element wrapper** — the unconditional `[<\|im_start\|>] + content + [<\|im_end\|>]` wrap added to every TEXT element.            |

The final packed sample's id sequence is the **concatenation of all
four layers' outputs**, applied at their respective stages on their
respective element types.

#### Concrete trace for the T2A walkthrough sample

| Layer | What this layer produced                                                                                          |
| :---: | ----------------------------------------------------------------------------------------------------------------- |
| 1     | `t2a_task_prompt = "Generate the following transcript:\n"` (constant in `OmniAudioSeqBuilder.Config`); SeqBuilder sets `TEXT.text_str = t2a_task_prompt + transcript` |
| 2     | `[TEXT(prompt, loss=False), NOISY_VAE_AUDIO(audio, loss=True)]` — two elements, prompt-then-target order           |
| 3     | For the audio element: `text_str = "<\|vision_start\|>" + "<\|image_pad\|>" × 157 + "<\|vision_end\|>"`             |
| 4     | For the TEXT element: tokenizer prepends `<\|im_start\|>` and appends `<\|im_end\|>` to the encoded prompt          |

The composed result is a single packed id sequence ≈ 189 tokens long
(30 text tokens including the wrap + 159 audio tokens), with each
layer's contribution sitting at a different position range.

#### Each layer is decided at a different abstraction level

What this layered design means in practice:

- **Layer 1 is the only place a human author writes natural-language
  template text.** For a new task, you write something like
  `"Describe the audio:\n"` or
  `"Transcribe and translate to French:\n"` and the per-task SeqBuilder
  Config holds it as a constant. Different tasks have different Layer-1
  templates; the `OmniAudioSeqBuilder` Config has one
  (`t2a_task_prompt`), an image-edit SeqBuilder has another, etc.
- **Layer 2 (element layout) decides what the model "sees" in what
  order.** For multi-turn / multi-modal tasks, this is where the
  conversational structure lives. For T2A it is dead simple
  (prompt-then-target). For an A2T or I2T task it would be
  `[input_media, TEXT(answer, loss=True)]`. For multi-turn editing it
  could be `[CLEAN_VAE(input), VIT(input), TEXT(cmd), NOISY_VAE(out)]`.
- **Layer 3 (span markers) is element-type-specific and hardcoded into
  the per-element processors.** You don't customize `<|vision_start|>`
  per task; it's always the same wrapper for every VAE / ViT element.
  Audio / video aliasing of these strings (per §8.5 of the deep-dive)
  is the only knob.
- **Layer 4 (text wrap) is uniform across the entire codebase.** Every
  TEXT element gets the same `<|im_start|>` / `<|im_end|>` wrap. There
  is no per-task override and no config flag to disable it.

#### Where chat-format-style templating *would* live

If you wanted to make a task look like Qwen3 chat format (with
`user\n` / `assistant\n` role keywords), the work happens in **Layer 1
or Layer 2**, not Layer 4. Two approaches:

| Approach                                              | Where the work goes                                                                                                                              | Resulting structure                                                                                                                                                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Render the chat template into `text_str` (Layer 1)    | The SeqBuilder constructs `text_str = "user\nWhat is 2+2?<\|im_end\|><\|im_start\|>assistant\n4"` directly                                          | After Layer 4 wraps it: `<\|im_start\|>user\n…<\|im_end\|><\|im_start\|>assistant\n4<\|im_end\|>` — Qwen3-chat-format-shaped, with one extra outer wrap pair                                                |
| Split one chat turn per TEXT element (Layer 2)         | The SeqBuilder emits `[TEXT("user\nWhat is 2+2?"), TEXT("assistant\n4", loss=True)]`                                                              | After Layer 4 wraps each: `<\|im_start\|>user\n…<\|im_end\|>` followed by `<\|im_start\|>assistant\n4<\|im_end\|>` — clean per-turn wrap pairs                                                              |

Neither approach is wired today. T2A's `text_str` has **no role keywords
at all**. The only place in the codebase where a chat-format role
keyword (the literal string `"assistant"`) is actively materialized is
[`TEXT_INCOMPLETE` at inference time](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_text_ops.py#L184-L189),
which prepends `[<|im_start|>, "assistant", "\n"]` as a generation cue.
Training never sees role keywords.

#### Templating decisions and where to make them

A small Q&A worth pinning down because the layered design makes the
location of each decision non-obvious:

**Q: Where does the literal text "Generate the following transcript:" come from?**
A: Layer 1 — `OmniAudioSeqBuilder.Config.t2a_task_prompt`. To change it
for T2A, you edit the SeqBuilder Config or override it at construction.

**Q: How do I make T2A multi-turn?**
A: Layer 2 — modify `handle_t2a` in `OmniAudioSeqBuilder` to emit
multiple `SequenceElement`s. Examples elsewhere in the codebase:
storyboard / multi-turn image editing.

**Q: How do I add system-prompt-style preamble to T2A?**
A: Layer 1 — extend `t2a_task_prompt` to include the preamble, OR Layer
2 — emit a separate TEXT element before the prompt with `loss=False`.
Both work; the multi-element approach gives you a separate
`<|im_start|>` / `<|im_end|>` wrap around the system prompt, which can
be useful at inference for prompt caching.

**Q: How do I switch between pretrain-style and SFT-style training for
the same task?**
A: There's no template-mode flag, so the change is in the SeqBuilder.
Pretrain-style: bare prompt without role keywords (current T2A behavior).
SFT-style: insert role keywords via Layer 1 or use multi-element layout
via Layer 2. The tokenizer's wrap is identical in both cases.

**Q: What's the "shortest" way to introduce chat-format support to
omni?**
A: Adopt the Layer 2 approach (one TEXT element per chat turn), put
role keywords in each element's `text_str`. The existing per-element
wrap then naturally produces correct Qwen3-chat-format ids. No
tokenizer or trainer changes needed.

#### Why this matters

- **"Prompt template" is not a single object you can grep for.** It
  composes across four stages and four owners. A reader who searches
  for "chat_template" or "system_prompt" in the codebase will find
  nothing meaningful — the template is decomposed.
- **Most templating knobs live on the SeqBuilder, not the tokenizer.**
  Anyone trying to understand or modify a task's training prompt
  should read the SeqBuilder's per-task handler first
  (`handle_t2a` for T2A, `handle_t2i` for T2I, etc.), not the tokenizer
  config.
- **Adding a new task = mostly Layer 1 + Layer 2 work.** New per-task
  SeqBuilder Config (Layer 1) and new SeqBuilder forward
  implementation that emits the right element layout (Layer 2). Layers
  3 and 4 rarely change.
- **Reading a tokenized id sequence requires knowing all four layers.**
  An `<|im_start|>` at position 0 of a TEXT element came from Layer 4.
  An `<|vision_start|>` came from Layer 3. The tokens in between are
  Layer 1 instruction + per-sample data. The element ordering is
  Layer 2.

### CE loss interaction at the wrap positions

This is the subtlest part. The wrappers participate in autoregressive
CE loss, but unevenly. For a supervised TEXT element with input ids
`[<|im_start|>, ans₁, ans₂, …, ansₙ, <|im_end|>]` (length N+2) where
the answer is the **last element** of the sample:

| Position | Input               | Predicted next token (autoregressive label)             | Contributes to CE? | Note                                                                  |
| :------: | ------------------- | -------------------------------------------------------- | :---------------: | --------------------------------------------------------------------- |
| 0        | `<\|im_start\|>`     | `ans₁`                                                   | **yes**            | Model learns to *start* generating after seeing the wrapper opening    |
| 1        | `ans₁`              | `ans₂`                                                   | yes                | Standard within-element AR shift                                       |
| …        | …                   | …                                                        | yes                | …                                                                     |
| N        | `ansₙ`              | `<\|im_end\|>`                                            | **yes**            | Model learns to *emit* `<\|im_end\|>` to terminate the answer          |
| N+1      | `<\|im_end\|>`       | (would be next-element-first-token, but no next exists)  | **no**             | "Don't supervise the very last token" branch fires — see below        |

Two takeaways from this:

- **`<|im_start|>` and `<|im_end|>` participate in CE loss in different
  roles.** `<|im_start|>` is the *input* at position 0 whose prediction
  (the first answer token) is supervised. `<|im_end|>` is both: a
  *prediction target* at position N (the model emits it) and an *input*
  at position N+1 (whose prediction is NOT supervised when the element
  is last).
- **The same token id (151645) at different positions has different
  semantic roles.** A reader who only sees the input ids can't tell
  whether a `<|im_end|>` they're looking at is "the model's emitted
  answer terminator" or "a wrapper closing the element with no
  prediction expected." The role is determined by position relative to
  the element boundary, not by the token id.

If the TEXT element is **not last** in the sample, the position-N+1
prediction IS supervised, with the next element's first token as
target. This is the cross-element AR linkage handled by
`next_first_token_id` in [sequence_packing.py:236-243](../../lumaverse/lib/ursa/ursa/models/omni/inference/sequence_packing.py#L236-L243).
The relevant code:

```python
if next_first_token_id is not None:
    # Supervise last token with first token of the next sequence element
    non_padding_supervised_text_ids = torch.cat([
        non_padding_supervised_text_ids,
        torch.tensor([next_first_token_id]),
    ])
else:
    # Don't supervise the very last token when there's no next sequence element
    txt_loss_positions = torch.where(s.txt_loss_mask)[0]
    if len(txt_loss_positions) > 0:
        s.txt_loss_mask[txt_loss_positions[-1]] = False
```

So the cross-element bridge is what teaches the model multi-element
generation patterns: "after `<|im_end|>` of an answer, emit
`<|vision_start|>` to begin the next image span," etc.

### A subtle Q&A on what these tokens mean

A few questions that come up repeatedly:

**Q: Are `<|im_start|>` / `<|im_end|>` the same as Qwen3's chat-format
role tokens?**
A: Same string ids, different semantic role. In Qwen3 chat format, the
pattern is `<|im_start|>user\n…<|im_end|><|im_start|>assistant\n…<|im_end|>`
— each pair brackets a chat message turn. In omni, the pair brackets
**any** TEXT element's content, with no role keyword inside. The TEXT
element might be a prompt, a response, an analysis turn, or a
caption — the wrapper says "this is one text element", nothing about
role.

**Q: Does that mean omni does pretrain-style training, not chat-format
SFT?**
A: It depends on what the SeqBuilder puts inside `text_str`. If the
SeqBuilder produces a bare prompt like `"Generate the following
transcript:\n…"` (T2A's pattern), the result is effectively a
pretrain-style sentence wrapper. If a future SeqBuilder were to render
the chat template into `text_str` (e.g.,
`"user\nQuery<|im_end|><|im_start|>assistant\nResponse"`), the result
would look like a Qwen3 chat sequence with one extra outer wrap. There
is no tokenizer-level mode switch between these.

**Q: Is there a separate `<EOS>` token in TEXT elements?**
A: No. `<|im_end|>` itself serves as the terminator. The token
`<|endoftext|>` (id 151643, named `QWEN3_PAD_TOKEN` in the codebase)
exists but is never inserted into TEXT element ids — see the next
subsection.

**Q: If `<|im_start|>` is at position 0, is it a "prediction target" or
an "input"?**
A: It is an *input* at position 0 — the model sees it as part of the
context. Its prediction (the autoregressive next-token at position 0,
which targets the first content token at position 1) IS supervised.

**Q: What about the model's prediction *of* `<|im_start|>`?**
A: That happens at the position whose label is `<|im_start|>`. For an
element whose preceding element ended at position k, position k's input
is the prior element's last token and the AR label at position k is
`<|im_start|>` of the next element (via `next_first_token_id`). If the
prior element was supervised (had `txt_loss_mask=1`), the model is
trained to *emit* `<|im_start|>` to begin the next element. If the
prior element wasn't supervised, no learning signal lands on this
prediction.

### `<|endoftext|>` (id 151643) — multiple roles, but not text-wrapping

A separate Qwen3 special token, [QWEN3_PAD_TOKEN](../../lumaverse/lib/koba_shared/koba_shared/common/omni_constants.py#L31)
= `"<|endoftext|>"` (id 151643), serves several roles in the codebase
that **do NOT** include "wrapping TEXT element content":

| Role of `<\|endoftext\|>` (id 151643) | Where used                                                                                                      |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Padding token                        | When `pad_text_tokens=True`, text is right-padded to `max_seq_len` using this id; `padding_mask=0` at those positions, so they don't contribute to CE loss |
| Register slot in noisy VAE elements  | Default `audio_register_token` / `vision_register_token` is this id; serves as the "summary slot" content (see deep-dive §8.4) |
| Document-level EOS (Qwen3 convention) | Theoretically the document boundary in raw Qwen3 use; rarely surfaces in omni training                           |

Inside a TEXT element, the structure is `[<|im_start|>, content...,
<|im_end|>]` with no `<|endoftext|>` in between. After `<|im_end|>`
there may be `<|endoftext|>` *padding* if `pad_text_tokens=True`, but
the padding sits outside the supervised region and never participates
in either AR labeling or loss.

### The naming collision (the "unfortunate naming")

Three conventions intersect on the same token strings, with three
different interpretations:

| Convention                              | What `<\|im_start\|>` / `<\|im_end\|>` means                                       |
| --------------------------------------- | ---------------------------------------------------------------------------------- |
| Qwen3 chat-format (upstream)            | Boundary of a chat **message turn** (user / assistant)                             |
| Omni's actual usage                     | Boundary of an opaque **per-text-element span** (any TEXT element, not necessarily chat) |
| Codebase variable name (`QWEN3_BEGIN_OF_SENTENCE_TOKEN`) | "Begin / end of a **sentence**" (third interpretation, doesn't quite fit either)   |

The token ids are reused literally from Qwen3, so the model's pretrained
embeddings for these ids carry chat-format priors. The omni training
data, however, mostly does NOT include chat-format role keywords — for
T2A, the TEXT element contains a bare prompt, with `<|im_start|>`
acting as a "this content begins" marker rather than "this is the start
of a chat turn." The embedding that the model has at id 151644 was
trained as "user / assistant turn boundary" in Qwen3 pretraining, but
in omni it gets used as a generic span boundary.

This is partly why the design works: the underlying chat-format
embedding semantics are general enough ("a structured boundary follows")
that they specialize fine to "an element boundary follows" through
omni training. But if you read the codebase looking for chat-format
patterns, you'll be confused. The `BEGIN_OF_SENTENCE_TOKEN` variable
name nudges in the right direction (closer to the actual usage) but
doesn't quite capture "per-`SequenceElement` wrap."

A cleaner naming, if anyone touches this in the future, would be
something like `OMNI_TEXT_ELEMENT_OPEN` / `OMNI_TEXT_ELEMENT_CLOSE`, or
just leave the constants named after their Qwen3 string ids and rename
the variable references to make it clear they're per-element wrappers.

### Counter-intuitive things to watch out for

**1. Same token id, different semantic role at different positions.** A
`<|im_end|>` at position N (predicted as the autoregressive target by
the previous content token) is "the model emitting the close marker."
A `<|im_end|>` at position N+1 (the actual input id sitting in the
sequence) is "the wrap that bookends the element." Both have id 151645,
but their loss contributions differ.

**2. The AR shift jumps across element boundaries via
`next_first_token_id`.** Within a single element, AR labels are just
"next position's input id." But for the last position of an element
(input is `<|im_end|>` for TEXT, `<|vision_end|>` for VAE/ViT), the
label has to come from the NEXT element's first id. The packer
[sequence_packing.py:236-268](../../lumaverse/lib/ursa/ursa/models/omni/inference/sequence_packing.py#L236-L268)
handles this — see the deep-dive on `next_first_token_id` if you need
the full mechanism.

**3. `TEXT_INCOMPLETE` is the only place in the data pipeline that uses
chat-format role keywords ("assistant\n").** It only fires at inference,
not training. If you're auditing the codebase to see where Qwen3
chat-format actually shows up, this is the lone site.

**4. There's no "disable wrap" config flag.** Every TEXT element gets
the wrap, period. If a downstream consumer needs unwrapped text (e.g.,
for analysis or debugging), it has to strip the first and last ids
manually — or look at `padding_mask` and notice that the wrap sits at
positions 0 and num_tokens-1.

**5. The wrap is mode-agnostic.** Pretrain-style and SFT-style training
use the **same** wrap structure; the only difference is what the
SeqBuilder put in `text_str`. There's no template-mode flag to switch
between them. A future audit that wants to document "is this pipeline
pretrain or SFT?" has to look at the SeqBuilder, not the tokenizer.

### Implications for T2A specifically

For the T2A walkthrough's TEXT element (the prompt), the lifecycle
counts work out like:

| Quantity                                    | Value                                                          |
| ------------------------------------------- | -------------------------------------------------------------- |
| `text_str` set by `OmniAudioSeqBuilder`      | `"Generate the following transcript:\n<transcript>"`          |
| Tokenizer-encoded content length             | ≈ 28 ids (depends on transcript length)                        |
| Wrap added by `OmniQwen3Tokenizer`           | `[<\|im_start\|>]` + 28 ids + `[<\|im_end\|>]` = 30 total      |
| `num_tokens`                                  | 30                                                             |
| `txt_loss_mask`                              | All zeros — the prompt is `loss=False` (no CE supervision)    |
| Cross-element AR linkage                     | Position 29 (input is `<\|im_end\|>`) would link to position 30, the audio element's first id (`<\|vision_start\|>`) — **but `txt_loss_mask=0` makes this moot for T2A** |

So in T2A, the `<|im_start|>` / `<|im_end|>` wrap is structurally
present (and adds 2 to `num_tokens`) but contributes nothing to CE
loss because the prompt isn't supervised. The wrap is "free
bookkeeping" — it doesn't cost training signal but does ensure the
prompt has clean boundary tokens visible in attention to the audio
element that follows.

For a hypothetical future A2T (audio-to-text) task — where the answer
TEXT element WOULD be supervised — the wrap interaction described
above (`<|im_start|>` predicts first answer token, last answer token
predicts `<|im_end|>`, etc.) becomes load-bearing. The wrap is what
teaches the model when to start and stop generating.

## Discussion: `x_vae` semantics across stages

This entry sits outside the §1–§3 discussion notes because it's a
single-field clarification rather than a multi-part design question.
It belongs here for the same reason: the field's name suggests
something it doesn't actually contain.

### The name misleads

The field name `x_vae` reads as "VAE-encoded latent embeddings," but
the field actually carries **input to the VAE encoder**, not output
from it, in the data-pipeline stages where it lives. The encoded
latents exist only ephemerally inside the trainer step, under a
different variable name. This naming mismatch is one of the
counter-intuitive details a reader hits when tracing the lifecycle.

### Where `x_vae` lives across stages

| Stage / location                                                | What `x_vae` contains                                                | Tensor shape (T2A example)                       |
| --------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------ |
| Stage 5 — `OmniElementVAEAudio` ([line 137](../../lumaverse/lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L137)) | `audio_tensor.clone()` — **raw 1-D audio waveform**                  | `Tensor[80000]` (5 s × 16 kHz)                    |
| Stage 9 — per-sample `TokenizedSequenceElement.x_vae`           | Same raw waveform                                                    | `Tensor[80000]`                                   |
| Stage 10 — per-batch `batch["x_vae"]`                           | List of raw waveforms, one per VAE element across the batch          | `list[Tensor]`, length = N_VAE_elements_in_batch  |
| Stage 11 — trainer step, before encoding ([trainer.py:209](../../lumaverse/projects/kuma/kuma/projects/omni/audio/trainer.py#L209)) | Same as Stage 10 (filtered by `x_vae_by_modality`)                   | `list[Tensor]`                                    |
| Stage 11 — trainer step, after `_encode_audio` ([trainer.py:214](../../lumaverse/projects/kuma/kuma/projects/omni/audio/trainer.py#L214)) | **Encoded audio latents** — bound to a new variable `z0`               | `list[Tensor]` of compressed latent shapes        |
| Stage 11 — after `del batch["x_vae"]` ([trainer.py:217](../../lumaverse/projects/kuma/kuma/projects/omni/audio/trainer.py#L217)) | (gone — original raw waveforms freed to save memory)                 | n/a                                               |
| Inside `model.forward` (`vis_tokens` from `xs[0]`)              | Encoded latents (passed under name `vis_tokens` / `xs[0]`, not `x_vae`) | `Tensor[B, S_V, C]`                              |

### What the boundary looks like

The transformation `raw_waveform → encoded_latent` happens at the
trainer step, NOT inside the data pipeline:

```python
# trainer.py:207-217 — inside train_step
x_audio = [
    x for x, m in zip(batch["x_vae"], batch["x_vae_by_modality"])
    if m in AUDIO_MODALITIES
]
with trace_io("audio_encode"):
    z0 = self._encode_audio(x_audio)        # ← VAE encoding happens HERE

del batch["x_vae"]                            # ← raw waveforms freed
```

After this, the encoded form is `z0` (a different variable). `model.forward`
receives the latents as `vis_tokens` / `xs[0]`, never under the name
`x_vae`. So:

| Name           | Meaning                                              | Where used                                  |
| -------------- | ---------------------------------------------------- | ------------------------------------------- |
| `x_vae`        | Input to the VAE encoder (raw waveform / pixel data) | Data pipeline (Stage 5–10) + trainer line 207 |
| `z0`           | VAE-encoded latents                                   | Trainer-internal, line 214 onward            |
| `vis_tokens` / `xs[0]` | Encoded latents passed to the denoiser              | `model.forward` arg                          |

Three names, two semantic forms, with the encoding boundary at the
trainer step.

### Implications

- **Don't try to find a `VAE.encode` call inside the data pipeline.**
  It isn't there. The data pipeline only carries inputs to the encoder;
  the encoding boundary is in `train_step`.
- **Memory profile: the data pipeline's biggest VAE-element tensors are
  raw waveforms / pixels.** For T2A's 5-second audio at 16 kHz, that's
  80,000 floats × 4 bytes = ~320 KB per sample. Encoded latents are
  ~100-200× smaller, but they exist only ephemerally in the trainer.
- **`x_vae_by_modality` matters because of this mid-flight handoff.**
  The trainer needs to know which `x_vae` entries are audio (→ DAC
  encoder) vs image (→ image VAE) vs video (→ video VAE) so it dispatches
  each to the right encoder. The per-VAE-element `x_vae_by_modality` tag
  is what makes that filter possible without recomputing from tensor
  shapes or guessing.
- **Same flavor of misnomer as `clean_vae_img_mask`** (deep-dive
  §11.6.3) — a name baked in early when the codebase was simpler, now
  stretched across multi-stage processing where the field's content
  evolves at the trainer boundary. A cleaner name would be `vae_input`
  or `pre_vae_data`. Worth flagging for the same parallel-cleanup track.