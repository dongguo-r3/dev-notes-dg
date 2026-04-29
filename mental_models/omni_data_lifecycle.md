# Omni T2A SequenceElement lifecycle — concrete walkthrough

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
(default 0.1), the TEXT element's `text_str` is replaced with the empty
string for classifier-free guidance training. Otherwise no-op. The
`SequenceType` and structure stay the same.

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

`create_sparse_mask(sample_lens, split_lens, attn_modes, ...)` builds
the FlexAttention mask:

- `document_id`: stamps which sample each token belongs to (length
  566). Tokens of sample 0 get id 1, sample 1 get id 2, sample 2 get id
  3.
- `full_and_noise_seq_id`: stamps the element index for noise/full
  elements (length 566). For the 6-element pack with `attn_modes`
  `["causal", "noise", "causal", "noise", "causal", "noise"]`, elements
  1, 3, 5 get their indices; elements 0, 2, 4 get -1.
- `noise_seq_id`: identical to `full_and_noise_seq_id` for this T2A
  pack since no `"full"` elements exist.

The mask is then
`(causal_mask OR full_and_noise_mask) AND remove_noise_mask AND sample_mask`,
with the visibility behavior described in deep-dive §8.1 / §9.7.

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
