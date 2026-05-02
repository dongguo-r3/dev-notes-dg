# Omni-audio extension v2 — A2T + A2A plan

> **Status (2026-04-29).** Consolidated plan after multi-round design
> discussion. Pre-implementation. Single-source-of-truth for adding
> audio-to-text (A2T) and audio-to-audio (A2A) capabilities to the
> existing omni-audio (T2A) stack.
>
> **In scope:** Phase 0 groundwork + Phase 1 (A2T) + Phase 2 (A2A).
> **Out of scope:** audio SSL (separate doc:
> [`audio_ssl_planning.md`](audio_ssl_planning.md)), per-modality
> VAE-mask rename (deep-dive §10/§12), `clean_vae_img_mask` rename
> (deep-dive §11.6.3), Phase 3 multi-task training (placeholder).
>
> **Companion docs:**
> [`omni_t2a_dataloader_deep_dive.md`](omni_t2a_dataloader_deep_dive.md)
> for the data-pipeline mental model;
> [`../mental_models/omni_data_lifecycle.md`](../mental_models/omni_data_lifecycle.md)
> for the per-sample lifecycle;
> [`qwen3_audio_graft_poc.md`](qwen3_audio_graft_poc.md) for the audio
> understanding-encoder integration.
>
> All cross-references to lumaverse code are pinned to commit
> `7fa0eb17a3c03f5386c0975f55ef7b6454405fd4` (branch `dongguo/omni-t2a-v2`).

---

## 1. Goal and scope

Extend the omni-audio model from T2A-only to **T2A + A2T + A2A**:

- **A2T** — speech transcription, audio captioning. Audio in, text out.
- **A2A** — TTS with reference audio (voice cloning). Reference audio +
  target text in, target audio out.

**Hardcoded simplifications** (deliberate, to keep the v2 scope tight):

- A2T uses the **AuT semantic encoder** path only — Qwen3-ASR audio_tower
  from the `Qwen3-VL-2B-Audio-POC` checkpoint (qwen3 POC). It does NOT
  use the VAE encoder.
- A2A uses the **VAE encoder** path only (DAC/MMAudio). It does NOT use
  the AuT encoder.
- Audio inputs to A2T and A2A use disjoint encoder routes; same packed
  batch can mix samples from both kinds (one row → one encoder).

**Explicitly not in scope for this plan** (each gets its own work):

- Audio SSL (inpainting / outpainting / random-frame masking) —
  `audio_ssl_planning.md`.
- Per-modality VAE-mask leaf-mask rename — deep-dive §10 / §12.
- `clean_vae_img_mask` per-element rename — deep-dive §11.6.3.
- Multi-task mixed-batch training — Phase 3 placeholder.

---

## 2. Mental-model anchors

A task in this stack is fundamentally **"what list of `SequenceElement`s
does the SeqBuilder emit, and what's the loss flag on each."** Once that
is decided, downstream is mechanical (lifecycle doc summary table;
deep-dive §11.0). The four levers per task:

| Lever | What it controls | Where it's set |
|---|---|---|
| Element list (Layer 2 templating) | structural shape of the sample | `OmniAudioSeqBuilder.handle_<task>` |
| `modality` string | per-task branching at sampler / loss | per-element field on `SequenceElement` |
| Per-element `loss` flag | which positions are supervised | per-element field |
| CFG dropout class | which conditioning is droppable | pipeline file (`default_<task>.py`) |

The two new tasks divide cleanly along the audio-encoder axis:

| Task | Audio route | Loss components | CFG class |
|---|---|---|---|
| T2A (existing) | VAE (noisy) | diffusion | `OmniCFGDropout["t2a"]` |
| **A2T** | **AuT** (semantic, understanding stream) | **CE** | disabled (`cfg_dropout_modalities=[]`) |
| **A2A** | VAE (clean ref + noisy target) | diffusion | `OmniCFGDropoutMixed["a2a"]` |

A2T's audio never goes through the VAE; A2A's audio never goes through
the audio_tower.

---

## 3. Phase 0 — groundwork (three parallel workstreams)

Phase 0 has three independent workstreams. Each can land as its own PR;
ordering between them doesn't matter for correctness.

### 3.1 Workstream 0A — AuT plumbing

**Purpose.** Wire the audio_tower (Qwen3-ASR semantic encoder, from
`Qwen3-VL-2B-Audio-POC`) into the data pipeline + model + trainer as
a **third** understanding-stream encoder, parallel to ViT.

This is the largest of the three workstreams. It reproduces the qwen3
POC §2.1 inference recipe, but lifted from a one-off script into the
omni training stack.

**Data pipeline:**

- Add field `aut_token_mask: torch.Tensor | None = None` to
  [`TokenizedSequenceElement`](lib/koba_shared/koba_shared/processor/tokenized_types.py).
- Add field `x_aut: torch.Tensor | None = None` (raw waveform, mirroring
  `x_vit`).
- New per-element processor `OmniElementVITAudio` in
  [`lib/koba_shared/koba_shared/processor/omni_audio_ops.py`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py),
  mirroring `OmniElementVITImage` (in
  [`omni_vit_ops.py`](lib/koba_shared/koba_shared/processor/omni_vit_ops.py)):
    - Sets `aut_token_mask=1` at audio_pad positions,
      `text_token_mask=1` at audio_start / audio_end boundaries,
      `attention_mode="causal"` (matches Qwen3 ViT understanding stream).
    - Computes `num_audio_tokens` from the audio-tower's compression
      schedule (post-CNN length, per qwen3 POC §2.1 step 3:
      `lengths → (lengths-1)//2+1` three times for the three stride-2
      conv stages, then 13 output tokens per chunk of 100 mel frames).
    - Stores raw waveform on `tok_element.x_aut` for the trainer's
      audio-tower forward pass.
- Add `VIT_AUDIO` branch to `OmniPositionIDMRoPE` in
  [`position_ids_dev.py:152`](lib/koba_shared/koba_shared/processor/position_ids_dev.py#L152) —
  currently noted as "VIT_AUDIO not yet supported by MRoPE." Audio_pad
  tokens use Axis-0 sequential, H/W axes share the Axis-0 value (per
  qwen3 POC §4.3 — same as the existing VAE_AUDIO piggyback). Copy
  that branch.
- Plumb `aut_token_mask`, `x_aut` through
  [`pack_sequence`](lib/ursa/ursa/models/omni/inference/sequence_packing.py)
  alongside `vit_token_mask` / `x_vit`.

**Model:**

- In [`OmniModel.forward` at model.py:600](lib/ursa/ursa/models/omni/model/model.py#L600),
  extend the und-stream union:
  ```python
  und_mask = text_token_mask | vit_token_mask | aut_token_mask
  ```
- Add `aut_in_und = aut_token_mask[und_mask]` slice and route those
  positions to `audio_tower.forward(x_aut)`, mirroring the existing
  `vit_in_und` / `vision_tower` block at
  [model.py:610](lib/ursa/ursa/models/omni/model/model.py#L610).
- The audio_tower's `(num_aut_tokens, hidden_size=2048)` output
  row-substitutes `inputs_embeds` at the `aut_in_und` positions
  (POC §2.1 step 6).

**Trainer:**

- In [`projects/.../audio/trainer.py`](projects/kuma/kuma/projects/omni/audio/trainer.py),
  add a pre-denoiser audio-tower step alongside the existing
  `_encode_audio` (VAE):
  ```python
  x_aut_inputs = batch.get("x_aut", [])
  if x_aut_inputs:
      aut_features = self._encode_audio_understanding(x_aut_inputs)
      batch["aut_features"] = aut_features
  ```
- The audio_tower runs on raw waveform, applies its internal mel
  feature extractor + 3-stride-2-conv stem + 24 transformer layers +
  proj1/proj2 (POC §1.1). Output is `(num_tokens, 2048)` consumed
  inside `model.forward`'s und-stream substitution.

### 3.2 Workstream 0B — audio token-id flip

**Purpose.** Replace the vision-token aliasing in
[`OmniElementVAEAudio.Config`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py)
(deep-dive §8.5) with the dedicated audio specials from the merged
`Qwen3-VL-2B-Audio-POC` tokenizer (POC §1.2).

| Role | Today (alias) | After flip |
|---|---|---|
| Audio span open | `<\|vision_start\|>` (151652) | `<\|audio_start\|>` (151669) |
| Audio span close | `<\|vision_end\|>` (151653) | `<\|audio_end\|>` (151670) |
| Audio pad placeholder | `<\|image_pad\|>` (151655) | `<\|audio_pad\|>` (151676) |

**Touch points:**

- `OmniElementVAEAudio.Config` defaults at
  [`omni_audio_ops.py:64-68`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L64-L68).
- Inference processors that build audio spans by hand:
  [`projects/.../omni/audio/inference/processor/t2a_processor.py`](projects/kuma/kuma/projects/omni/audio/inference/processor/t2a_processor.py).

**Verification:** the active backbone must be `Qwen3-VL-2B-Audio-POC`
(POC §1.5) before this flip lands — these ids only exist in that
tokenizer; using them with Qwen3-0.6B's tokenizer would silently
mistokenize. Per qwen3 POC §5.5, this sweep should also audit any
other hardcoded id constants in the omni-audio training/inference
scripts.

### 3.3 Workstream 0C — loss class refactor

**Purpose.** One coordinated PR with three changes to
[`projects/.../audio/losses/bagel_t2a.py`](projects/kuma/kuma/projects/omni/audio/losses/bagel_t2a.py):

#### (i) Rename

- `bagel_t2a.py` → `bagel_audio.py`
- `bagel_t2a_test.py` → `bagel_audio_test.py`
- `class BagelT2ALoss` → `class BagelAudioLoss`
- grep + replace importers in experiment configs and trainer wiring

The `_t2a` suffix becomes actively misleading once the class handles
A2T (CE only) and A2A (diffusion only with mixed clean+noisy audio).
Bundle the rename with the bug-fix PR so git-blame stays clean.

#### (ii) Add Guard 1 — per-clip noise-injection skip

Mirror image-side
[`bagel.py:397-404`](projects/kuma/kuma/projects/omni/bagel/losses/bagel.py#L397-L404):

```python
# inside the per-clip loop in __call__
log_alpha_i, log_sigma_i = self.corrupt_fn(logsnr_i)
if kwargs["clean_vae_img_mask"][i]:                # NEW
    sigma = torch.zeros_like(log_sigma_i)
    zt_i = z0_i
    log_alpha_i = torch.zeros_like(log_alpha_i)
    log_sigma_i = torch.full_like(log_sigma_i, float("-inf"))
else:
    # existing noise-adding code path
    sigma = torch.exp(log_sigma_i)
    shift = self.sigma_shift
    sigma = (shift * sigma) / (1 + (shift - 1) * sigma)
    sigma = torch.clamp(sigma, 0.0, 1.0)
    alpha = 1.0 - sigma
    zt_i = alpha * z0_i + sigma * eps0_i
    log_alpha_i = torch.log(alpha)
    log_sigma_i = torch.log(sigma)
```

The `clean_vae_img_mask` field is per-VAE-element (1 for CLEAN, 0 for
NOISY), already populated by `OmniElementVAEAudio` at
[`omni_audio_ops.py:181`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L181) —
despite the misleading `_img_` infix it carries audio side too
(deep-dive §11.6.3).

#### (iii) Add Guard 2 — per-token diffusion-loss masking

Mirror image-side
[`bagel.py:705-747`](projects/kuma/kuma/projects/omni/bagel/losses/bagel.py#L705-L747):

```python
if audio_out is not None and has_audio:
    noisy_in_audio_gen = kwargs["noisy_vae_token_mask"][kwargs["vae_token_mask"]]   # NEW
    if noisy_in_audio_gen.sum() > 0:
        target, prediction, ... = self.target_fn(
            model=audio_out.to(self.dtype)[:, noisy_in_audio_gen],
            x_noisy=zt_patched_cat[:, noisy_in_audio_gen],
            log_alpha=log_alphas_cat[:, noisy_in_audio_gen],
            log_sigma=log_sigmas_cat[:, noisy_in_audio_gen],
            x_original=z0_patched_cat[:, noisy_in_audio_gen],
            eps_original=eps0_patched_cat[:, noisy_in_audio_gen],
        )
        diffusion_loss = nn.functional.mse_loss(prediction, target, reduction="mean")
```

Verify `target_fn` accepts the masked tensors as-is, otherwise wrap
with `.contiguous()`.

#### (iv) Add CE block

Lift from
[`bagel.py:819-866`](projects/kuma/kuma/projects/omni/bagel/losses/bagel.py#L819-L866),
gated on a new config knob:

```python
@dataclass(kw_only=True, slots=True)
class Config:
    ...
    ce_loss_weight: float = 0.0       # NEW; T2A leaves at 0, A2T sets to 1
    use_fused_ce: bool = False        # NEW; mirror bagel.py
```

The CE block reads `txt_out` (already returned by the denoiser at
[`bagel_t2a.py:231`](projects/kuma/kuma/projects/omni/audio/losses/bagel_t2a.py#L231)
and currently discarded), uses `und_mask = text_token_mask |
vit_token_mask | aut_token_mask`, slices `txt_loss_mask` and
`label_ids` (already produced by `pack_sequence`), computes CE.

#### Behavior on existing T2A runs

Each change is a no-op on T2A:

- Guard 1: today every audio clip is noisy, so the new `if` branch is
  never taken.
- Guard 2: today every position has `noisy_vae_token_mask=1` within
  `vae_token_mask`, so the slice is a no-op.
- CE block: gated on `ce_loss_weight=0.0` default.

The PR is purely additive on T2A's behavior. The fixes become
load-bearing for A2A and any other future task with clean audio
elements.

---

## 4. Phase 1 — A2T (audio-to-text)

**Depends on:** Workstreams 0A (AuT) + 0C (CE block).
**Independent of:** 0B (T2A-only) and Phase 2.

### 4.1 Element layout

```python
[ SequenceElement(type=VIT_AUDIO, media=Media("audio", waveform), loss=False, modality="a2t"),
  SequenceElement(type=TEXT,      text_str=instruction,           loss=False, modality="a2t"),
  SequenceElement(type=TEXT,      text_str=target_text,           loss=True,  modality="a2t") ]
```

Key design points:

- **`VIT_AUDIO`** routes the audio through the audio_tower (semantic
  encoder), not through the VAE. The audio enters the **understanding
  stream**, exactly as in the qwen3 POC §2.2 zero-shot ASR test.
- Instruction and target are **separate TEXT elements** so the
  supervision boundary lines up with the per-element wrap
  (`<|im_start|>...<|im_end|>` from Layer 4 — lifecycle doc) — model
  learns to *emit* `<|im_end|>` to terminate the answer.

### 4.2 Source-table schema

```
audio_bytes      : bytes      # speech / general audio
instruction_text : str        # pre-rendered, e.g. "Transcribe the audio:" or "Describe this audio:"
target_text      : str        # the answer (transcript / caption)
```

The instruction column makes "transcription" vs "audio caption" the
same task with different prompts — `handle_a2t` is identical for both.
Domain-specific instruction strings are decided at data-prep time and
stored verbatim.

### 4.3 CFG dropout

**Disabled.** Pipeline sets `cfg_dropout_modalities=[]`. CFG isn't
meaningful for understanding-style outputs — there's no inference-time
guidance scale to amplify. If we wanted partial-conditioning CFG (drop
the instruction text but keep the audio), `OmniCFGDropoutLast` is the
right shape, but defer until empirically motivated.

### 4.4 Loss configuration

`BagelAudioLoss` with `ce_loss_weight=1.0, diffusion_loss_weight=0.0`.

The denoiser still runs (the model graph requires it), but the
diffusion branch contributes 0 to the loss because
`noisy_vae_token_mask.sum()==0` for every A2T sample (no noisy VAE
elements).

### 4.5 Files

**New files:**

```
lib/koba/koba/pipelines/default_a2t.py
projects/kuma/kuma/projects/omni/audio/configs/data/omni_a2t_packing_koba_v2.py
projects/kuma/kuma/projects/omni/audio/configs/a2t.py
projects/kuma/kuma/projects/omni/audio/inference/processor/a2t_processor.py
```

**Files modified:**

```
lib/koba/koba/processor/omni_audio_packed_ops.py
    OmniAudioSeqBuilder
        + handle_a2t(sample)
        + Config: a2t_default_instruction (Layer-1 templating constant)
```

### 4.6 Inference processor

`a2t_processor.py` builds:

```
[ VIT_AUDIO(audio),
  TEXT(instruction),
  TEXT_INCOMPLETE("<|im_start|>assistant\n") ]
```

Runs the LM in NTP mode (per the lifecycle doc's `TEXT_INCOMPLETE`
handling) — model generates target text autoregressively until
`<|im_end|>`.

### 4.7 Validation

- **Smoke test**: run A2T inference on the qwen3 POC's 4-clip test set
  (POC §2.2) at training-step-zero (just after wiring AuT, before any
  A2T training). Should reproduce ~0% WER because the AuT path is
  identical to the POC. If it doesn't, there's a wiring bug to find
  before A2T training starts.
- **Held-out eval**: WER on a transcription test set. The qwen3 POC
  hits 0% WER zero-shot on TTS-generated speech; A2T training on
  diverse data should at least match this on similar audio.

---

## 5. Phase 2 — A2A (audio-to-audio, voice clone)

**Depends on:** Workstream 0C (loss bug fixes).
**Independent of:** 0A, 0B, and Phase 1.

### 5.1 Element layout

```python
[ SequenceElement(type=TEXT,             text_str=instruction,                                 loss=False, modality="a2a"),
  SequenceElement(type=CLEAN_VAE_AUDIO,  media=Media("audio", reference_audio_waveform),       loss=False, modality="a2a"),
  SequenceElement(type=TEXT,             text_str=transcript,                                  loss=False, modality="a2a"),
  SequenceElement(type=NOISY_VAE_AUDIO,  media=Media("audio", target_audio_waveform),          loss=True,  modality="a2a") ]
```

Four elements, ordered: instruction → reference voice → transcript →
target audio. Each plays a structural role:

- `TEXT(instruction)` — the task framing ("Speak the following text in
  the voice of the reference audio:"). Constant per task variant,
  stamped per-row.
- `CLEAN_VAE_AUDIO(reference)` — voice characteristics conditioning.
  `attention_mode="full"`. Bidirectionally readable by the noisy
  target.
- `TEXT(transcript)` — the literal words to be spoken. Per-row.
- `NOISY_VAE_AUDIO(target)` — the diffusion target. `attention_mode="noise"`.
  Reads all prior context (instruction + reference + transcript);
  outbound-invisible to anything after (which is nothing — it's last).

### 5.2 Source-table schema

```
reference_audio_bytes : bytes
target_audio_bytes    : bytes
transcript_text       : str        # WHAT to say (literal words)
instruction_text      : str        # HOW to say it (constant per task variant)
```

`instruction_text` and `transcript_text` are distinct columns rather
than concatenated. Pre-rendered at data-prep time.

### 5.3 Why split the text into two elements

Compare merged vs split TEXT:

| Property | Merged TEXT (instruction+transcript) | Split TEXT (this design) |
|---|---|---|
| CFG dropout class fit | only drop-all viable → erases voice ref | `OmniCFGDropoutLast` blanks transcript, keeps voice ref + instruction → cleanly partial-conditional |
| Position-ID separation between ref and target | adjacent in position-id space | natural ~30-token gap from interleaved transcript |
| Transcript-to-target proximity | always close | always close (transcript immediately precedes target) |

Per
[`omni_interleaved_packed_ops_refactor.py:609+`](lib/koba/koba/processor/omni_interleaved_packed_ops_refactor.py#L609),
`OmniCFGDropoutLast` blanks all consecutive TEXT immediately preceding
the noisy target — applied here it blanks `TEXT(transcript)` while
leaving `TEXT(instruction)` and `CLEAN_VAE_AUDIO(reference)` intact.
This is the **partial-conditioning branch we want** at inference: dual
CFG over voice (CLEAN_VAE_AUDIO) and text (transcript), so a user can
control "how much like the reference voice" and "how literal the
transcript" independently.

The merged-TEXT layout couldn't access this branch — drop-all would
also erase the reference audio.

### 5.4 CFG dropout

`OmniCFGDropoutMixed["a2a"]`. Probabilistic blend of:

- **Drop-all branch** (`OmniCFGDropout`): trains
  `p(target | empty_conditioning)` — the fully-unconditional audio
  generation prior.
- **Drop-Last branch** (`OmniCFGDropoutLast`): trains
  `p(target | instruction, reference_voice, empty_transcript)` — the
  voice-conditioned-but-transcript-free branch.

At inference, both unconditionals are available for CFG combination.

### 5.5 Loss configuration

`BagelAudioLoss` with `diffusion_loss_weight=1.0, ce_loss_weight=0.0`.

**Critically depends on Workstream 0C's two guards**:

- Without **Guard 1**: every A2A sample's reference audio is silently
  noised (the encoder produces `z0` for it; `BagelT2ALoss` today
  unconditionally noises every clip in `z0_seq`). The denoiser sees
  garbage-noised conditioning and learns nothing useful from the
  reference voice.
- Without **Guard 2**: the diffusion-loss MSE denominator includes the
  clean reference audio's positions (which contribute zero to the sum
  if Guard 1 is also missing — but if Guard 1 is fixed and Guard 2 is
  not, `prediction == target` numerically at clean positions and they
  contribute zero terms to a mean over a larger denominator → the loss
  magnitude is silently rescaled by `noisy_count / (clean_count +
  noisy_count)`).

Both bugs land Phase 0C; Phase 2 inherits the fixes.

### 5.6 Position IDs

**No code change needed for Phase 2.** Default sequential wiring in
[`_wire_position_ids`](lib/ursa/ursa/models/omni/inference/sequence_packing.py#L132-L180)
gives position spans:

```
[0 .. L_instr-1]                                 TEXT(instruction)
[L_instr .. L_instr + L_ref - 1]                 CLEAN_VAE_AUDIO(reference)
[L_instr + L_ref .. L_instr + L_ref + L_trans - 1]  TEXT(transcript)
[L_instr + L_ref + L_trans ..]                   NOISY_VAE_AUDIO(target)
```

Reference and target are positionally separated by the transcript's
~20–50 tokens. Combined with distinct attention modes (`"full"` for
clean ref, `"noise"` for noisy target) and `<|audio_start|>` /
`<|audio_end|>` boundary tokens, the model has structural signal to
treat them as two separate audios.

**Escalation path** (deferred): if Phase 2 evaluation shows
reference-leakage artifacts (model copying acoustic content from
reference into target), add an explicit position-gap mechanism in
`_wire_position_ids` — a configurable position offset jump when
transitioning from CLEAN_VAE_AUDIO to NOISY_VAE_AUDIO under
`modality="a2a"`. Small additive change. Don't preemptively add.

### 5.7 Files

**New files:**

```
lib/koba/koba/pipelines/default_a2a.py
projects/kuma/kuma/projects/omni/audio/configs/data/omni_a2a_packing_koba_v2.py
projects/kuma/kuma/projects/omni/audio/configs/a2a.py
projects/kuma/kuma/projects/omni/audio/inference/processor/a2a_processor.py
```

**Files modified:**

```
lib/koba/koba/processor/omni_audio_packed_ops.py
    OmniAudioSeqBuilder
        + handle_a2a(sample)
        + Config: a2a_default_instruction (Layer-1 templating constant)
```

### 5.8 Inference processor

`a2a_processor.py` builds the four-element sequence with
NOISY_VAE_AUDIO replaced by an empty placeholder; diffusion sampler
fills it in over the denoising loop. Reference audio is pre-encoded
through the VAE encoder to provide conditioning latents.

### 5.9 Validation

Voice-clone quality on held-out (reference, target_text) pairs:

- **Speaker similarity** (SECS or equivalent): does generated audio
  sound like the reference voice?
- **Intelligibility** (WER via external ASR): is the transcript
  correctly spoken?
- **Subjective listening:** does the model copy reference content into
  the target instead of using it as voice-only conditioning? If yes,
  escalate to position-gap (§5.6 escalation path).

---

## 6. Phase 3 — multi-task training (deferred placeholder)

Once Phases 1 + 2 ship, the natural follow-up is one model trained on
a mix of T2A + A2T + A2A. The infra is already there
(`WeightedMultiSourceSampler` per the dataloader deep-dive §2-3); the
open work is loss-weight balancing across tasks (CE per-text-position
vs MSE per-VAE-frame, with different sample-level token counts).

Out of scope for this plan; will get its own design pass.

---

## 7. File-diff summary

```
NEW FILES (8):
  lib/koba/koba/pipelines/default_a2t.py
  lib/koba/koba/pipelines/default_a2a.py
  projects/.../omni/audio/configs/data/omni_a2t_packing_koba_v2.py
  projects/.../omni/audio/configs/data/omni_a2a_packing_koba_v2.py
  projects/.../omni/audio/configs/a2t.py
  projects/.../omni/audio/configs/a2a.py
  projects/.../omni/audio/inference/processor/a2t_processor.py
  projects/.../omni/audio/inference/processor/a2a_processor.py

RENAMED (2):
  projects/.../omni/audio/losses/bagel_t2a.py            → bagel_audio.py
  projects/.../omni/audio/losses/bagel_t2a_test.py       → bagel_audio_test.py

MODIFIED (Phase 0A — AuT plumbing):
  lib/koba_shared/koba_shared/processor/tokenized_types.py    + aut_token_mask, x_aut
  lib/koba_shared/koba_shared/processor/omni_audio_ops.py     + class OmniElementVITAudio
  lib/koba_shared/koba_shared/processor/position_ids_dev.py   + VIT_AUDIO branch
  lib/ursa/ursa/models/omni/inference/sequence_packing.py     plumb aut_token_mask + x_aut
  lib/ursa/ursa/models/omni/model/model.py                    extend und_mask, route to audio_tower
  projects/.../omni/audio/trainer.py                          + _encode_audio_understanding

MODIFIED (Phase 0B — token-id flip):
  lib/koba_shared/koba_shared/processor/omni_audio_ops.py     OmniElementVAEAudio.Config defaults
  projects/.../omni/audio/inference/processor/t2a_processor.py  mirror id substitutions

MODIFIED (Phase 0C — loss refactor):
  bagel_audio.py (post-rename)
    + ce_loss_weight, use_fused_ce config knobs
    + Guard 1: clean-clip noise skip
    + Guard 2: per-token diffusion-loss masking
    + CE block lifted from bagel.py:819-866
  importers grep + replace BagelT2ALoss → BagelAudioLoss

MODIFIED (Phase 1 + Phase 2):
  lib/koba/koba/processor/omni_audio_packed_ops.py            + handle_a2t, handle_a2a, Config knobs
```

---

## 8. Phasing and dependencies

```
Phase 0A (AuT plumbing)            ──┐
Phase 0B (token-id flip)           ──┼── parallel; any order
Phase 0C (loss rename + 2 guards   ──┘    ─── 0A+0C blocks Phase 1
          + CE block)                            ─── 0C alone blocks Phase 2
   │
   ├─►  Phase 1 (A2T)              ──┐
   │                                  ├── parallel; independent
   ├─►  Phase 2 (A2A)              ──┘
   │
   └─►  Phase 3 (multi-task)             after both Phase 1 + Phase 2 ship
```

**Single-engineer ordering:** 0C → 0A → 0B → Phase 1 → Phase 2 (Phase 1
first because A2T's CE-only loss exercises the new CE block in
isolation; A2A then exercises the dual-loss path with both branches
active).

**Two-engineer parallel:** one takes 0A + Phase 1, the other takes
0C + Phase 2; 0B is whichever finishes first.

---

## 9. Verification items not yet closed

Concrete checks/audits to do during implementation:

1. **Audit `BagelAudioLoss` after Phase 0C lands**: run a smoke
   A2A-style test (one fake batch with mixed CLEAN+NOISY audio
   elements) and confirm Guard 1 and Guard 2 fire as expected. Catch
   any subtle shape-contract issues with `target_fn` accepting masked
   tensors.
2. **Verify `_wire_position_ids` produces the expected
   adjacent-but-disjoint spans** for A2A's four-element layout.
   One-off test, ~5 lines.
3. **A2T eval calibration**: run A2T inference on the qwen3 POC's
   4-clip test set (POC §2.2) at training-step-zero (just after wiring
   AuT, before any A2T training). Should reproduce ~0% WER because
   the AuT path is identical to the POC. If it doesn't, there's a
   wiring bug to find before A2T training starts.
4. **A2A reference-leakage check** during Phase 2 eval: listen to
   outputs and check if the model is reproducing reference content
   rather than only voice characteristics. If yes, escalate to
   explicit position-gap (§5.6 escalation path).
5. **Backbone confirmation**: verify the active checkpoint path is
   `Qwen3-VL-2B-Audio-POC` before Phase 0B's token-id flip lands
   (using the dedicated audio specials with the wrong tokenizer would
   silently mistokenize).

---

## 10. Where this doc connects to the others

- [`omni_t2a_dataloader_deep_dive.md`](omni_t2a_dataloader_deep_dive.md):
  the comprehensive picture of how T2A data flows. Required reading
  for understanding Phase 0A and the loss bug fixes. Especially §8
  (attention modes), §11.0 (`SequenceElement` as anchor), §11.6.1
  (`aut_token_mask` proposal), §11.6.3 (`clean_vae_img_mask`
  misnomer).
- [`../mental_models/omni_data_lifecycle.md`](../mental_models/omni_data_lifecycle.md):
  the per-sample stage-by-stage walkthrough. Stage 2 is where new
  SeqBuilder handlers insert; Stage 5 is where per-element processors
  run; Stage 11 is where the trainer consumes the packed dict.
- [`qwen3_audio_graft_poc.md`](qwen3_audio_graft_poc.md): the
  audio-understanding-encoder integration. Phase 0A reproduces this
  POC's inference recipe in the training pipeline. Phase 0B's
  token-id flip is the §5.5 follow-up from that doc.
- [`audio_ssl_planning.md`](audio_ssl_planning.md): companion plan
  for the (out-of-scope-here) audio SSL task. Useful context for the
  big picture of the audio extension; not a dependency.

---

## 11. Change log

- 2026-04-29 — Initial draft. Consolidates multi-round design
  discussion: AuT-only A2T / VAE-only A2A simplifications, four-element
  A2A layout (instruction → reference → transcript → target),
  bagel.py-pattern bug fixes for clean+noisy audio coexistence,
  unified `BagelAudioLoss` with optional CE block, three-workstream
  Phase 0 structure. Pre-implementation. SSL deliberately excluded.
