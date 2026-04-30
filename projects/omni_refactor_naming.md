# Proposal: Naming and Special-Token Refactor for Omni Multi-Modal Stack

**Date:** 2026-04-30
**Author:** Dong Guo
**Status:** Draft — for review by image / audio / video / sampler / inference owners
**Companion docs:** [`omni_t2a_dataloader_deep_dive.md`](omni_t2a_dataloader_deep_dive.md) (the inventory and analysis this proposal acts on); [`qwen3_audio_graft_poc.md`](qwen3_audio_graft_poc.md) (the audio backbone swap that motivates Phase 0).

## TL;DR

The omni multi-modal data stack has accumulated five name-quality scars, each individually small but collectively a navigation tax that compounds as we add modalities (audio now, audio→audio / audio→text next):

1. **Audio reuses vision special-token strings** (`<|vision_start|>` / `<|vision_end|>` / `<|image_pad|>`) because Qwen3-base lacked dedicated audio specials.
2. **`vae_token_mask` is overloaded** to mean image-or-audio, with the modality split hidden in a per-element string (`x_vae_by_modality`).
3. **`clean_vae_img_mask` is misnamed** — audio sets it too, despite the `_img_` infix.
4. **`aut_token_mask` is missing** — there's no audio analog of `vit_token_mask` for the future audio-understanding stream, even though `SequenceType.VIT_AUDIO=8` is reserved.
5. **`VIT_AUDIO` is a category mistake** — `VIT` reads as "Vision Transformer," not "understanding-stream tower."

This proposal lays out a principled end-state where:

- Special tokens stop reusing vision strings on the audio side (the merged Qwen3-VL-2B-Audio-POC tokenizer ships dedicated audio specials at fresh IDs).
- Per-token masks shrink to the minimum needed for stream-membership routing (`text_token_mask`, `vae_token_mask`, `vit_token_mask`, `aut_token_mask`).
- All branch / modality information lives on the per-element `SequenceType` enum (the single source of truth), with a small helper module that derives any per-token slice from it on demand.
- Naming asymmetries (`VIT_AUDIO` → `AUT_AUDIO`; `clean_vae_img_mask` → `is_clean_vae_element`; `x_vae_by_modality` → `task`) are fixed.

Migration is phased so that no phase forces a breaking change on any consumer team in isolation. Phase 0 is additive only; phase 1 switches consumers file-by-file; phase 2 deletes the legacy fields once they have zero readers. Video and an optional `ElementKind` refactor are explicitly deferred — they live behind the same proposal but on later phases.

**Total blast radius:** ~30 files across `lib/koba_shared`, `lib/koba`, `lib/ursa`, `projects/kuma/kuma/projects/omni/{audio,bagel}`. CODEOWNERS for all of those teams need to sign off.

## 1. Why this proposal exists

The omni data pipeline grew up image-first. When audio was added it piggybacked on the image-shaped scaffolding because that was the cheap, low-risk path: audio reused image's special tokens, audio's frame slots set `vae_token_mask` (originally image-only), and audio's encoder dispatch was routed through a per-element string `x_vae_by_modality` rather than its own per-token mask. None of those decisions was wrong at the time — they reflected the shape of the data pipeline back when "audio" was a single new modality being grafted onto a stable image stack.

Two things have since changed:

1. **The audio backbone is being upgraded** ([`qwen3_audio_graft_poc.md` §5](qwen3_audio_graft_poc.md)). The merged Qwen3-VL-2B-Audio-POC tokenizer ships dedicated audio specials at fresh IDs. Audio no longer needs to borrow vision's special tokens — and after the swap, continuing to use `<|vision_start|>` for an audio span actively misleads anyone reading the tokenized text.
2. **Audio understanding is on the roadmap.** §5.1 of the audio-graft plan attaches an audio encoder ([`Qwen3-ASR-1.7B`'s audio_tower](qwen3_audio_graft_poc.md)) for ASR / A2T / audio-VLM. The understanding stream needs its own per-token routing mask the same way `vit_token_mask` routes image-understanding tokens; trying to serve that with `vit_token_mask` would route audio features through `self.vit_patchifier` (image-specific weights), which is wrong.

Both of these create natural pressure to clean up the names. This proposal collects the cleanup into one coherent design so individual teams aren't asked to make uncoordinated piecemeal changes in their own areas.

The motivation is **not** "the names are ugly and we have free time." It is "the next phase of audio work needs a place to put `aut_token_mask`, the audio backbone swap needs a place to put `<|audio_start|>`, and the existing scaffolding makes those additions awkward — let's fix the scaffolding while we're touching it."

## 2. Scope and non-goals

### In scope

- Audio special-token strings (audio side only — image / video unchanged).
- Per-token mask family on `TokenizedSequenceElement` (image, audio, text — the modalities in `koba_shared` per-element processors today).
- The `SequenceType` enum (rename `VIT_AUDIO` → `AUT_AUDIO`; consider whether `VIT_IMAGE` should mirror).
- The per-element classification field `x_vae_by_modality` (collapse / rename).
- The misnamed `clean_vae_img_mask`.
- A new helper module in `koba_shared` for deriving per-token slices from per-element classification.

### Explicit non-goals (this proposal does not change)

- **The image register token.** Switching `<|endoftext|>` → `<|vision_register|>` on the image side would force retraining of every loaded image checkpoint. Out of scope; covered as future work in [`omni_t2a_dataloader_deep_dive.md` §8.7](omni_t2a_dataloader_deep_dive.md) for the next time we touch the image tokenizer.
- **Video as a `koba_shared` per-element citizen.** Video lives entirely in `omni/bagel` packing code today; its top-level mask `video_vae_token_mask` is a `model.forward` kwarg, not a `TokenizedSequenceElement` field. Migrating it is **Phase 4 (optional)** and explicitly does not block phases 0–3.
- **Model architecture.** This proposal does not add or modify any neural-network module. Wiring up `aut_token_mask` to an actual audio_tower dispatch branch in [`model.py:580-614`](lib/ursa/ursa/models/omni/model/model.py#L580-L614) is `qwen3_audio_graft_poc.md` §5.1's job, not this proposal's; we ship the field declaration, the producer hook, and a CI assertion that catches any premature use.
- **Checkpoint compatibility.** Mask renames are data-pipeline / trainer-surface state. No model weights move. The audio special-token swap does change which embedding rows audio loads from at the LM input, but only on **new** training runs that adopt the merged Qwen3-VL-2B-Audio-POC tokenizer; pre-existing audio checkpoints (which were trained against `<|vision_start|>` / `<|image_pad|>` IDs) are unaffected by this proposal because they were never going to load against the new tokenizer.

## 3. The five problems, with evidence

The following five subsections give a code-grounded account of each problem. Tables show counts from a grep of the active codebase ([§11.5 of the deep-dive](omni_t2a_dataloader_deep_dive.md) verified consistency); pointers are to the canonical producer / consumer of each name.

### 3.1 Audio reuses vision special-token strings

| Role in audio span | Default string today | Where set | Note |
| --- | --- | --- | --- |
| Span open | `<\|vision_start\|>` (id 151652) | [`omni_audio_ops.py:64`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L64) | Same string the image stack uses to open a vision span |
| Span close | `<\|vision_end\|>` (id 151653) | [`omni_audio_ops.py:65`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L65) | Same as above |
| Frame placeholder | `<\|image_pad\|>` (id 151655) | [`omni_audio_ops.py:68`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L68) | Same string image VAE uses for its pad slots |
| Register slot | `<\|endoftext\|>` (id 151643) | [`omni_audio_ops.py:66`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L66) | Same as image register; `amount=0` today so never emitted |

The class docstring at [`omni_audio_ops.py:40-43`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L40-L43) flags this explicitly as temporary: *"Qwen3 has no dedicated audio special tokens, so boundary/register/pad tokens reuse the vision-side analogs (all single-token specials in Qwen3). Flip these once the tokenizer gains dedicated audio tokens."*

The merged Qwen3-VL-2B-Audio-POC tokenizer (see [`qwen3_audio_graft_poc.md` §1.2](qwen3_audio_graft_poc.md)) ships fresh audio specials at non-colliding IDs:

| Audio role | New string | New ID |
| --- | --- | --- |
| Span open | `<\|audio_start\|>` | 151669 |
| Span close | `<\|audio_end\|>` | 151670 |
| Frame placeholder | `<\|audio_pad\|>` | 151676 |

The flip is mechanically simple — change four defaults in `OmniElementVAEAudio.Config` and the new constants — but a few inference-side call sites also have to follow because they construct audio spans by hand using hardcoded image IDs (e.g. [`generate_modality_disaggregated.py:1828-1836`](lib/ursa/ursa/models/omni/inference/generate_modality_disaggregated.py#L1828-L1836) and the audio inference processor at [`projects/kuma/kuma/projects/omni/audio/inference/processor/t2a_processor.py`](projects/kuma/kuma/projects/omni/audio/inference/processor/t2a_processor.py)). [§5.5 of the audio-graft plan](qwen3_audio_graft_poc.md) calls this out as a separate task; we fold it into this proposal because the constants and the producer Config live in the same files and reviewing them together is cheaper than splitting.

### 3.2 `vae_token_mask` is overloaded

`vae_token_mask` currently means "this token's input embedding is replaced by a VAE encoder output, and its position is a denoising target if `noisy_vae_token_mask` is also set." That worked when image was the only VAE-stream modality. Audio's pad slots set `vae_token_mask=1` too ([`omni_audio_ops.py:154-163`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L154-L163)), so today the mask means image-or-audio.

The modality split is recovered downstream by the per-element string field `x_vae_by_modality` (e.g. `"t2a"` for audio, `"t2i"` for image) — a string at element granularity that the trainer broadcasts into per-token branching at every consumer site that cares about modality. This is the "scattered classifier" pattern called out in [§10.3.1 of the deep-dive](omni_t2a_dataloader_deep_dive.md):

```python
# audio diffusion loss head, today:
diffusion_loss_audio = mse(model_out[noisy_vae_token_mask & is_audio_modality(x_vae_by_modality, split_lens)])
                                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                          per-element string broadcast to per-token bool
                                                          via split_lens, intersected with mask
```

The failure modes:

- **Silent attribution bugs.** If a new audio task adds modality string `"a2a"` and an `is_audio_modality` classifier somewhere doesn't learn it, audio tokens get routed into the image diffusion loss with no error.
- **Maintenance locality.** "What counts as audio" is decided **N times** across consumer sites, not once at the producer.
- **Granularity mismatch.** The per-element string has to be broadcast to per-token booleans every time a consumer wants to act on a modality.

Counts (from [§11.5](omni_t2a_dataloader_deep_dive.md)):

| Field | ursa | koba | koba_shared | omni/bagel | omni/audio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vae_token_mask` | 14 | 6 | 10 | 22 | 10 |
| `x_vae_by_modality` | 3 | 1 | 6 | 15 | 5 |

Each `x_vae_by_modality` site is a place a developer added a per-modality predicate. Adding `aut_token_mask` extends the pattern; adding A2A modality extends it again.

### 3.3 `clean_vae_img_mask` is misnamed

| Field name | Type | Set on which elements | Set by |
| --- | --- | --- | --- |
| `clean_vae_img_mask` | `bool \| Tensor \| None` (per element, becomes `Tensor[N_elements]` after packing) | `CLEAN_VAE_IMAGE`, **and `CLEAN_VAE_AUDIO`** | [`omni_vae_ops.py`](lib/koba_shared/koba_shared/processor/omni_vae_ops.py) (image) and [`omni_audio_ops.py:181`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L181) (audio) |
| Read by | sampler indexing in [`tdm_sampler.py:181`](lib/ursa/ursa/models/omni/inference/tdm_sampler.py#L181) (the only consumer): `if not kwargs_cond["clean_vae_img_mask"][index]: ...` | | |

Three problems:

1. **Not image-specific.** Audio sets it too. The `_img_` infix is wrong by inspection.
2. **Not a per-token mask.** The `_mask` suffix in this codebase otherwise means "per-token bool tensor" by convention. `clean_vae_img_mask` is a per-element scalar bool, breaking the convention.
3. **Asymmetric without justification.** No `noisy_vae_img_mask` exists; the single 0/1 field encodes both branches inversely. That's fine as a representation, but inconsistent with the per-token family which has both `clean_vae_token_mask` and `noisy_vae_token_mask`.

### 3.4 Missing `aut_token_mask`

The understanding stream has a per-token mask for image-understanding tokens (`vit_token_mask`, set by [`OmniElementVit`](lib/koba_shared/koba_shared/processor/omni_vit_ops.py)) but **no analog for audio**. Today audio enters the omni stack only via the VAE generation stream; an audio-understanding stream is on the [§5.1 of the audio-graft plan](qwen3_audio_graft_poc.md) roadmap.

The needed mask is the audio analog of `vit_token_mask`:

```python
und_mask = text_token_mask | vit_token_mask                 # today
und_mask = text_token_mask | vit_token_mask | aut_token_mask # after §5.1
```

Status today: zero hits for `aut_token_mask` in the codebase. The enum value `SequenceType.VIT_AUDIO=8` is reserved at [`common/types.py:36`](lib/koba_shared/koba_shared/common/types.py#L36) but no producer or consumer exists yet — it's a half-done reservation waiting to be completed.

The trainer-side dispatch lives in the same vicinity as `vit_token_mask`'s dispatch ([`model.py:580-614`](lib/ursa/ursa/models/omni/model/model.py#L580-L614)) — a sibling branch that calls a future `audio_tower` and `audio_to_llm_proj` instead of `vit_patchifier` and `vit_to_llm_proj`. That wiring is §5.1 work, not this proposal's; we deliver the field name and the data-pipeline hook so §5.1 has a clean slot to attach to.

### 3.5 `VIT_AUDIO` is a category mistake

The enum name `VIT_AUDIO` reads as "Vision Transformer ... Audio?" — a category mistake on its face. The enum value's actual meaning is "this element belongs to the **understanding stream** carrying audio features," which has nothing intrinsic to do with vision or transformers.

Footprint of the rename:

| File | Touches |
| --- | --- |
| [`lib/koba_shared/koba_shared/common/types.py:36`](lib/koba_shared/koba_shared/common/types.py#L36) | 1 (declaration) |
| [`lib/koba_shared/koba_shared/processor/omni_text_ops.py`](lib/koba_shared/koba_shared/processor/omni_text_ops.py) lines 137, 147, 157, 165 | 4 (audio-aware switch in text processor) |
| [`lib/koba_shared/koba_shared/processor/position_ids_dev.py:152`](lib/koba_shared/koba_shared/processor/position_ids_dev.py#L152) | 1 (NOTE comment) |
| `lib/koba/docs/processors/tokenization.md` + audio project docs | 3 (cross-references) |

7 code touches, all in audio-aware code paths. No image / video / sampler / inference consumer reads the symbol.

A parallel concern: `VIT_IMAGE` and `vit_token_mask` carry the same category mistake on the image side. The honest fix would be `VIT_IMAGE → UND_IMAGE` (or `AUT_IMAGE`, awkward in the other direction) and `vit_token_mask → und_image_token_mask`. **This is much more expensive** — 12 hits in `ursa`, 12 in `omni/bagel`, 6 in `koba_shared` per [§11.5](omni_t2a_dataloader_deep_dive.md). We do NOT include this in the current proposal; see §5.1 for the deferred-work argument.

## 4. Proposed end-state

### 4.1 Special tokens

Add three audio-side strings to [`lib/koba_shared/koba_shared/common/omni_constants.py`](lib/koba_shared/koba_shared/common/omni_constants.py):

```python
AUDIO_START_TOKEN          = "<|audio_start|>"
AUDIO_END_TOKEN            = "<|audio_end|>"
# Placeholder used by BOTH audio understanding (aut_token_mask=1, row overwritten by audio_tower)
# and audio generation (vae_token_mask + audio_vae_token_mask=1, row fed to audio VAE).
# Embedding row is never read at these positions — mirrors <|image_pad|>'s dual role.
AUDIO_PAD_TOKEN            = "<|audio_pad|>"
QWEN3_ID_AUDIO_START_TOKEN = 151669
QWEN3_ID_AUDIO_END_TOKEN   = 151670
QWEN3_ID_AUDIO_PAD_TOKEN   = 151676
```

The block-comment on `AUDIO_PAD_TOKEN` is load-bearing for review: it tells a future reader that the dual role is intentional and matches the existing image precedent, not an accidental overload. The image side already does the same thing — `<|image_pad|>` is set both by `OmniElementVITImage` (understanding) at [`omni_vit_ops.py:34-38`](lib/koba_shared/koba_shared/processor/omni_vit_ops.py#L34-L38) and `OmniElementVAEImage` (generation) — distinguished by which mask the producer writes, not by the string.

Flip [`OmniElementVAEAudio.Config`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L64-L68) defaults from `VISION_*` / `QWEN3_IMAGE_PAD_TOKEN` → the three new constants. **Audio register stays as `QWEN3_PAD_TOKEN` (`<|endoftext|>`), shared with vision register.** Rationale in §5.4 below.

### 4.2 Per-token masks: shrink to stream-membership only

Materialized fields on [`TokenizedSequenceElement`](lib/koba_shared/koba_shared/processor/tokenized_types.py):

| Field | Granularity | Meaning | Producer (current → end-state) |
| --- | --- | --- | --- |
| `text_token_mask` | per token | LM embedding lookup | All producers (each writes 1 at its text-stream positions) |
| `vae_token_mask` | per token | VAE encoder dispatch (any branch, any modality) | `OmniElementVAE{Audio,Image}` |
| `vit_token_mask` | per token | ViT image-understanding projector dispatch | `OmniElementVit` |
| **`aut_token_mask`** (new) | per token | Audio understanding (audio_tower) dispatch | future `OmniElementAuT`; declared dormant in this proposal |
| `txt_loss_mask` | per token | Text CE loss positions | `OmniElementText` (unchanged) |
| `padding_mask` | per token | Sequence-tail padding | `OmniQwen3Tokenizer` / `pack_sequence` (unchanged) |

Dropped from the materialized set (computed on demand via the helper in §4.4):

| Dropped field | Today's role | How it's recovered |
| --- | --- | --- |
| `clean_vae_token_mask` | per-token clean-branch flag | `vae_mask(branch="clean")` from `SequenceType` + `split_lens` |
| `noisy_vae_token_mask` | per-token noisy-branch flag (also doubles as diffusion-loss target) | `vae_mask(branch="noisy")` |
| `clean_vae_img_mask` | per-element clean flag (sampler) | Helper exposes `is_clean_vae_element(seq_types)` returning `Tensor[N_elements]`; replaces the misnamed field. Same data, correct name, correct granularity-suffix. |
| `video_vae_token_mask` | per-token video dispatch (model.forward kwarg) | **NOT dropped in this proposal.** Stays as kwarg through Phase 3. Phase 4 (optional) migrates video into the per-element world; see §5.5. |

### 4.3 Per-element source of truth

| Per-element field | Status | Meaning |
| --- | --- | --- |
| `type: SequenceType` | unchanged location, **rename `VIT_AUDIO` → `AUT_AUDIO`** | Encodes (role, branch, modality) for image / audio / text exhaustively. |
| `task: str` (renamed from `x_vae_by_modality`) | renamed | Task tag (e.g. `"t2a"`, `"t2i"`, `"a2a"`) used by the sampler for timestep-shift mappings and per-task branching. **No longer used for stream / loss classification** — that's `SequenceType`'s job. |
| `vae_latent_shapes` | unchanged | Per-VAE-element shape: `(H, W)` for image, `(L,)` or `None` for audio. |
| `attention_mode` | unchanged | One of `causal` / `full` / `noise`. |

The `SequenceType` enum after the rename:

```python
class SequenceType(Enum):
    TEXT             = 0
    NOISY_VAE_IMAGE  = 1
    CLEAN_VAE_IMAGE  = 2
    VIT_IMAGE        = 3   # NOT renamed in this proposal (parallel cleanup deferred — see §5.1)
    PACKED           = 4
    TEXT_INCOMPLETE  = 5
    NOISY_VAE_AUDIO  = 6
    CLEAN_VAE_AUDIO  = 7
    AUT_AUDIO        = 8   # was VIT_AUDIO; same value, different name
```

### 4.4 The helper module

A new module — proposed location `lib/koba_shared/koba_shared/processor/seq_type_predicates.py`, ~80 lines — owns the broadcast-and-predicate logic that was previously scattered across consumers:

```python
def broadcast_seq_types(seq_types: list[SequenceType], split_lens: list[int]) -> Tensor:
    """Per-element list → per-token tensor of SequenceType ints.
    Materialized once per packed sample in the trainer; cached on the batch dict."""
    return torch.cat([torch.full((L,), int(t)) for t, L in zip(seq_types, split_lens)])

def vae_mask(per_token_seq_types, *, branch=None, modality=None) -> Tensor:
    """Return the per-token bool mask of VAE slots matching (branch, modality).
    Either argument can be omitted for an axis rollup ('any clean', 'any audio')."""
    target_types = _types_for(branch, modality)   # exhaustive switch on enum
    return torch.isin(per_token_seq_types, target_types)

def is_clean_vae_element(seq_types: list[SequenceType]) -> Tensor:
    """Replaces clean_vae_img_mask. Per-element bool. Used by tdm_sampler.py."""
    return torch.tensor([t.name.startswith("CLEAN_VAE_") for t in seq_types])
```

Three properties this gives us:

1. **Single source of truth.** "What counts as audio" is decided exactly once, in `_types_for(modality="audio")`. Adding a future modality (`video` in Phase 4, `a2a` task variant) is one line in this module; no consumer site changes.
2. **Type-safe predicates.** Consumers replace `noisy_vae_token_mask & is_audio_modality(x_vae_by_modality, split_lens)` with `vae_mask(branch="noisy", modality="audio")`. The argument names are checked at the predicate level; misspellings fail at call time, not silently in production.
3. **No materialization cost.** Each predicate is a single vectorized `torch.isin` call on a length-`S` int tensor, amortized to once per packed sample. Materializing the four `{clean,noisy} × {vision,audio}` leaves (the [§10 alternative](omni_t2a_dataloader_deep_dive.md)) buys nothing measurable here.

### 4.5 Side-by-side: current vs proposed

The table below is the heart of the proposal. Everything else is justification.

| Concern | Today | After this proposal |
| --- | --- | --- |
| Audio span open / close strings | `<\|vision_start\|>` / `<\|vision_end\|>` (image's) | `<\|audio_start\|>` / `<\|audio_end\|>` (fresh) |
| Audio frame placeholder | `<\|image_pad\|>` (image's) | `<\|audio_pad\|>` (fresh, dual-role for understanding+generation) |
| Audio register | `<\|endoftext\|>` (shared) | `<\|endoftext\|>` (shared, unchanged — see §5.4) |
| Stream-membership masks (per token) | `text` / `vae` / `vit` | `text` / `vae` / `vit` / **`aut`** |
| Per-token branch masks (clean / noisy) | `clean_vae_token_mask`, `noisy_vae_token_mask` | **dropped** — derive from `SequenceType` via `vae_mask(branch=...)` |
| Per-token modality classification | implicit via `x_vae_by_modality` broadcast (string) | derive from `SequenceType` via `vae_mask(modality=...)` (typed enum) |
| Per-element clean flag | `clean_vae_img_mask` (misnamed; audio sets it) | **dropped** — `is_clean_vae_element(seq_types)` returns same data with right name and granularity |
| Per-element classification | `x_vae_by_modality` (string, scattered classifiers) | `SequenceType` is canonical; `task` (renamed) carries only sampler-relevant task tag |
| Audio understanding enum value | `VIT_AUDIO=8` (category mistake) | `AUT_AUDIO=8` (same value, fixed name) |
| Image understanding enum value | `VIT_IMAGE=3` (same category mistake) | **unchanged in this proposal** — see §5.1 |
| Video VAE mask | `video_vae_token_mask` as `model.forward` kwarg | **unchanged** through Phase 3; Phase 4 (optional) migrates it |

## 5. Design rationale

This section answers the most-likely review questions in advance. Each subsection is a "why" — for each design choice, why this and not the cheap alternative, the slick alternative, the obvious alternative.

### 5.1 Why not also rename `VIT_IMAGE` / `vit_token_mask` symmetrically?

Honest answer: cost. The footprints are not symmetric:

| Symbol | Code touches | Teams |
| --- | --- | --- |
| `VIT_AUDIO` → `AUT_AUDIO` | 7 (4 in one file) | audio + tokenizer test |
| `vit_token_mask` → `und_image_token_mask` | ~50+ across `ursa`, `omni/bagel`, `koba_shared` | image + ViT + every inference processor |

Renaming `VIT_IMAGE` and `vit_token_mask` would land us in a ~4× larger PR with multi-team CODEOWNERS gating, and the audio-side wins from this proposal don't depend on it. The asymmetry "AUT_AUDIO + vit_token_mask" is ugly but stable; closing it is a follow-up PR that can ship on its own cadence without blocking anything else.

A second-order argument: the `VIT_AUDIO` rename actively prevents a category mistake from spreading. Today there's only one enum value with the bad shape. Once §5.1 of the audio-graft plan adds `OmniElementAuT` and an audio understanding dispatch branch, every line of that new code refers to "AuT" by name; if we leave `VIT_AUDIO` in place, half the audio-understanding code reads `VIT_AUDIO` (because that's the enum value it dispatches on) and half reads `AuT...`. Renaming now keeps the new code consistent with itself.

### 5.2 Why derive masks from `SequenceType` (§12) rather than materialize four leaf masks (§10)?

The deep-dive originally proposed [§10's 5-mask scheme](omni_t2a_dataloader_deep_dive.md): `vae_token_mask` (union) plus four leaves `{clean,noisy}_{vision,audio}_vae_token_mask`. After internal review, [§12.8](omni_t2a_dataloader_deep_dive.md) recommends the alternative — derive everything from `SequenceType`. The argument:

| Aspect | §10 (materialize 4 leaves) | §12 (derive from `SequenceType`) |
| --- | --- | --- |
| Information content | Same | Same |
| Source of truth | Per-element processor writes leaves; `SequenceType` is duplicated | `SequenceType` only |
| Drift risk (leaves vs union inconsistent) | Possible, needs phase-0 assertion | None — no separate stored copy |
| Drift risk (leaves diverge across producer files) | Possible — three processors must stay in sync | None — single helper |
| Memory cost per packed sample | 4 extra bool tensors (~tens of KB) | One `int` per element (negligible) |
| Hot-path cost | Free (direct read) | One vectorized `isin` (~free) |
| Maintenance per new modality | Touch every producer that writes the leaf | Add one entry in helper's enum→category map |
| Debug-ability | Direct: print materialized tensor | Slightly indirect: run helper |

§12 wins on **drift risk**, **memory**, **maintenance per new modality**, and **conceptual surface area**. §10 wins narrowly on debug-ability (direct tensor inspection) and on call-site readability (`noisy_audio_vae_token_mask` reads as a noun where `vae_mask(branch="noisy", modality="audio")` reads as a function call). Net: §12.

The decisive consideration is the **adding-a-modality** row. Today there are two `koba_shared` per-element processors (image, audio); §5.1 of the audio-graft plan adds a third (`OmniElementAuT`); Phase 4 (optional, see §5.5) adds a fourth (`OmniElementVAEVideo`). Each new processor with §10 in place has to write four bool tensors per element; with §12 in place, the new processor writes nothing extra and the helper's enum-to-category map gets one line.

### 5.3 Why share the `<|audio_pad|>` string between understanding and generation streams?

Three reasons:

1. **The image stack already does it.** `<|image_pad|>` plays both roles (set by [`OmniElementVITImage`](lib/koba_shared/koba_shared/processor/omni_vit_ops.py#L36) for understanding and by `OmniElementVAEImage` for generation), distinguished by which mask the producer writes. Adopting the same pattern on the audio side is the path of least conceptual surprise.
2. **The string's embedding row is never read at these positions.** Both consumers (image / audio understanding and image / audio VAE) overwrite the row at runtime — ViT projector output for understanding, VAE encoder output for generation. The string is just an ID slot for M-RoPE positioning; the mask decides which downstream weights produce the value.
3. **The opposite design — separate `<|audio_pad_und|>` and `<|audio_pad_gen|>` — costs us a dedicated tokenizer slot per modality per stream and gains nothing.** No consumer disambiguates by string today; they all disambiguate by per-element type (now `SequenceType`) and per-token mask. Two strings would just be two ways to write the same thing.

### 5.4 Why keep the register token shared across modalities?

The detailed analysis is in [§8.7 of the deep-dive](omni_t2a_dataloader_deep_dive.md). Summary:

- Registers are differentiated by **position + attention pattern**, not by token id. All `M` register slots inside a single noisy block already share the same id by design.
- Attention isolation (§8.1 noise-mode pass-2 column zero-out) prevents cross-modality leakage — an image-block register cannot be read by an audio-block register and vice versa, regardless of id.
- The mixed-modality concern (one packed sample with both image-noise and audio-noise blocks) is real but mild: id 151643's embedding row receives gradient from two summary roles, but the model recovers via positional and surrounding-token context.
- Adding `<|vision_register|>` and `<|audio_register|>` would be a tokenizer-config change with near-zero risk on the audio side **but force retraining of every loaded image checkpoint** (the image register row's prior is currently in slot 151643 and would move). Cost is too high for a register-table cleanup that doesn't unblock anything immediate.

Recommendation: keep shared. Revisit when we next have appetite to retrain the image side.

### 5.5 Why defer video?

Video is structurally different from image and audio in the omni stack:

| Concept | Image | Audio | Video |
| --- | --- | --- | --- |
| `SequenceType` enum value(s) | `*_VAE_IMAGE`, `VIT_IMAGE` | `*_VAE_AUDIO`, `AUT_AUDIO` (after rename) | **none** |
| `koba_shared` per-element processor | `OmniElementVAEImage`, `OmniElementVit` | `OmniElementVAEAudio`, future `OmniElementAuT` | **none** |
| Per-token mask declared on `TokenizedSequenceElement` | yes | yes | **no** |
| Top-level `model.forward` mask arg | n/a | n/a | `video_vae_token_mask` ([model.py:496](lib/ursa/ursa/models/omni/model/model.py#L496)) |
| Origin of the mask at runtime | per-element processor in `koba_shared` | per-element processor in `koba_shared` | upstream packing in `omni/bagel`, fed in as a kwarg |

Video lives entirely in the `omni/bagel` packing path. There is no `OmniElementVAEVideo`, no `SequenceType.*_VAE_VIDEO`, no per-token field on `TokenizedSequenceElement`. Migrating it into the per-element world requires:

- Adding `NOISY_VAE_VIDEO` / `CLEAN_VAE_VIDEO` to `SequenceType`.
- Writing `OmniElementVAEVideo` in `koba_shared`.
- Re-routing the mask construction in `omni/bagel` packing code to come from the per-element processor.
- Updating `model.py:660-680` to derive `video_vae_token_mask` from the helper instead of taking it as a kwarg.

That's its own multi-team PR with `omni/bagel` and video-task CODEOWNERS gating. Bundling it into this proposal would inflate the diff by ~50% and add a team to the review path that the audio cleanup doesn't otherwise need.

**Plan:** Phase 4 (optional, after Phase 3 settles) does the video migration. Phases 0–3 leave video exactly as it is — `video_vae_token_mask` stays a `model.forward` kwarg, packed by `omni/bagel`. The helper module's predicate library covers `modality ∈ {image, audio}` only until Phase 4 lights up `modality="video"`.

### 5.6 Why optionally consider an `ElementKind` refactor (Phase 5)?

`SequenceType` today is a flat 9-value enum. Adding a modality (e.g. `*_VAE_VIDEO` in Phase 4) means adding two enum values; adding a stream role for that modality (`*_VIT_VIDEO`?) adds another. The combinatorial explosion is small at three modalities (image, audio, video) but visible.

[§12.9 of the deep-dive](omni_t2a_dataloader_deep_dive.md) sketches a refactor:

```python
@dataclass(kw_only=True, slots=True, frozen=True)
class ElementKind:
    role:     Literal["text", "vae", "vit", "aut", "packed", "text_incomplete"]
    branch:   Literal["clean", "noisy"] | None     # only meaningful for vae
    modality: Literal["image", "audio", "video"] | None  # only meaningful for vae/vit/aut
```

Pros: the helper's predicate library becomes trivial (`role == "vae" and branch == "noisy" and modality == "audio"`); adding a future role × modality combination is one constructor line, not a new enum value.

Cons: every `if tok_element.type == SOMETHING_ENUM:` branch in the codebase becomes `if tok_element.kind.role == "..." and ...:`. That's a syntactic rewrite of every per-element processor and every dispatch switch. Larger blast radius than this proposal as a whole.

**Plan:** Phase 5 (optional, after Phase 4). Worth doing **only** if Phase 4 brings video into the per-element world (so the combinatorial pressure becomes real); not worth doing if video stays as-is. We mention it for completeness but do not commit.

## 6. Migration plan

The migration is structured as five additive phases, each shippable on its own. No phase forces a breaking change on a consumer team in isolation; legacy fields stay populated until Phase 2 explicitly removes them.

### Phase 0 — additive only

**Goal:** put the new names in place; keep all legacy fields populated; no consumer is forced to migrate yet.

| Step | File(s) |
| --- | --- |
| Add the three audio token strings + IDs | [`omni_constants.py`](lib/koba_shared/koba_shared/common/omni_constants.py) |
| Flip `OmniElementVAEAudio.Config` defaults to the new constants | [`omni_audio_ops.py:64-68`](lib/koba_shared/koba_shared/processor/omni_audio_ops.py#L64-L68) |
| Audit hand-built audio span construction in inference; route to `QWEN3_ID_AUDIO_*` | [`generate_modality_disaggregated.py:1828-1836`](lib/ursa/ursa/models/omni/inference/generate_modality_disaggregated.py#L1828-L1836), [`t2a_processor.py`](projects/kuma/kuma/projects/omni/audio/inference/processor/t2a_processor.py) |
| Rename `SequenceType.VIT_AUDIO → AUT_AUDIO` (value 8 preserved) | [`common/types.py:36`](lib/koba_shared/koba_shared/common/types.py#L36); 6 reference sites |
| Add `aut_token_mask: Tensor \| None = None` to `TokenizedSequenceElement` | [`tokenized_types.py`](lib/koba_shared/koba_shared/processor/tokenized_types.py) |
| Add CI assertion: `aut_token_mask is None or aut_token_mask.any() == False` until §5.1 wires producer | tests |
| Expose `seq_types: list[SequenceType]` in the packed-batch dict (carry-through from existing per-element `type` field) | [`sequence_packing.py`](lib/ursa/ursa/models/omni/inference/sequence_packing.py) |
| Add the helper module `seq_type_predicates.py` with `broadcast_seq_types`, `vae_mask`, `is_clean_vae_element` | new file |
| Add a phase-0 invariant assertion: at every batch boundary, `vae_token_mask == vae_mask(branch="any", modality="any")` (legacy equals derived) | tests / debug-mode |

Phase 0 ships with zero consumer changes. The new names exist; the old names still work; the helper is available for any consumer that wants to start using it.

### Phase 1 — switch consumers

**Goal:** every consumer of the legacy mask names migrates to either the helper or the new fields. Phase 0's assertions keep us honest during the migration.

| Consumer | Today | After |
| --- | --- | --- |
| Trainer audio path (`model.py`) | reads `vae_token_mask` + `x_vae_by_modality` to dispatch audio | reads `vae_mask(modality="audio")` |
| Audio diffusion loss head | `noisy_vae_token_mask & is_audio_modality(x_vae_by_modality, split_lens)` | `vae_mask(branch="noisy", modality="audio")` |
| Image diffusion loss head | `noisy_vae_token_mask & is_image_modality(x_vae_by_modality, split_lens)` | `vae_mask(branch="noisy", modality="image")` |
| Sampler ([`tdm_sampler.py:181`](lib/ursa/ursa/models/omni/inference/tdm_sampler.py#L181)) | `kwargs_cond["clean_vae_img_mask"][index]` | `is_clean_vae_element(seq_types)[index]` |
| Inference span builders (`generate_modality_disaggregated.py`, `t2a_processor.py`) | hand-broadcast `clean_vae_token_mask` etc. | helper-derived |
| Tests (`test_t2a_data_roundtrip.py`, `test_audio_position_ids_offset.py`, `dummy_dataset.py`) | populate legacy masks | populate via helper or new fields |

Phase 1 lands file-by-file. Phase 0's invariant assertion catches any consumer that forgets to migrate.

### Phase 2 — drop legacy fields

**Goal:** delete the old mask fields once they have zero readers.

After Phase 1 lands and bakes for a release cycle:

- Drop `clean_vae_token_mask` and `noisy_vae_token_mask` from `TokenizedSequenceElement`.
- Drop `clean_vae_img_mask` from `TokenizedSequenceElement`.
- Update producers (`OmniElementVAEAudio`, `OmniElementVAEImage`) to no longer write them.

After Phase 2, the data pipeline's per-token mask family is exactly: `text_token_mask`, `vae_token_mask`, `vit_token_mask`, `aut_token_mask`, `txt_loss_mask`, `padding_mask`. Six fields, each with one well-defined role.

### Phase 3 — collapse `x_vae_by_modality`

**Goal:** retire the per-element string field by splitting it into its remaining role.

After Phase 2, `x_vae_by_modality`'s only remaining roles are:

- `vae_latent_shapes` indexing (per-element shape lookups).
- Sampler-time per-task branching (timestep shift mappings, etc.).
- Telemetry / logging.

The first is structural and orthogonal to modality classification. The second and third are about **task tags** (`"t2a"`, `"i2i"`, etc.), not modality (`"image"`, `"audio"`). Rename `x_vae_by_modality` → `task` to reflect the remaining purpose; modality classification has already moved to `SequenceType` in Phase 1.

### Phase 4 — video into the per-element world (OPTIONAL)

**Goal (only if video team has appetite):** unify the data-pipeline surface across all VAE modalities.

| Step | File(s) |
| --- | --- |
| Add `NOISY_VAE_VIDEO` / `CLEAN_VAE_VIDEO` to `SequenceType` | [`common/types.py`](lib/koba_shared/koba_shared/common/types.py) |
| Write `OmniElementVAEVideo` in `koba_shared` (mirrors `OmniElementVAEImage`) | new file |
| Update `omni/bagel` video packing code to populate per-element fields instead of constructing `video_vae_token_mask` directly | `omni/bagel` |
| Update [`model.py:660-680`](lib/ursa/ursa/models/omni/model/model.py#L669-L680) to derive `video_vae_token_mask = vae_mask(modality="video")` instead of taking it as a kwarg | `model.py` |
| Drop `video_vae_token_mask` and `video_vae_latent_shapes` from `model.forward` signature | `model.py` |

After Phase 4, the helper's modality coverage is complete (`image`, `audio`, `video`); nothing in the data-pipeline surface special-cases video.

### Phase 5 — `ElementKind` refactor (OPTIONAL, after Phase 4)

**Goal:** replace `SequenceType` flat enum with orthogonal `(role, branch, modality)`. See §5.6 for trade-offs.

Worth doing **only** if Phase 4 lands and the combinatorial explosion of enum values starts to bite. If Phase 4 never lands, Phase 5 buys nothing.

## 7. Coordination with the audio backbone swap

[`qwen3_audio_graft_poc.md` §5](qwen3_audio_graft_poc.md) lays out the audio-graft work. Cross-referencing:

| §5 task | Phase of this proposal | Notes |
| --- | --- | --- |
| §5.1 swap omni-t2a backbone to Qwen3-VL-2B-Audio-POC | Independent — runs alongside Phase 0 | Backbone reads whatever masks the data pipeline produces; phase 0 leaves those untouched |
| §5.3 A2T data processing module | After Phase 0 | A2T builder needs `aut_token_mask` and the helper; both ship in Phase 0 |
| §5.4 CE + diffusion loss mixing | After Phase 1 | Loss head migration is exactly the case Phase 1 targets |
| §5.5 hardcoded special-token IDs | **Folded into Phase 0** | §5.5 and Phase 0's special-token swap touch the same files; coordinate into one PR |

The dependency graph:

```
Phase 0 ──┬──▶ §5.3 (A2T builder) ──▶ §5.4 (loss switch)
          └──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 3
                    │
                    └──▶ §5.4 (loss switch)

§5.1 (backbone swap)        ⊥  Phase 0/1/2/3
§5.5 (token id swap)        ≈  Phase 0 (folded)
Phase 4 (video, optional)   ⊥  §5.x (independent)
Phase 5 (ElementKind, opt)  ⇐  Phase 4
```

Phases 0–3 unblock all of §5; Phases 4–5 are independent of audio's critical path.

## 8. Risks and open questions

### Risks

- **Phase 1 has a dual-edit hazard.** Inference paths (`generate_modality_disaggregated.py`, `t2a_processor.py`) construct audio spans by hand. If a consumer migration skips one of these, the trainer-side reads new fields while the inference-side writes legacy fields and the masks silently disagree. Phase 0's invariant assertion (`vae_token_mask` legacy equals helper-derived) catches this on the trainer side; we should add an analogous assertion at the inference-side span-builder boundary.
- **CI cost.** The phase-0 invariant assertion runs on every batch in debug mode. If profiling shows it on the hot path, gate it behind a `KOBA_VALIDATE_MASKS=1` env var rather than always-on. Free in opt-out mode.
- **Helper memory footprint.** `broadcast_seq_types(seq_types, split_lens)` produces a length-`S` int tensor per packed sample. For `max_num_tokens=8000` that's 8 KB per sample — negligible. Cache it on the batch dict so multiple consumers share the broadcast.
- **Checkpoint compatibility.** Mask renames are data-pipeline state; no model weights move. The audio special-token swap changes which embedding rows audio loads from at the LM input, but only on **new** training runs that adopt the merged Qwen3-VL-2B-Audio-POC tokenizer; pre-existing audio checkpoints are unaffected.

### Open questions for review

1. **Helper module location.** Proposed at `lib/koba_shared/koba_shared/processor/seq_type_predicates.py`. Alternative: inline into `tokenized_types.py` to keep all per-token-mask machinery in one file. Preference?
2. **Should `task` (renamed from `x_vae_by_modality`) become typed?** Today it's a free-form string (`"t2a"`, `"t2i"`, ...). A `Literal["t2a", "t2i", "a2a", ...]` or an enum would catch typos and document the task vocabulary, but every new task definition would have to update the type alias. Worth doing, or out of scope?
3. **Phase-0 invariant assertion: opt-in or opt-out?** Always-on costs ~1% of step time at 8000 tokens (rough estimate, untested). Opt-out (`KOBA_VALIDATE_MASKS=0` to disable) is safer; opt-in (`KOBA_VALIDATE_MASKS=1` to enable) is cheaper. Default?
4. **Image register cleanup (`<|endoftext|>` → `<|vision_register|>`).** This proposal explicitly defers it (§5.4). Is there appetite to ship it in Phase 0 if a vision retraining is already planned in the same window? The flip costs ~10 lines of code but invalidates loaded image checkpoints.
5. **`VIT_IMAGE` / `vit_token_mask` parallel rename (§5.1).** Deferred here. Should it be a separate proposal with its own review cycle, or simply a follow-up PR after Phase 3? Preference?
6. **Phase 4 (video) — committed or speculative?** Video team should weigh in: is there appetite to migrate video into the `koba_shared` per-element world in the next 2 quarters, or should we treat Phase 4 as "if/when, not when"?
7. **`ElementKind` refactor (Phase 5).** Same question — if Phase 4 happens, is Phase 5 worth doing back-to-back (cheap moment because both touch every per-element processor), or is the enum-vs-dataclass gap not worth the rewrite cost?

## 9. Summary

The omni multi-modal data stack carries five name-quality scars from its image-first history. The audio backbone swap and the upcoming audio-understanding stream make this the right moment to clean them up: we have to touch the audio side anyway, and the helper module that derives slices from `SequenceType` is the cleanup that lets every future modality come in additively.

The end-state is:

- **Special tokens:** fresh audio strings on the audio side; image side untouched.
- **Per-token masks:** four stream-membership fields (`text`, `vae`, `vit`, `aut`); branch and modality derive from `SequenceType`.
- **Per-element source of truth:** `SequenceType` (with `VIT_AUDIO → AUT_AUDIO`); a separate `task` field (renamed from `x_vae_by_modality`) carries only sampler-relevant task tagging.
- **Helper module:** owns broadcast and predicates; classification decided once.

The migration is five additive phases, each shippable on its own. Phase 0 is purely additive; no team is forced to migrate in isolation. Phases 4–5 are explicitly deferred / optional and live behind their respective open questions.

The total blast radius is ~30 files across `koba_shared`, `koba`, `ursa`, `omni/audio`, `omni/bagel`, with all of those teams' CODEOWNERS on the review path. The audio-graft plan ([qwen3_audio_graft_poc.md §5](qwen3_audio_graft_poc.md)) folds in cleanly: §5.5 combines with Phase 0; §5.3 and §5.4 follow Phases 0 and 1 respectively.

Reviewers from each team — please flag any consumer in your area that we've missed in §6, and answer §8's open questions for the parts of the codebase you own. The proposal is structured so that any single team's pushback can localize to a phase or a deferral, not a full rejection of the cleanup.
