# Omni-audio extension v2 — Implementation logs

> **Started:** 2026-04-30
> **Last updated:** 2026-05-01
> **Branch:** `dongguo/omni-t2a-v2`
> **Companion docs:**
> - [`audio_extension_v2_plan.md`](audio_extension_v2_plan.md) — design plan (A2T + A2A)
> - [`qwen3_audio_graft_poc.md`](qwen3_audio_graft_poc.md) — POC: zero-shot ASR via Qwen3-ASR-1.7B audio_tower grafted onto Qwen3-VL-2B-Instruct
> - [`omni_t2a_project_notes.md`](omni_t2a_project_notes.md) — overall T2A project context

This doc captures **what landed today** (the project-local audio-encoder
scaffold) plus the design rationale from the multi-round discussion that
preceded it. It is the implementation log for the work specified in
[`audio_extension_v2_plan.md`](audio_extension_v2_plan.md).

---

## 1. Summary

Added a project-local audio-encoder + text-tokenizer scaffold under
`projects/kuma/kuma/projects/omni/audio/model/tokenizers/` that mirrors
the existing `lib/ursa/ursa/models/omni/tokenizers/` layout. The audio
encoder is a faithful vendor of the Qwen3-ASR-1.7B `audio_tower` (verified
byte-for-byte against the released ckpt). Vision encoder is **reused
unchanged** from ursa via `qwen_vit_forge_config(model_size="2B")`. Text
tokenizer is **reused unchanged** from `koba_shared.OmniQwen3Tokenizer` —
only the `hf_path` points at the merged Qwen3-VL-2B-Audio S3 dir.

**Planned usage pattern (decided 2026-05-01, drives §5 / §6 below):**

- The audio encoder is grafted-from-Qwen3-ASR and is intended to be
  trained **exactly once** — a one-time A2T head-only finetuning run
  that adapts `ln_post`/`proj1`/`proj2` to the joint LM-hidden space.
- After that one run, the encoder is **frozen forever** for all
  downstream uses (T2A, A2A, …) and consumed as a fixed feature
  extractor.
- This drives two corollaries:
  1. The trainable surface is **only** these head layers, and only in
     a single experiment. `trainable_param_patterns=("ln_post",
     "proj1", "proj2")` is the canonical setting.
  2. FSDP-wrapping the encoder is unnecessary and counter-productive —
     **replicate** instead (see §6 for the full reasoning).

Smoke-tested in the kuma venv:

- Total params: **317.48M** (24× transformer encoder layers dominate)
- Head-only mode (`trainable_param_patterns=("ln_post", "proj1", "proj2")`):
  3.15M trainable (0.99% of total)
- `state_dict()` keys prefixed `audio_tower.*` — matches both upstream HF
  model (`Qwen3ASRModel.audio_tower`) and the released-ckpt convention
- Forward `(B=1, mels=128, T=100)` → `(1, 13, 2048)` ✓

---

## 2. Directory tree

```
projects/kuma/kuma/projects/omni/audio/model/tokenizers/
├── __init__.py
├── model/
│   ├── __init__.py
│   └── qwen_audio.py                                    # Qwen3ASRAudioModel + Config
├── third_party/
│   ├── __init__.py
│   └── qwen_audio/
│       ├── __init__.py
│       ├── _compat.py                                   # 4.57.1 shims, copied verbatim from POC vendor
│       ├── configuration_qwen_audio.py                  # trimmed: encoder config only
│       └── modeling_qwen_audio.py                       # trimmed: encoder + deps only
└── configs/
    ├── __init__.py
    ├── audio_configs.py                                 # qwen_audio_forge_config + head-only variant
    └── text_tokenizer_configs.py                        # OmniQwen3Tokenizer.Config factory
```

**Mapping to ursa convention.** The layout deliberately mirrors
`lib/ursa/ursa/models/omni/tokenizers/`:

| ursa | omni/audio (project-local) | Reason |
|---|---|---|
| `model/qwen_vit.py` | `model/qwen_audio.py` | Lumaverse-owned wrapper around vendored layer code |
| `third_party/qwen_vit/modeling_qwen_vit.py` | `third_party/qwen_audio/modeling_qwen_audio.py` | Vendored layer implementations |
| `configs/vit_configs.py` | `configs/audio_configs.py` | Forge factories |

Lives under `projects/kuma` (not `lib/ursa`) because T2A is still
experimental — graduate to ursa later, per the convention noted in
`audio/model/preprocess.py` ("Will need to move to
`ursa.models.omni.model.model` in the future").

---

## 3. Design decisions and rationale

### 3.1 Why own the audio-encoder class in lumaverse (not import from `transformers`)

The released Qwen3-ASR-1.7B ckpt has **no `auto_map` in `config.json`** —
it expects upstream `transformers` to provide the class natively. As of
the kuma venv's `transformers 4.57.1`, `qwen3_asr` is not in the library
(only in PR #43838, unreleased). So importing from `transformers` directly
is not even an option today.

Beyond pragmatics, the reasons to own the class internally:

- **Customization fidelity.** The merged Qwen3-VL-2B-Audio model is no
  longer stock Qwen — merged tokenizer (36 ASR specials at IDs
  151669-151704), audio_tower grafted onto VL-2B's LM. Half-HF /
  half-lumaverse is a Frankenstein.
- **No churn from `transformers` version bumps.** PR #43838 may rename or
  refactor classes before merge.
- **Full control of FSDP wrap, AC, parameter naming.** `Qwen3MMDiT` and
  the parallelism config assume in-repo class names.

The bagel side already follows this — `omni/bagel/models/omni_models/`
defines its joint model in-repo, and HF `transformers` is touched only in
one-off ingest scripts (`omni/bagel/scripts/copy_language_tower.py`) and
eval-side helpers, never in the trained graph.

### 3.2 Why reuse `Qwen3VLVisionModel` from ursa for vision (no fork)

`qwen_vit_forge_config(model_size="2B")` in
`lib/ursa/ursa/models/omni/tokenizers/configs/vit_configs.py:51` already
exposes the right hyperparams (depth=24, hidden=1024,
deepstack=[5,11,17], out_hidden=2048) and an `osc://` checkpoint path.
The 2B vision tower differs from the 32B tower only in hyperparams (no
architecture changes), so a project-local fork would be pure duplication.

### 3.3 Why no subclass for `OmniQwen3Tokenizer`

Initial assumption: ASR special tokens (`<|audio_start|>`, etc.) require
extending the tokenizer wrapper. **Wrong.** The four audio role tokens
are already plumbed through `OmniElementVAEAudio.Config` (data side, see
`omni/audio/data/tokenizer_validation.py:30-35`) and the validation
function resolves them to IDs at pipeline-build time via
`tokenizer.encode(token_string, add_special_tokens=False)` — dynamically,
off whatever `tokenizer.json` the `hf_path` points at.

The merged tokenizer at
`s3://ai-lumalabs-checkpoints-ap-se-2/dongguo/ckpts/Qwen3-VL-2B-Audio`
already has the four ASR tokens in `added_tokens_decoder`. So the
"extension" reduces to **a one-line `hf_path` change** in our text
tokenizer factory — no new wrapper code.

### 3.4 Why no module split (`encoder` + `adapter`)

Considered: extract `ln_post` / `proj1` / `proj2` into a sibling submodule
so freezing reduces to `encoder.requires_grad_(False)`. Rejected because:

1. **Vendor fidelity.** Those three are direct attributes on
   `Qwen3ASREncoder` and the `forward()` bakes them inline. Splitting
   means editing the vendored file or wrapping with custom `forward()` —
   either way deviates from the released structure.
2. **Checkpoint-key compat.** Splitting into `body.*` + `adapter.*`
   breaks the `audio_tower.ln_post.*` key match with the released ckpt.
   You'd need a key-rewrite at load time. Doable, but pure friction.

Use **name-pattern freezing** instead, matching the existing lumaverse
convention at `omni/layering/rgba_vae/trainer.py:142-197`. Tomorrow if we
want to also unfreeze the last 2 transformer layers, the tuple grows
from `("ln_post", "proj1", "proj2")` to
`("ln_post", "proj1", "proj2", "layers.22.", "layers.23.")` — no
model-class refactor.

### 3.5 Two non-trivial patches preserved from upstream PR #43838

Both required to load the released Qwen3-ASR-1.7B ckpt without silently
dropping weights:

1. **`Qwen3ASRAttention.k_proj` constructed with `bias=bias`** (default
   `True`). Upstream PR has `bias=False`, but the released ckpt has 24
   trained `k_proj.bias` tensors (verified —
   `thinker.audio_tower.layers.{0..23}.self_attn.k_proj.bias`). Without
   the patch, `load_state_dict(..., strict=True)` silently drops them.
2. **`ALL_ATTENTION_FUNCTIONS.get(...)`** in place of `.get_interface(...)`
   for the 4.57.1 dict-like API. PR uses the newer API not yet in 4.57.1.

Both flagged with `NOTE (vendor patch)` comments inline.

---

## 4. Audio encoder architecture (Whisper-style, chunked)

Pipeline for batch B (defaults from released 1.7B ckpt; T_mel must be a
multiple of `n_window * 2 = 100`):

| Stage | Module | Out shape | Notes |
|---|---|---|---|
| **Input** | — | `(B, 128, T_mel)` | log-mel spectrogram |
| **Chunking** | reshape | `(B*nc, 1, 128, 100)` | 1-second chunks at 16 kHz / 10 ms hop |
| **Conv stem** | `conv2d1` (1→480, k=3, s=2) + GELU | `(B*nc, 480, 64, 50)` | mel and time both halved |
|  | `conv2d2` (480→480) + GELU | `(B*nc, 480, 32, 25)` | |
|  | `conv2d3` (480→480) + GELU | `(B*nc, 480, 16, 13)` | total 8× time downsample |
|  | `conv_out`: `Linear(7680→1024)` | `(B*nc, 13, 1024)` | flattens (channels, freq) |
| **Position** | `SinusoidsPositionEmbedding(1500, 1024)` | `(B*nc, 13, 1024)` | added per-chunk; reset at chunk start |
| **Reshape** | concat chunks | `(B, nc·13, 1024)` | + valid-mask from `_post_cnn_length` |
| **24× encoder layer** | pre-norm Whisper-style | `(B, nc·13, 1024)` | bidirectional self-attn (16 heads, head_dim=64); FFN 1024→4096→1024; valid-mask re-applied per layer |
| **Output head** | `ln_post` → `proj1` (1024→1024) → GELU → `proj2` (1024→2048) | `(B, nc·13, 2048)` | `output_dim=2048` matches Qwen3-VL-2B LM hidden — POC graft works without an external projector |

**Token rate:** 1 second of audio → 100 mel frames → 13 audio tokens at
hidden=2048. ~12.5 ms per output token (cf. Whisper's ~20 ms).

**Per-chunk math:** `(num_mel_bins+1)//2 = 64`, `(64+1)//2 = 32`,
`(32+1)//2 = 16` → `conv_out: Linear(480·16=7680, 1024)`.

**Trainable parameter breakdown** (all of the above):

- conv stem: ~12M
- 24× encoder layer ≈ 12.6M each → ~302M
- output head (`ln_post` + `proj1` + `proj2`): ~3.15M
- Sinusoidal posemb: fixed buffer (not a parameter)
- **Total: 317.48M** (verified via smoke test)

---

## 5. Trainability / freezing strategy

### 5.0 When does any of this run?

Per §1's planned usage pattern: the head-only adapter trains **exactly
once**, in a dedicated A2T finetuning experiment that maps the
ASR-encoder features into the joint LM-hidden space. After that one
run, the encoder is frozen for all downstream consumers (T2A, A2A, …).

So §5.1 / §5.2 describe **the configuration of that single A2T
adapter run**. Every other experiment that *uses* the encoder loads it
with all params frozen — no `trainable_param_patterns` needed (or
equivalently, `trainable_param_patterns=()` to be explicit).

### 5.1 Recommendation for the A2T adapter run: full output head, not just `proj2`

For the one-time A2T adapter pass, tune
`ln_post + proj1 + proj2` (3.15M params), not `proj2` alone (2.1M).

Per-component reasoning:

- **`ln_post`** (~2K params) — LayerNorm scale + bias absorbs
  feature-distribution shift cheaply. Always include.
- **`proj1` + GELU** (~1M params) — only nonlinearity in the post-encoder
  path. Without it, adaptation is *linear* (rotation only), can't compose
  or gate features. Domain shifts that change which features matter
  typically need this nonlinear remap.
- **`proj2`** (~2.1M params) — final 1024→2048 projection into LM hidden
  space. Necessary for the output coordinate system but linear-only.

3.15M is small enough not to overfit on tens of hours of new-domain audio
yet large enough to have real expressive capacity. Standard adapter
surface for "tune the top of an audio encoder" — matches how Whisper-style
encoders are usually adapted.

**LR note:** when refining a pretrained mapping, run the head at 1e-5 to
5e-5, not 1e-3. A larger LR wipes out the alignment that the POC's
zero-shot transfer relies on.

### 5.2 When to deviate

- **`proj2` only** — <10h adaptation data + close domain (accent
  variation, mild noise). Lowest overfit risk.
- **Last 2-4 transformer layers + full head** — head-only plateaus on
  val loss, or large domain shift (new language family, music vs. speech,
  very noisy). Each layer adds ~12.6M params.

### 5.3 How to configure: `trainable_param_patterns`

`Qwen3ASRAudioModel.Config.trainable_param_patterns: tuple[str, ...] | None`.
Substring match against `named_parameters()`. `None` = all trainable.

```python
# Head-only (recommended starting point):
Qwen3ASRAudioModel.Config(trainable_param_patterns=("ln_post", "proj1", "proj2"))

# Convenience factory in audio_configs.py:
qwen_audio_head_only_forge_config(checkpoint_path="osc://...")
```

The wrapper logs `n_trainable / n_total` at construction:

```
Qwen3ASRAudioModel: 3,150,848/317,477,504 params trainable (0.99%)
under patterns ('ln_post', 'proj1', 'proj2').
```

**Operational habit (recommended):** assert `n_trainable / n_total`
matches expectation in the trainer setup — easy to typo a pattern
(`"proj_1"` vs `"proj1"`) and silently freeze something you meant to
train.

### 5.4 Optimizer construction

When using head-only mode, the optimizer **must** filter by `requires_grad`,
or it allocates state for ~317M frozen params (~750 MB Adam state per
rank, sharded by FSDP, but still wasteful):

```python
optim.AdamW([p for p in model.parameters() if p.requires_grad], ...)
```

---

## 6. FSDP / HSDP integration

**Decision: replicate, do NOT FSDP-wrap.**

This supersedes an earlier "single FSDP unit" recommendation that was
made before §1's usage pattern was nailed down. The full analysis of
*why* — including the question of whether frozen, sometimes-unused
encoder params cause torch/distributed bugs at all — is in §6.1; the
three rationale bullets that drive the decision are in §6.2.

### 6.1 Distributed-correctness analysis: do frozen, never-touched encoder params cause torch/distributed bugs?

Worth capturing the full three-layer analysis from the design
discussion, since the answer is non-obvious and the decision in §6.2
hinges on it.

#### 6.1.a Single-rank torch — **no bug.**

Frozen params (`requires_grad=False`) that are never reached in
forward have no graph node, no grad. The optimizer — filtered by
`requires_grad` — skips them. They sit in HBM at rest. The only
cost is idle memory.

#### 6.1.b DDP `find_unused_parameters` — **no bug.**

Classic DDP hang scenario: a param has `requires_grad=True` but is
not reached in forward, so DDP waits forever for a grad that never
arrives. **Doesn't apply here** because everything we freeze has
`requires_grad=False` (set inside the wrapper's
`_apply_trainable_param_patterns`). DDP only waits for grads from
trainable params; frozen params are skipped entirely. ✓

#### 6.1.c FSDP/HSDP collective participation — **the real concern, but only for mixed-modality batches.**

FSDP's `all_gather` (forward-side parameter unshard) and
`reduce_scatter` (backward-side gradient sync) are **collective**
operations: every rank in the process group must call them, in the
same order, or the ones that did will block forever waiting for the
ones that didn't.

If the encoder is FSDP-wrapped, the wrap unit's `all_gather` fires
on `forward()` entry. Concretely:

| Configuration | Collective behaviour | Risk |
|---|---|---|
| FSDP-wrapped encoder, **every batch on every rank uses audio input** (e.g. pure A2T) | Every rank calls `all_gather` on every step → fully consistent. | None. |
| FSDP-wrapped encoder, **no batch on any rank uses audio input** (e.g. current pure T2A — encoder is dead weight) | No rank ever calls `all_gather` on the encoder unit → no collective fires → no participation requirement. | None. |
| FSDP-wrapped encoder, **mixed-modality batches: rank 0's batch has audio, rank 1's doesn't** (future T2A + A2T + A2A under the v2 plan) | Rank 0's `all_gather` blocks waiting for rank 1, which never calls it. | **Deadlock.** |

So the hazard is *specifically* about mixed-modality batch composition
across ranks, not about frozen unused params per se.

#### 6.1.d Three mitigations for the mixed-modality case

1. **Don't FSDP-wrap; replicate.** Frozen + replicated has no
   collective requirement at all on the encoder's params. Memory
   cost ~635 MB / GPU at fp16 — cheap. **This is what we choose
   (§6.2).** Strictly the cleanest given §1's usage pattern.
2. **Always-call-forward.** For batches without audio, pass a dummy
   1-chunk silent input through the encoder, discard the output.
   Every rank participates in collectives uniformly. Keeps FSDP's
   memory-at-rest savings but adds compute waste on every audio-less
   batch and adds non-trivial pipeline-side wiring (the dummy must
   be safe to encode and the output must be safely discarded
   downstream).
3. **Rank-balanced data loading.** Enforce identical modality
   composition per rank — every rank's batch has the same number of
   audio samples. Brittle: bucket schedulers, restarts, and
   variable-length sequences make this hard to maintain across long
   runs. Considered and rejected.

#### 6.1.e Adjacent torch quirks (none of which were issues, listed for closure)

- **Checkpointing**: `state_dict()` includes frozen encoder params; ckpt
  size includes them. Loading via `InitializerFromSingleCheckpoint`
  with `strict=True` works since every encoder key has a destination.
- **`torch.compile` + dynamic-shape**: with `dynamic=False,
  fullgraph=False` (per `base_t2a`), control-flow over
  audio-presence-per-batch could trigger guard rebuilds. Not an
  issue for the planned usage (encoder is consumed via a fixed
  per-pipeline path), but flagged for future mixed-batch work.
- **Activation memory**: 24-layer encoder × seq_len activations can
  be non-trivial for very long audio. Not a problem at typical
  T2A token counts (~13 tokens / second of audio) but worth
  re-checking under A2T's longer-audio settings.

### 6.2 Why replicate (the decision)

Three reasons, in order of weight:

1. **Memory cost is negligible.** ~317M params replicated at fp16 =
   ~635 MB per GPU. Well under noise on H100 / B200 setups; saving
   this is not worth any added complexity.
2. **Frozen forever (§1) → no benefit from FSDP wrapping.** FSDP
   pays off when grads need to be reduce-scattered across the shard
   group. Frozen params have no grads → no reduce-scatter saving →
   the only thing FSDP buys is HBM-at-rest, which is cheap to give
   up at this size.
3. **Replication sidesteps the §6.1.c hazard entirely.** With no
   FSDP wrap there are no per-encoder collectives → no deadlock
   surface, regardless of how mixed-modality batch composition
   evolves. This becomes load-bearing once T2A + A2T + A2A
   mixed-batch training lands per the v2 plan.

### 6.3 Concrete settings

- **Don't register `Qwen3ASRAudioModel` in `ParallelismConfig`'s
  FSDP-wrap entries.** Leave it unwrapped → replicated on every rank.
- **No activation checkpointing on the encoder.** It's frozen
  forever in the trained graph; backward through it short-circuits
  via `requires_grad=False` and there are no grads to recompute.
  Even during the one A2T adapter run, only the head is trainable
  (3.15M params) — AC on the body wastes recompute for grads that
  will be discarded.
- **Keep `attn_implementation="sdpa"`** as the default (set on the
  vendored encoder config in our wrapper). No flash-attn opt-in
  needed.
- **Optimizer must filter on `requires_grad`** during the A2T
  adapter run to avoid allocating Adam state for the ~314M frozen
  params (~750 MB / rank, wasted): `[p for p in m.parameters() if
  p.requires_grad]`.

### 6.4 The one wrinkle for the A2T adapter run

During that single A2T training experiment, the encoder *does*
participate in forward and backward (head grads flow through the
adapter). Two sub-decisions:

- **Replication still applies.** No FSDP wrap on the encoder; it
  lives replicated on every rank. The trainable head's grads (3.15M
  params) are tiny — DDP-style all-reduce on those is cheap and
  doesn't need FSDP.
- **Per-rank participation is consistent.** Every batch in the A2T
  run carries audio input by construction, so the encoder fires on
  every rank for every step → no cross-rank divergence in the
  trainable head's grad-sync collective.

### 6.5 Things explicitly NOT done

- Per-layer FSDP wrapping of `Qwen3ASREncoderLayer`.
- Kuma's AC on encoder layers.
- Mixed-precision dtype overrides per-submodule.
- `use_orig_params` opt-ins beyond the default.
- Treating the encoder as a candidate for HSDP-style sharding (the
  trainable surface is too small to benefit).

### 6.6 Param-count corrections from earlier discussion

I quoted "~95M params" for the audio encoder during the initial FSDP
discussion. The smoke test (§8) shows it's **~317M**. The earlier
"single FSDP unit" recommendation was made under both the wrong
size estimate and a more general "the encoder might be tuned" assumption;
neither holds. Replication is the cleaner choice for the actual
317M-param, frozen-forever case.

---

## 7. Source provenance

### Vendor source

`/fsx/dongguo/adhoc/omni-t2a/poc/qwen3_asr_vendor/` — verified by
`run_zero_shot_audio.py` on 2026-04-28: zero-shot ASR transcribed 4/4
test wavs near-verbatim. This is the same vendor copy of upstream PR
#43838 (commit `0b932ecb3e09c6efa1f0a96c6621bf77be23a08d`).

The trim (modeling): keeps `Qwen3ASRPreTrainedModel`, `eager_attention_forward`,
`Qwen3ASRAttention`, `Qwen3ASREncoderLayer`, `SinusoidsPositionEmbedding`,
`Qwen3ASREncoder`. Drops `Qwen3ASRModel`, `Qwen3ASRForConditionalGeneration`,
`Qwen3ASRForForcedAlignment`, `_get_feat_extract_output_lengths`. Updates
type annotations from `Qwen3ASRConfig` → `Qwen3ASREncoderConfig`. Drops
`"Qwen3DecoderLayer"` from `_no_split_modules`.

The trim (configuration): keeps `Qwen3ASREncoderConfig`. Drops
`Qwen3ASRConfig`, `Qwen3ForcedAlignerConfig`, `CONFIG_MAPPING`/`AutoConfig`
imports.

`_compat.py` copied verbatim.

### Verification of vendor fidelity to released ckpt

Checked all `audio_tower.*` keys in `model.safetensors.index.json` against
the trimmed vendor. Every key has a matching `nn.Module` field:

| Released ckpt key | Vendor location |
|---|---|
| `conv2d1/2/3`, `conv_out` | `modeling_qwen_audio.py` L334-340 |
| `ln_post`, `proj1`, `proj2` | L332, L342, L344 |
| `layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}` | L150-153 (with `bias=bias` patch) |
| `layers.{i}.self_attn_layer_norm` | L243 |
| `layers.{i}.fc1/fc2/final_layer_norm` | L247-249 |

All 24 `k_proj.bias` tensors confirmed loadable.

Released-ckpt audio_config (from `thinker_config.audio_config` in
`/fsx/dongguo/adhoc/ckpts/Qwen3-ASR-1.7B/config.json`) matches our
`Qwen3ASRAudioModel.Config` defaults, with `output_dim=2048` (released
ckpt overrides the upstream PR's default 3584 — and we mirror that
override at the wrapper level).

### Tokenizer source

`s3://ai-lumalabs-checkpoints-ap-se-2/dongguo/ckpts/Qwen3-VL-2B-Audio` —
the merged tokenizer dir (VL-2B fast tokenizer + 36 ASR audio specials at
IDs 151669-151704). Already wired into the data pipeline by commit
`b8718746` ("Wire merged Qwen3-VL-2B-Audio tokenizer into exp001c").

Same path used here in `text_tokenizer_configs.py:DEFAULT_QWEN3_VL_2B_AUDIO_HF_PATH`
— consistency required because the embedding rows for the 36 audio
specials are baked into the LM `.pt` at exactly those IDs, so LM and
tokenizer must agree.

---

## 8. Smoke test result

Run from `projects/kuma/.venv`:

```python
from kuma.projects.omni.audio.model.tokenizers.model.qwen_audio import Qwen3ASRAudioModel
import torch

cfg = Qwen3ASRAudioModel.Config()
m = Qwen3ASRAudioModel(cfg)
# Total params: 317.48M; Trainable: 317.48M

cfg2 = Qwen3ASRAudioModel.Config(trainable_param_patterns=("ln_post", "proj1", "proj2"))
m2 = Qwen3ASRAudioModel(cfg2)
# Head-only trainable: 3.15M (0.99%)

# state_dict() keys all under audio_tower.* prefix → matches released ckpt
# audio_tower.layers.0.self_attn.k_proj.weight
# audio_tower.layers.0.self_attn.k_proj.bias
# ...

# Forward pass: 1 second of audio → 13 audio tokens at hidden=2048
out = m(torch.randn(1, 128, 100), torch.ones(1, 100, dtype=torch.long))
# out.shape == (1, 13, 2048)
```

---

## 9. Open items / next steps

1. **The single A2T head-only adapter run.** Only experiment that
   ever trains the encoder. Sets `trainable_param_patterns=("ln_post",
   "proj1", "proj2")`, freezes everything else, runs once, ckpt is
   the source of truth for all downstream consumers. Not yet
   designed/launched — needs an A2T data path and loss function.
2. **Audio-tower-only ckpt extraction script.** Sibling to
   `audio/utils/convert_qwen_vl_weights.py` — extract `audio_tower.*`
   keys from `/fsx/dongguo/adhoc/ckpts/Qwen3-ASR-1.7B/model.safetensors{,.index.json}`,
   save tower-only `.pt`, upload to `osc://`. Until this lands,
   `qwen_audio_forge_config` requires `checkpoint_path` to be supplied
   explicitly.
3. **Integration into the joint Qwen3MMDiT.** Once the unified omni
   architecture lands with `visual` + `audio_tower` submodules, the
   `drop_understanding_encoder_keys` remap_fn in `audio/configs/utils.py:43`
   becomes obsolete — `audio_tower.*` keys load directly. The
   `freeze_understanding_stream` helper added at
   `audio/configs/utils.py` (2026-04-30) already handles freezing the
   wired-up encoder — its `audio_tower.` / `visual.` patterns are
   no-ops today, auto-catch the encoders once they appear.
4. **Bind `Qwen3ASRAudioModel` to replication, not FSDP wrap, in
   `ParallelismConfig`** — opposite of what `Qwen3VLVisionModel`
   does. See §6 for rationale; the encoder is frozen forever after
   the one A2T run, and replication sidesteps the cross-rank
   collective-participation hazard for future mixed-modality batches.
5. **Promote to ursa.** Once stable, move
   `omni/audio/model/tokenizers/` → `lib/ursa/ursa/models/omni/tokenizers/`
   and update imports.
6. **Delete `_compat.py` shim.** When PR #43838 merges and a transformers
   release ships, replace local imports with
   `from transformers.models.qwen3_asr import Qwen3ASREncoder, ...`.

### 9.1 Adjacent work landed during the same window

Logged here for cross-reference (these were not part of the audio-encoder
scaffold, but interact with it):

- **2026-04-30** Added `freeze_understanding_stream` helper to
  [`audio/configs/utils.py`](../../../Projects/lumaverse/projects/kuma/kuma/projects/omni/audio/configs/utils.py) — strict superset of `freeze_text_stream` that
  also catches `audio_tower.*` and `visual.*`. exp001c switched to the
  broader helper; exp001a/b kept on `freeze_text_stream`. No-op today
  (joint model has no understanding-encoder submodules yet); auto-catches
  once integration lands.
- **2026-05-01** Audio role-token rename in
  [`koba_shared/processor/omni_audio_ops.py`](../../../Projects/lumaverse/lib/koba_shared/koba_shared/processor/omni_audio_ops.py).
  `OmniElementVAEAudio.Config` defaults now point at the merged
  tokenizer's audio specials (`<|audio_start|>`=151669,
  `<|audio_end|>`=151670, `<|audio_pad|>`=151676) instead of the legacy
  vision-token placeholders. Affects all T2A configs that don't
  explicitly override the role tokens. New constants live in
  [`koba_shared/common/omni_constants.py`](../../../Projects/lumaverse/lib/koba_shared/koba_shared/common/omni_constants.py). End-to-end smoke verified
  against the merged tokenizer in S3 — all three role tokens encode to
  exactly their expected IDs, single-token, atomic.

---

## 10. Known minor items

- **Pyright false positives.** All imports in the new files (torch,
  transformers, configo, kuma.\*, ursa.\*, koba_shared) are flagged as
  unresolved by Pyright in the IDE. These are venv-resolution issues,
  not real errors; the kuma venv resolves them all and the smoke test
  passes.
- **`@dataclass(slots=True)` deviation from MLR-04.01.** The wrapper's
  `Config` uses `@dataclass(kw_only=True)` without `slots=True`.
  `DataclassSetup` (from configo) already declares `__slots__`, and
  combining the two raises `TypeError: Config already specifies
  __slots__`. The qwen_vit wrapper has the same constraint. Inline
  comment documents the deviation.

---

## 11. Design iterations

Captures the multi-turn analysis behind decisions that aren't obvious
from looking at the final code. The FSDP/HSDP analysis is in §6.1; this
section covers everything else.

### 11.1 ID-binding fragility — single-ID assert is necessary but not sufficient (2026-04-30)

Question raised during the exp001c review: does
`assert_t2a_special_tokens_atomic` (existing assertion that each audio
role token encodes to exactly one ID) close the binding-fragility
gap between the `.pt`'s patched embedding rows (151669-151704) and the
tokenizer's IDs?

**Answer: it catches BPE fragmentation but not ID drift.** Two regimes:

| Regime | Single-ID assert sufficient? | Why |
|---|---|---|
| Data pipeline emits vision tokens (the legacy state — see §11.2) | ✓ Yes | Binding to 151669-151704 isn't exercised; the dormant ASR-patched rows can drift in future tokenizer rebuilds without affecting training. |
| Data pipeline emits actual `<\|audio_start\|>` etc. (new state, post-rename) | ✗ No on its own | Single-ID assert validates `<\|audio_start\|>` encodes to *one* ID. If a tokenizer rebuild shifted it from 151669 → 151700, encode still returns `[151700]` (length 1) — assert passes silently. The model's row 151669 (ASR-patched) is then never read; row 151700 contains untouched VL-2B init for an unused-special-token slot, so the audio-special embedding bootstrap is lost. Training proceeds, just slower / worse. |

**End-to-end smoke (post-rename) verified the new state is correct:**
the merged tokenizer at the canonical S3 path encodes
`<|audio_start|>` → 151669, `<|audio_end|>` → 151670, `<|audio_pad|>` →
151676 — exactly the rows the `.pt` patches. So the binding holds today.

**To future-proof against tokenizer rebuilds:** when the data pipeline
graduates to actual audio specials (effectively, today after
[§11.3](#113-audio-role-token-rename-2026-05-01)), extend
`assert_t2a_special_tokens_atomic` with an optional
`expected_ids: dict[str, int]` parameter and pin to the canonical
mapping from `MERGE_NOTES.md`. Filed as a future-work item; not done
in this iteration because the rename + smoke test was the immediate
win and a value-check assertion is a small follow-up.

### 11.2 Surprise: legacy vision-token placeholders (2026-04-30 → 2026-05-01)

While answering §11.1, found that the data pipeline was using **vision
tokens** as audio role tokens — not the audio-specific specials at
151669-151704. Specifically, `OmniElementVAEAudio.Config` defaults pre-rename:

```python
audio_start_token:    str = VISION_START_TOKEN     # "<|vision_start|>" → 151652
audio_end_token:      str = VISION_END_TOKEN       # "<|vision_end|>"   → 151653
audio_register_token: str = QWEN3_PAD_TOKEN        # "<|endoftext|>"   → 151643
audio_pad_token:      str = QWEN3_IMAGE_PAD_TOKEN  # "<|image_pad|>"   → 151655
```

So the 36 ASR-patched embedding rows (151669-151704) in the merged
ckpt were *not exercised* by current T2A training — the pipeline
emitted vision IDs, which read from VL-2B's untouched (and
trained-as-vision) embedding rows.

**Origin:** before the audio specials existed in the tokenizer, the
T2A pipeline reused vision specials as placeholder boundary markers.
The vision tokens are single-token specials that happened to be
present in the base Qwen3 tokenizer, so they were a quick way to
unblock T2A development. The `.pt` embedding patches at IDs
151669-151704 were forward-looking — bootstrap for when the pipeline
graduated to real audio specials. That graduation hadn't happened
until §11.3.

### 11.3 Audio role-token rename (2026-05-01)

System-wide replacement of vision-token placeholders with the
audio-specific specials added by the merged tokenizer. Touched two
files:

- [`koba_shared/common/omni_constants.py`](../../../Projects/lumaverse/lib/koba_shared/koba_shared/common/omni_constants.py) — added
  `AUDIO_START_TOKEN`, `AUDIO_END_TOKEN`, `QWEN3_AUDIO_PAD_TOKEN` and
  their IDs (151669, 151670, 151676).
- [`koba_shared/processor/omni_audio_ops.py`](../../../Projects/lumaverse/lib/koba_shared/koba_shared/processor/omni_audio_ops.py) —
  swapped `OmniElementVAEAudio.Config` defaults to the new constants;
  `audio_register_token` kept on `QWEN3_PAD_TOKEN` (no dedicated audio
  register slot exists, and registers are off by default with
  `audio_register_token_amount=0`).

**Why a single source-of-truth change is enough.** Surveyed all
downstream consumers before editing — every call site
(`default_t2a_pipeline_processors`, `omni_t2a_packing_koba_v2`,
`tokenizer_validation`, `test_t2a_pipeline_equivalence`) instantiates
`OmniElementVAEAudio.Config(compression_factor=…)` without overriding
the role tokens. So the default-value change propagates to all of them
automatically. No call-site edits needed.

**Per-experiment effect:**

- **exp001c** (Qwen3-VL-2B-Audio merged ckpt): strict win. Audio role
  tokens now point at the embedding rows that were specifically
  ASR-bootstrapped for them. The 36 ASR-patched rows are no longer
  dormant — they're exercised on every T2A step.
- **exp001a/b** (Qwen3-0.6B base, frozen text embedding): role tokens
  shift to 151669/151670/151676, where 0.6B Qwen3 has untrained-empty
  slot init (those IDs are reserved-but-unused in the base
  tokenizer). The audio role markers still work as identifiable
  boundary tokens (frozen but distinct vectors), just without the
  vision-token semantic carryover. If you ever need to re-run exp001a/b
  faithful to historical W&B results, override
  `audio_start_token=VISION_START_TOKEN` etc. explicitly.

**Smoke verification** (kuma venv against the merged tokenizer in S3):

- `<|audio_start|>` → 151669 ✓
- `<|audio_end|>` → 151670 ✓
- `<|audio_pad|>` → 151676 ✓
- `assert_t2a_special_tokens_atomic` passes (single-token, atomic).
- All four audio role fields on `OmniElementVAEAudio.Config` resolve
  correctly.
- `default_t2a_pipeline_processors` imports cleanly.
- `koba.processor.omni_audio_ops` re-exports correctly through to
  `koba_shared`.

---

## Session checkpoint — 2026-05-01

End-of-session state on `dongguo/omni-t2a-v2`. Captured so the next
session can resume cleanly without re-deriving where things stopped.

### Commits pushed this session

Two commits landed on `dongguo/omni-t2a-v2` and were pushed to origin
(both feed PR #8064):

1. **`92a593b0a8` `[omni, t2a] Finalize exp001c: audio tokens, understanding-stream freeze, 32k pack`**
   - `lib/koba_shared/koba_shared/common/omni_constants.py` — added
     `AUDIO_START_TOKEN` / `AUDIO_END_TOKEN` / `QWEN3_AUDIO_PAD_TOKEN`
     constants (IDs 151669/151670/151676).
   - `lib/koba_shared/koba_shared/processor/omni_audio_ops.py` — flipped
     `OmniElementVAEAudio.Config` boundary/pad defaults from vision-token
     placeholders to the audio-specific tokens. Completes the POC
     tokenizer swap previously gated on the merged tokenizer landing.
   - `projects/kuma/kuma/projects/omni/audio/configs/utils.py` — added
     `freeze_understanding_stream` (strict superset of
     `freeze_text_stream`); refactored both via shared
     `_set_frozen_param_substrings`. The `audio_tower.*` / `visual.*`
     patterns are no-ops today (the joint Qwen3MMDiT lacks those
     submodules) but auto-catch once the unified
     understanding+generation architecture lands.
   - `projects/kuma/kuma/projects/omni/audio/configs/tasks/t2a.py` —
     bumped `exp001c_2b_mmaudio_softcap` to 32k packed-token budget,
     swapped freeze policy to `freeze_understanding_stream`, added
     `debug_local_exp001c` debug config (depth-2, FSDP-only, no
     compile/AC).
   - `projects/kuma/kuma/projects/omni/audio/studies/t2a.py` — added
     `study001c` study entry (2 nodes / 16 GPUs on kiwi, dp_shard=8 ×
     dp_replicate=2).

2. **`b12dc4f351` `[omni, audio] Add project-local audio encoder + tokenizer scaffold`**
   - `projects/kuma/kuma/projects/omni/audio/model/tokenizers/` — entire
     subtree (~1107 LOC across 11 files): `Qwen3ASRAudioModel` wrapper,
     audio + text-tokenizer configs, vendored HF
     `modeling_qwen_audio.py` / `configuration_qwen_audio.py` plus
     `_compat.py` shim.
   - This is the scaffold §1 of this doc described as "added today" but
     was sitting untracked for ~24h before this commit landed it.

### Pre-commit findings worth knowing for review

The hooks ran on both commits. Auto-fixed and non-blocking findings:

- **Auto-fixed by ruff** on commit 1 (no manual intervention): a few
  formatting nits across the 5 modified files. Re-staged and committed
  cleanly on second attempt.
- **Auto-fixed by ruff** on commit 2 (no manual intervention): 3 errors
  (1107 LOC of vendored HF code triggered some automatic rewrites in
  the third-party files). Re-staged on second attempt.
- **MLR-04.01 blocking violation** on commit 2: `qwen_audio.py:37` had
  `@dataclass(kw_only=True)` without `slots=True`. The author had
  already documented why `slots=True` is omitted (`DataclassSetup`
  provides `__slots__`; combining the two raises
  `TypeError: Config already specifies __slots__`; matches qwen_vit
  wrapper precedent). Resolved by adding
  `# noqa: MLR-04.01 - DataclassSetup provides __slots__ (see comment above)`
  to the decorator line. The justification comment was already
  immediately above; the noqa just suppresses the heuristic check that
  doesn't read prose.
- **Non-blocking ML Rules warnings** (commit 2):
  `MLR-04.07 self.config in nn.Module`,
  `MLR-04.08 DataclassSetup deprecated`,
  `MLR-04.13 Config constructor with 5+ keyword args`,
  `MLR-04.14 Mixed config systems in same file`,
  `MLR-08.04 missing Raises: in docstring`. All pre-existing in the
  scaffold as written; punted to a follow-up cleanup pass.
- **Non-blocking ML Rules warnings** (commit 1):
  `MLR-04.08 DataclassSetup deprecated`,
  `MLR-04.14 Mixed config systems`. Same pre-existing kind. Not
  blocking; left as-is.

### What stayed uncommitted (and why)

These files were in the working directory at session end and are
**deliberately not in source control**:

- `.claude/settings.json` — local Claude Code permissions config; not
  source code. Stays in working tree across branch switches; harmless.
- `projects/kuma/kuma/projects/omni/audio/inference/_dev_notes_compile.md`
  — file's own header explicitly states "Local scratchpad —
  intentionally not tracked." Records `torch.compile` smoke-test
  findings on T2A inference (mode="default" passes, mode="reduce-overhead"
  hits a known sampler-side CUDA Graphs issue).
- `projects/kuma/kuma/projects/omni/audio/inference/{configs,processor}/`
  — pycache-only directories from earlier inference development; no
  source files inside.
- `projects/kuma/noise_distributions.png` — debug-time plot artifact.
- `projects/kuma/wandb-metadata.json` — wandb run output.

If a future session wants to commit any of these, audit content first
and decide explicitly — they're not currently in any `.gitignore` and
will keep showing as untracked across branch switches.

### Branch context for resuming

- **Current branch at session end**: `dongguo/omni-a2a-plumbing` (new),
  based on `origin/main` at `b0592047d1`. Created for the parallel A2A
  workstream — see `audio_extension_a2a_logs.md` §8 (revised phase
  plan) for the v1 PR scope. **Tracks `origin/main`** until first push;
  use `git push -u origin dongguo/omni-a2a-plumbing` to create the
  proper remote tracking branch.
- **`dongguo/omni-t2a-v2` at session end**: contains both new commits;
  PR #8064 picked them up. To resume T2A work on this branch:
  `git checkout dongguo/omni-t2a-v2` (no pull needed; up to date with
  origin at session end).

### How to resume T2A work

If the next session is continuing the T2A v2 path on the same branch:

1. `git checkout dongguo/omni-t2a-v2` and confirm clean tree
   (`git status`). The five "uncommitted" files listed above will
   reappear (they live in working tree, not branch state) — expected.
2. `git pull --ff-only` to pick up any reviewer-merged commits on the
   PR side or `origin/main` merges.
3. Check PR #8064's review state (`gh pr view 8064`) before adding
   more commits — if the PR is close to merging, additional commits
   bloat the diff late. New T2A iterations may want their own follow-up
   branch off main after #8064 lands.
4. Outstanding T2A v2 follow-ups (none of these block PR #8064):
   - Address the non-blocking ML Rules warnings on the audio
     scaffold (`MLR-04.07`, `04.13`, `04.14`, `08.04`) when the API
     stabilizes enough to justify a refactor.
   - Decide whether to graduate the scaffold from
     `projects/kuma/.../model/tokenizers/` to
     `lib/ursa/ursa/models/omni/tokenizers/`. Per the scaffold's own
     `__init__.py` comment: "Graduate to `ursa.models.omni.tokenizers`
     once stable."

### How to resume A2A work

If the next session is continuing A2A development:

1. `git checkout dongguo/omni-a2a-plumbing`. Empty branch (no commits
   beyond `origin/main`); ready for PR-1a's plumbing work.
2. PR-1a scope is in
   [`audio_extension_a2a_logs.md`](audio_extension_a2a_logs.md) §8
   ("Phase 1a") and §10 (per-file change list). Net-new code:
   `MultiAudioDecoder`, `RefAudioTruncator`, `handle_a2a` method on
   `OmniAudioSeqBuilder`, `default_a2a.py`, `OmniA2APackingConfigKobaV2`,
   `DummyA2ADatasetConfig`, `debug_local_a2a_dummy` config, plus tests
   (determinism + smoke). Roughly 800 LOC + tests.
3. PR-1a is independent of PR #8064 (no overlap with T2A v2 changes
   currently in flight). Can ship in parallel.
4. **Suggested first sub-step**: write `MultiAudioDecoder` +
   `RefAudioTruncator` (both koba processors), then the `handle_a2a`
   seq-builder method, then the kuma-side wiring. The koba processors
   are the foundation everything else consumes.
