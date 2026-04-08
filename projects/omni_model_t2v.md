# Uni-1 (Omni) T2V Training Project

> **What this note is for:** Extending the uni-1 (omni) model from T2I to T2V, reusing Ray3's data infrastructure and 3D VAE. Tracks key design decisions, code changes, data pipeline, and training configs.
>
> | Area | Status | Latest |
> |---|---|---|
> | Code changes (9 modified + 7 new files) | ✅ Complete | Branch `dongguo/omni-t2v` |
> | Pipeline validation (synthetic) | ✅ Passing | `t2v_smoke_test.py --synthetic` |
> | 0.6B GPU smoke test | 🔲 Not yet run | Config: `exp_t2v_360p_smoke_0_6b` |
> | 32B real training | 🔲 Blocked | Needs T2I stage 5 checkpoint path |
> | Sigma shift tuning | 🔲 TODO | Current override=7.0 is estimate |

---

**Branch:** `dongguo/omni-t2v` on `lumalabs/lumaverse`
**Date:** 2026-04-06
**Status:** Code complete, ready for GPU smoke test

---

## Goal

Extend the uni-1 (omni) model from text-to-image (T2I) to text-to-video (T2V) training, reusing Ray3's data infrastructure and 3D VAE.

## Key Decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | VAE | Ray3 3D VAE, 16ch with patching_factor=2 (c_in=64) |
| 2 | VAE encoding | Online (during training) |
| 3 | Packing | Full multi-modal: text + ViT + VAE in one sequence |
| 4 | Resolution | Single 360p (actually 352p for even latent dims), quantized aspect ratio buckets |
| 5 | Video params | 5s clips, 24fps (Ray3 setting) |
| 6 | Token budget | 128K tokens per packed sequence |
| 7 | Initialization | From T2I stage 5 checkpoint (wan/Ray3 VAE, c_in=64) |
| 8 | Training mode | T2V only (first experiment), I2V later |
| 9 | Freezing | Freeze text embeddings, ViT encoder, VAE encoder/decoder |
| 10 | Diffusion | Uni-1 rectified flow with mode-based logSNR sampling |
| 11 | Sigma shift | Override=7.0 for first experiment |
| 12 | Dataset | Ray3 baked stage1_v2_5s.lance (386K rows, 5s clips) |

## Critical Insight: patching_factor=2

Ray3's 3D VAE outputs 16ch latents. With `patching_factor=2` (PixelUnshuffle), this becomes **64ch at half spatial resolution**:
- Without patching: 5s 360p 16:9 video = 32x45x80 = **115K tokens** (too many)
- With patching: 30x22x40 = **26,400 tokens** (fits in 128K budget)
- **c_in=64 matches the existing T2I stage 5 checkpoint** -- projection layers transfer directly

The `make_wan_compatible()` function at `projects/kuma/.../configs/experiment_utils.py:673` already does this.

## Resolution Note

"360p" in Ray3 means **352p** (352x640 for 16:9) because 360/8=45 is odd and PixelUnshuffle(2) requires even spatial dims. LATENT_DIMS registry: `"360p": {30: [(22, 40), (26, 35), (30, 30), (35, 26), (40, 22)]}`.

## Architecture: Why It Works

Uni-1 is **already partially video-aware**:
- **M-RoPE position encoding**: 3D `(temporal, height, width)` -- already supports T>1
- **FinalVidLayerPacked**: processes `(T*H*W)` tokens
- **Loss computation**: `z0` reshaped as `(b, c, t, h, w)` then flattened to `(t*h*w)` tokens
- **logSNR sampling**: has mode-based sampler for T2V/I2V
- **Sigma shifting**: token-count-dependent shifting already maps higher counts to stronger shifts

---

## Code Changes (Branch: dongguo/omni-t2v)

### Modified Files (9) -- backward-compatible

| File | Change |
|------|--------|
| `lib/koba_shared/koba_shared/common/types.py` | `NOISY_VAE_VIDEO = 7` enum value |
| `lib/koba_shared/koba_shared/processor/tokenized_types.py` | `vae_latent_shapes` accepts 2-tuples and 3-tuples |
| `lib/koba_shared/koba_shared/processor/position_ids_dev.py` | Handle `NOISY_VAE_VIDEO` with 3D `(pt, ph, pw)` positions |
| `lib/ursa/ursa/models/omni/inference/sequence_packing.py` | Treat video same as noisy image in position wiring |
| `lib/ursa/ursa/models/omni/model/model.py` | `patched_f` as `list[int]`, 3-tuple vae_latent_shapes unpacking |
| `projects/kuma/.../bagel/typing.py` | Type annotation: `list[tuple[int, ...]]` |
| `projects/kuma/.../bagel/encoder_offload/encoder_actors.py` | Skip `[:, :, None]` temporal unsqueeze for 5D video |
| `projects/kuma/.../bagel/trainer.py` | Handle 4D video tensors in batched/unbatched VAE encoding |
| `projects/kuma/.../bagel/losses/bagel.py` | T2V sigma shift mapping + `"t2v"` modality routing |

### New Files (7)

| File | Purpose |
|------|---------|
| `lib/koba_shared/koba_shared/processor/omni_vae_video.py` | `OmniElementVAEVideo` -- computes 3D latent shapes, token masks for video |
| `lib/koba/koba/processor/omni_vae_video.py` | Re-export shim |
| `lib/koba/koba/processor/omni_t2v_ops.py` | `VideoDecodeFromLance`, `CaptionSamplerFromLance`, `OmniPackedVideoBuilder` |
| `lib/koba/koba/pipelines/default_t2v.py` | `T2VPipelineParams` + processor chain definition |
| `projects/kuma/.../bagel/datasets/t2v_dataset_config.py` | Dataset config wiring `stage1_v2_5s.lance` into omni pipeline |
| `projects/kuma/.../bagel/configs/scaling_t2v.py` | `exp_t2v_360p_first()` (32B) + `exp_t2v_360p_smoke_0_6b()` (0.6B) |
| `projects/kuma/.../bagel/configs/t2v_smoke_test.py` | Standalone pipeline validation script |

---

## Data Pipeline Flow

```
Lance row (stage1_v2_5s.lance)
  |-- video_path, captions, caption_weights
  v
VideoDecodeFromLance  (wraps Ray3's CustomVideoDecodeProcessor)
  |-- Decodes video from S3, resizes to 352p, extracts 5s clip
  v
CaptionSamplerFromLance  (weighted caption selection)
  v
OmniPackedVideoBuilder  (creates sequence_plan)
  |-- [TEXT element, NOISY_VAE_VIDEO element]
  v
OmniAddTokenizedSequenceElement  (creates TokenizedSequenceElement objects)
  v
OmniElementVAEVideo  (computes vae_latent_shapes=(T,H,W), token masks)
  v
OmniElementText + OmniQwen3Tokenizer  (tokenize caption)
  v
OmniPositionIDMRoPE  (3D position IDs for video: temporal, height, width)
  v
pack_sequence()  (packs multiple samples into 128K-token batch)
  v
OmniTrainer.step()
  |-- Online VAE encoding (Ray3 3D VAE with patching)
  |-- BagelTransfusionLoss (rectified flow, mode-based logSNR, sigma_shift=7.0)
```

## Training Configs

### 0.6B Smoke Test (recommended first)
- **Config:** `exp_t2v_360p_smoke_0_6b()`
- **Hardware:** 1 node, 8 GPUs (any A100/H100)
- **Token budget:** 32K
- **Init:** Random (no checkpoint needed)
- **Purpose:** Validate full pipeline end-to-end

### 32B Real Training
- **Config:** `exp_t2v_360p_first()`
- **Hardware:** 1 node, 8x H100 80GB + Ulysses SP + activation checkpointing
- **Token budget:** 128K
- **Init:** T2I stage 5 checkpoint (TODO: fill in path)
- **Purpose:** First T2V training run

## How to Run

```bash
# 1. Pipeline-only validation (no GPU needed)
python -m kuma.projects.omni.bagel.configs.t2v_smoke_test --synthetic

# 2. 0.6B smoke test on GPU cluster
#    Register exp_t2v_360p_smoke_0_6b from:
#    kuma.projects.omni.bagel.configs.scaling_t2v

# 3. 32B training (after filling checkpoint path)
#    Register exp_t2v_360p_first from:
#    kuma.projects.omni.bagel.configs.scaling_t2v
```

## Remaining TODOs

1. **T2I stage 5 checkpoint path** -- needed for 32B `exp_t2v_360p_first`, not for 0.6B smoke test
2. **Sigma shift tuning** -- current override=7.0 is an estimate, needs calibration
3. **I2V extension** -- future work, reuses clean VAE image conditioning

## Key Files Reference

| What | Where |
|------|-------|
| Training configs | `projects/kuma/kuma/projects/omni/bagel/configs/scaling_t2v.py` |
| Dataset config | `projects/kuma/kuma/projects/omni/bagel/datasets/t2v_dataset_config.py` |
| Video VAE processor | `lib/koba_shared/koba_shared/processor/omni_vae_video.py` |
| T2V ops (decode, caption, builder) | `lib/koba/koba/processor/omni_t2v_ops.py` |
| Loss (sigma shift) | `projects/kuma/kuma/projects/omni/bagel/losses/bagel.py` |
| Sequence packing | `lib/ursa/ursa/models/omni/inference/sequence_packing.py` |
| Model preprocess | `lib/ursa/ursa/models/omni/model/model.py` |
| Smoke test script | `projects/kuma/kuma/projects/omni/bagel/configs/t2v_smoke_test.py` |
| Ray3 LATENT_DIMS | `projects/kuma/kuma/projects/ray3/registries/training.py:65` |
| make_wan_compatible | `projects/kuma/kuma/projects/omni/bagel/configs/experiment_utils.py:673` |
| Lance dataset | `s3://ai-lumalabs-datasets-ap-se-2/lance_datasets/video/ray3_sft_baked/stage1_v2_5s.lance` |
