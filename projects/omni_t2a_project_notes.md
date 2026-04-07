# Omni T2A Project Notes

**Started:** 2026-04-07  
**Author:** Dong Guo  
**Branch:** development branch (not merged to main)

---

## Overview

Added a **text-to-audio generation stream** to the existing omni MoT (Mixture of Transformers) model. The omni model currently has two streams (text understanding + vision generation). We add audio as a new generation stream using the same 2-stream MoT infrastructure.

The model is trained to generate audio from text prompts. Two encoder backends are supported:

- **MMAudio** (16kHz, 20-dim latents) — matches reference run `tage001-internal-audio-v2-k3600`
- **Hunyuan DAC VAE** (48kHz, 128-dim latents) — higher fidelity option

---

## Architecture

### Key Design Decisions

**Reuse 2-stream MoT** — Audio replaces vision as stream 1 (generation stream). No changes to `blocks.py` needed:

- Stream 0 (understanding): Text tokens, causal attention
- Stream 1 (generation): Audio VAE latents, noise attention

**Asymmetric attention** — achieved naturally by the attention mask:

- Text tokens: `causal` → can only see previous text, cannot see audio
- Audio tokens: `noise` → can see all text + all audio tokens

This gives audio cross-modal conditioning on text, without text attending back to audio — matching the Ray3 T2A cross-attention design but implemented via packed self-attention.

**Audio RoPE** — 1D temporal `Qwen3RotaryEmbedding` (same as text), not M-RoPE (which is vision-only).

**Audio modulation** — same "double" adaLN scheme as vision stream (6-param shift/scale from diffusion timestep).

### New Model Components

File: `lib/ursa/ursa/models/omni/model/model.py`

- `AudioPreprocess` — embeds audio latents + 1D temporal RoPE + timestep modulation
- `Qwen3TextAudioPackedPreprocess` — simplified preprocess (no ViT), maps `vae_token_mask` to audio generation stream

### Model Sizes

| Config | Hidden | Layers | Heads | Audio Latent | VAE |
|--------|--------|--------|-------|--------------|-----|
| `qwen3_0_6B_t2a_mmaudio()` | 1024 | 28 | 16 | 20-dim | MMAudio 16k |
| `qwen3_0_6B_t2a_dacvae()` | 1024 | 28 | 16 | 128-dim | Hunyuan DAC 48k |

---

## Training

### Loss: `BagelT2ALoss`

```
total_loss = audio_diffusion_loss x 1.0 + text_ce_loss x 0.25
```

- Audio: rectified flow diffusion MSE on noisy latents, sigma_shift=3.0
- Text: cross-entropy on text tokens (next-token prediction)
- Noise schedule: `SampleLogSNRGeneric(shift=-1.6)`, truncated normal (matches reference run)

### Trainer: `OmniT2ATrainer`

Step:

1. Encode raw audio through **frozen** encoder -> `z0 (B, C, T)`
2. Noise z0 with rectified flow
3. Forward through MoT denoiser -> `(text_logits, audio_prediction)`
4. Compute combined loss -> backward -> optimizer step

---

## Data Pipeline

### Audio Token Count

**Critical**: `OmniElementAudio.compression_factor` must match the encoder's `hop_length`:

| Encoder | Sample Rate | hop_length | Tokens for 5s |
|---------|------------|------------|---------------|
| MMAudio 16k | 16kHz | **512** | 156 |
| Hunyuan DAC | 48kHz | **960** | 250 |

### Pipeline (per sample)

```
Lance dataset
  -> AudioDecoder (decode + resample to target SR)
  -> OmniT2AAudio (sequence plan: [TEXT(transcript), AUDIO(waveform)])
  -> OmniElementAudio (vae_token_mask=True, num_tokens=ceil(frames/hop))
  -> OmniQwen3Tokenizer (text tokens; audio uses image_pad placeholder)
  -> OmniPositionIDMRoPE (order="THW"; audio position IDs 1D temporal)
  -> pack_sequence() -> batch dict
```

### Datasets

| Dataset | Rows | SR | Notes |
|---------|------|----|-------|
| `emilia` | -- | any | Multilingual, used for debug |
| `internal-audio-v2` | ~142M (filtered) | 16kHz | Primary training set; filter: `round1_pass_all_filter = 1` |

---

## Files Created / Modified

All code lives in `lumaverse` repo (separate development branch, not merged to main).

### New files (kuma project)

| File | Purpose |
|------|---------|
| `projects/kuma/.../omni/bagel/configs/t2a.py` | Model configs + Job configs (MMAudio + DAC variants) |
| `projects/kuma/.../omni/bagel/configs/data/omni_t2a_dataset.py` | `OmniT2ADatasetConfig` + `OmniT2APackingDataset` |
| `projects/kuma/.../omni/bagel/losses/bagel_t2a.py` | `BagelT2ALoss` (rank-1 audio diffusion + text CE) |
| `projects/kuma/.../omni/bagel/trainers/omni_t2a.py` | `OmniT2ATrainer` |
| `projects/kuma/.../omni/bagel/docs/omni_t2a_architecture.md` | Architecture diagrams |
| `projects/kuma/.../omni/bagel/docs/omni_t2a_survey.md` | Codebase survey + reference run details |

### Modified files (shared libs)

| File | Change |
|------|--------|
| `lib/ursa/.../omni/model/model.py` | Added `AudioPreprocess`, `Qwen3TextAudioPackedPreprocess` |
| `lib/ursa/.../omni/inference/sequence_packing.py` | Propagate `x_audio` + `audio_token_mask` through packing |
| `lib/koba/.../processor/omni_audio_ops.py` | Fix `vae_token_mask=True`, `shape[-1]` for frames, `image_pad` placeholder, `compression_factor` param |
| `lib/koba/.../pipelines/pipelines.py` | Add `audio_compression_factor` to `T2APipelineParams` |
| `lib/koba/.../pipelines/default_t2a.py` | Wire `audio_compression_factor`; fix `mrope_order="THW"` |
| `lib/koba_shared/.../processor/position_ids_dev.py` | `.long()` cast for audio position IDs |
| `projects/kuma/.../audio/t2a_dataset_config.py` | Register `internal-audio-v2` dataset |

---

## Running

### Smoke Tests (Single GPU)

```bash
cd /fsx/dongguo/Projects/lumaverse/projects/kuma
source .venv/bin/activate

# MMAudio (16kHz, 20-dim)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 \
    main.py --config kuma.projects.omni.bagel.configs.t2a.debug_local \
    --name debug_omni_t2a

# DAC VAE (48kHz, 128-dim)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 \
    main.py --config kuma.projects.omni.bagel.configs.t2a.debug_local_dacvae \
    --name debug_omni_t2a_dacvae
```

Monitor: `tail -f /tmp/omni_t2a_smoke_test.log`

### Full Training (8 GPU, single node)

```bash
torchrun --standalone --nproc_per_node 8 \
    main.py --config kuma.projects.omni.bagel.configs.t2a.exp_0_6b_mmaudio \
    --name omni_t2a_0_6b_mmaudio
```

---

## Reference Run

W&B: `luma-ai/t2a/runs/ovah4wwn` (inkyu, `tage001-internal-audio-v2-k3600`)

Key hyperparameters adopted:

- Optimizer: AdamW lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95)
- LR warmup: 5,000 steps
- Noise shift: -1.6, truncated normal logSNR
- Dataset: internal-audio-v2, duration buckets 5s/10s
- Encoder: MMAudio 16k, scaling_factor=1/2.3563

---

## Key Bugs Found During Smoke Test

| # | Error | Root Cause | Fix |
|---|-------|------------|-----|
| 1 | `RecursiveHSDP assert count > 0` | FSDP not valid on 1 GPU | `data_parallelism=None` for debug |
| 2 | `MRoPE order "T" not supported` | MRoPE only supports 3D orders | Changed to `"THW"` |
| 3 | S3 `ExpiredToken` | AWS credentials expired | Refresh `~/.bashrc` |
| 4 | `num_tokens mismatch 1 != 6` | audio_pad token not in Qwen3 vocab | Use `image_pad` (single token, ID 151655) |
| 5 | `Long/Float dtype in position_ids` | MRoPE produces floats, tensor is Long | `.long()` cast in `position_ids_dev.py` |
| 6 | `list has no attribute num_tokens` | `tokenized_sequence_plan` is a list | Use raw list directly in packing loop |
| 7 | `pack_sequence() unexpected kwarg` | Wrong function signature | Remove `use_flex_attention` arg |
| 8 | MMAudio `NotImplementedError: 4D padding` | Passed `(B, 1, L)`, expects `(B, L)` | Squeeze before encoding |
| 9 | `flex_attn_mask required` | `use_flex_attention=False` in model config | Set `use_flex_attention=True` |
| 10 | Shape mismatch `x=59, gate=16165` | `OmniElementAudio` read `shape[0]` (channel=1) not `shape[-1]` (frames) | `num_frames = audio_tensor.shape[-1]` |
| 11 | Shape mismatch after fix 10 | `compression_factor=960` (DAC) but MMAudio uses `hop_length=512` | `compression_factor=512` for MMAudio |

---

## Next Steps

- [ ] Multi-node Flyte launch config (study file)
- [ ] Audio generation inference processor
- [ ] Evaluation metrics (FAD, CLAP score) against reference run
- [ ] Extend to 3-stream model (text + vision + audio) when ready to merge
