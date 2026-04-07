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

## Sequence Parallelism: Ulysses, Ring Attention, and Flash Attention

### Problem Context

Training on very long sequences (long audio or video) hits two bottlenecks:

- **Memory**: Standard attention materializes a full (S, S) attention matrix in HBM → O(S²) memory
- **Compute**: Attention computation is O(S²) — doesn't fit or runs too slowly on one GPU

### Flash Attention (solves the memory bottleneck)

Flash attention tiles the (S, S) computation in SRAM rather than materializing it in HBM. HBM usage becomes O(S), not O(S²). Compute is still O(S²) but memory is not. This alone is sufficient for moderately long sequences.

### DeepSpeed Ulysses (distributes compute across GPUs, head dimension)

Named after the famously long James Joyce novel. Each GPU starts with a sequence shard, and redistributes for attention:

```
Before all-to-all:  each GPU holds (S/N, H,   D)  ← shard by sequence, all heads
       all-to-all
After all-to-all:   each GPU holds (S,   H/N, D)  ← full sequence, shard by head
       run full attention for H/N heads (independent across heads)
       all-to-all back
After all-to-all:   each GPU holds (S/N, H,   D)  ← back to sequence shard
```

Key insight: heads are fully independent — each GPU computes exact attention for its H/N heads over the full sequence. No approximation.

Limitations:
- Per-GPU compute is still O(S²) — doesn't scale with more GPUs
- Requires H ≥ N (can't shard more than H heads)
- Communication: 2× all-to-all (fast, single collective)

### Ring Attention (reduces per-GPU compute)

Each GPU holds Q for S/N tokens. K/V chunks are passed around a ring of N GPUs, and each GPU accumulates its partial attention result using flash attention on each chunk:

```
GPU 0: Q(tokens 0..S/N) × K/V(chunk 0) → partial result
                        × K/V(chunk 1)  ← received from GPU 1
                        × K/V(chunk 2)  ← received from GPU 2
                        ...accumulate → final output for tokens 0..S/N
```

Per-GPU compute = O((S/N) × S) = O(S²/N) — scales linearly with more GPUs.

Tradeoff vs Ulysses:
- Better compute scaling (O(S²/N) vs O(S²))
- Higher communication overhead: N-1 ring hops vs 2 all-to-alls
- No constraint on H

### Combining Ulysses + Ring (2D Parallelism)

Split GPUs into an M×N grid:

```
Total GPUs = M × N

         Ring →  (N=4 sequence chunks)
Ulysses  GPU(0,0)  GPU(0,1)  GPU(0,2)  GPU(0,3)   ← heads 0..H/M
  ↓      GPU(1,0)  GPU(1,1)  GPU(1,2)  GPU(1,3)   ← heads H/M..2H/M
(M=2)
```

Each GPU holds H/M heads and S/N sequence tokens. Ulysses all-to-all runs within each column; ring passes K/V chunks within each row. Per-GPU compute = O(S²/N), head constraint = M ≤ H.

### Relation to FSDP

FSDP and Ulysses are orthogonal and composable:

| | FSDP | Ulysses |
|---|---|---|
| **What it shards** | Model weights + gradients + optimizer states | Activations (Q/K/V) along head/sequence dim |
| **Problem solved** | Model too large to fit on one GPU | Sequence too long (activation memory + compute) |
| **Communication** | all-gather weights before forward, reduce-scatter gradients after | all-to-all Q/K/V before attention, all-to-all after |

They run simultaneously using separate `DeviceMesh` dimensions in PyTorch.

### In This Codebase

`ulysses_enabled` and `ulysses_mesh` in `Qwen3TextAudioPackedPreprocess` and `Qwen3MMDiT` control whether Ulysses is active. When enabled:

- `_prepare_sequence_length_metadata` pads sequence lengths to be divisible by mesh size
- `maybe_in_shard` / `maybe_out_shard` are the all-to-all scatter/gather ops wrapping attention blocks
- `maybe_in_shard_modulation_mask` / `maybe_out_shard_modulation_mask` handle sharding the adaLN modulation mask accordingly

---

## OmniPreprocessData: Masks and Indices Explained

### The Packed Sequence Layout

All masks operate over a single flat sequence of length `S = S_U + S_V`, where multiple training samples are concatenated:

```
packed sequence: [sample0_text | sample0_audio | sample1_text | sample1_audio | ...]
                  ← und tokens → ← gen tokens →  ← und tokens → ← gen tokens →
```

### Summary Table

| Field | Shape | Purpose |
|---|---|---|
| `packed_und_token_masks` | `Bool[S]` | Which positions in packed seq are text tokens |
| `packed_gen_token_masks` | `Bool[S]` | Which positions in packed seq are audio tokens |
| `packed_und_token_indexes` | `Int[N_U]` | Integer positions of text tokens (for gather/scatter) |
| `packed_gen_token_indexes` | `Int[N_V]` | Integer positions of audio tokens (for gather/scatter) |
| `scatter_indices` | `Int[S]` | Reorder concatenated [und\|gen] → interleaved packed order |
| `flex_attn_mask` | BlockMask | Asymmetric attention pattern for FlexAttention kernel |
| `flash4_attn_mask` | FA4SparseMask | Same but for Flash4 kernel |
| `modulation_mask` | `Bool[B, S_V]` | Per-token timestep assignment for adaLN modulation |

### Field Details

**`packed_und_token_masks` / `packed_gen_token_masks`** — `Bool[S]`

Boolean position masks for each stream in the flat packed sequence. Used to slice stream-specific tensors, e.g. extracting text-only position IDs:

```python
text_position_ids = position_ids[text_token_mask]
```

**`packed_und_token_indexes` / `packed_gen_token_indexes`** — `Int[N_U]` / `Int[N_V]`

Integer positions (nonzero indices) of each stream's tokens. Used in `blocks.py` to scatter the joint attention output back into per-stream tensors:

```python
attns_out_und = attns[:, packed_und_token_indexes]
attns_out_gen = attns[:, packed_gen_token_indexes]
```

**`scatter_indices`** — `Int[S]`

The most subtle field. Transformer blocks compute Q/K/V per stream separately, producing a concatenated `[und_tokens | gen_tokens]` tensor. But the attention kernel needs tokens in interleaved packed order. `scatter_indices` is a precomputed lookup that maps each packed position to its source in the concatenated order:

```python
# q_cat = concat([q_und, q_audio], dim=1)  — shape (B, N_U+N_V, H, D)
packed_q = torch.index_select(q_cat, 1, scatter_indices)  # reorder to packed layout
```

Precomputed once per forward pass in the preprocessor, reused in every block.

**`flex_attn_mask` / `flash4_attn_mask`**

Mutually exclusive, at most one is non-None. These encode the **asymmetric attention pattern** (text=causal, audio=full/noise) as a sparse block structure for the attention kernel. Not simple boolean tensors — precomputed structured objects that tell the kernel which (Q, K) pairs to compute. See the dedicated section below for full details.

**`modulation_mask`** — `Bool[B, S_V]`

Not an attention mask — maps each audio token to its corresponding diffusion timestep modulation vector. Needed when a generation stream contains multiple audio segments each with a different timestep (packed modulation). The adaLN layer uses it to apply the right `shift/scale` per token.

---

## Attention Masking: FlexAttention and Flash4

### Why a Custom Mask?

Standard causal attention is a simple lower-triangular matrix. The T2A model needs something asymmetric:

- **Text tokens** (`"causal"`) — attend to all previous text, **cannot** see audio tokens
- **Audio tokens** (`"noise"`) — attend to **all** text tokens + all audio tokens bidirectionally
- **No cross-sample attention** — tokens from sample 0 cannot attend to sample 1

### Concrete Example

2 samples packed together, each with 4 text + 4 audio tokens (`BLOCK_SIZE=4`, S=16):

```
pos:   0  1  2  3  |  4  5  6  7  |  8  9 10 11  | 12 13 14 15
type:  T  T  T  T  |  A  A  A  A  |  T  T  T  T  |  A  A  A  A
       ← text s0 → | ← audio s0 → | ← text s1  → | ← audio s1 →
       Block 0          Block 1        Block 2          Block 3
```

Full 16×16 attention matrix (1=attend, 0=blocked):

```
Q\KV   T0 T1 T2 T3  A4 A5 A6 A7  T8 T9 T10 T11  A12 A13 A14 A15
T0  [  1  0  0  0 |  0  0  0  0 |  0  0  0   0 |  0   0   0   0 ]
T1  [  1  1  0  0 |  0  0  0  0 |  0  0  0   0 |  0   0   0   0 ]
T2  [  1  1  1  0 |  0  0  0  0 |  0  0  0   0 |  0   0   0   0 ]
T3  [  1  1  1  1 |  0  0  0  0 |  0  0  0   0 |  0   0   0   0 ]  ← text cannot see audio
    |             |              |               |                 |
A4  [  1  1  1  1 |  1  1  1  1 |  0  0  0   0 |  0   0   0   0 ]  ← audio sees all text+audio in sample
A5  [  1  1  1  1 |  1  1  1  1 |  0  0  0   0 |  0   0   0   0 ]
A6  [  1  1  1  1 |  1  1  1  1 |  0  0  0   0 |  0   0   0   0 ]
A7  [  1  1  1  1 |  1  1  1  1 |  0  0  0   0 |  0   0   0   0 ]
    |             |              |               |                 |
T8  [  0  0  0  0 |  0  0  0  0 |  1  0  0   0 |  0   0   0   0 ]  ← sample 1 cannot see sample 0
T9  [  0  0  0  0 |  0  0  0  0 |  1  1  0   0 |  0   0   0   0 ]
T10 [  0  0  0  0 |  0  0  0  0 |  1  1  1   0 |  0   0   0   0 ]
T11 [  0  0  0  0 |  0  0  0  0 |  1  1  1   1 |  0   0   0   0 ]
    |             |              |               |                 |
A12 [  0  0  0  0 |  0  0  0  0 |  1  1  1   1 |  1   1   1   1 ]
A13 [  0  0  0  0 |  0  0  0  0 |  1  1  1   1 |  1   1   1   1 ]
A14 [  0  0  0  0 |  0  0  0  0 |  1  1  1   1 |  1   1   1   1 ]
A15 [  0  0  0  0 |  0  0  0  0 |  1  1  1   1 |  1   1   1   1 ]
```

### Block-Level Classification

`create_block_mask` evaluates `mask_mod` for every element in each 4×4 block, then classifies each block:

```
         KV Block 0    KV Block 1    KV Block 2    KV Block 3
         (text s0)     (audio s0)    (text s1)     (audio s1)

Q Blk 0   MIXED         ZERO          ZERO          ZERO
(text s0)  causal diag   text→audio    cross-sample  cross-sample

Q Blk 1   FULL          FULL          ZERO          ZERO
(audio s0) audio→text    audio→audio   cross-sample  cross-sample

Q Blk 2   ZERO          ZERO          MIXED         ZERO
(text s1)  cross-sample  cross-sample  causal diag   text→audio

Q Blk 3   ZERO          ZERO          FULL          FULL
(audio s1) cross-sample  cross-sample  audio→text    audio→audio
```

MIXED blocks have element-wise masks (causal triangles):

```
Block(0,0):    Block(2,2):
1 0 0 0        1 0 0 0
1 1 0 0        1 1 0 0
1 1 1 0        1 1 1 0
1 1 1 1        1 1 1 1
```

8 of 16 blocks are ZERO → **50% of computation skipped** at block level. In real training with many packed samples, the savings are much larger.

### What `BlockMask` Actually Stores

Not a matrix — a compact index structure per Q block:

```
Q block 0 (text s0):   kv_indices=[0],    full_kv_indices=[]      ← 1 mixed block
Q block 1 (audio s0):  kv_indices=[0,1],  full_kv_indices=[0,1]   ← 2 full blocks
Q block 2 (text s1):   kv_indices=[2],    full_kv_indices=[]      ← 1 mixed block
Q block 3 (audio s1):  kv_indices=[2,3],  full_kv_indices=[2,3]   ← 2 full blocks
```

- `kv_indices` — all non-zero KV blocks (FULL + MIXED)
- `full_kv_indices` — subset that are FULL (no element-wise mask needed, fastest path)
- ZERO blocks are simply absent — the kernel never visits them

### Kernel Execution at Runtime

```
for each Q block q:
    for kv in full_kv_indices[q]:
        compute attention(Q[q], K[kv], V[kv])                    # no masking, fastest
    for kv in kv_indices[q] \ full_kv_indices[q]:
        compute attention(Q[q], K[kv], V[kv], mask=mask_mod(...)) # element-wise mask
    # all other KV blocks: skip entirely
```

The full S×S boolean matrix is **never materialized** — it only exists as a Python function `mask_mod(b, h, q_idx, kv_idx) → bool` that the kernel calls inline for MIXED blocks.

### How `create_sparse_mask` Builds the `mask_mod`

Three sub-masks composed with logical AND/OR (from `flex_attn.py`):

```python
return and_masks(
    or_masks(causal_mask, full_and_noise_mask),  # each token's base pattern
    remove_noise_mask,                            # block text→audio
    sample_mask                                   # block cross-sample
)
```

- **`causal_mask`**: `q_idx >= kv_idx` — lower triangular everywhere
- **`full_and_noise_mask`**: audio tokens get a `seq_id`; tokens with same seq_id attend to each other → audio sees all audio in its split
- **`remove_noise_mask`**: prevents text Q from attending to audio KV (even though causal would allow it for earlier positions)
- **`sample_mask`**: `document_id[q] == document_id[kv]` — hard boundary between samples

### `flex_attn_mask` vs `flash4_attn_mask`

Both represent the same logical mask, different kernel formats:

| | `flex_attn_mask` (BlockMask) | `flash4_attn_mask` (FA4SparseMask) |
|---|---|---|
| **Kernel** | PyTorch FlexAttention | Flash Attention 4 (CuTe DSL) |
| **Block size** | 128 (production) | same block-level sparsity |
| **mask_mod** | called inline for MIXED blocks | translated to CuTe mask descriptor |
| **Selection** | `use_flex_attention=True` | `use_flash4_attention=True` |

Both are precomputed once per forward pass in the preprocessor and reused in every transformer block.

---

## Next Steps

- [ ] Multi-node Flyte launch config (study file)
- [ ] Audio generation inference processor
- [ ] Evaluation metrics (FAD, CLAP score) against reference run
- [ ] Extend to 3-stream model (text + vision + audio) when ready to merge
