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
| 12 | `RecursiveHSDP assert count > 0` (8-GPU) | `RecursiveHSDP` looks for `TransformerBlock` type, but omni uses `BagelMultiStreamBlock` | Use `GenericTransformerHSDP2(block_module_name=["blocks"])` (finds by name not type) |
| 13 | `LanceDataset.Config() got unexpected kwarg 'to_tensor_fn'` | `AllModalityDatasetWithMultithreading` passes `to_tensor_fn=None` but `LanceDataset.Config` doesn't accept it | Remove `to_tensor_fn=None` from `all_modality_dataset_with_multithreading.py:142` |
| 14 | `LanceDataset row_filter` treats SQL string as file path | `row_filter` expects Lance dataset path for row-index filtering, not SQL | Use processor-level `RowFieldFilter` instead of Lance SQL filter |
| 15 | `LanceDataset filter` raises `NotImplementedError` | `filter` intentionally blocked ("not well supported yet") | Same as #14 — processor-level filtering |
| 16 | SIGKILL at step 136 (8-GPU, max_num_tokens=8000) | Lance S3 Rust-side memory leak — fragment metadata cache grows unbounded with random access | Known issue: PR #7121 fixes koba V2 `LanceReader` but NOT V1 `AllModalityDatasetWithMultithreading`. Added periodic `gc.collect()` + `pa.release_unused()` every 50 batches as workaround |
| 17 | V2 LanceReader: `generator already executing` | Shared `_sampler_iter` (Python generator) accessed from multiple ThreadPoolExecutor threads | Python generators are NOT thread-safe. Fixed by using thread-local sampler iterators |
| 18 | V2 LanceReader: Lance Rust panic `take.rs:273 unwrap on None` | `ds.take(indices)` panics when called from multiple threads even with thread-local `LanceReader` instances | Lance library bug. `ds.take()` (used by V2 reader) is not thread-safe. Reverted to V1 `AllModalityDatasetWithMultithreading` which uses batch iteration (different code path, no `take()`) |
| 19 | V2 LanceReader: OOM at `max_num_tokens=16000` and `26000` | GPU memory spike during FlexAttention block mask compilation + FSDP all-gather of frozen text weights | Reduced `max_num_tokens` to 8000 (~33GB/GPU, well within 80GB H100). Future: try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

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

## Data Loader Upgrade (2026-04-08)

### Problem

Original data loader used single-threaded `BaseDataset`. Smoke test showed `data_time=0.44s` vs `step_time=0.38s` — data was the bottleneck.

### Fix

Replaced `BaseDataset` with `AllModalityDatasetWithMultithreading` (same infrastructure used by production omni T2I). Added `AudioToX` processor for audio normalization (peak norm, RMS norm, clamping).

### Results

| Mode | data_time | step_time | Status |
|------|-----------|-----------|--------|
| Single-threaded (before) | 0.44s | 0.38s | Data bottleneck |
| Multi-threaded (after) | **0.11s** | 4.4s | **Compute bottleneck** (correct) |

### Test Results

- Sync mode, 1 rank: **PASS** (5/5 batches valid)
- Multi-GPU simulation, 4 ranks: **PASS** (all ranks produce different valid batches)
- Async mode, 1 rank: **PASS** (10/10 batches, 37% throughput improvement)
- 8-GPU FSDP, full 0.6B model: **PASS** (300+ steps, loss decreasing)

---

## Loss Fix: Pure Diffusion (2026-04-08)

Removed CE (cross-entropy) loss from `BagelT2ALoss`. For T2A, the text transcript is **input context only** — not a prediction target. The model generates audio conditioned on text, not text itself.

Before: `loss = diffusion_loss x 1.0 + ce_loss x 0.25`
After: `loss = diffusion_loss x 1.0`

CE loss will be re-added for future CoT (chain-of-thought) generation, where the model first generates an extended text plan, then audio.

---

## Token Packing and Attention (2026-04-08)

### Packing Strategy

Multiple T2A samples are concatenated into a single packed sequence:

```
Pack: [text_1][audio_1][text_2][audio_2][text_3][audio_3]...
      <-------------- up to max_num_tokens (4000) ---------->
```

No padding within individual audio clips. Each clip is encoded to its natural length. Padding only at the end of the pack (for FlexAttention block alignment).

### Why max_num_tokens=4000?

With FlexAttention + per-sample block mask, attention compute is **O(N * K^2)**, not O(S^2):
- S = N * K (N samples, K tokens each). Block mask skips cross-sample attention.
- Doubling N (more samples) is **linear**. The quadratic cost is per-sample K only.

| max_num_tokens | N samples (K=200) | Compute | Scaling |
|---------------|-------------------|---------|---------|
| 4,000 | ~20 | N*K^2 = 800K | 1x |
| 16,000 | ~80 | 3.2M | 4x (linear) |
| 32,000 | ~160 | 6.4M | 8x (linear) |

The real bottleneck is **per-sample K** (individual clip duration). A 60s clip has K=1875 → K^2=3.5M, larger than 20 short samples combined. Sequence parallelism (Ulysses) is needed for long individual clips, not for more short clips.

### Per-Sample Isolation in Packed Attention

FlexAttention block mask ensures each sample only attends within itself:

```
                text_1  audio_1  text_2  audio_2  text_3  audio_3

text_1         [causal    .        .       .        .       .   ]
audio_1        [ full   noise      .       .        .       .   ]
text_2         [  .       .     causal     .        .       .   ]
audio_2        [  .       .      full    noise      .       .   ]
text_3         [  .       .        .       .     causal     .   ]
audio_3        [  .       .        .       .      full    noise ]

. = blocked (cross-sample)
```

Within each sample: text=causal (sees only text), audio=noise (sees everything).

---

## Dataset: internal-audio-v2 (2026-04-08)

The correct S3 path (was registered incorrectly):

```
s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance
```

(Note: bucket is `ai-lumalabs-datasets-ap-se-2-lance`, not `ai-lumalabs-datasets-ap-se-2`)

| Property | Value |
|----------|-------|
| Total rows | 221.8M |
| English rows (`language = 'en'`) | 166.4M (75%) |
| Language column | `language` (string) |
| Transcript column | `whisperx_asr_content` (format: `[SPEAKER_XX]"text"`) |
| Audio column | `audio_bytes` (`list<binary>` — needs unwrapping) |
| Quality filter | `round1_pass_all_filter = 1` |

English-only filter: `` `round1_pass_all_filter` = 1 AND `language` = 'en' ``

### Lance Filtering Gotcha

`LanceDataset` in koba does **NOT** support SQL-level filtering:
- `LanceDataset.Config(filter=...)` raises `NotImplementedError("Filter is not well supported yet")`
- `LanceDataset.Config(row_filter=...)` expects a **Lance dataset path** (for row-index filtering), not a SQL string

Workaround: filter at the **processor level** using a `RowFieldFilter` processor inserted at the start of the pipeline chain (before `AudioDecoder`, so no wasted decode compute on dropped rows):

```python
class RowFieldFilter:
    class Config(EasyConfig):
        required_fields: dict[str, object] | None = None  # e.g., {"language": "en", "round1_pass_all_filter": 1}

    def forward(self, sample: dict) -> dict | None:
        for field, value in self.config.required_fields.items():
            if sample.get(field) != value:
                return None  # drop this row
        return sample
```

The SQL filter string (e.g., `` `round1_pass_all_filter` = 1 AND `language` = 'en' ``) is parsed into `required_fields` dict and applied per-row. Placed first in the processor chain so filtered rows never reach the expensive `AudioDecoder`.

### audio_bytes Column Type

The `audio_bytes` column in `whisperx__multilingual_v1_compacted.lance` is `list<binary>` (a list containing one binary element), not plain `binary`. The `AudioDecoder` was patched to handle this:

```python
if isinstance(audio_bytes, list):
    audio_bytes = audio_bytes[0]  # unwrap list<binary>
```

---

## PoC Overnight Training Config (2026-04-09)

Config: `kuma.projects.omni.bagel.configs.t2a.exp_0_6b_mmaudio_pretrained`

```bash
torchrun --standalone --nproc_per_node 8 main.py \
    --config kuma.projects.omni.bagel.configs.t2a.exp_0_6b_mmaudio_pretrained \
    --name omni_t2a_0_6b_pretrained
```

| Setting | Value |
|---------|-------|
| **Model** | 0.6B (hidden=1024, 28 layers, 16 heads) |
| **Text stream** | Qwen3-0.6B pretrained (`osc://sam/qwen_3_single_weights_0_6B.pt`), **frozen** |
| **Audio stream** | Random init (xavier + zeros), **trainable** |
| **Audio VAE** | MMAudio 16k (20-dim latents, scaling_factor=1/2.3563) |
| **Dataset** | internal-audio-v2-english (~124M rows, quality + English filter) |
| **Transcript key** | `whisperx_asr_content` (format: `[SPEAKER_XX]"text"`) |
| **Audio key** | `audio_bytes` (`list<binary>`, unwrapped by AudioDecoder) |
| **max_num_tokens** | 4000 |
| **FSDP** | GenericTransformerHSDP2, intra_node=8 |
| **LR** | 1e-4, AdamW, warmup 5K steps |
| **Weight decay** | 0.01 |
| **Loss** | Pure diffusion MSE (no CE loss) |
| **Noise** | SampleLogSNRGeneric(shift=-1.6), truncated normal, sigma_shift=3.0 |
| **CFG dropout** | 0.1 (10% samples drop text for CFG training) |
| **Checkpoints** | Every 1000 steps |
| **W&B** | Enabled, project `omni-t2a` |

### Why 0.6B (not 2B)?

Originally planned 2B, but HuggingFace `Qwen/Qwen3-2B` now points to Qwen3.5 (MoE architecture, model type `qwen3_5`), not the dense Qwen3. Our installed transformers version doesn't support `qwen3_5`, and the omni team hasn't pre-converted a dense Qwen3-2B checkpoint. The 0.6B checkpoint exists and is confirmed working.

For the PoC, 0.6B is sufficient: ~300M trainable params (audio stream only), estimates 5K-10K steps overnight on 8xH100.

### Diffusion Parameters (matching reference run)

All diffusion-specific parameters match the reference run `tage001-internal-audio-v2-k3600`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `logsnr_shift` | -1.6 | Controls noise level distribution. Lower = more emphasis on higher noise |
| `noise_scale` | 1.0 | Std dev of truncated normal logSNR sampling |
| `sigma_shift` | 3.0 | Rescales sigma: `sigma_shifted = (shift*sigma) / (1 + (shift-1)*sigma)` |
| `latent_scaling` | 1/2.3563 | Normalizes MMAudio latents to ~unit variance |
| `corruption` | Rectified flow | `zt = (1-sigma)*z0 + sigma*eps` |
| `target` | Rectified flow velocity | Model predicts velocity (not noise or clean) |
| `cfg_dropout` | 0.1 | 10% samples train unconditional for CFG inference |

Goal: verify the model can produce "speech-like sound" after overnight training.

---

## Weight Initialization Strategy (2026-04-08)

### Text Stream (Stream 0) — Qwen3-0.6B Pretrained, Frozen

Load Qwen3-0.6B pretrained weights via `InitializerFromSingleCheckpoint(checkpoint_path="osc://sam/qwen_3_single_weights_0_6B.pt", strict=False)`. Text stream parameters match exactly:

- Text embedding (151,936 -> 1024)
- 28 transformer blocks (stream 0: self-attn Q/K/V/O, FFN, RMSNorm)
- Text out_proj / lm_head (1024 -> 151,936 vocab logits)

Frozen via `freeze_text_stream()` — optimizer's `parameters_fn` excludes all text params:
- `streams_pre.0.*` / `streams_post.0.*` (text stream blocks)
- `preprocess.text_processor.embedding.weight`
- `postprocess.out_projs.0.*`

No gradient, no optimizer state for these parameters. The text stream serves as a **fixed text encoder** (analogous to UMT5-XXL in the Ray3 reference run, but using packed joint attention instead of cross-attention).

### Audio Stream (Stream 1) — Random Init from Scratch

No existing checkpoint has matching architecture (0.6B Qwen3 block config trained on audio diffusion). All audio-specific weights initialized from scratch.

Init schemes (baked into the omni model configs):

| Component | Init | Why |
|-----------|------|-----|
| Audio embedding Linear(C_latent -> 1024) | `xavier_uniform_` + `zeros_` bias | Standard linear init |
| Stream 1 blocks (self-attn Q/K/V/O) x28 | `xavier_uniform_` + `zeros_` bias | Standard transformer init |
| Stream 1 blocks (FFN) x28 | `xavier_uniform_` + `zeros_` bias | Standard transformer init |
| Stream 1 RMSNorm x28 | `ones_` weight | Standard norm init |
| **Modulation MLP** (timestep -> 6x1024) | **`zeros_`** weight and bias | DiT convention |
| **FinalVidLayerPacked linear** (1024 -> C_latent) | **`zeros_`** weight and bias | DiT convention |
| **FinalVidLayerPacked adaLN** (1024 -> 2048) | **`zeros_`** weight and bias | DiT convention |
| Audio timestep embedding | Sinusoidal (fixed) + MLP (xavier) | Standard DiT |

Where C_latent = 20 for MMAudio (16kHz, ~31.25 latent frames/sec), 128 for Hunyuan DAC (48kHz, 50 frames/sec).

**Key insight**: Zeros init on modulation and output projection means at init, the audio stream contributes **zero** to the residual. The model starts as a pure (frozen) text LLM. Audio generation is learned from scratch — the zero-init provides a stable starting point (no random noise injected into text representations at step 0).

### What Transfers vs What's Random

```
Qwen3-0.6B pretrained checkpoint (osc://sam/qwen_3_single_weights_0_6B.pt)
  |-- Text embedding (151,936 -> 1024)       Y loads, frozen
  |-- 28 x BagelMultiStreamBlock
  |     |-- Stream 0 (text attn, FFN, norm)  Y loads, frozen
  |     |-- Stream 1 (audio attn, FFN, norm) X random (xavier), trainable
  |           |-- Modulation MLP             X random (zeros), trainable
  |-- Text out_proj (1024 -> 151,936)         Y loads, frozen
  |-- Audio embedding (C_latent -> 1024)      X random (xavier), trainable
  |-- Audio out_proj (1024 -> C_latent)       X random (zeros), trainable
  |-- Audio timestep embedding               X random (xavier), trainable
```

### Trainable Parameter Count (0.6B model)

With frozen text stream, ~50% of the 0.6B parameters are trainable:
- Stream 1 blocks: ~300M params (self-attn, FFN, modulation per block x28)
- Audio embedding + out_proj: ~0.02M params (tiny — C_latent is only 20)
- Timestep embedding: ~2M params
- Total trainable: ~300M (vs ~600M if training everything)

This halves optimizer memory and speeds up training.

---

## Batch Size Metric: "Minutes of Audio per Step"

### The Problem

Traditional batch size metrics (number of samples, number of tokens) are misleading for audio training:
- "32 samples" could mean 32 × 2s clips = 1 min of audio, or 32 × 30s clips = 16 min of audio
- "8000 tokens" depends on the VAE's compression ratio (MMAudio: 31.25 frames/sec, DAC: 50 frames/sec)
- Comparing across different VAEs, sample rates, or duration bucketing strategies becomes apples-to-oranges

### The Metric

**Minutes of audio per gradient step** is the natural unit for T2A training throughput:

```
audio_minutes_per_step = num_clips × avg_clip_duration / 60
```

This is directly comparable across:
- Different numbers of GPUs (more GPUs = more clips = more minutes)
- Different VAEs (MMAudio vs DAC compress differently, but audio duration is the same)
- Different packing strategies (bucketed vs greedy)
- Different sequence lengths (max_num_tokens)

### Concrete Comparison

| Run | GPUs | Config | Clips/step | Avg duration | Audio/step |
|-----|------|--------|-----------|--------------|------------|
| **Reference** (Ray3 T2A 2.9B) | 32 | bucketed [5-15s], batch=[32,24,16,12,8] | ~589 | ~10s | **~98 min** |
| **Our PoC** (Omni 0.6B, max_num_tokens=8000) | 8 | packed, variable length | ~240 | ~7s | **~28 min** |
| **Our PoC** (Omni 0.6B, max_num_tokens=4000) | 8 | packed, variable length | ~120 | ~7s | **~14 min** |

The reference run processes ~98 minutes of audio per gradient update across 32 GPUs. Our 8-GPU PoC at max_num_tokens=8000 processes ~28 minutes — roughly proportional to the GPU count ratio (8/32 = 25% GPUs, 28/98 = 29% audio throughput).

### Per-GPU Efficiency

Another useful view — **audio minutes per GPU per step**:

| Run | Audio/GPU/step |
|-----|---------------|
| Reference (32 GPUs) | ~3.1 min/GPU |
| Our PoC (8 GPUs, 8K tokens) | ~3.5 min/GPU |
| Our PoC (8 GPUs, 4K tokens) | ~1.8 min/GPU |

At max_num_tokens=8000, our per-GPU efficiency is comparable to the reference run.

### Synthetic Example: From Raw Data to Packed Batch

#### Step 1: One Training Sample

A single row from `internal-audio-v2-english`:

```
Lance row:
  audio_bytes: [<576810 bytes of WAV data>]      # 5.4s speech at 48kHz
  whisperx_asr_content: '[SPEAKER_00]"that as you continue to subscribe and listen to our"'
  language: "en"
  round1_pass_all_filter: 1
  segment_duration: 5.408
  sample_rate: 48000
```

After the pipeline processes this row:

```
1. RowFieldFilter:    checks language="en" and round1_pass_all_filter=1  -> PASS
2. AudioDecoder:      decode WAV bytes, resample 48kHz -> 16kHz
                      output: audio_tensor shape (86528,) = 5.408s × 16000 Hz
3. AudioToX:          peak normalize, clamp to [-1, 1]
4. OmniT2AAudio:      create sequence plan:
                        [TEXT] transcript, 42 text tokens (causal attention)
                        [AUDIO] 169 audio tokens = ceil(86528 / 512) (noise attention)
5. Tokenizer:         tokenize text, assign position IDs

Result: one sample = 42 text tokens + 169 audio tokens = 211 tokens total
```

#### Step 2: Packing Multiple Samples

The packing loop accumulates samples until `max_num_tokens=8000`:

```
Sample A: "that as you continue to subscribe..."     42 text +  169 audio =  211 tokens (5.4s)
Sample B: "the weather today is sunny and warm"       38 text +  125 audio =  163 tokens (4.0s)
Sample C: "I think the most important thing is..."    55 text +  250 audio =  305 tokens (8.0s)
...
Sample Z: "welcome back to the show everyone"         35 text +  156 audio =  191 tokens (5.0s)
                                                                     ─────────────────────
                                                              Total: ~8000 tokens (28 samples)
```

The packed sequence on ONE GPU:

```
[txtA][audioA][txtB][audioB][txtC][audioC]...[txtZ][audioZ][PAD]
  42    169     38    125     55    250   ...   35    156    ←pad to 8000→

Total: 8000 tokens = 28 samples packed end-to-end
```

This is ONE rank's batch. With 8 GPUs, total = 28 × 8 = 224 samples per gradient step.

#### Step 3: Attention Mask (FlexAttention Block Mask)

Each sample is isolated — tokens can only attend within their own sample:

```
             txtA  audA  txtB  audB  txtC  audC  ...  PAD
             (42)  (169) (38)  (125) (55)  (250)
    txtA  [ causal  .     .     .     .     .    ...   .  ]
    audA  [  full  noise   .     .     .     .    ...   .  ]
    txtB  [   .     .   causal   .     .     .    ...   .  ]
    audB  [   .     .    full  noise   .     .    ...   .  ]
    txtC  [   .     .     .     .   causal   .    ...   .  ]
    audC  [   .     .     .     .    full  noise  ...   .  ]
     ...
    PAD   [   .     .     .     .     .     .    ...   .  ]

    . = blocked (cross-sample or padding)
```

Zooming into Sample B (38 text + 125 audio = 163 tokens):

```
                t0  t1  t2  ... t37 | a0   a1   a2  ... a124
                <-- 38 text ------> | <---- 125 audio ------>

    t0         [ Y   .   .       .  |  .    .    .        .  ]
    t1         [ Y   Y   .       .  |  .    .    .        .  ]
    t2         [ Y   Y   Y       .  |  .    .    .        .  ]  <- text: causal
    ...                                                          (sees only prior text)
    t37        [ Y   Y   Y  ...  Y  |  .    .    .        .  ]

    a0         [ Y   Y   Y  ...  Y  |  Y    Y    Y  ...   Y  ]
    a1         [ Y   Y   Y  ...  Y  |  Y    Y    Y  ...   Y  ]  <- audio: noise/full
    a2         [ Y   Y   Y  ...  Y  |  Y    Y    Y  ...   Y  ]  (sees ALL text + ALL audio)
    ...
    a124       [ Y   Y   Y  ...  Y  |  Y    Y    Y  ...   Y  ]

    Y = attends    . = blocked
```

Key properties:
- **Audio -> Text**: YES (audio tokens attend to all text tokens in the same sample)
- **Text -> Audio**: NO (text tokens use causal mask, audio comes after text)
- **Cross-sample**: NO (Sample A cannot see Sample B)
- **Padding**: NO (padding tokens are masked out)

#### Detailed Walkthrough: Two Samples Through the MoT Pipeline

Let's trace exactly how two samples flow through concatenation, stream assignment, and attention.

**Sample A**: transcript="Hello world", 3 text tokens, 5 audio tokens
**Sample B**: transcript="Good morning", 4 text tokens, 3 audio tokens

##### 1. Concatenation into One Packed Sequence

The two samples are concatenated end-to-end into a single flat sequence of length 15:

```
Position:    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
Token:      tA0  tA1  tA2  aA0  aA1  aA2  aA3  aA4  tB0  tB1  tB2  tB3  aB0  aB1  aB2
            |<--- Sample A: 3 text + 5 audio --->|  |<--- Sample B: 4 text + 3 audio -->|
```

The data loader produces these masks (each is a boolean vector of length 15):

```
text_token_mask:  T  T  T  .  .  .  .  .  T  T  T  T  .  .  .     (7 text tokens total)
vae_token_mask:   .  .  .  T  T  T  T  T  .  .  .  .  T  T  T     (8 audio tokens total)
```

And per-sample boundaries for attention:

```
sample_lens:  [8, 7]                              (Sample A = 8 tokens, Sample B = 7 tokens)
split_lens:   [3, 5, 4, 3]                        (txtA=3, audA=5, txtB=4, audB=3)
attn_modes:   [causal, noise, causal, noise]       (text=causal, audio=noise, alternating)
```

##### 2. Stream Assignment: Text Stream vs Audio Stream

The model's preprocess separates the packed sequence into two streams using the masks:

```
Stream 0 (text/understanding):
  Extract tokens where text_token_mask=True:
    tokens:      tA0  tA1  tA2  tB0  tB1  tB2  tB3       (7 tokens)
    positions:    0    1    2    8    9   10   11           (from position_ids)
    RoPE:        cos/sin computed from positions             (Qwen3RotaryEmbedding)
    modulation:  None                                        (text has no timestep)

Stream 1 (audio/generation):
  Extract tokens where vae_token_mask=True:
    tokens:      aA0  aA1  aA2  aA3  aA4  aB0  aB1  aB2   (8 tokens)
    positions:    3    4    5    6    7   12   13   14       (continues from text)
    RoPE:        cos/sin computed from positions             (same Qwen3RotaryEmbedding)
    modulation:  timestep -> 6 x hidden_dim adaLN params    ("double" modulation)
```

Each stream computes its own Q, K, V independently.

##### 3. Pack into Joint Attention

The two streams' Q/K/V are concatenated and reordered back into the original packed sequence order using `scatter_indices`:

```
Before packing (concatenated order):
  [tA0 tA1 tA2 tB0 tB1 tB2 tB3 | aA0 aA1 aA2 aA3 aA4 aB0 aB1 aB2]
   <---- stream 0 (7 tokens) --> | <---- stream 1 (8 tokens) -------->

After packing (original interleaved order via scatter_indices):
  [tA0 tA1 tA2 aA0 aA1 aA2 aA3 aA4 tB0 tB1 tB2 tB3 aB0 aB1 aB2]
   pos 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
```

##### 4. FlexAttention with Block Mask

A single FlexAttention call runs on the packed Q/K/V with a block mask that encodes three rules simultaneously:

**Rule 1 — Sample isolation**: tokens from Sample A cannot attend to Sample B and vice versa.
**Rule 2 — Text is causal**: each text token only sees previous text tokens in its sample.
**Rule 3 — Audio sees everything**: each audio token sees all text + all audio in its sample.

```
Q \ K   tA0 tA1 tA2 aA0 aA1 aA2 aA3 aA4 tB0 tB1 tB2 tB3 aB0 aB1 aB2

tA0      Y   .   .   .   .   .   .   .  | .   .   .   .   .   .   .
tA1      Y   Y   .   .   .   .   .   .  | .   .   .   .   .   .   .
tA2      Y   Y   Y   .   .   .   .   .  | .   .   .   .   .   .   .
aA0      Y   Y   Y   Y   Y   Y   Y   Y  | .   .   .   .   .   .   .
aA1      Y   Y   Y   Y   Y   Y   Y   Y  | .   .   .   .   .   .   .
aA2      Y   Y   Y   Y   Y   Y   Y   Y  | .   .   .   .   .   .   .
aA3      Y   Y   Y   Y   Y   Y   Y   Y  | .   .   .   .   .   .   .
aA4      Y   Y   Y   Y   Y   Y   Y   Y  | .   .   .   .   .   .   .
         ─────────────────────────────────+────────────────────────────
tB0      .   .   .   .   .   .   .   .  | Y   .   .   .   .   .   .
tB1      .   .   .   .   .   .   .   .  | Y   Y   .   .   .   .   .
tB2      .   .   .   .   .   .   .   .  | Y   Y   Y   .   .   .   .
tB3      .   .   .   .   .   .   .   .  | Y   Y   Y   Y   .   .   .
aB0      .   .   .   .   .   .   .   .  | Y   Y   Y   Y   Y   Y   Y
aB1      .   .   .   .   .   .   .   .  | Y   Y   Y   Y   Y   Y   Y
aB2      .   .   .   .   .   .   .   .  | Y   Y   Y   Y   Y   Y   Y

Y = attends    . = blocked    | = sample boundary
```

Read any row to see what that token attends to:
- **tA1** (text token 1 of Sample A): sees tA0, tA1 only (causal within Sample A's text)
- **aA2** (audio token 2 of Sample A): sees ALL of tA0-tA2 + aA0-aA4 (full access within Sample A)
- **tB2** (text token 2 of Sample B): sees tB0, tB1, tB2 only (causal within Sample B's text)
- **aB0** (audio token 0 of Sample B): sees ALL of tB0-tB3 + aB0-aB2 (full access within Sample B)
- **aA4 -> tB0**: BLOCKED (cross-sample boundary)

##### 5. Unpack Back to Streams

After attention, the output is unpacked back into two streams using `packed_und_token_indexes` and `packed_gen_token_indexes`:

```
Attention output (packed, 15 tokens):
  [oA0 oA1 oA2 oA3 oA4 oA5 oA6 oA7 oB0 oB1 oB2 oB3 oB4 oB5 oB6]

Unpack to Stream 0 (text, 7 tokens):
  [oA0 oA1 oA2 oB0 oB1 oB2 oB3]  <- text outputs (used by text postprocess)

Unpack to Stream 1 (audio, 8 tokens):
  [oA3 oA4 oA5 oA6 oA7 oB4 oB5 oB6]  <- audio outputs (used by audio postprocess)
```

Each stream then goes through its own post-SDPA processing (residual + FFN + modulation for audio).

##### 6. Loss Computation

After 28 transformer blocks, the audio stream output is projected to latent dim (1024 -> 20) and the diffusion loss is computed:

```
Audio prediction (8 tokens, 20 channels):
  [predA0 predA1 predA2 predA3 predA4 predB0 predB1 predB2]

Target (rectified flow velocity):
  [targA0 targA1 targA2 targA3 targA4 targB0 targB1 targB2]

Loss = MSE(prediction, target) averaged over all 8 audio tokens
```

Text stream output is discarded (frozen, no CE loss in T2A mode).

#### Step 4: What the Model Sees

For each sample in the pack, the model:
1. **Text stream** (frozen Qwen3-0.6B): encodes the transcript into text representations
2. **Audio stream** (trainable): receives noisy audio latents + text conditioning via packed attention
3. **Loss**: MSE between model's audio prediction and the rectified flow target (velocity)

The text representations flow into audio via the packed attention — audio Q tokens fetch text K/V, giving the audio stream full access to the transcript. Text tokens never see audio (causal mask), so the frozen text stream produces the same representations regardless of what audio is present.

### Reading the Training Logs

`num_samples` in the training log is the number of audio clips in **one rank's** packed batch (NOT the total across all GPUs):

```
[48] diffusion_loss: 1.699, num_samples: 28.000, ...
                            ^^^^^^^^^^^^^^^^
                            28 clips on THIS GPU, not total
```

Total throughput per step:
```
total_clips  = num_samples × num_GPUs = 28 × 8 = 224 clips
audio_per_step = total_clips × avg_duration = 224 × 7s ≈ 26 min
```

### Known Issue: Lance S3 Memory Leak (2026-04-09)

Our overnight run was SIGKILL'd at step 136 (~18 min). Root cause: **Lance Rust-side memory leak** when reading from S3 with random access. This is a known issue across all T2A training (Slack thread: #training-updates 2026-04-06).

**Root cause** (from Richard Cai's investigation, PR #7121):
- Lance caches fragment metadata in Rust memory for each new fragment accessed via S3
- The cache grows with each `ds.take()` call touching new fragments and is **never evicted**
- With 222M-row datasets (2,227 fragments) and shuffled random access: ~350 MB/hr per worker
- This is NOT Python-level, NOT PyArrow, NOT audio decode — it's in Lance's Rust native code

**Fix** (PR #7121, merged): Recreate the `lance.dataset()` handle every 50 reads + `gc.collect()` + `pa.release_unused()`. However, this fix is in koba V2's `LanceReader` — our data loader uses V1's `AllModalityDatasetWithMultithreading` → `LanceDataset` which does NOT have this fix.

**Options**:
1. Port the connection-reset fix to `LanceDataset` / `AllModalityDatasetWithMultithreading`
2. Switch to koba V2's `LanceReader` (bigger refactor)
3. Reduce number of concurrent Lance readers to slow the leak (workaround)

### Memory vs Batch Size (0.6B model, 8× H100 80GB)

| max_num_tokens | GPU memory | Clips/step (total) | Audio/step | Status |
|---------------|------------|-------------------|------------|--------|
| 4,000 | 13 GB | ~120 | ~14 min | OK |
| 8,000 | 33 GB | ~224 | ~26 min | OK (current) |
| 16,000 | OOM | — | — | 67GB used + 15GB alloc failed |
| 26,000 | OOM | — | — | 45GB used + 40GB alloc failed |

The OOM at 16K+ is likely from FlexAttention block mask compilation spikes and FSDP all-gathering both frozen text + trainable audio weights. With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 12K-16K may work.

---

## Overnight PoC Run Log (2026-04-09)

### Run: omni_t2a_0_6b_pretrained

W&B: https://wandb.ai/luma-ai/omni-t2a/runs/g0ebocom

| Setting | Value |
|---------|-------|
| Model | 0.6B (28 layers, hidden=1024), pretrained Qwen3-0.6B text stream (frozen) |
| Audio stream | Random init, trainable (~300M params) |
| Audio VAE | MMAudio 16k (20-dim, scaling=1/2.3563) |
| Dataset | `internal-audio-v2-english` (multilingual table + runtime RowFieldFilter) |
| max_num_tokens | 8000 |
| FSDP | GenericTransformerHSDP2, intra_node_shard=8 |
| Loss | Pure diffusion MSE (no CE) |
| Checkpoint | Saved at step 200 to `os://ai-lumalabs-checkpoints-ap-se-2/root/omni_t2a_0_6b_pretrained/00000200/denoiser/` |

### Training Progress (before crash)

| Step | diffusion_loss | grad_norm | step_time | data_time | num_samples/rank |
|------|---------------|-----------|-----------|-----------|-----------------|
| 1 | 1.73 | 3.05 | 24.6s | 8.5s | ~25 (warmup) |
| 50 | 1.68 | 2.74 | 7.3s | 2.0s | ~28 |
| 100 | 1.65 | 2.80 | 7.3s | 2.0s | ~30 |
| 200 | 1.56 | 2.79 | 7.3s | 2.1s | ~30 |
| ~300 | ~1.5 | ~2.7 | ~7.3s | ~2.0s | ~30 |

Loss decreased from 1.73 to ~1.5 over 300 steps. Gradients stable (~2.7).

### Crash: OOM at ~300 Steps (Lance S3 Memory Leak)

Job killed by system cgroup OOM after ~300 steps (~37 min). Same root cause as before: Lance Rust-side fragment metadata cache grows unbounded when reading from S3 with random access.

**What we tried and what didn't work:**

| Approach | Result |
|----------|--------|
| Periodic `gc.collect()` + `pa.release_unused()` in main thread | Slowed leak slightly (136 → 300 steps before OOM) but didn't fix it. GC runs in main thread, leak is in V1 loader threads |
| V2 `LanceReader` (has the fix) | Lance `ds.take()` panics in Rust (`take.rs:273 unwrap on None`) when called from multiple threads, even with thread-local readers. Reverted |
| Thread-local `LanceReader` instances | Same Rust panic — `ds.take()` is not thread-safe |

**Root cause**: The fix in PR #7121 periodically recreates the `lance.dataset()` handle to flush the Rust cache. This fix is in V2 `LanceReader.read_batch()` (which uses `ds.take()`). But V1 `AllModalityDatasetWithMultithreading` uses `LanceDataset` which iterates via PyArrow batch iteration — a different code path. The V1 `LanceDataset` holds one persistent Lance handle per dataset that is never recreated.

**V2 `LanceReader` unusable because**: `ds.take(indices)` (used by V2 reader) triggers a Rust panic when called from ThreadPoolExecutor threads. The V1 path uses `ds.to_batches()` / scanner iteration which doesn't panic. This is a Lance library bug — `take()` is not thread-safe even with independent dataset handles.

### Performance Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| step_time | 7.3s | 4.9x slower than reference (1.5s) |
| data_time | 2.0s (26% of step) | Runtime RowFieldFilter rejects 25%+ of reads |
| compute_time | 5.3s | Forward (frozen text + trainable audio) + backward + optim |
| GPU memory | 33 GB / 80 GB | 41% utilization |
| GPU util | ~67% (drops to 0% during data loading) | Data loading is visible bottleneck |

**Why 7.3s/step vs reference 1.5s/step:**

1. **Data loading (2.0s)**: Runtime filtering wastes 25%+ of Lance reads. Reference has no runtime filter.
2. **Packed attention on 8000 tokens**: We pack 28 clips into one 8000-token FlexAttention call. Reference does independent self-attention on ~313 tokens per clip + separate cross-attention.
3. **Frozen text forward**: We run frozen Qwen3 text stream through all 28 blocks every step. Reference uses UMT5-XXL as a separate encoder (compute once, cache).

### Deep Dive: Why V2 LanceReader Panicked in Our Setup

When we tried using V2 `LanceReader` directly (medium fix attempt), we hit:
```
thread '<unnamed>' panicked at take.rs:273:27: called Option::unwrap() on a None value
```

**Investigation**: `AudioBatchingDatasetV2` uses the EXACT same pattern — one shared `self._reader` (LanceReader) called from multiple ThreadPoolExecutor worker threads via `self._reader.read_batch()`. The comment in their code says `"# Read from Lance (I/O is thread-safe)"`. So why does it work for them?

**Root cause**: The connection reset code in `LanceReader.read_batch()` (PR #7121) does:
```python
# Thread A (every 50 reads):
self._connections.pop(dataset_name, None)   # removes connection
gc.collect()

# Thread B (concurrent):
ds = self._get_connection(dataset_name)      # gets connection from dict
table = ds.take(indices)                     # uses connection
```

There's a **race condition**: Thread A pops the connection while Thread B is in the middle of `ds.take()` on the same handle. Thread B holds a reference to the old `ds` object that Thread A just destroyed → Rust internal state is corrupted → panic.

**Why `AudioBatchingDatasetV2` rarely hits this**: With `_connection_reset_interval=50` and only 2-4 workers, the probability of Thread A resetting at the exact moment Thread B is mid-`take()` is very low. It may occasionally panic, but the retry loop (`max_retries=5`) catches it. Our setup with 8 workers hit the timing window more frequently.

**Why even thread-local readers panicked**: We tried creating independent `LanceReader` instances per thread. Each thread had its own `_connections` dict and own `lance.dataset()` handle. But `lance.dataset()` may share **process-level Rust state** (e.g., a global async runtime, connection pool, or fragment cache). Multiple independent handles doing `ds.take()` concurrently can still conflict at the Rust level.

### Options to Fix the Memory Leak

#### Option 1: Port V2 fix to V1 `AllModalityDatasetWithMultithreading`

The V1 loader uses a different Lance read path than V2. Instead of `ds.take(indices)` (random access by row ID), V1 uses `ds.to_batches()` / scanner iteration (sequential batch reading). This path does NOT trigger the Rust panic.

The fix: add periodic `lance.dataset()` handle recreation in V1's loader thread:

```python
# In _batch_loader_worker_modality() of all_modality_dataset_with_multithreading.py:
read_count = 0
while not self._stop_event.is_set():
    read_count += 1
    if read_count % 50 == 0:
        # Recreate Lance handles to flush Rust fragment metadata cache
        for i, ds_config in enumerate(original_configs[modality]):
            self.datasets[modality][i] = ds_config.setup()  # new handle
        gc.collect()
        pa.default_memory_pool().release_unused()

    raw_batch, uri = self._get_next_raw_batch(modality)
    raw_batch = raw_batch.to_pylist()
    # ... enqueue items
```

This is safe because the loader thread is the **only thread** that calls `_get_next_raw_batch()` for a given modality — no concurrent access to the Lance handle within the same modality. The processor threads receive already-read Python dicts, never touching Lance directly.

| Pro | Con |
|-----|-----|
| Fixes the leak at its source | Modifies shared `lib/koba/` infrastructure |
| All V1 users benefit (omni T2I, VLM, etc.) | Needs code review from koba owners |
| Low risk: loader thread is single-writer per modality | Need to store original configs for recreation |
| No architecture change to our data loader | |

#### Option 2: Use `AudioBatchingDatasetV2` as data source + bridge layer

`AudioBatchingDatasetV2` is the proven Ray3 T2A data loader. It has V2 `LanceReader` built in, plus duration bucketing, ThreadPoolExecutor, per-bucket queues, TP broadcast — all battle-tested. The reference run (`ovah4wwn`) ran for days without OOM.

The architecture:

```
AudioBatchingDatasetV2 (existing, proven)
  ├── LanceReader (V2, with memory leak fix)
  ├── ThreadPoolExecutor (2-4 workers, thread-local pipelines)
  ├── Duration bucketing (per-bucket queues)
  └── Outputs: {"x": audio_tensor (B, L), "txt": caption_string, ...}
             flat batch, same-duration clips, already decoded + augmented

                              ↓

Bridge Layer (new code)
  ├── For each sample in the flat batch:
  │     1. Create sequence plan: [TEXT(caption), AUDIO(audio_tensor)]
  │     2. OmniAddTokenizedSequenceElement
  │     3. OmniElementAudio (token masks, vae_token_mask=True)
  │     4. OmniElementText + OmniQwen3Tokenizer
  │     5. OmniPositionIDMRoPE
  ├── Accumulate tokenized samples
  └── pack_sequence() → packed batch dict

                              ↓

OmniT2ATrainer.step() (unchanged)
  ├── frozen audio encoder(x_audio) → z0
  └── BagelT2ALoss(denoiser, z0, **batch)
```

The key difference: `AudioBatchingDatasetV2` manages the `LanceReader` and `ThreadPoolExecutor` as a **single cohesive unit** — the reader is created, initialized, and used within the same class's lifecycle. The worker threads call `self._reader.read_batch()` in a controlled loop with error handling and retry. The race condition on connection reset is rare and handled by retries.

Our bridge layer only needs to:
1. Iterate over `AudioBatchingDatasetV2` (get flat batches)
2. Run omni tokenization processors on each sample
3. Pack into omni packed format

| Pro | Con |
|-----|-----|
| Uses battle-tested infrastructure (no memory leak) | More code: bridge layer (~100-150 lines) |
| Gets duration bucketing for free (uniform batches) | Two-stage pipeline (AudioBatchingDatasetV2 → bridge → trainer) |
| Doesn't modify shared `lib/koba/` code | Audio already decoded by the time bridge runs (can't skip decode for filtered rows) |
| Reference run validates this exact data path | Need to configure AudioBatchingDatasetV2 in addition to omni configs |
| Thread safety proven in production | |

### Recommendation

**Option A** is simplest and lowest risk for unblocking the PoC tonight (20 lines, single-writer per modality). **Option C** (pre-encode latents) is the best long-term solution for production — eliminates both the memory leak and the audio decode overhead.

---

## Detailed Fix Options for Lance S3 Memory Leak (2026-04-09)

### Background: The Core Constraint

Lance `ds.take()` and scanner iteration from S3 leak Rust-side fragment metadata cache. The ONLY known fix is **periodically recreating the `lance.dataset()` handle** (drops the Rust cache). PR #7121 implements this in V2 `LanceReader` (connection reset every 50 reads + `gc.collect()` + `pa.release_unused()`).

The challenge: this fix lives in V2 `LanceReader.read_batch()`, but our data loader uses V1 `AllModalityDatasetWithMultithreading` → `LanceDataset` (different code path). And when we tried using V2 `LanceReader` directly from ThreadPoolExecutor threads, Lance's `ds.take()` panicked in Rust due to thread safety issues.

### Key Threading Analysis

**V1 `AllModalityDatasetWithMultithreading` threading model:**
```
Main thread: creates AllModalityDatasetWithMultithreading in __init__
  └── __init__ creates LanceDataset objects: self.datasets[modality] = [LanceDataset, ...]

Loader threads (1 per modality, started in _start_pipeline):
  └── _batch_loader_worker_modality(modality):
      while not stop:
          raw_batch, uri = self._get_next_raw_batch(modality)  # reads from self.datasets[modality]
          raw_batch = raw_batch.to_pylist()
          for item in raw_batch:
              items_to_process_queue[modality].put(item)

Processor threads (N shared, round-robin across modalities):
  └── _processor_worker(thread_id):
      while not stop:
          item = items_to_process_queue[modality].get()
          processed = _apply_processors(processors, item, modality)  # Python processing only
          processed_items_queue[modality].put(processed)
```

**Critical insight**: The loader thread for a given modality is the **ONLY thread** that touches `self.datasets[modality]` (the Lance handles). Processor threads only receive already-read Python dicts — they never touch Lance. This makes Option A safe.

**V2 `LanceReader` threading model (used by `AudioBatchingDatasetV2`):**
```
Main thread: creates ONE shared self._reader = LanceReader(...)
  └── self._reader.initialize_for_worker()
  └── self._executor = ThreadPoolExecutor(N workers)

Worker threads (N, from ThreadPoolExecutor):
  └── each calls self._reader.read_batch(name, indices)
      └── self._reader._get_connection(name)  # shared self._connections dict
      └── ds.take(indices)                     # Lance Rust code
      └── every 50 reads: self._connections.pop(name)  # RACE CONDITION
```

**Race condition**: Thread A does `_connections.pop(name)` while Thread B is mid-`ds.take()` on the same handle. Thread B's handle is partially destroyed → Rust panic at `take.rs:273`.

`AudioBatchingDatasetV2` survives this because with 2-4 workers and reset every 50 reads, the race window is tiny. Our setup with 8 workers hit it more often.

### Option A: Fix V1 Loader Thread (Simplest, Lowest Risk)

**What**: Add periodic `lance.dataset()` handle recreation in V1's per-modality loader thread.

**Where**: `lib/koba/koba/feeder/all_modality_dataset_with_multithreading.py`, in `_batch_loader_worker_modality()` (around line 282-317).

**Why it's safe**: The loader thread for modality "t2a" is the ONLY thread that reads from `self.datasets["t2a"]`. No concurrent access. The processor threads only see Python dicts.

**Implementation** (~20 lines):

```python
# In _batch_loader_worker_modality(modality):
import gc
import pyarrow as pa

read_count = 0
CONNECTION_RESET_INTERVAL = 50

while not self._stop_event.is_set():
    # Periodic Lance handle recreation to flush Rust fragment metadata cache
    read_count += 1
    if read_count % CONNECTION_RESET_INTERVAL == 0:
        for i, ds_obj in enumerate(self.datasets[modality]):
            # Recreate the lance.dataset() handle
            old_uri = ds_obj.dataset.uri
            storage_opts = {
                "timeout": "120s",
                "region": "ap-southeast-2",
                "connect_timeout": "60s",
                "request_timeout": "600s",
                "client_max_retries": "20",
                "client_retry_timeout": "600",
                "download_retry_count": "5",
            }
            ds_obj.dataset = lance.dataset(old_uri, storage_options=storage_opts)
        gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass

    # ... existing code: raw_batch, uri = self._get_next_raw_batch(modality)
```

**Effort**: ~20 lines in one file.
**Risk**: Low — single-writer per modality, no threading concern.
**Scope**: Modifies shared `lib/koba/` but change is minimal and isolated to loader thread.
**Benefit**: Fixes the memory leak for ALL V1 users (omni T2I, VLM, T2A).
**Limitation**: Doesn't improve data loading performance (still single-threaded Lance reads per modality).

### Option B: Use `AudioBatchingDatasetV2` + Bridge Layer

**What**: Use the proven Ray3 T2A data loader (which has V2 `LanceReader` with the memory fix built in) as the data source, then add a bridge layer to convert its flat batch output to the omni packed sequence format.

**Architecture**:

```
AudioBatchingDatasetV2 (existing, battle-tested)
  ├── V2 LanceReader (with memory leak fix, retry, connection pooling)
  ├── ThreadPoolExecutor (2-4 workers)
  │     └── Each worker: sampler.get_indices() → reader.read_batch() → V1 pipeline
  ├── Duration bucketing (per-bucket queues, uniform-length batches)
  ├── Audio augmentation (RMS norm, peak norm, clamp via AudioToX in V1 pipeline)
  └── Output: flat batch dict per bucket
      {
          "x": Tensor(batch_size, num_frames),     # decoded audio waveforms, all same duration
          "txt": list[str],                          # transcript captions
          "duration_bucket": float,                  # e.g., 5.04
          "original_durations": list[float],         # actual durations before padding
          "dataset_names": list[str],                # source dataset per sample
      }

                              ↓

OmniT2ABridgeDataset (new, ~150 lines)
  ├── Iterates over AudioBatchingDatasetV2 (gets flat batches)
  ├── For each sample in the flat batch:
  │     1. Create SequenceElement list:
  │         [SequenceElement(type=TEXT, text_str=caption, loss=False, modality="t2a"),
  │          SequenceElement(type=AUDIO, media=Media(data=audio_tensor), loss=True, modality="t2a")]
  │     2. Run omni tokenization pipeline on the sequence plan:
  │         - OmniAddTokenizedSequenceElement.Config().setup().forward(sample)
  │         - OmniElementAudio.Config(compression_factor=512).setup().forward(sample)
  │         - OmniElementText.Config().setup().forward(sample)
  │         - OmniQwen3Tokenizer.Config().setup().forward(sample)
  │         - OmniPositionIDMRoPE.Config(order="THW").setup().forward(sample)
  │     3. Collect tokenized_sequence_plan
  ├── Accumulate samples until max_num_tokens reached
  └── pack_sequence(samples) → packed batch dict
      {
          "txt": Tensor,
          "text_token_mask": Tensor,
          "vae_token_mask": Tensor,
          "audio_token_mask": Tensor,
          "x_audio": list[Tensor],
          "position_ids": Tensor,
          "sample_lens": list[int],
          "split_lens": list[int],
          "attn_modes": list[str],
          ...
      }

                              ↓

OmniT2ATrainer.step() (unchanged)
  ├── frozen MMAudio encoder(x_audio) → z0 latents
  └── BagelT2ALoss(denoiser, z0, **batch)
```

**Where**: New file `projects/kuma/.../omni/bagel/configs/data/omni_t2a_bridge_dataset.py`.

**Configuration**: The `AudioBatchingDatasetV2.Config` is created from our existing `OmniT2ADatasetConfig` parameters, or we create a separate config that wraps both.

**Key code for the bridge** (pseudocode):

```python
class OmniT2ABridgeDataset(torch.utils.data.IterableDataset):
    def __init__(self, audio_batching_config, max_num_tokens, ...):
        self.audio_dataset = audio_batching_config.setup()  # AudioBatchingDatasetV2
        self.max_num_tokens = max_num_tokens
        # Create tokenization processors (one-time, main thread)
        self.tokenizer = OmniQwen3Tokenizer.Config().setup()
        self.position_id = OmniPositionIDMRoPE.Config(order="THW").setup()
        # ... other processors

    def __iter__(self):
        curr_pack = []
        curr_tokens = 0
        for flat_batch in self.audio_dataset:
            # flat_batch["x"] shape: (batch_size, num_frames)
            # flat_batch["txt"]: list of caption strings
            for i in range(flat_batch["x"].shape[0]):
                audio_tensor = flat_batch["x"][i]  # (num_frames,)
                caption = flat_batch["txt"][i]

                # Build sequence plan
                sample = {
                    "sequence_plan": [
                        SequenceElement(type=SequenceType.TEXT, text_str=caption,
                                       loss=False, modality="t2a",
                                       supervise_last_text_token=True),
                        SequenceElement(type=SequenceType.AUDIO,
                                       media=Media(media_type="audio", data=audio_tensor),
                                       loss=True, modality="t2a"),
                    ]
                }

                # Run tokenization pipeline
                sample = self.add_tokenized(sample)
                sample = self.element_audio(sample)
                sample = self.element_text(sample)
                sample = self.tokenizer(sample)
                sample = self.position_id(sample)

                tok_plan = sample["tokenized_sequence_plan"]
                sample_len = sum(s.num_tokens for s in tok_plan)

                if curr_tokens + sample_len > self.max_num_tokens and curr_pack:
                    yield pack_sequence([s["tokenized_sequence_plan"] for s in curr_pack])
                    curr_pack = []
                    curr_tokens = 0

                curr_pack.append(sample)
                curr_tokens += sample_len
```

**Effort**: ~150 lines new code. No shared code modifications.
**Risk**: Low — uses proven AudioBatchingDatasetV2 infrastructure.
**Scope**: Only our T2A project files.
**Benefit**: Memory leak fixed + duration bucketing + proven threading model.
**Limitation**: Two-stage pipeline. Audio is decoded by AudioBatchingDatasetV2 before the bridge — can't skip decode for samples that would be filtered. Need to configure AudioBatchingDatasetV2 separately.

### Option C: Pre-encode Audio Latents into a New Lance Table

**What**: Run MMAudio encoder offline on the entire `internal-audio-v2-english` dataset. Save the resulting 20-dim latent tensors to a new Lance table. Training reads tiny latent tensors instead of 500KB raw audio bytes.

**Why this helps**:
1. **Memory leak**: Tiny columns (20×T floats ≈ 2KB per 5s clip) vs large audio bytes (500KB). Fragment metadata per read is proportionally smaller → leak grows 250x slower.
2. **No audio decode at training time**: Latents are pre-computed. No MMAudio encoder forward pass needed in the trainer.
3. **No frozen encoder GPU memory**: The ~73M param MMAudio encoder is no longer loaded on each GPU.
4. **Faster data loading**: 2KB reads vs 500KB reads = 250x less S3 bandwidth per sample.

**New Lance table schema**:

```
audio_latents: fixed_size_list<float32, 20>   # MMAudio latent channels (per frame)
audio_latent_length: int32                     # number of latent frames (T)
whisperx_asr_content: string                   # transcript
language: string                               # language code
segment_duration: float64                      # original audio duration in seconds
```

Or alternatively, store as a flat binary blob:

```
audio_latents_blob: binary                     # serialized tensor (20, T) as bytes
audio_latent_frames: int32                     # T
whisperx_asr_content: string
```

**Offline encoding job** (LAX or standalone script):

```python
# Pseudocode for the encoding job
import lance
import torch
from kuma.projects.ray3_t2av.models.mmaudio import PretrainedMMAudioEncoder

encoder = PretrainedMMAudioEncoder.Config(deterministic=True, mode="16k",
                                          scaling_factor=1/2.3563).setup()
encoder.eval().to("cuda")

src_ds = lance.dataset("s3://...whisperx__multilingual_v1_compacted.lance")
# Filter: round1_pass_all_filter=1 AND language='en'

output_rows = []
for batch in src_ds.to_batches(columns=["audio_bytes", "whisperx_asr_content", ...],
                                filter="`round1_pass_all_filter` = 1 AND `language` = 'en'",
                                batch_size=64):
    for row in batch.to_pylist():
        audio_bytes = row["audio_bytes"][0]  # unwrap list<binary>
        waveform = decode_audio(audio_bytes, sample_rate=16000)
        with torch.no_grad():
            z = encoder(waveform.unsqueeze(0).cuda())  # (1, 20, T)
        output_rows.append({
            "audio_latents": z.squeeze(0).cpu().numpy(),  # (20, T)
            "audio_latent_frames": z.shape[2],
            "whisperx_asr_content": row["whisperx_asr_content"],
            "segment_duration": row["segment_duration"],
        })

# Write new Lance table
lance.write_dataset(pa.Table.from_pylist(output_rows),
                    "s3://...internal_audio_v2_english_mmaudio_latents.lance")
```

**Modified trainer**: `OmniT2ATrainer.step()` would skip the audio encoding step:

```python
# Before (current):
x_audio = batch["x_audio"]
with trace_io("audio_encode"):
    z0_list = []
    for audio_tensor in x_audio:
        z0_i = self.audio_encoder(audio_input)  # GPU forward pass per clip
        z0_list.append(z0_i)
z0 = torch.cat(z0_list, dim=2)

# After (with pre-encoded latents):
z0 = batch["audio_latents"].to(self.device)  # already encoded, just move to GPU
```

**Modified data loader**: Read `audio_latents` column instead of `audio_bytes`. The `OmniElementAudio` processor would read latent frames instead of raw audio frames for token count calculation.

**Effort**: ~2 hours for encoding job + ~1 hour for trainer/loader modifications.
**Risk**: Medium — need to manage a derived dataset (re-encode when encoder changes).
**Scope**: New Lance table + modified trainer + modified data loader. No shared code changes.
**Benefit**: Eliminates memory leak + audio decode time + frozen encoder memory. Fastest training.
**Limitation**: Tied to one specific audio VAE (MMAudio 16k). If you switch to DAC, need to re-encode. The pre-encoded table is a snapshot — can't change augmentation (RMS norm etc.) at training time since raw audio is gone.

### Option D: Fix V2 LanceReader Thread Safety

**What**: Make `LanceReader.read_batch()` thread-safe by using per-thread connection tracking and reset, instead of the shared `_read_count` and `_connections` dict.

**Where**: `lib/koba/koba/v2/core/reader.py`.

**Implementation**:

```python
class LanceReader:
    def __init__(self, config):
        ...
        self._thread_local = threading.local()  # per-thread state
        # Remove shared: self._read_count, self._connection_reset_interval

    def _get_thread_local_state(self):
        if not hasattr(self._thread_local, 'connections'):
            self._thread_local.connections = {}
            self._thread_local.read_count = 0
        return self._thread_local

    def _get_connection(self, dataset_name: str) -> Any:
        state = self._get_thread_local_state()
        if dataset_name not in state.connections:
            import lance
            state.connections[dataset_name] = lance.dataset(
                self.datasets[dataset_name],
                storage_options=self.storage_config.to_storage_options(),
            )
        return state.connections[dataset_name]

    def read_batch(self, dataset_name, indices, with_indices=True):
        state = self._get_thread_local_state()
        state.read_count += 1

        # Per-thread connection reset (no race condition — only this thread's state)
        if (self._connection_reset_interval > 0
                and state.read_count % self._connection_reset_interval == 0):
            state.connections.pop(dataset_name, None)
            gc.collect()
            try:
                pa.default_memory_pool().release_unused()
            except Exception:
                pass

        # ... rest of read_batch using self._get_connection(dataset_name)
```

**Key change**: `_connections` dict and `_read_count` move from shared instance state to `threading.local()`. Each thread has its own Lance handles and its own reset counter. No shared mutable state between threads → no race condition.

**Backward compatibility**: `AudioBatchingDatasetV2` would automatically get the fix since it uses `LanceReader.read_batch()`. Single-threaded callers also work (main thread gets its own thread-local state).

**Effort**: ~50 lines modifying `lib/koba/koba/v2/core/reader.py`.
**Risk**: Medium — changes V2 shared code. Needs testing to ensure `AudioBatchingDatasetV2` still works. Also need to verify that `initialize_for_worker()` resets thread-local state properly after `fork()`.
**Scope**: Shared `lib/koba/` code.
**Benefit**: Fixes the race condition properly. Our V2 LanceReader approach (medium fix) would then work.
**Limitation**: Doesn't fix V1 users (omni T2I/VLM). Each thread creates its own Lance connection → more S3 connections. Need to verify Lance handles are truly independent at the Rust level (process-level Rust state may still be shared).

### Summary Comparison

| | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| **What** | Fix V1 loader thread | AudioBatchingDatasetV2 + bridge | Pre-encode latents | Fix V2 reader thread safety |
| **Effort** | ~20 lines | ~150 lines | ~2-3 hours | ~50 lines |
| **Risk** | Low | Low | Medium | Medium |
| **Fixes leak** | Yes | Yes | Yes (eliminates it) | Yes |
| **Fixes perf** | No | Yes (bucketing) | Yes (no decode) | No |
| **Modifies shared code** | Yes (minimal) | No | No | Yes |
| **Benefits others** | Yes (all V1 users) | No (T2A only) | No (T2A only) | Yes (all V2 users) |
| **Production-ready** | Good for short-term | Good for medium-term | Best for long-term | Good if V2 reader is the standard |
| **Unblocks PoC tonight** | Yes | No (more code) | No (need encoding job) | Maybe (if test passes) |

---

## Option B PoC Run (2026-04-09)

### Run: omni_t2a_0_6b_pretrained_v2

W&B: https://wandb.ai/luma-ai/omni-t2a/runs/u27stj53
Branch: `dongguo/omni-t2a-v2`

**Architecture**: AudioBatchingDatasetV2 (V2 LanceReader, memory leak fix) → OmniT2ABridgeDataset (tokenize + pack) → OmniT2ATrainer

| Setting | Value |
|---------|-------|
| Model | 0.6B, Qwen3-0.6B pretrained text (frozen), random audio stream |
| Data loader | Option B: `AudioBatchingDatasetV2` + bridge layer |
| Dataset | emilia + internal-audio-v1 (via `t2a_dataset()`) |
| Duration buckets | [5.04, 7.54, 10.04, 12.54, 15.04] s |
| Batch sizes per bucket | [32, 24, 16, 12, 8] |
| max_num_tokens | 8000 |
| FSDP | GenericTransformerHSDP2, intra_node_shard=8 |
| Loss | Pure diffusion MSE |

### Early Training Progress

| Step | diffusion_loss | step_time | data_time | Notes |
|------|---------------|-----------|-----------|-------|
| 1 | 1.898 | 80.1s | 41.6s | Cold start (torch.compile + queue fill) |
| 13 | 1.811 | 16.0s | 3.3s | Warming up |
| 39 | 1.869 | 13.0s | 1.6s | Stabilizing |

### Key Improvements over V1 Data Loader Run

| Metric | V1 (previous, OOM'd at ~300 steps) | Option B (current) |
|--------|-------------------------------------|-------------------|
| Memory leak | Yes (Lance S3, ~350 MB/hr) | **No** (V2 LanceReader resets handles) |
| Duration bucketing | No (variable per pack) | **Yes** (uniform duration per batch) |
| data_time (stabilized) | ~2.0s | ~1.6s |
| step_time (stabilized) | ~7.3s | ~13.0s (still warming up, expected to decrease) |

Note: step_time is higher because this is earlier in torch.compile warmup. The previous V1 run's 7.3s was measured at step 200+. Option B step_time should converge to similar values.

### Run Comparison: V1 vs Option B (300 steps)

After 300 steps, compared the two runs side-by-side:

- V1 run: https://wandb.ai/luma-ai/omni-t2a/runs/g0ebocom (orange, `omni_t2a_0_6b_pretrained`)
- Option B run: https://wandb.ai/luma-ai/omni-t2a/runs/u27stj53 (green, `omni_t2a_0_6b_pretrained_v2`)

| Metric | V1 (orange) | Option B (green) |
|--------|-------------|------------------|
| step_time | Stable ~7-10s | Baseline ~8-12s, frequent spikes to 20-40s, some extreme 60-80s |
| num_samples | 20-35, stable | 20-40, comparable |
| loss | 1.7 → 1.35 (smooth) | 1.7 → 1.5 (noisier, fewer effective steps) |
| grad_norm | 2.0-3.0, stable | 2.0-3.0, comparable |
| Memory | OOM at ~300 steps | Stable (no OOM) |

**Diagnosis**: Option B solved the memory leak but introduced step_time instability. Root cause: `OmniT2ABridgeDataset` runs the full omni tokenization pipeline (6 processors: CFGDropout → AddTokenized → ElementAudio → ElementText → Tokenizer → PositionID) **in the main thread** (`num_workers=0`). When `AudioBatchingDatasetV2` has an S3 latency spike or its internal buffer empties, the main thread blocks and the GPU sits idle.

V1 was faster because `AllModalityDatasetWithMultithreading` ran tokenization in `num_actors=8` parallel threads. The main thread only did packing from a prefetched queue.

### Fix: Prefetch Thread (2026-04-09)

Commit: `[omni] Add prefetch thread to T2A bridge dataset for stable step_time`

Moved the entire tokenize+pack pipeline into a background daemon thread (`t2a-bridge-prefetch`) that feeds packed batches into a `queue.Queue(maxsize=8)`. The main training thread just does `queue.get()`.

```
Before:  Main thread: [S3 wait] → [tokenize] → [pack] → [GPU forward] → repeat
After:   Prefetch:    [S3 wait] → [tokenize] → [pack] → queue.put()
         Main thread: queue.get() → [GPU forward] → queue.get() → ...
```

Key implementation details:

- `prefetch_queue_size=8` packed batches buffered ahead (configurable)
- Processors instantiated inside the thread via `_build_processors()` (not shared with main thread)
- `stop_event` + queue drain for clean shutdown
- Sentinel `None` value signals worker completion/crash
- Worker is a daemon thread — dies automatically if main process exits

Expected improvement: step_time should stabilize to ~7-10s (matching V1) without the memory leak.

### Prefetch Run Results (2026-04-09)

W&B: https://wandb.ai/luma-ai/omni-t2a/runs/3bhjps4k (`omni_t2a_0_6b_pretrained_v2_prefetch`)

At ~290 steps, the prefetch thread fixed data loading but exposed a compute-side bottleneck:

| Metric | V1 (g0ebocom) | Option B no prefetch (u27stj53) | Option B + prefetch (3bhjps4k) |
|--------|---------------|--------------------------------|-------------------------------|
| **data_time** | ~2.0s (steady) | ~0s baseline, spikes 15-20s every ~20-30 steps | **0.25s** (steady, no spikes) |
| **Normal step_time** | 7-10s (stable) | 8-12s | 8-10s |
| **Spike step_time** | none | 20-80s (irregular) | **~30s every ~3 steps** |
| **Effective avg step_time** | ~7.3s | ~13s | **~15.9s** (dragged up by periodic spikes) |
| **num_samples/rank** | 20-35, stable | 20-40 | 17-37 |
| **loss @ 280 steps** | ~1.5 | ~1.5 | ~1.55 |
| **grad_norm** | 2.0-3.0 | 2.0-3.0 | 2.2-2.8 |
| **GPU memory** | 33GB (OOM at ~300) | 33GB (stable) | **33GB (stable)** |
| **GPU utilization** | — | varied | 90-100% |

**What the prefetch thread fixed**: data_time dropped from ~1.6s (with 15-20s spikes) to a steady 0.25s. The queue successfully decouples S3 I/O from GPU training.

**What it did NOT fix**: ~30s step_time spikes occurring every ~3 steps. These do NOT correlate with num_samples changes (e.g., samples=19→19 still spikes at 30.2s), ruling out torch.compile recompilation from shape changes. The spikes are compute-side, likely from FlexAttention `create_block_mask` recomputation for each new packed layout, or periodic CUDA memory allocation overhead.

### Padding Strip Bug Fix (2026-04-09)

The initial padding strip commit (`196c9c0b50`) had a slicing bug: `sample_audio[:orig_frames]` truncated the **channel dimension** (dim 0) instead of the **time dimension** (dim -1) when audio was 2D `[channels, frames]`. Since channels=1, the slice was a no-op — padding was never actually stripped.

Fix: `sample_audio[..., :orig_frames]` — ellipsis preserves all leading dims, slices only time.

This explains why `num_samples` remained ~20-28/rank in runs `u27stj53`, `3bhjps4k`, and `28r7xpeu` despite the "strip" code being present.

### V3 Run: Prefetch + Padding Fix + 10K Tokens (2026-04-09)

W&B: https://wandb.ai/luma-ai/omni-t2a/runs/16gvru69 (`omni_t2a_0_6b_pretrained_v3`)

Changes from previous runs:

- Fixed padding strip bug (`[..., :orig_frames]`)
- Increased `max_num_tokens` from 8000 → 10000 (~25% more, targeting ~47GB of 80GB)

Early results (step 5-13, post torch.compile warmup):

| Metric | V2 no-fix (u27stj53) | V3 broken strip (28r7xpeu) | **V3 fixed (16gvru69)** |
|--------|---------------------|---------------------------|------------------------|
| num_samples/rank | ~22 | ~22 | **28-50 (avg ~38)** |
| loss @ step 10 | ~1.85 | ~1.90 | **~1.67** |
| step_time (post warmup) | 8-12s + spikes | 8-10s + spikes | **~8s so far** |
| GPU memory | 33GB | 31GB | **47GB** |

The padding fix is confirmed working: `num_samples` nearly doubled (22 → 38 avg), meaning the token budget is now filled with real audio instead of silence padding. Lower initial loss (1.67 vs 1.90) reflects cleaner latents from unpadded MMAudio encoding.

### Efficiency Deep Dive: Why Option B Compute Time is 2.6x Slower (2026-04-09)

After adding the prefetch thread and comparing all three runs on W&B (`train/data_time`), an important observation emerged:

**data_time comparison:**

| Run | Baseline data_time | Spikes |
|-----|-------------------|--------|
| V1 (orange, `omni_t2a_0_6b_pretrained`) | Consistent ~2-3s | None |
| Option B no prefetch (green, `omni_t2a_0_6b_pretrained_v2`) | ~0s | Every ~20-30 steps, spikes to 15-20s |
| Option B with prefetch (purple, `omni_t2a_0_6b_pretrained_v2_prefetch`) | ~0s | Less frequent, but still present |

**Paradox**: Option B has much lower data_time most of the time, yet step_time is nearly 2x higher. Where does the extra time go?

**Decomposing step_time:**

```
step_time = data_time + compute_time

V1:       step_time ~7.5s = data_time ~2.5s + compute ~5.0s
Option B: step_time ~13s  = data_time ~0s   + compute ~13.0s
```

The GPU is doing **2.6x more compute** per step in Option B. Same model, same `max_num_tokens=8000`. The cause: **duration bucket zero-padding**.

#### Root Cause: AudioBatchingDatasetV2 Pads Audio to Bucket Ceilings

`AudioBatchingDatasetV2` uses duration bucketing for uniform batch shapes. Every clip is zero-padded to its bucket ceiling:

```
Config: duration_buckets = [5.04, 7.54, 10.04, 12.54, 15.04] seconds
        probabilities = [0.2, 0.2, 0.2, 0.2, 0.2] (equal)
        compression_factor = 512, sample_rate = 16kHz
```

A 3-second clip assigned to the 5.04s bucket gets 2s of zero-padding appended. The bridge then passes the **padded** waveform to the tokenization pipeline. `OmniElementAudio` computes `num_tokens = ceil(padded_frames / compression_factor)`, inflating the token count.

#### Token Inflation Math

| Bucket | Padded frames | Audio tokens | + ~50 text tokens | Sample total |
|--------|--------------|-------------|-------------------|-------------|
| 5.04s  | 80,640       | 158         | ~50               | ~208        |
| 7.54s  | 120,640      | 236         | ~50               | ~286        |
| 10.04s | 160,640      | 314         | ~50               | ~364        |
| 12.54s | 200,640      | 392         | ~50               | ~442        |
| 15.04s | 240,640      | 470         | ~50               | ~520        |

**Average sample size** (equal bucket probability): ~364 tokens → **~22 samples per 8000-token pack**

**V1 comparison** (internal-audio-v2-english, no padding, whisperx segments ~3-8s):
Average ~5s clip → 156 audio tokens + 50 text → ~206 tokens → **~39 samples per 8000-token pack**

#### Impact on Compute

Both V1 and Option B packs have 8000 tokens → FlexAttention compute is identical. The extra cost comes from two sources:

**1. MMAudio encoder processes padded waveforms.** The trainer calls `audio_encoder(x_audio)` where `x_audio` is the raw waveform concatenated from all samples in the pack. In Option B, this includes all zero-padded regions:

- Option B: 22 samples × avg 10.04s × 16kHz = **~3.5M frames** (includes silence padding)
- V1: 39 samples × avg 5s × 16kHz = **~3.1M frames** (all real audio)

**2. More critically: padding consumes the token budget.** With 8000 tokens per pack, zero-padded audio tokens crowd out real samples. Option B fits ~22 real clips per step vs V1's ~39. Each gradient step updates on **44% fewer real audio clips**, which:

- Makes each step less sample-efficient (fewer real gradients per update)
- Explains why Option B's loss decreases slower (1.7→1.5 vs V1's 1.7→1.35 at step 300)
- Means more steps needed to reach the same loss → worse wall-clock efficiency

**3. Dataset difference also contributes.** V1 uses `internal-audio-v2-english` (whisperx segments, mostly 3-8s), while Option B uses `emilia + internal-audio-v1` (broader duration distribution). Longer clips inflate the average token count further.

#### Fixing the Efficiency Gap

**Option A: Strip padding in the bridge** — use `original_durations` from the flat batch to truncate padded audio back to actual length before passing to `OmniElementAudio`. This preserves bucketed batching (AudioBatchingDatasetV2 benefit) while eliminating token waste:

```python
# In OmniT2ABridgeDataset._prefetch_worker():
original_durations = flat_batch.get("original_durations")
for i in range(batch_size):
    sample_audio = audio_tensor[i]
    # Truncate padding: use original duration, not bucket ceiling
    if original_durations:
        orig_frames = int(original_durations[i] * 16000)
        sample_audio = sample_audio[:orig_frames]
```

Expected result: token budget used for real audio only → ~39 samples/pack (matching V1) → compute per step drops from ~13s to ~5s.

**Option B: Use the same dataset as V1** — switch to `internal-audio-v2-english` (shorter clips, less padding waste). Combined with padding stripping, this maximizes samples per pack.

**Option C: Tune bucket sizes** — use narrower buckets (e.g., [3, 5, 7, 9, 11]s) to reduce max padding per clip, at the cost of more bucket queues.

#### Summary

| Factor | V1 | Option B | Impact |
|--------|-----|----------|--------|
| Duration padding | None (variable-length) | Padded to bucket ceiling | ~1.8x token inflation |
| Avg sample size | ~206 tokens | ~364 tokens | 44% fewer clips per pack |
| MMAudio encoder frames | ~3.1M/step | ~3.5M/step | ~13% more encoder work |
| Real clips per step | ~39 | ~22 | Slower loss convergence |
| data_time | ~2.5s (steady) | ~0s (with spikes) | Lower baseline, but spiky |
| compute_time | ~5.0s | ~13.0s | **2.6x more** |

The key insight: **lower data_time does not mean faster training**. The padding overhead shifted the bottleneck from data loading to GPU compute. The priority fix is stripping duration padding in the bridge, which should recover V1-level compute efficiency while keeping Option B's memory stability.

### Task Formulation: Padding, Loss Masking, and Non-Causal Encoders (2026-04-09)

The efficiency gap above is a symptom of a deeper task formulation issue. The question: given a text transcript, the target audio has **variable duration** (same sentence can be spoken in 1.5s or 4s). How we handle this variance changes what the model learns.

#### How Ray3 T2A Handles Variable-Length Audio (Reference)

Ray3 uses a careful three-part approach (see `ray3_t2av/trainer.py` and `ray3_t2av/loss/diffusion_loss.py`):

1. **Sequential encoding**: MMAudio is a non-causal encoder — padding zeros corrupt ALL latent positions, not just the padded ones. Ray3 encodes each sample **individually at its true length**, then pads the resulting latents and stacks into a batch. This preserves latent fidelity.

2. **Per-sample length tracking**: `audio_lengths` is threaded through the pipeline alongside the audio tensor. After encoding, `latent_lengths` is computed and passed to the loss.

3. **Masked diffusion loss**: `_create_temporal_mask(lengths, shape)` creates a binary mask (1.0 for real, 0.0 for padding). Loss is computed as a masked mean — only real audio regions contribute to gradients:

   ```python
   # Ray3: loss/diffusion_loss.py
   if modality == "aud" and aud_lengths_cond is not None:
       temporal_mask = _create_temporal_mask(aud_lengths_cond, diff_loss.shape)
       diff_loss = _masked_mean(diff_loss, temporal_mask)
   ```

#### How Our Omni T2A Currently Handles It (Problems)

Our pipeline has three compounding issues:

**Problem 1: Non-causal encoder sees padded input.** `AudioBatchingDatasetV2` pads clips to bucket ceilings. The bridge passes padded waveforms directly to MMAudio encoder. Because MMAudio is non-causal (convolution-based), the zero-padded suffix contaminates **all** latent positions — not just the padding positions. This means even the "real" speech latents are corrupted.

```
Actual audio:  [speech speech speech 0 0 0 0 0]  (padded to bucket)
                          ↓ MMAudio (non-causal)
Latents:       [corrupt corrupt corrupt silence]   ← ALL positions affected
```

**Problem 2: No loss masking.** Our loss (`bagel_t2a.py` line 157-158) computes:

```python
diffusion_loss = nn.functional.mse_loss(prediction, target, reduction="mean")
```

MSE is averaged over **all** positions including silence padding. Consequences:

- **Diluted loss signal**: The model gets "free" loss reduction from trivially predicting near-zero silence latents. At step 300, the reported loss of 1.5 may be lower than actual speech quality warrants because silence predictions are easy.
- **Gradient pollution**: Gradients from silence regions push the model toward low-energy outputs, potentially suppressing dynamic range and expressiveness in real speech.
- **Incorrect weighting**: A 3s clip in a 15s bucket contributes 80% silence loss and 20% speech loss. The model optimizes for silence more than speech.

**Problem 3: No duration signal.** The model receives text tokens + audio tokens but has no way to know the intended audio duration. In the packed sequence `[TEXT AUDIO]`, the audio region is fixed per bucket, not per utterance. The model cannot learn "this sentence should be spoken in 2s vs 5s" because that information is lost in the padding.

#### Comparison Table

| Aspect | Ray3 T2A | Our Omni T2A (current) |
|--------|----------|----------------------|
| Audio encoding | Per-sample at true length | Batched with padding (non-causal corruption) |
| Latent quality | Clean (no padding artifacts) | Corrupted (padding bleeds into real positions) |
| Loss masking | `_masked_mean()` on real positions only | `mse_loss(reduction="mean")` over all positions |
| Duration signal | Implicit in latent length | Lost in bucket padding |
| Silence gradients | Zero (masked out) | Non-zero (learning to predict silence) |

#### Why This Matters Beyond Efficiency

This isn't just about wasting compute on silence. It affects model quality:

1. **Audio fidelity**: Non-causal encoder corruption means the model is trained on degraded latents. Even if it learns a perfect denoiser, the targets themselves are wrong.

2. **Speaking rate**: Without a duration signal, the model may learn the average rate across all bucket paddings rather than the natural rate of each utterance. This could produce monotonic, unnatural speech.

3. **Loss interpretation**: The reported `diffusion_loss` is not comparable between V1 and V2 runs because V2's loss includes trivially-easy silence predictions. A loss of 1.5 in V2 could correspond to worse real speech quality than a loss of 1.5 in V1.

#### Recommended Fixes (Priority Order)

**Fix 1 (Critical): Strip padding before encoding.**

In the bridge, use `original_durations` from the flat batch to truncate audio before passing to the omni tokenization pipeline:

```python
# In OmniT2ABridgeDataset._prefetch_worker():
original_durations = flat_batch.get("original_durations")
for i in range(batch_size):
    sample_audio = audio_tensor[i]
    if original_durations:
        orig_frames = int(original_durations[i] * 16000)
        sample_audio = sample_audio[:orig_frames]
```

This solves both the efficiency problem (more real samples per pack) and the encoder corruption problem (MMAudio sees only real audio).

**Fix 2 (Important): Add loss masking.**

If variable-length audio is ever packed (not just concatenated), add temporal masking to `BagelT2ALoss`:

```python
# Compute per-token loss mask from vae_token_mask + original lengths
# Only score positions corresponding to real audio
diff_loss = masked_mean(diff_loss, audio_valid_mask)
```

Note: If Fix 1 is applied (no padding at encoding time), loss masking becomes less critical because all audio tokens correspond to real content. But it remains good practice for robustness.

**Fix 3 (Future): Duration conditioning.**

For TTS quality, the model should eventually receive a duration signal (e.g., target duration in seconds as a conditioning token, or a separate duration predictor like FastSpeech 2). This lets the model learn speaking rate variation rather than averaging it out.

#### Concrete Walkthrough: Two Processing Settings with Synthetic Examples

To make the above analysis precise, here is a step-by-step comparison using three synthetic audio samples.

**Setup:** Three samples, all with transcript "Hello world":

| Sample | Actual duration | Actual frames (16kHz) | Bucket (Option B) | Padded frames |
|--------|----------------|----------------------|-------------------|---------------|
| A | 2.0s | 32,000 | 5.04s | 80,640 |
| B | 4.5s | 72,000 | 5.04s | 80,640 |
| C | 6.0s | 96,000 | 7.54s | 120,640 |

MMAudio 16k encoder: `compression_factor=512`, `latent_dim=20`.

##### Setting 1: Ray3 (Sequential Encoding, Masked Loss)

**Step 1: Encode each sample individually at its true length.**

MMAudio is a convolutional encoder with non-causal (bidirectional) convolutions. The encoder "sees" the entire input when computing any output position.

```
Sample A: waveform [32,000 frames] → MMAudio → latents [20, 63]   (63 = ceil(32000/512))
Sample B: waveform [72,000 frames] → MMAudio → latents [20, 141]  (141 = ceil(72000/512))
Sample C: waveform [96,000 frames] → MMAudio → latents [20, 188]  (188 = ceil(96000/512))
```

Each encoding is clean — the encoder only sees real speech waveform. No zero-padding enters the convolution receptive fields.

**Step 2: Pad latents to max length in batch, track lengths.**

```
Max latent length in batch = 188 (Sample C)

Sample A latents: [20, 63]  → pad → [20, 188]   latent_length = 63
Sample B latents: [20, 141] → pad → [20, 188]   latent_length = 141
Sample C latents: [20, 188] → no pad             latent_length = 188

Stacked: z0 = [3, 20, 188]
lengths = [63, 141, 188]
```

Note: padding happens **after** encoding — the zeros are in latent space, never seen by the encoder.

**Step 3: Diffusion forward (noise + denoise).**

```
eps = randn([3, 20, 188])        # noise
zt = alpha * z0 + sigma * eps     # noisy latents
prediction = model(zt, sigma, text_conditioning)  # denoiser output

raw_loss = (prediction - target)^2   # shape [3, 20, 188], per-element MSE
```

**Step 4: Masked loss — only real positions contribute.**

```
temporal_mask from lengths:
  Sample A: [1,1,...,1, 0,0,...,0]   63 ones, 125 zeros
  Sample B: [1,1,...,1, 0,0,...,0]   141 ones, 47 zeros
  Sample C: [1,1,...,1,1,1,1,1,1]   188 ones, 0 zeros

Shape: [3, 1, 188] (broadcast over channel dim)

Per-sample loss:
  loss_A = sum(raw_loss[0] * mask[0]) / (63 * 20)    ← only 63 real positions
  loss_B = sum(raw_loss[1] * mask[1]) / (141 * 20)   ← only 141 real positions
  loss_C = sum(raw_loss[2] * mask[2]) / (188 * 20)   ← all 188 real positions

final_loss = (loss_A + loss_B + loss_C) / 3   ← each SAMPLE weighted equally
```

**Key properties:**

- Each sample contributes equally regardless of duration
- Silence regions produce zero gradient
- 2s and 6s clips have equal influence on model update
- Total real latent positions scored: 63 + 141 + 188 = **392**

##### Setting 2: Our Omni T2A (Batch Encoding with Padding, Unmasked Loss)

**Step 1: AudioBatchingDatasetV2 pads waveforms to bucket ceilings.**

```
Sample A: waveform [32,000] → pad to 5.04s → [80,640 frames]
  Content: [speech speech ... speech 0 0 0 0 0 0 0 0 0 0 0 0]
                   32,000 real          48,640 zeros

Sample B: waveform [72,000] → pad to 5.04s → [80,640 frames]
  Content: [speech speech ... speech 0 0 0 0 0]
                   72,000 real      8,640 zeros

Sample C: waveform [96,000] → pad to 7.54s → [120,640 frames]
  Content: [speech speech ... speech 0 0 0 0 0 0 0 0]
                   96,000 real          24,640 zeros
```

**Step 2: Bridge passes padded waveforms to omni tokenization.**

`OmniElementAudio` computes token count from the **padded** length:

```
Sample A: num_tokens = ceil(80,640 / 512) = 158 tokens   (vs 63 real)
Sample B: num_tokens = ceil(80,640 / 512) = 158 tokens   (vs 141 real)
Sample C: num_tokens = ceil(120,640 / 512) = 236 tokens   (vs 188 real)
```

These token counts fill the `max_num_tokens=8000` packing budget.

**Step 3: Trainer encodes the full padded waveform through MMAudio.**

```
                    ┌─────────────────────────────────────────────────────┐
Sample A waveform:  │ speech speech speech | 0  0  0  0  0  0  0  0  0  │
                    └─────────────────────────────────────────────────────┘
                                        ↓
                              MMAudio encoder (non-causal conv)
                                        ↓
                    ┌─────────────────────────────────────────────────────┐
Sample A latents:   │  L1   L2   ...  L63 | L64  L65  ...  L158         │
                    └─────────────────────────────────────────────────────┘
                       ↑ corrupted ↑        ↑ silence-ish but also       ↑
                       by trailing           corrupted by edge effects
                       zeros in
                       receptive field
```

**Why non-causal encoding corrupts everything:**

MMAudio's encoder uses standard (non-causal) 1D convolutions. Each output position's receptive field extends both left and right. Consider latent position L60 (near the end of real speech in Sample A):

```
L60's receptive field (simplified, ~100 frames each side):

    waveform:  ... [speech] [speech] [speech] [0] [0] [0] [0] ...
                            ↑ L60 center ↑
                    ←── receptive field ──→

The convolution mixes real speech WITH trailing zeros.
L60's value is different from what it would be without padding.
```

For Sample A (2s speech, 3s padding), even L1's receptive field may reach into the zero region if the encoder's total receptive field is large enough. The corruption is worst near boundaries but affects all positions to some degree.

Compare: if Sample A were encoded at its true length (32,000 frames), L60 would only see real speech in its receptive field → clean latent.

**Step 4: Trainer computes diffusion loss with NO masking.**

```
z0 shape per sample after encoding:
  Sample A: [20, 158]   ← 63 positions from real speech, 95 from silence (all corrupted)
  Sample B: [20, 158]   ← 141 real, 17 silence (all corrupted)
  Sample C: [20, 236]   ← 188 real, 48 silence (all corrupted)
```

In the packed sequence format, these are concatenated:

```
packed audio tokens: [A's 158 tokens | B's 158 tokens | C's 236 tokens] = 552 tokens
```

Diffusion loss:

```
raw_loss = (prediction - target)^2   # shape [..., 552, 20]

diffusion_loss = raw_loss.mean()     # mean over ALL 552 × 20 = 11,040 values
                                     # no masking, no per-sample weighting
```

**What the loss "sees":**

```
Position breakdown of the 552 tokens:
  Tokens 1-158   (Sample A): 63 corrupted-speech + 95 silence
  Tokens 159-316 (Sample B): 141 corrupted-speech + 17 silence
  Tokens 317-552 (Sample C): 188 corrupted-speech + 48 silence

Real speech tokens:  63 + 141 + 188 = 392  (71%)
Silence tokens:      95 + 17 + 48  = 160   (29%)

But ALL tokens (including "real speech" ones) have corrupted latent values
due to non-causal encoder seeing the padding.
```

**Gradient implications:**

- 29% of gradient signal comes from learning to predict silence (trivially easy, near-zero latents)
- Sample A (2s clip) contributes 158 tokens, Sample C (6s clip) contributes 236 tokens → **longer clips get more gradient weight**, not proportional to real content
- Sample A is 60% silence by token count → the model mostly learns silence from this sample
- The "speech" latents are corrupted targets — the model is learning to predict the **wrong** values

##### Side-by-Side Summary

```
                    Setting 1 (Ray3)              Setting 2 (Our Omni T2A)
                    ─────────────────             ───────────────────────────

Waveform input      [32000]  (true length)        [80640]  (padded to bucket)
to encoder          [72000]                       [80640]
                    [96000]                       [120640]

Encoder calls       3 separate calls              3 calls on padded waveforms
                    (clean, no padding)           (non-causal: padding corrupts)

Latent tokens       63 + 141 + 188 = 392         158 + 158 + 236 = 552
scored in loss

Silence tokens      0 (masked out)               160 (29% of loss)
in loss

Latent quality      Clean (encoder saw            Corrupted (encoder's
                    only real speech)             receptive field mixed
                                                  speech with zero-padding)

Sample weighting    Equal (1/3 each)              By padded token count
                                                  (Sample C gets 1.5x weight
                                                  of Sample A)

Effective tokens    392 real tokens               392 corrupted tokens
for learning        (high quality gradients)      + 160 silence tokens
                                                  (wasted + distorted gradients)

Token budget used   392 / 8000 = 4.9%            552 / 8000 = 6.9%
(for same 3 clips)  → room for ~57 more clips    → room for ~43 more clips
```

The fix is straightforward: in the bridge, truncate `audio_tensor[i]` to `original_durations[i]` **before** passing to the omni tokenization pipeline. This makes our pipeline behave like Setting 1 — encode real audio only, no padding waste, no encoder corruption.

#### Audio Encoder Batching Strategy (2026-04-09)

**Question**: After stripping bucket padding, `x_audio` is a list of variable-length waveforms. Should we encode them one-by-one (sequential) or batch them?

**Current implementation** (`omni_t2a.py` lines 148-165):

```python
# Sequential: encode each sample individually at its true length
for audio_tensor in x_audio:
    audio_input = audio_tensor.unsqueeze(0)  # (1, L_i)
    z0_i = self.audio_encoder(audio_input)   # (1, C, T_latent_i)
    z0_list.append(z0_i)
z0 = torch.cat(z0_list, dim=2)  # (1, C, total_T_latent)
```

This is the same approach as Ray3's `_encode_audio_sequential()` (`ray3_t2av/trainer.py` lines 602-642), which also encodes one sample at a time with true-length inputs to avoid non-causal corruption.

**Is sequential encoding a bottleneck?**

MMAudio 16k encoder is a lightweight CNN (~5M params, mel spectrogram + conv stack). Encoding cost per sample:

```
5s clip at 16kHz = 80,000 frames → ~1-2ms on A100
Per training step: ~40 samples/pack × ~1.5ms ≈ 60ms total encoding
28-layer transformer on 8000 tokens: ~5000ms

Encoding is ~1% of step time — not a bottleneck.
```

**Batching options if encoder becomes heavier (future DAC VAE, larger models):**

| Approach | Encode calls/step | Padding waste | Latent quality | When to use |
|----------|-------------------|---------------|----------------|-------------|
| **Sequential** (current) | N (~40) | 0 | Perfect | Default — correct and fast for lightweight encoders (MMAudio) |
| **Pad to max-in-pack** | 1 | Bounded by max-min duration within pack | Slight corruption near sample boundaries | If encoder is heavy AND samples have similar lengths |
| **Group by similar length** | 3-5 groups | Minimal per group | Near-perfect | Best tradeoff for heavy encoders with diverse durations |

**Why pad-to-max-in-pack is reasonable after the padding strip fix:**

After stripping bucket padding, all samples within a single pack fit the `max_num_tokens=8000` budget. With an average sample size of ~200 tokens (~6s audio), a pack holds ~40 samples. Their duration range is naturally bounded — the packer greedily fills, so the longest clip in a pack is at most `max_per_seq_len` and the shortest is whatever fits the remaining budget. Typical spread within a pack: ~3-8s, much smaller than the old bucket spread of 2-15s. This means pad-to-max-in-pack wastes at most ~5s of padding per sample, vs the old bucket design that could waste ~13s.

**Decision**: Keep sequential encoding for now. MMAudio encoding is <1% of step time. Revisit if:

- Switching to DAC VAE (heavier encoder, 128-dim latents)
- Scaling to much larger batch sizes / longer sequences
- Profiling shows `audio_encode` trace exceeding ~5% of step time

---

## Medium-Term Roadmap

Three task types with different loss structures:

| Task | Input | Output | Loss |
|------|-------|--------|------|
| T2A (current) | text prompt | audio latents | diffusion only |
| Understanding (A2T) | audio + text prompt | text response | CE only |
| CoT generation | simple prompt | extended text -> audio | CE on text + diffusion on audio |
| Multi-speaker | reference audios + text | speech audio | diffusion (interleaved audio/text input) |

Multi-speaker format:
```
<speaker0> [audio_ref_0] <speaker1> [audio_ref_1]
Prompt: generate "<speaker0> sentence one <speaker1> sentence two"
-> [generated speech audio]
```

This requires the understanding stream to handle **interleaved text + audio input** (reference clips alongside text), similar to how ViT image features are injected in the omni vision model.

---

## Git Branch Structure

| Branch | Purpose | Status |
|--------|---------|--------|
| `dongguo/omni-t2a` | V1 data loader (AllModalityDatasetWithMultithreading + GC workaround) | Pushed, OOMs after ~300 steps |
| `dongguo/omni-t2a-v2` | Option B data loader (AudioBatchingDatasetV2 + bridge layer) | **Active**, running PoC |

Both branches share the same model/trainer/loss code. Only the data loading layer differs.

---

## Merge Notes

When merging `dongguo/omni-t2a-v2` to main, be aware:

- **`lib/koba/koba/pipelines/default_t2a.py`** was deleted in main (PR #6718, "graveyard unused/dead code") but we kept it because our `OmniT2ADatasetConfig` (V1 path) imports from it. The Option B path (`OmniT2ABridgeDatasetConfig`) does NOT depend on this file. If using Option B only, this file can be safely dropped during merge.
- **`lib/koba/koba/feeder/all_modality_dataset_with_multithreading.py`** — we removed `to_tensor_fn=None` (line 142) because `LanceDataset.Config` doesn't accept it. This fix benefits all V1 users.
- **`lib/koba/koba/processor/audio_ops.py`** — added `list<binary>` unwrapping in `AudioDecoder` for whisperx datasets. This is a backward-compatible enhancement.

---

## Lessons Learned

1. **Commit frequently**: A `git reset --hard HEAD` during cherry-pick conflict resolution wiped all uncommitted changes (~15 files across 2 days of work). Had to re-apply everything from conversation memory. Always commit before destructive git operations.

2. **Lance `ds.take()` is NOT thread-safe**: Even with independent `lance.dataset()` handles per thread, concurrent `ds.take()` calls can panic in Rust. The Rust runtime may share process-level state. Use Lance through `AudioBatchingDatasetV2` (which manages threading internally) rather than calling `ds.take()` from your own ThreadPoolExecutor.

3. **V1 `LanceDataset` SQL filters**: `filter=` raises `NotImplementedError`, `row_filter=` expects a Lance dataset path (not SQL). Use processor-level `RowFieldFilter` as a workaround, or use pre-filtered Lance tables.

4. **`AudioBatchingDatasetV2` is the right abstraction for audio**: It handles threading, bucketing, memory management, and the V2 LanceReader — all battle-tested. Don't reinvent this; use it as a data source and bridge to your format.

5. **HuggingFace model names change**: `Qwen/Qwen3-2B` now points to Qwen3.5 (MoE, model type `qwen3_5`), not the dense Qwen3-2B. The omni team uses pre-converted checkpoints (`osc://sam/qwen_3_single_weights_{size}.pt`) which are pinned to specific architectures.

---

## Launch Scripts

### Previous PoC Runs (on this node)

All runs launched from `/fsx/dongguo/Projects/lumaverse/projects/kuma` with `.venv` activated.

**V1 data loader run** (`dongguo/omni-t2a` branch) — OOM'd at ~300 steps:

```bash
cd /fsx/dongguo/Projects/lumaverse/projects/kuma
source .venv/bin/activate

# Config: AllModalityDatasetWithMultithreading + internal-audio-v2-english
torchrun --standalone --nproc_per_node 8 main.py \
    --config kuma.projects.omni.bagel.configs.t2a.exp_0_6b_mmaudio_pretrained \
    --name omni_t2a_0_6b_pretrained
```

W&B: https://wandb.ai/luma-ai/omni-t2a/runs/g0ebocom

**Option B data loader run** (`dongguo/omni-t2a-v2` branch) — no OOM, but step_time spikes:

```bash
cd /fsx/dongguo/Projects/lumaverse/projects/kuma
source .venv/bin/activate

# Config: AudioBatchingDatasetV2 + OmniT2ABridgeDataset (no prefetch thread)
# Dataset: emilia + internal-audio-v1
torchrun --standalone --nproc_per_node 8 main.py \
    --config kuma.projects.omni.bagel.configs.t2a.exp_0_6b_mmaudio_pretrained_v2 \
    --name omni_t2a_0_6b_pretrained_v2
```

W&B: https://wandb.ai/luma-ai/omni-t2a/runs/u27stj53

### Proposed: Test Prefetch Fix (on another 8-GPU node)

Branch `dongguo/omni-t2a-v2` now includes the prefetch thread fix. To verify step_time stabilizes:

```bash
# 1. Clone and set up on the new node
cd /fsx/<username>/Projects
git clone git@github.com:lumalabs/lumaverse.git  # or use existing clone
cd lumaverse
git checkout dongguo/omni-t2a-v2
git pull origin dongguo/omni-t2a-v2

# 2. Activate kuma venv (or create if not present)
cd projects/kuma
source .venv/bin/activate
# If .venv doesn't exist: uv sync

# 3. Source AWS credentials (required for S3 Lance datasets)
source ~/.bashrc

# 4. Launch the run — same config, now uses prefetch thread internally
torchrun --standalone --nproc_per_node 8 main.py \
    --config kuma.projects.omni.bagel.configs.t2a.exp_0_6b_mmaudio_pretrained_v2 \
    --name omni_t2a_0_6b_pretrained_v2_prefetch
```

**What to verify on W&B:**

- `train/step_time`: Should stabilize to ~7-10s without spikes (matching V1 run)
- `train/step_time` variance: Should be low (no 20-80s outliers)
- `train/loss`: Should decrease from ~1.7 toward ~1.4 over 300 steps
- Process memory: Should remain stable (no growth over time)

**If step_time is still spiking**, check whether the bottleneck shifted to `AudioBatchingDatasetV2`:

- Increase `max_workers` in `AudioBatchingDatasetV2.Config` (default 2 → try 4)
- Increase `prefetch_queue_size` in `OmniT2ABridgeDatasetConfig` (default 8 → try 16)

---

## Next Steps

- [x] Monitor Option B overnight run for memory stability — **confirmed no OOM after 300+ steps**
- [x] Strip duration padding in bridge — fixed slicing bug (`[..., :orig_frames]`), num_samples 22→38
- [x] Run Option B with prefetch thread + padding fix — **running as `16gvru69`, 47GB/GPU, ~8s/step, loss 1.67 @ step 10**
- [ ] Run 2B model once dense Qwen3-2B checkpoint is available
- [x] Audio generation inference processor (T2A sampling loop) — implemented in `omni/bagel/inference/`
- [ ] A2T support — audio understanding encoder in understanding stream
- [ ] CoT generation — combined CE + diffusion loss
- [ ] Multi-speaker / interleaved audio input
- [ ] Sequence parallelism (Ulysses) for longer sequences
- [ ] Multi-node Flyte launch config
- [ ] Evaluation metrics (FAD, CLAP score)

---

## Test Script: Prefetch + Padding Strip Fix (2026-04-09)

Run on a fresh 8-GPU node to validate three fixes applied in `dongguo/omni-t2a-v2`:

1. **Prefetch thread** — tokenize+pack in background thread (eliminates data_time spikes)
2. **Padding strip** — truncate to `original_durations` before tokenization (eliminates 2.6x compute overhead)
3. **Sample rate from config** — `audio_sample_rate` read from `AudioBatchingDatasetV2.Config` (correct for both MMAudio 16kHz and DAC VAE 48kHz)

Key commits:

```
99a3379b88 [omni] Read audio_sample_rate from AudioBatchingDatasetV2 config
196c9c0b50 [omni] Strip bucket padding in T2A bridge before tokenization
6a93771617 [omni] Add prefetch thread to T2A bridge dataset for stable step_time
```

### Setup and Launch

```bash
#!/bin/bash
# ============================================================
# Omni T2A PoC — Option B with prefetch + padding strip fix
# Run on an 8-GPU node (A100 80GB)
# Branch: dongguo/omni-t2a-v2
# ============================================================

set -euo pipefail

# 1. Navigate to repo (assumes lumaverse is already cloned at /fsx/<user>/Projects/)
cd /fsx/dongguo/Projects/lumaverse

# 2. Fetch latest and switch to the branch
git fetch origin dongguo/omni-t2a-v2
git checkout dongguo/omni-t2a-v2
git pull origin dongguo/omni-t2a-v2

# 3. Activate kuma venv
cd projects/kuma
source .venv/bin/activate

# 4. Source AWS credentials (required for S3 Lance datasets)
source ~/.bashrc

# 5. Launch 8-GPU training
#    - Config: exp_0_6b_mmaudio_pretrained_v2
#      (AudioBatchingDatasetV2 + bridge with prefetch + padding strip)
#    - Run name includes "v3" to distinguish from previous runs
nohup torchrun --standalone --nproc_per_node 8 main.py \
    --config kuma.projects.omni.bagel.configs.t2a.exp_0_6b_mmaudio_pretrained_v2 \
    --name omni_t2a_0_6b_pretrained_v3 \
    > /tmp/omni_t2a_v3.log 2>&1 &

echo "PID: $!"
echo "Monitor logs: tail -f /tmp/omni_t2a_v3.log"
echo "W&B project: https://wandb.ai/luma-ai/omni-t2a"
```

### What to Verify on W&B

Compare against previous runs:

- V1: https://wandb.ai/luma-ai/omni-t2a/runs/g0ebocom (`omni_t2a_0_6b_pretrained`)
- V2 no prefetch: https://wandb.ai/luma-ai/omni-t2a/runs/u27stj53 (`omni_t2a_0_6b_pretrained_v2`)
- V3 (this run): `omni_t2a_0_6b_pretrained_v3`

**Expected metrics at ~step 100+ (after torch.compile warmup):**

| Metric | V1 (target) | V2 (before fix) | V3 (expected) |
|--------|-------------|-----------------|---------------|
| `train/step_time` | ~7.5s, stable | ~13s, spiky (20-80s outliers) | **~7s, stable** |
| `train/data_time` | ~2.5s, steady | ~0s with 15-20s spikes | **~0s, rare small spikes** |
| `train/num_samples` | ~30 | ~22 | **~35-40** (more real clips per pack) |
| `train/loss` at step 300 | ~1.35 | ~1.5 | **~1.3** (cleaner latents + more samples) |
| Process memory | OOM at ~300 steps | Stable | **Stable** |

**Key things to watch:**

1. **`train/step_time` stability** — Should be flat ~7s with no large spikes. If spikes remain, the bottleneck is in `AudioBatchingDatasetV2` (try increasing `max_workers`).

2. **`train/num_samples`** — Should increase from V2's ~22 to ~35-40. This confirms padding is stripped and the token budget is used for real audio.

3. **`train/loss` convergence rate** — Should track closer to V1's curve (faster decrease), because each step now trains on more real audio clips with clean (uncorrupted) latents.

4. **Process memory** — Should remain stable (no growth). The padding strip does not affect memory stability since `AudioBatchingDatasetV2`'s V2 LanceReader still handles the leak fix.

### Troubleshooting

**If `num_samples` is still ~22 (not increasing):**

The padding strip may not be active. Check that `original_durations` is present in the flat batch:

```bash
# Add temporary debug logging in omni_t2a_bridge_dataset.py:
# In _prefetch_worker(), after getting original_durations:
#   loguru.info(f"original_durations: {original_durations}")
```

**If step_time is still spiking (>15s outliers):**

The `AudioBatchingDatasetV2` internal queues may be draining. Try:

```python
# In t2a.py exp_0_6b_mmaudio_pretrained_v2():
audio_config.max_workers = 4  # default is 2, try 4 or 8
```

**If loss is NaN or diverges:**

The padding strip may expose edge cases with very short clips (e.g., <0.5s → 1-2 audio tokens). Check if `max_per_seq_len` filtering is still working after the strip.

---

## Post-Mortem: Option B Run Crash at Step 1200 (2026-04-09)

Diagnosing run: https://wandb.ai/luma-ai/omni-t2a/runs/u27stj53?nw=nwuserdongguo

Run name: `omni_t2a_0_6b_pretrained_v2` (Option B, AudioBatchingDatasetV2 + bridge, no prefetch thread fix yet)
Branch: `dongguo/omni-t2a-v2`
Log file: `/tmp/omni_t2a_v2.log`

### Grounded (from logs)

- **Last successful step**: Step 1200 at `10:43:26`, diffusion_loss=0.629, num_samples=28
- **Checkpoint saved**: Step 1200 checkpoint uploaded to S3 at `10:43:48`
- **NCCL timeout**: At `11:08:32` (25 minutes after step 1200), the NCCL watchdog detected a collective timeout on all ranks
- **Stuck operation**: `_ALLGATHER_BASE` (SeqNum=110609), the operation immediately after the last completed one (110608). Timeout threshold was 900s (15 min).
- **No Python traceback**: Only C++ NCCL watchdog exceptions. FlightRecorder was disabled (`TORCH_NCCL_TRACE_BUFFER_SIZE` not set), so no stack trace of the stuck collective.
- **All ranks killed**: SIGABRT on all 8 ranks (exitcode -6). NCCL takes down the entire process group when a timeout is detected.
- **Earlier warning sign**: Step 1000→1100 had a 23-minute gap (`10:00:06` → `10:23:31`), but the run recovered from that stall.

### Not known from logs

- **Which rank was the straggler** — the timeout is reported by all ranks; the logs don't identify which rank failed to enter the collective
- **What the straggler was doing** — could have been blocked in data loading, forward pass, backward pass, checkpoint I/O, or a GPU-level hang
- **Whether the 23-min stall at step 1000 and the fatal hang at step 1200 share the same root cause**

### Possible causes (inferred, not proven)

1. **Data loader hang** — `AudioBatchingDatasetV2` or the prefetch thread blocked indefinitely on S3/Lance, preventing the rank from reaching the FSDP all-gather. Plausible because we observed frequent data_time spikes in this run, and the step 1000→1100 stall matches this pattern.
2. **Prefetch thread deadlock** — The background thread could have hit an unhandled exception or a blocking `queue.put(timeout=60)` that silently stalled. The `try/except` in `_prefetch_worker` should catch this, but a GIL deadlock or Rust-level hang in Lance would bypass Python exception handling.
3. **Silent GPU error** — A CUDA error on one rank could cause it to stop issuing NCCL operations. However, there are no CUDA error messages in the logs.
4. **Checkpoint upload blocking training** — The HSDPCheckpointing worker uploads asynchronously, but if S3 upload stalled on one rank, it could potentially block the next training step. The logs show uploads completed at `10:43:48`, but only for ranks 2 and 3 — other ranks' upload status is not visible.
5. **Network/infrastructure issue** — A transient NVLink or network failure on one GPU could cause it to hang on NCCL without any Python-level error.

### Assessment: likely data loading related

Evidence pointing toward data loading:

1. **The 23-min stall at step 1000→1100 recovered** — this matches a long data stall (eventually got data, resumed), not a GPU error (which wouldn't self-recover)
2. **The data_time spike pattern throughout the run** — we already confirmed AudioBatchingDatasetV2 periodically stalls for 15-20s; a longer stall (>15 min) would exceed the NCCL timeout
3. **No CUDA errors in the logs** — rules out GPU hardware failure
4. **`_ALLGATHER_BASE` is an FSDP unshard** — this happens at the start of each forward pass, right after `next(dataloader)` returns. If one rank is stuck waiting for data, the other ranks wait at this all-gather.

What makes it less than certain: the stall happened 25 minutes after step 1200, and we don't have per-rank logs showing the prefetch thread was the one blocked. A Lance Rust-level deadlock (not just slow S3) would look identical from the outside.

**Bottom line**: Data loading is the most likely cause, but the prefetch thread + padding strip fix may not be sufficient if the underlying issue is Lance/S3 hanging indefinitely rather than just being slow. For the next run, adding a watchdog log in the prefetch thread (e.g., log every 60s if no batch produced) would confirm or rule this out quickly.

### What would help diagnose in future runs

- `TORCH_NCCL_TRACE_BUFFER_SIZE=1000` — enables FlightRecorder, captures stack traces at the point of the stuck collective
- Per-rank logging in the prefetch thread (e.g., log every N batches to confirm the thread is alive)
- `dmesg` or kernel logs from the node (GPU Xid errors, OOM kills)

---

## Run Log (2026-04-07 — 2026-04-10)

All runs use 0.6B MoT (Qwen3-0.6B frozen text stream, random audio stream, ~300M trainable params), MMAudio 16k VAE, pure diffusion MSE loss.

Branch: `dongguo/omni-t2a-v2` (commit `084eabdc95 Merge zewang/omni-t2a-koba-v2-adapter`) unless noted.

### Phase 1: Data Loader Development (V1 → Option B → Koba V2)

| W&B ID | Run Name | Config | Data Loader | Dataset | LR | Nodes×GPUs | Where | Status | Notes |
|--------|----------|--------|-------------|---------|-----|------------|-------|--------|-------|
| [g0ebocom](https://wandb.ai/luma-ai/omni-t2a/runs/g0ebocom) | `omni_t2a_0_6b_pretrained` | `exp_0_6b_mmaudio_pretrained` | V1 (AllModalityDataset) | internal-audio-v2-english | 1e-4 | 1×8 | local | Crashed (~300 steps) | **OOM**: Lance S3 memory leak. Step_time ~7.5s, loss 1.7→1.35 |
| [u27stj53](https://wandb.ai/luma-ai/omni-t2a/runs/u27stj53) | `omni_t2a_0_6b_pretrained_v2` | `exp_0_6b_mmaudio_pretrained_v2` | Option B (AudioBatchingDatasetV2 + bridge, no prefetch) | emilia + internal-audio-v1 | 1e-4 | 1×8 | local | Crashed (~1200 steps) | **NCCL timeout**: data stall >15min. No OOM. Step_time ~13s (unstable), loss 1.7→0.6 |
| [32o437te](https://wandb.ai/luma-ai/omni-t2a/runs/32o437te) | `omni_t2a_0_6b_pretrained_v2_prefetch` | `exp_0_6b_mmaudio_pretrained_v2` | Option B + prefetch thread | emilia + internal-audio-v1 | 1e-4 | 1×8 | local | Finished | Prefetch reduced data_time spikes but step_time still ~13s (padding overhead) |
| [16gvru69](https://wandb.ai/luma-ai/omni-t2a/runs/16gvru69) | `omni_t2a_0_6b_pretrained_v3` | `exp_0_6b_mmaudio_pretrained_v2` | Option B + prefetch + padding strip | emilia + internal-audio-v1 | 1e-4 | 1×8 | local | Failed | Short run, details pending |

### Phase 2: Koba V2 Native Packing (Current)

| W&B ID | Run Name | Config | Data Loader | Dataset | LR | Nodes×GPUs | Where | Flyte Job ID | Status | Notes |
|--------|----------|--------|-------------|---------|-----|------------|-------|-------------|--------|-------|
| [fh7smmaz](https://wandb.ai/luma-ai/omni-t2a/runs/fh7smmaz) | `omni_t2a_koba_v2_test` | `exp_..._koba_v2` | Koba V2 (LanceReader + PackingDataset) | internal-audio-v2-english | 1e-4 | 1×8 | local | — | Finished | First koba v2 test |
| [7krtfbsc](https://wandb.ai/luma-ai/omni-t2a/runs/7krtfbsc) | `omni_t2a_koba_v2_compare` | `exp_..._koba_v2` | Koba V2 | internal-audio-v2-english | 1e-4 | 1×8 | local | — | Finished | Local comparison run |
| [9asejgsk](https://wandb.ai/luma-ai/omni-t2a/runs/9asejgsk) | `dongguo/omni_t2a_koba_v2_compare` | `exp_..._koba_v2` | Koba V2 | internal-audio-v2-english | 1e-4 | 1×8 | kiwi-flyte | `dongguo-uo-omni-t2a-koba-v2-compare-kd02b` | Running | Baseline koba v2 on Flyte |
| [5oyhy5dk](https://wandb.ai/luma-ai/omni-t2a/runs/5oyhy5dk) | `dongguo/omni_t2a_formal_koba_v2_gc` | `exp_..._formal_koba_v2` | Koba V2 | internal-audio-v2-english | 1e-4 | 1×8 | kiwi-flyte | `dongguo--omni-t2a-formal-koba-v2-gc-k79b8` | Running | Formal run with GC |

### Phase 3: LR Sweep (2026-04-10)

All use koba v2 packing, internal-audio-v2-english, frozen Qwen3-0.6B text stream.

| W&B ID | Run Name | Config | LR | Nodes×GPUs | Where | Flyte Job ID | Status | Notes |
|--------|----------|--------|----|------------|-------|-------------|--------|-------|
| [redxljkz](https://wandb.ai/luma-ai/omni-t2a/runs/redxljkz) | `dongguo/omni_t2a_koba_v2_lr1e5` | `exp_..._koba_v2_lr1e5` | 1e-5 | 1×8 | local | — | Failed | ExpiredToken (AWS creds) |
| [iwgfgo7c](https://wandb.ai/luma-ai/omni-t2a/runs/iwgfgo7c) | `dongguo/omni_t2a_koba_v2_lr1e5` | `exp_..._koba_v2_lr1e5` | 1e-5 | 1×8 | local | — | Finished | Retry after cred refresh; killed by SIGTERM (pod eviction) |
| [hk15yyf7](https://wandb.ai/luma-ai/omni-t2a/runs/hk15yyf7) | `dongguo/omni_t2a_koba_v2_lr1e6` | `exp_..._koba_v2_lr1e6` | 1e-6 | 1×8 | kiwi-flyte | `dongguo-gguo-omni-t2a-koba-v2-lr1e6-kd44e` | Running | |
| [8m75dgpu](https://wandb.ai/luma-ai/omni-t2a/runs/8m75dgpu) | `dongguo/omni_t2a_koba_v2_lr2e5` | `exp_..._koba_v2_lr2e5` | 2e-5 | 1×8 | kiwi-flyte | `dongguo-gguo-omni-t2a-koba-v2-lr2e5-k4c81` | Running | |

### Multi-Node Formal Runs

| W&B ID | Run Name | Config | LR | Nodes×GPUs | Where | Flyte Job ID | Status | Notes |
|--------|----------|--------|----|------------|-------|-------------|--------|-------|
| [bcyx2a4o](https://wandb.ai/luma-ai/omni-t2a/runs/bcyx2a4o) | `omni_t2a/0_6b_formal_16gpu` | `exp_..._formal` | 1e-4 | 2×8 | kiwi-flyte | `dongguo-omni-t2a-0-6b-formal-16gpu-kb2b4` | Running | 16 GPU formal |
| [ie0zgoca](https://wandb.ai/luma-ai/omni-t2a/runs/ie0zgoca) | `omni_t2a/0_6b_formal_32gpu` | `exp_..._formal` | 1e-4 | 4×8 | kiwi-flyte | `dongguo-omni-t2a-0-6b-formal-32gpu-ka1eb` | Running | 32 GPU formal |

---

## T2A Inference Implementation (2026-04-09)

Implemented standalone T2A inference: text prompt → Euler denoising → MMAudio decode → .wav file.

### Files Created / Modified

| File | Purpose |
|------|---------|
| `omni/bagel/inference/processor/t2a_processor.py` | **NEW** — Inference processor: `get_sequence_plan_t2a()`, `tokenize_sequence_plan_t2a()` |
| `omni/bagel/inference/processor/__init__.py` | **MODIFIED** — Registered `"t2a"` in both factory dicts |
| `omni/bagel/inference/run_t2a_inference.py` | **NEW** — Standalone inference script with Euler sampler |
| `omni/bagel/configs/t2a.py` | **MODIFIED** — Added `get_t2a_inference_params()` for centralized inference config |

### Architecture

```
Text prompt
  → get_sequence_plan_t2a()        [TEXT(prompt), AUDIO(placeholder)]
  → tokenize_sequence_plan_t2a()   OmniElementAudio + OmniQwen3Tokenizer + MRoPE(THW)
  → pack_sequence()                Packed batch dict (same format as training)
  → euler_sample_t2a()             Euler denoising loop (rectified flow + sigma shift)
  → PretrainedMMAudioDecoder       VAE decode (latents → mel) + BigVGAN vocoder (mel → waveform)
  → torchaudio.save()              .wav file
```

### Inference Processor (`t2a_processor.py`)

Follows the same pattern as `t2i_processor.py`:

- `get_sequence_plan_t2a(text, duration_sec, sample_rate)` — Creates `[TEXT, AUDIO]` sequence elements. Audio element uses a dummy tensor sized to produce the correct token count via `OmniElementAudio`.
- `tokenize_sequence_plan_t2a(plan, compression_factor)` — Tokenizes text (Qwen3) and audio (placeholder `<|image_pad|>` tokens with `noise` attention mode, `vae_token_mask=True`).
- Uses `order="THW"` for MRoPE (1D temporal for audio), matching training.

### Sigma Shift = Schedule Shift (no difference between Ray and Omni)

The `BagelT2ALoss` training code applies a sigma shift as a post-hoc rescaling:

```python
sigma_shifted = (S * sigma) / (1 + (S - 1) * sigma)   # S = sigma_shift = 3.0
```

This looks different from `RectifiedFlowsSchedule(shift=...)` which adds a constant in logsnr space. But they are **exactly the same transformation**:

```
sigma_shifted / (1 - sigma_shifted)
  = [S * sigma / (1 + (S-1)*sigma)] / [(1 - sigma) / (1 + (S-1)*sigma)]
  = S * sigma / (1 - sigma)

logit(sigma_shifted) = log(S) + logit(sigma)

Since logsnr = -2 * logit(sigma):
  logsnr_shifted = -2 * logit(sigma_shifted)
                 = -2 * (log(S) + logit(sigma))
                 = logsnr - 2*log(S)
```

So **`sigma_shift=S` in BagelT2ALoss is identical to `schedule.shift = -2*log(S)` in RectifiedFlowsSchedule**. For `S=3`: `shift = -2*log(3) ≈ -2.197`. This is the same parameterization Ray T2A uses (their configs use `-2*log(3)`, `-2*log(5)`, etc. depending on the training noise schedule).

There is no fundamental difference in the sampler between Ray T2A and Omni T2A. The only difference is which shift value matches each model's training config. Both use the same diffusion infrastructure.

### Sampler (`sample_t2a`)

Uses the same diffusion infrastructure as Ray T2A (`ursa.models.omni.inference.diffusion`):

- `RectifiedFlowsSchedule(shift=-2*log(3))` — matches training `sigma_shift=3.0`
- `DDPM(eta=0)` — deterministic DDIM one-step function
- `dm.sample_simple()` — standard sampling loop

The model_fn wrapper (`make_t2a_model_fn`) adapts the omni MoT calling convention for `dm.sample_simple`:

```python
# dm.sample_simple calls: model_fn(x, logsnr, it, num_steps, logsnr_next)
# Wrapper converts logsnr → sigma via NoiseCond, calls omni model, applies CFG
noise_cond = NoiseCond()(logsnr)                           # logsnr → sigma
xs = (x, text_ids)                                         # (audio_latents, text_token_ids)
ts = (noise_cond, None)                                    # (sigma, None)
(_, audio_out, *_), *_ = model(xs, ts, **batch_kwargs)     # velocity prediction
# CFG: v = v_uncond + cfg * (v_cond - v_uncond)
```

### CFG Support

When `--cfg > 0`, packs two samples into one batch:

- Sample 0 (conditioned): `[text_prompt | audio_placeholder]`
- Sample 1 (unconditioned): `[empty_text | audio_placeholder]`

Both share the same initial noise. FlexAttention block mask isolates the samples. After model forward:

```python
v_cond   = audio_out[0:1]   # conditioned velocity
v_uncond = audio_out[1:2]   # unconditioned velocity
v_cfg    = v_uncond + cfg * (v_cond - v_uncond)
```

This is simpler than Ray T2A's separate `CFG` class because the omni model already handles multi-sample packing natively.

### Model Calling Convention

Matches training exactly — the model (`Qwen3MMDiT`) receives:

- `xs = (audio_latents, text_ids)` where `audio_latents` is `(B, T, C)` (time-major) and `text_ids` is `batch["txt"][batch["text_token_mask"]]` (flat packed)
- `ts = (noise_cond, None)` where `noise_cond` is `(B, 1)` sigma values
- `**kwargs` = `position_ids, text_token_mask, vit_token_mask, vae_token_mask, sample_lens, split_lens, attn_modes, modulation_mask`

The preprocessor (`Qwen3TextAudioPackedPreprocess`) handles embedding, RoPE, and attention mask construction. The postprocessor (`FinalVidLayerPacked`) projects audio hidden states back to latent dim.

### Centralized Inference Config (`get_t2a_inference_params`)

All inference-specific constants live in one place in `configs/t2a.py`:

```python
params = get_t2a_inference_params(model_size="0.6b", vae_mode="mmaudio_16k")
# Returns:
#   model_config: Qwen3MMDiT.Config (0.6B, 28 layers, hidden=1024)
#   audio_latent_dim: 20
#   hop_length: 512
#   sample_rate: 16_000
#   latent_scaling_factor: 1/2.3563
#   schedule_shift: -2*log(3) ≈ -2.197 (same parameterization as Ray T2A)
#   default_num_steps: 50
#   default_cfg: 3.5
#   default_eta: 0.0
```

### Comparison with Ray T2A Inference

| Aspect | Ray T2A | Omni T2A | Different? |
|--------|---------|----------|------------|
| **Sampler infrastructure** | `dm.sample_simple()` + `DDPM(eta=0)` + `RectifiedFlowsSchedule` | **Same** | No |
| **Schedule shift** | `-2*log(5)`, `-2*log(3)`, or `0` (per training config) | `-2*log(3)` (matches training `sigma_shift=3.0`) | No — same mechanism, value depends on training |
| **Audio decoder** | `PretrainedMMAudioDecoder` | Same | No |
| **Text conditioning** | UMT5-XXL cross-attention | Packed joint attention (frozen Qwen3) | Yes — architecture |
| **CFG** | Separate `CFG` class, doubles batch externally | Packed samples (cond+uncond), split output | Yes — implementation, not math |
| **Comparey** | `DashboardLogger` with `field_audio` | Not yet integrated | Yes — TODO |

### Usage

```bash
cd projects/kuma && source .venv/bin/activate

# Smoke test (random weights, no checkpoint)
CUDA_VISIBLE_DEVICES=0 python kuma/projects/omni/bagel/inference/run_t2a_inference.py \
    --prompt "A person speaking clearly in English" \
    --duration 5.0 --num-steps 50 --output-dir ./t2a_output

# With checkpoint + CFG
CUDA_VISIBLE_DEVICES=0 python kuma/projects/omni/bagel/inference/run_t2a_inference.py \
    --prompt "A person speaking clearly in English" \
    --checkpoint-path osc://dongguo/omni_t2a_0_6b/step_01200.pt \
    --duration 5.0 --num-steps 50 --cfg 3.5 --output-dir ./t2a_output

# Skip audio decode (save latents only, faster for debugging)
CUDA_VISIBLE_DEVICES=0 python kuma/projects/omni/bagel/inference/run_t2a_inference.py \
    --prompt "Birds chirping in a forest" \
    --skip-decode --output-dir ./t2a_output
```

### Future: Comparey Integration

To post results to Comparey (like Ray T2A's `t2a_eval.py`), wrap the inference in an eval job with:

```python
dashboard_logger = DashboardLogger.Config()
dashboard_logger.field_audio = "audio"
dashboard_logger.field_audio_sample_rate = "sample_rate"
dashboard_logger.slack_channel = "#omni-t2a-evals"
```

This requires converting `run_t2a_inference.py` into an `Inference` subclass that yields `{"audio": waveform, "sample_rate": 16000, "txt": prompt}` dicts. The `DashboardLogger` handles S3 upload, Comparey experiment creation, and Slack notifications.

---

## Formal Multi-Node Training (2026-04-10)

### Goal

Launch a proper multi-node training run on Flyte, matching the Ray T2A reference config (`luma-ai/t2a/runs/ovah4wwn`, `stage001_internal_audio_v2`) as closely as possible.

### Reference Run

W&B: `luma-ai/t2a/runs/ovah4wwn` (`ray3_t2a_mmaudio/stage001_internal_audio_v2`)
PR: lumalabs/lumaverse#6888 (dataset config reorg, merged 2026-04-03)

### Config: `exp_0_6b_mmaudio_formal`

File: `omni/bagel/configs/t2a.py`

All hyperparameters match the reference except model size and text conditioning:

| Parameter | Reference (2.9B Ray3) | This run (0.6B Omni) |
|-----------|----------------------|----------------------|
| **Noise schedule** | `SampleLogSNRGeneric(shift=-1.6)`, truncated normal, scale=1.0 | Same |
| **sigma_shift** | 3.0 | Same |
| **Optimizer** | AdamW, lr=1e-4, wd=0.01, betas=(0.9, 0.95) | Same |
| **LR warmup** | 5,000 steps | Same |
| **CFG dropout** | 0.1 | Same |
| **Dataset** | internal-audio-v2 | Same |
| **Duration buckets** | [5.04, 10.04, 15.04] | Same |
| **Peak normalize** | headroom=1.1, clamp=True | Same |
| **Latent scaling** | 1/2.3563 (MMAudio 16k) | Same |
| **Save every** | 5,000 | Same |
| **max_num_tokens** | N/A (Ray uses fixed batch) | 10,000 |
| **Model** | 2.9B Ray3 DiT, all trainable | 0.6B Omni MoT, ~300M trainable |
| **Text conditioning** | UMT5-XXL cross-attention | Frozen Qwen3-0.6B packed attention |

### Node Count Recommendation

| | Reference (2.9B) | Ours (0.6B) |
|---|---|---|
| **Total params** | 2.9B | 0.6B (~5x smaller) |
| **Trainable params** | 2.9B (all) | ~300M (audio stream only, text frozen) |
| **GPUs** | 32 (4 nodes) | 16 (2 nodes) or 32 (4 nodes) |

With only ~300M trainable params and `max_num_tokens=10000`, 2 nodes (16 GPUs) is sufficient. 4 nodes (32 GPUs) doubles batch size (more audio minutes per step), useful for comparing batch size scaling.

### Launch Commands

```bash
cd /fsx/dongguo/Projects/lumaverse/projects/kuma
source .venv/bin/activate
flytecli activate --cluster kiwi-flyte

# 2 nodes (16 GPUs) — baseline
python flyte_main.py \
    --config kuma.projects.omni.bagel.configs.t2a.exp_0_6b_mmaudio_formal \
    --name omni_t2a/0_6b_formal_16gpu \
    --nodes 2 \
    --gpus-per-node 8 \
    --wandb --influxdb --kuma-profiler \
    --cluster kiwi-flyte \
    --username_override dongguo

# 4 nodes (32 GPUs) — 2x batch size
python flyte_main.py \
    --config kuma.projects.omni.bagel.configs.t2a.exp_0_6b_mmaudio_formal \
    --name omni_t2a/0_6b_formal_32gpu \
    --nodes 4 \
    --gpus-per-node 8 \
    --wandb --influxdb --kuma-profiler \
    --cluster kiwi-flyte \
    --username_override dongguo
```

Naming convention: `omni_t2a/0_6b_formal_{num_gpus}gpu` — encodes total GPU count so W&B runs and checkpoint paths don't collide. The 32gpu run has ~2x audio minutes per step; comparing loss curves at the same step count shows the effect of batch size scaling.

### What to Monitor on W&B

Project: `luma-ai/omni-t2a`

| Metric | Expected (16gpu) | Expected (32gpu) |
|--------|------------------|-------------------|
| `train/step_time` | ~7-10s (stable after compile warmup) | ~7-10s (same, compute-bound) |
| `train/loss` at step 5K | ~1.0-1.2 | ~0.9-1.1 (faster convergence from larger batch) |
| `train/num_samples` | ~35-40 per rank | ~35-40 per rank (same per-GPU) |
| Process memory | Stable ~30-50 GB/GPU | Same |
| Audio minutes per step | ~56 min (16 GPUs × 3.5 min/GPU) | ~112 min (32 GPUs × 3.5 min/GPU) |

---

## Comparey Inference Evaluation

### Context

The Ray3 T2A team (inkyushin) already has a working eval pipeline that generates audio samples from checkpoints and uploads them to the **comparey dashboard** for side-by-side listening. Example experiment:

- **Link:** `ki://ai-lumalabs-dashboard-samples-ap-se-2/dashboard_samples/by_id/69d8358fa54207dee464b612/`
- **Experiment:** `ablation_wd_mmaudio_5s_eval_158-exp=ray3_t2a_mmaudio_ablation_wd`
- **Checkpoint:** 155,000 (inkyushin's weight decay ablation)
- **Contents:** 30 WAV files (~316 KB each, 5s duration, 16kHz) + `index.json` with prompts and generation times
- **Prompts:** Diverse eval set — news, narration, announcements, poetry, emotional, technical — using `[SPEAKER_00]"..."` format

### Goal

Support the same comparey inference flow for omni T2A models with two modes:

1. **During training** — automatically run inference with the latest checkpoint (continuous eval)
2. **Post-hoc** — provide a list of already-trained checkpoints and run inference on all of them

### Existing Infrastructure (Ray3 T2A pattern)

Three layers in the Ray3 pipeline:

| Layer | File | Role |
|-------|------|------|
| `InferJob` | `lib/ursa/ursa/infer_job.py` | Orchestrates checkpoint loading + eval loop |
| `DashboardLogger` | `lib/ursa/ursa/infer_job.py:664` | Creates comparey experiments, uploads WAVs, posts to Slack |
| `basic_t2a_eval_job()` | `projects/kuma/kuma/projects/ray3_t2av/inference/t2a_eval.py` | Config factory: wires training configs → InferJob + DashboardLogger |

**`InferJob.run()` execution modes:**

- `continuous_eval()` — polls for latest training checkpoint, runs eval when new one appears
- `evaluate_specific_checkpoints()` — evaluates a given list of step numbers

**`DashboardLogger` flow:**

1. `initialize(step, name)` → creates `comparey.Experiment(exp_name, checkpoint=step)`
2. `add(data)` → extracts audio from inference results, saves WAV, calls `comparey_exp.add_media_file()`
3. `evaluate()` → returns metrics, logs dashboard URL

**Dashboard URL format:** `https://internal-dashboard.sandbox.labs.lumalabs.ai/dashboards/compare-samples?exp1={slug}&cp1={step:05d}`

### Problem

`basic_t2a_eval_job()` is tightly coupled to the Ray3 architecture (`Ray3DenoiserForgeWrapper`, `T2ATask`, `GenerativeTaskInferencer`, Forge model managers). The omni T2A model uses a completely different pipeline (Qwen3MMDiT with packed self-attention, custom batch building via `t2a_processor.py`).

### Design: Plug Omni T2A into InferJob

Reuse the `InferJob` framework (gets continuous_eval, checkpoint_steps, DashboardLogger, Slack for free). Three new files needed:

#### 1. `OmniT2AInferencer` — adapts omni inference to the InferJob protocol

File: `projects/kuma/kuma/projects/omni/bagel/inference/t2a_inferencer.py`

- Wraps existing functions from `run_t2a_inference.py`: `load_t2a_model()`, `load_audio_decoder()`, `build_t2a_batch()`, `sample_t2a()`
- Implements inferencer protocol: `setup_models()`, `setup_datasets()`, `inference_step()`
- `setup_models()` → builds Qwen3MMDiT + MMAudio decoder
- `inference_step()` → for each prompt: build batch → sample → decode → return `{"audio": waveform, "sample_rate": sr, "txt": prompt, "total_time_taken": elapsed}`
- Checkpoint loading via direct `load_state_dict()` (no Forge model manager)

#### 2. `basic_omni_t2a_eval_job()` — config factory

File: `projects/kuma/kuma/projects/omni/bagel/inference/t2a_eval.py`

- Analogous to Ray3's `basic_t2a_eval_job()` but for omni
- Creates `InferJob.Config` with DashboardLogger (audio field, Slack channel), OmniT2AInferencer, prompt dataset

#### 3. Eval config functions — launchable via `flyte_main.py`

File: `projects/kuma/kuma/projects/omni/bagel/configs/t2a_eval.py`

```python
def omni_t2a_mmaudio_5s_eval() -> InferJob.Config:
    """Continuous eval — polls latest checkpoint during training."""
    return basic_omni_t2a_eval_job(
        model_config=qwen3_0_6B_t2a_mmaudio(),
        duration_sec=5.0,
        log_prefix="omni_t2a_mmaudio_5s_eval",
    )

def omni_t2a_mmaudio_5s_eval_steps() -> InferJob.Config:
    """Evaluate specific checkpoints."""
    job = omni_t2a_mmaudio_5s_eval()
    job.checkpoint_steps = [50000, 100000, 150000, 200000]
    return job
```

### Launch Commands

```bash
cd projects/kuma && source .venv/bin/activate
flytecli activate --cluster kiwi-flyte

# Mode 1: Continuous eval (polls latest checkpoint during training)
python flyte_main.py \
    --config kuma.projects.omni.bagel.configs.t2a_eval.omni_t2a_mmaudio_5s_eval \
    --name dongguo/omni_t2a_experiment \
    --username_override dongguo \
    --nodes 1 --cluster kiwi-flyte

# Mode 2: Evaluate specific checkpoints
python flyte_main.py \
    --config kuma.projects.omni.bagel.configs.t2a_eval.omni_t2a_mmaudio_5s_eval_steps \
    --name dongguo/omni_t2a_experiment \
    --username_override dongguo \
    --nodes 1 --cluster kiwi-flyte
```

### Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Framework | Reuse `InferJob` | Free continuous_eval, checkpoint_steps, DashboardLogger, Slack |
| Model manager | Skip Forge, direct `load_state_dict` | Omni T2A doesn't use Forge's model manager pattern |
| Prompt dataset | Reuse `PromptTxtDataset` | Same format as Ray3 T2A prompts, already works |
| Batch size | 1 (sequential prompts) | Omni T2A packs CFG samples along T, not batch dim |

### Open Questions

- **Prompt set:** Reuse Ray3 T2A prompts (`[SPEAKER_00]"..."` format) or create new prompt format for omni T2A?
- **Checkpoint path convention:** Are omni T2A checkpoints saved via Forge (standard `get_checkpoint_path` with username/name/step), or custom path?

## Dataloader Refinement Plan (2026-04-10)

Context: Ze Wang's PR #7209 provides a koba v2 native T2A packing path (`OmniT2APackingConfigV2`) as a cleaner replacement for our `OmniT2ABridgeDataset`. The current bridge path has a known memory leak (from `AllModalityDatasetWithMultithreading`) that likely causes dataloader crashes after a few hundred/thousand steps.

### P0 — Test Ze's v2 path on his PR branch directly ✅ DONE (2026-04-10)

- [x] Check out Ze's PR #7209 branch (`zewang/omni-t2a-koba-v2-adapter`)
- [x] Run `debug_local_koba_v2` config on single GPU to verify basic functionality
- [ ] Run `exp_0_6b_mmaudio_pretrained_koba_v2` on multi-node (skipped — 8-GPU local OOM'd due to 240GB cgroup limit; single-GPU was sufficient to validate)
- [ ] Compare loss/grad_norm curves against our existing proof-of-concept run (deferred to multi-node run)

**Result**: Koba v2 dataloader is stable. 37,300+ steps over 5h15m with 0 errors, 0 filtered items, no crashes. Loss stable ~0.65-0.79, grad_norm healthy 0.5-1.2, consistent step_time 0.225s. WandB: https://wandb.ai/luma-ai/omni-t2a/runs/fh7smmaz

**Note**: `exp_0_6b_mmaudio_pretrained_koba_v2` OOM'd on local 8-GPU pod (240GB cgroup, 8 Lance readers exceeded 95% threshold). This is a pod resource issue, not a dataloader bug. Multi-node Flyte test needed for full validation.

### P1 — Run equivalence tests from PR #7209 ✅ DONE (2026-04-10)

- [x] Run `test_t2a_pipeline_equivalence.py` — ALL 3 TESTS PASSED
  - Test 1: Tokenized plans are element-by-element identical (5/5 samples match)
  - Test 2: Packed batches are bit-identical (seq_len=4000, 6 samples)
  - Test 3: Both paths produce identical model behavior (same error on synthetic data)
- [x] Run `compare_t2a_dataloaders.py` — All structural checks PASSED
  - All 3 paths (A: original, B: bridge, C: koba v2) produce 5 valid batches each
  - Path A: 10000 tokens, 8814 audio tokens
  - Path B: 10000 tokens, 7194 audio tokens
  - Path C: 10000 tokens, 8442 audio tokens
  - All keys, masks, position IDs, attention modes structurally valid

**Conclusion**: Switching to koba v2 path will NOT silently change training behavior. Output is bit-identical when given the same input.

### P2 — Coordinate with Ze on merge order ✅ DONE (2026-04-10)

- [x] Merged Ze's branch (`zewang/omni-t2a-koba-v2-adapter`) into `dongguo/omni-t2a-v2` locally
- [x] Resolved 1 conflict in `omni_t2a_bridge_dataset.py` (formatting + indexing style — took Ze's version)
- [ ] Push to remote and get PR #7206 through CI + review to merge to main

Approach: Ze's PR #7209 targeted our branch, so we merged it in. PR #7206 now contains both our T2A code and Ze's v2 dataloader — one combined merge to main.

### P3 (low) — Final-pack drop bug fix

- Only matters if we continue using the bridge path long-term
- If we switch to v2, this becomes moot
- The bug only loses the last partial batch per epoch — it is NOT the cause of training crashes

---

## Gradient Explosion Diagnosis (2026-04-11)

### Symptom

Every omni T2A training run follows the same pattern regardless of learning rate or gradient clipping:

1. **Phase 1 (healthy):** Loss decreases, grad norms are stable (2–3), model is learning
2. **Phase 2 (onset):** Grad norms start increasing monotonically
3. **Phase 3 (divergence):** Grad norms reach 10,000–80,000+, loss spikes and becomes unstable

The transition occurs at different steps depending on LR, but the pattern is universal.

### Experiment Summary

W&B project: `luma-ai/omni-t2a`

| Run ID | Config | LR | Grad Clip | Grad Norm Bottom | Step at Bottom | Final Grad Norm |
|--------|--------|-----|-----------|-----------------|----------------|-----------------|
| `bcyx2a4o` | `exp_0_6b_mmaudio_formal` | 1e-4 | No | ~2.0 | ~500 | 77,567 |
| `5oyhy5dk` | `exp_0_6b_mmaudio_formal_koba_v2` | 1e-4 | Yes (1.0) | ~2.0 | ~500 | 7,885 (clipped) |
| `9asejgsk` | `exp_0_6b_mmaudio_pretrained_koba_v2` | 1e-4 | No | ~1.7 | ~400 | 21,834 |
| `8m75dgpu` | `..._koba_v2_lr2e5` | 2e-5 | No | 0.84 | ~1,150 | 1.14 (rising) |
| `hk15yyf7` | `..._koba_v2_lr1e6` | 1e-6 | No | 2.80 | — | Barely learning |

Phase-by-phase for the lr=1e-4 no-clip run (`bcyx2a4o`):

| Phase | Steps | Loss | Grad Norm | LR |
|-------|-------|------|-----------|-----|
| Warmup | 0–500 | 1.63→0.77 | 2–3 (stable) | 0→10e-6 |
| Onset | 500–800 | 0.76 (flat) | 2→5 | 10e-6→13e-6 |
| Escalation | 800–1200 | 0.75 | 3→44 | ~20e-6 |
| Runaway | 1200–2000 | 0.79 | 26→2,036 | ~32e-6 |
| Diverging | 2000–3000 | 1.55 | 767→42,514 | ~50e-6 |
| Blown up | 3000+ | 3.33 | 9,234→77,567 | ~73e-6 |

### Comparison with Ray3 T2A Reference

The Ray3 reference run (`luma-ai/t2a/ovah4wwn`, `stage001_internal_audio_v2`) uses the **same LR schedule** (5K warmup to 1e-4) and has monotonically **decreasing** grad norms:

| Phase | Ray3 Reference (2.9B, healthy) | Omni T2A (0.6B, explodes) |
|-------|-------------------------------|--------------------------|
| 0–500 | grad_norm 5.7 → decreasing | grad_norm 2.9 → stable |
| 500–1K | grad_norm 2.6 | grad_norm 2.0–5.0 (onset) |
| 1K–2K | grad_norm **1.0** | grad_norm **14→582** |
| 2K–5K | grad_norm **0.4→0.25** | grad_norm **5,000→30,000** |

Key config differences between Ray3 and omni:

| Setting | Ray3 T2A (`main_mmaudio.py`) | Omni T2A (`t2a.py`) |
|---------|------------------------------|---------------------|
| Model | 2.9B Ray3 DiT | 0.6B Qwen3 MoT (~300M trainable) |
| Text conditioner | UMT5-XXL (**trainable**) | Frozen Qwen3-0.6B text stream |
| Architecture | Cross-attention to UMT5 | Packed self-attention (shared QKV) |
| Modulation | adaLN on all params (both streams trainable) | adaLN only on audio stream, text has none |
| LR warmup | **1,000 steps** | **5,000 steps** |
| Weight decay | 1e-5 | 0.01 |
| Noise shift | -2\*log(5) ≈ -3.22 | -1.6 |
| **softcap_cap** | N/A (different arch) | **None (disabled)** |

### Root Cause: Modulation Parameter Feedback Loop

The problem is **structural**, not a hyperparameter issue. Lowering LR only delays the onset; the trajectory is the same.

#### Why the model learns well first, then fails

All modulation parameters (`shift`, `scale`, `gate`) are initialized to **exactly zero** (`nn.init.zeros_` in `mmdit_block.py`). At init:

- `modulate(x, shift=0, scale=0)` → `x * (1 + 0) + 0 = x` (identity)
- `apply_gate(x, gate=0)` → `x * 0 = 0` (kills residual branch)

So early in training, the audio stream behaves like a **plain transformer with no diffusion conditioning and no residual MLP**. This is a well-behaved regime — gradients flow cleanly, loss drops.

To learn diffusion, the model **must** learn non-zero modulation values to condition on the noise timestep. The moment these grow away from zero, the feedback loop begins:

#### The three amplification mechanisms

**1. Unbounded modulation compounds across depth (`blocks_pre_post.py:77-101`)**

In every audio-stream layer:

```python
x_modulated = norm1(x)                           # unit variance
x_modulated = x * (1 + scale) + shift            # modulate() — lib/ursa/ursa/models/utils.py:176
x_modulated = x_modulated * norm_scale            # fixed constant
x_qkv = attn_qkv(x_modulated)                    # linear projection
```

The gradient of the loss w.r.t. `scale` is `dL/d(scale) = dL/d(x_modulated) * norm1(x)` — proportional to the activation magnitude. As `scale > 0`, `x_modulated` is larger, which feeds into the next layer, making its `dL/d(scale)` larger.

Across 28 layers, this compounds exponentially:

- If average `|1 + scale|` is 1.05/layer → output amplified `1.05^28 = 3.9x`
- If average `|1 + scale|` is 1.10/layer → output amplified `1.10^28 = 14.4x`
- If average `|1 + scale|` is 1.15/layer → output amplified `1.15^28 = 39.2x`

**2. No normalization before final output (`model.py:1151-1169`)**

`FinalVidLayerPacked.forward()`:

```python
shift, scale = self.adaLN_modulation(self.activation(vec)).chunk(2, dim=-1)
x = modulate(self.norm_final(x), shift=shift, scale=scale)  # norm BEFORE modulate
x = self.linear(x)  # directly into loss — NO norm after modulate
```

`norm_final` normalizes to unit variance *before* modulation. After applying `x * (1 + scale) + shift`, the scale change flows directly into the MSE loss with no buffer. If final scale=2.0, output is ~3x, MSE is ~9x, gradient is ~3x.

**3. Gated residuals compound in backward pass (`blocks_pre_post.py:216-249`)**

```python
x = x + apply_gate(self.attn_proj(attn), gate=mod_params[2])     # gate = x * gate
x = x + apply_gate(self.mlp(modulate(norm2(x), shift, scale)), gate=mod_params[5])
```

Each layer has 2 gates (attention + MLP). Across 28 layers = 56 multiplicative factors in the backward pass. Each gate's gradient is proportional to the attention/MLP output, which is influenced by all previous layers' gates.

#### Why Ray3 doesn't have this problem

Ray3's UMT5-XXL text conditioner is **trainable** — it co-adapts with the denoiser. If activations grow, the text conditioner can adjust its output downward. In omni, the frozen text outputs are fixed, so the audio stream has no relief valve. 100% of gradients funnel into the audio stream parameters.

### The Fix: `softcap_cap` (Already Used by T2I)

The omni T2I (text-to-image) model had the **exact same problem** and solved it. The `softcap` mechanism already exists in the codebase:

```python
# lib/ursa/ursa/math/numeric.py:71
def softcap(x, cap=1.0):
    return cap * tanh(x / cap)   # bounds output to [-cap, cap]
```

Both `modulate()` and `apply_gate()` in `lib/ursa/ursa/models/utils.py` already accept `softcap_cap` and call `softcap()` when it's not None. The T2A configs simply have it set to `None` (disabled).

#### T2I production configs all enable softcap

Every T2I training config enables softcap on the generation stream (stream[1]):

```python
# scaling_vl.py:77-78 (stages < 5.0)
job.trainer.denoiser.module.blocks[0].streams_pre[1].softcap_cap = 1.0
job.trainer.denoiser.module.blocks[0].streams_post[1].softcap_cap = 1.0

# scaling_vl.py:446-448 (stages >= 5.0) — added postprocess
job.trainer.denoiser.module.blocks[0].streams_pre[1].softcap_cap = 1.0
job.trainer.denoiser.module.blocks[0].streams_post[1].softcap_cap = 1.0
job.trainer.denoiser.module.postprocess.out_projs[1].softcap_cap = 1.0

# scaling_vl.py:1054-1056 (stages >= 6.2) — relaxed to 10.0
job.trainer.denoiser.module.blocks[0].streams_pre[1].softcap_cap = 10.0
job.trainer.denoiser.module.blocks[0].streams_post[1].softcap_cap = 10.0
job.trainer.denoiser.module.postprocess.out_projs[1].softcap_cap = 10.0
```

Consistent across `scaling_vl.py`, `scaling_8b.py`, `scaling.py`, `scaling_vl_dmd.py`, `scaling_vl_cfg_distill.py`, `scaling_vl_tdm.py`, `moe_experiments.py`. Zero exceptions.

#### What softcap does at each location

| Location | What it bounds | Which amplifier it prevents |
|----------|---------------|----------------------------|
| `streams_pre[1].softcap_cap` | `shift` and `scale` in pre-attention `modulate()` | #1 — unbounded modulation |
| `streams_post[1].softcap_cap` | `gate` in `apply_gate()`, `shift`/`scale` in MLP | #3 — gated residual compounding |
| `postprocess.out_projs[1].softcap_cap` | `shift` and `scale` in `FinalVidLayerPacked` | #2 — unguarded final output |

#### T2I evolution: 1.0 → 10.0

The T2I team started with `softcap_cap=1.0` (tight bound) for training stability, then relaxed to `10.0` at stage 6.2. This suggests tight bounds are needed for initial training; they can be loosened once the model is well-trained.

### Proposed Fix for T2A

Apply the same 3-line config change to all T2A configs:

```python
# In get_t2a_baseline() or per-experiment config:
job.trainer.denoiser.module.blocks[0].streams_pre[1].softcap_cap = 1.0
job.trainer.denoiser.module.blocks[0].streams_post[1].softcap_cap = 1.0
job.trainer.denoiser.module.postprocess.out_projs[1].softcap_cap = 1.0
```

No architecture changes needed — the infrastructure is fully wired, just disabled via `None` default.

### Additional Stabilization (Lower Priority)

These are secondary to softcap but worth noting:

- **Lower LR for smaller model:** Ray3 reference uses lr=1e-4 for 2.9B params. Omni has ~300M trainable. lr=2e-5 to 5e-5 may be more appropriate.
- **Shorter warmup:** Ray3 reference uses 1,000 steps vs omni's 5,000. Shorter warmup reaches stable LR sooner.
- **Higher weight decay:** T2I uses 0.1; current T2A configs use 0.01. Higher weight decay provides additional regularization on modulation parameters.
- **Per-layer gradient norm monitoring:** Diagnostic only — would confirm which layers diverge first (deep layers from compounding, or final layer from unguarded output).

### Softcap Experiment Runs (2026-04-11)

New config: `exp_0_6b_mmaudio_koba_v2_softcap` in `projects/kuma/kuma/projects/omni/bagel/configs/t2a.py`

Changes vs `exp_0_6b_mmaudio_pretrained_koba_v2`:

- `softcap_cap=1.0` on `streams_pre[1]`, `streams_post[1]`, `postprocess.out_projs[1]`
- `lr=2e-5` (down from 1e-4)
- Warmup: 1,000 steps (down from 5,000, matches Ray3 reference)
- `grad_clip=1.0` as safety net

Helper function `_apply_softcap(job, cap)` added for reuse across configs.

#### Run 1: Local 8-GPU (softcap validation)

- **Goal:** Quick validation that softcap stabilizes training. Compare grad norm trajectory against the no-softcap `lr2e5` run (`8m75dgpu`) which showed grad norms starting to rise at step ~1,200.
- **Config:** `exp_0_6b_mmaudio_koba_v2_softcap`
- **Launch:** `torchrun --standalone --nproc_per_node 8 main.py --config kuma.projects.omni.bagel.configs.t2a.exp_0_6b_mmaudio_koba_v2_softcap --name dongguo/omni_t2a_softcap_local`
- **Nodes:** 1 (8 GPUs, local interactive pod)
- **What to watch:** Grad norm should stay bounded (< 3.0) even past step 1,200 where the no-softcap run started diverging. If softcap works, grad norms should follow the Ray3 reference pattern (monotonically decreasing after warmup).

#### Run 2: Kiwi 4-node (softcap at scale)

- **Goal:** Full-scale training with softcap fix. This is the first production-quality run that should train stably to convergence.
- **Config:** `exp_0_6b_mmaudio_koba_v2_softcap`
- **Job ID:** `dongguo-gguo-omni-t2a-softcap-4node-kcfb7`
- **Launch:** `python flyte_main.py --config kuma.projects.omni.bagel.configs.t2a.exp_0_6b_mmaudio_koba_v2_softcap --name dongguo/omni_t2a_softcap_4node --nodes 4 --gpus-per-node 8 --wandb --influxdb --kuma-profiler --cluster kiwi-flyte --username_override dongguo`
- **Nodes:** 4 (32 GPUs)
- **What to watch:** Same as local run, plus loss convergence at scale. Target: loss < 0.7 by step 5K with stable grad norms.

#### Stopped Run

- **Job:** `dongguo--omni-t2a-formal-koba-v2-gc-k79b8` (W&B: `5oyhy5dk`)
- **Config:** `exp_0_6b_mmaudio_formal_koba_v2` (lr=1e-4, grad_clip=1.0, NO softcap)
- **Reason stopped:** Grad norms at 5,000–8,000 (pre-clip) after 2,400 steps. Grad clip was firing every step — model was not learning meaningfully. Replaced by softcap run.
