# MASTER

## Project Index

| Project | Owner | Status | Priority | Last Updated |
|---|---|---|---|---|
| [Audio Data Notes](projects/audio_data_notes.md) | Dong | 🟡 In Progress | Medium | 2026-03-19 |
| [Audio Data Processing Logs](projects/audio_data_processing_logs.md) | Dong | 🟡 In Progress | Medium | 2026-03-18 |
| [Audio Processing Notes](projects/audio_processing_notes.md) | Dong | 🟢 Done | Low | 2026-03-18 |
| [Data Infrastructure Notes](projects/data_infra_notes.md) | Dong | 🟢 Done | Low | 2026-03-18 |
| [Omni T2A Training](projects/omni_t2a_project_notes.md) | Dong | 🟡 In Progress | High | 2026-04-07 |
| [Uni-1 Omni T2V Training](projects/omni_t2v_project_notes.md) | Dong | 🟡 In Progress | High | 2026-04-06 |
| [Video Director-Style Captioning](projects/video_director_captioning.md) | Dong | 🟡 In Progress | Medium | 2026-04-06 |
| [VibeVoice ASR — podcast_10m](projects/vibevoice_asr.md) | Dong | 🟡 In Progress | High | 2026-04-06 |

---

## Daily Logs

### 2026-04-07 — Omni T2A architecture deep dive
- **Omni T2A**: Deep-read `Qwen3TextAudioPackedPreprocess` and related model components in `lib/ursa`
- **Architecture notes**: Documented Config/setup() pattern, adaLN modulation, sequence packing design (text flat-packed vs audio batched), RoPE flow (computed in preprocessor, applied per-layer in Q/K)
- **Sequence parallelism**: Documented Ulysses, Ring Attention, Flash Attention, FSDP — their roles, tradeoffs, and 2D M×N GPU grid combination
- **Attention masking**: Documented FlexAttention BlockMask mechanics — mask_mod function, block classification (FULL/MIXED/ZERO), `kv_indices`/`full_kv_indices` index structure, concrete 16-token 2-sample example
- **OmniPreprocessData**: Documented all mask/index fields — `packed_und/gen_token_masks`, `packed_und/gen_token_indexes`, `scatter_indices`, `flex_attn_mask`, `flash4_attn_mask`, `modulation_mask`
- All notes appended to `projects/omni_t2a_project_notes.md`

### 2026-04-06 — System setup
- **System**: Project management system initialized. 6 existing worker files migrated into `projects/`.

---

## Archive

_(See `archive/` folder for completed projects.)_

---

## Historical Notes

> The following was the previous contents of MASTER.md before the project management system was initialized:
>
> **Daily Plan — 2026-03-19**
> - Check the detail description (EP) for the golden T2AV dataset
> - Check status of 3M multilingual fidelity job (`raysubmit_76JnAh13s4psimmn`)
> - Delete kiwi-flyte cluster when 3M job completes
> - Run fidelity pipeline on mosaic AVGU table (38K rows, video_path mode)
