# MASTER

## Project Index

| Project | Owner | Status | Priority | Last Updated |
|---|---|---|---|---|
| [Audio Data](projects/audio_data.md) | Dong | 🟡 In Progress | Medium | 2026-03-19 |
| [Audio Data Pipeline](projects/audio_data_pipeline.md) | Dong | 🟡 In Progress | Medium | 2026-04-08 |
| [Audio Processing](projects/audio_processing.md) | Dong | 🟢 Done | Low | 2026-03-18 |
| [Data Infra](projects/data_infra.md) | Dong | 🟢 Done | Low | 2026-03-18 |
| [Omni Model T2A](projects/omni_t2a_project_notes.md) | Dong | 🟡 In Progress | High | 2026-04-08 |
| [Omni Model T2V](projects/omni_model_t2v.md) | Dong | 🟡 In Progress | High | 2026-04-06 |
| [Video Captioning](projects/video_captioning.md) | Dong | 🟡 In Progress | Medium | 2026-04-08 |
| [Video Captioning — Director Style](projects/video_captioning_director_style.md) | Dong | 🟡 In Progress | Medium | 2026-04-08 |
| [Audio ASR — VibeVoice](projects/audio_asr_vibevoice.md) | Dong | 🟡 In Progress | High | 2026-04-06 |
| [Sticky Notes](projects/sticky_notes.md) | Dong | 🟡 In Progress | Low | 2026-04-08 |

---

## Daily Logs

### 2026-04-08 — Omni T2A: data loader + parameter init
- **Omni T2A**: Verifying data loader correctness and efficiency; verifying model parameter loading and initialization

### 2026-04-08 — EP dataset exploration + dense captioning sync
- **Sticky Notes**: Preliminary findings from reading ep_dataset examples at `s3://ai-lumalabs-datasets-ap-se-2/haoxiang/ep_vlm/0319_mix2.lance`
- **Video Captioning** (new): Synced with Riddhish — T5 not suited for dense captions; LLM top-layer embeddings can't simply replace T5 in Ray models; PoC branch obtained from Riddhish

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
