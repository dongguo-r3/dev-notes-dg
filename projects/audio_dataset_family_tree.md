# Audio Dataset Family Tree

> **What this note is for:** A structured map of all audio source tables and their
> downstream processed tables. Tracks which pipelines have been applied, output URIs,
> row counts, and job status. All downstream tables join to source via `original_row_id`.
>
> **Last updated:** 2026-04-17 22:30 UTC

---

## Summary

### Source Table Families

| Family | Tables | Total Rows | Description |
|---|---|---|---|
| **SFT** | 5 | 66,085,511 | Curated English speech for SFT training |
| **multilingual_v1** | 1 | 221,842,325 | Podcast pretrain, 36 languages |
| **en50m_nonen50m** | 1 | 106,898,690 | Podcast pretrain, balanced EN/non-EN split |
| **internal_audio_v1** | 1 | 92,221,138 | Mixed audio (speech/music/SFX), Qwen3-Omni captions |

### Processing Status Overview

| Family | Fidelity v1 | VibeVoice ASR | Speech Metadata v2 |
|---|---|---|---|
| **SFT (5 tables)** | Done (2 need cleanup pass) | ✅ Done (9 partitions, 66M rows, finished 2026-04-16) | 3/5 tables done; 2 big tables (hours_140k, podcast p17-20) in 6-partition processing |
| **multilingual_v1** | Done (221.2M/221.8M, 0.27% gap) | ✅ Done — 16/16 partitions (221.3M/221.8M, 0.23% gap, all committed 2026-04-06) | ✅ Done — all 12 partitions SUCCEEDED (221.84M rows) |
| **en50m_nonen50m** | Done (merged from 8 partitions) | Not started | Not started |
| **internal_audio_v1** | Running (8 partitions) | Not started | Not started |

### Pipelines

| Pipeline | What it produces | Key models | Throughput (8×H100) |
|---|---|---|---|
| **Fidelity v1** | bandwidth, AES quality scores, sound events, audio tags | torch.stft, audiobox-aesthetics ONNX, PANNs CNN14, EAT ViT | ~380 rows/s |
| **VibeVoice ASR** | Re-transcription with VibeVoice model | VibeVoice vLLM | ~28 rows/s (~3.5 rows/s/GPU) |
| **Speech Metadata v2** | pitch, volume, speed, gender, emotion, age | torchcrepe, wav2vec2-base ONNX (prithiv gender, dpngtm emotion), ECAPA-TDNN + SVR | ~380-420 rows/s (measured 2026-04-17) |
| **Speech Metadata v2 + Audeering** | + audeering_gender (3-class with child), audeering_age_years, audeering_arousal/dominance/valence | + audeering wav2vec2-large ONNX (non-commercial) | ~250 rows/s (not yet run on cluster) |

### Speech Metadata v2 — Pipeline Notes

- **Commercial pipeline** (`run_speech_metadata_pipeline_gpu`): default variant used
  in all 17 jobs documented in this file. Outputs 16 columns above.
- **Non-commercial variant** (`run_speech_metadata_pipeline_gpu_with_audeering`):
  adds 6 Audeering comparison columns (CC-BY-NC-SA). Available for research but
  not used for production tables.
- **DSP-only variant** (`run_speech_metadata_pipeline_gpu_dsp_only`): pitch + volume
  + speaking rate only (no gender/emotion/age). Useful for quick runs.
- **Per-row latency** (local H100, single actor): ~41ms commercial, ~57ms with Audeering.
- **Key optimizations applied** (see `DESIGN_SPEECH_METADATA.md` in the lax repo):
  fixed 8s input window (avoids ORT dynamic-shape replanning, ~15-20x speedup),
  batch=2 windowing for emotion (first 8s + last 8s averaged), custom GPU decode
  for torchcrepe (~15x faster than stock), swapped gender to wav2vec2-base
  (prithiv) from wav2vec2-large (alefiury) with 100% label agreement, dropped
  redundant `emotion_superb` model.

### Active Clusters (omniva-flyte, 2026-04-17 22:30 UTC)

| Cluster | Family | Current Speech Metadata Job | Status |
|---|---|---|---|
| vibevoice-s0 | SFT hours_140k p2 (2,6,1) | `raysubmit_5hV5PLAEYPPBPS3E` | Running |
| vibevoice-s1 | SFT hours_140k p3 (3,6,1) | `raysubmit_X2u43hgz8qSeXhjE` | ✅ SUCCEEDED |
| vibevoice-s2 | SFT podcast_p17to20 p2 (2,6,1) | `raysubmit_ApraG7urL9zUhm8f` | Running |
| vibevoice-s3 | SFT podcast_p17to20 p3 (3,6,1) | `raysubmit_cQSFxY9E3KUD9zgM` | Running |
| vibevoice-s4 | SFT podcast_p17to20 p4 (4,6,1) | `raysubmit_PahNjgh64BxDdTXr` | Running |
| vibevoice-s5..s8 | multilingual_v1 p8-p11of12 | (SUCCEEDED earlier) | ✅ Idle |
| metadata-s0 | SFT hours_140k p4 (4,6,1) | `raysubmit_PDH1aB95TZbaPgvx` | Running (launched 22:20) |
| metadata-s1 | SFT hours_140k p5 (5,6,1) | `raysubmit_fr2hDgwWuP4zTGXd` | Running (launched 22:24) |
| metadata-s2 | SFT podcast_p17to20 p5 (5,6,1) | `raysubmit_ADLXTfAuab6thAY9` | Running (launched 22:28) |
| metadata-s3..s7 | multilingual_v1 p3-p7of12 | (SUCCEEDED earlier) | ✅ Idle |

---

## Family 1: SFT Tables

5 curated English speech tables used for supervised fine-tuning. All share the same
schema from the WhisperX ASR pipeline.

**Common columns:** `audio_bytes`, `audio_path`, `sample_rate`, `language`, `segment_start`,
`segment_end`, `segment_duration`, `whisperx_asr_content`, `whisperx_timestamp`,
`num_speakers`, `total_speakers_in_file`, `lufs_gain_db`, `snr_db`, `speech_ratio`,
`avg_word_score`, `overlap_ratio`, `original_row_id`

---

### 1.1 hours_140k

**Source:** `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance`
**Rows:** 21,745,714 | **Fragments:** 5,381

#### Fidelity v1

| Output | Rows | Gap | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/metadata/fidelity_prefiltered_english__whisperx_v1.lance` | 21,721,789 | 23,925 (0.11%) | Done — needs cleanup pass with `batch_size=1024` |

#### VibeVoice ASR

3 partitions via `partitions_range N,3,1`. Launched 2026-04-16 attempt 3 (00:41 PDT). All complete by 2026-04-16 22:40 UTC.

| Output | Partition | Rows | Job ID | Status |
|---|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p1of3.lance` | `0,3,1` | 7,275,309 | `raysubmit_gmPJG1hGBJiNfSyi` | ✅ Complete (v26, 2026-04-16 22:40) |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p2of3.lance` | `1,3,1` | 7,235,639 | `raysubmit_sRfGyqjmqPCxE1x3` | ✅ Complete (v25, 2026-04-16 22:19) |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p3of3.lance` | `2,3,1` | 7,254,118 | `raysubmit_T9L15MFHmDYa5TsD` | ✅ Complete (v25, 2026-04-16 22:20) |
| **Total** | | **21,765,066** | | (vs 21,745,714 source, +19K rows from concurrent writes) |

#### Speech Metadata v2

**Output dest:** `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/hours_140k_speech_metadata.lance`

Originally launched as a whole-table job (`raysubmit_VWqSuD6Yan3yjs6v`) on vibevoice-s0
at 07:49 UTC. Stopped at 08:47 UTC (~5% progress, ~1.09M rows committed) and
repartitioned into 6 fragment partitions to enable parallel processing across
multiple clusters after s1-s3 completed their SFT jobs.

All 6 partitions share the same destination URI (above); LAX's checkpoint mechanism
handles the combined writes and prevents duplicate rows via `original_row_id`.

| Partition | Range | Est. Rows | Cluster | Job ID | Status |
|---|---|---|---|---|---|
| p0+p1 | `0,6,2` | ~7.2M | vibevoice-s0 | `raysubmit_u3cXsqFMgCdtVprE` | ✅ SUCCEEDED (08:48 UTC) |
| p2 | `2,6,1` | ~3.6M | vibevoice-s0 (relaunch) | `raysubmit_5hV5PLAEYPPBPS3E` | Running (first attempt `raysubmit_LWZxpUgPKKqnFHGt` FAILED, rowid_mappings race) |
| p3 | `3,6,1` | ~3.6M | vibevoice-s1 | `raysubmit_X2u43hgz8qSeXhjE` | ✅ SUCCEEDED |
| p4 | `4,6,1` | ~3.6M | metadata-s0 | `raysubmit_PDH1aB95TZbaPgvx` | Running (launched 22:20 UTC) |
| p5 | `5,6,1` | ~3.6M | metadata-s1 | `raysubmit_fr2hDgwWuP4zTGXd` | Running (launched 22:24 UTC) |

---

### 1.2 convspeech

**Source:** `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/convspeech/asr/prefiltered_english__whisperx.lance`
**Rows:** 6,514,097 | **Fragments:** 1,614

#### Fidelity v1

| Output | Rows | Gap | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/convspeech/metadata/fidelity_prefiltered_english__whisperx.lance` | 6,514,097 | 0 | Done |

#### VibeVoice ASR

| Output | Rows | Cluster | Job ID | Status |
|---|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/convspeech_vibevoice_asr.lance` | 6,514,097 | vibevoice-s3 | `raysubmit_1dCMTYz6JywcHE13` | ✅ Complete (v21, 2026-04-16 20:12) |

#### Speech Metadata v2

| Output | Cluster | Job ID | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/convspeech_speech_metadata.lance` | vibevoice-s1 | `raysubmit_paKM7EAr6EpCdjRZ` | ✅ SUCCEEDED (6.51M / 6.51M rows, 100%) |

---

### 1.3 podcast_10m p11-14 (clean)

**Source:** `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p11to14_whisperx_clean.lance`
**Rows:** 7,499,644 | **Fragments:** 1,858

#### Fidelity v1

| Output | Rows | Gap | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/metadata/fidelity_podcast_10m_p11to14_whisperx_clean.lance` | 7,499,326 | 318 (0.004%) | Done — needs cleanup pass |

#### VibeVoice ASR

| Output | Rows | Cluster | Job ID | Status |
|---|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p11to14_vibevoice_asr.lance` | 7,499,644 | vibevoice-s4 | `raysubmit_sFwQKfbEUAtVFxae` | ✅ Complete (v21, 2026-04-16 22:30) |

#### Speech Metadata v2

| Output | Cluster | Job ID | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p11to14_speech_metadata.lance` | vibevoice-s2 | `raysubmit_Ny9YeHH8mUCL8LPw` | ✅ SUCCEEDED (7.50M / 7.50M rows, 100%) |

---

### 1.4 podcast_10m p14-17 (clean)

**Source:** `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p14to17_whisperx_clean.lance`
**Rows:** 7,670,431 | **Fragments:** 1,902

#### Fidelity v1

| Output | Rows | Gap | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/metadata/fidelity_podcast_10m_p14to17_whisperx_clean.lance` | 7,670,431 | 0 | Done |

#### VibeVoice ASR

| Output | Rows | Cluster | Job ID | Status |
|---|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p14to17_vibevoice_asr.lance` | 7,670,431 | vibevoice-s5 | `raysubmit_aDDdJWmhedpKKfHc` | ✅ Complete (v21, 2026-04-16 22:57) |

#### Speech Metadata v2

| Output | Cluster | Job ID | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p14to17_speech_metadata.lance` | vibevoice-s3 | `raysubmit_6ZjL95CAtdUiQkJb` | ✅ SUCCEEDED (7.67M / 7.67M rows, 100%) |

---

### 1.5 podcast_10m p17-20 (wild)

**Source:** `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance`
**Rows:** 22,655,625 | **Fragments:** 5,606

#### Fidelity v1

| Output | Rows | Gap | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/metadata/fidelity_podcast_10m_p17to20_whisperx_wild_v1.lance` | 22,655,625 | 0 | Done |

#### VibeVoice ASR

3 partitions via `partitions_range N,3,1`. Launched 2026-04-16 attempt 3 (00:41 PDT). All complete by 2026-04-16 23:36 UTC.

| Output | Partition | Rows | Cluster | Job ID | Status |
|---|---|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p1of3.lance` | `0,3,1` | 7,533,353 | vibevoice-s6 | `raysubmit_87F1nYJGZsLPDrzA` | ✅ Complete (v27, 2026-04-16 23:13) |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p2of3.lance` | `1,3,1` | 7,534,293 | vibevoice-s7 | `raysubmit_rVTvJyEutpn8uYR1` | ✅ Complete (v26, 2026-04-16 22:54) |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p3of3.lance` | `2,3,1` | 7,615,367 | vibevoice-s8 | `raysubmit_r2RXzxTC8tLb3mfS` | ✅ Complete (v26, 2026-04-16 23:36) |
| **Total** | | **22,683,013** | | | (vs 22,655,625 source, +27K rows from concurrent writes) |

#### Speech Metadata v2

**Output dest:** `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p17to20_speech_metadata.lance`

Originally launched as a whole-table job (`raysubmit_yeg5RNY6vzCHHchK`) on vibevoice-s4
at 07:57 UTC. Stopped at 08:47 UTC (~4% progress, ~900K rows committed) and
repartitioned into 6 fragment partitions. Same resume mechanism as hours_140k above.

| Partition | Range | Est. Rows | Cluster | Job ID | Status |
|---|---|---|---|---|---|
| p0+p1 | `0,6,2` | ~7.6M | vibevoice-s4 | `raysubmit_RynKYsTH2fUv1K8W` | ✅ SUCCEEDED (08:48 UTC) |
| p2 | `2,6,1` | ~3.8M | vibevoice-s2 (relaunch) | `raysubmit_ApraG7urL9zUhm8f` | Running (first attempt `raysubmit_11WRmFeAhcxtfXvG` FAILED, rowid_mappings race) |
| p3 | `3,6,1` | ~3.8M | vibevoice-s3 (relaunch) | `raysubmit_cQSFxY9E3KUD9zgM` | Running (first attempt `raysubmit_VGyMv5KzpsD2MEHe` FAILED, rowid_mappings race) |
| p4 | `4,6,1` | ~3.8M | vibevoice-s4 (relaunch) | `raysubmit_PahNjgh64BxDdTXr` | Running |
| p5 | `5,6,1` | ~3.8M | metadata-s2 | `raysubmit_ADLXTfAuab6thAY9` | Running (launched 22:28 UTC) |

---

## Family 2: whisperx__multilingual_v1_compacted

The largest audio dataset — 222M podcast speech segments across 36 languages.

**Source:** `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance`
**Rows:** 221,842,325 | **Fragments:** 2,227

**Columns:** `audio_bytes`, `sample_rate`, `segment_duration`, `language`, `snr_db`,
`whisperx_asr_content`, `whisperx_timestamp`, `num_speakers`

**Note:** Use `batch_size <= 2048` when reading — fragments are ~100K rows each, and
large audio batches can hit the Arrow 2GB per-column limit.

### Fidelity v1

Processed as 16 partitions, then merged. Fidelity filter scores also computed.

| Output | S3 URI | Rows | Status |
|---|---|---|---|
| **Fidelity (merged)** | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_multilingual_v1_round1.lance` | 221,238,296 | Done |
| **Fidelity Scores** | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_multilingual_v1_round1_scores.lance` | 221,238,296 | Done |

Gap: 604,029 rows (0.27%) — Arrow 2GB overflow batches. Needs cleanup pass with `batch_size=1024`.

### VibeVoice ASR

16 partitions processed separately on omniva-flyte and kiwi-flyte clusters,
completed 2026-04-01 — 2026-04-06. Resume jobs for 3 late partitions (p2, p6,
p13) all finished on 2026-04-06 by 22:47 UTC.

| Output Pattern | Partitions | Rows | Status |
|---|---|---|---|
| `s3://...podcast_10m/asr/vibevoice_multilingual_v2_p{N}_16_1.lance` | 16/16 | 221,324,671 | ✅ All complete — 99.77% of source (0.23% gap, ~517K rows) |

See `audio_asr_vibevoice.md` for per-partition row counts, resume job IDs, and
analysis dashboards (2026-04-07 podcast comparison / language detection / EN
progressive filtering dashboards uploaded to S3).

### Speech Metadata v2

12 partitions across 12 clusters. Launched 2026-04-17 08:22-08:26 UTC. **ALL SUCCEEDED** by ~20:00 UTC (~12h total). Cluster throughput averaged ~420 rows/s each.

| Partition | Range | Output S3 URI | Cluster | Job ID | Rows Committed | Status |
|---|---|---|---|---|---|---|
| p0of12 | `0,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p0of12.lance` | metadata-s0 | `raysubmit_PNBuTxkxSEwud2cV` | 18,559,064 | ✅ SUCCEEDED |
| p1of12 | `1,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p1of12.lance` | metadata-s1 | `raysubmit_s82UjvNqwZR2RbEe` | 18,409,406 | ✅ SUCCEEDED |
| p2of12 | `2,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p2of12.lance` | metadata-s2 | `raysubmit_6MvCZWshq2GMBT2s` | 18,599,430 | ✅ SUCCEEDED |
| p3of12 | `3,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p3of12.lance` | metadata-s3 | `raysubmit_d53KsLTV4Z8rDrpM` | ~18.4M | ✅ SUCCEEDED |
| p4of12 | `4,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p4of12.lance` | metadata-s4 | `raysubmit_PHtVFSiJvQ4GjZtT` | ~18.6M | ✅ SUCCEEDED |
| p5of12 | `5,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p5of12.lance` | metadata-s5 | `raysubmit_jXVxu9Sy57wccWne` | ~18.4M | ✅ SUCCEEDED |
| p6of12 | `6,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p6of12.lance` | metadata-s6 | `raysubmit_mAhxXZV6J8LVa6gV` | ~18.4M | ✅ SUCCEEDED |
| p7of12 | `7,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p7of12.lance` | metadata-s7 | `raysubmit_gbpXui9gPpwvYcuw` | ~18.4M | ✅ SUCCEEDED |
| p8of12 | `8,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p8of12.lance` | vibevoice-s5 | `raysubmit_jt2X3sL6pANRjdGt` | ~18.6M | ✅ SUCCEEDED |
| p9of12 | `9,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p9of12.lance` | vibevoice-s6 | `raysubmit_m5FYHLZDzgii3DAn` | 18,178,106 | ✅ SUCCEEDED |
| p10of12 | `10,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p10of12.lance` | vibevoice-s7 | `raysubmit_vhskJN81bnnfmZSi` | ~18.5M | ✅ SUCCEEDED |
| p11of12 | `11,12,1` | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p11of12.lance` | vibevoice-s8 | `raysubmit_NQt7whEbEGujyJMB` | ~18.5M | ✅ SUCCEEDED |
| **Total** | | | | | **221,842,325** | **100% of source** |

**Post-processing:** Merge 12 partitions → `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1.lance` (pending, run with `lax.scripts.infra.concat_tables`).

---

## Family 3: en50m_nonen50m (Podcast Pretrain, Balanced)

A balanced 50M English + ~57M non-English subset of podcast speech. Separate from
`multilingual_v1` — different partitioning and compaction.

**Source:** `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__en50m_nonen50m_compacted.lance`
**Rows:** 106,898,690

**Columns:** Same schema as `multilingual_v1` — `audio_bytes`, `sample_rate`, `segment_duration`,
`language`, `snr_db`, `whisperx_asr_content`, `whisperx_timestamp`, `num_speakers`

### Related ASR tables under podcast_10m

| Table | S3 URI | Rows | Notes |
|---|---|---|---|
| `whisperx__multilingual_v1_compacted` | `s3://...podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | 221,842,325 | Full multilingual (Family 2) |
| `whisperx__en50m_nonen50m_compacted` | `s3://...podcast_10m/asr/whisperx__en50m_nonen50m_compacted.lance` | 106,898,690 | **This table** — balanced EN/non-EN |
| `whisperx__eng_v1` | `s3://...podcast_10m/asr/whisperx__eng_v1.lance` | 102,116,557 | English-only subset |

### Fidelity v1

Processed as 8 partitions, then merged.

| Output | S3 URI | Status |
|---|---|---|
| **Partitions** | `s3://...podcast_10m/metadata/fidelity_en50m_nonen50m_p{0..7}of8.lance` | Done |
| **Merged** | `s3://...podcast_10m/metadata/fidelity_en50m_nonen50m.lance` | Done |

### VibeVoice ASR

Not started.

### Speech Metadata v2

Not started.

---

## Family 4: internal_audio_v1

92M rows of mixed audio (speech, music, SFX) with Qwen3-Omni captions.
Uses `audio_bytes_ori` (original sample rate) instead of `audio_bytes`.

### Source Table

**Source:** `s3://ai-lumalabs-datasets-ap-se-2/inkyu/lax/audio_segmentation/internal_audio_v1_captioned_original_sr_v3_merged_v4.lance`
**Rows:** 92,221,138

**Key columns:** `audio_bytes`, `audio_bytes_ori`, `sample_rate`, `segment_duration`,
`lufs_gain_db`, `is_speech`, `compression_ratio`,
`audio_caption_Qwen3-Omni-30B-A3B-Thinking_gemini_refined_prompt` (JSON caption)

**Note:** Fidelity pipeline uses `audio_key=audio_bytes_ori` for this table (not `audio_bytes`).

### Upstream Lineage

The `v3_merged_v4` table was assembled from 10 partitioned captioning runs, with
some rows filtered during merge (captioning/schema cleanup across iterations).

```
internal_audio_v1_captioned_original_sr_v3_part{0..9}.lance   (10 parts, ~97M rows total)
  ↓ [merge + filter]
internal_audio_v1_captioned_original_sr_v3_merged_v4.lance    (92,221,138 rows)
  ↓ [English filter]
internal_audio_v1_captioned_original_sr_english_v1_compacted.lance  (31,679,067 rows, 34.4%)
```

| Stage | S3 URI | Rows |
|---|---|---|
| v3_part0 | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_v3_part0.lance` | 9,591,199 |
| v3_part1 | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_v3_part1.lance` | 9,573,378 |
| v3_part2 | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_v3_part2.lance` | 9,546,883 |
| v3_part3 | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_v3_part3.lance` | 9,796,357 |
| v3_part4 | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_v3_part4.lance` | 9,747,456 |
| v3_part5 | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_v3_part5.lance` | 9,716,255 |
| v3_part6 | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_v3_part6.lance` | 9,702,031 |
| v3_part7 | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_v3_part7.lance` | 9,685,706 |
| v3_part8 | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_v3_part8.lance` | 9,658,528 |
| v3_part9 | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_v3_part9.lance` | 10,225,897 |
| **Sum of parts** | | **97,243,690** |
| **v3_merged_v4** | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_v3_merged_v4.lance` | **92,221,138** |
| **Filtered out** | | **5,022,552** (5.2%) |
| **English subset** | `s3://...audio_segmentation/internal_audio_v1_captioned_original_sr_english_v1_compacted.lance` | **31,679,067** (34.4%) |

Earlier versions (v1, v2) also exist on S3 but are superseded by v3.

### Fidelity v1

8 partitions, launched 2026-04-08 on mixed kiwi (Sydney) / omniva (US) clusters.

| Output Pattern | Status | Notes |
|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_fidelity_p{N}of8.lance` | Running | kiwi ~3.4 frags/min, omniva ~1.1 frags/min (cross-region S3 latency) |

Post-processing plan: merge 8 partitions → `internal_audio_v1_fidelity.lance`, then run fidelity filter.

### VibeVoice ASR

Not started — pending fidelity completion.

### Speech Metadata v2

Not started — pending fidelity completion.

---

## Output Column Reference

### Fidelity v1 columns

| Column | Type | Description |
|---|---|---|
| `bandwidth_hz_50` | float64 | Spectral center of mass (Hz) |
| `bandwidth_hz_90` | float64 | Upper bound of useful content (Hz) |
| `bandwidth_hz_95` | float64 | Effective bandwidth (Hz) |
| `bandwidth_hz_99` | float64 | Full spectral extent (Hz) |
| `aes_ce` | float64 | Content Enjoyment (1-10) |
| `aes_cu` | float64 | Content Usefulness (1-10) |
| `aes_pc` | float64 | Production Complexity (1-10) |
| `aes_pq` | float64 | Production Quality (1-10) |
| `sound_events` | string (JSON) | Per-second top-K acoustic events |
| `audio_tags` | string (JSON) | Clip-level top-K audio tags |

### Speech Metadata v2 columns

Production pipeline (`run_speech_metadata_pipeline_gpu`) outputs the following 16
columns. The `emotion_superb` (4-class) columns from an earlier design were
dropped on 2026-04-17 as redundant with `emotion_dpngtm` (7-class subsumes 4-class).

| Column | Type | Description |
|---|---|---|
| `pitch_mean_hz` | float64 | Mean f0 (Hz) via torchcrepe tiny + custom GPU decode |
| `pitch_median_hz` | float64 | Median f0 (Hz) |
| `pitch_std_hz` | float64 | f0 standard deviation (Hz) |
| `pitch_category` | string | low / normal / high (gender-aware thresholds) |
| `volume_db` | float64 | RMS energy (dBFS, pure DSP on GPU) |
| `volume_category` | string | low / normal / high |
| `speaking_rate_cps` | float64 | Characters per second (from transcript) |
| `speaking_rate_wps` | float64 | Words per second (from transcript) |
| `speaking_rate_category` | string | slow / normal / fast |
| `gender` | string | male / female (wav2vec2-base ONNX, prithivMLmods) |
| `gender_confidence` | float64 | Classifier confidence (0-1) |
| `emotion_dpngtm` | string | 7-class: angry / calm / disgust / fearful / happy / sad / surprised (wav2vec2-base ONNX, batch=2 windowing) |
| `emotion_dpngtm_confidence` | float64 | Confidence (0-1) |
| `emotion_dpngtm_scores` | string (JSON) | All 7 class probabilities |
| `age_years` | float64 | Predicted age (years) via ECAPA-TDNN + SVR regressor |
| `age_group` | string | child / teen / young_adult / adult / older_adult |

**Known limitations:**

- Emotion `dpngtm` tends to over-classify "angry" on natural conversational/podcast
  speech (it was trained on acted RAVDESS-style emotion datasets). Its "angry"
  label often corresponds to high-energy speech with neutral valence. For more
  reliable emotion, consider adding the non-commercial Audeering dimensional
  model (arousal/dominance/valence) in a separate comparison run, or threshold
  dpngtm predictions by confidence.
- `age_years` correlates moderately with true age (0.58-0.81 depending on
  dataset) but absolute values are not trustworthy to better than ±8 years.
  Use `age_group` bins for more robust downstream filtering.
- `pitch_*` uses the `weighted_argmax` CREPE decoder (not Viterbi) — about 3%
  higher std on per-clip f0 but 2.8x faster. Suitable for coarse pitch stats.

---

## Operational Notes / Lessons Learned

### Concurrent writers to a shared Lance destination (2026-04-17)

When multiple jobs share the same destination Lance table, each new job reads the
destination on startup to build a `_rowid_mappings_tmp_*` manifest (what's already
done). If several jobs start within seconds of each other, they race on these temp
folders: one job may 404 while trying to read a file another job is mid-writing.

**Error signature:**

```
LSUFatalError: Not a S3 file:
  ('ai-lumalabs-datasets-ap-se-2-lance',
   'dongguo/lax/metadata/sft/{table}_speech_metadata.lance/_rowid_mappings_tmp_.../original_row_id/*.index.json')
  err: 404 NoSuchKey / HeadObject Not Found
```

**Workaround:** stagger launches of jobs that share a destination by ~3 minutes
(long enough for the first job to finish its initial resume phase).

**Observed on 2026-04-17 19:46 UTC:** launched 5 partition jobs simultaneously
(s0, s1, s2, s3, s4). 3 of 5 failed with this error. Relaunching the 3 failed
jobs one at a time with 3-min gaps succeeded.

### Repartition semantics with `--partitions_range`

`--partitions_range start,total,size` uses modular arithmetic on fragment indices:

```python
for i in range(min(size, total - start)):
    fragments_for_this_partition.extend(src_fragments[start + i :: total])
```

- `0,12,1` → every 12th fragment starting at 0 → partition 0 of 12-way split
- `0,6,2` → fragments where `idx % 6 in (0, 1)` → 1/3 of the fragments (union of partitions 0 and 1)
- `2,6,1` → fragments where `idx % 6 == 2` → 1/6 of the fragments (just partition 2)

**LAX's checkpoint mechanism resumes across partition_range changes** — if you
restart a job with a different `--partitions_range` but the same destination,
LAX reads the destination's `original_row_id` column, figures out what's
already committed, and skips those fragments in the new partition. This means
you can stop a whole-table job and restart with finer partition splits without
losing committed work (as we did with hours_140k and podcast_p17to20 repartition
at 08:47 UTC).

### Ray cluster inventory (omniva-flyte, dongguo)

Active production clusters (used across VibeVoice ASR and Speech Metadata v2):

| Family | Clusters | Nodes × GPUs | Current use (2026-04-17) |
|---|---|---|---|
| `vibevoice-omniva-s0..s8` | 9 | 1 × 8 H100 each | Speech Metadata v2 SFT partition jobs (s0-s4); idle (s5-s8, finished multilingual) |
| `metadata-s0..s7` | 8 | 1 × 8 H100 each | Finished multilingual v1 partitions; now running SFT partition 4/5 jobs on s0/s1/s2 |

Total: **17 clusters × 8 H100 GPUs = 136 GPUs** available for audio metadata work.
