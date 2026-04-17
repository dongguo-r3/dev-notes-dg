# Audio Dataset Family Tree

> **What this note is for:** A structured map of all audio source tables and their
> downstream processed tables. Tracks which pipelines have been applied, output URIs,
> row counts, and job status. All downstream tables join to source via `original_row_id`.
>
> **Last updated:** 2026-04-17

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
| **SFT (5 tables)** | Done (2 need cleanup pass) | Running | Running |
| **multilingual_v1** | Done (221.2M/221.8M, 0.27% gap) | ~81% (13/16 partitions) | Running (12 partitions) |
| **en50m_nonen50m** | Done (merged from 8 partitions) | Not started | Not started |
| **internal_audio_v1** | Running (8 partitions) | Not started | Not started |

### Pipelines

| Pipeline | What it produces | Key models | Throughput (8×H100) |
|---|---|---|---|
| **Fidelity v1** | bandwidth, AES quality scores, sound events, audio tags | torch.stft, audiobox-aesthetics ONNX, PANNs CNN14, EAT ViT | ~380 rows/s |
| **VibeVoice ASR** | Re-transcription with VibeVoice model | VibeVoice vLLM | TBD |
| **Speech Metadata v2** | pitch, volume, speed, gender, emotion, age | torchcrepe, wav2vec2 classifiers, ECAPA-TDNN + SVR | TBD (first run in progress) |

### Active Clusters (omniva-flyte, 2026-04-17)

| Cluster | Family | Job | Status |
|---|---|---|---|
| metadata-s0..s7 | multilingual_v1 p0-p7of12 | speech_metadata | Running |
| vibevoice-s0 | SFT hours_140k | speech_metadata | Running |
| vibevoice-s1 | SFT convspeech | speech_metadata | Idle |
| vibevoice-s2 | SFT podcast p11-14 | speech_metadata | Running |
| vibevoice-s3 | SFT podcast p14-17 | speech_metadata | Running |
| vibevoice-s4 | SFT podcast p17-20 | speech_metadata | Running |
| vibevoice-s5..s8 | multilingual_v1 p8-p11of12 | speech_metadata | Idle |

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

| Output | Partitions | Cluster | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p1of3.lance` | p1of3 (frags 0-1793) | vibevoice-s0 | TBD |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p2of3.lance` | p2of3 (frags 1793-3586) | vibevoice-s1 | TBD |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p3of3.lance` | p3of3 (frags 3586-5381) | vibevoice-s2 | TBD |

#### Speech Metadata v2

| Output | Cluster | Job ID | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/hours_140k_speech_metadata.lance` | vibevoice-s0 | `raysubmit_VWqSuD6Yan3yjs6v` | Running |

---

### 1.2 convspeech

**Source:** `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/convspeech/asr/prefiltered_english__whisperx.lance`
**Rows:** 6,514,097 | **Fragments:** 1,614

#### Fidelity v1

| Output | Rows | Gap | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/convspeech/metadata/fidelity_prefiltered_english__whisperx.lance` | 6,514,097 | 0 | Done |

#### VibeVoice ASR

| Output | Cluster | Status |
|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/convspeech_vibevoice_asr.lance` | vibevoice-s3 | TBD |

#### Speech Metadata v2

| Output | Cluster | Job ID | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/convspeech_speech_metadata.lance` | vibevoice-s1 | `raysubmit_paKM7EAr6EpCdjRZ` | Idle |

---

### 1.3 podcast_10m p11-14 (clean)

**Source:** `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p11to14_whisperx_clean.lance`
**Rows:** 7,499,644 | **Fragments:** 1,858

#### Fidelity v1

| Output | Rows | Gap | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/metadata/fidelity_podcast_10m_p11to14_whisperx_clean.lance` | 7,499,326 | 318 (0.004%) | Done — needs cleanup pass |

#### VibeVoice ASR

| Output | Cluster | Status |
|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p11to14_vibevoice_asr.lance` | vibevoice-s4 | TBD |

#### Speech Metadata v2

| Output | Cluster | Job ID | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p11to14_speech_metadata.lance` | vibevoice-s2 | `raysubmit_Ny9YeHH8mUCL8LPw` | Running |

---

### 1.4 podcast_10m p14-17 (clean)

**Source:** `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p14to17_whisperx_clean.lance`
**Rows:** 7,670,431 | **Fragments:** 1,902

#### Fidelity v1

| Output | Rows | Gap | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/metadata/fidelity_podcast_10m_p14to17_whisperx_clean.lance` | 7,670,431 | 0 | Done |

#### VibeVoice ASR

| Output | Cluster | Status |
|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p14to17_vibevoice_asr.lance` | vibevoice-s5 | TBD |

#### Speech Metadata v2

| Output | Cluster | Job ID | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p14to17_speech_metadata.lance` | vibevoice-s3 | `raysubmit_6ZjL95CAtdUiQkJb` | Running |

---

### 1.5 podcast_10m p17-20 (wild)

**Source:** `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance`
**Rows:** 22,655,625 | **Fragments:** 5,606

#### Fidelity v1

| Output | Rows | Gap | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/metadata/fidelity_podcast_10m_p17to20_whisperx_wild_v1.lance` | 22,655,625 | 0 | Done |

#### VibeVoice ASR

| Output | Partitions | Cluster | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p1of3.lance` | p1of3 (frags 0-1868) | vibevoice-s6 | TBD |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p2of3.lance` | p2of3 (frags 1868-3736) | vibevoice-s7 | TBD |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p3of3.lance` | p3of3 (frags 3736-5606) | vibevoice-s8 | TBD |

#### Speech Metadata v2

| Output | Cluster | Job ID | Status |
|---|---|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p17to20_speech_metadata.lance` | vibevoice-s4 | `raysubmit_yeg5RNY6vzCHHchK` | Running |

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

16 partitions processed separately.

| Output Pattern | Status |
|---|---|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p{N}_16_1.lance` | ~13/16 complete (as of 2026-04-06) |

### Speech Metadata v2

12 partitions across 12 clusters. Launched 2026-04-17.

| Partition | Output S3 URI | Cluster | Job ID | Status |
|---|---|---|---|---|
| p0of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p0of12.lance` | metadata-s0 | `raysubmit_PNBuTxkxSEwud2cV` | Running |
| p1of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p1of12.lance` | metadata-s1 | `raysubmit_s82UjvNqwZR2RbEe` | Running |
| p2of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p2of12.lance` | metadata-s2 | `raysubmit_6MvCZWshq2GMBT2s` | Running |
| p3of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p3of12.lance` | metadata-s3 | `raysubmit_d53KsLTV4Z8rDrpM` | Running |
| p4of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p4of12.lance` | metadata-s4 | `raysubmit_PHtVFSiJvQ4GjZtT` | Running |
| p5of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p5of12.lance` | metadata-s5 | `raysubmit_jXVxu9Sy57wccWne` | Running |
| p6of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p6of12.lance` | metadata-s6 | `raysubmit_mAhxXZV6J8LVa6gV` | Running |
| p7of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p7of12.lance` | metadata-s7 | `raysubmit_gbpXui9gPpwvYcuw` | Running |
| p8of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p8of12.lance` | vibevoice-s5 | `raysubmit_jt2X3sL6pANRjdGt` | Idle |
| p9of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p9of12.lance` | vibevoice-s6 | `raysubmit_m5FYHLZDzgii3DAn` | Idle |
| p10of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p10of12.lance` | vibevoice-s7 | `raysubmit_vhskJN81bnnfmZSi` | Idle |
| p11of12 | `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1_p11of12.lance` | vibevoice-s8 | `raysubmit_NQt7whEbEGujyJMB` | Idle |

**Post-processing:** Merge 12 partitions → `s3://...podcast_10m/metadata/speech_metadata_multilingual_v1.lance`

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

| Column | Type | Description |
|---|---|---|
| `pitch_mean_hz` | float64 | Mean f0 (Hz) |
| `pitch_median_hz` | float64 | Median f0 (Hz) |
| `pitch_std_hz` | float64 | f0 standard deviation (Hz) |
| `pitch_category` | string | low / normal / high (gender-aware) |
| `volume_db` | float64 | RMS energy (dBFS) |
| `volume_category` | string | low / normal / high |
| `speaking_rate_cps` | float64 | Characters per second |
| `speaking_rate_wps` | float64 | Words per second |
| `speaking_rate_category` | string | slow / normal / fast |
| `gender` | string | male / female |
| `gender_confidence` | float64 | Classifier confidence (0-1) |
| `emotion_superb` | string | 4-class: neu / hap / ang / sad |
| `emotion_superb_confidence` | float64 | Confidence (0-1) |
| `emotion_superb_scores` | string (JSON) | All class probabilities |
| `emotion_dpngtm` | string | 7-class: angry / calm / disgust / fearful / happy / sad / surprised |
| `emotion_dpngtm_confidence` | float64 | Confidence (0-1) |
| `emotion_dpngtm_scores` | string (JSON) | All class probabilities |
| `age_years` | float64 | Predicted age (years) |
| `age_group` | string | child / teen / young_adult / adult / older_adult |
