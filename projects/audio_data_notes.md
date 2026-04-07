# Audio Data Notes

Survey of audio datasets in lumaverse (main branch, 2026-03-19).

## T2A Training Datasets

| Dataset | S3 URI | Rows | Audio Key | Transcript Key | Notes |
|---------|--------|------|-----------|---------------|-------|
| Emilia | `s3://ai-lumalabs-datasets-ap-se-2/lance_datasets/audio/raw/emilia.lance` | ~40M | `audio_bytes` | `raw_transcript` | TTS training, MP3→WAV |
| Internal Audio V1 | `s3://ai-lumalabs-datasets-ap-se-2/inkyu/lax/audio_segmentation/internal_audio_v1_captioned_original_sr_v3_merged_v4.lance` | 92M | `audio_bytes_ori` | `audio_caption_Qwen3-Omni-30B-A3B-Thinking_gemini_refined_prompt` (JSON) | All languages, original SR |
| Internal Audio V1 EN | `s3://ai-lumalabs-datasets-ap-se-2/inkyu/lax/audio_segmentation/internal_audio_v1_captioned_original_sr_english_v1_compacted.lance` | 32M | `audio_bytes_ori` | same as above (JSON) | English-only subset, `is_variant=True` |
| Podcast Golden | `s3://ai-lumalabs-datasets-ap-se-2/inkyu/podcast_10m/podcast_5s_44p1k_whisper_v3.lance` | 24M | `audio_bytes` | `transcription` | 5s clips, 44.1/48kHz, English, minimal noise |
| Internal Audio V2 EN | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance` | 100M | `audio_bytes` | `whisperx_asr_content` | Per-bucket loading (Koba v2) |

### Pipeline settings per dataset

| Dataset | transcript_key | audio_length_prob | tag_dropout | is_json |
|---------|---------------|-------------------|-------------|---------|
| emilia | raw_transcript | 0.5 | - | False |
| internal-audio-v1 | Qwen3-Omni JSON caption | 0.5 | 0.2 | True |
| internal-audio-v1-english | Qwen3-Omni JSON caption | 0.5 | 0.2 | True |
| podcast-golden-set | transcription | 0.0 | - | False |
| internal-audio-v2-eng | whisperx_asr_content | 0.0 | - | False |

Config: `projects/kuma/kuma/projects/audio/t2a_dataset_config.py`

## T2AV Training Datasets

| Dataset | S3 URI | Rows | Audio Key | Notes |
|---------|--------|------|-----------|-------|
| AI Humans (fixed 5s) | `s3://ai-lumalabs-datasets-ap-se-2/inkyu/t2av/ai_humans_002_conf_audio_video_captioning_batch_all_cate_merged.lance` | TBD | `audio` | 48kHz, Qwen3-Omni captions |
| AI Humans (variable 3-20s) | `s3://ai-lumalabs-datasets-ap-se-2/inkyu/t2av/other_combined__v1_filtered_audio_cut_prob_p1_merged_av_captioned_merged_v3.lance` | TBD | `audio` | 16kHz, shot boundaries |
| AVGU Mosaic v3 | `s3://ai-lumalabs-datasets-ap-se-2-lance/inkyu/t2av/mosaic_golden/mosaic_avgu_seg_audiobox_qwen3omni_av_caption_v3.lance` | 38K | `media.clip` (video path) | 1-10s buckets, 99% 16:9 |

Config: `projects/kuma/kuma/projects/ray3_t2av/configs/t2av/main_train.py`

## Audio Resource Datasets

| Dataset | S3 URI | Audio Key | Notes |
|---------|--------|-----------|-------|
| Zapsplat | `s3://ai-lumalabs-datasets-ap-se-2/lance_datasets/audio/zapsplat.lance` | `audio_bytes` | SFX / sound effects |
| LibriSpeech | `s3://ai-lumalabs-datasets-ap-se-2/lance_datasets/audio/librispeech_{lang}_{split}.lance` | `audio_bytes` | Multilingual speech, FLAC |
| Common Voice 17 | `s3://ai-lumalabs-datasets-ap-se-2/lance_datasets/audio/common_voice_17.lance` | `audio_bytes` | Crowdsourced multilingual speech |

Conversion scripts: `projects/kuma/kuma/projects/audio/{zapsplat,librispeech,common_voice}/`

## Raw Audio Resources on S3

| Path | Content |
|------|---------|
| `s3://ai-lumalabs-datasets-ap-se-2/audio_resource/supplier_data/podcast-enclosures-20m-7m-2/` | 7M podcast MP3s |
| `s3://ai-lumalabs-datasets-ap-se-2/audio_resource/supplier_data/podcast-enclosures-20m-10m-1/` | 10M podcast MP3s |
| `s3://ai-lumalabs-datasets-ap-se-2/audio_resource/zapsplat/` | Zapsplat MP3s |
| `s3://ai-lumalabs-datasets-ap-se-2/audio_resource/multilingual_librispeech/` | LibriSpeech FLAC |

## Processing Pipelines (LAX)

| Pipeline | Location | Input → Output |
|----------|----------|---------------|
| Podcast Segmentation | `lax/projects/audio_segmentation/podcast/` | Raw MP3 → 5-15s segments with VAD |
| ASR Diarization | `lax/projects/av_data_processing/audio/asr_diarize/` (PR #6485) | Segments → WhisperX transcription + speaker IDs |
| AGU Speech | `lax/projects/av_data_processing/audio/agu/speech/` (PR #6485) | ASR output → cut check + Qwen3-Omni dense captions |
| AGU SFX | `lax/projects/av_data_processing/audio/agu/sfx/` (PR #6485) | Raw audio → energy segmentation + SFX captions |
| Fidelity | `lax/projects/av_data_processing/audio/audio_metadata/` (dongguo/datasets) | Any audio → bandwidth + audiobox scores + PANNs AED |
| Audio Duration | `lax/projects/audio_duration/` | Any audio → duration extraction |

### Pipeline flow

```
Raw podcast MP3s
  → Audio Segmentation (VAD, 5-15s clips)
    → ASR Diarization (WhisperX + pyannote)
      → AGU Speech (cut check + dense captioning)
      → AGU SFX (energy segmentation + SFX captioning)
      → Fidelity (bandwidth + audiobox + sound events)
```

## Audio Loading (Koba V2)

- Default pipeline: `lib/koba/koba/pipelines/default_t2a.py`
- Default sample rate: 16kHz
- Decoder: torchcodec `SimpleAudioDecoder`
- Per-bucket loading for duration-balanced batching
- Duration buckets for internal-audio-v1: 2-5s (34%), 5-10s (62%), 10-15s (3%)

Examples: `lib/koba/koba/v2/examples/audio_*.py`

## Key Column Name Conventions

| Column | Meaning |
|--------|---------|
| `audio_bytes` | Audio at 16kHz (standard) |
| `audio_bytes_ori` | Audio at original sample rate |
| `raw_transcript` | Plain text transcript |
| `transcription` | WhisperV3 transcript |
| `whisperx_asr_content` | WhisperX multi-speaker transcript |
| `audio_caption_Qwen3-Omni-*` | Qwen3-Omni JSON caption (General_Caption, Transcription, Script_Summary) |
