# Audio Processing Notes

Technical notes on audio data I/O, signal processing, resampling, and bandwidth measurement across the lumaverse repo. Compiled from internal investigation (2026-03-18).

---

## Table of Contents

- [Audio I/O and Signal Processing Inventory](#audio-io-and-signal-processing-inventory)
  - [Shared Libraries (lib/)](#shared-libraries-lib)
  - [Data Pipelines (projects/lax/)](#data-pipelines-projectslax)
  - [Key Libraries Used](#key-libraries-used)
- [Resampling Methods](#resampling-methods)
  - [Six Backends in the Repo](#six-backends-in-the-repo)
  - [Common Target Sample Rates](#common-target-sample-rates)
  - [Input Priority (koba v2)](#input-priority-koba-v2)
- [Bandwidth Measurement](#bandwidth-measurement)
  - [How is bandwidth calculated in the fidelity pipeline?](#how-is-bandwidth-calculated-in-the-fidelity-pipeline)
  - [Is there any other version of bandwidth calculation in the repo?](#is-there-any-other-version-of-bandwidth-calculation-in-the-repo)
- [Upsampling and Artifacts](#upsampling-and-artifacts)
  - [What is the Nyquist frequency?](#what-is-the-nyquist-frequency)
  - [Does upsampling to 48kHz affect measured bandwidth?](#does-upsampling-to-48khz-affect-measured-bandwidth)
  - [Filter-level artifacts by backend](#filter-level-artifacts-by-backend)
  - [The real problem: spectral gap (major)](#the-real-problem-spectral-gap-major)
  - [Recommendations for podcast data in TTA training](#recommendations-for-podcast-data-in-tta-training)
- [Demucs Source Separation in the ASR/Diarize Pipeline](#demucs-source-separation-in-the-asrdiarize-pipeline)
  - [What is Demucs?](#what-is-demucs)
  - [How It's Used](#how-its-used)
  - [Pipeline Flow](#pipeline-flow)
  - [When to Use](#when-to-use)
  - [Diarization Trade-off](#diarization-trade-off)
  - [GPU Memory for Large-Scale Processing](#gpu-memory-for-large-scale-processing)

---

## Audio I/O and Signal Processing Inventory

### Shared Libraries (lib/)

#### audiotools — `lib/audiotools/`

Primary audio processing library built around the `AudioSignal` class.

- **`audiotools/core/audio_signal.py`** — `AudioSignal` class

  - `load_from_file()` — loads audio from file paths using **soundfile**
  - `write(audio_path)` — saves audio to file using **soundfile**
  - `load_from_array(audio_array, sample_rate)` — creates AudioSignal from numpy/torch arrays
  - `resample(sample_rate)` — uses **julius.resample_frac** for high-quality sinc interpolation
  - `excerpt(audio_path, offset, duration)` — random excerpt extraction
  - `salient_excerpt()` — excerpt based on loudness thresholds
  - `info()` — file duration and format metadata
  - Supported formats: WAV, FLAC, OGG (via soundfile)

- **`audiotools/core/ffmpeg.py`** — `FFMPEGMixin`

  - `ffmpeg_resample(sample_rate)` — resampling using ffmpeg subprocess
  - `load_from_file_with_ffmpeg(audio_path)` — FFmpeg-based decoding for unsupported formats (MP3, M4A, etc.)
  - `ffmpeg_loudness()` — loudness computation via ffmpeg ebur128 filter
  - `r128stats(filepath)` — extracts loudness statistics using ffmpeg ebur128 filter
  - `ffprobe_offset_and_codec(path)` — gets codec and timing info from audio files
  - Handles codec latency (MP3 0.027s threshold), audio sync offsets
  - Libraries: ffmpy, subprocess

#### hplv — `lib/hplv/`

High-performance low-level video/audio library.

- **`hplv/audio_file.py`** — `AudioFile` class

  - High-performance audio decoder with random access
  - `get_audio_in_time_range(start_sec, end_sec)` — time-based access
  - `get_samples_in_range(start, end)` — sample-based access
  - `get_samples_at(indices)` — random sample access
  - `get_batch_audio(duration_sec)` — sequential streaming
  - Supports remote files (s3://, gs://, ki://, fra://) via LumaFile
  - Raw bytes input support, audio stream selection for multi-track files
  - `target_sample_rate` parameter for automatic resampling
  - Output format: float32 stereo by default
  - Libraries: PyAV (FFmpeg bindings)

- **`hplv/resample.py`** — `resample_framebatch(batch, source_fps, target_fps, sync_info)`

  - FPS resampling with A/V sync alignment
  - Pads audio with silence when video starts before audio
  - Handles audio shorter than video
  - Nearest-neighbor frame selection + silence padding
  - Libraries: numpy, torch (dual support)

- **`hplv/backends/pyav_audio_backend.py`** — `PyAVAudioBackend`

  - PyAV-based audio decoder
  - Supported formats: WAV, MP3, FLAC, AAC, and any FFmpeg format
  - Extract audio from video containers (MP4, MKV, AVI, etc.)
  - Multi-stream audio selection, target sample rate / channel conversion
  - Resampling via PyAV AudioResampler internally
  - Custom error types: `HPLVAudioError`, `HPLVDecodingError`

- **`hplv/backends/pyav_demuxer.py`** — audio helper functions

  - `create_audio_resampler(target_sample_rate, target_channels)` — creates `av.AudioResampler` configured for float32 output (default: 48000 Hz, 2-channel stereo)
  - `resample_audio_frame_to_array(frame, resampler, target_channels)` — converts resampled PyAV frames to numpy/torch arrays, handles planar and interleaved format conversions
  - `flush_audio_resampler(resampler, target_channels)` — flushes remaining samples from resampler buffer
  - `ensure_stereo(samples, target_channels)` — channel conversion (mono <-> stereo)
  - Libraries: PyAV, numpy

#### koba — `lib/koba/`

- **`koba/v2/audio/processors.py`** — `AudioDecodeProcessor` (v2, multi-source)

  - Auto-detects input mode:
    1. Embedded audio from video → uses hplv (PREFERRED)
    2. Standalone audio file → uses torchaudio
    3. Raw bytes → uses torchcodec
  - Configuration:
    - `audio_sample_rate` — target sample rate (default: 16000 Hz)
    - `num_channels` — 1=mono, 2=stereo (default: 1)
    - `normalize` — normalize to [-1, 1] range
    - `vgu_duration` — video-aligned duration control (e.g., "5s")
    - `target_duration_sec` — fixed target duration
    - `too_short_policy` — "pad" or "discard" for short audio
    - `too_long_policy` — "truncate_start" or "truncate_random"
    - `variable_length` — skip trim/pad (for bucketed loading)
    - `max_duration_sec` — maximum duration for variable-length mode
  - Output fields: `audio` (tensor [channels, samples]), `audio_sample_rate`, `audio_duration`, `audio_source` ("file"/"video"/"bytes"), `num_audio_samples`, `_clip_duration_sec`
  - Supported input formats: WAV, MP3, FLAC, AAC, M4A, MP4, MKV, AVI (embedded audio), raw bytes
  - Resampling: `torchaudio.functional.resample`
  - Channel conversion: multi-channel → mono (averaging), mono → stereo (duplication)
  - Core method: `_resample_and_convert(audio, source_sr)` uses `torchaudio.functional.F.resample()`

- **`koba/processor/audio_ops.py`** — `AudioDecoder` (v1)

  - Uses torchcodec for byte decoding
  - Takes audio bytes as input, outputs tensor [num_samples, num_channels]
  - Configuration: target sample_rate, num_channels
  - Libraries: torchcodec, io

#### taro — `lib/taro/`

- **`taro/yielders/audio/audio_sampler.py`** — `AudioSampler`

  - Audio loader from WAV bytes with frame alignment
  - Load from bytes: `torchaudio.load(BytesIO(wav_bytes))`
  - Get frame-aligned audio: extracts audio indices based on frame indices
  - Resample: `torchaudio.transforms.Resample(orig_freq, new_freq)`
  - Mono conversion: `waveform.mean(dim=0)` for stereo → mono
  - Pad/truncate to target audio length
  - Configuration: `pad_to_fps` (default: 25.0), `audio_sample_rate` (default: 44100), `mono` (default: True), `allow_empty_audio`
  - Libraries: torchaudio, torch

- **`taro/yielder_pipelines/audio_pipeline.py`** and **`taro/yielder_pipelines/audio_flattened_parquet_pipeline.py`**

  - Audio/video synchronized loading pipelines using AudioSampler

#### ursa — `lib/ursa/`

- **`ursa/utils/audio_preprocessor.py`** — `AudioPreprocessor`

  - Speech-to-video audio preprocessing
  - Input modes: `from_path(path)` via torchaudio.load, `from_tensor(audio, sample_rate)`
  - Operations: resample (`torchaudio.transforms.Resample`), RMS-based normalization, mono/stereo conversion, dtype handling (int16/int32/uint8 → float32), silence padding at start and middle (chunks)
  - Configuration: `sample_rate` (default: 16000), `as_mono` (default: True), `normalize_audio` (default: False), `desired_rms` (default: 0.1), `output_dtype` (default: float32)
  - Libraries: torchaudio, torch, numpy

- **`ursa/eval_job_video_utils.py`** — `save_audio(path, audio, sample_rate)`

  - Saves audio tensor to WAV file using **scipy.io.wavfile.write**
  - Handles shape transpose for librosa compatibility

### Data Pipelines (projects/lax/)

#### Fidelity Pipeline — `projects/lax/lax/projects/audio_pipeline/fidelity/`

- **`processor.py`** — `FidelityProcessor`

  - Purpose: audio bandwidth, aesthetics scoring, sound event detection
  - Input: audio_bytes (binary) + sample_rate
  - Outputs: `bandwidth_hz` (spectral rolloff), `aes_ce`/`aes_cu`/`aes_pc`/`aes_pq` (aesthetic scores via WavLM-based audiobox-aesthetics), `sound_events` (per-second acoustic event detections via PANNs CNN14)
  - Core operations:
    1. Decode audio bytes via io.BytesIO + librosa/torchaudio
    2. Resampling via **librosa.resample** with `resample_type="soxr_hq"` (libsoxr, 3-5x faster than kaiser_best)
    3. Bandwidth: FFT (n_fft=2048, hop_length=512) + spectral rolloff at 95% energy
    4. Aesthetics: Meta audiobox-aesthetics model (WavLM-based)
    5. Sound events: PANNs CNN14_DecisionLevelMax (527 AudioSet classes, ~10ms resolution)
  - Resampling optimization:
    - Pre-resample to 16kHz before aesthetics (makes internal torchaudio.functional.resample a no-op)
    - Resample directly from original SR to 32kHz for PANNs (no chained resampling)
    - Uses libsoxr C library via librosa for high-speed, high-quality resampling
  - Configuration: `resample_type` ("soxr_hq" default), `audio_key` (default: "audio_bytes"), `enable_aesthetics`, `enable_sed`, `aesthetics_device`, `sed_device`
  - Libraries: librosa, torch, audiobox_aesthetics, panns_inference

#### Audio Segmentation — `projects/lax/lax/projects/audio_segmentation/`

- **`mp4_audio_extractor.py`** — `Mp4AudioExtractor`

  - Extract and convert audio from MP4 to WAV 16kHz mono
  - Operations: read MP4 via lumastore → PyAV decode → frame to ndarray → mono conversion (`audio_data.mean(axis=1)`) → resample via **librosa.resample** → normalize int16/int32 to float32 → encode via **soundfile.write** (WAV PCM_16)
  - Configuration: `sample_rate` (default: 16000), `channels` (default: 1)
  - Libraries: PyAV, librosa, soundfile, lumastore

- **`mp4_audio_passthrough_extractor.py`** — `Mp4AudioPassthroughExtractor`

  - Zero-transcode audio extraction from MP4 (preserves original format — AAC, MP3, etc.)
  - Operations: read MP4 via lumastore → open container via PyAV → extract audio stream → remux to M4A (raw packet copy, no transcoding)
  - Benefits: no quality loss, preserves compression, very fast
  - Limitations: downstream processors must handle multiple formats
  - Libraries: PyAV, lumastore

#### Podcast Pipelines — `projects/lax/lax/projects/audio_segmentation/podcast/`

- **`mp3_audio_loader.py`** — `Mp3AudioLoader` (flat_map processor)

  - Input: S3/local audio path
  - Output: raw audio bytes via `lsu.cat(audio_path)`
  - Error handling: returns empty list for 404/missing files (graceful degradation)
  - Libraries: lumastore

- **`podcast_audio_segmentation_processor.py`** — podcast-specific audio segmentation

#### ASR Podcast — `projects/lax/lax/projects/asr/podcast/`

- **`mp3_audio_loader.py`** — similar to audio_segmentation variant, loads MP3 files for ASR processing
- **`vocal_separator.py`** — vocal/instrumental separation for podcast audio

#### Audio Duration — `projects/lax/lax/projects/audio_duration/`

- **`audio_duration_processor.py`** — computes audio duration from audio bytes using librosa, outputs duration in seconds

### Key Libraries Used

| Library | Locations | Purpose |
|---|---|---|
| **torchaudio** | 8 | Primary for file loading and resampling |
| **PyAV/av** | 5 | FFmpeg Python bindings for advanced formats |
| **soundfile/sf** | 4 | WAV/FLAC I/O |
| **torch** | 6 | Tensor operations |
| **librosa** | 3 | Resampling and feature extraction |
| **lumastore** | 3 | Remote file access (S3, Ki, Fra, Gs) |
| **torchcodec** | 2 | Byte decoding from raw bytes |
| **julius** | 1 | High-quality sinc resampling |
| **scipy.io.wavfile** | 1 | WAV writing |
| **ffmpy** | 1 | FFmpeg subprocess wrapper |

---

## Resampling Methods

### Six Backends in the Repo

| Backend | Quality | Speed | Where Used | Default Params |
|---|---|---|---|---|
| **julius.resample_frac** | Sinc (highest) | GPU-optimized | audiotools | Sinc interpolation |
| **torchaudio.transforms.Resample** | High (kaiser best) | GPU/CPU | koba, taro, ursa | Configurable kwargs |
| **torchaudio.functional.F.resample** | High (kaiser best) | GPU/CPU | koba v2 audio | Default kaiser_best |
| **librosa.resample** | High (soxr or kaiser) | Variable | lax/mp4_extractor, fidelity | soxr_hq (~3-5x faster) |
| **PyAV av.AudioResampler** | High (FFmpeg) | FFmpeg-tuned | hplv | float32, configurable layout |
| **FFmpeg aresample filter** | FFmpeg default | Good | audiotools/ffmpeg | aresample=async=1000 |

### Common Target Sample Rates

- **16,000 Hz** — speech/ASR models (most common default)
- **32,000 Hz** — PANNs sound event detection
- **44,100 Hz** — music/general (CD quality)
- **48,000 Hz** — video audio (hplv default)

Automatic detection: most processors detect source SR and resample only when necessary.

### Input Priority (koba v2)

Koba v2's `AudioDecodeProcessor` uses a priority order (auto-detected):

1. **Standalone audio file** (e.g., `data_paths.wav`) → torchaudio
2. **Audio bytes** (e.g., `audio_bytes`) → torchcodec (fastest for bytes)
3. **Video with embedded audio** (e.g., `media_clip`) → hplv (most flexible)

This design ensures optimal performance for each input type.

---

## Bandwidth Measurement

### How is bandwidth calculated in the fidelity pipeline?

The fidelity pipeline computes bandwidth via **spectral rolloff at 95% energy** (`roll_percent=0.95`).

Implementation in `processor.py:203-224` (`_compute_bandwidth`):

1. For normal-length clips (>= n_fft=2048 samples):
   - Compute `librosa.feature.spectral_rolloff` per frame (n_fft=2048, hop_length=512)
   - This gives the frequency below which 95% of the **squared magnitude** (energy) lies, for each frame
   - Take the **median** across all frames → single `bandwidth_hz` value

2. For short clips (< 2048 samples), fallback:
   - Manual FFT via `np.fft.rfft`
   - Compute cumulative sum of squared magnitudes
   - Find frequency where cumulative energy reaches 95% of total via `np.searchsorted`

Configuration defaults:

- `roll_percent`: 0.95
- `n_fft`: 2048
- `hop_length`: 512

### Is there any other version of bandwidth calculation in the repo?

No. The fidelity pipeline's `_compute_bandwidth()` is the **only** audio spectral bandwidth calculation in the repo. Other uses of the word "bandwidth" in the codebase (e.g., stable-audio-tools) refer to Gaussian kernel bandwidth for MMD loss — unrelated to audio spectral bandwidth.

---

## Upsampling and Artifacts

### What is the Nyquist frequency?

The **Nyquist frequency** is half the sample rate. It is the maximum frequency that can be represented in a digital audio signal.

- 16kHz sample rate → Nyquist = 8kHz (can only represent frequencies up to 8kHz)
- 48kHz sample rate → Nyquist = 24kHz

When you upsample 16kHz → 48kHz, the new Nyquist becomes 24kHz, but there is no real audio content between 8-24kHz — those frequencies were never captured in the original recording.

### Does upsampling to 48kHz affect measured bandwidth?

**No.** None of the 6 resampling backends will extend the actual bandwidth. All apply an anti-imaging low-pass filter at the original Nyquist frequency, so the spectrum above the original Nyquist remains near-zero.

Example: upsampling 16kHz audio → 48kHz:

```
0-8kHz:    ████████████████  (original content, preserved)
8-24kHz:   ................  (silence / near-zero — no real content)
```

If you measure `bandwidth_hz` (spectral rolloff at 95% energy) **after** upsampling to 48kHz, you will get the **true original bandwidth** — the rolloff will still land around 8kHz for a 16kHz source. This is actually desirable: upsampling does not inflate the bandwidth metric.

The only edge case: some resamplers have slightly different **transition band steepness** near the original Nyquist. For example, `torchaudio` with default `rolloff=0.99` cuts off slightly below Nyquist (~7.92kHz for a 16kHz source), while `soxr_hq` has a steeper rolloff preserving closer to the full 8kHz. The difference is negligible for practical bandwidth measurement.

### Filter-level artifacts by backend

These are minor artifacts introduced by the resampling filter itself. They differ slightly by backend but are unlikely to matter for model training.

| Backend | Ringing (Gibbs) | Passband Ripple | Rolloff Behavior | Severity for TTA |
|---|---|---|---|---|
| **julius** | Moderate — ideal sinc has the most ringing around transients | Very low | Sharp cutoff at Nyquist | Low |
| **torchaudio** (kaiser) | Controlled by beta param | Low | `rolloff=0.99` attenuates top ~1% of band | Low |
| **librosa soxr_hq** | Well-controlled | Very low | Very steep, preserves almost full band | **Lowest** |
| **PyAV/libswresample** | Moderate | Moderate | Less steep than soxr | Low-moderate |
| **FFmpeg aresample** | Same as PyAV | Same as PyAV | Same as PyAV | Low-moderate |

Details on each artifact type:

- **Passband ripple** — small amplitude variations in the original frequency range
- **Transition band behavior** — how the filter rolls off near Nyquist
- **Pre-ringing / post-ringing (Gibbs phenomenon)** — sinc-based filters cause temporal ringing around transients
- **Phase distortion** — linear phase vs minimum phase filters
- **Numerical precision** — float32 vs float64 processing
- **Spectral imaging** — if anti-imaging filter is imperfect, mirror images appear above Nyquist

### The real problem: spectral gap (major)

The dominant artifact from upsampling is **not** from the filter — it is from the **empty spectrum above the original Nyquist**. This is the same across all 6 backends and is the primary concern for text-to-audio (TTA) model training.

#### Why this matters for TTA training

1. **The model learns a false spectral prior.** If you mix genuinely wideband 48kHz data (e.g., studio music) with upsampled 16kHz→48kHz data (e.g., podcasts), the model sees inconsistent high-frequency distributions. It may learn to produce muffled output or hallucinate high frequencies unpredictably.

2. **Spectrogram/codec representations amplify the gap.** If the TTA model uses mel spectrograms or neural audio codecs (EnCodec, DAC, etc.), the empty high-frequency bins become explicit zero-valued features. The model must learn when to "fill in" vs "leave empty" — this is a hard, ambiguous task.

3. **Loss function confusion.** During training, the model gets penalized for generating high-frequency content on upsampled samples (where the target is silence above 8kHz), but rewarded for it on genuine 48kHz samples. This creates conflicting gradients.

### Recommendations for podcast data in TTA training

Given that podcast audio is typically 16-24kHz source sample rate being upsampled to 48kHz:

- **Option A: Train at native sample rate.** Do not upsample — keep podcasts at 16kHz or 24kHz and either train a separate model / use a separate codec for this sample rate, or use `bandwidth_hz` from the fidelity pipeline as a conditioning signal so the model knows the true bandwidth.

- **Option B: Upsample but condition on bandwidth.** Upsample to 48kHz for uniform data format, but include `bandwidth_hz` as metadata so the model can learn the bandwidth distribution. At inference, condition on full bandwidth.

- **Option C: Filter out low-bandwidth samples.** Use the fidelity pipeline's `bandwidth_hz` to exclude samples below a threshold (e.g., <20kHz), ensuring all 48kHz training data has genuine wideband content.

- **Option D: Bandwidth extension model.** Apply a neural bandwidth extension model (e.g., AudioSR) before training to synthesize plausible high-frequency content. Adds compute and potential artifacts of its own.

**Bottom line**: The choice of resampler barely matters — `soxr_hq` is marginally cleanest. The real risk is the empty spectrum above the original Nyquist, which affects model training regardless of backend. Use the `bandwidth_hz` metric to handle this at the data curation level.

---

## Demucs Source Separation in the ASR/Diarize Pipeline

Notes from investigation of `projects/lax/lax/projects/av_data_processing/audio/asr_diarize/` (2026-04-08).

### What is Demucs?

Demucs (htdemucs variant) is Meta's **Hybrid Transformer Demucs** — a U-Net style encoder-decoder for music source separation. It is **not autoregressive**; it processes audio in a single forward pass. Architecture:

- Two parallel U-Net branches: one on raw waveforms (temporal), one on spectrograms (spectral)
- Cross-attention transformer layers bridging the two branches in the bottleneck
- Outputs all stems (vocals, drums, bass, other) simultaneously

Model size: ~80–85M parameters, ~330MB in float32 weights.

### How It's Used

Demucs is invoked as a **Python library** (not CLI) in `models/vocal_separator.py`:

```python
import demucs.pretrained
from demucs.apply import apply_model

model = demucs.pretrained.get_model("htdemucs")
sources = apply_model(model, waveform, overlap=0.25, shifts=1)
```

Configuration:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model_name` | `htdemucs` | Hybrid Transformer variant |
| `device` | `cuda` | GPU inference |
| `overlap` | `0.25` | Overlap ratio between chunks |
| `shifts` | `1` | No extra random shifts (fastest) |
| `segment_length` | `None` | Uses model default chunking |

### Pipeline Flow

3-stage pipeline defined in `podcast_asr_pipeline.py` → `run_podcast_pipeline_gpu_with_separation()`:

```
Stage 1: Mp3AudioLoader
  └─ Download audio from S3

Stage 2: VocalSeparator (Demucs)
  └─ Load audio at model native SR (44.1kHz)
  └─ Run htdemucs, extract "vocals" stem only
  └─ Convert to mono WAV → output as vocal_bytes
  └─ Pass original audio through as audio_bytes
  └─ On failure: graceful fallback to original audio

Stage 3: PodcastV2Processor
  └─ Run WhisperX ASR on clean vocals (vocal_bytes)
  └─ Run diarization on original or vocals (configurable)
  └─ Segment into 3–15s TTS training clips
```

### When to Use

This is an **opt-in pipeline** for noisy audio (field recordings, live events). Clean podcast audio uses `run_podcast_pipeline_gpu()` without Demucs.

### Diarization Trade-off

A `diarize_on_original` flag in `processor.py` controls the diarization source:

- **`True` (on original)**: Better speaker embeddings, but background music may confuse diarization
- **`False` (on vocals)**: Cleaner audio, but Demucs may alter speaker embeddings

### GPU Memory for Large-Scale Processing

For large-scale short audio datasets (~10s clips) on H100 (80GB HBM3):

- Model weights: ~330MB per copy (~1.6GB for 5 copies)
- Activations for 10s audio: a few hundred MB per inference
- **5+ actors per GPU is very comfortable** — well under 10GB total, small fraction of 80GB
- Bottleneck is likely compute throughput, not memory
- Consider sharing a single model across actors for maximum memory efficiency

---

## internal_audio_v1 Fidelity Pipeline Run (2026-04-08)

### Goal

Compute audio fidelity metadata (bandwidth, AES, SED, EAT) for the `internal_audio_v1` source table and save to the fidelity output table.

| Stage | Dataset | S3 URI |
|-------|---------|--------|
| **Source** | `internal_audio_v1_captioned_original_sr_v3_merged_v4` | `s3://ai-lumalabs-datasets-ap-se-2/inkyu/lax/audio_segmentation/internal_audio_v1_captioned_original_sr_v3_merged_v4.lance` |
| **Fidelity output** | `internal_audio_v1_fidelity_p{N}of8` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_fidelity_p{N}of8.lance` |

- **Source rows**: 92,221,138 (~92M)
- **Partitioning**: 8 partitions (p0-p7), ~11.5M rows each
- **Audio column**: `audio_bytes_ori`
- **Branch**: `dongguo/audio-metadata` at commit `2644403` ("[lax] Override __call__ in FidelityProcessor to skip per-row gc/cache clear")

### Throughput Bottleneck Fix

`BaseProcessor.__call__` (in `lax/core/processors/base_processor.py`) runs `gc.collect()` + `torch.cuda.empty_cache()` after every row (added in PRs #6914 and #6991). For lightweight GPU processors like FidelityProcessor (~14ms/row ONNX inference), this overhead causes ~290x throughput regression (380 rows/s → 1.3 rows/s).

The fix: `FidelityProcessor` overrides `__call__` to skip these calls. This was originally implemented in commit `2644403` (Apr 3), accidentally removed in the next commit `d19d2cc` during documentation cleanup, then restored on Apr 8 by resetting the branch back to `2644403`.

### First Attempt — Failed (Apr 8)

Launched 8 partitions on kiwi clusters (`dongguo-metadata-tmp-*` and `dongguo-vibevoice-tmp-s*`). Issues:
- Code was from branch tip `35ec70f` which lacked the `__call__` override → throughput bottleneck
- 4 of 8 clusters had 0 available workers → jobs failed with `ValueError: min_size must be >= 1`
- 1 cluster (p6 on `dongguo-vibevoice-tmp-s3-543e8a`) had no GPU worker → stuck pending
- All jobs stopped, all `dongguo-vibevoice-tmp-s*` clusters deleted

### Second Attempt — Running (Apr 8)

After resetting `dongguo/audio-metadata` to `2644403` (with `__call__` override), relaunched all 8 partitions:

| Partition | Cluster | Location | Job ID |
|-----------|---------|----------|--------|
| p0 | `dongguo-metadata-tmp-5cfb4d` | kiwi-flyte | `raysubmit_ttt5TKXNA8LVxATY` |
| p1 | `dongguo-vibevoice-omniva-s6-fecd96` | omniva-flyte | `raysubmit_TLhgEd9jNV1bfRhJ` |
| p2 | `dongguo-vibevoice-omniva-s0-b76d15` | omniva-flyte | `raysubmit_Z8pnJu8s97JQRDgS` |
| p3 | `dongguo-vibevoice-omniva-s1-502299` | omniva-flyte | `raysubmit_DekX5cajTSmNXh73` |
| p4 | `dongguo-vibevoice-omniva-s2-dfb184` | omniva-flyte | `raysubmit_ejwt3F4ThBtsDqKZ` |
| p5 | `dongguo-vibevoice-omniva-s3-9ec3db` | omniva-flyte | `raysubmit_XYJUHz5EsxqDt4Wf` |
| p6 | `dongguo-vibevoice-omniva-s4-eef6d4` | omniva-flyte | `raysubmit_eNBZ8YcJ6wVcDMiX` |
| p7 | `dongguo-vibevoice-omniva-s5-2e0b2e` | omniva-flyte | `raysubmit_szGErCCsmC3mkEcv` |

Each cluster: 1 node × 8 H100 GPUs. Expected throughput ~380 rows/s per node.

### Checking Job Status

```bash
# Setup proxy (kiwi for p0, omniva for p1-p7)
source scripts/setup-ray-proxy.sh kiwi-flyte dongguo-metadata-tmp-5cfb4d
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-vibevoice-omniva-s6-fecd96

# Check status
ray job status <job_id>
ray job logs <job_id>
```

### Kiwi vs Omniva Throughput Analysis

p0 (kiwi) processes at ~3.4 fragments/min vs p2-p6 (omniva) at ~1.1 fragments/min — omniva runs at **~30% of kiwi's throughput** for this pipeline.

**Root cause: cross-region S3 read latency.** The source Lance table is in `s3://ai-lumalabs-datasets-ap-se-2` (Sydney). Kiwi is also in Sydney → low-latency reads. Omniva is in US → every S3 read has cross-region latency.

Evidence from job logs (object store ramp-up in first 40s):
- **p0 (kiwi):** 0 → 25 → 42 → 60 → 74 → 90 → 106 → 119 GiB — reader floods the object store
- **p2 (omniva):** stuck at 3.4 GiB — reader starves the GPU actors

The pipeline config is identical (same image, `num_gpus=0.25`, `min_concurrency=32`). GPUs are 8/8 allocated on both, but omniva actors idle waiting for data.

**Cluster selection rule of thumb:**

| Per-row latency | Bottleneck | Use omniva? |
|----------------|------------|-------------|
| <50ms (fidelity, bandwidth) | Reader / S3 I/O | No — use kiwi (same region as S3) |
| >500ms (VLM captioning, LLM) | GPU compute | Yes — cross-region latency negligible |

For compute-intensive tasks (e.g. multimodal LLM dense captioning at seconds/row), the GPU processing time dominates and cross-region S3 latency becomes negligible. The reader easily keeps up because actors consume data slowly. Omniva's available GPU capacity makes it a good fit for these workloads.

### Post-Processing (TODO)

After all 8 partitions complete:
1. Merge 8 partition tables into final `internal_audio_v1_fidelity.lance`
2. Run fidelity filter to produce filter scores table
3. Update README dataset table with final URIs