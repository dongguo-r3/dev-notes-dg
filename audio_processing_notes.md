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