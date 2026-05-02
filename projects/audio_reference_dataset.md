# Audio Reference Dataset Survey + Sample Pull

Notes from a working session on 2026-04-30. Goal: pick open-sourced Hugging Face datasets to train a "TTS with reference speech" model, where each training example has the structure:

```
Here are some speech by multiple speakers:
<speaker0> [audio by speaker 0]
<speaker1> [audio by speaker 1]
...
<speakerM> [audio by speaker M]
Speak out the following transcripts using the sound of speaker-k: <transcript_to_speak>
[audio to generate by a diffusion model]
```

`M` can be 1 (single reference) or larger (multi-shot prompt). The model picks one of the M references and synthesizes the target text in that speaker's voice.

## 1. What "TTS with reference speech" looks like as a dataset

Two distinct packagings exist on the Hub:

- **Training packaging.** Multi-speaker corpus where each utterance carries a `speaker_id`. The training pipeline samples N utterances from the same speaker on the fly — one becomes the reference prompt, another becomes the target. Almost every voice-cloning model (XTTS, F5-TTS, CosyVoice, Llasa, VoxCPM, …) is trained this way; you do not usually find pre-baked `(ref, target_text, target_audio)` triples.
- **Evaluation packaging.** Pre-baked triples: a reference audio clip + a target text whose target speaker is held out at training time.

For our M-shot construction we want training packaging — and we'll build the M-shot prompt at sampling time.

## 2. Survey of training-side datasets

| Dataset | Hours | Languages | Notes |
|---|---|---|---|
| [amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) | ~216k (Emilia-Large) | EN/ZH/DE/FR/JA/KO | In-the-wild podcasts/talk shows. Per-clip `speaker` field. Current SOTA-scale TTS pretraining set. CC BY-NC 4.0 + CC BY 4.0 (YODAS half) |
| [TTS-AGI/emilia-yodas](https://huggingface.co/datasets/TTS-AGI/emilia-yodas) | ~114k | multilingual | The CC BY 4.0 (commercial-friendly) half of Emilia-Large |
| [Wenetspeech4TTS/WenetSpeech4TTS](https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS) | 12,800 | Mandarin | Re-segmented by speaker similarity specifically for TTS |
| [mythicinfinity/libritts](https://huggingface.co/datasets/mythicinfinity/libritts) | 585 | EN | 24 kHz, ~2,500 audiobook readers, gold transcripts. Default multi-speaker baseline |
| [parler-tts/mls_eng](https://huggingface.co/datasets/parler-tts/mls_eng) | ~44k | EN | English MLS audiobooks with stable speaker IDs |
| [speechcolab/gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech) | 10k | EN | Audiobooks/podcasts/YT; weaker per-speaker grouping |
| VCTK (multiple Hub mirrors) | ~44 | EN | 110 speakers, classic TTS benchmark |
| HiFi-TTS / Expresso (parler-tts mirrors) | small | EN | Higher fidelity / emotional axes |
| [legacy-datasets/common_voice](https://huggingface.co/datasets/legacy-datasets/common_voice) | 7k+ validated | 60+ langs | `client_id` ≈ speaker. Quality varies; only realistic option for many low-resource languages |

The Emilia paper showed that training on Emilia matches MLS audiobooks for speaker similarity and intelligibility, so for production-grade voice-cloning training Emilia is the de facto choice.

## 3. Survey of eval-side datasets

| Dataset | Notes |
|---|---|
| [zhaochenyang20/seed-tts-eval](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval) | ByteDance Seed-TTS eval set, EN+ZH. Each row is a prompt audio (held-out speaker) + target text. Used for WER + speaker-similarity. Also has tongue-twister / repetition stress sets |
| [MiniMaxAI/TTS-Multilingual-Test-Set](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set) | Multilingual zero-shot voice cloning eval |

## 4. Deep dive: Emilia-YODAS + MLS-eng

The pair we settled on. Both are CC BY 4.0 (commercial-friendly).

| | parler-tts/mls_eng | TTS-AGI/emilia-yodas |
|---|---|---|
| **Disk** | ~705 GB (FLAC) | ~2.14 TB (MP3) |
| **Hours** | ~44,660 | ~113,900 |
| **Utterances** | ~10.8 M | ~24.7 M |
| **Unique speakers** | 5,490 (gold IDs) | tens of thousands of pseudo-IDs (per-source diarization) |
| **Languages** | EN only | EN, ZH, DE, FR, JA, KO |
| **Format / SR** | FLAC (48 kHz in parler mirror) | MP3, 24 kHz |
| **Transcripts** | Gold (book-aligned) | Whisper ASR |
| **Per-clip length** | ~10–20 s | ~3–20+ s |
| **License** | CC BY 4.0 | CC BY 4.0 |

### MLS-eng (parler-tts mirror)

- Speaker IDs are reliable and **global** — each LibriVox reader has a persistent ID across all their books.
- Many utterances per speaker (dozens to hundreds), great for sampling distinct ref/target pairs.
- Gold, book-aligned transcripts.
- Single domain (read audiobook prose), single language. Voice-diverse but style-monotonic.
- Sharded as 1,418 parquet files (`data/train-{NNNNN}-of-01416.parquet`).

### Emilia-YODAS

- Speaker IDs are **pseudo-IDs from diarization within a source** (`{LANG}_B{book}_S{speaker}`). Stable within a podcast/video, **not across sources**. Same person on two different podcasts gets two different IDs. Usable, with label noise from diarization errors.
- Multilingual, in-the-wild styles (podcasts, talk shows, debates, interviews) — adds the spontaneous/expressive prosody MLS lacks.
- Whisper-ASR'd transcripts (not gold).
- Sharded as WebDataset tar files: `{LANG}/{LANG}-B{NNNNNN}.tar`. Each tar contains paired `.mp3` + `.json`.
- Languages: DE, EN, FR, JA, KO, ZH. Tars are sharded by language directory.

## 5. Pitfalls for the M-shot prompt format

Designing around these matters:

- **Don't mix the M reference speakers across datasets in the same prompt.** If 3 references are MLS audiobook reads and one is an Emilia podcast clip, the model can identify speaker-k from domain/recording-channel cues alone and skip learning timbre. Keep all M references from the same source dataset (and ideally same source bucket for Emilia).
- **Verify Emilia speaker labels before training.** Re-embed all utterances of a claimed speaker with a speaker-verification model (WavLM-large-SV, ECAPA-TDNN) and reject the speaker if intra-cluster cosine similarity is below a threshold. Cheap, removes a lot of diarization noise.
- **Filter by per-speaker utterance count.** Need at least `K = max(M) + 1` clips per speaker (M references + 1 target). Knocks out the long tail in Emilia; almost everyone passes in MLS.
- **Length-filter both refs and target.** 3–15 s is typical. Both datasets are pre-segmented but distributions differ.
- **Distractor sampling matters.** Draw the M-1 distractor speakers from the same dataset and ideally same language. Random distractors are too easy.
- **License hygiene.** Emilia-YODAS is CC BY 4.0, MLS is CC BY 4.0. If you ever pull in the non-YODAS half of Emilia (CC BY-NC 4.0), keep it in a separate pool you can drop for any commercial release.
- **Sample-rate hygiene.** MLS is 48 kHz in the parler mirror, Emilia is 24 kHz. Resample to one rate up front.

## 6. Recipe: pull a 1K-sample slice (re-runnable)

We pulled 1K samples from each into `/fsx/dongguo/adhoc/dataset/{mls_eng,emilia_yodas_en}/`. Each sample is a `.flac` file plus a row in `metadata.jsonl`.

### 6a. Sharp edge: streaming MLS in-order gives one speaker

The parquet shards are sorted by `speaker_id`. A naive `load_dataset(..., streaming=True)` of mls_eng yielded 1000 utterances all from speaker 4800 (one book). Useless for M-shot.

**Fix:** round-robin across N evenly-spaced parquet shards. With 50 shards × 20 rows we got broad speaker coverage in the 1K slice.

Emilia-YODAS streams cleanly because the WebDataset tars are already organized by source rather than speaker, so the natural read order interleaves speakers. The first DE shard naturally yields ~31 unique speakers in 1K rows.

### 6b. Self-contained download script

Save as `download_samples.py`. Runs anywhere with `datasets`, `soundfile`, `torch`, `torchcodec`, `huggingface_hub`. Adjust `ROOT` for your machine.

```python
"""Stream ~1K samples each from parler-tts/mls_eng and TTS-AGI/emilia-yodas (EN)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from datasets import load_dataset

ROOT = Path("./dataset")  # override per machine
N = 1000


def audio_to_numpy(decoder):
    samples = decoder.get_all_samples()
    data = samples.data
    sr = samples.sample_rate
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data.ndim == 2 and data.shape[0] in (1, 2):
        data = data.T
    return data, sr


def dump_mls():
    """Round-robin across evenly-spaced parquet shards for speaker diversity."""
    out_dir = ROOT / "mls_eng"
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    n_shards = 50
    rows_per_shard = N // n_shards  # 20
    total_train = 1416
    step = total_train / n_shards
    shard_files = [
        f"data/train-{int(i * step):05d}-of-{total_train:05d}.parquet"
        for i in range(n_shards)
    ]

    print(f"[mls_eng] {n_shards} shards x {rows_per_shard} rows -> {out_dir}", flush=True)
    written = 0
    with meta_path.open("w") as f:
        for shard_idx, shard_file in enumerate(shard_files):
            ds = load_dataset(
                "parler-tts/mls_eng",
                split="train",
                streaming=True,
                data_files={"train": [shard_file]},
            )
            taken = 0
            for row in ds:
                if taken >= rows_per_shard:
                    break
                arr, sr = audio_to_numpy(row["audio"])
                key = f"mls_{written:05d}_spk{row['speaker_id']}_book{row['book_id']}"
                rel_audio = f"audio/{key}.flac"
                sf.write(out_dir / rel_audio, arr, sr, format="FLAC")
                f.write(json.dumps({
                    "key": key,
                    "audio_path": rel_audio,
                    "sample_rate": int(sr),
                    "speaker_id": row["speaker_id"],
                    "book_id": row["book_id"],
                    "transcript": row["transcript"],
                    "audio_duration": row["audio_duration"],
                    "begin_time": row["begin_time"],
                    "end_time": row["end_time"],
                    "original_path": row["original_path"],
                    "source_shard": shard_file,
                }, ensure_ascii=False) + "\n")
                written += 1
                taken += 1
            if (shard_idx + 1) % 5 == 0:
                print(f"[mls_eng] shard {shard_idx + 1}/{n_shards}, written={written}", flush=True)
    print(f"[mls_eng] done: {written} samples", flush=True)


def dump_emilia_en():
    out_dir = ROOT / "emilia_yodas_en"
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    print(f"[emilia_en] streaming -> {out_dir}", flush=True)
    ds = load_dataset(
        "TTS-AGI/emilia-yodas",
        split="train",
        streaming=True,
        data_files={"train": ["EN/EN-B000000.tar", "EN/EN-B000001.tar"]},
    )
    written = 0
    with meta_path.open("w") as f:
        for row in ds:
            if written >= N:
                break
            j = row["json"]
            if j.get("language") != "en":
                continue
            arr, sr = audio_to_numpy(row["mp3"])
            key = j["_id"]
            rel_audio = f"audio/{key}.flac"
            sf.write(out_dir / rel_audio, arr, sr, format="FLAC")
            f.write(json.dumps({
                "key": key,
                "audio_path": rel_audio,
                "sample_rate": int(sr),
                "speaker": j["speaker"],
                "language": j["language"],
                "duration": j["duration"],
                "dnsmos": j["dnsmos"],
                "phone_count": j["phone_count"],
                "text": j["text"],
                "tar_url": row.get("__url__"),
            }, ensure_ascii=False) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"[emilia_en] {written}/{N}", flush=True)
    print(f"[emilia_en] done: {written} samples", flush=True)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "both"
    if target in ("mls", "both"):
        dump_mls()
    if target in ("emilia", "both"):
        dump_emilia_en()
```

Run:

```bash
python download_samples.py both
# or just one:
python download_samples.py mls
python download_samples.py emilia
```

## 7. Re-running on macOS

Two practical concerns on Mac:

1. **`torchcodec` / FFmpeg**. The `datasets` library decodes audio via `torchcodec`, which links against FFmpeg. Install FFmpeg first:

   ```bash
   brew install ffmpeg
   ```

   Then in a fresh Python env:

   ```bash
   uv venv .venv && source .venv/bin/activate
   uv pip install datasets soundfile huggingface_hub torch torchcodec
   ```

   If torchcodec fails to find FFmpeg, set `DYLD_LIBRARY_PATH` to the Homebrew lib dir (`/opt/homebrew/lib` on Apple Silicon, `/usr/local/lib` on Intel).

2. **HF auth** is not needed for these two datasets — both are public. If you hit a gated-repo error, run `huggingface-cli login` once.

3. **Disk budget for 1K samples**: ~700 MB total (~406 MB MLS + ~285 MB Emilia EN). Streaming reads roughly 2–3× that off the network because parquet shards are pulled whole.

4. **Adjust `ROOT`** in the script to e.g. `Path("~/data/audio_reference_samples").expanduser()`.

5. **Listening on Mac**: any FLAC-aware player works (QuickTime, VLC). For inline browsing, a quick `streamlit` app reading `metadata.jsonl` and rendering `st.audio(audio_path)` per row makes A/B listening trivial:

   ```python
   # listen.py — streamlit run listen.py
   import json
   from pathlib import Path
   import streamlit as st

   ROOT = Path(__file__).parent
   ds = st.selectbox("dataset", ["mls_eng", "emilia_yodas_en"])
   rows = [json.loads(l) for l in (ROOT / ds / "metadata.jsonl").open()]
   spk_field = "speaker_id" if ds == "mls_eng" else "speaker"
   spk = st.selectbox("speaker", sorted({r[spk_field] for r in rows}))
   for r in [r for r in rows if r[spk_field] == spk][:20]:
       st.write(r.get("transcript") or r.get("text"))
       st.audio(str(ROOT / ds / r["audio_path"]))
   ```

## 8. What we observed in the 1K-sample slice

(Numbers from the actual pull on the FSx machine; Mac re-run should match modulo shard contents.)

| | mls_eng (round-robin) | emilia_yodas_en |
|---|---|---|
| Rows | 1000 | 1000 |
| Disk | 416 MB | 285 MB |
| Total audio | 4.13 h | 2.69 h |
| Unique speakers | 48 (50 books) | 31 |
| Per-speaker count | min 20 / median 20 / max 40 | min 1 / median 16 / max 185 |
| Top-3 speaker share | 10% | ~48% |
| Sample rate | 48 kHz FLAC | 24 kHz FLAC |

The MLS round-robin pull was triggered after the naive in-order pull yielded **1 unique speaker** for 1000 rows. The fix (50 shards × 20 rows) brought speaker count from 1 → 48 with an evenly-distributed top tail. Always shard-randomize when sampling MLS for prompt construction.

The Emilia EN slice is heavy-tailed (top 3 speakers ≈ 48% of clips) because the first two tars happen to be dominated by a few prolific YouTube channels. For a real training mix you'd interleave many more tars.

## 9. Open questions / next steps

- Decide max `M` for the M-shot prompt; that drives the per-speaker minimum utterance count for the speaker filter.
- Run a WavLM-SV cleanup pass on the Emilia speaker IDs before any training — quantify how many speaker clusters survive at e.g. mean intra-cluster cosine ≥ 0.7.
- If we want ZH/multilingual coverage, add the matching Emilia language directories. WenetSpeech4TTS is the alternative for ZH-only at higher quality.
- Decide whether the model should ever see an empty/zero-shot prompt (M=0) or always M ≥ 1.
