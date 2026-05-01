# A2A Training Dataset Table Design (`mls_eng_zs_tts_a2a_v1.lance`)

> **Audience:** Someone tasked with creating the Lance table from a source dataset (initially MLS-Eng).
> **Purpose:** Specifies the row schema, the hard invariants the curator must guarantee, the audio-format requirements, and the loader contract — enough that a curation script can be written from this doc alone.
> **Status:** Schema locked (PR-1b plan, designed 2026-05-01). The table-creation work itself is the next step; this doc is the handoff.

---

## 1. TL;DR

Build a Lance table that pairs a **target audio clip** with **one reference audio clip from the same speaker** and the target's **transcript**. The model trained on this table will learn to synthesize the target transcript spoken in the reference's voice — a "zero-shot text-to-speech" (zs-tts) task, which we expose to the loader as audio-to-audio (a2a).

The table is **per-row**: each row is one self-contained training example (target + 1 ref + text). No external lookups required at training time — audio bytes are stored inline.

Hard requirements per row:

- Target audio: **WAV, 48 kHz, mono**.
- Reference audio: **same speaker as target, sampled from a different recording session** (cross-book preference, see §6); WAV, 48 kHz, mono.
- Reference is **never the same clip as the target**.
- Reference transcript exists in source data but is **not stored** in the Lance row (deliberate — the model should learn from voice properties only, not from ref content).

Storage path:

```
s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/a2a/mls_eng_zs_tts_a2a_v1.lance
```

---

## 2. Task background

### 2.1 What zero-shot TTS / A2A does

A reference-conditioned text-to-speech model. At inference, the user provides:

- A short reference audio clip (~3 s, any speaker) — the **voice cue**.
- A text transcript — what to speak.

The model synthesizes audio that says the transcript in the reference voice. The "zero-shot" part means the speaker doesn't need to be in training — the model generalizes from voice cues, not from a fixed speaker enumeration.

For training, we need (target audio, reference audio, transcript) triples where:

- The target audio is the supervised signal — the model must produce this.
- The reference audio is the conditioning — same speaker as target, but **a different utterance** (so the model can't shortcut by copying audio frame-for-frame).
- The transcript tells the model what's being said in the target.

### 2.2 Why "A2A v1" specifically

This dataset supports the v1 of the A2A pipeline: **M=1 same-speaker reference** per target. Every row has exactly one reference clip, and that clip is from the same speaker as the target.

A future v2 (M ∈ {1, 2, 3}) will support multiple refs per target with distractor-speaker slots, but v1 keeps it minimal: one ref, one match. Schema design preserves a clean v2 upgrade path (see §10).

### 2.3 Why the reference is from a *different* clip

If the curator paired (target = clip X, reference = clip X), the model would learn to copy audio frame-for-frame instead of extracting voice properties. To force voice-property generalization, the reference must come from a **different utterance** by the same speaker. Cross-recording-session (cross-book in MLS-Eng) is even better — see §6 on the curation policy.

---

## 3. Source data assumptions (initial: MLS-Eng)

### 3.1 What we're starting from

Initial source: **Multilingual LibriSpeech, English split** (`mls_eng`). Audiobook recordings from LibriVox, English readings, multi-speaker, multiple clips per speaker.

Key MLS-Eng structure:

- **Audio**: 48 kHz mono FLAC files. We re-encode to WAV bytes for storage (per §5).
- **Transcripts**: gold-quality, sentence-level alignment.
- **Speaker IDs** (`speaker_id`): stable LibriVox-account identifiers. One person = one ID across all their recordings.
- **Book IDs** (`book_id`): each LibriVox recording session corresponds to one book reading; same speaker may record multiple books on different days, in different rooms, possibly with different mic setups.
- **Clip IDs**: a deterministic per-clip identifier, e.g. `mls_spk{speaker_id}_book{book_id}_{clip_index:06d}`.

### 3.2 Future sources

The table format is designed to support additional sources later (e.g. Emilia-YODAS at native 24 kHz, then upsampled to 48 kHz for uniform storage). Each future source gets its own Lance table file (`<source>_zs_tts_a2a_v1.lance`) with the same schema.

---

## 4. Per-row schema specification

A single row is one training example. All fields are top-level (no nested structures other than the list-typed `ref_*` fields).

### 4.1 Columns

```text
─── Identity / namespacing ────────────────────────────────────────────
sample_id                  : str            — globally unique primary key
source_dataset             : str            — "mls_eng" (extensibility tag)
target_language            : str            — "en"

─── Target (the supervised audio + its transcript) ───────────────────
target_audio_bytes         : bytes          — WAV 48 kHz mono
target_audio_sr            : int            — 48000
target_duration            : float          — seconds (used for bucketing in PR-3)
target_text                : str            — transcript of the target audio
target_speaker_global_id   : str            — stable per-speaker ID (e.g. LibriVox account)
target_key                 : str            — provenance handle for the target clip

─── Reference (the voice cue) ────────────────────────────────────────
ref_audio_bytes            : list<binary>   — length 1 in v1; WAV 48 kHz mono
ref_audio_srs              : list<int>      — length 1 in v1; values all 48000
ref_durations              : list<float>    — length 1 in v1; informational
ref_speaker_global_ids     : list<str>      — length 1 in v1; entry equals target_speaker_global_id
ref_keys                   : list<str>      — length 1 in v1; provenance handles

─── Curator metadata ─────────────────────────────────────────────────
M                          : int            — 1 in v1 (always)
target_speaker_local_idx   : int | None     — 0 in v1 (always); None reserved for future
                                               cross-speaker tasks
```

### 4.2 Column-by-column notes

#### `sample_id` (str, primary key)

Globally unique, deterministically derivable from the source. Use:

```
sample_id = f"{source_dataset}/{target_key}"
# e.g.  "mls_eng/mls_spk4800_book10003_000000"
```

This lets the same row be regenerated from source data without value drift, and serves as the dedup key if curation is re-run.

#### `source_dataset` (str)

Free-form tag identifying the source: `"mls_eng"` for v1; `"emilia_yodas"`, etc. for future. Used by mixing pipelines that combine multiple sources in one table; can also serve as a SQL-filter axis for source-specific debugging.

#### `target_language` (str)

ISO-639-1-ish code: `"en"` for English. Used for language-filtering and for future multilingual training mixtures. The loader filter expression (§9.5) typically pins this.

#### `target_audio_bytes` (bytes)

The supervised target audio, encoded as WAV bytes. Format requirements: see §5. Stored as the literal binary contents of a `.wav` file; the loader uses torchcodec to decode.

#### `target_audio_sr` (int)

Sample rate of `target_audio_bytes`. **Must equal 48000 for v1.** The loader asserts this per row at decode time and raises if any row violates it. Acts as defensive insurance against partially-converted shards.

#### `target_duration` (float)

Duration of target audio in seconds. Used by the loader's bucketing scheduler (PR-3) to filter rows by duration bucket (e.g., `target_duration BETWEEN 5.0 AND 10.0`). v1's loader doesn't bucket but the column is needed for future PRs.

Compute as `len(decoded_samples) / sample_rate` after loading the source clip.

#### `target_text` (str)

The transcript that the target audio speaks. The model is trained to produce `target_audio` given `target_text` + the reference voice. UTF-8, no normalization beyond what the source provides (the loader does no normalization either; the tokenizer handles whatever is there).

For MLS-Eng, use the gold transcripts directly. Don't lowercase or strip punctuation unless the source has done so already.

#### `target_speaker_global_id` (str)

Stable speaker identifier from the source. For MLS-Eng, the LibriVox account ID (e.g. `"4800"`). Used for:

- The same-speaker invariant (§7.2): `target_speaker_global_id == ref_speaker_global_ids[0]`.
- Per-speaker frequency analysis during curation (e.g., capping clips per speaker).
- Debug / per-speaker quality probing.

**Convention**: store as a string even if the source uses ints. Avoids ambiguity with later speaker-id systems that aren't integer-shaped.

#### `target_key` (str)

Provenance handle — uniquely identifies the target *clip* within the source. Format suggestion:

```
target_key = f"mls_spk{speaker_id}_book{book_id}_{clip_index:06d}"
# e.g.  "mls_spk4800_book10003_000000"
```

Used for:

- The "target not in ref pool" invariant (§7.1): `target_key NOT IN ref_keys`.
- Forensics — given a sample_id, lets you re-locate the original clip in the source.
- Dedup if curation is re-run.

#### `ref_audio_bytes` (list&lt;binary&gt;, length 1)

The reference audio, stored as a **list of length 1** even in v1 where there's only one ref. See §10 for why we picked list-typed over scalar.

The single entry is WAV bytes for the reference clip — sampled per the §6 policy. Format requirements identical to `target_audio_bytes`.

#### `ref_audio_srs` (list&lt;int&gt;, length 1)

Sample rates of the ref audio entries. v1: list of length 1, value `48000`. Loader asserts every entry is 48000.

#### `ref_durations` (list&lt;float&gt;, length 1)

Duration of each ref audio in seconds. Informational — the loader truncates each ref to a fixed 3 s window before VAE-encoding, so the source duration matters for sampling policy (refs should be > 3 s when possible) but isn't directly consumed at training time.

#### `ref_speaker_global_ids` (list&lt;str&gt;, length 1)

Speaker IDs for each ref entry. **In v1, the single entry must equal `target_speaker_global_id`** — the same-speaker invariant.

#### `ref_keys` (list&lt;str&gt;, length 1)

Provenance handles for each ref clip. Same format convention as `target_key`. The single entry must be different from `target_key` (per the "target not in ref pool" invariant).

#### `M` (int)

Number of reference slots in this row. **Always 1 for v1**. The loader's default SQL filter is `M = 1` — even though every row will already be M=1, the filter acts as a defensive check against schema drift if a v2 row sneaks into a v1-named table.

#### `target_speaker_local_idx` (int | None, **nullable**)

For v1: **always `0`** (the single ref slot is the target speaker by invariant).

For v2 (future): index in `[0, M-1]` pointing at whichever ref slot is from the target speaker. When multiple slots happen to be same-speaker, the curator picks one deterministically.

For future cross-speaker / voice-conversion tasks: `None` — meaning "no ref slot is from the target speaker; refs are style references, not voice references."

**Important encoding decision**: this column is `int | None` (Lance nullable), **not** an int with `-1` as a "no match" sentinel. Reason: Python's negative-indexing convention (`ref_audio_bytes[-1]` = last element) makes `-1` a real bug source. Use null (`None` / SQL `NULL`) for the no-match case.

For v1, the column is never null. The nullable type is preserved for v2+.

---

## 5. Audio format requirements

Every audio entry — both target and ref, both inside `target_audio_bytes` and inside `ref_audio_bytes[*]` — must satisfy:

| Requirement | Value |
|---|---|
| **Encoding** | WAV (PCM). The loader uses `torchcodec.decoders.AudioDecoder` which handles standard WAV. |
| **Sample rate** | Exactly 48 000 Hz |
| **Channels** | Mono (1 channel) |
| **Bit depth** | Not strictly constrained (16-bit, 24-bit, 32-bit float all decode). The loader resamples and converts to float32 internally. 16-bit PCM is the minimum-storage default for clean audio. |
| **Compression** | None (PCM only). FLAC is *not* used for storage, even though the source data may be FLAC. |

### 5.1 Why WAV (not FLAC)

The original spec considered FLAC. We've moved to WAV because:

- **Decoder simplicity**: torchcodec handles WAV with no special-casing.
- **Decode determinism**: WAV is byte-for-byte reproducible across decode tools; FLAC's lossless decode is also reproducible but adds a decompression step in the worker.
- **Storage cost**: ~2× FLAC for typical speech, but not prohibitive at the scale of A2A v1 (mls_eng is ~25 K speakers × ~50 hr = ~10 TB at 48 kHz mono 16-bit WAV). Acceptable trade-off for decode simplicity.

### 5.2 Why 48 kHz everywhere

Uniform per-row SR removes a class of branching from the loader:

- The loader resamples to 16 kHz at decode time (model SR for MMAudio).
- If the source is natively 48 kHz (mls_eng), no upsample is needed at curation time.
- If the source is natively 24 kHz (future Emilia-YODAS), the curator upsamples to 48 kHz at write time. ~2× storage cost is accepted to keep the table SR-uniform.

### 5.3 Why mono

Speech is mono by nature; stereo ref audio would just double the storage without information gain. The loader is built with multi-channel readiness (the `num_channels=1` default in `MultiAudioDecoder` is flippable later), but v1 commits to mono.

### 5.4 Encoding suggestion

Pseudocode for converting one source clip to the storage format:

```python
import io
import soundfile as sf
import numpy as np

def encode_clip(samples: np.ndarray, source_sr: int) -> bytes:
    """Encode a numpy waveform to v1's WAV-48k-mono storage format."""
    # 1. Resample to 48 kHz if needed
    if source_sr != 48000:
        from librosa import resample
        samples = resample(samples, orig_sr=source_sr, target_sr=48000)
    # 2. Force mono
    if samples.ndim > 1:
        samples = samples.mean(axis=0)
    # 3. Encode as 16-bit PCM WAV
    buf = io.BytesIO()
    sf.write(buf, samples, 48000, format="WAV", subtype="PCM_16")
    return buf.getvalue()
```

The exact resample / encode tooling is up to the curator; the only constraint is that the output is decodable by torchcodec at SR 48000.

---

## 6. Reference sampling policy

For each target clip, the curator must sample exactly one reference clip subject to:

### 6.1 Hard constraints

1. **Same speaker** as the target (`ref_speaker_global_id == target_speaker_global_id`).
2. **Different clip** than the target (`ref_key != target_key`). The reference cannot be the supervised target itself.
3. **Decodable as WAV 48 kHz mono** (per §5).

### 6.2 Strong preference (curation policy)

**Cross-recording-session preference**: with probability ~80%, sample the ref from a recording session different from the target's. For MLS-Eng, "recording session" = `book_id`. So:

```
if random() < 0.80 and speaker_has_clips_in_other_books:
    sample ref from a clip with `book_id != target.book_id`
else:
    sample ref uniformly from any same-speaker clip (excluding target_key)
```

**Why cross-book**: same recording session shares mic, room acoustics, register, and microphone-channel cues. Without cross-book preference, the model can identify "this speaker" from acoustic-channel artifacts (mic noise, room reverb, recording-software EQ) rather than voice timbre. Cross-book pairs share *speaker identity* but not *acoustic channel*, forcing the model to extract the part we want.

The 80% number is from the original v2 spec; it's the operating point that worked in earlier ref-TTS literature (VALL-E and follow-ups). Don't deviate without reason.

**When falling through to same-book**: only if the speaker has no clips in any other book (i.e., they only recorded one session). In this case, just sample uniformly from their other clips.

### 6.3 Minimum ref duration

The loader truncates every ref to a fixed 3-second window. So the curator should prefer refs that are **at least 3 seconds** long. If shorter refs are rare in the source, fine — the loader will silence-pad them. If they're common, consider filtering them at curation time so the model isn't trained on silence-padded refs.

For MLS-Eng, clips are typically 5–20 seconds; the under-3 s case is rare.

### 6.4 Per-speaker capping

To avoid speaker frequency imbalance (a common audiobook-corpus pathology where one prolific reader has 10× more clips than median), cap the number of target rows per speaker:

```
T = count of clips for the speaker at rank floor(0.10 * N_speakers) when sorted by clip count desc.
Each speaker contributes between 2 and T target rows.
Speakers with only 1 clip are dropped (can't supply a same-speaker ref).
```

This is the original v2 spec's policy — preserve it for v1 consistency.

### 6.5 Sampling determinism

The curation pipeline should be **deterministic from a fixed seed**: rerunning the curator with the same seed should produce byte-identical Lance shards. This is critical for:

- Reproducibility (a research run that pinned a specific table version can be re-derived).
- Resume / restart of curation jobs without contamination.
- Diff'ing curation iterations during development.

---

## 7. Hard invariants the curator must guarantee

These are the contracts the loader trusts at training time. **Any violation is a curation bug**, not a loader bug.

### 7.1 Target is not in its own reference pool

```
target_key NOT IN ref_keys
# In v1 with len(ref_keys) == 1, this means: ref_keys[0] != target_key.
```

If violated, the model can shortcut by memorizing target audio frame-for-frame. Catches: enforce at sampling time (`while sampled_ref.key == target.key: resample`).

### 7.2 Reference is the same speaker as target

```
ref_speaker_global_ids[0] == target_speaker_global_id
```

If violated, the row is a cross-speaker training example, which v1 doesn't support (the loader would still train on it, but the supervised loss would push the model to produce target speaker's voice from a different speaker's ref — confusing signal).

### 7.3 Exactly one reference

```
M == 1
len(ref_audio_bytes) == 1
len(ref_audio_srs) == 1
len(ref_durations) == 1
len(ref_speaker_global_ids) == 1
len(ref_keys) == 1
```

The loader raises `ValueError` if `len(ref_audio_bytes) != 1`. The other length-1 constraints are by-construction in v1.

### 7.4 All audio is WAV 48 kHz mono

```
target_audio_sr == 48000
ref_audio_srs[0] == 48000
both audios decode as WAV mono (not stereo, not FLAC)
```

The loader asserts SR == 48000 and raises on any row that violates it. WAV-vs-FLAC and mono-vs-stereo would surface as a decode error (torchcodec couldn't produce a single-channel float tensor).

### 7.5 `target_speaker_local_idx` consistency

```
if target_speaker_local_idx is None:
    # No ref is target-speaker. v1 should never have this state
    # (every v1 row has exactly one ref and it's same-speaker).
    raise CurationBug

if target_speaker_local_idx == 0:
    # The ref at slot 0 is target-speaker.
    assert ref_speaker_global_ids[0] == target_speaker_global_id

# Other values (≥ 1) are not valid for v1 with M=1.
```

For v1, this column is always `0`. For v2, the value indicates which slot to consult.

---

## 8. Excluded columns (deliberately)

For clarity, here's what the schema does **not** include and why:

| Column | Why excluded |
|---|---|
| `ref_text` / `ref_texts` | The model trains on voice cues only; ref transcripts would tempt it to learn ref→target text alignment, which is the wrong signal. |
| `pass_filter` / quality column | Source dataset is clean academic audio; no quality gate needed for v1. (T2A's `round1_pass_all_filter` is for crawl-quality data; not applicable here.) |
| `ref_speaker_local_ids` | Always `[0]` for M=1 by construction. Add when v2 ships if needed. |
| `task_template` | Loader builds the prompt locally for v1 (`"Generate the following transcript:\n" + target_text`). v2 will need this column once slot-aware prompts ship. |
| `book_id` (per-clip) | Used at curation time to enforce cross-book preference, but **not exposed per row** in the table — would just bloat storage. If needed for analysis, look up via `target_key` against the source `metadata.jsonl`. |

---

## 9. Loader contract (what the consumer expects)

This section spells out what the training loader will do with the table, so the curator can predict consumer behavior.

### 9.1 Loader entry point

The loader is `OmniA2APackingConfigKobaV2`, exposed via the kuma factory:

```python
mls_eng_a2a_v1(max_num_tokens=8000, batch_size=4)
# → OmniA2APackingConfigKobaV2 instance pointing at the curator's Lance table.
```

The factory points at the curator's S3 path:

```python
audio_datasets=[
    "s3://ai-lumalabs-datasets-ap-se-2-lance"
    "/audio/pretrain/a2a/mls_eng_zs_tts_a2a_v1.lance"
]
```

### 9.2 SQL filter applied at load time

```
`M` = 1 AND `target_language` = 'en'
```

Belt-and-suspenders: the table is supposed to contain only M=1 English rows, but the SQL filter catches stale shards. If the curator wants to support other languages (future Emilia-YODAS de/es/fr), the loader filter is the per-job override knob.

### 9.3 Per-row read sequence

For each row that passes the filter, the loader:

1. Asserts `target_audio_sr == 48000` and `ref_audio_srs[i] == 48000`. **Raises** on mismatch.
2. Decodes `target_audio_bytes` to a `(1, T_target)` float32 tensor at 16 kHz.
3. Decodes `ref_audio_bytes[0]` to a `(1, T_ref)` float32 tensor at 16 kHz.
4. Normalizes both via `AudioToX` (peak / clamp; same kwargs for both).
5. Truncates the ref tensor to a fixed 3-second window (`(1, 48000)`) — random crop in training, center crop in eval. Pads with silence if shorter than 3 s.
6. Builds the sequence plan: `[CLEAN_VAE_AUDIO(ref), TEXT(prompt + target_text), NOISY_VAE_AUDIO(target)]`.
7. Tokenizes + packs.

The ref-truncation step is at training time, not curation time. **The curator should not pre-truncate refs** — let the loader do it (the loader does random crop per epoch, which is regularization the curator can't replicate).

### 9.4 Column-name mapping

The loader's column-key knobs default to the names in §4. **The curator using these names means zero loader-side overrides are needed.** If the curator uses different names, the loader can be configured per-knob:

| Loader knob (default value) | Maps to schema column |
|---|---|
| `target_audio_bytes_key="target_audio_bytes"` | `target_audio_bytes` |
| `target_audio_sr_key="target_audio_sr"` | `target_audio_sr` |
| `transcript_key="target_text"` | `target_text` |
| `ref_audio_bytes_key="ref_audio_bytes"` | `ref_audio_bytes` |
| `ref_audio_srs_key="ref_audio_srs"` | `ref_audio_srs` |

### 9.5 What the loader does *not* read

For information — the curator can include these columns or omit them with no effect on training:

- `sample_id`, `target_key`, `ref_keys`: provenance only; not read by the loader. Useful for debugging.
- `target_speaker_global_id`, `ref_speaker_global_ids`: not read at training; useful for invariant assertions in a curation-validation script.
- `target_duration`, `ref_durations`: not read by the v1 un-bucketed loader. `target_duration` will be needed by PR-3's bucketing.
- `source_dataset`: not read; available as a SQL-filter axis if multiple sources are mixed in one table later.
- `target_speaker_local_idx`: not read by v1; v2's seq-builder will read it.

Even though the loader doesn't read most of these, **include them anyway** — they cost little storage relative to audio bytes, and the column mapping is the contract.

---

## 10. Forward compatibility (don't accidentally break v2)

When v2 (M ∈ {1, 2, 3}) ships in PR-4, the curator will produce a richer table. To keep v1 → v2 a smooth migration:

### 10.1 Use list-typed columns even when length is 1

`ref_audio_bytes` is `list<binary>` even in v1 (length always 1). When v2 ships, the column type doesn't change — only the list length grows from 1 to up-to-3. No schema migration, no column rename.

If the curator stored `ref_audio_bytes` as scalar `binary` in v1, the v2 transition would require either rewriting every row or maintaining two parallel schemas. Don't.

### 10.2 Include `target_speaker_local_idx` in v1

Even though v1 always has `target_speaker_local_idx = 0`, include the column. v2's seq-builder reads it; pre-populating it in v1 means v1 rows can be loaded by a v2-aware loader without any conversion.

### 10.3 Don't pre-compute `task_template` in v1

`task_template` is a v2 column. v1's loader builds the prompt locally (`"Generate the following transcript:\n" + target_text`). v2 will need richer prompts that reference slot indices (e.g. `"using the voice of speaker-2"`). The curator can leave `task_template` out of v1 and add it in v2.

### 10.4 Schema versioning lives in the filename

`mls_eng_zs_tts_a2a_v1.lance` is the v1 table. When v2 ships, build `mls_eng_zs_tts_a2a_v2.lance` as a separate file. Don't try to grow v1 in place.

The version in the filename should advance whenever:

- The schema gains a column whose absence breaks loaders.
- Hard invariants change (e.g. M lifts from {1} to {1,2,3}).
- Column types change.

The version should NOT advance for:

- Adding/dropping rows (just overwrite).
- Including/excluding optional metadata columns the loader doesn't read.

---

## 11. Validation: how the curator should sanity-check the output

Before publishing `mls_eng_zs_tts_a2a_v1.lance`, run these checks. Even better: bake them into the curation pipeline so a malformed shard never gets published.

### 11.1 Schema check

```python
import lance
ds = lance.dataset("s3://.../mls_eng_zs_tts_a2a_v1.lance")
print(ds.schema)
```

Verify every column from §4.1 is present with the correct Arrow type. In particular:

- `ref_audio_bytes`: `list<binary>`, NOT `binary`.
- `ref_audio_srs`: `list<int64>`, NOT `int64`.
- `target_speaker_local_idx`: nullable `int64`.

### 11.2 Per-row invariants (sample 1000 rows)

```python
for row in ds.scanner(columns=[...]).to_table().to_pylist()[:1000]:
    # 7.1: target not in ref pool
    assert row["target_key"] not in row["ref_keys"]

    # 7.2: same-speaker
    assert row["ref_speaker_global_ids"][0] == row["target_speaker_global_id"]

    # 7.3: exactly one ref
    assert row["M"] == 1
    assert len(row["ref_audio_bytes"]) == 1
    assert len(row["ref_audio_srs"]) == 1
    # ... and so on for the other ref_* lists

    # 7.4: SR
    assert row["target_audio_sr"] == 48000
    assert row["ref_audio_srs"][0] == 48000

    # 7.5: speaker-idx consistency
    assert row["target_speaker_local_idx"] == 0
```

### 11.3 Decode check (sample 100 rows)

```python
import io, torchcodec

for row in sample_100_rows:
    # Target decodes as 48 kHz mono
    dec = torchcodec.decoders.AudioDecoder(
        source=io.BytesIO(row["target_audio_bytes"]),
        sample_rate=48000,
        num_channels=1,
    )
    target = dec.get_all_samples().data
    assert target.shape[0] == 1  # mono channel dim
    # Sample count matches reported duration
    assert abs(target.shape[-1] - row["target_duration"] * 48000) <= 48  # 1 ms tolerance

    # Ref decodes likewise
    dec = torchcodec.decoders.AudioDecoder(
        source=io.BytesIO(row["ref_audio_bytes"][0]),
        sample_rate=48000,
        num_channels=1,
    )
    ref = dec.get_all_samples().data
    assert ref.shape[0] == 1
```

### 11.4 Distribution sanity (over the full table)

```python
import polars as pl
df = ds.to_table().to_pandas()

# Speaker frequency: should match the curation cap policy (§6.4).
print(df.groupby("target_speaker_global_id").size().describe())

# Cross-book share: should be ≈80% per the §6.2 policy.
# (This requires access to source book_ids; check at curation time, not post-hoc.)

# Target duration distribution (informs PR-3 bucketing).
print(df["target_duration"].describe())
print(df["target_duration"].quantile([0.05, 0.5, 0.95]))
```

### 11.5 Loader smoke test

The cleanest end-to-end check is to pass the table through the loader on a tiny config:

```python
from kuma.projects.omni.audio.data.omni_a2a_packing_koba_v2 import OmniA2APackingConfigKobaV2

cfg = OmniA2APackingConfigKobaV2(
    audio_datasets=["s3://.../mls_eng_zs_tts_a2a_v1.lance"],
    max_num_tokens=2000,
    batch_size=4,
    num_workers=0,
)
loader = cfg.get_loader(dp_rank=0, dp_world_size=1)
batch = next(iter(loader))
print(f"Got batch with sequence_length={batch['sequence_length']}")
```

If this returns one batch without crashing, the loader's per-row asserts (SR, list shape, decode) are all passing.

---

## 12. Estimated storage

For sizing:

- Average MLS-Eng clip: ~10 s.
- WAV 48 kHz 16-bit mono: 96 KB/s.
- Per-clip storage: ~960 KB.
- Per-row storage: 2 × ~960 KB = ~1.9 MB (target + ref).

For ~10 M rows (curator's per-speaker-cap policy applied to MLS-Eng), the table is ~19 TB. The curator should plan for ~20 TB of S3 with this estimate.

Compression note: Lance applies row-level compression to `bytes` columns by default, but PCM WAV is already incompressible (random-ish at the bit level), so don't expect compression to bring this down meaningfully. The 19 TB is essentially the raw audio bytes plus negligible metadata.

If storage becomes a concern, two follow-ups (not v1):

1. Switch to FLAC bytes — ~50% reduction at the cost of decoder complexity.
2. Pre-truncate refs to 3 s — eliminates the redundancy that the loader truncates anyway. Costs forward-compat (v2 might want longer refs).

---

## 13. Open decisions for the curator team

A few things weren't pinned down during the schema design and need a decision before curation starts:

### 13.1 `source_dataset` column: include or skip?

Including the column lets multiple sources (mls_eng + future Emilia-YODAS) be mixed into one table with `source_dataset` as a SQL-filter axis. Skipping it means each source gets its own table file — simpler operationally, but harder to mix sources in one training run.

**Recommendation**: **include** `source_dataset = "mls_eng"` even if v1 only has one source. The cost is one short-string column; the value is multi-source readiness.

### 13.2 Sample count target

For the v1 table, what's the target row count? Options:

- **Minimum viable**: ~100 K rows (~30 hr of speech). Enough for the smoke test, far short of production training.
- **First useful**: ~1 M rows (~300 hr). Enough for a small zs-tts model to converge.
- **Production v1**: ~10 M rows (~3 K hr, applying the §6.4 per-speaker cap). The curator's full output for mls_eng.

The §6.4 capping policy implicitly sets the upper bound at ~10 M for mls_eng; the curator should confirm what the lower-bound is (how many rows are needed for the smoke / first training run).

### 13.3 Same-book fallback ratio

§6.2 says "with probability 80% sample cross-book; otherwise sample uniformly." The 20% same-book share is from the original v2 spec. If the curator team has reason to push the cross-book preference higher (e.g., 95%) for stronger acoustic-channel decorrelation, that's a one-line change in the sampling code. **Confirm**: stay at 80% or push higher?

### 13.4 Determinism seed value

The curation pipeline should be deterministic from a fixed seed (§6.5). What's the seed?

- A fresh seed per curation iteration (e.g., `42`, `43`, ...) gives reproducibility within an iteration but lets us re-roll if v1's first table has issues.
- A fixed seed (e.g., `42` forever) gives full reproducibility across re-runs but means we can't easily "try a different sample" without changing the schema version.

**Recommendation**: pin a seed (`42`) for v1's first publication; advance the filename version (`v1.1.lance` etc.) if the same schema is re-rolled.

### 13.5 Curation-time SR conversion tolerance

If a source clip is reported as 48 kHz but the actual file is 44.1 kHz (metadata corruption), the curator's encode step should detect this and either fix or drop the clip. **Decide**: drop silently, or log + drop, or raise (require manual review)?

---

## 14. Summary checklist

A v1 publication is ready when:

- [ ] Lance table exists at `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/a2a/mls_eng_zs_tts_a2a_v1.lance`.
- [ ] Schema matches §4.1 exactly (column names, types, nullability of `target_speaker_local_idx`).
- [ ] Every row has `M = 1`, `target_audio_sr == 48000`, `ref_audio_srs[0] == 48000`, `len(ref_audio_bytes) == 1`.
- [ ] Every row satisfies `target_speaker_global_id == ref_speaker_global_ids[0]` (same-speaker invariant).
- [ ] Every row satisfies `target_key NOT IN ref_keys` (target-not-in-ref-pool invariant).
- [ ] At least 80% of rows have `book_id != target.book_id` for the ref (cross-book preference policy, validated at curation time).
- [ ] Per-speaker cap (§6.4) applied.
- [ ] Sample audio decodes as WAV 48 kHz mono via torchcodec (verified on ≥100 sampled rows).
- [ ] Loader smoke test (§11.5) returns a batch without crashing.

Once those are checked, point the loader factory `mls_eng_a2a_v1` at the path, run PR-1b's `debug_local_a2a` smoke test, and the table is ready for production use.

---

## Appendix A: Worked example row

For concreteness, here's what one row looks like, populated from a hypothetical mls_eng pair:

```python
{
    # Identity
    "sample_id": "mls_eng/mls_spk4800_book10003_000000",
    "source_dataset": "mls_eng",
    "target_language": "en",

    # Target
    "target_audio_bytes": <bytes — ~1.5 MB of WAV 48 kHz mono PCM>,
    "target_audio_sr": 48000,
    "target_duration": 15.81,
    "target_text": "oh my dear you must see him he expects you she answered almost gayly...",
    "target_speaker_global_id": "4800",
    "target_key": "mls_spk4800_book10003_000000",

    # Reference (length-1 lists)
    "ref_audio_bytes":         [<bytes — ~720 KB of WAV 48 kHz mono PCM>],
    "ref_audio_srs":           [48000],
    "ref_durations":           [7.5],
    "ref_speaker_global_ids":  ["4800"],   # same as target_speaker_global_id
    "ref_keys":                ["mls_spk4800_book9842_000123"],   # different book
                                                                   # different clip
    # Curator metadata
    "M": 1,
    "target_speaker_local_idx": 0,
}
```

Note that the ref's `book_id=9842` differs from the target's `book_id=10003` — this is a cross-book pair (the dominant case under the 80% policy). The speaker (`4800`) is the same. The clip IDs are different (target is `clip 000000`, ref is `clip 000123`).

---

## Appendix B: Cross-references to the loader-side code

For the curator who wants to confirm the loader's expectations against the actual code:

| Schema field | Loader-side code that consumes it |
|---|---|
| `target_audio_bytes` | `MultiAudioDecoder.forward` in `lib/koba/koba/processor/audio_ops.py` |
| `ref_audio_bytes` (list-of-1) | Same; reads `[0]` |
| `target_audio_sr` / `ref_audio_srs` | `MultiAudioDecoder.forward` (defensive assert against `expected_source_sr=48000`) |
| `target_text` | `OmniAudioSeqBuilder.handle_a2a` in `lib/koba/koba/processor/omni_audio_packed_ops.py` (concatenated with the prompt prefix) |
| Filter expression `M = 1 AND target_language = 'en'` | `OmniA2APackingConfigKobaV2.dataset_row_filter` default in `projects/kuma/.../data/omni_a2a_packing_koba_v2.py` |

---

End of document. Use this as the anchor for the table-creation discussion; the curator-side decisions in §13 are the fastest way to unblock implementation.
