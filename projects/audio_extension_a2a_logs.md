# Omni-audio extension — A2A design notes

> **Started:** 2026-05-01
> **Branch:** `dongguo/omni-t2a-v2`
> **Companion docs:**
>
> - [`audio_extension_v2_logs.md`](audio_extension_v2_logs.md) — A2T encoder scaffold + freeze policy (already-landed work)
> - [`audio_extension_v2_plan.md`](audio_extension_v2_plan.md) — overall design plan (A2T + A2A)
> - [`omni_t2a_dataloader_deep_dive.md`](omni_t2a_dataloader_deep_dive.md) — definitive reference for the T2A data loader, sequence-element abstraction, attention modes, per-token mask machinery
> - [`mental_models/omni_data_lifecycle.md`](../mental_models/omni_data_lifecycle.md) — pipeline-stage walkthrough + the CFG-dropout special-topic section

This doc captures the design of the A2A (audio-to-audio reference-TTS)
data path arrived at over a multi-round design discussion. It is
forward-looking — it documents the implementation we are about to land,
not what already exists in the tree.

---

## 1. TL;DR

- **Task.** Reference-TTS. Given (a) one or more reference audio clips
  and (b) a transcript, generate the transcript spoken in the reference
  speaker's voice.
- **Two-stage rollout.**
  - **v1 (this design's first deliverable):** simplified to **M = 1
    same-speaker reference**. Sequence plan is just T2A plus a
    `CLEAN_VAE_AUDIO(ref)` element prefix. Mirrors the canonical zero-shot
    TTS setup (VALL-E / NaturalSpeech-2 style: acoustic prompt + phonetic
    prompt → generate).
  - **v2 (follow-up):** extend to `M ∈ {1, 2, 3}` with distractor refs.
    Adds `speaker-{i}` slot tags interleaved with refs and a templated
    instruction with a slot pointer. Adds bucketing on
    `target_duration`. Re-evaluates CFG dropout.
- **Reuses the existing T2A / bagel machinery almost end-to-end.** The
  abstraction `SequenceElement` is variable-length and supports
  arbitrary interleaved element types; the existing element types
  (`CLEAN_VAE_AUDIO`, `NOISY_VAE_AUDIO`, `TEXT`) and per-element
  processors handle everything we need. Only **3 net-new pieces** ship
  in v1: `MultiAudioDecoder`, `RefAudioTruncator`, and a new
  `handle_a2a` method on `OmniAudioSeqBuilder`.
- **No tokenizer changes.** Slot tags are plain BPE text in v2; v1
  doesn't use slot tags at all.
- **No new sequence-plan fields.** Refs vs. target supervision is
  encoded by element *type* (`CLEAN_VAE_AUDIO` = conditioning-no-loss,
  `NOISY_VAE_AUDIO` = denoising target). The deep-dive's
  per-element-type loss-mask machinery (§8.2, §9.4) handles it.

---

## 2. Task definition

A2A trains a model to synthesize speech in a given reference voice. The
canonical row format (curator output) is fully described in the project
root README; the loader-relevant fields are:

```text
target_audio_bytes        : <bytes>          ← raw FLAC/WAV
target_audio_sr           : 48000            ← see §3.1 below
target_duration           : 15.81            ← seconds (used for bucketing in v2)
target_text               : "oh my dear..."  ← what to speak
ref_audio_bytes           : [<bytes>, ...]   ← list, length M ∈ {1, 2, 3}
ref_audio_srs             : [48000, ...]
ref_durations             : [16.31, ...]
M                         : 3                ← scalar count
target_speaker_local_idx  : 2                ← which slot holds target speaker
task_template             : "Speak the following text using the voice of
                            speaker-2: oh my dear..."
```

Hard invariants the curator guarantees (loader can rely on them):

- `target_key ∉ ref_keys` — the target clip is never in the reference
  pool. No memorization shortcut.
- `target_speaker_global_id ∈ ref_speaker_global_ids` — the target
  speaker is always present in the reference pool, at minimum at slot
  `target_speaker_local_idx`.
- `M ∈ {1, 2, 3}` per row (sampling prior `[0.60, 0.20, 0.20]`).
- `M = 1` rows are **always same-speaker** (only one slot, which by
  invariant 2 is the target speaker). This is what makes the v1 SQL
  filter `M = 1` give us a clean same-speaker training set for free.

### 2.1 What v1 does and doesn't do

- v1 **uses only `M = 1` rows.** Filtered via SQL predicate at the
  loader. `M = 1` rows are 60% of the curated dataset by the sampling
  prior — for `mls_eng` that's still tens of millions of same-speaker
  pairs, plenty for a first run.
- v1 **does not use `task_template` from the row.** Instead it
  constructs a T2A-style prompt locally (`"Generate the following
  transcript:\n" + target_text`). Reasoning: with `M = 1` there is no
  slot to point at, so the curator's templated instruction with
  `speaker-{x}` is unnecessary verbose for the v1 task. T2A-byte-parity
  also lets a T2A checkpoint init A2A v1 cleanly. (v2 will switch to
  the curator's `task_template` when slot pointers become meaningful.)
- v1 **does not emit slot tags.** With one ref slot, there is nothing
  to label. (v2 will add `TEXT("speaker-{i}")` interleaved with refs.)
- v1 **does not use `target_speaker_local_idx`** (always 0 for `M = 1`
  rows; not consumed in the seq-plan).

---

## 3. Pipeline-level simplifications upstream of the loader

### 3.1 Audio format normalization

Curator pre-converts every audio clip to **48 kHz mono WAV bytes** before
writing to Lance, regardless of native rate (mls_eng is natively 48 kHz;
emilia would be 24 kHz upsampled). This decision is upstream of the
loader.

Loader implications:

- The decoder's `sample_rate` parameter (target rate, 16 kHz for
  MMAudio) is the only resampling that happens in the loader — uniform
  per row. No per-row SR branching.
- `target_audio_sr` / `ref_audio_srs` columns become **defensive
  asserts**, not branching inputs. We assert `== 48000` per row to
  catch a partially-converted shard before it silently corrupts
  training. Cheap insurance.

Cost: ~2× storage bloat for any 24 kHz native source (no quality gain
from the upsample). Acceptable trade-off; can revisit ("store native,
resample in loader") in a later iteration if storage becomes binding.

### 3.2 Reference truncation

Reference clips can be 12–16 s in the curator output. Voice cloning
needs voice properties (timbre, prosody), not transcript content, so we
**truncate refs to a fixed window** during loading.

- Default window: **3 s** (config knob `ref_window_sec`). Rationale: long
  enough for timbre + prosody, short enough to not bloat the token
  budget. Common practice in zero-shot TTS literature ranges from 1 s
  (ECAPA-style speaker embed) to 3–10 s (VALL-E style acoustic prompt);
  3 s is the middle of the band.
- **Random crop** at training time (cheap regularization — different
  3 s chunks of the same clip across epochs), **center crop** at eval
  time (deterministic).
- Random crop's per-row RNG must be a `Seedable`; otherwise we get
  silent non-determinism per koba's CLAUDE.md.

This collapses what was structurally a "variable M × variable SR ×
variable per-ref duration" 3-axis problem to "variable M × fixed
window-shape per slot" — much easier to bucket and pack. After
truncation, refs are uniform-shape `(window_samples,)` tensors at the
model SR (16 kHz × 3 s = 48 000 samples).

---

## 4. v1 design (M = 1 same-speaker reference)

### 4.1 Sequence plan

```
[
  CLEAN_VAE_AUDIO(ref),                           attn=full,  loss=False, modality="a2a"
  TEXT("Generate the following transcript:\n" + target_text),
                                                  attn=causal, loss=False, modality="a2a"
  NOISY_VAE_AUDIO(target),                        attn=noise, loss=True,  modality="a2a"
]
```

Three elements. Structurally one element more than T2A (which is
`[TEXT, NOISY_VAE_AUDIO]`) — the only addition is the prefix
`CLEAN_VAE_AUDIO(ref)` element.

This works because of the abstraction laid out in the deep-dive (§9.7):
**"Element composition encodes task semantics. The task is exactly
'which sequence of elements does the data builder emit, and what's the
loss flag on each.' Everything downstream is mechanical."** All three
element types and their per-token mask wiring already exist for T2I /
T2A.

What each element does mechanically (cross-references to deep-dive §8):

| Element type | `attention_mode` | Loss masks set | Stream routing |
| --- | --- | --- | --- |
| `CLEAN_VAE_AUDIO(ref)` | `full` (bidirectional self + visible to all priors and downstream) | `clean_vae_token_mask=1`, `noisy_vae_token_mask=0`, `txt_loss_mask=0` | VAE stream via `vae_token_mask=1` on pad slots; boundary tokens carry `text_token_mask=1` |
| `TEXT(prompt+transcript)` | `causal` (LR triangular self + visible to all priors) | `txt_loss_mask=0` (no supervision on prompt) | Text stream |
| `NOISY_VAE_AUDIO(target)` | `noise` (bidirectional self + visible to priors; outbound-invisible to downstream) | `noisy_vae_token_mask=1` (denoising target), `clean_vae_token_mask=0` | VAE stream |

The diffusion loss is automatically applied only at the target's
`noisy_vae_token_mask=1` positions; the ref contributes zero loss; the
prompt contributes zero loss. No per-element `loss_mask` flag needed —
the type tag does it.

### 4.2 `handle_a2a` method on `OmniAudioSeqBuilder`

The seq-builder dispatches on `sample["conversation_modality"]` to a
per-task handler. T2A is wired today; A2A is a new branch.

New `Config` fields (prefixed `a2a_*` per the existing convention):

```python
# A2A-specific (v1: single same-speaker ref)
a2a_target_audio_tensor_key: str = "target_audio_tensor"
a2a_ref_audio_tensor_key:    str = "ref_audio_tensor"     # singular (v1)
a2a_transcript_key:          str = "target_text"          # column from row schema
a2a_task_prompt:             str = "Generate the following transcript:\n"
```

Note: `a2a_task_prompt` deliberately byte-matches `t2a_task_prompt`. A
T2A-trained checkpoint can plausibly init A2A v1 with most of the
text-side priors already aligned.

Handler:

```python
def handle_a2a(self, sample: dict) -> dict | None:
    """A2A (v1, M=1): same-speaker ref + transcript → target audio.

    Sequence plan:
        [CLEAN_VAE_AUDIO(ref), TEXT(prompt + transcript), NOISY_VAE_AUDIO(target)]
    """
    cfg = self.config
    if cfg.a2a_target_audio_tensor_key not in sample:
        return None
    if cfg.a2a_ref_audio_tensor_key not in sample:
        return None
    if cfg.a2a_transcript_key not in sample:
        return None

    target_tensor = sample[cfg.a2a_target_audio_tensor_key]
    ref_tensor    = sample[cfg.a2a_ref_audio_tensor_key]
    transcript    = sample[cfg.a2a_transcript_key]

    ref_media    = Media(media_type="audio", data=ref_tensor)
    target_media = Media(media_type="audio", data=target_tensor)
    sample["media"] = [ref_media, target_media]

    sample["sequence_plan"] = [
        SequenceElement(
            type=SequenceType.CLEAN_VAE_AUDIO,
            media=ref_media,
            loss=False,
            modality="a2a",
        ),
        SequenceElement(
            type=SequenceType.TEXT,
            text_str=cfg.a2a_task_prompt + transcript,
            loss=False,
            modality="a2a",
            supervise_last_text_token=False,
        ),
        SequenceElement(
            type=SequenceType.NOISY_VAE_AUDIO,
            media=target_media,
            loss=True,
            modality="a2a",
        ),
    ]
    return sample
```

Dispatch update in `forward`:

```python
def forward(self, sample: dict) -> dict | None:
    modality = sample.get(self.config.conversation_modality_key)
    if modality == "t2a":
        return self.handle_t2a(sample)
    if modality == "a2a":
        return self.handle_a2a(sample)
    raise NotImplementedError(...)
```

### 4.3 New koba processors

Two new processors at the front of the pipeline, decoding+normalizing
the A2A row's two audio fields.

**`MultiAudioDecoder`** — replaces `AudioDecoder` for A2A rows.

- Inputs: `target_audio_bytes` (scalar bytes), `ref_audio_bytes` (list
  column; v1 reads `[0]`), `target_audio_sr` / `ref_audio_srs`
  (asserted `== 48000`), `sample_rate` config (target SR, 16 kHz).
- Outputs: `target_audio_tensor`, `ref_audio_tensor` (singular in v1).
- Internally uses the same WAV decode + resample path as `AudioDecoder`
  twice. No format branching (always WAV after curator normalization).

**`RefAudioTruncator`** — fixed-window crop on the ref tensor.

- Inputs: `ref_audio_tensor`, config `ref_window_sec` (default 3.0),
  `crop_mode` ∈ {`"random"`, `"center"`}.
- Outputs: `ref_audio_tensor` overwritten with the cropped tensor of
  shape `(window_samples,)`.
- Implements `Seedable.reseed(seed)` and is registered in
  `KobaDataset.Config.seedables` so the random crop is deterministic
  across epochs (per koba CLAUDE.md determinism contract).

For v2, both processors gain list-aware modes (or are wrapped). The
seq-builder's `handle_a2a` is what changes structurally; the processors
are additive extensions.

### 4.4 Pipeline (`default_a2a.py`)

Mirrors `default_t2a.py:31-138`. Diff vs. T2A:

1. Replace `AudioDecoder` with `MultiAudioDecoder`.
2. Insert `RefAudioTruncator` after the decoder.
3. Loop `AudioToX` over `[target_audio_tensor, ref_audio_tensor]` (or
   wrap in a small helper). Negligible per-tensor cost; AudioToX's job
   is normalization, the M=1 case is `(M+1)=2` calls.
4. `OmniAudioSeqBuilder` is the same class — the `handle_a2a` branch
   reads the new `a2a_*` keys.
5. `OmniElementVAEAudio` is unchanged — it dispatches on element type,
   so the ref's `CLEAN_VAE_AUDIO` element gets the right (no-noise,
   `clean_vae_token_mask=1`) treatment; the target's `NOISY_VAE_AUDIO`
   is identical to T2A.
6. `OmniElementText`, `OmniQwen3Tokenizer`, `OmniPositionIDMRoPE`,
   `OmniSequenceLengthFilter` — all reuse unchanged.

Result: `default_a2a.py` is a near-clone of `default_t2a.py`. The
diff's surface is small and reviewable as a single PR.

### 4.5 `OmniA2APackingConfigKobaV2`

Mirrors `OmniT2APackingConfigKobaV2` byte-for-byte except:

- New row-schema fields exposed as config knobs (column names listed
  in §2 above).
- `dataset_row_filter: str = "M = 1"` — restricts to the M=1
  same-speaker subset.
- Wires the v1 `default_a2a_pipeline_processors` (above) instead of
  the T2A one.
- `conversation_modality_value="a2a"` so the seq-builder dispatches to
  `handle_a2a`.

Bucketing: **deferred** to a later phase (see §5 and §8). v1 ships
un-bucketed because the production T2A loader ran un-bucketed for many
million steps before bucketing was introduced; the modest VAE
inefficiency is acceptable while validating correctness.

### 4.6 CFG dropout (v1)

For v1's M=1 case, the standard CFG-dropout machinery works cleanly.

Reasoning chain:

- A2A is structurally an **editing-shape** task in the lifecycle doc's
  taxonomy (§ "Special topic: CFG dropout"): non-text conditioning
  (the ref) the task is impossible without, plus a trailing instruction.
- For editing-shape tasks, **drop-all is conceptually wrong** (replaces
  the editing task with unconditional generation — useless).
- For the M=1 case, drop-text leaves a **well-defined narrower task**:
  `p(target | ref_voice, no_text)` = "given this voice, produce some
  plausible speech in it." Same shape as image-edit's drop-text
  branch. The text-CFG axis at inference is the standard ref-TTS
  slider: amplify text-faithfulness while preserving voice from the
  ref.
- For the M>1 case, drop-text breaks because the slot-selection signal
  goes away — see §5.4 for the v2 analysis. **This is the v2 problem,
  not v1's.**

**v1 plan, in order:**

1. **First runs**: ship with `cfg_dropout_modalities=[]` (CFG dropout
   off entirely). Minimum moving parts; validate the rest of the
   pipeline first. Bagel VL configs already use this empty-list pattern
   when CFG isn't yet right for a task.
2. **Once basic training works**: flip to `OmniCFGDropoutLast` with
   `cfg_dropout_modalities=["a2a"]` and `cfg_dropout_prob=0.1`.
   Mirrors the T2A pattern (explicit modality list, T2A parity on
   prob) plus image-edit's class choice (drop-trailing-text-only is
   the right behavior for editing-shape tasks, per the lifecycle
   doc's Special-topic table).

Pipeline wiring:

```python
OmniCFGDropoutLast.Config(
    cfg_dropout_prob=0.1,
    cfg_dropout_modalities=[],          # off for first runs
    # cfg_dropout_modalities=["a2a"],   # flip on once training works
)
```

### 4.7 What v1 explicitly does *not* do

| Item | v1 stance | When it lands |
| --- | --- | --- |
| Slot tags `TEXT("speaker-{i}")` | Not emitted (no slots to label) | v2 |
| Curator's `task_template` field | Not consumed (we build a T2A-style prompt instead) | v2 |
| `target_speaker_local_idx` | Not consumed | v2 |
| Multi-ref decoding (list-aware) | Singular tensor only | v2 |
| Bucketing on `target_duration` | Un-bucketed loader | Phase 4 |
| Two-shape-class VAE batching extension | Not needed (refs and target are in the same un-bucketed pack) | Phase 4 |
| Custom slot-aware CFG variant | n/a — drop-text works for M=1 | v2+ if needed |
| Tokenizer additions | None | Never (decided: plain BPE) |

---

## 5. v2 design (M ∈ {1, 2, 3} with distractors)

Brought along to keep the v1→v2 transition explicit and minimize
re-derivation. Targeted at the curator's full row schema, including
distractor refs.

### 5.1 Sequence plan

```
[
  TEXT("speaker-0"), CLEAN_VAE_AUDIO(ref_0),     attn=causal, full
  TEXT("speaker-1"), CLEAN_VAE_AUDIO(ref_1),     ...
  ...
  TEXT("speaker-{M-1}"), CLEAN_VAE_AUDIO(ref_{M-1}),
  TEXT(task_template),                           causal
  NOISY_VAE_AUDIO(target),                       noise
]
```

`2*M + 2` elements with `M ∈ {1, 2, 3}`. Slot tags are interleaved with
their refs (not collected into one block) so positional binding is
local — the model doesn't have to traverse arbitrary distance to bind
"speaker-1" with `ref_1`.

Visibility (per deep-dive §8.1):

- All ref blocks are `full` mode — outbound-visible. Both the
  instruction text and the target's noisy block attend to all refs
  (this is how voice cues reach the denoiser).
- Refs see prior refs transitively. Cross-ref contrast happens at the
  target's attention layer (priors include all M refs).
- Target is `noise` mode — outbound-invisible. Same diffusion
  isolation as T2A.

### 5.2 Slot tags as plain BPE text

`speaker-{i}` plain text, not special tokens. Reason: the row's
`task_template` already uses the surface form `"speaker-2"` as plain
BPE in the instruction body. If we made `<speaker_2>` a dedicated
special token in the prefix, we'd have an alignment mismatch — prefix
tag is one token id, instruction reference is a different id sequence.
Plain text gives identical token ids in both places, the easiest signal
for the model to align.

Cost: ~2-3 BPE tokens per slot tag × up to 3 slots = ~6-9 extra tokens
per row. Negligible against a 16k–32k packed-token budget. Surface form
is a config knob `a2a_speaker_tag_template: str = "speaker-{i}"` so we
can ablate later (e.g. `"Speaker {i}:"`).

### 5.3 `handle_a2a` (M ≥ 1)

```python
def handle_a2a(self, sample: dict) -> dict | None:
    """A2A: M reference audios + templated instruction → target audio."""
    cfg = self.config
    target_tensor = sample[cfg.a2a_target_audio_tensor_key]
    ref_tensors   = sample[cfg.a2a_ref_audio_tensors_key]   # list[Tensor]
    task_template = sample[cfg.a2a_task_template_key]
    M = len(ref_tensors)

    ref_medias   = [Media(media_type="audio", data=t) for t in ref_tensors]
    target_media = Media(media_type="audio", data=target_tensor)
    sample["media"] = ref_medias + [target_media]

    plan: list[SequenceElement] = []
    for i, ref_m in enumerate(ref_medias):
        plan.append(SequenceElement(
            type=SequenceType.TEXT,
            text_str=cfg.a2a_speaker_tag_template.format(i=i),
            loss=False, modality="a2a", supervise_last_text_token=False,
        ))
        plan.append(SequenceElement(
            type=SequenceType.CLEAN_VAE_AUDIO,
            media=ref_m, loss=False, modality="a2a",
        ))
    plan.append(SequenceElement(
        type=SequenceType.TEXT,
        text_str=task_template, loss=False, modality="a2a",
        supervise_last_text_token=False,
    ))
    plan.append(SequenceElement(
        type=SequenceType.NOISY_VAE_AUDIO,
        media=target_media, loss=True, modality="a2a",
    ))
    sample["sequence_plan"] = plan
    return sample
```

(The v1 handler from §4.2 is the M=1 special case of this with no
slot-tag loop. v2 supersedes v1's handler in place.)

Config additions for v2:

```python
# A2A-specific (v2: M ∈ {1, 2, 3})
a2a_target_audio_tensor_key: str = "target_audio_tensor"
a2a_ref_audio_tensors_key:   str = "ref_audio_tensors"     # list[Tensor]
a2a_task_template_key:       str = "task_template"
a2a_speaker_tag_template:    str = "speaker-{i}"
```

### 5.4 CFG dropout under distractors — open

Standard CFG-dropout breaks for M>1 with distractors, in a way that
doesn't apply to image-edit:

- **Drop-text** removes the slot-selection signal. The element list
  becomes `[(speaker_i, ref_i)*M, TEXT(""), target]`. Slot tags survive
  (interleaved with refs), but the instruction telling the model
  *which* slot to imitate is gone. With M ≥ 2 and distractors, the
  task becomes underdetermined: by symmetry any slot could be x, but
  the loss target is fixed in slot x's voice. The model probably
  learns some biased default (e.g. "lean toward slot 0," which the
  curator's slot-0 cross-book preference might amplify), not a useful
  unconditional reference.
- **Drop-all** = unconditional generation = non-task ("speak some
  arbitrary text in some arbitrary voice"). Same conceptual failure as
  for image-edit (lifecycle doc, §"Why drop-all CFG is conceptually
  wrong for editing tasks").

The natural CFG axis for ref-TTS — voice-strength — would require a
**custom variant** that drops slot-x's ref specifically (using
`target_speaker_local_idx` from the row) while keeping distractors and
the instruction intact. Not in scope for v1 or v2 baseline; flagged as
a follow-up if a use case emerges.

**v2 baseline plan:** disable CFG dropout (`cfg_dropout_modalities=[]`)
for the first M>1 runs. Decide whether to build the slot-aware variant
based on whether v2 inference shows a gap that voice-CFG would fill.

### 5.5 Bucketing on `target_duration`

After truncation, refs become **fixed-shape per slot** (window_samples
each). The only structural variability axis is `target_duration` —
plus the bounded M overhead (≤ 282 ref tokens at M=3, vs. ~470 target
tokens for a 15 s clip; refs contribute < 2% of pack-level dispersion).

So bucket on `target_duration` alone, same boundaries as production
T2A (`[0–5, 5–10, 10–15]`). The Lance schema already exposes
`target_duration` per row — one-line config addition.

### 5.6 Two-shape-class batched VAE encoding

In a bucketed pack, two audio shape classes coexist:

| Class | Shape per element | Count per pack |
| --- | --- | --- |
| ref | `(window_samples,)` (e.g. `(48000,)` for 3 s × 16 kHz) | `sum_pack(M_i)` |
| target | `(target_max_samples_for_bucket,)` | `N_rows_in_pack` |

The trainer's batched VAE call needs to dispatch by shape class —
group elements at `(window_samples,)` into one VAE call, group elements
at the bucket-max length into another. This is a small extension to
`OmniAudioTrainer._encode_audio_batched`. It is **the** non-trivial
trainer-side change introduced by A2A (the entire seq-plan / mask
machinery is data-pipeline only).

The padding ceiling (`_PadAudioToCeiling`) needs a small change too:
apply only to target-audio elements, not refs. Refs are already at
exactly `window_samples` from the truncator — no rounding needed.
Cleanest gate is on the per-element type tag (refs are
`CLEAN_VAE_AUDIO`, target is `NOISY_VAE_AUDIO`).

---

## 6. Key design decisions (with reasoning)

| Area | Decision | Why |
| --- | --- | --- |
| **Element types for refs vs. target** | `CLEAN_VAE_AUDIO` for refs, `NOISY_VAE_AUDIO` for target — no per-element loss flag | Existing per-element-type masks (deep-dive §8.2, §9.4) already encode "no loss / conditioning" vs. "denoising target." Adding a parallel `is_target` field would duplicate the mechanism. |
| **Seq-builder structure** | New `handle_a2a` method on existing `OmniAudioSeqBuilder` | Codebase pattern (§T2A: `handle_t2a`); the seq-builder is task-dispatched on `conversation_modality`. Parallel class would fragment task-handling logic. |
| **Audio format** | Curator pre-converts to 48 kHz WAV; loader resamples to 16 kHz uniformly | Removes per-row SR branching from the loader. ~2× storage cost on 24 kHz native sources is acceptable; revisit if storage becomes binding. |
| **SR fields in row** | Keep, asserted `== 48000` per row | Defensive insurance against partially-converted shards. Negligible overhead. |
| **Reference truncation window** | 3 s, fixed across rows | Long enough for timbre+prosody, short enough to keep token budget close to T2A. Random crop at train (regularization), center at eval. |
| **Random-crop seeding** | `Seedable.reseed` on the truncator, registered in `seedables` | Per koba's determinism contract (CLAUDE.md). Without it, silent non-determinism. |
| **Slot tag form (v2)** | Plain BPE text `"speaker-{i}"`, not special tokens | Aligns with `task_template`'s BPE-encoded `"speaker-2"`. Avoids an alignment mismatch that would force the model to learn prefix-tag-↔-instruction-body binding. ~6-9 extra BPE tokens per row, negligible. |
| **Tokenizer changes** | None | No specials needed. `tokenizer_validation` analog also unneeded. |
| **`x_vae_by_modality` for refs and target** | Same `"a2a"` for both | Element type already distinguishes refs from target. Splitting into `"a2a_ref"` vs. `"a2a"` would duplicate the type-vs-modality information. |
| **CFG dropout (v1, M=1)** | Start with `[]` (off), flip to `OmniCFGDropoutLast` once training works | M=1 case has a well-defined drop-text branch. Image-edit precedent applies cleanly. |
| **CFG dropout (v2, M>1)** | Disable for baseline; consider custom slot-aware variant later | Standard branches break under distractor structure. See §5.4. |
| **Bucketing axis** | `target_duration` alone | Refs are fixed-shape after truncation, contribute bounded overhead. M-axis variability is < 2% of pack-token dispersion. |
| **First-PR scope (v1)** | Un-bucketed loader, dummy data first, then real Lance | Production T2A ran un-bucketed for many million steps; correctness before perf. |

---

## 7. Rejected alternatives

Documented so future-me doesn't re-litigate them.

| Alternative | Rejected because |
| --- | --- |
| New `OmniA2ASeqBuilder` class (parallel to `OmniAudioSeqBuilder`) | Codebase pattern is per-task handlers on a single seq-builder. Parallel class fragments the task-handling logic. |
| Per-element `loss_mask: bool` field on `SequenceElement` | The existing `CLEAN_VAE_AUDIO` vs. `NOISY_VAE_AUDIO` element types already encode this. Adding a parallel field is redundant. |
| Special `<speaker_0..2>` tokens in the tokenizer | Forces a token-id mismatch with the instruction body's plain-BPE `"speaker-2"`. Costs more for no functional gain (and adds a tokenizer-validation surface). |
| `<speaker_0>` (angle-bracket-shaped) plain-text tag | Looks tag-shaped to a human but tokenizes to 4-5 BPE pieces — more tokens than `"speaker-0"` and misleading because it isn't a real special token. |
| Substituting `"speaker-{x}"` → `<speaker_{x}>` tag at preprocessing time | Couples preprocessing to the template surface form; the model has to learn alignment anyway. Plain text in both places is the simplest signal. |
| Preserving native per-row SR in the loader | Adds branching for every audio op. The 2× storage cost of upsampling 24 kHz → 48 kHz upstream is the cheaper tradeoff for now. |
| Two `x_vae_by_modality` tags (`"a2a_ref"` vs `"a2a"`) | Element type already distinguishes them more cleanly. |
| `OmniCFGDropout` (drop-all) as the CFG variant for A2A | Replaces the editing task with unconditional generation. Lifecycle doc explicitly anticipates A2A and rules out drop-all for the same reason as image-edit. |
| 2D bucketing on `(target_duration, M)` | M-axis variability is bounded and small (< 2% of pack dispersion). Cross-product gives 9 buckets, thinning per-bucket sample density. |
| Padding refs to bucket-max with the same `_PadAudioToCeiling` knob | Wastes 5–6× VAE compute on refs (a 3 s ref padded to 15 s for the [10–15] bucket). The right fix is two-shape-class batched VAE; this is the wrong shortcut. |
| Including `ref_texts` (transcripts of refs) in the prompt | The user's framing is voice-only refs; transcripts are unnecessary. Truncation to 3 s also makes ref transcripts mostly meaningless. |
| Using all rows (not filtering `M = 1`) for v1 | M>1 rows have distractors that v1 isn't designed to handle. SQL-filter to M=1 gives clean same-speaker pairs without code complexity. |
| Building the custom slot-aware CFG variant before v1 ships | Not blocking. Drop-text works for M=1; v1 can ship without any custom CFG code. |

---

## 8. Phase plan

Each phase ships as a separate PR. Each phase ships with determinism
tests (`koba/v2/core/determinism_test.py` precedent).

| Phase | PR | Status | Scope | Validates |
| --- | --- | --- | --- | --- |
| **1. Plumbing on dummy data** | PR-1a | **✓ landed 2026-05-01** | `MultiAudioDecoder`, `RefAudioTruncator`, `handle_a2a` (M=1), `default_a2a.py`, `OmniA2APackingConfigKobaV2`, `DummyA2ADatasetConfig`, `debug_local_a2a_dummy` config (2-layer model, no FSDP/compile). | Seq-plan structure, tokenization, packing, MRoPE wiring, model fwd/bwd, optimizer step. **No** Lance, **no** real audio. **See §14 for the implementation log.** |
| **2. Real Lance, tiny model** | PR-1b | scope + schema locked (§15) | Replace dummy with real `mls_eng_zs_tts_a2a_v1.lance` table + `M = 1` filter. Keep 2-layer model. | Multi-audio decode, ref truncation determinism, schema-column wiring, end-to-end on a real row. ~5 min smoke test. |
| **3. Real Lance, real model** | PR-2 | planned | 0.6B / 2B Qwen3-VL+MMAudio configs (mirror `exp001a` / `exp001c`). 4–8 GPUs, ~200 steps. CFG flipped to `OmniCFGDropoutLast` once basic training works. | Loss declines, FSDP behaves, audio quality at decode time looks plausible. |
| **4. Bucketed loader** | PR-3 | planned | Add `target_duration` bucketing, two-shape-class VAE batching extension. | Throughput parity / improvement vs. un-bucketed; correctness regression test against phase 3. |
| **5. M>1 (v2)** | PR-4 | planned | Extend `handle_a2a` to multi-ref + slot tags, list-aware decoder/truncator, switch from v1 prompt to curator `task_template`. Lift `M = 1` filter. | Multi-ref training stability; voice cloning under distractors. |

Phase 5 (v2) re-opens the CFG question and may add the custom
slot-aware variant if needed.

---

## 9. Open questions / TBD

1. **`a2a_task_prompt` exact wording (v1).** Sketched as
   `"Generate the following transcript:\n"` for T2A byte-parity. Could
   also be A2A-flavored (`"Speak the following transcript in the same
   voice:\n"`) to make the ref's role explicit. v1 starts with T2A
   parity (lets a T2A checkpoint init cleanly); ablate later if voice
   cloning is weak.
2. **Schema-field column names.** The §2 table is from the curator
   spec; verify against the actual Lance writer output once curation
   v2 lands. Action: dump `lance.dataset.schema` and pin config knobs.
3. **`drop_all_ratio` if we ever switch to `OmniCFGDropoutMixed`.**
   Currently irrelevant (v1 uses `OmniCFGDropoutLast`, no drop-all
   branch). If v2's experimentation argues for Mixed, check
   `image_edit_supermax.py` / `storyboard.py` for the parity value
   (likely 0.1).
4. **First-run dataset.** `mls_eng` is the only ready source.
   `emilia-yodas` (24 kHz native) joins later under the same row
   schema.
5. **`sample["media"]` ordering convention with multiple Media
   elements.** v1 sets `[ref_media, target_media]`. The convention is
   plausibly "match `sequence_plan` VAE-element order," but no
   downstream consumer in T2A exercises a multi-Media list. Worth
   verifying once the implementation hits real consumers — if a
   downstream consumer assumes a different order, fix at that seam.

---

## 10. Quick reference — what changes by file

For the v1 PR.

| File | Change |
| --- | --- |
| `lib/koba/koba/processor/omni_audio_packed_ops.py` | Add `handle_a2a` method, add `a2a_*` config fields, add `"a2a"` branch in `forward` dispatch |
| `lib/koba/koba/processor/audio_ops.py` (or new file) | New `MultiAudioDecoder` class (or extend `AudioDecoder`) |
| `lib/koba/koba/processor/omni_audio_ops.py` (or new file) | New `RefAudioTruncator` class with `Seedable` |
| `lib/koba/koba/pipelines/default_a2a.py` (new) | A2A-pipeline factory; near-clone of `default_t2a.py` |
| `projects/kuma/kuma/projects/omni/audio/data/omni_a2a_packing_koba_v2.py` (new) | A2A packing config; near-clone of T2A's |
| `projects/kuma/kuma/projects/omni/audio/configs/datasets.py` | Add `mls_eng_a2a` dataset factory (with `M = 1` SQL filter) |
| `projects/kuma/kuma/projects/omni/audio/configs/tasks/a2a.py` (new) | Experiment configs (`debug_local_a2a`, `exp001a_a2a` mirroring `exp001a_0p6b_mmaudio_softcap`) |
| `projects/kuma/kuma/projects/omni/audio/data/dummy_dataset.py` | Add `DummyA2ADatasetConfig` |
| `projects/kuma/kuma/projects/omni/audio/studies/a2a.py` (new) | Study entries for launch |

No changes required in:

- The trainer (`OmniAudioTrainer`) — un-bucketed v1 doesn't exercise
  the two-shape-class batched-VAE path.
- Tokenizer code — no new specials.
- `OmniElementVAEAudio`, `OmniElementText`, `OmniQwen3Tokenizer`,
  `OmniPositionIDMRoPE` — all reused.
- `koba_shared/processor/tokenized_types.py` — no new fields on
  `SequenceElement` or `TokenizedSequenceElement`.
- Loss heads — diffusion loss already targets `noisy_vae_token_mask=1`
  positions, which only fire inside `NOISY_VAE_AUDIO` elements (i.e.
  the target).

---

## 11. Pre-implementation insights from companion docs

A few high-leverage observations from the deep-dive and lifecycle docs
that drove this design. Captured here so the next reader doesn't have
to re-derive.

- **The `SequenceElement` abstraction is variable-length and
  task-shape-agnostic.** T2A's two-element list is a degenerate case;
  T2I/bagel already exercises arbitrary interleaved elements. A2A
  "just" emits a different element list. (Deep-dive §11.0.)
- **`CLEAN_VAE_AUDIO` already exists as a `SequenceType` enum value,
  with full per-element-type mask machinery wired up.** Refs slot in
  with no plumbing changes — the existing `OmniElementVAEAudio`
  processor produces the right per-token masks based on the element
  type tag (no loss, `clean_vae_token_mask=1`). (Deep-dive §11.3,
  §8.2.)
- **`attention_mode = "full"` for refs and `"noise"` for target gives
  exactly the right visibility.** Refs are bidirectional + outbound
  visible (downstream attends to them); target is bidirectional self
  + outbound-invisible (diffusion isolation). No new attention modes
  needed. (Deep-dive §8.1.)
- **Element-list ordering is task-pipeline state, not model state.**
  The model's only awareness of A2A vs T2A is via
  `x_vae_by_modality="a2a"` plus the element list it sees. This is
  why the seq-builder is the single source of truth for task semantics.
  (Deep-dive §9.7.)
- **CFG-dropout class choice is task-shape-driven, not modality-driven.**
  Pure-generation tasks (T2A, T2I, SISO) use `OmniCFGDropout`
  (drop-all). Editing-shape tasks (image-edit, storyboard, A2A) use
  `OmniCFGDropoutLast` or `OmniCFGDropoutMixed`. (Lifecycle doc,
  Special-topic table — explicitly anticipates A2A.)
- **Bucketing exists to make batched-VAE encoding efficient by giving
  every row uniform audio shape per pack.** A2A's two-shape-class
  pack is a small extension, not a structural break. (T2A v2
  `_PadAudioToCeiling` precedent.)

---

## 12. Reference-audio mental model: static vs dynamic, and what it implies

Added 2026-05-01 after a multi-round discussion on what we want the
reference audio to *signal* to the model — and what we explicitly do
**not** want it to signal. This section captures the mental model
behind the v1 design choices (truncation window, random crop, refs
through the same VAE as targets) plus the reasoning that ruled out
more aggressive approaches like splice-and-reverse augmentation.

The audience is a future reader (or future-us) trying to understand
"why does the v1 reference path look like this and not like X" without
re-deriving the trade-offs from scratch.

### 12.1 The decomposition: static vs dynamic voice properties

Voice information decomposes cleanly into two axes:

| Axis | What it carries | Examples | Speaker-bound? |
|---|---|---|---|
| **Static** | Vocal tract anatomy → spectral envelope, formant positions, baseline pitch range, timbre | "How the speaker's voice fundamentally sounds" | Yes — invariant across utterances |
| **Dynamic** | Prosody, emotion, speaking rate, energy, pause patterns, intonation contour | Same speaker reading angrily vs calmly, fast vs slow | No — varies with context, mood, content |

For a reference-TTS model, we want the model to learn that **static
properties are the speaker-ID signal it should imitate** (the same
speaker should produce the same timbre regardless of what they're
saying or how they feel), while **dynamic properties should be driven
by the text instruction** (or in v1's bare-bones case, just emerge
naturally from the model's prior over normal speech).

The key consequence: a same-speaker reference clip carries *both*
axes. The static axis is what we want; the dynamic axis is incidental,
and ideally the model learns to be *invariant* to it.

### 12.2 The prosody-leakage failure mode we're guarding against

The failure mode: the model takes a shortcut. Instead of extracting
"this speaker's static voice properties" from the reference and
applying them to whatever prosody the text asks for, it extracts
**both** static and dynamic from the ref and copies the dynamic
component verbatim. At inference:

- Reference is calm → model generates calm speech, regardless of
  text content.
- Reference is angry → model generates angry speech, regardless of
  text content.
- Reference has a specific intonation arc → model copies that arc,
  bending the target text to fit.

This is a real failure mode in zero-shot TTS, well-documented in the
literature. The cure is to train the model on data where ref and
target have **decorrelated** dynamic content but **correlated** static
content, so the only reliable signal the model can extract from the
ref is the static part.

### 12.3 Three approaches in the literature

Different systems address prosody leakage in different ways:

| Approach | Examples | How it works | What it costs |
|---|---|---|---|
| **Architectural decomposition** | StyleTTS, StyleTTS 2 | Two separate encoders — one for static identity, one for dynamic style. Decoder consumes them separately. Prevents cross-pollination by construction. | Extra model component; more training complexity. |
| **Cross-utterance pairing** | VALL-E, NaturalSpeech 2, VoiceBox | Data-side: pair ref and target from *different utterances* of the same speaker. Their dynamics are naturally different. The model learns prosody varies across same-speaker pairs while voice identity doesn't. | Curator-side complexity; relies on dataset structure. |
| **Multi-ref averaging** | Tortoise TTS | Feed multiple ref clips. Each has its own dynamics; only static identity is shared. Averaging suppresses the dynamic component. | More refs per row; longer effective context. |

Our design uses **approach 2 (cross-utterance pairing)**, which is
the simplest and most-deployed. v2's distractor-sampling rule
implicitly adds **approach 3 (multi-ref averaging)** when M ≥ 2 and
multiple slots happen to land on the target speaker (~5.8% of rows
per the curator spec).

We do **not** use approach 1 (architectural decomposition) — refs go
through the same MMAudio VAE as targets, not a dedicated speaker
encoder. This is a deliberate constraint, not an accident. See §12.7
for the implications.

### 12.4 Our v1 design: three stacked decorrelation layers

v1 stacks three independent layers of prosody-decorrelation, each
contributing differently to making the static signal the only
exploitable shortcut. None of the three is novel on its own; the value
is in stacking them.

#### Layer 1 (curator-side): cross-utterance pairing via cross-book preference

The curator's slot-0 sampling rule (per the row schema spec):

> "Slot 0 prefers cross-book. With probability 80%, the slot-0
> same-speaker reference is sampled from a clip whose `book_id`
> differs from the target clip (mls_eng's `book_id` corresponds to
> one LibriVox recording session)."

This is the dominant source of decorrelation. Same speaker, different
recording sessions = same static voice ID, **completely different
dynamic context** (different book, different reading session, possibly
weeks apart, different room/mic settings). The curator spec also
calls out the acoustic-channel mitigation angle:

> "without it, the model can identify 'speaker-x' by mic / room /
> register cues that share across the same recording session, instead
> of learning timbre."

#### Layer 2 (loader-side): random cropping at training time

`RefAudioTruncator` does random crop during training, center crop at
eval. Random crop's purpose isn't just regularization — it's
specifically that **across epochs, the model sees different 3 s
slices of the same ref clip**. Each slice has its own local prosodic
fragment. No single prosodic pattern can be exploited as a shortcut
because it's not stable across epochs.

Center crop at eval is for determinism (same input → same output for
reproducible benchmarks); the prosody-decorrelation argument applies
only at training.

#### Layer 3 (window length): 3 s as the static/dynamic sweet spot

Within 3 s of speech:

- **Voice timbre and formant structure are stable** — saturated
  speaker-ID signal. Speaker-verification models routinely achieve
  > 95% closed-set accuracy from 3 s.
- **Prosody is fragmentary** — you might catch a single phrase
  contour or two, but not a full sentence's prosodic arc. Multi-clause
  intonation patterns exceed 3 s.
- **Random-crop variance is meaningful** — different 3 s slices of
  the same clip have visibly different prosodic fragments.

This is the static/dynamic sweet spot. Going to 5 s or 10 s would give
the model more coherent prosody to potentially copy. Going below 2 s
would risk losing static identity stability. The literature converges
on 3 s for similar reasons (VALL-E, NaturalSpeech 2, VoiceBox all use
3 s).

### 12.5 RefAudioTruncator's role and why we keep it even when upstream pre-truncates

A natural question raised during design: **if the curator already
writes refs at exactly 3 s upstream, is `RefAudioTruncator` just
no-op overhead?**

The answer is "no" — we keep the class for five reasons that make it
a structural element of the pipeline rather than removable cruft:

1. **Single source of truth for the ref-window contract.** Without
   the truncator, the contract ("refs are 3 s") is implicit in the
   curator's behavior. With the truncator, it's explicit in the
   loader. When someone changes the curator later (new dataset variant,
   parameter rename, forgotten audit), the truncator catches the
   drift; downstream code never sees mismatched shapes.
2. **Defensive against drift.** Curator-side invariants are easy to
   violate accidentally. The truncator runs every row, makes the
   assumption explicit, and is cheap (microseconds for a tensor slice
   or pad).
3. **Future-augmentation hook.** Pitch shift, random crop offset,
   time stretch, multi-clip concat — the natural place for any of
   these is the truncator. Having the class skeleton + position in
   the pipeline already wired means future iterations touch one file,
   not three.
4. **Consistency with existing pipeline patterns.** T2A's
   `_PadAudioToCeiling` is also a no-op when `pad_duration_ceil_sec=0`.
   The "always-present, sometimes-no-op" pattern is established;
   following it is one less surprise for readers.
5. **Dummy/debug datasets benefit.** When v1 ships with
   `DummyA2ADatasetConfig` emitting seeded random tensors, the dummy
   may produce arbitrary durations. The truncator normalizes them to
   `ref_window_sec` consistently with the real-data path. Without it,
   the dummy would need its own truncation logic.

The right framing: **the truncator's job is to enforce the
ref-window contract regardless of upstream behavior, with the
no-op-when-already-fixed case being a happy path.** That's a sound
architectural pattern.

### 12.6 v2's implicit upgrade: Tortoise-style multi-ref via distractor sampling

A subtle benefit of the v2 design (M ∈ {1, 2, 3} with distractors)
that's worth flagging now:

When 2+ ref slots happen to land on the target speaker (per the
curator's distractor rule, ~5.8% of M ≥ 2 rows), the model sees
**multiple-utterance same-speaker refs** — exactly Tortoise's setup.
Each ref has its own prosody (different sentences, possibly different
sessions); only static identity is shared. The model's downstream
attention layer over multiple distinct ref encodings naturally
suppresses the dynamic component — implicit averaging.

This is approach 3 from §12.3 (multi-ref averaging) layered on top of
approach 2 (cross-utterance pairing) for free, just by virtue of the
curator's distractor-sampling rule. v2 effectively combines two
literature approaches without paying for the explicit Tortoise-style
multi-clip retrieval.

If v1 prosody leakage is observable, **the natural escalation path
is to ramp up the same-speaker fraction in non-slot-0 slots**
(curator-side change), making this implicit averaging more frequent
without changing the loader.

### 12.7 The constraint that quietly drives most v1 decisions: frozen MMAudio VAE

v1 sends both refs and targets through the **same frozen MMAudio
VAE**. We do not use a dedicated speaker encoder. This is a deliberate
constraint reflecting:

- **Architectural simplicity.** No second model component to train,
  serve, or version.
- **Implementation reuse.** Refs and targets share the entire
  acoustic encoding stack — no special-casing in the trainer or
  inference path.
- **Conservative scope.** Adding a speaker encoder is a meaningful
  axis of model complexity; we defer it until we know whether it's
  needed.

The cost: refs must look like *natural speech* to the VAE. The VAE
was trained on natural speech and will produce useful latents only
for in-distribution inputs. This constraint cascades into several v1
design choices:

| v1 choice | What the VAE constraint forces |
|---|---|
| Reference is raw waveform, not a pre-extracted embedding | The VAE is the encoder; we can't bypass it. |
| No splice-of-disparate-clips augmentation (see §12.8) | Splice boundaries are OOD for the VAE. |
| No time-reversal augmentation (see §12.8) | Reversed speech is OOD for the VAE. |
| Augmentations limited to "still-natural-speech" transforms | E.g., mild pitch shift, mild time stretch, random natural-window crop — all *might* stay in-distribution. |
| If we want aggressive augmentation later, we have to swap the encoder | E.g., dedicated speaker encoder trained on aggressive augs. Real cost. |

This is the constraint to remember when evaluating future iteration
ideas: **does this transform produce something the MMAudio VAE was
trained to encode?** If yes, viable; if no, we either stay with the
status quo or commit to encoder-swap-level changes.

### 12.8 Augmentations considered and ruled out for v1

#### Concat-multiple-clips + time-reversal

**The idea:** Synthesize a "speaker-stationary but content-incoherent"
reference by splicing together disparate same-speaker 3 s slices,
optionally time-reversing some of them. Intent: aggressively destroy
all dynamic content (sentence-level prosody, intonation arcs,
phonetic content) while preserving static voice identity.

**Why it's clever:** Maps cleanly onto the static/dynamic
decomposition. Splicing destroys cross-segment dynamic continuity;
time-reversal destroys phonetic content and intonation arcs while
preserving the *spectral* characteristics that carry timbre. Both
are well-known augmentations in speaker-encoder training literature.

**Why we ruled it out for v1:**

1. **VAE distribution mismatch (§12.7's constraint).** Splice
   boundaries have spectral discontinuities — even with crossfade,
   they don't look like natural speech. Time-reversed speech has
   unnatural envelope shapes (slow onsets, fast decays — opposite of
   natural). Both are OOD for the frozen MMAudio VAE; the latents
   won't be representative of speaker identity.
2. **Train/inference distribution mismatch.** At inference, refs are
   real natural audio. If we train on splices/reverses, the encoder
   has learned something specific to that augmentation that may not
   transfer to natural-ref inference.
3. **Marginal benefit over stacked layers 1–3.** The decorrelation
   already happens via cross-book + random crop + 3 s window. The
   marginal gain from splice/reverse is "destroys *more* dynamic
   content" — but the existing layers already destroy enough that the
   model can only extract static signal in expectation.
4. **Multi-ref averaging (v2 free) does the same job without OOD
   audio.** Each ref in v2's M ≥ 2 same-speaker case is real natural
   audio (in-distribution), and dynamic averaging happens implicitly
   via the model's attention over multiple distinct utterances.

**Where this *could* live:** v3+ when we're willing to swap the frozen
MMAudio VAE for a dedicated speaker encoder trained on aggressive
augmentations including splice + reverse. Real cost; only worth it if
we observe persistent prosody leakage that the cheaper escalations
(§12.6, §12.9) don't fix.

#### Per-batch adaptive ref shape (longest-in-batch)

**The idea:** Instead of a global fixed 3 s window, pick the ref
duration per batch — e.g., truncate every ref in the batch to the
longest ref in that batch.

**Why ruled out:**

1. **No silence-padding waste.** With longest-in-batch, batches
   containing one 15 s ref force every other ref to pad to 15 s with
   silence. The padded silence still gets VAE-encoded — burning
   ~5× compute for no information gain.
2. **Predictable shape across steps matters for compile/kernel
   selection.** Variable shape per batch invalidates compile caches
   and forces kernel re-selection. Fixed shape gets one warm path.
3. **Voice-cue argument cuts the other way.** We only need
   static voice properties; 3 s is enough. Going longer doesn't add
   useful voice signal — just compute cost.

**The compute math (recap from earlier discussion):** for a pack with
N rows, fixed-3-s refs at 16 kHz cost `N × 48000` VAE samples.
Adaptive-to-bucket-max (e.g., 15 s) costs `N × 240000` — 5× more.
Compute savings dominate any dispatch-overhead savings from a single
combined VAE call.

**Decision:** Per-row fixed 3 s window. Predictable, compute-efficient,
literature-precedented.

### 12.9 Future iteration paths, ordered by VAE-compatibility

If v1 prosody leakage shows up in evals (e.g., generated speech
inappropriately mirrors reference emotion/style), here's the
escalation ladder, easiest first:

| # | Axis | What changes | VAE-compat? |
|---|---|---|---|
| 1 | **Stronger curator cross-book preference** | Curator-side: push the 80% cross-book preference to 95–100%. | Yes. No model change. |
| 2 | **More multi-same-speaker rows in v2** | Curator-side: increase the same-speaker fraction in non-slot-0 slots (Tortoise-style averaging more often). | Yes. No model change. |
| 3 | **Random-window-offset augmentation** | Loader-side: instead of one random crop per epoch, sample multiple random 3 s crops per row across packs. | Yes. Stays within natural-speech distribution. |
| 4 | **Mild pitch-shift / time-stretch augmentation** (~5%) | Loader-side: apply small natural perturbations to refs. Forces invariance within a margin of static pitch range. | Likely yes — small perturbations may stay in-distribution for the VAE. Needs verification. |
| 5 | **Larger pitch-shift / time-stretch** (~20%) or **VTLP** | Loader-side: more aggressive perturbations. Strong prosody-decorrelation. | Maybe — larger transforms increase OOD risk for the frozen VAE. |
| 6 | **Splice / time-reversal** | Loader-side: synthetic speaker-stationary content-incoherent refs. | No — OOD for frozen VAE. Requires encoder swap. |
| 7 | **Dedicated speaker encoder** (ECAPA, GE2E, or trained-from-scratch) | Architectural: frozen-MMAudio-VAE on refs replaced by a model whose only job is speaker embedding. | N/A — replaces the VAE-on-refs path entirely. |
| 8 | **Two-encoder architecture** (StyleTTS-style: separate static + dynamic encoders) | Architectural: full split between speaker-ID encoder and prosody encoder. | N/A — model-architecture change. |

The v1 plan is to ship with the three-layer stack (§12.4) and observe.
v2 picks up #2 implicitly via distractor sampling. Anything past #2
is a deliberate escalation triggered by observed leakage.

### 12.10 Cross-references

- §3.1 (audio format normalization) — upstream curator pre-converts
  to 48 kHz mono WAV.
- §3.2 (reference truncation) — the "3 s window, random/center crop"
  mechanic that this section justifies.
- §4.3 (`MultiAudioDecoder` / `RefAudioTruncator`) — the v1 processors.
- §5.4 (CFG dropout under distractors) — separate concern, but shares
  the "what does the model actually learn from the reference" framing.

### 12.11 Summary table

| Question | Answer |
|---|---|
| What is the reference *for*? | To signal **static** voice properties (timbre, vocal-tract characteristics) to the model. |
| What should the reference *not* signal? | **Dynamic** properties (prosody, emotion, intonation arc) — those should come from the text instruction at inference. |
| How do we prevent dynamic leakage in v1? | Three stacked layers: curator's 80% cross-book pairing + per-epoch random crop + 3 s window length. |
| Why 3 s specifically? | Long enough for static voice ID to saturate, short enough to limit prosodic coherence within the window. Matches VALL-E / NaturalSpeech 2 / VoiceBox precedent. |
| Why a fixed window and not adaptive? | Predictable shape; no silence-padding compute waste; compute math favors fixed 3 s by ~5× over bucket-adaptive. |
| Why keep `RefAudioTruncator` if upstream pre-truncates? | Single source of truth for the contract; defensive against drift; future-augmentation hook; pattern consistency; dummy-dataset normalization. |
| What ruled out splice + time-reversal? | Frozen MMAudio VAE — splice boundaries and reversed speech are out-of-distribution. Multi-ref averaging (v2 free) achieves the same goal without OOD audio. |
| If v1 leaks, what's the next step? | Curator: 80% → 95–100% cross-book. v2: ramp same-speaker fraction in distractors. Then mild pitch/time augmentation, then encoder swap. |

---

## 13. Channel-dim convention: pre-VAE tensor-rank discipline

Added 2026-05-01 after a discussion during PR-1a implementation. The
A2A audio path stretches from raw bytes (Lance row) through several
processors (decode, normalize, truncate) and into the VAE. At which
stages does the "audio channel" concept exist, and how should pre-VAE
processors handle it consistently?

This section captures the convention adopted in PR-1a, scoped tightly:
**the channel dim is a pre-VAE concept only**.

### 13.1 The pre-VAE / post-VAE distinction

| Stage | Tensor shape | What "channel dim" means |
|---|---|---|
| **Pre-VAE** (decoder → AudioToX → truncator → batch stacking) | `(C=1, T_samples)` per row, `(N, 1, T_samples)` after stacking | **Audio channels** — mono = 1, stereo = 2, etc. v1 fixes at 1; future stereo would flip it. |
| **VAE encoding boundary** (`OmniElementVAEAudio` reads `shape[-1]`; trainer's `_encode_audio` calls VAE) | input `(N, 1, T_samples)`, output `(N, T_latent, 20)` for MMAudio 16k | The `1` in input is audio channels; the `20` in output is the VAE's **internal feature dim** (latent channels), entirely unrelated. |
| **Post-VAE** (latents flow through transformer streams, MRoPE, attention, denoiser) | `(N, T_latent, 20)` flattened into per-token slots in the packed sequence | "Channels" no longer means audio channels. The packed-sequence-level mask machinery (`vae_token_mask`, `clean_vae_token_mask`, etc.) operates on per-token slots, agnostic to any channel notion. |

The convention applies **only** to the pre-VAE column. Once the VAE
runs, we leave the channel-dim regime behind — code that consumes
post-VAE latents must not reuse the pre-VAE convention.

### 13.2 Why we made the channel dim explicit

Before PR-1a, the koba T2A path had a **soft inconsistency**:

- `torchcodec.decoders.AudioDecoder(..., num_channels=1)` returns 2D
  `(C=1, T)`.
- `OmniElementVAEAudio` and `CaptionAugmentation` both consume the
  tensor via `shape[-1]`, so they handle 1D and 2D uniformly.
- The legacy `AudioDecoder` (singular) wrote
  `audio_duration_seconds = shape[0] / sr` — buggy for 2D output
  (returned `num_channels / sr ≈ 6.25 × 10⁻⁵` for mono at 16 kHz).
  The bug was latent because no production consumer reads
  `audio_duration_seconds` (`CaptionAugmentation` computes its own
  duration from the tensor's last dim).

PR-1a's audit caught this and fixed it (`shape[0]` → `shape[-1]` in
`AudioDecoder`). More broadly, the audit revealed that the codebase
**handles** 2D input but doesn't **commit** to it. The PR-1a
convention is: commit to 2D `(C=1, T)` for raw waveforms, document
the convention, and assert it in tests.

### 13.3 The four pre-VAE shape-handling rules

For every processor that operates on raw waveform tensors:

1. **Outputs are 2D `(num_channels, num_samples)`.** With v1's
   `num_channels=1`, every tensor is `(1, T)`. Never squeeze to 1D
   without a documented reason.
2. **Sample count comes from `shape[-1]`.** Never `shape[0]` (that's
   the channel dim). `OmniElementVAEAudio` already follows this
   convention at `omni_audio_ops.py:139`.
3. **Slicing / padding samples acts on dim `-1`.**
   `RefAudioTruncator` slices via `tensor[..., start:start+window]`
   and pads via `F.pad(tensor, (0, pad_amount))`. Both work for any
   leading dims, so they handle `(1, T)` and `(C, T)` for any `C`
   uniformly.
4. **Test assertions pin both `ndim == 2` and `shape[0] == 1`.** This
   is the regression guard that prevents future code from accidentally
   squeezing the channel dim or producing a 1D tensor.

### 13.4 Why this matters: tensor-rank discipline, not semantic info

The leading channel dim of `(1, T)` doesn't carry semantic
information for v1 mono audio — it's `1`, period. So why keep it?

- **Stable layout** for processors that need to slice/pad/stack
  without case analysis. `tensor[..., :N]` always means "first N
  samples"; never have to disambiguate "samples on dim 0 or dim 1?".
- **Multi-channel readiness.** If a future stereo or multi-channel
  mode lands (some speaker-encoder literature uses stereo refs), it's
  a one-line `num_channels` Config change. Nothing else in the
  pipeline assumes mono. Without rank discipline, the change would
  ripple through every shape access.
- **Aligns with torchcodec's native output.** No squeezing or
  reshaping noise.

### 13.5 Where the convention is enforced

- **`MultiAudioDecoder` class docstring** (`audio_ops.py`): explicitly
  scopes the convention to the pre-VAE stage; warns that post-VAE
  shapes are different.
- **`RefAudioTruncator.forward`**: slices on dim `-1`, preserving the
  leading channel dim.
- **`audio_ops_test.py:test_decode_basic`**: asserts `ndim == 2` and
  `shape[0] == 1` for both target and ref tensors.
- **The `AudioDecoder` `audio_duration_seconds` bug fix** (commit
  `fc841e8d00`): also indexes via `shape[-1]`, aligning the legacy
  writer with the convention.

### 13.6 Cross-references

- §3.1 (audio format normalization) — upstream curator pre-converts
  to 48 kHz mono WAV.
- §4 (v1 design) — pipeline stages that operate under this convention.
- `omni_t2a_dataloader_deep_dive.md` §11.0 — "`SequenceElement` is
  the anchor"; per-token masks and post-VAE concepts live below the
  element abstraction.

---

## 14. PR-1a implementation log (landed 2026-05-01)

PR-1a delivered the koba-side processors + kuma-side packing config +
Lance-free dummy + smoke-test debug config. **All plumbing is built;
the only runnable entry point uses dummy data** (real Lance wiring is
PR-1b — see §15).

### 14.1 Branch + commit summary

Branch: `dongguo/omni-a2a-plumbing` (off `origin/main` at
`b0592047d1`). 7 commits, 110 net-new tests, ~3 K LOC across koba +
kuma.

| Commit | Lines | Tests | Content |
|---|---|---|---|
| `fc841e8d00` | +506 / −8 | 14 | `MultiAudioDecoder` (A2A row decoder) + `AudioDecoder.audio_duration_seconds` bug fix |
| `fc14623d40` | +338 | 12 | `RefAudioTruncator` (3 s window, random/center crop, silence/reject pad) |
| `97620bab7e` | +359 / −9 | 17 | `handle_a2a` method on `OmniAudioSeqBuilder` (3-element seq plan) |
| `79d6918dc1` | +468 | 15 | `default_a2a` pipeline factory + `A2APipelineParams` |
| `fc4371cad8` | +572 | 21 | `OmniA2APackingConfigKobaV2` (kuma packing config + kuma → koba field-bridge tests) |
| `80ae447e58` | +597 / −5 | 21 | `DummyA2ADatasetConfig` (3-element-per-sample packed-batch generator) |
| `7901923b26` | +240 | 10 | `debug_local_a2a_dummy` experiment config (smoke-test entry point) |

### 14.2 Per-component shape

#### `MultiAudioDecoder` (`lib/koba/koba/processor/audio_ops.py`)

- Reads `target_audio_bytes` (scalar) + `ref_audio_bytes[0]`
  (`list<binary>`, length 1 in v1).
- Decodes both at the model SR (16 kHz) via shared
  `_decode_audio_bytes` helper.
- Defensive asserts: `target_audio_sr == 48000`,
  `ref_audio_srs[i] == 48000` (skipped when fields absent — dummy
  datasets).
- Raises `ValueError` if `len(ref_audio_bytes) != 1` (v1 invariant
  enforcement).
- Emits T2A-compatible metadata keys (`audio_sample_rate`,
  `audio_duration_seconds` for the **target**) so any future opt-in
  T2A consumer (e.g. `CaptionAugmentation`) reused for A2A continues
  to work.

#### `RefAudioTruncator` (same file)

- Per-row: random / center crop + silence / reject pad to a fixed
  window (default 3 s × 16 kHz = 48 000 samples).
- Slices on dim `-1`, preserves channel dim.
- Determinism: uses Python's module-level `random.randint`, seeded
  by koba's per-worker per-epoch `_seed_everything(worker_seed)` (see
  koba/CLAUDE.md "Determinism" section). No `Seedable` instance RNG.
- Acts as a no-op when input is already at exactly `window_samples`.
  Class kept anyway for the five reasons in §12.5 (single source of
  truth, drift defense, augmentation hook, pattern consistency,
  dummy normalization).

#### `OmniAudioSeqBuilder.handle_a2a` (`omni_audio_packed_ops.py`)

- New per-task handler alongside `handle_t2a`. Dispatched on
  `sample["conversation_modality"]`.
- New `a2a_*` config fields: `a2a_target_audio_tensor_key`,
  `a2a_ref_audio_tensor_key`, `a2a_transcript_key`,
  `a2a_task_prompt`. Defaults match the curator schema
  (`target_audio_tensor`, `ref_audio_tensor`, `target_text`,
  `"Generate the following transcript:\n"`).
- Emits the 3-element seq plan
  `[CLEAN_VAE_AUDIO(ref), TEXT(prompt+transcript), NOISY_VAE_AUDIO(target)]`
  with `loss=False`, `loss=False`, `loss=True` respectively.
- Sets `sample["media"] = [ref_media, target_media]` (VAE-element
  order); the seq-plan's `Media` references are the **same objects**
  as the media-list entries (verified by test).

#### `default_a2a.py` pipeline factory (`lib/koba/koba/pipelines/`)

- `A2APipelineParams` dataclass holds all the pipeline-shaping knobs
  (column keys, ref window, CFG dropout list, tokenizer path, etc.).
- `default_a2a_pipeline_processors` composes 12 processors:
  `MultiAudioDecoder` → `AudioToX(target)` → `AudioToX(ref)` →
  `RefAudioTruncator` → `AddDummyConversationModality(value="a2a")`
  → `OmniAudioSeqBuilder` → `OmniCFGDropoutLast` →
  `OmniAddTokenizedSequenceElement` → `OmniElementVAEAudio` →
  `OmniElementText` → `OmniQwen3Tokenizer` →
  (optional `OmniSequenceLengthFilter`) → `OmniPositionIDMRoPE`.
- CFG dropout default: `cfg_dropout_modalities=[]` (no-op for first
  runs; flip to `["a2a"]` once basic training works — see §4.6).

#### `OmniA2APackingConfigKobaV2` (`projects/kuma/.../data/`)

- Kuma-side wrapper that ties `default_a2a_pipeline_processors` to
  V2's `PackingDataset`. Reuses the inner `OmniT2APackingDatasetKobaV2`
  IterableDataset (its body is task-agnostic; the "T2A" in the name
  is historical and worth a future cleanup).
- `dataset_row_filter="M = 1"` default — restricts to same-speaker
  rows for v1.
- Kuma → koba field propagation via `configo.merge_into_copy` plus
  explicit overrides for fields whose names differ
  (`audio_sample_rate` ⇒ `sample_rate`).

#### `DummyA2ADatasetConfig` (`projects/kuma/.../data/dummy_dataset.py`)

- 3-element-per-sample packed-batch generator that bypasses the
  entire pipeline. Yields packed dicts with the A2A v1 schema
  directly.
- Per sample: `[ref_tokens (clean VAE), text_tokens, target_tokens
  (noisy VAE)]`. Token budget:
  `ref_tokens = ref_window_sec × audio_sample_rate /
  audio_compression_factor` (≈ 93 by default), text =
  `text_tokens_per_sample` (default 50), target = remainder.
- 2 `x_vae` entries per sample (ref + target, ordered ref-first),
  flattened to `2 × batch_size` total.
- `clean_vae_token_mask` non-empty (covers ref positions; T2A's
  dummy has it all-zero).
- "Schema-faithful enough" for trainer + loss + denoiser smoke
  testing — not byte-perfect with the real pipeline (boundary tokens
  inside VAE elements skipped; per-sample summary fields stay
  T2A-shaped).

#### `debug_local_a2a_dummy` (`projects/kuma/.../configs/tasks/a2a.py`)

- Inherits `exp001a_0p6b_mmaudio_softcap` (T2A scaffolding) and
  overrides:
  - `tracker.project_name → "omni-a2a"`, all trackers off.
  - `denoiser.module.depths → [2]` (production-only depth shrunk).
  - `parallelism_config`: `dp_shard=-1` (auto-scale from world
    size), `dp_replicate=1`, `ulysses_cp=ring_cp=1` (CP pinned
    so a production CP turn-on doesn't silently break debug).
  - `denoiser.compiler / checkpointing / activation_checkpointing →
    None` for fast iteration.
  - Dataset: `OmniA2APackingConfigKobaV2(audio_datasets=[])` first
    (exercises kuma → koba field-bridge at config-build time), then
    `merge_into_copy(DummyA2ADatasetConfig(), …)` to swap the
    actual data source to the dummy.
- Runs as `torchrun --standalone --nproc_per_node 1 main.py
  --config kuma.projects.omni.audio.configs.tasks.a2a.debug_local_a2a_dummy
  --name debug_omni_a2a_dummy`. Auto-scales to any `nproc_per_node`.

### 14.3 Implementation-time decisions made (not in the original design)

These were resolved during PR-1a coding, beyond what the design doc
specified up-front:

| Decision | What we picked | Why |
|---|---|---|
| `MultiAudioDecoder` writes T2A-compat metadata keys | Yes — `audio_sample_rate`, `audio_duration_seconds` (for target). | Avoids silent feature-loss if any future config reuses `CaptionAugmentation` (which reads `audio_sample_rate`). Originally proposed to skip; pushback during audit corrected to write. |
| `audio_duration_seconds` bug in legacy `AudioDecoder` | Fixed: `shape[0]` → `shape[-1]`. | Bug was latent (no production reader). Fix is correctness-neutral but aligns the writer with `CaptionAugmentation` and `MultiAudioDecoder`'s `shape[-1]` convention. Pre-existing TODOs to add `slots=True` also resolved. |
| Test fixture for synthetic WAV bytes | `torchaudio.save` to a temp `.wav` file (not BytesIO). | New torchcodec-backed `torchaudio.save` determines format from file extension, not a kwarg. BytesIO without an extension fails. |
| `RefAudioTruncator` RNG seeding | Module-level `random.randint`, no `Seedable`. | Module-level RNG is auto-seeded per worker per epoch by koba's `_seed_everything(worker_seed)`. Adding `Seedable` would be cleaner protocol-wise but added boilerplate without functional benefit. Determinism test (`test_random_crop_determinism_with_seed`) verifies the contract. |
| `OmniA2APackingConfigKobaV2` dataset class reuse | Reused `OmniT2APackingDatasetKobaV2` (T2A-named) instead of duplicating. | The inner IterableDataset's body is task-agnostic — only forwards `dataset_configs` + `modalities` + `batch_size` to `create_packing_dataset_from_v1`. The "T2A" in the name is historical; a rename to `OmniAudioPackingDataset` is a future cleanup. Avoids ~70 lines of copy-pasted boilerplate. |
| Pre-commit MLR-04.01 cleanup | Added `slots=True` to `MultiAudioDecoder.Config`, `AudioDecoder.Config`, `AudioToTokens.Config`. | Verified `EasyConfig` doesn't conflict with `slots=True` (unlike `DataclassSetup`). The pre-existing TODO comments on `AudioDecoder.Config` and `AudioToTokens.Config` were stale; PR-1a finishes them off. |
| AudioToX two calls (target + ref) | Two distinct `AudioToX.Config` instances in the pipeline, with per-tensor input/output keys. | `AudioToX.forward` writes only `sample[output_key]` (no global side-effect keys), so the two calls don't collide. Verified during audit. |
| AudioToX-then-truncate vs. truncate-then-AudioToX | AudioToX runs **before** the truncator. | Peak/RMS norm uses full-clip statistics; a 3 s window's RMS would differ from the full clip. Order is load-bearing. |
| Channel-dim convention | Explicit 2D `(C=1, T_samples)`, scoped to pre-VAE only (§13). | Stable shape contract for processors; multi-channel readiness is a one-line config change. |

### 14.4 What's runnable in PR-1a vs not

| Component | Built? | Tested? | Runnable at runtime? |
|---|---|---|---|
| `MultiAudioDecoder` | ✓ | 13 unit tests | Not exercised — dummy bypasses it |
| `RefAudioTruncator` | ✓ | 12 unit tests | Not exercised — dummy bypasses it |
| `handle_a2a` | ✓ | 17 unit tests | Not exercised — dummy bypasses it |
| `default_a2a` pipeline | ✓ | 15 shape + plumbing tests | Not exercised — dummy bypasses it |
| `OmniA2APackingConfigKobaV2` | ✓ | 21 field-bridge tests | **Constructed** during config build (catches kuma → koba wiring errors); no real Lance read |
| `DummyA2ADatasetConfig` | ✓ | 21 schema tests | **Used at runtime** — generates the actual training batches |
| `debug_local_a2a_dummy` | ✓ | 10 config-shape tests | **Yes** — runnable smoke-test entry point |

PR-1b will switch the runtime data path to a real Lance table and
exercise the koba-side processors at runtime.

### 14.5 Pre-commit hook outcomes

All blocking hooks pass on every commit. Notable interactions:

- **ruff format** auto-fixed several files; re-staged after each
  auto-fix. No manual reformat needed.
- **dataclass-check (MLR-04.01)** initially blocked the
  `MultiAudioDecoder` commit (missing `slots=True`); resolved by
  verifying compatibility with `EasyConfig` and adding `slots=True`
  uniformly. Same fix retroactively cleaned up the pre-existing
  TODOs on `AudioDecoder.Config` and `AudioToTokens.Config`.
- **Non-blocking ML Rules warnings** on `default_a2a.py`:
  `MLR-04.13 Config constructor with 5+ keyword args` (3 instances).
  Same shape T2A's pipeline factory carries; documented as "pipeline
  plumbing forwards X knobs."
- **Pyright / IDE diagnostics**: many import-resolution warnings
  throughout PR-1a coding (kuma-venv mismatch in the editor's
  configured Python). All are venv-config noise, not real issues —
  the kuma venv has every dependency.

---

## 15. PR-1b plan: data table schema

Designed 2026-05-01 in advance of the curator's first A2A v1 table
landing. Captures the schema, naming, hard invariants, and forward-compat
path discussed when planning PR-1b.

### 15.1 Final schema (v1, M=1 only)

```text
─── Identity / namespacing ────────────────────────────────────────────
sample_id                  : str             — globally unique (primary key)
source_dataset             : str             — "mls_eng" (optional if per-source tables)
target_language            : str             — "en"

─── Target ───────────────────────────────────────────────────────────
target_audio_bytes         : bytes           — WAV 48 kHz mono
target_audio_sr            : int             — 48000 (asserted by loader)
target_duration            : float           — seconds (used for PR-3 bucketing)
target_text                : str             — transcript (the only text used)
target_speaker_global_id   : str             — for invariant-checking + debug
target_key                 : str             — provenance / dedup

─── References (list-of-length-1 in v1; up to length 3 in v2) ────────
ref_audio_bytes            : list<binary>    — WAV 48 kHz mono (all entries)
ref_audio_srs              : list<int>       — all 48000 (asserted)
ref_durations              : list<float>     — informational
ref_speaker_global_ids     : list<str>       — speaker IDs per slot
ref_keys                   : list<str>       — provenance per slot

─── Curator metadata ─────────────────────────────────────────────────
M                          : int             — 1 in v1; 1..3 in v2
target_speaker_local_idx   : int | None      — 0 in v1; in [0, M-1] for v2;
                                                None for future cross-speaker tasks
```

### 15.2 Field-level rationale

#### Why list-typed ref columns (not scalar)

**Decision: `list<binary>` of length 1 for all `ref_*` columns in v1.**

Two reasons to pick list over scalar:

1. **PR-1a's loader is already built to read it.**
   `MultiAudioDecoder.Config.ref_audio_bytes_key` expects
   `list<binary>` and reads `[0]`. The 13 unit tests pin this
   contract. Switching to scalar would require modifying
   `MultiAudioDecoder`, the packing config, and the field-bridge
   tests — extra surface in PR-1b for negligible gain.
2. **v2 forward-compat without schema migration.** When v2
   (M ∈ {1, 2, 3}) ships, the reference columns just grow from length
   1 to length 3. No column rename, no migration, no curator
   rewrite. Just lift the `M = 1` SQL filter.

The cost: a few bytes of list overhead per row. Negligible vs
~50 KB of audio bytes.

#### `target_speaker_local_idx` and the nullable-vs-sentinel decision

**Decision: nullable int (`int | None`), not `-1` sentinel.**

The field points to the slot in `ref_audio_bytes` whose audio is from
the target speaker. For v1 M=1, this is always `0`. For v2 M ∈ {1,2,3},
it's in `[0, M-1]` (deterministically picked when multiple refs share
speaker — the v2 spec already specifies this).

For the corner case "no ref shares speaker" (e.g., future
cross-speaker style transfer / voice-conversion tasks), three
encoding options were weighed:

| Option | Encoding | Verdict |
|---|---|---|
| **A. `-1` sentinel** | `int`, value `-1` means none | **Rejected** — Python's negative-indexing convention will mislead readers (`ref_audio_bytes[-1]` is the last element, not "none"). Real bug source in production DataFrames + Lance tables. |
| **B. Nullable int (`None`)** | `int \| None`, `None` means none | **Picked** — unambiguous semantics; idiomatic; Lance null bitmap overhead is ~1 bit/row (negligible); loader-side handling is one ternary. |
| **C. Separate boolean** | `same_speaker_ref_present: bool` + `target_speaker_local_idx: int` (only meaningful when `True`) | Rejected — two columns instead of one; redundant when `same_speaker_ref_present=True` always (v1 + v2 baseline cases). |

For v1, the field is **never null** (same-speaker invariant
guarantees a match). Nullable just records *future* flexibility
without affecting current rows.

#### What's intentionally excluded

| Column | Why excluded |
|---|---|
| Quality / `pass_filter` | Source dataset (`mls_eng`) is clean academic; no quality gate needed for v1. |
| `ref_text` / `ref_texts` | Per the user's constraint 3 — never used downstream. |
| `ref_speaker_local_ids` | Always `[0]` for M=1; redundant. Add when v2 ships if needed. |
| `task_template` | Loader builds the prompt locally for v1 (T2A-byte-parity, see §4.2). v2 will need this column. |

### 15.3 Hard invariants the curator must guarantee

These are the contract the loader trusts at read time. Any violation
is a curation bug:

1. **`target_key NOT IN ref_keys`** — target clip is never in its
   own reference pool. Same as v2 spec; prevents memorization
   shortcuts.
2. **`target_speaker_global_id == ref_speaker_global_ids[0]`** — v1
   is "same-speaker only" by construction. (v2 weakens this to
   "target speaker is *somewhere* in the ref pool," with the
   `target_speaker_local_idx` field pointing at the matching slot.)
3. **`M == 1`** for every row in this table. Either enforce by table
   content, or rely on the loader's `dataset_row_filter="M = 1"`
   (PR-1a default).
4. **All audio bytes are WAV 48 kHz mono.** Loader asserts SR = 48000;
   the WAV/mono part is implicit (decode would fail otherwise).
5. **`len(ref_audio_bytes) == 1`** for every row. Loader raises
   `ValueError` if length differs (`MultiAudioDecoder`'s v1 mode).

### 15.4 Mapping to existing config knobs

Every column lines up with a default field on
`OmniA2APackingConfigKobaV2`:

| Schema column | Maps to config field | Default value | Match? |
|---|---|---|---|
| `target_audio_bytes` | `target_audio_bytes_key` | `"target_audio_bytes"` | ✓ |
| `target_audio_sr` | `target_audio_sr_key` | `"target_audio_sr"` | ✓ |
| `target_text` | `transcript_key` | `"target_text"` | ✓ |
| `target_duration` | (PR-3 bucketing column) | not yet a config knob | (added later) |
| `ref_audio_bytes` | `ref_audio_bytes_key` | `"ref_audio_bytes"` | ✓ |
| `ref_audio_srs` | `ref_audio_srs_key` | `"ref_audio_srs"` | ✓ |

**PR-1b's dataset factory needs zero column-key overrides** if the
curator uses these names.

### 15.5 Lance path + naming convention

**Path: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/a2a/mls_eng_zs_tts_a2a_v1.lance`**

Path layers (mirrors the existing T2A path
`audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance`):

| Layer | Value |
|---|---|
| Bucket | `ai-lumalabs-datasets-ap-se-2-lance` |
| Modality | `audio` |
| Stage | `pretrain` (room for sibling `sft/`, `eval/`) |
| Task | `a2a` |
| Filename | `mls_eng_zs_tts_a2a_v1.lance` |

Filename tokens, in reading order:

| Token | Meaning |
|---|---|
| `mls_eng` | Source dataset |
| `zs_tts` | Zero-shot text-to-speech (data-side task framing) |
| `a2a` | Audio-to-audio (loader-side / consumer modality) |
| `v1` | Schema version (M=1 only) |

`zs_tts` (with underscore) was picked over `zstts` (run-together) so
the abbreviation is parseable as two distinct concepts (`zs` =
zero-shot; `tts` = text-to-speech). Underscores act as visual word
boundaries; readers don't need to be deep in the TTS literature to
parse `zs_tts`.

### 15.6 PR-1b's `mls_eng_a2a_v1` factory entry

Mirroring T2A's pattern:

```python
def mls_eng_a2a_v1(
    *,
    max_num_tokens: int = 8000,
    batch_size: int = 4,
    **kwargs,
) -> OmniA2APackingConfigKobaV2:
    return OmniA2APackingConfigKobaV2(
        audio_datasets=[
            "s3://ai-lumalabs-datasets-ap-se-2-lance"
            "/audio/pretrain/a2a/mls_eng_zs_tts_a2a_v1.lance",
        ],
        max_num_tokens=max_num_tokens,
        batch_size=batch_size,
        # Default dataset_row_filter is already "M = 1"; tighten with
        # language gate to mirror T2A's pattern.
        dataset_row_filter="`M` = 1 AND `target_language` = 'en'",
        **kwargs,
    )
```

The Lance path is the only thing PR-1b needs from the curator team.
Everything else (column names, SR conventions, `M = 1` invariant) is
already locked in by PR-1a's loader defaults.

### 15.7 Filter expression considerations

Belt-and-suspenders pattern with the curator-side enforcement:

- **`M = 1` filter** — redundant with the curator's "single-purpose
  table" intent, but acts as a defensive check if a M>1 row ever
  slips into a v1-named table by mistake.
- **`target_language = 'en'`** — explicit per-language gate at the
  factory, even when the table is single-language. Matches T2A's
  pattern (`language = 'en'`).
- **No quality column** for v1 (clean academic source).

### 15.8 v2 transition outline

When PR-4 ships, the same physical table can grow without breaking
the v1 loader:

1. Curator regenerates with `M ∈ {1, 2, 3}` rows (lifts the
   M=1-only constraint).
2. Schema gains two columns:
   - `task_template: str` — full curator-rendered instruction
     including the slot pointer (`"...using the voice of speaker-2:
     ..."`).
   - (`target_speaker_local_idx` is already in v1 — no schema change
     needed.)
3. The list-typed `ref_*` columns just grow in length (1 → up to 3).
4. Loader lifts `dataset_row_filter="M = 1"` → `None` (or
   `M IN (1, 2, 3)` for defensiveness).
5. Loader switches `OmniAudioSeqBuilder.handle_a2a` to v2's
   slot-aware variant (see §5.3).

No column rename, no migration, no schema break.

### 15.9 Future-task envelope

The `target_speaker_local_idx: int | None` field opens a third row in
the task table:

| Task | M | `target_speaker_local_idx` value | What the model does |
|---|---|---|---|
| **A2A v1** (PR-1b) | 1 | always `0` | Speak target text in ref's voice |
| **A2A v2** (PR-4, with distractors) | 1, 2, 3 | uniformly in `[0, M-1]` | Speak target text in `ref[idx]`'s voice; refs at other indices are distractors |
| **Cross-speaker style transfer** (future) | 1, 2, 3 | `None` | Speak target text in *some other* voice — refs are style references, not voice references |

If/when the third row becomes a real task, the schema is ready: just
curate rows with `target_speaker_local_idx = None` and ref clips that
demonstrate the desired *style* without sharing the target speaker's
identity.

### 15.10 Open questions for the curator team

1. **Single-purpose vs unified tables**: do we want one
   `mls_eng_zs_tts_a2a_v1.lance` for v1-only-data, then a separate
   v2 table later? Or one growing table that the loader filters? Both
   work; per-table per-version is simpler operationally.
2. **`source_dataset` column**: useful for multi-source mixing (mls_eng
   + emilia_yodas) within one table, but redundant if going
   per-source-table. Curator's choice.
3. **Embedding the same-speaker invariant**: should the curator also
   write a `same_speaker_ref_idx == 0` column or rely on
   `target_speaker_global_id == ref_speaker_global_ids[0]` SQL? The
   nullable-int field already encodes this.

### 15.11 PR-1b scope summary

What lands in PR-1b:

- `mls_eng_a2a_v1` factory in `projects/kuma/.../configs/datasets.py`.
- Schema-column verification: dump `lance.dataset.schema` from the
  curator's table and pin the config knobs against the verified
  names. Loader expectations match curator output.
- `debug_local_a2a` config (real Lance, 2-layer model). Mirrors
  PR-1a's `debug_local_a2a_dummy` but swaps the dummy back to the
  real packing config + Lance path.
- Smoke test: ~5 min run through the real-Lance loader, single
  multi-audio decode, ref truncation, end-to-end forward pass on a
  small batch.

Out of scope for PR-1b (deferred to PR-2):

- Production-scale (0.6B / 2B) configs.
- CFG dropout flip from `[]` to `["a2a"]`.
- Multi-node launch.

---

## 16. PR-1a follow-up: cleanups + schema rename + naming alignment

Two follow-up commits landed on `dongguo/omni-a2a-plumbing` after
PR-1a's body to (a) act on audit findings from §14's review,
(b) align the loader column-name defaults with the curator schema's
locked convention (`a2a_curator_design.md` §3.4 + §A.5), and
(c) trim PR-1a's design-discussion verbosity from docstrings to
formal-code concision.

PR draft on GitHub: <https://github.com/lumalabs/lumaverse/pull/8113>.

### 16.1 Branch + commit summary

| Commit | Lines | Scope |
|---|---|---|
| `276cffcf56` | +50 / −25 | Audit cleanups (§14 findings B1, B2, R1) |
| `ddd127580b` | +626 / −1117 | Schema rename, SR-assert removal, T2A naming alignment, `A2APipelineParams` promotion, comment trim |

19 files touched in commit 2; net ~500 lines removed (the comment
trim dominates).

### 16.2 Audit-driven cleanups (commit `276cffcf56`)

Three small fixes, no behavior change:

- **Drop redundant tensor-key overrides** in
  `OmniA2APackingConfigKobaV2._build_dataset_configs`. Two override
  lines duplicated `configo.merge_into_copy`'s automatic same-name
  field propagation; only `audio_sample_rate → sample_rate` is a
  real rename and needs explicit assignment.
- **Fix misleading test docstring** in `a2a_test.py`:
  `test_dummy_dataset_a2a_specific_defaults` claimed
  `ref_window_sec` was "A2A-only (not on the source config)", but
  both configs define the field with the same default. Renamed to
  `test_dummy_dataset_ref_window_default` with an honest docstring
  describing it as a value-pinning regression guard.
- **Fix misleading two-step-swap comment** in `a2a.py`. The prior
  comment claimed the swap exercised the kuma → koba field-bridge
  "during config construction", but that bridge runs in
  `_build_dataset_configs()` (called from `get_loader()`), not
  `__init__()`.
- **Strengthen CFG silent-no-op warning** in `default_a2a.py`.
  `OmniCFGDropoutLast._forward` hardcodes
  `elem.type == SequenceType.NOISY_VAE_IMAGE`; flipping
  `cfg_dropout_modalities=["a2a"]` would be a silent no-op (no
  error, no log, no CFG). Comment now spells out the failure mode
  and the gating fix.

### 16.3 Schema rename to align with curator convention (commit `ddd127580b`)

The curator schema design (`a2a_curator_design.md` §3.3 + §3.4)
locked a "central audio gets bare names, refs get `ref_` prefix"
convention. This commit propagates that convention into the
loader's Config field defaults.

| old loader default | new loader default |
|---|---|
| `target_audio_bytes` | `audio_bytes` |
| `target_audio_sr` + `ref_audio_srs` | (dropped — see §16.4) |
| `target_text` | `transcript` |
| `target_audio_tensor` | `audio_tensor` (intermediate decoder output) |
| SQL filter `M = 1` | `num_ref_audios = 1` |

Touches `MultiAudioDecoder.Config`, `OmniAudioSeqBuilder.Config`,
`A2APipelineParams`, `OmniA2APackingConfigKobaV2`, plus the
field-bridge tests.

### 16.4 SR-assert removal

Drops `audio_sr_key` + `expected_source_sr` Config fields and the
SR-assert block in `MultiAudioDecoder.forward`. Reasoning (full
rationale in `a2a_curator_design.md` §A.10.1):

- The assert reads the *column value* (curator's claim about source
  SR), not the actual file content. If a row claims
  `audio_sr=48000` but the bytes were 24 kHz, the assert passes —
  the wrong direction for what it appeared to guard against.
- torchcodec's `AudioDecoder(..., sample_rate=target)` resamples
  whatever rate the bytes carry to `target`; same-rate input is a
  passthrough no-op. The decode path is correct on its own.

The curator may keep `audio_sr` as `aux_metadata_json` provenance;
the loader does not read it.

### 16.5 `speech_duration` vs `audio_length` naming

A2A adopts a two-column convention: `speech_duration`
(speech-content length) + `audio_length` (post-decode container
length). Distinct whenever the source has leading/trailing silence
or when the loader silence-pads to a target shape (PR-3 bucketing
case). For mls_eng (edge-trimmed) the values coincide.

`MultiAudioDecoder` writes `audio_length_seconds` (was
`audio_duration_seconds`) as a metadata side-effect.

T2A's `T2APipelineParams.duration_key="duration"` is **kept
unchanged** — existing Emilia / podcast / WavCaps / Jamendo tables
use the bare `duration` column. A `TODO(naming-cleanup)` comment
in the docstring tracks the future migration to the
`speech_duration` / `audio_length` two-column convention when those
T2A tables are re-curated. Full rationale:
`a2a_curator_design.md` §A.10.2 + §A.10.3.

### 16.6 T2A naming alignment

To converge T2A and A2A vocabulary on the koba-v2 distinction
(`audio_key` reserved for standalone-file paths,
`audio_bytes_key` for inline-bytes columns — see
`koba/v2/config.py:199` vs `:320`):

- `T2APipelineParams.audio_key` → `audio_bytes_key`.
- `OmniT2APackingConfigKobaV2.audio_key` → `audio_bytes_key`.
- Same rename on `OmniT2ABucketedPackingConfigKobaV2`.
- `configs/datasets.py` (4 factory call sites) updated.
- `tests/test_t2a_data_roundtrip.py` updated.

The legacy `AudioDecoder.Config.audio_key` field is **left
unchanged** (would touch docs + research configs); the T2A factory
bridges from new `audio_bytes_key` to legacy `audio_key=` kwarg in
one line.

### 16.7 Inner-class rename + `A2APipelineParams` promotion

`OmniT2APackingDatasetKobaV2` → `OmniAudioPackingDatasetKobaV2`.
The inner IterableDataset's body is task-agnostic; A2A already
reused it. `dp_sharding_test.py`'s monkeypatch target updated.

`A2APipelineParams` moved from `default_a2a.py` to `pipelines.py`
(next to `T2APipelineParams`). `default_a2a.py` imports from there.
Same back-compat as T2A: `from koba.pipelines.default_a2a import
A2APipelineParams` still works because Python module-level names
are visible after import (no `__all__` needed).

### 16.8 Comment / docstring trim

Removed dev-note prose carried over from PR-1a's design phase:
§-references to `audio_extension_a2a_logs.md` and
`a2a_curator_design.md`; "v1 default" / "v2 will switch"
versioning narrative; multi-paragraph rationale; migration
history; process meta-commentary.

Preserved:

- Class/function purpose (1-3 sentences).
- Input/output keys + types.
- The `speech_duration` vs `audio_length` distinction (concise).
- Channel-dim convention (one line).
- Failure modes ("returns None on …", "raises on …").
- Test intent (one line per test).

The 18-line "silent no-op trap" warning on `OmniCFGDropoutLast`
was trimmed to 7 lines, preserving the actionable instruction
(extend `OmniCFGDropoutLast` to handle `NOISY_VAE_AUDIO` before
flipping `cfg_dropout_modalities=["a2a"]`) and dropping the
design-doc narrative.

### 16.9 Test counts

110 (PR-1a body) → 115 (PR-1a + follow-up) tests passing across
the A2A surface; 52 collateral T2A-side tests still pass with no
regressions. The +5 delta is from renaming/restructuring existing
assertions, not new test logic.

---

## Status

- **Design**: locked through v1; v2 + future-task envelopes
  documented but not implemented.
- **PR-1a** (Phase 1, plumbing on dummy data): ✓ **landed**
  2026-05-01 → 02 on `dongguo/omni-a2a-plumbing`. 9 commits
  (7-commit body + 2 follow-up commits), 115 tests, ~3 K LOC.
  Smoke-test entry point: `debug_local_a2a_dummy`. PR draft
  awaiting GPU smoke run before un-drafting:
  <https://github.com/lumalabs/lumaverse/pull/8113>. See §14 for
  the body log, §16 for the follow-up log.
- **PR-1b** (Phase 2, real Lance + tiny model): scope locked,
  schema designed (§15). Curator-side schema also aligned with the
  `speech_duration` / `audio_length` two-column convention
  (`a2a_curator_design.md` §A.10.2). Awaiting curator's first
  `mls_eng_zs_tts_a2a_v1.lance` table.
- **PR-2** (Phase 3, production-scale): planned. Mirrors `exp001a` /
  `exp001c` for 0.6B / 2B configs. Includes CFG dropout flip-on,
  gated on `OmniCFGDropoutLast` learning to handle
  `NOISY_VAE_AUDIO` (see §16.2 / the warning comment in
  `default_a2a.py`).
- **PR-3** (Phase 4, bucketing): planned. Adds `audio_length`
  bucketing + two-shape-class VAE batching extension in the
  trainer. T2A's `duration_key="duration"` migrates to
  `speech_duration` here too (TODO marker in `T2APipelineParams`).
- **PR-4** (Phase 5, M>1 v2): planned. List-aware decoder /
  truncator, slot-tag seq builder, lift `num_ref_audios = 1`
  filter.
- **Open in v1**: `a2a_task_prompt` wording (§9.1), GPU smoke run
  to convert PR #8113 from draft.
- **Open in v2**: CFG dropout behavior under distractors, possible
  custom slot-aware variant (§5.4).
- **Mental model anchors**: §12 (static/dynamic + prosody decorrelation),
  §13 (channel-dim convention, pre-VAE only). The augmentation ideas
  considered and deferred (§12.8) plus the implementation-time
  decisions (§14.3, §16.3-§16.7) are documented with rationale so
  they aren't re-litigated each iteration.

---

## Session checkpoint — 2026-05-02

End-of-session state on `dongguo/omni-a2a-plumbing`. Captured so
the next session can resume cleanly without re-deriving where
things stopped. Mirrors the pattern in
`audio_extension_v2_logs.md`'s "Session checkpoint — 2026-05-01".

### Branch + PR state

- **Branch**: `dongguo/omni-a2a-plumbing` (9 commits ahead of
  `origin/main`); pushed.
- **PR**: <https://github.com/lumalabs/lumaverse/pull/8113> —
  draft.

### What's runnable

- 115 unit tests pass (58 koba A2A + 52 kuma A2A + 5 cross-cutting)
  via `pytest` from the kuma venv. `dp_sharding_test` plus four
  collateral T2A-side test files also pass (52 tests, no
  regressions).
- `debug_local_a2a_dummy` builds without error (10 a2a_test smoke
  tests cover config-construction).

### What's not yet exercised

- **`torchrun` smoke run on a real GPU.** The
  `DummyA2ADataset → OmniAudioTrainer → forward + backward +
  optimizer step` path is unverified end-to-end. The packed-batch
  schema didn't change in the follow-up (trainer reads `txt`,
  `vae_token_mask`, `x_vae`, etc. — all unchanged), so I don't
  expect failure, but it's untried.

### Deferred items (not blockers for PR-1a)

- **T2A `duration_key="duration"` migration** to A2A's
  `speech_duration` / `audio_length` two-column convention.
  Tracked by a `TODO(naming-cleanup)` in `T2APipelineParams`'s
  docstring; will land when existing T2A tables (Emilia / podcast
  / WavCaps / Jamendo) are re-curated to write the new column
  names. PR-3 (bucketing) is the natural landing place — that's
  when `audio_length` becomes load-bearing for the loader.
- **CFG dropout flip for A2A** (`cfg_dropout_modalities=["a2a"]`).
  Gated on extending `OmniCFGDropoutLast` to recognize
  `NOISY_VAE_AUDIO` (today it hardcodes `NOISY_VAE_IMAGE` and
  would silently no-op on A2A's seq plan). Warning comment in
  `default_a2a.py` flags the trap.
- **PR-1b** (real-Lance debug variant) — awaiting the curator's
  first `mls_eng_zs_tts_a2a_v1.lance` table; loader-side defaults
  already align (no further code change needed once the table
  lands).

### How to resume

1. `git checkout dongguo/omni-a2a-plumbing` (already up to date
   with `origin`).
2. Run the GPU smoke test:
   ```bash
   cd /fsx/dongguo/Projects/lumaverse/projects/kuma
   source .venv/bin/activate
   CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 \
       main.py \
       --config kuma.projects.omni.audio.configs.tasks.a2a.debug_local_a2a_dummy \
       --name debug_omni_a2a_dummy_smoke
   ```
3. If 5-10 steps complete with declining (or at least non-NaN)
   loss, mark the test-plan checkbox on PR #8113 and run
   `gh pr ready 8113` to convert from draft.
4. If the run fails (e.g., on an unexpected key in the trainer's
   packed-batch reader), the fix is likely one-line + a missing
   test; land it as commit 3 on the same branch.

### Untracked working-tree files (deliberately not committed)

These were in the working tree at session start; intentionally
left out of PR #8113:

- `.claude/settings.json` — local Claude Code permissions config.
- `projects/kuma/kuma/projects/omni/audio/inference/` — pycache
  only.
- `projects/kuma/noise_distributions.png` — debug-time plot
  artifact.
- `projects/kuma/wandb-metadata.json` — wandb run output.
