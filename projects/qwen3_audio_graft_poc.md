# Qwen3-VL-2B + Qwen3-ASR-1.7B Audio Graft — Zero-Shot ASR POC

**Date:** 2026-04-28
**Author:** Dong Guo
**Status:** POC complete; zero-shot ASR working at 0% WER on a 4-sample TTS test set.

## Motivation

We want a small **dense** Qwen3-family model that accepts audio input. The Qwen3-Omni release only ships at 30B-A3B (MoE), which is too large for our research budget. The cleanest dimensional match within the Qwen3 family is **Qwen3-ASR-1.7B** (the audio encoder source) onto **Qwen3-VL-2B-Instruct** (the LM/vision base):

| | Qwen3-VL-2B-Instruct | Qwen3-ASR-1.7B |
| --- | --- | --- |
| `hidden_size` | 2048 | 2048 |
| `num_hidden_layers` | 28 | 28 |
| `num_attention_heads` | 16 | 16 |
| `num_key_value_heads` | 8 | 8 |
| `head_dim` | 128 | 128 |
| `vocab_size` | 151,936 | 151,936 |
| `mrope_section` | `[24, 20, 20]` | `[24, 20, 20]` |
| `rope_theta` | **5,000,000** | **1,000,000** |
| `max_position_embeddings` | 262,144 | 65,536 |
| Audio encoder `output_dim` | — | **2048** (matches LM hidden) |

The `output_dim=2048` of ASR-1.7B's audio encoder already matches VL-2B's LM hidden size, so **no separately-trained projector is needed** — the encoder's bundled proj1/proj2 layers do the lift internally. The two models share most architectural fields, with two known divergences (`rope_theta`, `max_position_embeddings`) that we noted but did not adjust.

A separate byte-similarity check on `embed_tokens.weight` was done before the merge: across the 151,643 base-vocab rows, **0 / 151,936 rows are byte-identical**, but mean cosine similarity is **0.9654** (median 0.9714). This is consistent with a **shared Qwen3-2B base, separately fine-tuned** for vision and ASR — they share an ancestor but neither is a snapshot of the same fine-tune.

## Part 1 — Integration Design

The deliverable is a single HF-loadable checkpoint directory `Qwen3-VL-2B-Audio-POC/` that contains the full Qwen3-VL-2B-Instruct model plus the ASR-1.7B audio encoder, loadable as VL-2B for the text/vision path with a separate inference-time wiring for the audio encoder.

### 1.1 Audio Encoder Attachment

**What "audio encoder" means here.** The ASR-1.7B checkpoint exposes 397 tensors under the `thinker.audio_tower.*` prefix:

| Component | Tensor count | Purpose |
| --- | ---:| --- |
| `conv2d{1,2,3}` (3 stride-2 convs) | 6 | Mel-spectrogram downsampling stem |
| `layers.{0..23}.*` (24 transformer layers) | 384 | Encoder body |
| `ln_post` | 2 | Final layer norm |
| `conv_out` | 1 | Length-axis output conv |
| `proj1`, `proj2` | 4 | **Adapter** — lifts encoder hidden (`d_model=1024`) to LM hidden (`output_dim=2048`) |

The "adapter" the user originally asked about is the `proj1` + `proj2` pair. There is no Q-Former, no Perceiver, no Linear bridge external to this — the encoder ships with the projection bundled.

**Renaming for the merged checkpoint.** ASR-1.7B's keys live under `thinker.audio_tower.*` because that model has a Thinker–Talker structure. VL-2B uses a different prefix scheme (`model.language_model.*`, `model.visual.*`). To match VL-2B's convention, we remap on write:

```text
thinker.audio_tower.*   →   model.audio_tower.*
```

This places `model.audio_tower.*` parallel to `model.visual.*` in the merged tensor namespace, matching VL-2B's nesting.

**Loading at inference.** The vanilla `Qwen3VLForConditionalGeneration` class does not know about audio. When it loads our merged checkpoint:

- All 625 VL-2B keys load cleanly into the LM + vision tower.
- The 397 `model.audio_tower.*` keys produce HF "unused weights" warnings.
- We instantiate the audio tower **separately** as a vendored `Qwen3ASREncoder` (see §3) and load just those 397 keys (with the prefix stripped) into it.

This avoids needing to register a custom HF model class for the POC.

### 1.2 Special Token Embedding Rows in VL-2B's Vocab

The audio encoder produces feature vectors at the LM's hidden size, but those vectors only enter the LM via the embedding-row mechanism: the chat template inserts `<|audio_pad|>` placeholder tokens, and at inference we **overwrite the corresponding rows of `inputs_embeds`** with the audio encoder's output. For this to make semantic sense, the audio-related token embeddings must come from a model that actually trained those rows on audio.

**The key empirical fact: ID alignment.** Both checkpoints declare `vocab_size=151936`. VL-2B's tokenizer registers 26 `added_tokens` at IDs 151643–151668 (text/vision/tool/think). **IDs 151669–151935 are unused empty slots** in VL-2B's tokenizer — the embedding tensor has rows there (training side-effects), but no string maps to them. ASR-1.7B uses **36 of those slots** (151669–151704):

```text
151669  <|audio_start|>     # frame marker (used by inference)
151670  <|audio_end|>       # frame marker (used by inference)
151671  <tts_pad>           # ASR-only, unused at inference
151672  <tts_text_bos>      # ASR-only, unused at inference
151673  <tts_text_eod>      # ASR-only, unused at inference
151674  <tts_text_bos_single># ASR-only, unused at inference
151675  <non_speech>        # ASR-only, unused at inference
151676  <|audio_pad|>       # frame placeholder (used by inference; spliced)
151677–151703  <blank1..27> # CTC blank slots, unused at inference
151704  <asr_text>          # ASR-only, unused at inference
```

**There are zero ID collisions** with anything VL-2B has trained on (IDs ≤ 151668 are bit-identical between the two tokenizers; ASR's 36 extras land in VL-2B's empty zone).

**Embedding row patch.** We overwrite VL-2B's `model.language_model.embed_tokens.weight` rows `[151669, 151705)` (inclusive of 151704) with the corresponding rows from ASR-1.7B's `thinker.model.embed_tokens.weight`. All other 151,900 rows are untouched VL-2B values.

```python
LO, HI = 151669, 151705       # half-open: 36 rows
new_embed = vl_embed.clone()
new_embed[LO:HI] = asr_embed[LO:HI].to(new_embed.dtype)
# Verified post-write:
#   rows [0, 151669) byte-identical to VL-2B
#   rows [151705, 151936) byte-identical to VL-2B
#   rows [151669, 151705) byte-identical to ASR-1.7B
```

**Why not transplant more rows?** We considered but rejected copying ASR-1.7B's full embedding table. The cosine drift across body rows (mean 0.965, max difference rel-L2 1.33) is large enough that overwriting VL-2B's text-token rows would corrupt the text/vision alignment that VL-2B was instruction-tuned for. The 36 audio-special rows are the **only** rows where ASR-1.7B has trained values and VL-2B has empty slots — copying them is risk-free; copying anything else is not.

### 1.3 Tokenizer Construction

The **ASR-1.7B repo does not ship a `tokenizer.json`** — it has only `vocab.json` + `merges.txt` + `tokenizer_config.json` (the slow Python tokenizer files). VL-2B ships a full 7 MB Rust-backed `tokenizer.json` (fast tokenizer) plus a vision-aware chat template. We want both, so we **start from VL-2B's tokenizer files** and merge in only the 36 ASR specials:

| File | Source | Modification |
| --- | --- | --- |
| `tokenizer.json` | VL-2B (fast Rust tokenizer) | Append 36 entries to `added_tokens` array |
| `tokenizer_config.json` | VL-2B | Append 36 entries to `added_tokens_decoder`; copy `audio_bos_token`/`audio_eos_token`/`audio_token`/`extra_special_tokens` metadata from ASR-1.7B |
| `vocab.json`, `merges.txt` | VL-2B | Unchanged (ASR's are 14 bytes different — trailing whitespace; functionally identical) |
| `chat_template.json` | VL-2B + injected audio branch | See below |
| `preprocessor_config.json` | VL-2B | Image preprocessor — unchanged |
| `audio_preprocessor_config.json` | ASR-1.7B's `preprocessor_config.json` | New file (Whisper feature extractor: 128 mel bins, 16 kHz, 30 s window, hop 160) |

**Chat template patch.** VL-2B's template renders user-message content items by branching on `image` / `video` / `text`. We add an `audio` branch alongside, mirroring ASR-1.7B's audio block:

```jinja
{%- elif content.type == 'audio' or 'audio' in content or 'audio_url' in content %}
    {%- set audio_count.value = audio_count.value + 1 %}
    {%- if add_audio_id %}Audio {{ audio_count.value }}: {% endif -%}
    <|audio_start|><|audio_pad|><|audio_end|>
```

We also add an `audio_count = namespace(value=0)` declaration alongside `image_count` / `video_count` at the top of the template. The template is duplicated into `tokenizer_config.json` (loaders prefer one over the other depending on version, so syncing both avoids a stale-template bug we initially hit).

**Verified:** the merged tokenizer round-trips all 62 added tokens, encodes the audio specials as single tokens (`<|audio_pad|>` → `[151676]`, etc.), and renders a mixed audio+image+text user message with the expected three special-token blocks in the right positions. `tokenizer.is_fast == True`.

### 1.4 Sidecar config

The merged `config.json` is VL-2B's config plus:

- `audio_config`: a cleaned `Qwen3ASREncoderConfig` block (encoder architecture: `encoder_layers=24`, `d_model=1024`, `output_dim=2048`, `num_mel_bins=128`, `n_window=50`, `n_window_infer=800`, etc.)
- `audio_token_id=151676`, `audio_start_token_id=151669`, `audio_end_token_id=151670`
- `_provenance`: a custom block recording the base model, the audio source, the key remap, and the embedding row patch range. This field is ignored by HF but documents the recipe in-band.

The top-level `architectures` and `model_type` fields stay as VL-2B's so `AutoModel` / `AutoConfig` resolve to the existing `Qwen3VLForConditionalGeneration` class without a custom modeling file.

### 1.5 Final merged checkpoint

```text
/fsx/dongguo/adhoc/ckpts/Qwen3-VL-2B-Audio-POC/  (4.90 GB total)
├── model.safetensors                     4.89 GB   1022 keys (VL-2B 625 + audio_tower 397)
├── config.json                            2.8 KB   VL-2B base + audio_config + 3 audio token IDs
├── tokenizer.json                         5.2 MB   VL-2B fast tokenizer + 36 ASR specials
├── tokenizer_config.json                   18 KB   matching added_tokens_decoder + audio metadata
├── chat_template.json                     6.2 KB   VL-2B template + audio branch
├── audio_preprocessor_config.json          330 B   Whisper feature extractor settings
├── preprocessor_config.json                       VL-2B image preprocessor (unchanged)
├── video_preprocessor_config.json                 VL-2B (unchanged)
├── vocab.json, merges.txt, generation_config.json
└── MERGE_NOTES.md                         4.3 KB   Provenance + caveats
```

Sanity-checked end-to-end:

- `AutoConfig` exposes `audio_config.encoder_layers=24, output_dim=2048` and the three audio token IDs.
- `AutoTokenizer` returns `Qwen2TokenizerFast`, `len(tokenizer)=151705`.
- `Qwen3VLForConditionalGeneration.from_pretrained` loads with the expected `model.audio_tower.*` "unused weights" warning (397 tensors); LM forward on a sequence containing `<|audio_pad|>` produces finite logits.
- Embedding rows `[151669:151705)` byte-match ASR-1.7B post-load.

---

## Part 2 — Zero-Shot ASR Test

### 2.1 Methodology

**Goal.** Probe whether the LM can interpret features produced by a foreign audio encoder *without any audio training* — i.e., does the proj1/proj2 adapter happen to land in a token-embedding space VL-2B already understands?

**Test data.** Four short English clips generated by an internal **omni-T2A model** (`/fsx/dongguo/adhoc/omni-t2a-eval/smoke_v2_step100000/omni_t2a_kq81ynyc_smoke_v2_step100000/{0000..0003}.wav`). Each is 5 s at 16 kHz with the speech occupying ~4.6 s (fixed `expected_speech_duration_sec`). The accompanying JSON records the original prompt text passed to the TTS model — that text is the **ground truth** for the round-trip test.

**Round-trip evaluation.** TTS gave the model the *prompt text* and produced *speech*; we now feed that *speech* to our merged Qwen3-VL-2B-Audio-POC and ask it to write down what was said. If the system works, the predicted text should match the original TTS prompt.

**Important caveat: punctuation is out of scope.** The audio renders only spoken words; commas, periods, and capitalization are absent from the acoustic signal. We strip the TTS prompt wrapper (`[SPEAKER_00]"..."`) before scoring, lowercase both sides, and remove non-alphanumeric characters except apostrophes — i.e., we score on **spoken-word identity**, not orthographic match. This is the standard convention for ASR evaluation.

**Inference recipe (`run_zero_shot_audio.py`):**

1. Load wav → `WhisperFeatureExtractor` (128 mel, 16 kHz, hop 160, 30 s right-padded) → input features `(1, 128, 3000)` + frame mask `(1, 3000)`.
2. Audio encoder forward → `(1, 390, 2048)` features. (3000 mel frames / chunk_len 100 = 30 chunks; 13 output tokens per chunk; 30 × 13 = 390.)
3. Compute the valid-prefix length from the frame mask using `_post_cnn_length` (3 stride-2 conv stages: `lengths → (lengths-1)//2+1` three times). For our ~4.6 s clips: 460 mel frames → 4 full chunks (13 tokens each) + 1 partial chunk (8 tokens) = **65 valid audio tokens** out of 390. Trim to those 65.
4. Render the chat template with one audio + one text content item:
   ```text
   <|im_start|>user
   <|audio_start|><|audio_pad|><|audio_end|>The audio above contains a person speaking English. Please write down the exact words that were spoken.<|im_end|>
   <|im_start|>assistant
   ```
5. Locate the single `<|audio_pad|>` (id 151676) in the rendered IDs and **expand it to 65 copies** in-place (one ID per valid audio token).
6. Build `inputs_embeds` by embedding the expanded IDs through `model.get_input_embeddings()`, then **overwrite the rows at `<|audio_pad|>` positions** with the trimmed audio encoder output.
7. Greedy generate with `model.generate(inputs_embeds=..., max_new_tokens=96, do_sample=False)`.

### 2.2 Results

| Sample | Words | Edits | WER | Pred (normalized) vs Ref (normalized) |
| --- | ---: | ---: | ---: | --- |
| 0000 | 15 | 0 | 0.00% | `good morning today's forecast calls for clear skies and a gentle breeze across the city` |
| 0001 | 20 | 0 | 0.00% | `once upon a time in a quiet village by the sea a young girl dreamed of flying among the clouds` |
| 0002 | 13 | 0 | 0.00% | `please press one to continue in english or press two for more options` |
| 0003 | 13 | 0 | 0.00% | `the quantum realm challenges our understanding of scale causality and even time itself` |
| **Total** | **61** | **0** | **0.00%** | — |

**Aggregate WER (normalized for spoken-word identity): 0.00% (0 errors / 61 reference words)** across all four clips.

Inference latency on H100: 1.9–4.3 s end-to-end per 5 s clip (first sample is slower due to model warmup).

**Raw output of sample 0000** (illustrative — note the spontaneous instruction-following format the model added):

```
The exact words spoken in the audio are:

"Good morning! Today's forecast calls for clear skies and a gentle breeze across the city."
```

The other three return the transcript directly, without the explanatory preamble. None of the four are instruction-tuned for ASR — the format variation is incidental.

### 2.3 Analysis — what this tells us

This result strongly supports the hypotheses laid out during the survey / byte-similarity / token-overlap stages:

1. **Qwen3-ASR-1.7B and Qwen3-VL-2B-Instruct share a common Qwen3-2B base.** This is consistent with the byte-similarity finding (embed body cosine ~0.965 across 151,643 base-vocab rows, but 0 / 151,936 rows byte-identical) — common ancestor, separate fine-tunes.
2. **The audio encoder's adapter (proj1/proj2) projects audio features into the *shared base's* text-token semantic space**, not into ASR-1.7B's fine-tune-specific space. If it had been the latter, feeding the same features into VL-2B's LM (a sibling fine-tune) would produce gibberish; instead it produces verbatim transcription. The adapter is base-aligned, not fine-tune-aligned.
3. **The RoPE-base mismatch (VL-2B θ=5M vs ASR-1.7B θ=1M) produces no visible degradation** in this short-audio regime. The audio tokens splice into a sequence ≤ 100 positions long, well inside the regime where both RoPE schedules behave nearly identically. Long-audio or long-context behaviour was not tested.
4. **VL-2B's instruction tuning + chat template absorb the audio segment naturally.** Sample 0000 spontaneously emits *"The exact words spoken in the audio are: ..."* — that is the model treating the audio block as another modality of user input and answering the text instruction. No special prompting tricks were needed; the standard chat template path with a new audio branch was sufficient.

These four points together amount to a working **architectural recipe** for grafting Qwen3-family audio encoders onto Qwen3-family LMs of compatible hidden size, *with no training*, when both endpoints derive from the same base.

### 2.4 Limitations (this POC, not the broader recipe)

- **N=4, English only, ~5 s clips.** No noise, no accents, no cross-talk, no long-form, no multi-language. The 0% WER says the pipeline plumbing is correct and the architectural transfer works under easy conditions; it does not characterize the model's robustness.
- **TTS-generated speech is in-distribution for ASR-1.7B's training data** (clean studio-quality English with neutral prosody). Real-world audio will be harder.
- **Punctuation is not in the spoken signal** and we score after stripping it. A real ASR product would need a separate punctuation-restoration stage if punctuation matters downstream.
- **The audio encoder still does ASR-1.7B's full pipeline including its 30-second padding** — only ~17% of the encoder's capacity (65 / 390 output tokens) carries information for a 5 s clip. There is no efficiency win here over standalone Qwen3-ASR-1.7B.
- **No comparison against a baseline.** We did not run Qwen3-ASR-1.7B end-to-end on the same 4 clips to see if the graft preserves accuracy or degrades it (presumably it can't *exceed* it).

### 2.5 What this is *not* a claim about

This experiment shows that **zero-shot ASR transcription** works at near-trivial difficulty. It does **not** demonstrate that:

- The model can answer non-transcription questions about audio (e.g., "what emotion is the speaker conveying?", "is there music in the background?", "how many speakers?")
- Audio + image + text mixed reasoning works
- Long-form audio or multi-segment audio works
- Any audio-output capability exists (this is input-only)

Those would require additional probing and are out of scope here.

---

## Part 3 — Implementation Notes

### 3.1 Vendored Qwen3-ASR modeling code

`Qwen3-ASR` is **not yet merged** into HuggingFace transformers main branch (open PR [#43838](https://github.com/huggingface/transformers/pull/43838)). The Qwen3-ASR-1.7B HF repo also does not bundle `modeling_*.py` for `trust_remote_code` loading. We therefore vendor the modeling code from the open PR, pinned to a specific commit:

- Source: `mbtariq82/transformers`, branch `qwen3-asr`, commit `0b932ecb3e09c6efa1f0a96c6621bf77be23a08d`
- Vendored at: `/fsx/dongguo/adhoc/omni-t2a/poc/qwen3_asr_vendor/` (4 files, ~45 KB)
- Compatibility: transformers 4.57.1 lacks several utilities the PR uses. A small `_compat.py` shim provides no-op decorators (`auto_docstring`, `capture_outputs`, `merge_with_config_defaults`), a minimal `create_bidirectional_mask`, and a `torch.nn.init` re-export with `copy_` added.
- Two inline patches to vendored code (both annotated with `# NOTE (vendor patch):` comments):
  1. `Qwen3ASRAttention.k_proj` switched from `bias=False` to `bias=bias`. The PR's spec says `bias=False`, but the released ASR-1.7B checkpoint contains 24 trained `k_proj.bias` tensors (one per layer). Loading without this patch silently drops them.
  2. `ALL_ATTENTION_FUNCTIONS.get_interface(...)` rewritten to `ALL_ATTENTION_FUNCTIONS.get(...)`. 4.57.1's `AttentionInterface` is dict-like with a `.get()` API; `.get_interface()` is newer.

The vendor follows the existing repo precedent at `lib/ursa/ursa/models/omni/tokenizers/third_party/{siglip2,qwen_vit}/` (full upstream copies, kept intact, opt-in vendoring). When the PR merges and a release ships, the vendor directory can be deleted and `from transformers.models.qwen3_asr import ...` will work as a drop-in replacement.

### 3.2 File index

| Path | Role |
| --- | --- |
| `/fsx/dongguo/adhoc/ckpts/Qwen3-VL-2B-Instruct/` | Original VL-2B checkpoint (3.2 GB) |
| `/fsx/dongguo/adhoc/ckpts/Qwen3-ASR-1.7B/` | Original ASR-1.7B checkpoint (3.7 GB) |
| `/fsx/dongguo/adhoc/ckpts/Qwen3-VL-2B-Audio-POC/` | **Merged POC checkpoint** (4.90 GB) |
| `/fsx/dongguo/adhoc/omni-t2a/poc/qwen3_asr_vendor/` | Vendored Qwen3-ASR modeling code with 4.57.1 compat shims |
| `/fsx/dongguo/adhoc/omni-t2a/poc/run_zero_shot_audio.py` | Zero-shot inference script (used to produce §2.2 results) |
| `/fsx/dongguo/adhoc/omni-t2a-eval/smoke_v2_step100000/omni_t2a_kq81ynyc_smoke_v2_step100000/` | Test wavs + per-sample JSON ground truth |

## Part 4 — Position-ID Technical Details

The graft mixes weights from two separately-tuned models. Position encoding is the most prominent axis along which their training-time conventions diverge, so it is worth pinning down what each side does, what numerically differs, and why none of it bit us in the §2.2 results.

### 4.1 Two unrelated position encoding schemes are at play

There are **two** position-encoding mechanisms in the merged model, and they are independent of each other:

1. **Audio encoder side — absolute sinusoidal embeddings.** Inside `Qwen3ASREncoder`, post-CNN audio frames receive a Whisper-style `SinusoidsPositionEmbedding` added once at the encoder input. No RoPE is involved anywhere in the audio tower. This means the audio encoder's positional behavior is governed entirely by parameters in `audio_config`, and is unaffected by anything on the LM side.
2. **LM side — M-RoPE applied to Q/K at every attention layer.** The LM (text + visual + audio_pad tokens, all interleaved into one sequence) uses Qwen3's multimodal RoPE: the head-dim-128 rotation is partitioned across multiple position axes, and `rope_theta` plus `rope_scaling` control its frequency schedule and modality partitioning. This is the side where VL-2B and ASR-1.7B differ.

The audio encoder's output features become **values** in the LM's sequence (via the `<|audio_pad|>` row substitution), and acquire **positions** from the LM's RoPE schedule applied at the audio_pad slots. The audio tower's own positional embedding never crosses into the LM; the LM's RoPE never touches the audio encoder. They are concatenated in series, not blended.

### 4.2 Audio encoder positional encoding

The relevant `audio_config` fields:

| Field | Value | Meaning |
| --- | ---: | --- |
| `num_mel_bins` | 128 | Input feature size |
| `max_source_positions` | 1500 | Length of the absolute sinusoidal table |
| `n_window` | 50 | Half mel-frame chunk size (chunk_len = 2 × n_window = 100) |
| `n_window_infer` | 800 | Attention-window span in mel frames at inference |
| `conv_chunksize` | 500 | Conv-pipeline chunking |

The 1500 entries in `max_source_positions` correspond to post-CNN time steps. With three stride-2 convs (each `(L−1)//2 + 1`), 100 mel frames per chunk reduce to 13 post-CNN tokens; over a 30-second window (3000 mel frames = 30 chunks) you get 30 × 13 = 390 audio output tokens. 1500 leaves plenty of slack for longer clips. The sinusoidal table is precomputed at module construction and added once before the encoder layers.

Because this is absolute sinusoidal (Whisper convention), there is no learnable parameter to retune and no concept of `rope_theta` for the audio side. Whatever the audio tower produces at output position *t* depends only on the input acoustics, not on the LM's position schedule.

### 4.3 LM positional encoding (M-RoPE via `Qwen3VLTextRotaryEmbedding`)

The merged checkpoint inherits VL-2B's LM unchanged, so the LM-side RoPE is implemented by the **`Qwen3VLTextRotaryEmbedding`** class from `transformers.models.qwen3_vl.modeling_qwen3_vl`. Each of the 28 LM decoder layers instantiates one as `self.rotary_emb`, and it is responsible for rotating Q and K at every attention call — including for the `<|audio_pad|>` token positions where audio encoder features get spliced in. The vision tower uses a separate, simpler class (`Qwen3VLVisionRotaryEmbedding`); we are not concerned with that one here.

> **Note on naming.** Qwen3-ASR-1.7B's text config also declares the same M-RoPE fields (`mrope_section=[24, 20, 20]`, `rope_scaling.rope_type="default"`), but its modeling code lives in a different package. Since our merged ckpt loads with `Qwen3VLForConditionalGeneration`, the *runtime* RoPE class for both text tokens and audio_pad tokens is unambiguously `Qwen3VLTextRotaryEmbedding`. The ASR-1.7B-trained audio_tower weights enter that schedule as feature values, not as positions.

Both models share the same head geometry: `head_dim=128`, which factors into 64 RoPE rotation pairs (each pair rotates two adjacent feature dimensions together). M-RoPE generalizes vanilla RoPE by partitioning those 64 pairs across multiple **position axes** rather than rotating all 64 by a single scalar position.

`Qwen3VLTextRotaryEmbedding.forward` expects a `position_ids` tensor of shape `(3, batch, seq)`. The three rows correspond to:

- **Axis 0 — temporal/text position.** Sequential 0, 1, 2, … for text tokens. For video frames, this is the frame index. For interleaved chat the text axis runs through the whole sequence including any image/audio placeholder positions.
- **Axis 1 — visual height.** Row coordinate within an image's patch grid. Zero for non-visual tokens.
- **Axis 2 — visual width.** Column coordinate within an image's patch grid. Zero for non-visual tokens.

For audio tokens (the `<|audio_pad|>` slots in our POC), all three axes get the same sequential value as Axis 0 — i.e., audio tokens are positioned the same way text tokens are. The two visual-coordinate axes carry no audio-specific meaning.

The relevant config fields, side-by-side:

| Field | VL-2B | ASR-1.7B |
| --- | --- | --- |
| `head_dim` | 128 | 128 |
| `num_attention_heads` | 16 | 16 |
| `num_key_value_heads` | 8 | 8 |
| `mrope_section` | `[24, 20, 20]` | `[24, 20, 20]` |
| `mrope_interleaved` | `True` | `True` |
| `rope_type` | `'default'` | `'default'` |
| `rope_theta` | **5,000,000** | **1,000,000** |
| `max_position_embeddings` | **262,144** | **65,536** |

`mrope_section = [24, 20, 20]` sums to 64 = `head_dim / 2` and declares **how many RoPE pairs each of the three position axes drives**: 24 pairs for T, 20 for H, 20 for W. **It does not specify a contiguous chunked layout.** The actual mapping from RoPE-pair index to axis is computed by `apply_interleaved_mrope` and follows an interleaved 3-cycle, with leftover pairs falling back to T:

```text
RoPE pair index  : 0  1  2  3  4  5  6  7  8  …  57 58 59 60 61 62 63
position axis    : T  H  W  T  H  W  T  H  W  …   T  H  W  T  T  T  T
                   └────────── 20 × THW ──────────┘ └ 4 × T ┘
```

H lands at indices `1, 4, 7, …, 58`; W at `2, 5, 8, …, 59`; the remaining indices (every third starting from 0, plus 60–63) stay T. Counts: 24 T (= 20 from triplets + 4 trailing) + 20 H + 20 W = 64 ✓.

This interleaving is a **load-bearing design choice**, not a permutation: each axis gets RoPE pairs covering a *spread* of frequencies (from the highest-frequency end down to nearly the lowest), so each axis retains both fine-grained and long-range position discriminability. A chunked layout (T at lanes 0–23, H at 24–43, W at 44–63) would force the T axis into only the top 24 frequencies — fine-grained but with limited long-range discriminability — which is a poor trade for the text axis on long contexts.

In transformers 4.57.1's implementation, `Qwen3VLTextRotaryEmbedding.forward` calls `apply_interleaved_mrope` **unconditionally** — there is no branch on `rope_scaling.mrope_interleaved`. The `mrope_interleaved: True` field in both configs is therefore documentation rather than a runtime switch; the class is hardcoded to interleave.

`rope_type='default'` means **standard RoPE without any extrapolation scaling** — no NTK-aware, no dynamic-NTK, no linear, no LLaMA-3 style frequency scaling. The rotation angle for the pair at dimension index *i* (0 ≤ *i* < 64) at position *p* is:

```text
θ(i, p) = p · base^(-2i / head_dim) = p · base^(-i / 64)
```

with `base = rope_theta`. After interleaving, what each *axis* sees is the rotation angle indexed by **the lane indices that axis owns** (e.g. T sees `θ(0, p) , θ(3, p), θ(6, p), …`).

### 4.4 What `rope_theta` controls

`rope_theta` is the **base** of the geometric frequency schedule. The wavelength along which RoPE pair *i* completes one full rotation is `λ_i = 2π · base^(i / 64)`. Plugging in numbers:

| RoPE pair *i* | Wavelength at `θ=1M` (ASR-1.7B) | Wavelength at `θ=5M` (VL-2B) | Ratio |
| ---: | ---: | ---: | ---: |
| 0 (highest freq) | 2π ≈ 6.28 | 2π ≈ 6.28 | 1.00× |
| 16 | ≈ 6.28 · 1Mᐟ¼ ≈ 198 | ≈ 6.28 · 5Mᐟ¼ ≈ 297 | 1.50× |
| 32 | ≈ 6.28 · 1Mᐟ½ ≈ 6.3 K | ≈ 6.28 · 5Mᐟ½ ≈ 14 K | 2.24× |
| 48 | ≈ 6.28 · 1Mᐟ¾ ≈ 199 K | ≈ 6.28 · 5Mᐟ¾ ≈ 668 K | 3.34× |
| 63 (lowest freq) | ≈ 6.28 · 1M^(63/64) ≈ 5.86 M | ≈ 6.28 · 5M^(63/64) ≈ 28 M | 5.00× (saturates to base ratio) |

The **highest-frequency pairs** (small *i*) rotate at essentially the same rate in both schedules — at *i*=0 they are exactly identical. This carries fine-grained relative-position information. The **lowest-frequency pairs** (large *i*) carry coarse, long-range position information, and that is where the two schedules diverge: VL-2B's slow-rotating dims have wavelengths up to roughly 5× longer than ASR-1.7B's.

The intuition is straightforward: a model with a higher `rope_theta` is engineered to **distinguish positions over longer ranges before periodicity wraps around**. VL-2B's `θ=5M` with `max_position_embeddings=262K` gives it ≈ 100× headroom between the longest representable wavelength and its trained context — comfortable. ASR-1.7B's `θ=1M` with `max_position_embeddings=65K` gives ≈ 90× headroom — also comfortable. The two are tuned for their respective context budgets; neither is incorrect.

### 4.5 What `max_position_embeddings` controls

Despite the name, this field does **not** allocate a learned position-embedding table (RoPE is parameter-free). It is a declared invariant: the maximum sequence length the model has been trained / validated to operate at. Practically it gates how high the position counter is allowed to climb without warnings, and it sets the size of position-related KV-cache buffers in some implementations.

| Model | `max_position_embeddings` | Tuned for |
| --- | ---: | --- |
| Qwen3-VL-2B-Instruct | 262,144 | Long-context interleaved video + document input |
| Qwen3-ASR-1.7B | 65,536 | At most a 30-second audio window plus surrounding text — far short of 65K |

VL-2B's 262K is consistent with its higher `rope_theta=5M`: the team retuned RoPE for long-context behavior. ASR-1.7B's 65K with `rope_theta=1M` is the legacy Qwen3 short-context default.

### 4.6 Net divergence between the two LMs (for our POC)

Putting the comparison together, three fields differ between the two LMs in ways the merged checkpoint inherits from VL-2B (since we kept VL-2B's LM untouched):

| Aspect | Inherited value | Notes for the audio path |
| --- | --- | --- |
| `rope_theta` | 5,000,000 (VL-2B) | Audio_pad tokens get rotated by VL-2B's slower schedule, not ASR-1.7B's |
| `max_position_embeddings` | 262,144 (VL-2B) | Plenty of headroom for any audio-bearing sequence |
| `mrope_section` | `[24, 20, 20]` (identical on both sides) | No divergence; audio_pad tokens use Axis 0 only |
| Audio encoder positional embedding | Whisper-style absolute sinusoidal | Independent of LM RoPE |

### 4.7 Impact on §2.2 results

The first row of the table above is the only place the two models actually differ in a way that touches the audio path: audio_pad tokens are RoPE-rotated by VL-2B's schedule even though the audio encoder weights producing those features were trained against ASR-1.7B's schedule. **Why this didn't measurably degrade behavior on the 4-sample test:**

1. **Position counts in our prompts are tiny.** The longest expanded prompt was 94 tokens, all sequential text positions (no visual axes used). With *p* ≤ 100, the rotation angles in the slow-rotating dims that actually differ between the two schedules are infinitesimal:

   ```text
   θ(63, 100) at base=1M:  100 / 1Mᐟ¹·⁰  ≈ 1.0e-04 rad
   θ(63, 100) at base=5M:  100 / 5Mᐟ¹·⁰  ≈ 2.0e-05 rad
   ```

   Both effectively zero; the *difference* between them is negligible. The fast-rotating dims (small *i*) where actual position information is encoded behave identically under both schedules at any position.

2. **The audio encoder's output is *position-free*.** Sinusoidal positional encoding is added inside the encoder before the encoder layers; by the time those features exit at proj2, the position information is fully absorbed into the feature vectors. The LM's RoPE then re-positions them at audio_pad slots, exactly the same way it would for any non-RoPE-aware modality (e.g. image tokens from `model.visual`).

3. **Axis 1 / Axis 2 RoPE pairs (40 of 64) play no role.** Audio tokens have all three axes set to the same Axis-0 value, so the height/width RoPE rotation pairs rotate by the same scalar as Axis 0 — they don't introduce any audio-specific positional signal that would be sensitive to fine-tune divergence.

In short: the position-encoding divergence between VL-2B and ASR-1.7B is real but concentrates in the *long-position* regime, which the §2.2 prompts do not enter. The same divergence would matter for two scenarios the POC did not probe:

- **Long-form audio** that produces hundreds or thousands of `<|audio_pad|>` tokens. At *p* in the thousands, the slow-rotation dims start to differ in ways the audio encoder's training-time schedule (ASR-1.7B's `θ=1M`) didn't anticipate. Whether VL-2B's `θ=5M` is *better* or *worse* than ASR-1.7B's `θ=1M` for unseen audio features isn't predictable from the math alone — only empirically.
- **Mixed-modality long-context inputs** where audio tokens sit far into the sequence (say *p* > 10K) because of preceding video / document content. VL-2B was trained for that regime; ASR-1.7B's audio encoder was not exposed to it. This is the most likely failure mode of the architectural recipe and is a sensible target for a follow-up stress test.

## Part 5 — Plan (2026-04-28)

The POC validates that a Qwen3-VL-2B-based backbone can usefully consume audio features produced by Qwen3-ASR-1.7B's encoder. The next step is to **fold this into the existing omni-t2a training stack**: replace its current 0.6B Qwen3 LM backbone with the merged VL-2B+audio variant, and make the rest of the pipeline (data, loss, hardcoded IDs) follow. Five concrete tasks, in execution order:

### 5.1 Swap the omni-t2a backbone from Qwen3-0.6B to Qwen3-VL-2B (audio-only modality for now)

Replace the LM init source in the omni-t2a training code with our merged `Qwen3-VL-2B-Audio-POC` checkpoint. **Skip the ViT (`model.visual.*`) for now** — image inputs are not part of the omni-t2a task, and pulling the vision tower into the multi-stream architecture would add training-time complexity (extra streams, extra position-axis logic, extra data conventions) that is out of scope for the immediate scale-up. **Keep the audio encoder (`model.audio_tower.*`) wired into the understanding stream** as the source of audio-token features at `<|audio_pad|>` positions. Net result: the new backbone is the LM (28 layers, 2048 hidden) + audio_tower; the visual tower's weights live in the checkpoint but are unused at training/inference time. Fits cleanly with ursa's local Qwen3 re-implementation (`lib/ursa/ursa/models/qwen.py`) since the LM-side weight shapes are unchanged from the standard Qwen3 layout.

### 5.2 Rerun omni-t2a at 2B with the same data + training code as the 0.6B baseline

With the backbone swapped, retrain the 2B variant using the **identical** data mixture, recipe, and code path as the existing 0.6B run. The point is a clean apples-to-apples capacity ablation: same data, same loss, same multi-stream architecture, only the LM size + audio understanding initialization changed. This will surface (a) whether the 2B base meaningfully improves T2A audio quality, and (b) whether the audio-encoder-bearing init helps convergence vs. a from-scratch 2B (this is a separate ablation we may want later — out of scope for the first run).

### 5.3 Build an A2T data processing module mirroring the T2A pipeline

The existing pipeline produces (text → audio) supervision pairs. We need its mirror: (audio → text) for ASR-style training. Mirror the existing T2A data layout module symbol-for-symbol where possible — same dataset class structure, same row-level interface, same chunk-and-pack conventions — but with audio as the conditioning input and text as the target. The §2.2 zero-shot result implies real audio-bearing data exists in formats compatible with the audio_tower's input contract (16 kHz mono, ≤30 s windows, Whisper-style mel features); the new module's job is to package it into the training pipeline's expected schema. Once this module exists, the 2B model from §5.2 can be fine-tuned on a mixed T2A + A2T objective.

### 5.4 Mix CE loss (text targets) with diffusion loss (audio targets)

Today the omni-t2a loss is diffusion-only because the only output stream being supervised is audio. Adding A2T introduces text-token outputs, which require autoregressive cross-entropy supervision. The training step must therefore: (i) detect each example's target-stream type, (ii) compute CE loss on text positions and diffusion loss on audio positions, (iii) combine the two with appropriate scaling. Open design questions to settle when implementing: relative weight between the two losses, whether to normalize per-token or per-example, whether to enforce per-batch modality balance or allow free mixing.

### 5.5 Update hardcoded special-token IDs across training/inference scripts

The current omni-t2a code was written against Qwen3-0.6B's tokenizer, where audio/text/control token IDs differ from the merged VL-2B-Audio-POC tokenizer's IDs. From §1 of this document, the IDs we now use are:

- Audio frame markers: `<|audio_start|>=151669`, `<|audio_end|>=151670`, `<|audio_pad|>=151676`
- Text framing (shared with VL-2B): `<|im_start|>=151644`, `<|im_end|>=151645`, `<|endoftext|>=151643`

A sweep is needed over the omni-t2a training and inference scripts to (a) replace any hardcoded constants with values pulled from the tokenizer / model config, and (b) audit chat template / collator paths for assumptions about the old ID layout. Risk: silent miscoding bugs if any hardcoded ID slips through and lands on a now-occupied vocabulary slot.

### Dependencies

- §5.1 → §5.2 (swap before rerun)
- §5.5 must precede §5.2 to avoid silent miscoding
- §5.3 → §5.4 (need A2T data shape locked before designing the loss switch)
- §5.4 unblocks the first true mixed-objective training run; §5.2 above is single-objective (T2A only) at first.
