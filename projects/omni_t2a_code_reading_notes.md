# Omni T2A Data Pipeline — Code Reading Notes

**Date:** 2026-04-10
**Author:** Dong Guo
**Context:** Code review of PR #7206, focusing on the T2A data loading pipeline in `lib/koba/koba/pipelines/default_t2a.py`

---

## 1. `supervise_last_text_token` — Bridging Text and Audio

### What it does

In the T2A sequence plan, the TEXT element has `loss=False` (no supervision on transcript tokens), but `supervise_last_text_token=True` overrides this for a single token.

The implementation is in `lib/koba_shared/koba_shared/processor/omni_text_ops.py:237-242`:

```python
if og_element.supervise_last_text_token:
    last_non_padding_idx = (tok_element.padding_mask).nonzero(as_tuple=True)[0][-1]
    tok_element.txt_loss_mask[last_non_padding_idx] = True
```

### Which token is supervised?

Since the TEXT element wraps tokens with `<|im_start|>` and `<|im_end|>` (see `omni_text_ops.py:205-206`), the actual last non-padding token is **`<|im_end|>`**, not the last word of the transcript.

For example, `"hello world!"` becomes:

```
[<|im_start|>, he, llo, _wor, ld, !, <|im_end|>]
                                       ^^^^^^^^^ this is supervised
```

### What is the label?

The label construction happens in `lib/ursa/ursa/models/omni/inference/sequence_packing.py:228-261`. The key logic:

```python
next_first_token_id = sequence_plan[s_idx + 1].input_ids[0]
# ...
s_label[s.txt_loss_mask] = non_padding_supervised_text_ids[1:]
```

The label at the `<|im_end|>` position is `sequence_plan[next].input_ids[0]` — the first token ID of the next sequence element. This is a standard **text cross-entropy loss**, not a diffusion loss.

---

## 2. Audio Element Dual Representation

Audio elements carry **two parallel representations** for the two MoT branches:

| Branch | Field | Content | Loss type |
|--------|-------|---------|-----------|
| Diffusion/VAE | `x_audio` | Continuous audio waveform tensor | Diffusion loss |
| Text/LLM | `input_ids` | `[151655, 151655, ...]` (`<\|image_pad\|>` repeated) | None (`txt_loss_mask=0`) |

### Where do audio `input_ids` come from?

Audio is continuous embeddings, not discrete tokens. The placeholder IDs are manufactured explicitly:

1. `OmniElementAudio` (`lib/koba/koba/processor/omni_audio_ops.py:110-111`) sets:
   ```python
   tokens = [self.config.audio_pad_token] * num_audio_tokens  # "<|image_pad|>"
   og_element.text_str = "".join(tokens)
   ```

2. `OmniQwen3Tokenizer` then tokenizes this string via the `else` branch (`omni_text_ops.py:209-215`), producing `input_ids = [151655, 151655, ...]`.

These placeholder IDs serve two purposes:

- Give the text/LLM head something to process at audio positions
- Provide a discrete target for the `supervise_last_text_token` mechanism (the `<|im_end|>` before audio predicts `<|image_pad|>`)

---

## 3. Full Loss Chain for T2A (After System Prompt Addition)

After adding the task prompt and assistant prefix, the sequence plan has 3 elements:

```
[TEXT: "user\nSpeak the following transcript:\n<transcript>"]
  → [TEXT: "assistant\n"]
  → [AUDIO: audio tensor]
```

Tokenized form:

```
<|im_start|>user\nSpeak the following transcript:\n<transcript><|im_end|>
<|im_start|>assistant\n<|im_end|>
<|image_pad|><|image_pad|>...<|image_pad|>
```

Loss supervision chain:

1. **User text `<|im_end|>`** → text CE loss, label = `<|im_start|>` (first token of assistant element)
2. **Assistant `<|im_end|>`** → text CE loss, label = `151655` (`<|image_pad|>`, first audio placeholder)
3. **Audio positions** → diffusion loss on `x_audio` via VAE branch (`vae_token_mask=True`)

All other text tokens have `loss=False` (no text CE loss).

---

## 4. Audio Resampling

Source audio (often 48kHz) is resampled to 16kHz inside `AudioDecoder` (`lib/koba/koba/processor/audio_ops.py:50-53`):

```python
decoder = torchcodec.decoders.AudioDecoder(
    source=io.BytesIO(audio_bytes),
    sample_rate=self.config.sample_rate,  # 16000
    num_channels=self.config.num_channels,
)
```

`torchcodec` handles resampling internally during decode. No separate resampling step is needed.

---

## 5. Pipeline Architecture: Params vs Processors vs Config

The naming convention in `koba` differs from typical ML codebases:

### `T2APipelineParams`

A plain dataclass holding **scalar hyperparameters** (sample rate, dropout prob, compression factor, etc.). This is what most codebases would call "config". It does NOT build anything — just a bag of values used to construct processor Configs.

Defined in `lib/koba/koba/pipelines/pipelines.py:83`.

### `processors: list[EasyConfig]`

A list of nested `Config` objects (e.g., `AudioDecoder.Config`, `OmniT2AAudio.Config`). Each is a **factory/builder**: `EasyConfig` has a `.setup()` method that uses `__qualname__` to find the enclosing class and instantiate it.

```python
AudioDecoder.Config(sample_rate=16000).setup()  # → AudioDecoder(config)
```

### `Pipeline.Config`

Top-level container: holds `name` + `processors` list. On init (`pipelines.py:351`):

```python
self.processors = [p.setup() for p in config.processors]
```

### Key point: Composition, not inheritance

`AudioDecoder`, `OmniT2AAudio`, etc. are **NOT** subclasses of `Pipeline`. The relationship is composition — `Pipeline` holds a list of processor Configs and calls `.setup()` on each. The only shared base class is `EasyConfig`, and only for the inner `Config` classes, not the processors themselves.

```
Pipeline.Config                          # "what pipeline to build"
  └── processors: list[EasyConfig]       # "what steps to run" (each is a factory)
        └── built using T2APipelineParams  # "what scalar values to use"
```
