> ⚠️ **TEMPORARY NOTE — created 2026-04-08**
> This file is a working scratch-pad for the director-centric dense video captioning PoC (local Gemini API runs).
> **It will be removed within a few days** once the PoC runs are complete.
> Findings will be merged into a formal captioning notes file (likely under `projects/`).

---

# PoC: Director-Centric Dense Video Captioning (Local Gemini API)

## Goal

Run a quick local PoC on a MacBook:

1. Download **~8 sample videos** from each of the datasets enumerated below
2. Call the **Gemini API** directly (no infra, no vLLM, no throughput concerns) to generate dense director-centric captions
3. Verify and compare captioning prompt variants
4. Zero production concerns — this is purely exploratory

---

## Context & Background

- Branch under investigation: `riddhish/captioning_baselines` ([GitHub compare](https://github.com/lumalabs/lumaverse/compare/riddhish/captioning_baselines?expand=1))
- Core frame extraction PR: [lumaverse#6996](https://github.com/lumalabs/lumaverse/pull/6996) — hplv-based frame-level video processing
- Director-style captioning script reference (Dong Guo): `projects/lax/lax/projects/vllm/prompts/video_director.py#L17` on branch `dongguo/omni-t2v`
- Riddhish's prior baseline comparison report (Qwen 235B / 397B / Gemini Pro / Flash):
  - S3: `s3://ai-lumalabs-datasets-ap-se-2/lance_datasets/riddhishb/video_captioning/reports/video_caption_comparison.html`
  - [Internal dashboard viewer](https://internal-dashboard.sandbox.labs.lumalabs.ai/dashboards/html-viewer?path=s3://ai-lumalabs-datasets-ap-se-2/lance_datasets/riddhishb/video_captioning/reports/video_caption_comparison.html)
- Camera grounding experiment report: `s3://ai-lumalabs-dashboard-samples-ap-se-2/report/camera_caption_experiments_report.html`

---

## Datasets & S3 Data Pointers

All datasets are **Lance format** on S3. Use the `lance` Python package to open them.

| # | Dataset | S3 Path | Notes |
|---|---|---|---|
| 1 | **Ray SFT Videos** (source) | `s3://ai-lumalabs-datasets-ap-se-2/lance_datasets/video/ray3_sft_baked/stage1_v2.lance` | Raw 5s video clips used for storyboard extraction |
| 2 | **Ray SFT Storyboard** | `s3://ai-lumalabs-datasets-ap-se-2-lance/lance_datasets/riddhishb/ray_sft_storyboard_dataset_nvidia.lance` | ~440k clips (incl. ~70k 4K, may need resize); storyboard frames extracted |
| 3 | **Shotdeck + Refs** | `s3://ai-lumalabs-datasets-ap-se-2-lance/lance_datasets/riddhishb/shotdeck_storyboard_with_ref.lance` | ~490k movie clips with actor reference images |
| 4 | **Handheld batch 1** | `s3://ai-lumalabs-datasets-ap-se-2-lance/lance_datasets/riddhishb/overseas_handheld_storyboard1.lance` | Vertical handheld videos, good for product/subject refs |
| 5 | **Handheld batch 2** | `s3://ai-lumalabs-datasets-ap-se-2-lance/lance_datasets/riddhishb/overseas_handheld_storyboard2.lance` | Same format as above |
| 6 | **Handheld batch 3** | `s3://ai-lumalabs-datasets-ap-se-2-lance/lance_datasets/riddhishb/overseas_handheld_storyboard3.lance` | Same format as above |

**Relevant lance columns:**
- `media_bytes` — raw video bytes (or keyframe images depending on dataset)
- `keyframe_indices` — indices of selected keyframes
- `captions` — existing captions (for comparison)
- `fps`, `clip_start_time`, `clip_end_time` — timing info
- `reference_image_paths`, `reference_caption` — actor/subject refs (Shotdeck + Handheld)
- `keyframe_aesthetic_scores`, `keyframe_clarity_scores` — quality signals (Handheld)

> **Schema note (from team discussion):** Column naming is being standardized.
> Prefer `keyframes` for keyframe images, `references` for reference images going forward.
> Avoid `media_bytes` in new datasets — it's overloaded (used for both video and reference images historically).

---

## Local Setup (MacBook)

### 1. Prerequisites

```bash
pip install lance boto3 google-generativeai pillow opencv-python tqdm
```

AWS credentials must be configured (`~/.aws/credentials` or env vars `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).

Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-key-here"
```

### 2. Sample Video Download Script

```python
#!/usr/bin/env python3
"""
poc_download_samples.py

Downloads 8 sample videos from each of the 6 lance datasets for local captioning PoC.
Saves to ./samples/<dataset_name>/<idx>.mp4
"""

import os
import lance
import pyarrow as pa
from pathlib import Path

DATASETS = {
    "ray_sft_source": "s3://ai-lumalabs-datasets-ap-se-2/lance_datasets/video/ray3_sft_baked/stage1_v2.lance",
    "ray_sft_storyboard": "s3://ai-lumalabs-datasets-ap-se-2-lance/lance_datasets/riddhishb/ray_sft_storyboard_dataset_nvidia.lance",
    "shotdeck_storyboard_ref": "s3://ai-lumalabs-datasets-ap-se-2-lance/lance_datasets/riddhishb/shotdeck_storyboard_with_ref.lance",
    "handheld_1": "s3://ai-lumalabs-datasets-ap-se-2-lance/lance_datasets/riddhishb/overseas_handheld_storyboard1.lance",
    "handheld_2": "s3://ai-lumalabs-datasets-ap-se-2-lance/lance_datasets/riddhishb/overseas_handheld_storyboard2.lance",
    "handheld_3": "s3://ai-lumalabs-datasets-ap-se-2-lance/lance_datasets/riddhishb/overseas_handheld_storyboard3.lance",
}

N_SAMPLES = 8
OUT_DIR = Path("./samples")

def download_samples():
    for name, uri in DATASETS.items():
        out_path = OUT_DIR / name
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"\n[{name}] Opening {uri}")
        try:
            ds = lance.dataset(uri)
            # Sample N random rows
            total = ds.count_rows()
            print(f"  Total rows: {total}")
            tbl = ds.take(list(range(0, min(N_SAMPLES * 10, total), max(1, total // N_SAMPLES))))
            rows = tbl.to_pydict()

            saved = 0
            for i in range(min(N_SAMPLES, len(rows.get("_rowid", rows.get("media_bytes", [None]))))):
                # Try to get video bytes from media_bytes column
                video_bytes = None
                for col in ["media_bytes", "video", "video_bytes"]:
                    if col in rows and rows[col][i] is not None:
                        video_bytes = rows[col][i]
                        break

                if video_bytes is None:
                    print(f"  [!] No video bytes in row {i}, skipping")
                    continue

                if isinstance(video_bytes, (bytes, bytearray)):
                    data = bytes(video_bytes)
                else:
                    data = bytes(video_bytes.as_py())

                fpath = out_path / f"sample_{i:02d}.mp4"
                fpath.write_bytes(data)
                saved += 1
                print(f"  Saved {fpath} ({len(data)/1024:.1f} KB)")

            print(f"  [{name}] Saved {saved}/{N_SAMPLES} samples")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

if __name__ == "__main__":
    download_samples()
```

---

## Director-Centric Dense Caption Prompt

Based on Dong Guo's `video_director.py` approach (branch `dongguo/omni-t2v`, `projects/lax/lax/projects/vllm/prompts/video_director.py#L17`) and Riddhish's `CAPTIONING_PROMPT_CAPTION_MAX` shared in `#proj-prism`.

### System Prompt

```
You are an expert cinematographer and film director writing a technical shot description
for a training dataset. Your goal is to produce a dense, objective, single-paragraph
caption that a director or DP would write in a shot report.
```

### User Prompt (director-centric dense caption)

```python
VIDEO_DIRECTOR_DENSE_CAPTION_PROMPT = """
Analyze this video clip and write a single dense paragraph describing it as a
director/cinematographer would in a detailed shot report. Cover all of the following
in order — do not use bullet points or headers:

1. SHOT TYPE & FRAMING: (e.g., extreme wide shot, medium close-up, over-the-shoulder,
   two-shot, bird's-eye, Dutch angle, POV). Describe the frame composition, subject
   placement, and rule-of-thirds positioning.

2. CAMERA MOTION: Be precise. Distinguish camera movement from subject movement.
   Use exact filmmaking terminology: static, pan (left/right), tilt (up/down),
   dolly (in/out/lateral), truck, pedestal, handheld (shake level), steadicam,
   crane/jib (arc, boom up/down), zoom (in/out). State direction explicitly.

3. SUBJECT & ACTION: Identify every visible person (count, gender if discernible,
   approximate age, clothing, body orientation). Describe exact body pose — joint
   positions, limb angles, head orientation. Describe any motion or action with
   precise spatial detail (e.g., "walks left-to-right crossing frame center").

4. SCENE & ENVIRONMENT: Setting, location type, time of day, weather/lighting
   conditions. Describe background elements that are clearly visible.

5. LIGHTING: Quality (hard/soft), direction (key light angle, fill, backlight),
   color temperature (warm/cool/neutral), practical light sources if visible.

6. VISUAL STYLE: Identify the medium (live-action, 3D render, animation, archival,
   screen recording). Note lens characteristics if apparent (wide distortion, telephoto
   compression, shallow vs. deep depth of field, lens flare, bokeh).

7. COLOR & TONE: Overall palette, saturation level, notable color grading choices
   (teal-orange, desaturated, high-contrast, etc.).

8. TEXT ON SCREEN: If any text is visible, transcribe it exactly as shown including
   formatting, capitalization, and positioning on screen.

Output: One continuous, dense paragraph. No bullet points. No section headers.
Be objective — describe only what is directly visible. Do not infer intent or emotion
unless physically explicit in facial muscle positions.
""".strip()
```

---

## Gemini API Captioning Script

```python
#!/usr/bin/env python3
"""
poc_gemini_caption.py

Runs Gemini API dense captioning on downloaded sample videos.
Outputs captions to ./captions/<dataset_name>/<idx>.txt and a summary JSON.
"""

import os
import json
import time
from pathlib import Path
import google.generativeai as genai

# --- Config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"   # or "gemini-2.0-pro" for higher quality
SAMPLES_DIR = Path("./samples")
CAPTIONS_DIR = Path("./captions")
DELAY_BETWEEN_CALLS = 1.0           # seconds; no throughput concerns here

# --- Prompt ---
SYSTEM_PROMPT = (
    "You are an expert cinematographer and film director writing a technical shot "
    "description for a training dataset. Your goal is to produce a dense, objective, "
    "single-paragraph caption that a director or DP would write in a shot report."
)

VIDEO_DIRECTOR_DENSE_CAPTION_PROMPT = """
Analyze this video clip and write a single dense paragraph describing it as a
director/cinematographer would in a detailed shot report. Cover all of the following
in order — do not use bullet points or headers:

1. SHOT TYPE & FRAMING: (e.g., extreme wide shot, medium close-up, over-the-shoulder,
   two-shot, bird's-eye, Dutch angle, POV). Describe the frame composition, subject
   placement, and rule-of-thirds positioning.

2. CAMERA MOTION: Be precise. Distinguish camera movement from subject movement.
   Use exact filmmaking terminology: static, pan (left/right), tilt (up/down),
   dolly (in/out/lateral), truck, pedestal, handheld (shake level), steadicam,
   crane/jib (arc, boom up/down), zoom (in/out). State direction explicitly.

3. SUBJECT & ACTION: Identify every visible person (count, gender if discernible,
   approximate age, clothing, body orientation). Describe exact body pose — joint
   positions, limb angles, head orientation. Describe any motion or action with
   precise spatial detail.

4. SCENE & ENVIRONMENT: Setting, location type, time of day, weather/lighting
   conditions. Describe background elements that are clearly visible.

5. LIGHTING: Quality (hard/soft), direction, color temperature, practical light sources.

6. VISUAL STYLE: Medium (live-action, 3D render, animation), lens characteristics.

7. COLOR & TONE: Overall palette, saturation, color grading choices.

8. TEXT ON SCREEN: Transcribe exactly if visible.

Output: One continuous, dense paragraph. No bullet points. No section headers.
Be objective — describe only what is directly visible.
""".strip()


def caption_video(video_path: Path, model) -> str:
    """Upload video to Gemini Files API and generate caption."""
    print(f"  Uploading {video_path.name}...")
    video_file = genai.upload_file(str(video_path))

    # Wait for processing
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_path}")

    print(f"  Generating caption...")
    response = model.generate_content(
        [VIDEO_DIRECTOR_DENSE_CAPTION_PROMPT, video_file],
        generation_config=genai.types.GenerationConfig(
            temperature=0.4,        # moderate: creative but grounded
            top_p=0.9,
            max_output_tokens=1024,
            # NOTE: do NOT set presence_penalty — harmful for captioning
            # (prevents model from repeating text/OCR/metadata it sees)
        )
    )

    # Clean up uploaded file
    genai.delete_file(video_file.name)
    return response.text


def run_captioning():
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    results = {}

    for dataset_dir in sorted(SAMPLES_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        out_dir = CAPTIONS_DIR / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Dataset: {dataset_name} ===")
        results[dataset_name] = {}

        for video_path in sorted(dataset_dir.glob("*.mp4")):
            caption_path = out_dir / f"{video_path.stem}.txt"
            if caption_path.exists():
                print(f"  [skip] {video_path.name} (already captioned)")
                results[dataset_name][video_path.stem] = caption_path.read_text()
                continue

            try:
                caption = caption_video(video_path, model)
                caption_path.write_text(caption)
                results[dataset_name][video_path.stem] = caption
                print(f"  ✓ {video_path.name}")
                print(f"    → {caption[:120]}...")
                time.sleep(DELAY_BETWEEN_CALLS)
            except Exception as e:
                print(f"  [ERROR] {video_path.name}: {e}")
                results[dataset_name][video_path.stem] = f"ERROR: {e}"

    # Save summary JSON
    summary_path = CAPTIONS_DIR / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    run_captioning()
```

---

## Key Findings from Prior Experiments (from Slack / #proj-prism)

### Captioning approach comparison (Riddhish, 2026-03-19, #proj-prism)

Testing **5 approaches** for injecting camera motion signal into Qwen3-VL-235B captions:

| Approach | Result |
|---|---|
| ✅ Baseline (no camera info) | Most natural cinematic language — interprets visual effect of motion correctly. But can get direction wrong or hallucinate motion. |
| ⚠️ Text labels (DA3 per-frame labels) | Too granular, reads like a motion log ("performs a series of dynamic movements including dolly ins, pans, arcs..."). Not useful for training. |
| ❌ Overlays on frames | Did not work |
| ❌ Interleaved arrows on frames | Did not work |
| ✅ Two-pass pipeline (Camera_Motion field) | Most promising — use a separate dedicated `Camera_Motion` field and a specialist pass |

**Key insight:** DA3 rotation (pan/tilt) signals are reliable; zoom/focal are not (entangled with depth estimation, especially on animation).

### Qwen3-VL frame count
Current default captioning pipeline uses **12 equally spaced frames** (Manuel Kansy, #proj-prism).

### ⚠️ Critical: Qwen 3.5 sampling parameters
When using **Qwen 3.5 non-thinking mode** (e.g., 397B), do **NOT** set `presence_penalty=1.5`
(the official recommended default). It prevents the model from repeating visible text/OCR/metadata
from the prompt — which is harmful for captioning. **Remove it entirely.** (Haoxiang Wang, 2026-04-07)

### VLMs under active comparison
- Qwen3-VL 235B-A22B
- Qwen3.5 397B-A17B MoE (hybrid linear attn + sparse MoE)
- Gemini 2.0 Pro / Flash
- Kimi 2.5 (planned)
- Gemini 3.1 (planned comparison with Qwen 3.5)

---

## Prompt Variants to Test in This PoC

Track results across at least these prompt variations:

| Variant | Description |
|---|---|
| `director_dense` | Full director-centric prompt above (all 8 axes) |
| `director_camera_only` | Only camera motion + shot type (for camera motion eval) |
| `director_no_camera` | Full prompt minus camera motion section (for subject/scene eval) |
| `caption_max_baseline` | Riddhish's existing exhaustive `CAPTIONING_PROMPT_CAPTION_MAX` (one dense paragraph, no structure) |

---

## Output Structure

```
./
├── poc_download_samples.py
├── poc_gemini_caption.py
├── samples/
│   ├── ray_sft_source/          sample_00.mp4 ... sample_07.mp4
│   ├── ray_sft_storyboard/      ...
│   ├── shotdeck_storyboard_ref/ ...
│   ├── handheld_1/              ...
│   ├── handheld_2/              ...
│   └── handheld_3/              ...
└── captions/
    ├── ray_sft_source/          sample_00.txt ... sample_07.txt
    ├── ...
    └── results_summary.json
```

---

## People to Loop In / References

| Person | Area |
|---|---|
| **Riddhish** (`@U0A97DKLHQD`) | Owner of captioning_baselines branch, storyboard data |
| **Manuel Kansy (Manny)** (`@U0A7C7FC9C0`) | Temporal captioning, JSON prompt format, Ray captioning pipeline |
| **Dong Guo** (`@U0AJ8DWQSKB`) | Director-style captioning script (`video_director.py`) |
| **Haoxiang Wang** (`@U09M1HU6HH9`) | Qwen 3.5 captioning pipeline, sampling param findings |
| **Wendy Xian** (`@U09N27Y63T2`) | 3D grounding (DA3 depth/pose) for camera-aware captioning |
| **Shyamal Buch** (`@U0948LJ7JER`) | Eval standards, CameraBench terminology |

**Slack thread references:**
- Riddhish camera grounding experiments: [#proj-prism 2026-03-19](https://luma-ai.slack.com/archives/C0A3YH84Y85/p1773984826783809)
- Riddhish Friday update (datasets + report): [#proj-omni 2026-04-03](https://luma-ai.slack.com/archives/C0816AKQ4SJ/p1775265471937659)
- Riddhish daily update (data progress): [#proj-omni 2026-04-07](https://luma-ai.slack.com/archives/C0816AKQ4SJ/p1775612548646999)
- Frame extraction bug thread (hplv): [#C097TBLP72A 2026-04-07](https://luma-ai.slack.com/archives/C097TBLP72A/p1775578611787019)
- Haoxiang Qwen 3.5 sampling param tip: [#proj-omni daily update 2026-04-07 thread](https://luma-ai.slack.com/archives/C0816AKQ4SJ/p1775611804101989)
- CameraBench reference (Manny): https://linzhiqiu.github.io/papers/camerabench/

---

*Last updated: 2026-04-08 by Dong (via Cowork)*
