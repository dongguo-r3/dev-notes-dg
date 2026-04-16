# Video Director-Style Captioning

> **What this note is for:** A specific captioning approach that prompts a VLM to describe video clips from a film director's perspective — annotating intentional compositional and narrative choices rather than literal pixel content. Tracks prompts, implementation, and results.
>
> - **Approach:** Treat the VLM as a film director annotating shots (camera movement, shot type, lighting mood, pacing) rather than a generic dense captioner.
> - **Implementation:** v4.1 director-style prompts + two-pipeline Qwen3-VL-235B on branch `dongguo/omni-t2v`.
> - **Parent effort:** See `video_captioning.md` for the broader video captioning research.

## Meta
- **Owner:** Dong
- **Status:** 🟡 In Progress
- **Priority:** Medium
- **Created:** 2026-04-06
- **Last Updated:** 2026-04-08
- **Tags:** lax, vllm, captioning, video, qwen3-vl

---

## Goal

Generate high-quality, structured descriptions of cinematic/movie video clips using a "director" framing — treating the VLM as a film director annotating shots rather than a generic dense captioner.

---

## Key Insight

For movie/cinematic video clips, simply prompting a VLM to generate dense captions is **not the best fit** for this domain. Generic dense captioning tends to describe literal pixel content without capturing the intentional compositional and narrative choices that define cinematic footage (camera movement, shot type, lighting mood, pacing, etc.).

The director-centered framing instead asks the model to annotate *why* a shot was composed the way it was, which produces richer, more semantically meaningful descriptions for training data.

---

## Implementation

**Branch:** `dongguo/omni-t2v` on `lumalabs/lumaverse`

- `projects/lax/lax/projects/vllm/prompts/video_director.py` — v4.1 director-style system/user prompts
- `projects/lax/lax/projects/vllm/qwen3_vl_235b/pipeline_director_captioning.py` — two-pipeline Qwen3-VL-235B implementation (shot-level captioning + director-style annotation)

---

## Notes / Running Log

- **2026-04-06**: Director-centered description approach inspired by a demo in Slack (see References). v4.1 prompts and two-pipeline implementation committed.

---

## To Explore

- [ ] **2026-04-08**: Survey the concept of "camera-guided ____" — understand what it means to use camera motion/trajectory as an explicit conditioning or guiding signal in video generation/captioning. Fill in the blank and document findings here.

---

## References

- Slack demo that inspired the director-centered approach: https://luma-ai.slack.com/archives/C0816AKQ4SJ/p1775265471937659?thread_ts=1775262604.370749&cid=C0816AKQ4SJ
- Broader video captioning effort: [video_captioning.md](video_captioning.md)
