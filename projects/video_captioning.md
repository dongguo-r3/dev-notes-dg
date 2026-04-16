# Video Captioning

> **What this note is for:** Umbrella doc for all video captioning R&D — including dense captions, embedding strategies, and PoC experiments. Specific captioning approaches are tracked in their own files (e.g. `video_captioning_director_style.md`).
>
> - **Dense captions for T2V:** Current T2V training data lacks dense captions. T5 encoders not suited for dense caption encoding (even beyond 512-token limit). LLM top-layer embeddings cannot simply replace T5 — no natural "encoder" layer in decoder-only LLMs.
> - **Project status:** Not officially kicked off; some consensus on challenges and directions.
> - **Next step:** Run PoC experiments on Riddhish's captioning baselines branch.

## Meta
- **Owner:** Dong
- **Status:** 🟡 In Progress
- **Priority:** Medium
- **Created:** 2026-04-08
- **Last Updated:** 2026-04-08
- **Tags:** captioning, video, dense-captions, t2v, t5, llm

---

## Goal

Research and develop video captioning capabilities for T2V training data — including dense captions, embedding strategies, and PoC experiments. This is the umbrella project tracking all video captioning efforts; specific approaches (e.g. director-style prompting) are tracked in their own worker files.

---

## Tasks
- [ ] Run preliminary PoC experiments on video data using Riddhish's captioning baselines branch

---

## Blockers
_None_

---

## Notes / Running Log

- **2026-04-08**: File created.
- **2026-04-08**: Synced with Riddhish on dense video captions. Key takeaways:
  - Current T2V data does not have dense captions. T5 encoders are not trained for dense caption scenarios — even without the 512-token limit, T5 is not well-suited to encoding dense captions.
  - Cannot simply replace T5 embeddings with LLM top-layer embeddings in Ray models: LLM top layers target token-level embeddings, and since LLMs are not encoder-decoder models, no single transformer layer naturally plays the role of "encoder embedding".
  - The dense video captioning project has not been officially kicked off; some consensus exists on the challenges and directions.
  - Got a branch from Riddhish to run preliminary PoC experiments on video data (see References).

---

## References

- Riddhish's captioning baselines branch: https://github.com/lumalabs/lumaverse/compare/riddhish/captioning_baselines?expand=1
- Director-style captioning (specific approach): [video_director_captioning.md](video_director_captioning.md)
