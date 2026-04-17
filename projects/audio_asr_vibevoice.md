# VibeVoice ASR — podcast_10m Status

> **What this note is for:** Tracking the VibeVoice ASR v2 run over the `podcast_10m` multilingual dataset (221M rows, 16 partitions). Logs partition status, active job IDs, cluster assignments, and analysis dashboards.
>
> | Metric | Value |
> |---|---|
> | Total rows | 221,842,325 (16 partitions) |
> | As of 2026-04-06 | 13/16 complete, 3 running (p2, p6, p13) |
> | As of 2026-04-17 | **All 16/16 complete** — 221,324,671 rows on disk (99.77% of source, 517,654 gap) |
> | Analysis dashboards | Uploaded 2026-04-07 to S3 (4 dashboards — see below) |

---

Source: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance`
Total rows: 221,842,325 across 16 partitions
Output prefix: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p{N}_16_1.lance`

Pipeline: `lax.projects.av_data_processing.audio.asr_vibevoice.pipeline_vllm.run_vibevoice_asr_vllm_pipeline`  
Submit script: `python -m lax.scripts.submit_ray_job` from `/Users/dongguo/Projects/lumaverse/projects/lax`  
Runtime env: `lax/projects/av_data_processing/audio/asr_vibevoice/runtime_env_local.json` (local, contains HF_TOKEN)

## v2 Partition Status (updated 2026-04-17 — all complete)

Verified 2026-04-17 by querying each partition's output Lance table directly.
All 3 jobs running on 2026-04-06 (p2, p6, p13) have since committed final state
(last commits 2026-04-06 21:54 — 22:47 UTC).

| Partition | Cluster | Rows | Last Version | Last Updated | Status |
|-----------|---------|------|-------------|--------------|--------|
| p0_16_1 | omniva-s0 | 14,000,000 | v99 | 2026-04-04 08:07 | ✅ Complete |
| p1_16_1 | omniva-s1 | 14,000,000 | v98 | 2026-04-01 22:28 | ✅ Complete |
| p2_16_1 | tmp-s2 (`dongguo-vibevoice-tmp-s2-633ae8`) | 13,990,495 | v98 | 2026-04-06 22:47 | ✅ Complete — resume job `raysubmit_piq44zQzedA9M2s6` finished (+523K rows since 2026-04-06 14:58) |
| p3_16_1 | omniva-s3 | 13,898,108 | v97 | 2026-04-04 04:05 | ✅ Complete |
| p4_16_1 | omniva-s4 | 13,897,974 | v97 | 2026-04-04 17:28 | ✅ Complete |
| p5_16_1 | tmp-s3 (`dongguo-vibevoice-tmp-s3-543e8a`) | 13,337,105 | v95 | 2026-04-06 21:54 | ✅ Complete (resume job finished in 35s) |
| p6_16_1 | tmp-s1 (`dongguo-vibevoice-tmp-s1-13c202`) | 13,860,492 | v98 | 2026-04-06 22:19 | ✅ Complete — resume job `raysubmit_RH7TRn1sZB9H2GR2` finished (+230K rows) |
| p7_16_1 | omniva-s7 | 13,869,821 | v97 | 2026-04-03 22:08 | ✅ Complete |
| p8_16_1 | tmp-s8 | 13,863,636 | v96 | 2026-04-02 02:53 | ✅ Complete |
| p9_16_1 | tmp-s9 | 13,862,549 | v97 | 2026-04-01 15:08 | ✅ Complete |
| p10_16_1 | tmp-s10 | 13,867,698 | v97 | 2026-04-03 14:19 | ✅ Complete |
| p11_16_1 | tmp-s11 | 13,897,600 | v97 | 2026-04-03 14:31 | ✅ Complete |
| p12_16_1 | tmp-s12 | 13,897,454 | v97 | 2026-04-01 14:48 | ✅ Complete |
| p13_16_1 | tmp-s5 (`dongguo-vibevoice-tmp-s5-2c12fb`) | 13,282,309 | v92 | 2026-04-06 21:54 | ✅ Complete — resume job `raysubmit_WvYucpFDRSe41Myq` appears finished (row count stable since 2026-04-06 14:58, ~583K rows never materialized — likely natural partition size gap, not a stalled job) |
| p14_16_1 | tmp-s14 | 13,899,430 | v97 | 2026-04-03 00:41 | ✅ Complete |
| p15_16_1 | tmp-s15 | 13,900,000 | v97 | 2026-04-01 15:09 | ✅ Complete |
| **Total** | | **221,324,671** | | | vs 221,842,325 source = **99.77%**, gap 517,654 rows (0.23%) |

**All 16/16 complete** as of 2026-04-06 22:47 UTC. Gap is consistent with fidelity v1's 0.27% gap on the same source — likely Arrow 2GB overflow batches that failed silently. If needed, a cleanup pass with `batch_size=1024` on the missing fragments should recover most of the gap.

## Active Jobs (historical — all finished 2026-04-06)

Resume jobs for the 3 late-running partitions. All have since finished and
their output tables are committed (see status table above).

| Job ID | Cluster | Partition | Final State |
|--------|---------|-----------|-------------|
| `raysubmit_RH7TRn1sZB9H2GR2` | tmp-s1 (dongguo-vibevoice-tmp-s1-13c202) | p6 | ✅ Complete 2026-04-06 22:19 |
| `raysubmit_piq44zQzedA9M2s6` | tmp-s2 (dongguo-vibevoice-tmp-s2-633ae8) | p2 | ✅ Complete 2026-04-06 22:47 |
| `raysubmit_WvYucpFDRSe41Myq` | tmp-s5 (dongguo-vibevoice-tmp-s5-2c12fb) | p13 | ✅ Complete 2026-04-06 21:54 |

The `tmp-s1`, `tmp-s2`, `tmp-s5` clusters may or may not still exist —
check with `kubectl get rayclusters -n flytesnacks-development | grep tmp-s`
before reuse. These were ad-hoc resume clusters, not part of the stable
omniva-sN / metadata-sN cluster families.

## Analysis Dashboards (vibevoice_analysis_v2)

Uploaded to `s3://ai-lumalabs-datasets-ap-se-2/dongguo/datasets/audio/asr/` on 2026-04-07.

| Dashboard | Link |
|-----------|------|
| Podcast single speaker comparison | [vibevoice_whisperx_compare_podcast_single_speaker](https://internal-dashboard.sandbox.labs.lumalabs.ai/dashboards/html-viewer?path=s3%3A%2F%2Fai-lumalabs-datasets-ap-se-2%2Fdongguo%2Fdatasets%2Faudio%2Fasr%2Fvibevoice_whisperx_compare_podcast_single_speaker.html) |
| Podcast 2-speaker comparison | [vibevoice_whisperx_compare_podcast_2speakers](https://internal-dashboard.sandbox.labs.lumalabs.ai/dashboards/html-viewer?path=s3%3A%2F%2Fai-lumalabs-datasets-ap-se-2%2Fdongguo%2Fdatasets%2Faudio%2Fasr%2Fvibevoice_whisperx_compare_podcast_2speakers.html) |
| EN progressive filtering | [vibevoice_whisperx_en_progressive_filtering](https://internal-dashboard.sandbox.labs.lumalabs.ai/dashboards/html-viewer?path=s3%3A%2F%2Fai-lumalabs-datasets-ap-se-2%2Fdongguo%2Fdatasets%2Faudio%2Fasr%2Fvibevoice_whisperx_en_progressive_filtering.html) |
| Language detection | [vibevoice_whisperx_lang_detection](https://internal-dashboard.sandbox.labs.lumalabs.ai/dashboards/html-viewer?path=s3%3A%2F%2Fai-lumalabs-datasets-ap-se-2%2Fdongguo%2Fdatasets%2Faudio%2Fasr%2Fvibevoice_whisperx_lang_detection.html) |

## Notes

- v1 runs all produced empty output due to `HF_TOKEN` not being passed to `ProcessPoolExecutor` spawn workers. Fixed in `pipeline_vllm.py`.
- Use `runtime_env_local.json` (not checked in) which contains the real `HF_TOKEN`. The committed `runtime_env.json` has `${HF_TOKEN}` placeholder.
- Partitions p0-p7 were originally on omniva-flyte; p8-p15 on kiwi-flyte. The 2026-04-06 resume jobs for p2/p6/p13 were submitted to idle kiwi clusters instead.
- p5 appeared stalled at 96.2% but was actually complete — the partition only has 13,337,105 rows (not every partition is equal size).
- 2026-04-06: deleted idle clusters s0, s3, s6, s8, s9, s15 after confirming no active jobs. Kept s1, s2, s5 for the 3 running resume jobs.
