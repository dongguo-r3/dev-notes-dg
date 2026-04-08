# VibeVoice ASR — podcast_10m Status

Source: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance`  
Total rows: 221,842,325 across 16 partitions  
Output prefix: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p{N}_16_1.lance`

Pipeline: `lax.projects.av_data_processing.audio.asr_vibevoice.pipeline_vllm.run_vibevoice_asr_vllm_pipeline`  
Submit script: `python -m lax.scripts.submit_ray_job` from `/Users/dongguo/Projects/lumaverse/projects/lax`  
Runtime env: `lax/projects/av_data_processing/audio/asr_vibevoice/runtime_env_local.json` (local, contains HF_TOKEN)

## v2 Partition Status (updated 2026-04-06)

| Partition | Cluster | Rows | Last Version | Last Updated | Status |
|-----------|---------|------|-------------|--------------|--------|
| p0_16_1 | omniva-s0 | 14,000,000 | v99 | 2026-04-04 | **Complete** |
| p1_16_1 | omniva-s1 | 14,000,000 | v98 | 2026-04-01 | **Complete** |
| p2_16_1 | tmp-s2 (`dongguo-vibevoice-tmp-s2-633ae8`) | 13,467,200 | v93 | 2026-04-06 | **Running** — Job `raysubmit_piq44zQzedA9M2s6`, ~398K rows left |
| p3_16_1 | omniva-s3 | 13,898,108 | v97 | 2026-04-03 | **Complete** |
| p4_16_1 | omniva-s4 | 13,897,974 | v97 | 2026-04-04 | **Complete** |
| p5_16_1 | tmp-s3 (`dongguo-vibevoice-tmp-s3-543e8a`) | 13,337,105 | v94 | 2026-04-06 | **Complete** — Resume job finished in 35s, partition fully processed |
| p6_16_1 | tmp-s1 (`dongguo-vibevoice-tmp-s1-13c202`) | 13,630,387 | v95 | 2026-04-06 | **Running** — Job `raysubmit_RH7TRn1sZB9H2GR2`, ~235K rows left |
| p7_16_1 | omniva-s7 | 13,869,821 | v97 | 2026-04-03 | **Complete** |
| p8_16_1 | tmp-s8 | 13,863,636 | v96 | 2026-04-01 | **Complete** |
| p9_16_1 | tmp-s9 | 13,862,549 | v97 | 2026-04-01 | **Complete** |
| p10_16_1 | tmp-s10 | 13,867,698 | v97 | 2026-04-03 | **Complete** |
| p11_16_1 | tmp-s11 | 13,897,600 | v97 | 2026-04-03 | **Complete** |
| p12_16_1 | tmp-s12 | 13,897,454 | v97 | 2026-04-01 | **Complete** |
| p13_16_1 | tmp-s5 (`dongguo-vibevoice-tmp-s5-2c12fb`) | 13,282,309 | v92 | 2026-04-06 | **Running** — Job `raysubmit_WvYucpFDRSe41Myq`, ~583K rows left |
| p14_16_1 | tmp-s14 | 13,899,430 | v97 | 2026-04-03 | **Complete** |
| p15_16_1 | tmp-s15 | 13,900,000 | v97 | 2026-04-01 | **Complete** |

**13/16 complete, 3 running (p2, p6, p13)** — as of 2026-04-06 14:58 UTC

## Active Jobs (2026-04-06)

| Job ID | Cluster | Partition | Port-forward |
|--------|---------|-----------|--------------|
| `raysubmit_RH7TRn1sZB9H2GR2` | tmp-s1 (dongguo-vibevoice-tmp-s1-13c202) | p6 | `kubectl port-forward -n flytesnacks-development svc/dongguo-vibevoice-tmp-s1-13c202-head-svc 18261:8265` |
| `raysubmit_piq44zQzedA9M2s6` | tmp-s2 (dongguo-vibevoice-tmp-s2-633ae8) | p2 | `kubectl port-forward -n flytesnacks-development svc/dongguo-vibevoice-tmp-s2-633ae8-head-svc 18262:8265` |
| `raysubmit_WvYucpFDRSe41Myq` | tmp-s5 (dongguo-vibevoice-tmp-s5-2c12fb) | p13 | `kubectl port-forward -n flytesnacks-development svc/dongguo-vibevoice-tmp-s5-2c12fb-head-svc 18264:8265` |

Check status:
```bash
cd /Users/dongguo/Projects/lumaverse/projects/lax && source .venv/bin/activate
RAY_ADDRESS=http://localhost:18261 ray job status raysubmit_RH7TRn1sZB9H2GR2  # p6
RAY_ADDRESS=http://localhost:18262 ray job status raysubmit_piq44zQzedA9M2s6  # p2
RAY_ADDRESS=http://localhost:18264 ray job status raysubmit_WvYucpFDRSe41Myq  # p13
```

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
