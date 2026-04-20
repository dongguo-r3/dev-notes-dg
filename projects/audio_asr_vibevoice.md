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

## 2026-04-18 — Invalid-row cleanup + backfill plan (25 tables)

### Goal

Delete rows where the VibeVoice output is missing/empty/invalid across all 25 output tables (9 SFT + 16 multilingual_v2), then re-run the pipeline with new hyperparameters to backfill.

**Delete predicate:** `num_tokens = 0 OR segments = '[]' OR truncated = true`

- `num_tokens = 0` — model returned nothing (likely vLLM batch failure / OOM)
- `segments = '[]'` — JSON parse failure or empty segment list
- `truncated = true` — output hit `max_new_tokens` cap; transcript incomplete

### Scripts

- Dry-run / delete tool: `projects/lax/lax/projects/av_data_processing/audio/audio_metadata/demos/delete_invalid_vibevoice_rows.py`
- Older narrower predecessor (only `num_tokens=0 OR segments='[]'`): `delete_empty_vibevoice_rows.py` (kept for history; don't edit)

### Dry-run results (2026-04-18)

Total across 25 tables: **287,456,922 rows; 860,206 deletable (0.299%)**.

| Family | Tables | Total rows | Deletable | Pct |
|---|---:|---:|---:|---:|
| SFT | 9 | 66,132,251 | 541,066 | 0.82% |
| multilingual_v2 | 16 | 221,324,671 | 319,140 | 0.14% |

Hot spots (>1%):

- `podcast_10m_p17to20_p1of3`: 123,712 / 7.53M (1.64%)
- `podcast_10m_p17to20_p2of3`: 118,807 / 7.53M (1.58%)
- `podcast_10m_p17to20_p3of3`: 102,155 / 7.62M (1.34%)
- `hours_140k_p3of3`: 72,960 / 7.25M (1.01%) — 10× worse than its siblings p1of3/p2of3 (~0.1%). Asymmetric across the 3 shards.

By predicate component across all 25: `num_tokens=0` ≈ 535k (dominant driver, heavily overlaps with `segments='[]'`), `segments='[]'` ≈ 724k, `truncated=true` ≈ 304k (fairly uniform across partitions, additive).

### Why soft-delete + rerun is safe (LAX pipeline behavior)

Verified by tracing `projects/lax/lax/core/data/lance/`:

1. `pipeline_vllm.py` sets `should_write_row_ids=True` → output carries `original_row_id` from source.
2. `ray_sources/base.py:314-322` — at each run, calls `create_rowid_mappings(..., reuse=False)` to build a fresh source→dest row-id diff mapping. `reuse=False` forces old cached mappings to be deleted first (`rowid_mappings.py:95`), so soft deletes are picked up.
3. `rowid_mappings_workers/build_mapping_per_bucket.py:122` — fresh build uses `frag.scanner(...)`, which **respects Lance deletion bitmaps**. Soft-deleted rows are excluded from the mapping.
4. `ray_sources/base.py:562-567` — `rowids = np.setdiff1d(source_rowids, dest_row_ids)`. Deleted rows are absent from `dest_row_ids` → survive the setdiff → get re-processed.

**End state after rerun:** old bad rows stay soft-deleted (invisible, reclaimable until `cleanup_old_versions()`); new good rows appended with same `original_row_id`; queries by `original_row_id` return the new row. No duplicates, no holes.

### Asymmetries to investigate before re-running

- `hours_140k_p1of3` / `p2of3` have 0 rows with `num_tokens=0` but ~2.2k with `segments='[]'`. Distinct failure mode: "model emitted tokens that failed JSON parse", not the OOM-dropout pattern. New hyperparameters should ideally address this too.
- `hours_140k_p3of3` is 10× worse than p1of3/p2of3 — different job run or config change mid-sequence. Check the specific job history before assuming the same rerun fixes it.
- `p17to20_*` all show ~115k `num_tokens=0` each — a single bad job likely affected all three.

### Stage-by-stage plan

| Stage | Action | Expected outcome |
|---|---|---|
| **0. Dry-run** ✅ 2026-04-18 | `python delete_invalid_vibevoice_rows.py` (no `--execute`) | 860,206 deletable rows across 25 tables (0.299%). No data touched. |
| **1. Execute deletion** | `python delete_invalid_vibevoice_rows.py --execute` → type `yes` | 25 tables get `ds.delete(...)` sequentially. Each version bumps +1 (e.g. multilingual_v2_p0 v99→v100). Live row counts drop by deletable numbers. Parquet untouched (soft delete). Runtime ~10–30 min. |
| **2. Post-delete verification** | Re-run dry-run | All tables report `Deletable=0`. Total live rows ≈ 286,596,716. Anything else = investigate. |
| **3. Update pipeline hyperparameters** | Edit `pipeline_vllm.py` for the failure modes (raise `max_new_tokens` for truncated; tune vLLM memory/batch for `num_tokens=0`; fix parser for `segments='[]'` with tokens emitted) | New code committed to a branch, ready to deploy. |
| **4. Backfill rerun** | Submit LAX jobs, source → same destination as before, one per table or grouped | Framework builds fresh `dest_rowid_mapping` (`reuse=False`), diffs source vs live dest → processes only the ~860k missing rows (not all 287M). New rows appended with same `original_row_id`s as soft-deleted ones. Runtime: ~few hours if parallelized across fleet at ~200ms/sample on 8 GPUs. |
| **5. Backfill verification** | Re-run dry-run a third time | `Total rows` back near 287.4M. `Deletable` much lower than 860k. Residual rate = genuinely hard samples (audio decode errors, corrupt bytes, extreme duration) the new hyperparameters couldn't fix. Target <0.05%. |
| **6. (Optional) Compaction** | `ds.compact_files()` per table | Physically drops soft-deleted rows, merges sparse backfill fragments. Reduces storage + improves scan speed. Only needed if read perf degrades. |

**Go/no-go gates:**

- Before Stage 1: hyperparameter change (Stage 3) is committed and you trust the rerun will produce better output than what's being deleted. If unsure, pilot on a single partition first.
- Before Stage 4: Stage 2 confirms zero residual deletable rows.

### Stage 3 detailed task list (run from a computer with omniva access)

Stages 1 and 2 completed on 2026-04-18 from the fsx compute host. 860,206 rows deleted across 25 tables; verification confirmed `Deletable=0` everywhere. Live row count: 286,596,716.

The hyperparameter edits in §3.1 below were also made on the fsx host on 2026-04-18 (on branch `dongguo/datasets`) but are not yet committed. The remaining tasks are: verify, commit, push, then discover clusters to prepare for Stage 4.

#### 3.1 Code changes already staged (uncommitted) on fsx host

Branch: `dongguo/datasets`
File: `projects/lax/lax/projects/av_data_processing/audio/asr_vibevoice/pipeline_vllm.py`

| Line | Param | Before | After | Purpose |
|---|---|---|---|---|
| 143 | `VibeVoiceASRVLLMActor.max_new_tokens` default | 256 | 512 | 2× transcript budget — addresses 304k `truncated=true` rows |
| 474 | Ray `map_batches` `batch_size` | 64 | 16 | Smaller per-actor working set — reduces vLLM OOM (addresses 535k `num_tokens=0`) |
| 499 | `_BatchProcessor.max_new_tokens` default | 256 | 512 | Consistency with factory default |
| 668 | `run_vibevoice_asr_vllm_pipeline.max_new_tokens` default | 256 | 512 | Production factory default |
| 703 | `read_control_row_based_batch_size` | 256 | 128 | Reduces head object-store pressure (still 4× below the historical 2048 OOM threshold) |

Rationale recap: the 860k deleted rows split as ~535k `num_tokens=0` (vLLM returned nothing — suspected OOM) + ~304k `truncated=true` (hit 256-token cap) + ~189k JSON parse failures. The edits target the first two; the parser issue is not addressed in this round.

#### 3.2 Prereqs on your Mac

```bash
cd /Users/dongguo/Projects/lumaverse/projects/kuma && source .venv/bin/activate
flytecli activate --cluster omniva-flyte   # opens browser for Teleport
```

Sanity-check `lax/projects/av_data_processing/audio/asr_vibevoice/runtime_env_local.json` contains a real `HF_TOKEN` (not the `${HF_TOKEN}` placeholder from the committed `runtime_env.json`).

#### 3.3 Sync the uncommitted edits to your Mac

The fsx host has the edits in its working tree but not pushed. Two options:

**Option A — re-apply manually from §3.1** (safest, 5 one-line changes).

**Option B — transfer the working tree.** If you have ssh to the fsx host:

```bash
# From Mac
scp fsx:/fsx/dongguo/Projects/lumaverse/projects/lax/lax/projects/av_data_processing/audio/asr_vibevoice/pipeline_vllm.py \
    /Users/dongguo/Projects/lumaverse/projects/lax/lax/projects/av_data_processing/audio/asr_vibevoice/pipeline_vllm.py
```

Then verify:

```bash
cd /Users/dongguo/Projects/lumaverse
git diff projects/lax/lax/projects/av_data_processing/audio/asr_vibevoice/pipeline_vllm.py
```

Expected diff: exactly 5 single-line changes matching §3.1. No whitespace churn, no other edits.

#### 3.4 Optional smoke test (recommended before launching 25 jobs)

Debug-mode single-process run — no Ray, no cluster needed, just confirms the new config runs end-to-end:

```bash
cd /Users/dongguo/Projects/lumaverse/projects/lax && source .venv/bin/activate
python -m lax.scripts.lax_processing.run_data_debug \
    -s "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    -p "lax.projects.av_data_processing.audio.asr_vibevoice.pipeline_vllm.run_vibevoice_asr_vllm_pipeline" \
    -n 5 --print_output
```

Expected: 5 rows of non-empty `segments` + `raw_text`, `num_tokens` values in a reasonable range (< 512). If any row shows `truncated=true` or `num_tokens=0`, pause and investigate before Stage 4.

Alternatively, pilot on one small partition via the full submit path before the full rollout — pick a single SFT table like `convspeech_vibevoice_asr` (smallest at 6.5M rows; ~35k to backfill) and run the Stage 4 submit command once. Verify the output before batch-launching the other 24.

#### 3.5 Commit + push

```bash
cd /Users/dongguo/Projects/lumaverse
git add projects/lax/lax/projects/av_data_processing/audio/asr_vibevoice/pipeline_vllm.py
git commit -m "[lax vibevoice] Tune batch sizes + max_new_tokens for backfill rerun

- max_new_tokens: 256 -> 512 (addresses 304k truncated=true rows)
- map_batches batch_size: 64 -> 16 (reduces vLLM OOM, addresses 535k num_tokens=0)
- read_control_row_based_batch_size: 256 -> 128 (lighter head object-store pressure)

Backfills 860,206 rows soft-deleted 2026-04-18 across 25 output tables
(9 SFT + 16 multilingual_v2)."
git push origin dongguo/datasets
```

#### 3.6 Discover ready + idle omniva Ray clusters

Per notes §2026-04-06, idle clusters s0, s3, s6, s8, s9, s15 were deleted; tmp-s1/s2/s5 were used for resume jobs and may or may not still exist. Inventory first:

```bash
flytecli ray-cluster --cluster omniva-flyte --list
```

For each listed cluster, check state + running jobs:

```bash
flytecli ray-cluster --cluster omniva-flyte --name <cr-name> --status
```

Target state: cluster is `Ready` (phase) and job list is empty (zero SUBMITTED / RUNNING / PENDING).

If no ready+idle clusters exist, create new single-node ones as needed (§ "Creating Ray Clusters" in `run_podcast10m_partitions.md`):

```bash
flytecli ray-cluster --cluster omniva-flyte --name dongguo-audio-omniva-sN --nodes 1
```

#### 3.7 Rerun targets — per-table backfill manifest

25 tables, 860,206 rows total to regenerate. "Rows to regenerate" = exact count of rows soft-deleted in Stage 1 (matches dry-run deletable count). Each LAX job runs against `--source_uri` (source), writes to `--destination_uri` (destination), and re-processes only the missing `original_row_id`s via the framework's row-id diff (see §"Why soft-delete + rerun is safe" above). `--partitions_range` values below follow the observed naming conventions (`p{N}of3` → `N-1,3,1`, `p{N}_16_1` → `N,16,1`) — **verify against original launch commands before running**, since p0of3-style partitioning conventions aren't in the dev-notes.

**Family 1 — SFT (9 tables, 541,066 rows to regenerate)**

| Destination | Source | `--partitions_range` | Rows to regenerate |
|---|---|---|---:|
| `…/sft/hours_140k_vibevoice_asr_p1of3.lance` | `…-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance` | `0,3,1` | 7,494 |
| `…/sft/hours_140k_vibevoice_asr_p2of3.lance` | (same) | `1,3,1` | 7,381 |
| `…/sft/hours_140k_vibevoice_asr_p3of3.lance` | (same) | `2,3,1` | 72,960 |
| `…/sft/convspeech_vibevoice_asr.lance` | `…-lance/audio/sft/convspeech/asr/prefiltered_english__whisperx.lance` | (none — full) | 35,207 |
| `…/sft/podcast_10m_p11to14_vibevoice_asr.lance` | `…-lance/audio/sft/podcast_10m/asr/podcast_10m_p11to14_whisperx_clean.lance` | (none — full) | 36,155 |
| `…/sft/podcast_10m_p14to17_vibevoice_asr.lance` | `…-lance/audio/sft/podcast_10m/asr/podcast_10m_p14to17_whisperx_clean.lance` | (none — full) | 37,195 |
| `…/sft/podcast_10m_p17to20_vibevoice_asr_p1of3.lance` | `…-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance` | `0,3,1` | 123,712 |
| `…/sft/podcast_10m_p17to20_vibevoice_asr_p2of3.lance` | (same) | `1,3,1` | 118,807 |
| `…/sft/podcast_10m_p17to20_vibevoice_asr_p3of3.lance` | (same) | `2,3,1` | 102,155 |

Destination prefix for all SFT: `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/`
Source prefix: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/…`

**Family 2 — multilingual_v2 (16 tables, 319,140 rows to regenerate)**

All share source `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance`.
Destination prefix: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p{SID}_16_1.lance`.

| SID | `--partitions_range` | Rows to regenerate |
|---:|---|---:|
| 0 | `0,16,1` | 16,920 |
| 1 | `1,16,1` | 17,106 |
| 2 | `2,16,1` | 16,407 |
| 3 | `3,16,1` | 15,994 |
| 4 | `4,16,1` | 16,193 |
| 5 | `5,16,1` | 15,667 |
| 6 | `6,16,1` | 16,100 |
| 7 | `7,16,1` | 15,782 |
| 8 | `8,16,1` | 16,293 |
| 9 | `9,16,1` | 17,483 |
| 10 | `10,16,1` | 21,282 |
| 11 | `11,16,1` | 17,725 |
| 12 | `12,16,1` | 17,436 |
| 13 | `13,16,1` | 35,330 |
| 14 | `14,16,1` | 31,288 |
| 15 | `15,16,1` | 32,134 |

**Hot-spot tables (>1% deletable)** — worth piloting these first to confirm the new hyperparameters actually address the failure mode before launching the others:

1. `podcast_10m_p17to20_vibevoice_asr_p1of3` — 123,712 rows (1.64%)
2. `podcast_10m_p17to20_vibevoice_asr_p2of3` — 118,807 rows (1.58%)
3. `podcast_10m_p17to20_vibevoice_asr_p3of3` — 102,155 rows (1.34%)
4. `hours_140k_vibevoice_asr_p3of3` — 72,960 rows (1.01%)

#### 3.8 Handoff to Stage 4

After §3.6, list the ready+idle cluster names + their CR names (the `dongguo-...-xxxxxx` format). Stage 4 will need one cluster per concurrent partition job — the 25 backfill jobs can be sequenced through fewer clusters if parallelism isn't critical (total work is ~860k rows, vs. 287M for the original runs, so each cluster finishes much faster than before).

## 2026-04-19 — Backfill result audit (Stage 5 verification)

Re-ran the delete predicate (`num_tokens = 0 OR segments = '[]' OR truncated = true`)
as a read-only count across the same 25 tables to measure how much the backfill
actually reduced invalid rows.

### Headline

**Total invalid rows dropped from 860,206 → 601,894 (−30.0%).**
Invalid fraction: **0.299% → 0.207%**.

The new hyperparameters (`max_new_tokens 256→512`, vLLM `batch_size 64→16`,
`read_control_row_based_batch_size 256→128`) recovered most of the
truncation-driven failures and a good fraction of the vLLM-OOM failures, but
several tables plateau — the residual looks like a qualitatively different
failure mode that this hyperparameter round did not address.

### Family-level comparison

| Family | Before: rows | Before: invalid | Before % | After: rows | After: invalid | After % | Δ invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT (9 tables) | 66,132,251 | 541,066 | 0.818% | 68,417,366 | 396,859 | 0.580% | **−144,207 (−26.6%)** |
| multilingual_v2 (16 tables) | 221,324,671 | 319,140 | 0.144% | 221,838,460 | 205,035 | 0.092% | **−114,105 (−35.8%)** |
| **TOTAL** | **287,456,922** | **860,206** | **0.299%** | **290,255,826** | **601,894** | **0.207%** | **−258,312 (−30.0%)** |

### Per-table detail (all 25)

Rows and invalid counts as observed at the table's current Lance version; `Δ`
is `after − before`, so negative = improvement.

| Table | Rows (before) | Rows (after) | Invalid before | Invalid after | Δ invalid | % reduction |
|---|---:|---:|---:|---:|---:|---:|
| hours_140k_p1of3 | 7,275,309 | 7,275,295 | 7,494 | 1,835 | −5,659 | −75.5% |
| hours_140k_p2of3 | 7,235,639 | 7,235,630 | 7,381 | 2,638 | −4,743 | −64.3% |
| hours_140k_p3of3 | 7,254,118 | 7,251,238 | 72,960 | 1,901 | −71,059 | **−97.4%** |
| convspeech | 6,514,097 | 6,514,097 | 35,207 | 25,759 | −9,448 | −26.8% |
| podcast_p11to14 | 7,499,644 | 9,756,046 | 36,155 | 3,838 | −32,317 | **−89.4%** |
| podcast_p14to17 | 7,670,431 | 7,670,431 | 37,195 | 32,933 | −4,262 | −11.5% |
| podcast_p17to20_p1of3 | 7,533,353 | 7,546,490 | 123,712 | 114,602 | −9,110 | −7.4% |
| podcast_p17to20_p2of3 | 7,534,293 | 7,544,848 | 118,807 | 112,253 | −6,554 | −5.5% |
| podcast_p17to20_p3of3 | 7,615,367 | 7,623,291 | 102,155 | 101,100 | −1,055 | −1.0% |
| multilingual_v2_p0 | 14,000,000 | 14,016,920 | 16,920 | 17,768 | **+848** | **+5.0%** |
| multilingual_v2_p1 | 14,000,000 | 14,000,000 | 17,106 | 16,953 | −153 | −0.9% |
| multilingual_v2_p2 | 13,990,495 | 13,990,495 | 16,407 | 16,081 | −326 | −2.0% |
| multilingual_v2_p3 | 13,898,108 | 13,898,108 | 15,994 | 8,646 | −7,348 | −45.9% |
| multilingual_v2_p4 | 13,897,974 | 13,897,974 | 16,193 | 15,055 | −1,138 | −7.0% |
| multilingual_v2_p5 | 13,337,105 | 13,321,438 | 15,667 | 0 | −15,667 | **−100.0%** |
| multilingual_v2_p6 | 13,860,492 | 13,860,492 | 16,100 | 8,619 | −7,481 | −46.5% |
| multilingual_v2_p7 | 13,869,821 | 13,869,821 | 15,782 | 8,513 | −7,269 | −46.1% |
| multilingual_v2_p8 | 13,863,636 | 13,863,636 | 16,293 | 8,716 | −7,577 | −46.5% |
| multilingual_v2_p9 | 13,862,549 | 13,862,549 | 17,483 | 15,603 | −1,880 | −10.8% |
| multilingual_v2_p10 | 13,867,698 | 13,862,580 | 21,282 | 5,970 | −15,312 | −71.9% |
| multilingual_v2_p11 | 13,897,600 | 13,897,600 | 17,725 | 17,725 | **0** | **0.0%** |
| multilingual_v2_p12 | 13,897,454 | 13,897,454 | 17,436 | 10,025 | −7,411 | −42.5% |
| multilingual_v2_p13 | 13,282,309 | 13,799,963 | 35,330 | 16,655 | −18,675 | −52.9% |
| multilingual_v2_p14 | 13,899,430 | 13,899,430 | 31,288 | 16,891 | −14,397 | −46.0% |
| multilingual_v2_p15 | 13,900,000 | 13,900,000 | 32,134 | 21,815 | −10,319 | −32.1% |

### What worked (clear wins)

- **hours_140k_p3of3** — 72,960 → 1,901 (−97.4%). The anomalous 10× hot-spot
  among the three shards is gone; it now matches its siblings' residual rate.
  The original failure mode was consistent with vLLM OOM (`num_tokens=0`),
  which the `batch_size 64→16` change addresses directly.
- **multilingual_v2_p5** — 15,667 → **0** (100% clean).
- **multilingual_v2_p3 / p6 / p7 / p8** — uniformly ~46% reduction each,
  consistent with a shared upstream failure mode that the new hyperparameters
  fully address for the rerun subset but leaves a residual from the original
  run that didn't get re-processed.
- **multilingual_v2_p10 / p13 / p14** — 32–72% reduction each on the larger
  deletable sets.
- **podcast_p11to14** — 36,155 → 3,838 (−89%). Note row count **grew from 7.50M
  to 9.76M** (+2.26M); this is out of scope for the backfill, unrelated append
  — investigate separately (suspicious: only 36,155 were supposed to regenerate).

### What barely moved (stubborn residuals)

- **podcast_p17to20_p{1,2,3}of3** — still 1.3–1.5% invalid (114k / 112k / 101k).
  Reduction of only ~1–7% per part despite full rerun. The "wild" podcast
  split's failures are not OOM/truncation — likely malformed audio bytes,
  extreme durations, or non-speech content that the model genuinely can't
  transcribe. Needs a different intervention (e.g., pre-filter on audio
  validity, or lower the `truncated=true` bar).
- **multilingual_v2_p11** — 17,725 → 17,725 (**exact same count**). Strongly
  suggests this partition was not part of the rerun, or the rerun reproduced
  the identical failure set. Worth checking the job history for p11 specifically.
- **multilingual_v2_p0** — 16,920 → 17,768 (**slightly worse, +848**). The
  backfill introduced new failures. Row count grew by exactly 16,920 (matching
  the pre-delete deletable), then +848 of those rerun outputs were themselves
  invalid — the rerun's new-hyperparameter failure rate on the hardest residual
  subset is ~5% of that subset.
- **convspeech**, **podcast_p14to17**, **multilingual_v2_p1 / p2 / p4 / p9** —
  all < 15% reduction. Low-hanging fruit is exhausted.

### Anomalies to chase later

1. **podcast_p11to14 row-count jump (+2,256,402)**. Only 36,155 rows were
   flagged for regeneration, yet the table grew by 2.26M. Could indicate a
   separate append, a rerun that targeted a wider partition range than
   intended, or a double-write. Low priority (invalid rate dropped 89%), but
   worth verifying nothing duplicated.
2. **multilingual_v2_p11 identical counts**. Zero change before/after — check
   whether the p11 backfill job was actually submitted and succeeded, or
   whether it was skipped.
3. **multilingual_v2_p0 regression**. Only table where invalid count went up.
   Confirms that even with the new hyperparameters, a small fraction of
   already-flagged rows remain unparseable on rerun.

### Next steps (optional)

- **Targeted third-pass** on the 4 stubborn SFT tables and p11/p15 if downstream
  consumers need <0.5% invalid everywhere. Needs a different knob than the
  previous round (pre-filter on audio duration / byte length, parser-tolerant
  path for JSON failures).
- **Compaction pass** (`ds.compact_files()`) per table, if read performance is
  degrading from the mix of soft-deleted rows + sparse backfill fragments.
- **Escalate to deletion** for the truly unrecoverable ~600k rows — they're a
  constant 0.2% tax on all downstream consumers otherwise.

---

## 4. Transcript Cross-Validation Pipeline (WhisperX vs VibVoice)

Date: 2026-04-17 — 2026-04-19

### 4.1 Motivation

For TTS training, we need transcripts that faithfully represent the spoken content, including filler words. VibVoice is preferred (preserves fillers, matches phonemes), but it's an AR LLM that can hallucinate or produce truncated/repetitive output. WhisperX serves as an independent cross-check for Latin-script languages where it is reliable.

### 4.2 Pipeline Design

The pipeline selects one transcript per audio segment and outputs it in WhisperX format (`[SPEAKER_00]"text"`). Non-speech and low-quality transcripts are rejected with `<no_transcript>`.

**Class-1 gates (all languages):**

| Gate | Condition | Catches |
|---|---|---|
| 1 | VV transcript empty | vLLM inference failures (OOM, GPU crash) |
| 2a | VV truncated (`num_tokens=256`) | Incomplete transcripts from AR token limit |
| 2b | VV LLM repetition | Pathological AR loops (≥20 consecutive repeats, ≥50% of text, WX cross-check fails) |
| 2c | VV `[Lyric]`/`[Music]` tag | Non-speech content |
| 3 | Either version ≥3 speakers | Too complex for TTS |

**Latin language cross-check (Gates 4–6):**

| Gate | Condition | Threshold |
|---|---|---|
| 4 | WX transcript empty | Reject |
| 5 | Length ratio (after dedup + nofiller) | `len_ratio_nf < 0.7` → reject |
| 6 | WER + char_sim (2 tiers) | Tier a: `wer_nf ≤ 0.20` → accept |
| | | Tier b: `wer_nf ≤ 0.25` AND `char_sim_nf > 0.90` → accept (phonetic rescue) |

**Non-Latin languages:** Auto-accept VibVoice after class-1 gates (WhisperX unreliable for CJK/Arabic/etc.).

**Version selection (when accepted):**
- Near-identical (`wer_nf < 0.02`): prefer version with more speakers; default VibVoice
- Phonetic rescue: VibVoice
- Speaker count differs: version with more speakers
- Default: VibVoice (filler preservation)

### 4.3 Threshold Calibration

Manually inspected 50 transcript pairs at each (WER, char_sim) boundary:

| Config | Risk | Problematic % | Key Pattern |
|---|---|---|---|
| wer~0.25, cs~0.90 | LOW-MEDIUM | 16% | Mostly fillers/stutters — safest rescue band |
| wer~0.25, cs~0.85 | MEDIUM | 36% | Numbers dropped by WX; some real word errors |
| wer~0.30, cs~0.85 | MEDIUM-HIGH | 56% | Wrong words, proper nouns, meaning changes |
| wer~0.30, cs~0.80 | HIGH | 80% | Nearly all pairs have genuine content errors |
| wer~0.35, cs~0.85 | HIGH | 60% | Wrong names/phrases, numbers lost |

**Conclusion:** `char_sim` is a stronger safety signal than WER alone. The `wer ≤ 0.25 + char_sim > 0.90` rescue band has only ~16% problematic pairs, while `wer ≤ 0.30 + char_sim > 0.85` jumps to 56%.

### 4.4 Results on Podcast 10M Sample (~500K rows)

Latin acceptance: ~89.7% (86.4% Tier a + 3.3% phonetic rescue). Non-Latin: ~95%+ (class-1 gates only).

### 4.5 Production Deployment — 6 Families, 25 Tables

**Full table manifest with S3 paths:**

| Family | WhisperX Source | VibVoice Table(s) | Output Table |
|---|---|---|---|
| **hours_140k** | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p1of3.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_transcript_cv.lance` |
| | (same) | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p2of3.lance` | (same) |
| | (same) | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p3of3.lance` | (same) |
| **convspeech** | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/convspeech/asr/prefiltered_english__whisperx.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/convspeech_vibevoice_asr.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/convspeech_transcript_cv.lance` |
| **podcast_p11to14** | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p11to14_whisperx_clean.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p11to14_vibevoice_asr.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_p11to14_transcript_cv.lance` |
| **podcast_p14to17** | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p14to17_whisperx_clean.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p14to17_vibevoice_asr.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_p14to17_transcript_cv.lance` |
| **podcast_p17to20** | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p1of3.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_p17to20_transcript_cv.lance` |
| | (same) | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p2of3.lance` | (same) |
| | (same) | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p3of3.lance` | (same) |
| **multilingual_v2** | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p{0-15}_16_1.lance` (16 partitions) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cross_validation.lance` |

**Cluster assignment (omniva-flyte):**

| Cluster | Family | Jobs | ~Rows |
|---|---|---|---|
| `metadata-s0` | multilingual_v2 (16 partitions) | 16 | 220M |
| `metadata-s1` | hours_140k (3 partitions) | 3 | 21M |
| `metadata-s2` | podcast_p17to20 (3 partitions) | 3 | 23M |
| `metadata-s3` | podcast_p11to14 | 1 | 10M |
| `metadata-s4` | podcast_p14to17 | 1 | 8M |
| `metadata-s5` | convspeech | 1 | 7M |

**Execution mode:** Each partition submitted as a Ray job with `--append-to-lance`. Jobs within a family append to the same output Lance table. After all jobs complete, gap-filling adds `<no_transcript>` for WhisperX rows without VibVoice coverage.

**Output schema:** `original_row_id` (uint64), `transcript` (string), `choice` (string), `reason` (string). Joins back to the WhisperX source table via `original_row_id == _rowid`.

### 4.6 Code Locations

| File | Purpose |
|---|---|
| `lax/projects/av_data_processing/audio/audio_metadata/demos/transcript_cross_validation.py` | Main script — all logic + batch processing |
| `lax/projects/av_data_processing/audio/audio_metadata/demos/launch_transcript_cv.sh` | Shell helper to submit Ray jobs per family |
| `lax/projects/av_data_processing/audio/audio_metadata/demos/runtime_env_transcript_cv.json` | Minimal pip deps |
| `vibevoice_analysis_v4/compare_sft_asr.ipynb` | Development notebook with analysis and threshold calibration |

### 4.7 Interactive Report

HTML report with audio demos at each filtering stage:
`s3://ai-lumalabs-dashboard-samples-ap-se-2/dongguo/html-viewer/transcript_selection_whisperx_vibevoice_cross_validation_pipeline.html`

### 4.8 Launch Scripts

**Submit convspeech (metadata-s5):**
```bash
cd /Users/dongguo/Projects/lumaverse/projects/lax && source .venv/bin/activate
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s5-d38cd4

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env lax/projects/av_data_processing/audio/audio_metadata/demos/runtime_env_transcript_cv.json \
    --script lax.projects.av_data_processing.audio.audio_metadata.demos.transcript_cross_validation \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/convspeech/asr/prefiltered_english__whisperx.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/convspeech_vibevoice_asr.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/convspeech_transcript_cv.lance" \
    --append-to-lance --batch-size 10000
```

**Submit podcast_p11to14 (metadata-s3):**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s3-c857b1

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env lax/projects/av_data_processing/audio/audio_metadata/demos/runtime_env_transcript_cv.json \
    --script lax.projects.av_data_processing.audio.audio_metadata.demos.transcript_cross_validation \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p11to14_whisperx_clean.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p11to14_vibevoice_asr.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_p11to14_transcript_cv.lance" \
    --append-to-lance --batch-size 10000
```

**Submit podcast_p14to17 (metadata-s4):**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s4-720f5b

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env lax/projects/av_data_processing/audio/audio_metadata/demos/runtime_env_transcript_cv.json \
    --script lax.projects.av_data_processing.audio.audio_metadata.demos.transcript_cross_validation \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p14to17_whisperx_clean.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p14to17_vibevoice_asr.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_p14to17_transcript_cv.lance" \
    --append-to-lance --batch-size 10000
```

### 4.9 Job Tracking — Pilot (2026-04-19)

First 3 jobs launched to validate the pipeline:

| Family | Cluster | Job ID | Status |
|---|---|---|---|
| convspeech | `metadata-s5` | `raysubmit_RgQVv1pQbSz1HmV3` | Submitted 02:54 UTC |
| podcast_p11to14 | `metadata-s3` | `raysubmit_SwCfrM3UuPaJwpsx` | Submitted 02:54 UTC |
| podcast_p14to17 | `metadata-s4` | `raysubmit_ryqGR1ALEdtYS6Uc` | Submitted 02:58 UTC |

Note: first 3 pilot jobs failed due to `lance.write_dataset` not available on the cluster (runtime env installed wrong package). Fixed by using empty pip list (cluster base image already has pylance). Resubmitted:

| Family | Cluster | Job ID | Status |
|---|---|---|---|
| convspeech | `metadata-s5` | `raysubmit_vKHAg5matEv2tn3p` | Running (~760 rows/s) |
| podcast_p11to14 | `metadata-s3` | `raysubmit_7nHFigpHqdpwwqBs` | Running |
| podcast_p14to17 | `metadata-s4` | `raysubmit_ybWSbnTkBm3fPf2z` | Running |

### 4.10 Per-Partition Output Tables

Each VibVoice partition writes to its own output Lance table to avoid concurrent writer contention. After all jobs complete, merge per-partition outputs into one table per family.

**Complete manifest — 25 VibVoice tables → 25 output tables:**

| # | VibVoice Input Table | WhisperX Source Table | Output Table | Cluster |
|---|---|---|---|---|
| 1 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p1of3.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_hours_140k_p1of3.lance` | `metadata-s1` |
| 2 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p2of3.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_hours_140k_p2of3.lance` | `metadata-s1` |
| 3 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p3of3.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_hours_140k_p3of3.lance` | `metadata-s1` |
| 4 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/convspeech_vibevoice_asr.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/convspeech/asr/prefiltered_english__whisperx.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/convspeech_transcript_cv.lance` | `metadata-s5` |
| 5 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p11to14_vibevoice_asr.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p11to14_whisperx_clean.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_p11to14_transcript_cv.lance` | `metadata-s3` |
| 6 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p14to17_vibevoice_asr.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p14to17_whisperx_clean.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_p14to17_transcript_cv.lance` | `metadata-s4` |
| 7 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p1of3.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_podcast_p17to20_p1of3.lance` | `metadata-s2` |
| 8 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p2of3.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_podcast_p17to20_p2of3.lance` | `metadata-s2` |
| 9 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p3of3.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance` | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_podcast_p17to20_p3of3.lance` | `metadata-s2` |
| 10 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p0_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p0.lance` | `metadata-s0` |
| 11 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p1_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p1.lance` | `metadata-s0` |
| 12 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p2_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p2.lance` | `metadata-s0` |
| 13 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p3_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p3.lance` | `metadata-s0` |
| 14 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p4_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p4.lance` | `metadata-s0` |
| 15 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p5_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p5.lance` | `metadata-s0` |
| 16 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p6_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p6.lance` | `metadata-s0` |
| 17 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p7_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p7.lance` | `metadata-s0` |
| 18 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p8_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p8.lance` | `metadata-s0` |
| 19 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p9_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p9.lance` | `metadata-s0` |
| 20 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p10_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p10.lance` | `metadata-s0` |
| 21 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p11_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p11.lance` | `metadata-s0` |
| 22 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p12_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p12.lance` | `metadata-s0` |
| 23 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p13_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p13.lance` | `metadata-s0` |
| 24 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p14_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p14.lance` | `metadata-s0` |
| 25 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p15_16_1.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p15.lance` | `metadata-s0` |

### 4.11 Submit Scripts (22 remaining tables)

Prerequisite: `cd /Users/dongguo/Projects/lumaverse/projects/lax && source .venv/bin/activate && flytecli activate --cluster omniva-flyte`

Common variables:
```bash
RUNTIME_ENV="lax/projects/av_data_processing/audio/audio_metadata/demos/runtime_env_transcript_cv.json"
SCRIPT="lax.projects.av_data_processing.audio.audio_metadata.demos.transcript_cross_validation"
```

**Table 1: hours_140k_p1of3 (metadata-s1)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s1-fed951

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p1of3.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_hours_140k_p1of3.lance" \
    --append-to-lance --batch-size 10000
```

**Table 2: hours_140k_p2of3 (metadata-s1)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s1-fed951

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p2of3.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_hours_140k_p2of3.lance" \
    --append-to-lance --batch-size 10000
```

**Table 3: hours_140k_p3of3 (metadata-s1)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s1-fed951

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p3of3.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_hours_140k_p3of3.lance" \
    --append-to-lance --batch-size 10000
```

**Table 7: podcast_p17to20_p1of3 (metadata-s2)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s2-de5369

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p1of3.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_podcast_p17to20_p1of3.lance" \
    --append-to-lance --batch-size 10000
```

**Table 8: podcast_p17to20_p2of3 (metadata-s2)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s2-de5369

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p2of3.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_podcast_p17to20_p2of3.lance" \
    --append-to-lance --batch-size 10000
```

**Table 9: podcast_p17to20_p3of3 (metadata-s2)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s2-de5369

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p3of3.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_podcast_p17to20_p3of3.lance" \
    --append-to-lance --batch-size 10000
```

**Table 10: multilingual_v2_p0 (metadata-s0)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s0-6a0225

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p0_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p0.lance" \
    --append-to-lance --batch-size 10000
```

**Table 11: multilingual_v2_p1 (metadata-s0)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s0-6a0225

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p1_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p1.lance" \
    --append-to-lance --batch-size 10000
```

**Table 12: multilingual_v2_p2 (metadata-s0)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s0-6a0225

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p2_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p2.lance" \
    --append-to-lance --batch-size 10000
```

**Table 13: multilingual_v2_p3 (metadata-s0)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s0-6a0225

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p3_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p3.lance" \
    --append-to-lance --batch-size 10000
```

**Table 14: multilingual_v2_p4 (metadata-s0)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s0-6a0225

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p4_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p4.lance" \
    --append-to-lance --batch-size 10000
```

**Table 15: multilingual_v2_p5 (metadata-s0)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s0-6a0225

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p5_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p5.lance" \
    --append-to-lance --batch-size 10000
```

**Table 16: multilingual_v2_p6 (metadata-s0)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s0-6a0225

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p6_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p6.lance" \
    --append-to-lance --batch-size 10000
```

**Table 17: multilingual_v2_p7 (metadata-s0)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s0-6a0225

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p7_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p7.lance" \
    --append-to-lance --batch-size 10000
```

**Table 18: multilingual_v2_p8 (metadata-s6)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s6-30970a

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p8_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p8.lance" \
    --append-to-lance --batch-size 10000
```

**Table 19: multilingual_v2_p9 (metadata-s6)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s6-30970a

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p9_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p9.lance" \
    --append-to-lance --batch-size 10000
```

**Table 20: multilingual_v2_p10 (metadata-s6)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s6-30970a

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p10_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p10.lance" \
    --append-to-lance --batch-size 10000
```

**Table 21: multilingual_v2_p11 (metadata-s6)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s6-30970a

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p11_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p11.lance" \
    --append-to-lance --batch-size 10000
```

**Table 22: multilingual_v2_p12 (metadata-s7)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s7-ffd6fc

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p12_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p12.lance" \
    --append-to-lance --batch-size 10000
```

**Table 23: multilingual_v2_p13 (metadata-s7)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s7-ffd6fc

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p13_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p13.lance" \
    --append-to-lance --batch-size 10000
```

**Table 24: multilingual_v2_p14 (metadata-s7)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s7-ffd6fc

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p14_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p14.lance" \
    --append-to-lance --batch-size 10000
```

**Table 25: multilingual_v2_p15 (metadata-s7)**
```bash
source scripts/setup-ray-proxy.sh omniva-flyte dongguo-metadata-s7-ffd6fc

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --script "${SCRIPT}" \
    -- \
    --wx-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance" \
    --vv-uri-template "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/vibevoice_multilingual_v2_p15_16_1.lance" \
    --vv-partitions 0 \
    --output-uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p15.lance" \
    --append-to-lance --batch-size 10000
```

**Cluster assignment summary (16 multilingual partitions spread across 4 clusters):**

| Cluster | Tables |
|---|---|
| `metadata-s0` (`dongguo-metadata-s0-6a0225`) | multilingual_v2_p0 through p7 (8 jobs) |
| `metadata-s1` (`dongguo-metadata-s1-fed951`) | hours_140k_p1of3, p2of3, p3of3 (3 jobs) |
| `metadata-s2` (`dongguo-metadata-s2-de5369`) | podcast_p17to20_p1of3, p2of3, p3of3 (3 jobs) |
| `metadata-s3` (`dongguo-metadata-s3-c857b1`) | podcast_p11to14 (1 job) |
| `metadata-s4` (`dongguo-metadata-s4-720f5b`) | podcast_p14to17 (1 job) |
| `metadata-s5` (`dongguo-metadata-s5-d38cd4`) | convspeech (1 job) |
| `metadata-s6` (`dongguo-metadata-s6-30970a`) | multilingual_v2_p8 through p11 (4 jobs) |
| `metadata-s7` (`dongguo-metadata-s7-ffd6fc`) | multilingual_v2_p12 through p15 (4 jobs) |

### 4.12 Output Lance Tables — Full Paths

25 output tables containing cross-validated transcripts. Each row has: `original_row_id` (uint64, join key to the WhisperX source table), `transcript` (string, WhisperX format or `<no_transcript>`), `choice` (string), `reason` (string).

**SFT Family — hours_140k** (WhisperX source: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance`)

| # | Output Table |
|---|---|
| 1 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_hours_140k_p1of3.lance` |
| 2 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_hours_140k_p2of3.lance` |
| 3 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_hours_140k_p3of3.lance` |

**SFT Family — convspeech** (WhisperX source: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/convspeech/asr/prefiltered_english__whisperx.lance`)

| # | Output Table |
|---|---|
| 4 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/convspeech_transcript_cv.lance` |

**SFT Family — podcast_p11to14** (WhisperX source: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p11to14_whisperx_clean.lance`)

| # | Output Table |
|---|---|
| 5 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_p11to14_transcript_cv.lance` |

**SFT Family — podcast_p14to17** (WhisperX source: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p14to17_whisperx_clean.lance`)

| # | Output Table |
|---|---|
| 6 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_p14to17_transcript_cv.lance` |

**SFT Family — podcast_p17to20** (WhisperX source: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance`)

| # | Output Table |
|---|---|
| 7 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_podcast_p17to20_p1of3.lance` |
| 8 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_podcast_p17to20_p2of3.lance` |
| 9 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/transcript_cv_podcast_p17to20_p3of3.lance` |

**Pretrain Family — multilingual_v2** (WhisperX source: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance`)

| # | Output Table |
|---|---|
| 10 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p0.lance` |
| 11 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p1.lance` |
| 12 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p2.lance` |
| 13 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p3.lance` |
| 14 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p4.lance` |
| 15 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p5.lance` |
| 16 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p6.lance` |
| 17 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p7.lance` |
| 18 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p8.lance` |
| 19 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p9.lance` |
| 20 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p10.lance` |
| 21 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p11.lance` |
| 22 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p12.lance` |
| 23 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p13.lance` |
| 24 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p14.lance` |
| 25 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p15.lance` |

**How to use these tables:**

```python
import lance

# Read an output table
ds = lance.dataset("s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/transcript_cv_multilingual_v2_p0.lance")

# Get accepted transcripts
accepted = ds.to_table(filter="choice != 'reject'")

# Join back to WhisperX source table for audio bytes
wx = lance.dataset("s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance")
oids = accepted.column("original_row_id").to_pylist()
audio = wx._take_rows(oids, columns=["audio_bytes", "language", "segment_duration"])
```

**Output schema:**

| Column | Type | Description |
|---|---|---|
| `original_row_id` | uint64 | Join key to the WhisperX source table (`_rowid`) |
| `transcript` | string | WhisperX format (`[SPEAKER_00]"text"`) or `<no_transcript>` |
| `choice` | string | `vibevoice`, `whisperx`, `either`, or `reject` |
| `reason` | string | Human-readable explanation of the decision |

### 4.13 Job Completion Status (2026-04-20)

All 25 jobs completed. Row counts verified against source VibVoice tables:

| # | Table | Output Rows | Versions | Status |
|---|---|---:|---:|---|
| 1 | hours_140k_p1of3 | 7,275,295 | 71 | Complete |
| 2 | hours_140k_p2of3 | 7,235,630 | 70 | Complete |
| 3 | hours_140k_p3of3 | 7,251,238 | 71 | Complete |
| 4 | convspeech | 6,514,097 | 64 | Complete |
| 5 | podcast_p11to14 | 9,756,046 | 94 | Complete |
| 6 | podcast_p14to17 | 7,670,431 | 74 | Complete |
| 7 | podcast_p17to20_p1of3 | 7,546,490 | 73 | Complete |
| 8 | podcast_p17to20_p2of3 | 7,544,848 | 73 | Complete |
| 9 | podcast_p17to20_p3of3 | 7,623,291 | 74 | Complete |
| 10 | multilingual_p0 | 14,016,920 | 134 | Complete |
| 11 | multilingual_p1 | 14,000,000 | 134 | Complete |
| 12 | multilingual_p2 | 13,990,495 | 134 | Complete |
| 13 | multilingual_p3 | 13,898,108 | 133 | Complete |
| 14 | multilingual_p4 | 13,897,974 | 133 | Complete |
| 15 | multilingual_p5 | 13,321,438 | 128 | Complete |
| 16 | multilingual_p6 | 13,860,492 | 133 | Complete |
| 17 | multilingual_p7 | 13,869,821 | 133 | Complete |
| 18 | multilingual_p8 | 13,863,636 | 133 | Complete |
| 19 | multilingual_p9 | 13,862,549 | 133 | Complete |
| 20 | multilingual_p10 | 13,862,580 | 133 | Complete |
| 21 | multilingual_p11 | 13,897,600 | 133 | Complete |
| 22 | multilingual_p12 | 13,897,454 | 133 | Complete |
| 23 | multilingual_p13 | 13,799,963 | 132 | Complete |
| 24 | multilingual_p14 | 13,899,430 | 133 | Complete |
| 25 | multilingual_p15 | 9,685,845 | 93 | Check — expected ~13.9M, got 9.7M |
| | **Total** | **~286M** | | **24 complete, 1 to verify** |

Note: Tables 4-6 (convspeech, podcast_p11to14, podcast_p14to17) were the pilot jobs and used a slightly different naming convention (`{family}_transcript_cv.lance` instead of `transcript_cv_{family}.lance`). The actual paths are documented in section 4.12 above.

Note: multilingual_p15 has 9,685,845 rows vs expected ~13,900,000. This may indicate the job was still writing when checked, or it terminated early. Needs verification — check Ray job logs for `raysubmit_NTGvs3NNbdDeZUKj` on `vibevoice-omniva-s4`.
