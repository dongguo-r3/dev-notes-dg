# Audio Data Processing Logs

> **What this note is for:** Run logs for specific audio data pipeline jobs — job IDs, cluster assignments, throughput, failures, and outcomes. Each entry is a discrete pipeline execution against a dataset.
>
> | Date | Pipeline | Dataset | Outcome |
> |---|---|---|---|
> | 2026-03-18 | Fidelity (bandwidth + AES + SED) | `whisperx__eng_v1` (podcast_10m, 10K rows) | ✅ 10K rows in 36 min, 8×H100 kiwi-flyte |
> | 2026-04-08 | Fidelity | `internal_audio_v1` (~92M rows, 8 partitions) | 🟡 Running — p0 on kiwi (Sydney), p1-p7 on omniva (US, ~30% throughput) |

---

## 2026-03-18 Summary

- Refactored `fidelity_pipeline.py` to align with `podcast_asr_pipeline.py` patterns (dynamic concurrency via `_detect_cluster_gpus()`, `num_gpus=1` per actor, removed dead fractional GPU code)
- Renamed `pipeline.py` → `fidelity_pipeline.py`, `processor.py` → `fidelity_processor.py`
- Attempted 5 job submissions: ashburn down (migration), chicago failed (AMD GPUs, no CUDA), kiwi-flyte preempted (low priority), then succeeded with high-priority kiwi-flyte
- **Final result**: 10K rows processed successfully in 36 min on 8 H100 GPUs (~4.6 rows/sec)
- Output: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance`
- Key learnings: always use `--priority high-priority` for dev clusters, ashburn is the only NVIDIA data-ray-cluster, runtime env pip install adds ~10-15 min cold start (need custom Docker image for production)

---

## 2026-03-18 Detailed Logs

### Fidelity pipeline — 1K row verification run

- **Job ID**: `raysubmit_21dzJzD6HiVAhTSG`
- **Cluster**: chicago / data-ray-cluster
- **Source**: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance`
- **Destination**: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance`
- **Pipeline**: `lax.projects.av_data_processing.audio.audio_metadata.fidelity_pipeline.run_fidelity_pipeline_gpu`
- **Params**: `audio_key=audio_bytes`
- **Limit**: 1000 rows
- **Status**: Submitted

**Launch scripts**:

```bash
cd projects/lax && source .venv/bin/activate

# Connect to chicago (ashburn was down due to migration)
source scripts/setup-ray-proxy.sh chicago

# Submit 1K verification run
python -m lax.scripts.submit_ray_job --no-wait \
    --source_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance \
    --destination_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance \
    --pipeline_config lax.projects.av_data_processing.audio.audio_metadata.fidelity_pipeline.run_fidelity_pipeline_gpu \
    --pipeline_params audio_key=audio_bytes \
    --limit 1000
```

**Check status**:

```bash
ray job status raysubmit_21dzJzD6HiVAhTSG \
  --address "$RAY_ADDRESS" \
  --headers "{\"Authorization\": \"Bearer $RAY_PROXY_TOKEN\"}"

ray job logs raysubmit_21dzJzD6HiVAhTSG \
  --address "$RAY_ADDRESS" \
  --headers "{\"Authorization\": \"Bearer $RAY_PROXY_TOKEN\"}"
```

**Dashboard**: https://ray-dashboard.protected.sydney3.labs.lumalabs.ai/chicago/data-ray-cluster/#/jobs

**Notes**: Ashburn was down due to migration. Used chicago instead. This is a verification run before processing the full dataset.

**Result**: FAILED — `ModuleNotFoundError: No module named 'librosa'`. Chicago cluster doesn't have the required pip packages pre-installed.

---

### Fidelity pipeline — 1K row verification run (attempt 2, with runtime env)

- **Job ID**: `raysubmit_R3P5wafq1wPVELKW`
- **Cluster**: chicago / data-ray-cluster
- **Status**: FAILED

**Launch scripts**:

```bash
# Create runtime env with pip dependencies
echo '{"pip": ["librosa", "audiobox-aesthetics", "panns-inference", "soxr"]}' > /tmp/fidelity_runtime_env.json

python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env /tmp/fidelity_runtime_env.json \
    --source_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance \
    --destination_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance \
    --pipeline_config lax.projects.av_data_processing.audio.audio_metadata.fidelity_pipeline.run_fidelity_pipeline_gpu \
    --pipeline_params audio_key=audio_bytes \
    --limit 1000
```

**Result**: FAILED — `ValueError: ('min_size must be >= 1', 0)`. Pip install succeeded, but the pipeline crashed because Chicago has **AMD MI300X GPUs** (not NVIDIA). The CUDA-dependent models (audiobox-aesthetics, PANNs) cannot run on AMD GPUs. The `num_gpus=1` resource request likely returned 0 available CUDA GPUs.

**Lesson learned**:

- **Ashburn** (NVIDIA A10) is the only data-ray-cluster with NVIDIA GPUs — required for this pipeline
- **Chicago/Osaka** have AMD MI300X — only for ROCm-compatible workloads (e.g. vLLM, large language models)
- Need to wait for ashburn migration to complete, or use a dev cluster on omniva-flyte/kiwi-flyte (NVIDIA H100s)

---

### Fidelity pipeline — 1K row verification run (attempt 3, kiwi-flyte dev cluster)

- **Job ID**: `raysubmit_hBgv74Kxj4CSgwr2`
- **Cluster**: kiwi-flyte / dongguo-dongguo-fidelity-a1f30a (NVIDIA H100, 1 node)
- **Source**: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance`
- **Destination**: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance`
- **Limit**: 1000 rows
- **Status**: Submitted

**Launch scripts**:

```bash
# Step 1: Create a temporary Ray cluster on kiwi-flyte (NVIDIA H100)
cd projects/kuma && source .venv/bin/activate
flytecli activate --cluster kiwi-flyte
flytecli ray-cluster --name dongguo-fidelity --nodes 1

# Wait for Ready status
flytecli ray-cluster --name dongguo-fidelity --status

# Step 2: Set up proxy
cd projects/lax && source .venv/bin/activate
source scripts/setup-ray-proxy.sh kiwi-flyte dongguo-dongguo-fidelity-a1f30a

# Step 3: Create runtime env with pip dependencies
echo '{"pip": ["librosa", "audiobox-aesthetics", "panns-inference", "soxr"]}' > /tmp/fidelity_runtime_env.json

# Step 4: Submit
python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env /tmp/fidelity_runtime_env.json \
    --source_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance \
    --destination_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance \
    --pipeline_config lax.projects.av_data_processing.audio.audio_metadata.fidelity_pipeline.run_fidelity_pipeline_gpu \
    --pipeline_params audio_key=audio_bytes \
    --limit 1000
```

**Check status**:

```bash
ray job status raysubmit_hBgv74Kxj4CSgwr2 \
  --address "$RAY_ADDRESS" \
  --headers "{\"Authorization\": \"Bearer $RAY_PROXY_TOKEN\"}"
```

**Dashboard**: https://ray-dashboard.protected.sydney3.labs.lumalabs.ai/kiwi-flyte/dongguo-dongguo-fidelity-a1f30a/

**Clean up** (after job completes):

```bash
cd projects/kuma && source .venv/bin/activate
flytecli ray-cluster --name dongguo-fidelity --delete --yes
```

**Result**: FAILED — Cluster was **preempted** by a higher-priority workload after ~10 minutes. The job had allocated all 8 GPUs and was loading models (0 rows completed) when it was evicted.

**Root cause**: The cluster was created with `--priority low-priority` (the default). Kueue preempted it to make room for a higher-priority job:

```
Preempted to accommodate a workload (UID: 40399c9c...) due to prioritization in the ClusterQueue
```

**Key takeaway**: Always create dev clusters with `--priority high-priority` to avoid preemption:

```bash
# ❌ Default (low-priority, can be preempted)
flytecli ray-cluster --name dongguo-fidelity --nodes 1

# ✅ High priority (less likely to be preempted)
flytecli ray-cluster --name dongguo-fidelity --nodes 1 --priority high-priority
```

---

### Fidelity pipeline — 1K row verification run (attempt 4, high-priority kiwi-flyte) ✅ SUCCESS

- **Job ID**: `raysubmit_Q5HkcnbgPgujaige`
- **Cluster**: kiwi-flyte / dongguo-dongguo-fidelity-9e7626 (NVIDIA H100, 1 node, **high-priority**)
- **Source**: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance`
- **Destination**: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance`
- **Limit**: 1000 rows
- **Status**: SUCCESS
- **Duration**: ~37 minutes (2197s) — dominated by pip install + model checkpoint downloads on first run
- **Output**: 1000 rows, 678KB

**Launch scripts**:

```bash
# Step 1: Create high-priority cluster
cd projects/kuma && source .venv/bin/activate
flytecli activate --cluster kiwi-flyte
flytecli ray-cluster --name dongguo-fidelity --nodes 1 --priority high-priority
flytecli ray-cluster --name dongguo-fidelity --status  # wait for Ready

# Step 2: Set up proxy
cd projects/lax && source .venv/bin/activate
source scripts/setup-ray-proxy.sh kiwi-flyte dongguo-dongguo-fidelity-9e7626

# Step 3: Runtime env
echo '{"pip": ["librosa", "audiobox-aesthetics", "panns-inference", "soxr"]}' > /tmp/fidelity_runtime_env.json

# Step 4: Submit
python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env /tmp/fidelity_runtime_env.json \
    --source_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance \
    --destination_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance \
    --pipeline_config lax.projects.av_data_processing.audio.audio_metadata.fidelity_pipeline.run_fidelity_pipeline_gpu \
    --pipeline_params audio_key=audio_bytes \
    --limit 1000
```

**Notes**:

- Progress bar showed `Completed 0` for ~35 minutes while pip packages installed and models downloaded on each actor — this is normal for first run with runtime env pip dependencies
- All 1000 rows were flushed at the end in one batch
- Actual processing throughput unclear from logs — need a second run (models cached) to measure
- The cluster (`dongguo-dongguo-fidelity-9e7626`) is still running and can be reused for follow-up jobs

**Next steps**:

- Run a second 1000-row batch to measure actual processing speed (no download overhead)
- For production: build a Docker image with deps pre-installed to avoid the 35-min cold start
- Don't forget to delete the cluster when done: `flytecli ray-cluster --name dongguo-fidelity --delete --yes`

---

### Fidelity pipeline — 10K row formal run (attempt 5, reusing kiwi-flyte cluster) ✅ SUCCESS

- **Job ID**: `raysubmit_4zRutCD6M4SY93eN`
- **Cluster**: kiwi-flyte / dongguo-dongguo-fidelity-9e7626 (reused from attempt 4)
- **Source**: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance`
- **Destination**: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance`
- **Limit**: 10,000 rows
- **Status**: SUCCESS
- **Duration**: 2180s (~36 minutes)
- **Output**: 10,000 rows, 6.7MB
- **Throughput**: ~4.6 rows/sec (~275 rows/min) across 8 H100 GPUs

**Launch script**:

```bash
python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env /tmp/fidelity_runtime_env.json \
    --source_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance \
    --destination_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance \
    --pipeline_config lax.projects.av_data_processing.audio.audio_metadata.fidelity_pipeline.run_fidelity_pipeline_gpu \
    --pipeline_params audio_key=audio_bytes \
    --limit 10000
```

**Notes**:

- Reused the cluster from attempt 4 — pip packages and models were cached, so cold start was minimal
- 10K run took ~36 min vs 1K run's ~37 min → the 1K run was dominated by cold start (pip install + model download)
- Actual processing throughput: ~4.6 rows/sec across 8 GPUs (~0.58 rows/sec per GPU)
- Progress bar stayed at `Completed 1` for most of the run, then flushed all rows at the end
- GPU utilization is likely low (models only ~3GB VRAM on 80GB H100s) — future optimization: use fractional GPU allocation to pack more actors per GPU

---

### Fidelity pipeline — 90K row run (attempt 6, new kiwi-flyte cluster)

- **Cluster**: kiwi-flyte (1 node, NVIDIA H100, high-priority)
- **Source**: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance`
- **Destination**: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance`
- **Limit**: 100,000 rows (10K already done → ~90K new rows via dedup)
- **Job ID**: `raysubmit_cMEA1TCAqTUT8Nqq`
- **Cluster CR**: `dongguo-dongguo-fidelity-ecc0f9`
- **Status**: SUCCESS
- **Duration**: 2205s (~37 minutes)
- **Output**: 100,000 rows, 67.3MB

**Complete launch script**:

```bash
# === Step 1: Create cluster (from kuma venv) ===
cd projects/kuma && source .venv/bin/activate
flytecli activate --cluster kiwi-flyte
flytecli ray-cluster --name dongguo-fidelity --nodes 1 --priority high-priority

# Wait for Ready (~2-3 min)
flytecli ray-cluster --name dongguo-fidelity --status

# === Step 2: Set up proxy and submit (from lax venv) ===
cd projects/lax && source .venv/bin/activate
source scripts/setup-ray-proxy.sh kiwi-flyte dongguo-dongguo-fidelity-ecc0f9

# Create runtime env (if not already at /tmp/fidelity_runtime_env.json)
echo '{"pip": ["librosa", "audiobox-aesthetics", "panns-inference", "soxr"]}' > /tmp/fidelity_runtime_env.json

# Submit 100K rows (dedup skips the 10K already processed)
python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env /tmp/fidelity_runtime_env.json \
    --source_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance \
    --destination_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance \
    --pipeline_config lax.projects.av_data_processing.audio.audio_metadata.fidelity_pipeline.run_fidelity_pipeline_gpu \
    --pipeline_params audio_key=audio_bytes \
    --limit 100000

# === Step 3: Monitor ===
ray job status <JOB_ID> \
  --address "$RAY_ADDRESS" \
  --headers "{\"Authorization\": \"Bearer $RAY_PROXY_TOKEN\"}"

# === Step 4: Clean up when done ===
cd projects/kuma && source .venv/bin/activate
flytecli ray-cluster --name dongguo-fidelity --delete --yes
```

**Notes**: Cold start ~10-15 min (pip install + model download). Actual processing was much faster than initially estimated.

---

### Fidelity pipeline — 100K row run (attempt 7, reusing cluster, warm) ✅ SUCCESS

- **Job ID**: `raysubmit_S3KCXjew6uWtuhku`
- **Cluster**: kiwi-flyte / dongguo-dongguo-fidelity-ecc0f9 (reused)
- **Limit**: 200,000 rows (100K already done → ~100K new rows via dedup)
- **Status**: SUCCESS
- **Duration**: 2221s (~37 minutes)
- **Output**: 200,000 total rows, 133.6MB

**Launch script** (reused same cluster, no setup needed):

```bash
python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env /tmp/fidelity_runtime_env.json \
    --source_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__eng_v1.lance \
    --destination_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance \
    --pipeline_config lax.projects.av_data_processing.audio.audio_metadata.fidelity_pipeline.run_fidelity_pipeline_gpu \
    --pipeline_params audio_key=audio_bytes \
    --limit 200000
```

---

## Throughput Analysis

### Raw Data

| Run | New rows | Wall time (s) | Total rows/sec | Notes |
|-----|----------|---------------|----------------|-------|
| Attempt 4 (1K) | 1,000 | 2,197 | 0.5 | Cold start: pip install + model download from scratch |
| Attempt 5 (10K) | 9,000 | 2,180 | 4.1 | Warm cluster from attempt 4, pip still reinstalls per job |
| Attempt 6 (100K) | 90,000 | 2,205 | 40.8 | New cluster, cold start |
| Attempt 7 (100K) | 100,000 | 2,221 | 45.0 | Warm cluster reuse, pip still reinstalls per job |

### Key Observations

1. **Wall time is nearly constant (~37 min) regardless of row count.** This means the overhead (pip install + model loading + Ray Data setup) dominates for small runs, but actual processing is fast once actors are warm.

2. **Overhead breakdown (estimated)**:

   - pip install per actor: ~10-15 min (happens every job, even on reused cluster)
   - Model download (first run only): ~5-10 min
   - Ray Data setup + dedup scan: ~2-5 min
   - Total overhead: ~15-20 min per job

3. **Actual processing speed (after overhead)**:

   - Attempt 7: 100K rows in ~17 min of actual processing = **~98 rows/sec** (~5,900 rows/min)
   - Per GPU: ~12 rows/sec (8 H100 GPUs)
   - Per GPU utilization: very low (~3GB / 80GB VRAM used)

4. **Scaling projections** (1 node, 8 H100 GPUs, ~98 rows/sec after warm-up):

   | Dataset size | Processing time | Total time (incl. overhead) |
   |---|---|---|
   | 100K rows | ~17 min | ~37 min |
   | 500K rows | ~85 min | ~105 min (~1.75 hr) |
   | 1M rows | ~170 min | ~190 min (~3.2 hr) |
   | 10M rows | ~28 hr | ~28 hr |

5. **Optimization opportunities**:

   - **Custom Docker image**: eliminates ~15 min pip install per job → biggest single improvement
   - **Fractional GPU (`num_gpus=0.1`)**: pack ~10 actors per GPU instead of 1 → potential ~10x throughput
   - **Multi-node**: add more nodes (`--nodes 2+`) for linear scaling
   - Combined: custom image + fractional GPU on 2 nodes could process 10M rows in ~1.5 hr

---

### Fidelity pipeline — 3M multilingual run (attempt 8, reusing cluster) ⏳ RUNNING

- **Job ID**: `raysubmit_VncbGvL3BfZVGd9W`
- **Cluster**: kiwi-flyte / dongguo-dongguo-fidelity-ecc0f9 (reused)
- **Source**: `s3://ai-lumalabs-datasets-ap-se-2/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1.lance`
- **Destination**: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_multilingual_v1.lance`
- **Limit**: 3,000,000 rows
- **ETA**: ~8.5 hours processing + ~15 min overhead (at ~98 rows/sec)
- **Status**: FAILED → resubmitted twice
- **Failure 1** (`raysubmit_cMEA1TCAqTUT8Nqq`): Dataset has 54,905 fragments — randomized read with limit=3M was too slow. Resubmitted with `--no-randomized`.
- **Failure 2** (`raysubmit_GByqaBLBHyRF3jEX`): Ray head pod restarted, losing job state (404 not found). Cluster showed Ready but job was gone.
- **Failure 3** (`raysubmit_76JnAh13s4psimmn`): Ray head pod restarted again (404). Cluster was unstable.
- **Final resubmit** (`raysubmit_A5fEikcCdtFXSNjZ`): Fresh cluster `dongguo-dongguo-fidelity-4f6501`. Submitted 2026-03-19 00:38 PDT. Running overnight.

**Launch script**:

```bash
python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env /tmp/fidelity_runtime_env.json \
    --source_uri s3://ai-lumalabs-datasets-ap-se-2/audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1.lance \
    --destination_uri s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_multilingual_v1.lance \
    --pipeline_config lax.projects.av_data_processing.audio.audio_metadata.fidelity_pipeline.run_fidelity_pipeline_gpu \
    --pipeline_params audio_key=audio_bytes \
    --limit 3000000 \
    --no-randomized
```

**Check status**:

```bash
ray job status raysubmit_A5fEikcCdtFXSNjZ \
  --address "$RAY_ADDRESS" \
  --headers "{\"Authorization\": \"Bearer $RAY_PROXY_TOKEN\"}"
```

**Clean up when done**:

```bash
cd projects/kuma && source .venv/bin/activate
flytecli ray-cluster --name dongguo-fidelity --delete --yes
# New cluster CR: dongguo-dongguo-fidelity-4f6501
```

---

### Fidelity pipeline — mosaic AVGU segments (planned for 2026-03-19)

- **Source**: `s3://ai-lumalabs-datasets-ap-se-2-lance/inkyu/t2av/mosaic_golden/mosaic_avgu_seg_audiobox_qwen3omni_v2.lance`
- **Destination**: `s3://ai-lumalabs-datasets-ap-se-2-lance/dongguo/t2av/mosaic_golden/mosaic_avgu_seg_fidelity.lance`
- **Limit**: TBD (check row count first)
- **Status**: PLANNED

**Pre-flight** (check schema and row count before running):

```bash
cd projects/lax && source .venv/bin/activate
python -c "
import lance
ds = lance.dataset('s3://ai-lumalabs-datasets-ap-se-2-lance/inkyu/t2av/mosaic_golden/mosaic_avgu_seg_audiobox_qwen3omni_v2.lance')
print(f'Rows: {ds.count_rows()}')
print(f'Schema: {ds.schema}')
# Check audio column name — may be audio_bytes, audio_bytes_ori, etc.
print(f'Columns: {ds.schema.names}')
"
```

**Launch script** (complete, copy-paste ready):

```bash
# === Step 1: Create cluster (or reuse if still running) ===
cd projects/kuma && source .venv/bin/activate
flytecli activate --cluster kiwi-flyte
flytecli ray-cluster --name dongguo-fidelity --nodes 1 --priority high-priority
flytecli ray-cluster --name dongguo-fidelity --status  # wait for Ready, note the CR name

# === Step 2: Set up proxy and submit ===
cd projects/lax && source .venv/bin/activate
source scripts/setup-ray-proxy.sh kiwi-flyte <CR_NAME_FROM_STATUS>

echo '{"pip": ["librosa", "audiobox-aesthetics", "panns-inference", "soxr"]}' > /tmp/fidelity_runtime_env.json

# Uses video_path mode — extracts audio from MP4 via ffmpeg (no audio_bytes column in this table)
python -m lax.scripts.submit_ray_job --no-wait \
    --runtime-env /tmp/fidelity_runtime_env.json \
    --source_uri s3://ai-lumalabs-datasets-ap-se-2-lance/inkyu/t2av/mosaic_golden/mosaic_avgu_seg_audiobox_qwen3omni_v2.lance \
    --destination_uri s3://ai-lumalabs-datasets-ap-se-2-lance/dongguo/t2av/mosaic_golden/mosaic_avgu_seg_fidelity.lance \
    --pipeline_config lax.projects.av_data_processing.audio.audio_metadata.fidelity_pipeline.run_fidelity_pipeline_gpu_from_video \
    --pipeline_params video_key=key

# === Step 3: Clean up when done ===
cd projects/kuma && source .venv/bin/activate
flytecli ray-cluster --name dongguo-fidelity --delete --yes
```

**Notes**:

- This is a T2AV (text-to-audio-video) dataset — **no audio_bytes column**. Audio is in MP4 videos on S3.
- Uses `run_fidelity_pipeline_gpu_from_video` which extracts audio from video via ffmpeg
- The `key` column contains S3 paths like `s3://ai-lumalabs-datasets-ap-se-2/audio_resource/mosaic_20260210/...mp4`
- Table already has `avgu__filtering_audiobox` (audiobox CE/CU/PC/PQ scores from Inkyu's AVGU pipeline). Fidelity pipeline adds bandwidth_hz + sound_events.
- 38,110 rows — small table, should complete in ~10 min after cold start
- Estimated time depends on row count: ~30K rows/hr on 1 node with 8 H100 GPUs (after cold start)

---

## 2026-03-19

### Fidelity Output Tables — Inventory

All tables produced by the fidelity pipeline. Schema: `bandwidth_hz`, `aes_ce`, `aes_cu`, `aes_pc`, `aes_pq`, `sound_events`, `original_row_id`.

#### Production tables

| Table | Rows | Fragments | Source | Status |
|-------|------|-----------|--------|--------|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance` | 311,000 | 41 | `whisperx__eng_v1.lance` (English podcast ASR) | Complete |
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_multilingual_v1.lance` | 461,937 | 57 | `whisperx__multilingual_v1.lance` (Multilingual podcast ASR) | In progress (500K job running) |

#### Test / experiment tables

| Table | Rows | Fragments | Notes |
|-------|------|-----------|-------|
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_fidelity_metadata.lance` | 5,000 | 1 | Full fidelity on internal_audio_v1 |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_fidelity_metadata_test.lance` | 3,000 | 1 | Test run |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_fidelity_test.lance` | 3,000 | 2 | Test run |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_1_fidelity_test.lance` | 3,000 | 1 | Test run (variant 1) |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_2_fidelity_test.lance` | 3,000 | 1 | Test run (variant 2) |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/podcast_10m_segments_fidelity_metadata.lance` | 30,000 | 6 | Local experiment (eager/streaming/processpool comparison) |

#### Local parquet copies

| File | Rows | Source |
|------|------|--------|
| `/Users/dongguo/Projects/adhoc/fidelity_eng_v1.parquet` | 311,000 | Snapshot of eng production table |
| `/Users/dongguo/Projects/adhoc/fidelity_multilingual_v1.parquet` | 461,937 | Partial snapshot of multilingual table |
| `/Users/dongguo/Projects/adhoc/internal_audio_v1_fidelity_metadata.parquet` | 5,000 | From earlier experiment |

---

## Full Lance Table Inventory (as of 2026-03-19)

### Active tables

| Table | Rows | Fragments | Status | Notes |
|-------|------|-----------|--------|-------|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1.lance` | 311,000 | 41 | Active | Production, English podcast |
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_multilingual_v1.lance` | 461,937 | 57 | In progress | Target: 3M, multilingual podcast |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_fidelity_metadata.lance` | 5,000 | 1 | Active | internal_audio_v1 sample |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/podcast_10m_segments_fidelity_metadata.lance` | 30,000 | 6 | Active | Local experiment results |

### Test tables (can delete)

| Table | Rows | Status | Notes |
|-------|------|--------|-------|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/bandwidth_only_test.lance` | 100 | Test | Bandwidth-only pipeline test, can delete |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_fidelity_metadata_test.lance` | 3,000 | Test | AccessDenied on delete |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_fidelity_test.lance` | 3,000 | Test | AccessDenied on delete |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_1_fidelity_test.lance` | 3,000 | Test | AccessDenied on delete |
| `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/audio_pipeline/internal_audio_v1_2_fidelity_test.lance` | 3,000 | Test | AccessDenied on delete |

### Deleted tables

| Table | Notes |
|-------|-------|
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1_demo.lance` | Broken (NULL columns), deleted 2026-03-19 |
| `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_eng_v1_1_demo.lance` | Case study demo, deleted 2026-03-19 |

### Local parquet copies

| File | Rows | Source |
|------|------|--------|
| `internal_audio_v1_5K.parquet` | 5,000 | `internal_audio_v1_fidelity_metadata.lance` |
| `internal_audio_v1_fidelity_metadata.parquet` | 25,000 | Older run (not matching S3 table) |
| `podcast_segment_30K.parquet` | 30,000 | `podcast_10m_segments_fidelity_metadata.lance` |
| `fidelity_eng_v1.parquet` | 311,000 | `fidelity_eng_v1.lance` |
| `fidelity_multilingual_v1.parquet` | 461,937 | `fidelity_multilingual_v1.lance` (partial) |

---

## LAX Docker Images for Audio Pipelines

### Current Image Variants

The LAX Docker build system (`.github/workflows/ci-lax-docker.yaml` + `projects/lax/infrastructure/docker/Dockerfile`) supports 4 official variants:

| Image prefix | Docker target | GPU | Use case |
|---|---|---|---|
| `gpu` | `base_gpu` | NVIDIA | Default for most pipelines |
| `cpu` | `base_cpu` | None | CPU-only jobs |
| `gpu-serve` | `gpu_serve` | NVIDIA | RayService vLLM serving |
| `gpu-amd` | `base_gpu_amd` | AMD ROCm | AMD MI300X clusters (chicago, osaka) |

### The `gpu-asr` Image

The ASR pipeline documentation references a `gpu-asr` image (`phx.ocir.io/axsgshhbf0lb/lax:gpu-asr-902f1084`) with WhisperX, pyannote, and cuDNN 8+9 pre-installed. However, **there is no `base_gpu_asr` target in the Dockerfile** — this was a one-off custom build, not part of the CI pipeline.

### Fidelity Pipeline Dependencies

The fidelity pipeline requires `librosa`, `audiobox-aesthetics`, `panns-inference`, and `soxr`, which are not in the default `gpu` image. Current workaround: runtime env with pip install (`--runtime-env` JSON), which adds ~10-15 min cold start per job.

For production, a custom Docker image should be built with these deps pre-installed (similar to `gpu-asr`).

### Where to Discuss / Who to Ask

| Channel | Purpose |
|---|---|
| **#proj-ml-infra** | Docker image builds, cluster infra. **Yichen Wang** is the primary person who builds LAX images. |
| **#proj-data-processing-prs** | PRs for Docker/infra changes (e.g. PR #6136 "building a source of truth for lax docker images") |
| **#sre-team** | Image pull issues, registry auth, node problems |

### Image Registries

| Registry | Clusters |
|---|---|
| `808558726171.dkr.ecr.ap-southeast-2.amazonaws.com/lax:gpu-*` | kiwi-flyte (AWS, H100) |
| `iad.ocir.io/axsgshhbf0lb/lax:gpu-amd-*` | ashburn, chicago, osaka (OCI) |

### How to Build a Custom Image (TODO)

To add a `gpu-audio-metadata` variant:

1. Add a new Docker stage in `projects/lax/infrastructure/docker/Dockerfile` based on `base_gpu`
2. Add `RUN uv pip install librosa audiobox-aesthetics panns-inference soxr`
3. Add to CI workflow matrix in `.github/workflows/ci-lax-docker.yaml`
4. Use `--image` flag when creating Ray clusters: `flytecli ray-cluster --name my-cluster --image <new-image>`

---

## KubeRay Cluster Image Notes (2026-04-15)

### Image Comparison: `gpu-08c77f73` vs `gpu-63d5cafd`

Two OCIR Phoenix images used across omniva-flyte clusters:

- **`gpu-08c77f73`** — Old "ecr-gpu-stable" image, pinned 2026-02-20 (PR #5764). Ray 2.50.0, no hplv.
- **`gpu-63d5cafd`** — Current `ocir-phx-gpu-ray2_54-vllm0_14-hplv` image, built 2026-04-02 (PR #7011). Ray 2.54.0, vLLM 0.14.1, hplv >= 1.0.39, hplv-native >= 1.0.52.

| Attribute | `gpu-08c77f73` | `gpu-63d5cafd` |
|---|---|---|
| Ray version | 2.50.0 | 2.54.0 |
| vLLM version | older | 0.14.1 |
| hplv | not included | >= 1.0.39 |
| Image age | Feb 20 | Apr 2 |

### Image Source of Truth

All image tags are defined in `lib/lax-images/images.yaml`. Key entries for KubeRay clusters:

- `ecr-gpu-ray2_54-vllm0_14-hplv` → `808558726171.dkr.ecr.ap-southeast-2.amazonaws.com/lax:gpu-73cf19e3` (kiwi-flyte)
- `ocir-phx-gpu-ray2_54-vllm0_14-hplv` → `phx.ocir.io/axsgshhbf0lb/lax:gpu-63d5cafd` (omniva-flyte)

### Image vs Application Code

The image provides the **base environment** (Ray, vLLM, system libs, pip packages). When using `submit_ray_job`, your local `projects/lax/` directory is uploaded as a working directory overlay, so the **latest application code** runs regardless of what's baked into the image.

To use a specific image: `flytecli ray-cluster --name my-cluster --image <image-uri>`

### Cluster Inventory (2026-04-15)

**vibevoice-omniva-s0..s7** (old image `gpu-08c77f73`, created 2026-03-30):
- Head: ~307Gi memory, ~1031Gi worker object store (64%, older flytecli proportions)
- Used for fidelity `en50m_nonen50m` processing (all 8 partitions completed)

**metadata-s0..s7** (latest image `gpu-63d5cafd`, created 2026-04-15):
- Head: 256Gi memory, ~128Gi object store (50%)
- Worker: 88 CPUs, 8 GPUs, 1600Gi, ~515Gi object store (30%)
- Ready and idle, awaiting next workload

**metadata-s8** (old image `gpu-08c77f73`, created 2026-04-15):
- Same config as metadata-s0..s7 but using the old image for A/B comparison
- CR: `dongguo-metadata-s8-0e7583`

---

## 2026-04-15: Image A/B Test — gpu-63d5cafd vs gpu-08c77f73

### Goal

Compare the two omniva Docker images on the same fidelity pipeline workload to determine if the newer image (Ray 2.54, hplv) has any performance difference vs the older one (Ray 2.50, no hplv).

### Setup

- **Cluster A** (new image): `dongguo-metadata-s7-ffd6fc` — `gpu-63d5cafd` (Ray 2.54.0, vLLM 0.14.1, hplv)
- **Cluster B** (old image): `dongguo-metadata-s8-0e7583` — `gpu-08c77f73` (Ray 2.50.0, no hplv)
- Both clusters: 1 worker (88 CPUs, 8 GPUs, 1600Gi), head 256Gi, omniva-flyte
- Same source table, same partition (0,4,1 = 25% of table), same application code overlay

### Jobs

- **Job A** (new image): `raysubmit_xPbtRc7btXTGn7s8`
  - Destination: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_en50m_nonen50m_testrun_img_gpu-63d5cafd.lance`
- **Job B** (old image): `raysubmit_r6zUQirQPdzxVuyx`
  - Destination: `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_en50m_nonen50m_testrun_img_gpu-08c77f73.lance`

### Scripts

```bash
cd /Users/dongguo/Projects/lumaverse/projects/lax

LAX_PYTHON=".venv/bin/python"
DATA_API_URL="https://b0b7c37fc317-data-api-staging.sydney3.labs.lumalabs.ai"
SOURCE="s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/asr/whisperx__en50m_nonen50m_compacted.lance"
PIPELINE="lax.projects.av_data_processing.audio.audio_metadata.fidelity_pipeline.run_fidelity_pipeline_gpu"
RUNTIME_ENV="lax/projects/av_data_processing/audio/audio_metadata/runtime_env.json"

source ../../projects/lax/scripts/setup-ray-proxy.sh omniva-flyte "dongguo-metadata-s7-ffd6fc" <<< "n" > /dev/null 2>&1

# Test 1: New image (gpu-63d5cafd) on metadata-s7
$LAX_PYTHON -m lax.scripts.submit_ray_job --no-wait \
  --ray-address "${DATA_API_URL}/api/v1/ray-proxy/omniva-flyte/dongguo-metadata-s7-ffd6fc" \
  --source_uri "${SOURCE}" \
  --destination_uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_en50m_nonen50m_testrun_img_gpu-63d5cafd.lance" \
  --pipeline_config "${PIPELINE}" \
  --pipeline_params '{"audio_key": "audio_bytes"}' \
  --no-randomized \
  --runtime-env "${RUNTIME_ENV}" \
  --partitions_range "0,4,1"

# Test 2: Old image (gpu-08c77f73) on metadata-s8
$LAX_PYTHON -m lax.scripts.submit_ray_job --no-wait \
  --ray-address "${DATA_API_URL}/api/v1/ray-proxy/omniva-flyte/dongguo-metadata-s8-0e7583" \
  --source_uri "${SOURCE}" \
  --destination_uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/fidelity_en50m_nonen50m_testrun_img_gpu-08c77f73.lance" \
  --pipeline_config "${PIPELINE}" \
  --pipeline_params '{"audio_key": "audio_bytes"}' \
  --no-randomized \
  --runtime-env "${RUNTIME_ENV}" \
  --partitions_range "0,4,1"
```

### Expectation

The fidelity pipeline is not heavy on Ray-specific features (no hplv usage, no vLLM), so we expect **similar performance** between the two images. The main variable is Ray 2.50 vs 2.54 — any difference would come from Ray Data scheduling improvements in 2.54.

### Results

- **Job A** (gpu-63d5cafd): TBD
- **Job B** (gpu-08c77f73): TBD

---

## 2026-04-16: VibeVoice ASR — SFT Tables (Group 62)

### Goal

Run VibeVoice ASR on 5 SFT lance tables (table-2 through table-6 from the internal dashboard group 62) to produce vibevoice transcripts alongside existing WhisperX transcripts. Table-1 (americanrhetoric, 28K rows) is skipped for now due to small size.

### Source Tables

| Table | Dataset | Source URI | Rows | Fragments |
|-------|---------|-----------|------|-----------|
| table-2 | hours_140k | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance` | 21,745,714 | 5,381 |
| table-3 | convspeech | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/convspeech/asr/prefiltered_english__whisperx.lance` | 6,514,097 | 1,614 |
| table-4 | podcast p11-14 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p11to14_whisperx_clean.lance` | 7,499,644 | 1,858 |
| table-5 | podcast p14-17 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p14to17_whisperx_clean.lance` | 7,670,431 | 1,902 |
| table-6 | podcast p17-20 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance` | 22,655,625 | 5,606 |

All tables share the same schema: `audio_bytes` (list\<binary\>), `audio_path`, `sample_rate`, `language`, `segment_start`, `segment_end`, `segment_duration`, `whisperx_asr_content`, `whisperx_timestamp`, `num_speakers`, `total_speakers_in_file`, `lufs_gain_db`, `snr_db`, `speech_ratio`, `avg_word_score`, `overlap_ratio`, `original_row_id`.

### Cluster & Partition Plan

9 omniva Ray clusters (`vibevoice-omniva-s0` through `vibevoice-omniva-s8`), each with 1 worker node (8 GPUs). The two largest tables (table-2 and table-6) are split into 3 partitions each by fragment range.

| Cluster | Source Table | Partition | Fragment Range | Dest URI |
|---------|-------------|-----------|----------------|----------|
| s0 | hours_140k | p1of3 | 0–1793 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p1of3.lance` |
| s1 | hours_140k | p2of3 | 1793–3586 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p2of3.lance` |
| s2 | hours_140k | p3of3 | 3586–5381 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/hours_140k_vibevoice_asr_p3of3.lance` |
| s3 | convspeech | all | all 1,614 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/convspeech_vibevoice_asr.lance` |
| s4 | podcast p11-14 | all | all 1,858 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p11to14_vibevoice_asr.lance` |
| s5 | podcast p14-17 | all | all 1,902 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p14to17_vibevoice_asr.lance` |
| s6 | podcast p17-20 | p1of3 | 0–1868 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p1of3.lance` |
| s7 | podcast p17-20 | p2of3 | 1868–3736 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p2of3.lance` |
| s8 | podcast p17-20 | p3of3 | 3736–5606 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/asr/sft/podcast_10m_p17to20_vibevoice_asr_p3of3.lance` |

Naming convention: replace `whisperx` with `vibevoice_asr` in the source name, add `_p{i}of3` suffix for split tables.

### Pipeline

- **Pipeline**: `lax.projects.av_data_processing.audio.asr_vibevoice.pipeline_vllm.run_vibevoice_asr_vllm_pipeline`
- **Model**: VibeVoice-ASR (7B, Qwen2.5-7B backbone) — joint ASR + speaker diarization + timestamps
- **Output schema**: `segments` (JSON), `raw_text`, `num_speakers`, `num_segments`, `num_tokens`, `truncated`, `audio_duration_s`
- **Throughput**: ~3.5 rows/s per GPU, ~28 rows/s per 8-GPU node

### Cluster CR Names

| Cluster | CR Name |
|---------|---------|
| s0 | `dongguo-vibevoice-omniva-s0-b76d15` |
| s1 | `dongguo-vibevoice-omniva-s1-502299` |
| s2 | `dongguo-vibevoice-omniva-s2-dfb184` |
| s3 | `dongguo-vibevoice-omniva-s3-9ec3db` |
| s4 | `dongguo-vibevoice-omniva-s4-eef6d4` |
| s5 | `dongguo-vibevoice-omniva-s5-2e0b2e` |
| s6 | `dongguo-vibevoice-omniva-s6-fecd96` |
| s7 | `dongguo-vibevoice-omniva-s7-8c00e1` |
| s8 | `dongguo-vibevoice-omniva-s8-23eb17` |

### Launch Script

Script: `/Users/dongguo/Projects/adhoc/audio_caption/launch_vibevoice_asr_9jobs.sh`

```bash
# Launch all 9 jobs
bash launch_vibevoice_asr_9jobs.sh

# Launch specific clusters
bash launch_vibevoice_asr_9jobs.sh 3 4 5

# Launch single cluster
bash launch_vibevoice_asr_9jobs.sh 0
```

Each job runs:

```bash
source scripts/setup-ray-proxy.sh omniva-flyte <CR_NAME>

python -m lax.scripts.submit_ray_job --no-wait \
    --source_uri <SOURCE> \
    --destination_uri <DEST> \
    --pipeline_config lax.projects.av_data_processing.audio.asr_vibevoice.pipeline_vllm.run_vibevoice_asr_vllm_pipeline \
    --disable-tracking true \
    -u dongguo \
    [--partitions_range "START,END,1"]  # only for split tables
```

### Job IDs

**Attempt 1** (2026-04-16 00:11 PDT) — FAILED: missing `--runtime-env`, `No module named 'vibevoice'`.

| Cluster | Job ID (failed) |
|---------|--------|
| s0–s8 | `raysubmit_iHCsH5FRHk538s6A`, `raysubmit_9CNmrPKV3pf1LYbz`, `raysubmit_UyVXmv1dbrn2RSUF`, `raysubmit_vgNRejB8SffzYkwg`, `raysubmit_dxgqHvxjA4wR4PNj`, `raysubmit_UjU1SrS6VnzwLmT8`, `raysubmit_cr6n69K9L5MW92kW`, `raysubmit_GGabDwLh2HbdNAsH`, `raysubmit_73Wm6JGESBFx2AwC` |

**Attempt 2** (2026-04-16 00:25 PDT) — with `--runtime-env runtime_env_local.json` (real HF_TOKEN, not placeholder).

| Cluster | Dataset | Job ID |
|---------|---------|--------|
| s0 | hours_140k p1of3 | `raysubmit_dViV4vn2ENARrBUB` |
| s1 | hours_140k p2of3 | `raysubmit_7YWdaGekYDkkvPCS` |
| s2 | hours_140k p3of3 | `raysubmit_gKaYz51UMCDxp54S` |
| s3 | convspeech | `raysubmit_1dCMTYz6JywcHE13` |
| s4 | podcast p11-14 | `raysubmit_sFwQKfbEUAtVFxae` |
| s5 | podcast p14-17 | `raysubmit_aDDdJWmhedpKKfHc` |
| s6 | podcast p17-20 p1of3 | `raysubmit_x8mt5a5Uv93ewuNC` |
| s7 | podcast p17-20 p2of3 | `raysubmit_28MM1Kmrh4jqcSPE` |
| s8 | podcast p17-20 p3of3 | `raysubmit_h1pkcgaBi6mXCBHk` |

### Attempt History

**Attempt 1** (00:11 PDT) — FAILED: missing `--runtime-env`, `No module named 'vibevoice'`.

**Attempt 2** (00:25 PDT) — PARTIAL FAILURE: added `--runtime-env runtime_env_local.json`, but `partitions_range` format was wrong. Used `0,1793,1` (contiguous range) instead of `0,3,1` (modular). The format is `start_position,total,size` where fragments are selected by `fragment_index % total in [start, start+size)`. s0–s2 and s6–s8 processed only ~3 fragments each (~16K rows) and exited. s3/s4/s5 (no partitions) were unaffected.

**Attempt 3** (00:41 PDT) — Resubmitted s0–s2 and s6–s8 with corrected partition ranges (`0,3,1` / `1,3,1` / `2,3,1`). Deleted stale output data before resubmission. s3/s4/s5 continued from attempt 2.

| Cluster | Dataset | Attempt 3 Job ID |
|---------|---------|-----------------|
| s0 | hours_140k p1of3 | `raysubmit_gmPJG1hGBJiNfSyi` |
| s1 | hours_140k p2of3 | `raysubmit_sRfGyqjmqPCxE1x3` |
| s2 | hours_140k p3of3 | `raysubmit_T9L15MFHmDYa5TsD` |
| s3 | convspeech | `raysubmit_1dCMTYz6JywcHE13` (from attempt 2) |
| s4 | podcast p11-14 | `raysubmit_sFwQKfbEUAtVFxae` (from attempt 2) |
| s5 | podcast p14-17 | `raysubmit_aDDdJWmhedpKKfHc` (from attempt 2) |
| s6 | podcast p17-20 p1of3 | `raysubmit_87F1nYJGZsLPDrzA` |
| s7 | podcast p17-20 p2of3 | `raysubmit_rVTvJyEutpn8uYR1` |
| s8 | podcast p17-20 p3of3 | `raysubmit_r2RXzxTC8tLb3mfS` |

### Key Lesson: `partitions_range` format

The `--partitions_range` argument is `start_position,total,size` — NOT `start_fragment,end_fragment,step`. It uses modular arithmetic on fragment indices:

```python
for i in range(min(size, total - start)):
    src_fragment_ids_.extend(src_fragment_ids[start + i :: total])
```

For a 3-way split: `0,3,1` / `1,3,1` / `2,3,1` (every 3rd fragment, offset by 0/1/2).

### Status

- [x] Create s8 cluster (`dongguo-vibevoice-omniva-s8-23eb17`)
- [x] Launch all 9 jobs (attempt 3)
- [x] Verified s3 output: 331K rows, 90.4% non-empty, real transcripts
- [ ] All jobs complete
- [ ] Verify final outputs
