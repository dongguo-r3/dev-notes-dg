# Audio Data Processing Logs

> **What this note is for:** Run logs for specific audio data pipeline jobs — job IDs, cluster assignments, throughput, failures, and outcomes. Each entry is a discrete pipeline execution against a dataset.
>
> | Date | Pipeline | Dataset | Outcome |
> |---|---|---|---|
> | 2026-03-18 | Fidelity (bandwidth + AES + SED) | `whisperx__eng_v1` (podcast_10m, 10K rows) | ✅ 10K rows in 36 min, 8×H100 kiwi-flyte |
> | 2026-04-08 | Fidelity | `internal_audio_v1` (~92M rows, 8 partitions) | 🟡 Running — p0 on kiwi (Sydney), p1-p7 on omniva (US, ~30% throughput) |
> | 2026-04-17 | Speech Metadata v2 (pitch/gender/emotion/age) | 5 SFT tables (~66M rows) + whisperx__multilingual_v1_compacted (221.8M rows, 12-way split) | 🟡 Running — 17 clusters on omniva (5 SFT + 12 multilingual partitions), batch_size=2048, --no-randomized |

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

---

## 2026-04-17: Speech Metadata v2 — SFT Tables (Group 62)

### Goal

Run the new speech metadata v2 pipeline (gender + pitch + volume + speaking rate + emotion + age, commercial-only) on the 5 SFT lance tables (table-2 through table-6 from internal dashboard group 62), same input tables as the 2026-04-16 VibeVoice run. Table-1 (americanrhetoric, 28K rows) is small enough to run locally if needed.

### Source & Destination Tables

| Cluster | Source Table | Source URI | Rows | Dest URI |
|---------|-------------|------------|------|----------|
| s0 | hours_140k | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance` | 21,745,714 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/hours_140k_speech_metadata.lance` |
| s1 | convspeech | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/convspeech/asr/prefiltered_english__whisperx.lance` | 6,514,097 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/convspeech_speech_metadata.lance` |
| s2 | podcast p11-14 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p11to14_whisperx_clean.lance` | 7,499,644 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p11to14_speech_metadata.lance` |
| s3 | podcast p14-17 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p14to17_whisperx_clean.lance` | 7,670,431 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p14to17_speech_metadata.lance` |
| s4 | podcast p17-20 | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance` | 22,655,625 | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p17to20_speech_metadata.lance` |

**No partition splits** — pipeline at ~500-700 rows/s per 8-GPU cluster is fast enough to process 22M-row tables in 10-15 hours per cluster.

### Cluster Assignment

Reused the existing omniva-flyte clusters from the 2026-04-16 VibeVoice run (s0-s4). The remaining s5-s8 continue to run VibeVoice on the partitioned tables.

| Cluster | CR Name |
|---------|---------|
| s0 | `dongguo-vibevoice-omniva-s0-b76d15` |
| s1 | `dongguo-vibevoice-omniva-s1-502299` |
| s2 | `dongguo-vibevoice-omniva-s2-dfb184` |
| s3 | `dongguo-vibevoice-omniva-s3-9ec3db` |
| s4 | `dongguo-vibevoice-omniva-s4-eef6d4` |

### Pipeline

- **Pipeline**: `lax.projects.av_data_processing.audio.audio_metadata.speech_metadata_pipeline.run_speech_metadata_pipeline_gpu`
- **Runtime env**: `lax/projects/av_data_processing/audio/audio_metadata/speech_metadata_runtime_env.json`
  - `numpy<2.3` (avoids numba conflict pulled in transitively by torchcrepe → resampy)
  - `torchcrepe>=0.0.22`, `speechbrain>=1.0.0`, `joblib>=1.3.0`
  - `onnxruntime-gpu>=1.23.0` (critical — prevents CPU-only `onnxruntime` from overriding)
- **Models loaded per actor (~2.5 GB VRAM)**:
  - torchcrepe tiny (3M params) for pitch
  - gender_prithiv.onnx (wav2vec2-base, 95M)
  - emotion_dpngtm.onnx (wav2vec2-base, 95M) — batch=2 windowing (first 8s + last 8s)
  - ECAPA-TDNN + SVR for age (15M + SVR)
- **Output schema** (16 columns): pitch (4), volume (2), speaking_rate (3), gender (2), emotion_dpngtm (3), age_years, age_group
- **Resource per actor**: `num_cpus=1, num_gpus=0.25, memory=6GB` → 4 actors per GPU × 8 GPUs = 32 actors per cluster
- **Read batch size**: `read_control_row_based_batch_size=2048` (key tuning lever — tested locally at 24 rows/s per actor, dominant lever on cluster throughput)

### Local Benchmark (before launch)

On 5× H100 GPUs (one per table, single actor each) with 4096 samples:

| Dataset | Throughput (single actor) | Per-row latency |
|---------|---------------------------|-----------------|
| americanrhetoric | 24.65 rows/s | 41 ms |
| hours_140k | 24.62 rows/s | 41 ms |
| convspeech | 24.24 rows/s | 42 ms |
| podcast_clean | 25.10 rows/s | 40 ms |
| podcast_wild | 25.23 rows/s | 40 ms |

Per-model breakdown (ms/row, steady state):

| Step | Latency | % |
|------|---------|---|
| gender (ONNX, wav2vec2-base) | 5 | 16% |
| pitch (optimized torchcrepe + GPU decode) | 6 | 19% |
| emotion_dpngtm (ONNX, batch=2 windowing) | 6 | 19% |
| age (ECAPA + SVR) | 13 | 42% |
| decode + resample + volume + rate | 1 | 4% |

### Key Optimizations Applied

1. **Switched gender model** from alefiury wav2vec2-large (316M) to prithivMLmods wav2vec2-base (95M) — ~5 ms/row savings with 100% agreement on the 5-dataset benchmark.
2. **Fixed 8s input window** for all ONNX models (tile-repeat for short clips, first 8s for long) — avoids ORT dynamic-shape replanning (~15-20x speedup per ONNX call).
3. **Batch=2 windowing** for emotion_dpngtm only — captures variation across long clips (first 8s + last 8s, averaged probabilities). Not applied to Audeering (fixed batch=1 ONNX, would require 2x loop).
4. **Custom GPU decode for torchcrepe** — skip Python postprocess overhead (`torchcrepe.predict` → `torchcrepe.infer` + manual argmax on GPU). 15x speedup vs stock.
5. **Dropped emotion_superb** — redundant with dpngtm (42% label agreement), no meaningful added signal.
6. **Dropped Audeering from default pipeline** — non-commercial license (CC-BY-NC-SA). Available as `run_speech_metadata_pipeline_gpu_with_audeering` variant for research comparisons only.

### S3 Checkpoints

```
s3://ai-lumalabs-checkpoints-ap-se-2/dongguo/speech_metadata/
  commercial/
    gender_prithiv.onnx + .onnx.data           (362 MB)
    emotion_dpngtm.onnx + .onnx.data           (362 MB)
    age_ecapa/                                  (85 MB)
    age_svr_model.joblib + age_svr_scaler.joblib
  non_commercial/
    audeering_age_gender.onnx                  (1.2 GB)
    audeering_emotion_dim.onnx                 (630 MB)
```

### Launch Script

`lax/projects/av_data_processing/audio/audio_metadata/launch_speech_metadata_5jobs.sh`

```bash
# Verification run on one cluster (8192 rows)
bash launch_speech_metadata_5jobs.sh --verify 1

# Launch single cluster (full table)
bash launch_speech_metadata_5jobs.sh 0

# Launch all 5 full jobs
bash launch_speech_metadata_5jobs.sh
```

### Job IDs

**Attempt 1** (2026-04-17 07:39 PDT) — verification on s1 with `--limit 8192`. Job ran slowly (limit + randomized sampling forces cross-fragment reads). Stopped manually to avoid wasting cluster time.

- s1 verify: `raysubmit_V59zkuhHFUK6Pdj4` (stopped)

**Attempt 2** (2026-04-17 07:49–07:57 PDT) — full-table runs with `--no-randomized`:

| Cluster | Dataset | Job ID | Notes |
|---------|---------|--------|-------|
| s0 | hours_140k | `raysubmit_VWqSuD6Yan3yjs6v` | Launched 07:49, pipeline uploaded with old `batch_size=1024` |
| s1 | convspeech | `raysubmit_paKM7EAr6EpCdjRZ` | Launched 07:56, `batch_size=2048` |
| s2 | podcast p11-14 | `raysubmit_Ny9YeHH8mUCL8LPw` | Launched 07:56, `batch_size=2048` |
| s3 | podcast p14-17 | `raysubmit_6ZjL95CAtdUiQkJb` | Launched 07:57, `batch_size=2048` |
| s4 | podcast p17-20 | `raysubmit_yeg5RNY6vzCHHchK` | Launched 07:57, `batch_size=2048` |

### Key Lessons

1. **`--limit` with default `--randomized` is slow on large tables** — it forces cross-fragment sampling which reads sparse data from across the lance dataset. Use `--no-randomized` (first N rows, sequential) for verification runs.
2. **`read_control_row_based_batch_size` is the dominant cluster-throughput lever** (per fidelity pipeline's STATUS.md: 256 → 2048 was ~10x). Always default to 2048 unless the table has very large audio_bytes payloads that risk Arrow 2GB overflow.
3. **`onnxruntime-gpu` must be explicitly pinned in runtime_env** — otherwise transitive deps (e.g. from `audonnx` or any other package) can install the CPU-only `onnxruntime` and silently fall back to CPU with no visible error (just no `CUDAExecutionProvider` in the provider list).
4. **Reusing clusters from VibeVoice run** avoided waiting for cluster creation. Useful pattern when doing follow-up metadata extraction on the same tables.

### Status

- [x] Local benchmark on 5 datasets (4096 samples each) — 24 rows/s per actor
- [x] Upload all checkpoints to S3
- [x] Launch all 5 jobs (attempt 2)
- [ ] Verify cluster throughput matches local benchmark × 4 actors × 8 GPUs (~500-700 rows/s per cluster projected)
- [ ] All jobs complete
- [ ] Verify final outputs (row counts, schema, sample predictions)
- [ ] Consider re-launching s0 with `batch_size=2048` if it's noticeably slower than s1-s4

---

## 2026-04-17: Speech Metadata v2 — 17-Job Run Inventory

Unified table of all 17 speech_metadata_v2 jobs running on omniva-flyte as of 2026-04-17 08:26 PDT. Two families:

- **5 SFT jobs** (whole-table): one cluster per table, no partition split.
- **12 multilingual_v1 jobs**: 12-way fragment split of the 221.8M-row podcast table, one partition per idle cluster.

### Full Run Table

| Cluster (CR name) | Job ID | Source Dataset | Partition | Target Lance Table |
|-------------------|--------|----------------|-----------|---------------------|
| `dongguo-vibevoice-omniva-s0-b76d15` | `raysubmit_VWqSuD6Yan3yjs6v` | `audio/sft/hours_140k/asr/prefiltered_english__whisperx.lance` (21.7M rows) | all (whole table) | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/hours_140k_speech_metadata.lance` |
| `dongguo-vibevoice-omniva-s1-502299` | `raysubmit_paKM7EAr6EpCdjRZ` | `audio/sft/convspeech/asr/prefiltered_english__whisperx.lance` (6.5M rows) | all (whole table) | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/convspeech_speech_metadata.lance` |
| `dongguo-vibevoice-omniva-s2-dfb184` | `raysubmit_Ny9YeHH8mUCL8LPw` | `audio/sft/podcast_10m/asr/podcast_10m_p11to14_whisperx_clean.lance` (7.5M rows) | all (whole table) | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p11to14_speech_metadata.lance` |
| `dongguo-vibevoice-omniva-s3-9ec3db` | `raysubmit_6ZjL95CAtdUiQkJb` | `audio/sft/podcast_10m/asr/podcast_10m_p14to17_whisperx_clean.lance` (7.7M rows) | all (whole table) | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p14to17_speech_metadata.lance` |
| `dongguo-vibevoice-omniva-s4-eef6d4` | `raysubmit_yeg5RNY6vzCHHchK` | `audio/sft/podcast_10m/asr/podcast_10m_p17to20_whisperx_wild.lance` (22.7M rows) | all (whole table) | `s3://ai-lumalabs-datasets-ap-se-2/dongguo/lax/metadata/sft/podcast_10m_p17to20_speech_metadata.lance` |
| `dongguo-metadata-s0-6a0225` | `raysubmit_PNBuTxkxSEwud2cV` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` (221.8M rows) | `--partitions_range 0,12,1` (p0of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p0of12.lance` |
| `dongguo-metadata-s1-fed951` | `raysubmit_s82UjvNqwZR2RbEe` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `--partitions_range 1,12,1` (p1of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p1of12.lance` |
| `dongguo-metadata-s2-de5369` | `raysubmit_6MvCZWshq2GMBT2s` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `--partitions_range 2,12,1` (p2of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p2of12.lance` |
| `dongguo-metadata-s3-c857b1` | `raysubmit_d53KsLTV4Z8rDrpM` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `--partitions_range 3,12,1` (p3of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p3of12.lance` |
| `dongguo-metadata-s4-720f5b` | `raysubmit_PHtVFSiJvQ4GjZtT` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `--partitions_range 4,12,1` (p4of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p4of12.lance` |
| `dongguo-metadata-s5-d38cd4` | `raysubmit_jXVxu9Sy57wccWne` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `--partitions_range 5,12,1` (p5of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p5of12.lance` |
| `dongguo-metadata-s6-30970a` | `raysubmit_mAhxXZV6J8LVa6gV` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `--partitions_range 6,12,1` (p6of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p6of12.lance` |
| `dongguo-metadata-s7-ffd6fc` | `raysubmit_gbpXui9gPpwvYcuw` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `--partitions_range 7,12,1` (p7of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p7of12.lance` |
| `dongguo-vibevoice-omniva-s5-2e0b2e` | `raysubmit_jt2X3sL6pANRjdGt` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `--partitions_range 8,12,1` (p8of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p8of12.lance` |
| `dongguo-vibevoice-omniva-s6-fecd96` | `raysubmit_m5FYHLZDzgii3DAn` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `--partitions_range 9,12,1` (p9of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p9of12.lance` |
| `dongguo-vibevoice-omniva-s7-8c00e1` | `raysubmit_vhskJN81bnnfmZSi` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `--partitions_range 10,12,1` (p10of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p10of12.lance` |
| `dongguo-vibevoice-omniva-s8-23eb17` | `raysubmit_NQt7whEbEGujyJMB` | `audio/pretrain/podcast_10m/asr/whisperx__multilingual_v1_compacted.lance` | `--partitions_range 11,12,1` (p11of12, ~18.5M rows) | `s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1_p11of12.lance` |

### Pipeline & Config (all 17 jobs)

- **Pipeline**: `lax.projects.av_data_processing.audio.audio_metadata.speech_metadata_pipeline.run_speech_metadata_pipeline_gpu`
- **Runtime env**: `lax/projects/av_data_processing/audio/audio_metadata/speech_metadata_runtime_env.json` (`numpy<2.3`, `torchcrepe`, `speechbrain`, `joblib`, `onnxruntime-gpu`)
- **Launch scripts**:
    - 5 SFT: `launch_speech_metadata_5jobs.sh`
    - 12 multilingual: `launch_speech_metadata_multilingual_12jobs.sh`
- **Common flags**: `--commit_percentage 0.01 --disable-tracking true --no-randomized`
- **Reader batch size**: `read_control_row_based_batch_size=2048` (except s0 which was launched with old `batch_size=1024` before the pipeline update at 07:56)

### Cluster Resource Total

All 17 clusters × 8 H100 GPUs = **136 GPUs** fully utilized. At ~500 rows/s per cluster the aggregate throughput is ~8,500 rows/s across all jobs. Expected finish times:

- SFT jobs (~6-23M rows each): 3-13 hours per cluster
- Multilingual partitions (~18.5M rows each): ~10 hours per partition → all 12 finish within ~10-12h

### Post-processing

After all 12 multilingual partitions finish, merge with:

```bash
python -m lax.scripts.infra.concat_tables \
    --src_uris "s3://...speech_metadata_multilingual_v1_p0of12.lance,s3://...p1of12.lance,...,s3://...p11of12.lance" \
    --dst_uri "s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/speech_metadata_multilingual_v1.lance"
```

The 5 SFT outputs are standalone and do not need merging.

---

## 2026-04-17 08:47 PDT: Repartition s0 and s4 (6-way split of the 2 biggest SFT tables)

### Rationale

s0 (hours_140k, 21.7M rows) and s4 (podcast p17-20, 22.7M rows) had the longest ETAs (~15-16h each) because they're single-cluster jobs on the two largest tables. Meanwhile s1-s3 (convspeech, podcast p11-14, podcast p14-17) are projected to finish in ~4-5h, after which their 8-GPU clusters become idle.

Strategy: split each of hours_140k and podcast_p17to20 into **6 fragment partitions** so that the work can be spread across multiple clusters when s1-s3 free up. Each partition is ~1/6 of the source = ~3.6M-3.8M rows.

- s0 restarted with `--partitions_range 0,6,2` (partitions 0+1 = 1/3 of hours_140k, ~7.2M rows)
- s4 restarted with `--partitions_range 0,6,2` (partitions 0+1 = 1/3 of podcast p17-20, ~7.6M rows)
- Remaining 4+4 = **8 partitions** (partitions 2/3/4/5 of each table) will be distributed to s1, s2, s3 (and any other idle clusters) as those jobs finish.

### Checkpointing / resume

Both jobs keep their original destination URIs (`hours_140k_speech_metadata.lance` and `podcast_10m_p17to20_speech_metadata.lance`). LAX's checkpoint mechanism reads the destination on startup and skips source fragments whose `original_row_id`s have already been committed. Progress from the aborted run (~5% committed for s0, ~4% for s4) is preserved — the relaunched jobs simply skip the done fragments in their new partitions.

### Job Changes

| Cluster | Previous Job (stopped) | New Job | Partition | Rows in partition |
|---------|-----------------------|---------|-----------|--------------------|
| s0 | `raysubmit_VWqSuD6Yan3yjs6v` (STOPPED at ~5% done) | `raysubmit_u3cXsqFMgCdtVprE` | `--partitions_range 0,6,2` | ~7.2M (1/3 of hours_140k) |
| s4 | `raysubmit_yeg5RNY6vzCHHchK` (STOPPED at ~4% done) | `raysubmit_RynKYsTH2fUv1K8W` | `--partitions_range 0,6,2` | ~7.6M (1/3 of podcast p17-20) |

### Remaining partition assignments (pending)

When s1, s2, s3 finish (~4-5h from initial launch), these 8 partitions need to be distributed:

**hours_140k remaining partitions** (4 slots):
- `--partitions_range 2,6,1` (partition 2, ~3.6M rows)
- `--partitions_range 3,6,1` (partition 3, ~3.6M rows)
- `--partitions_range 4,6,1` (partition 4, ~3.6M rows)
- `--partitions_range 5,6,1` (partition 5, ~3.6M rows)

**podcast p17-20 remaining partitions** (4 slots):
- `--partitions_range 2,6,1` (partition 2, ~3.8M rows)
- `--partitions_range 3,6,1` (partition 3, ~3.8M rows)
- `--partitions_range 4,6,1` (partition 4, ~3.8M rows)
- `--partitions_range 5,6,1` (partition 5, ~3.8M rows)

Total remaining work across 8 partitions: ~29.6M rows. With 3 helper clusters (s1, s2, s3) running at ~380 rows/s each: 29.6M / (3 × 380) / 3600 = ~7.2h after s1-s3 finish.

### Projected total wall time

- Without repartition: max(s0 ETA, s4 ETA) ≈ ~16h
- With repartition (if we distribute all 8 remaining partitions across s0, s1, s2, s3, s4 when s1-s3 free up at T+5h):
  - s0 and s4 handle 1/3 of their own tables each: 7.2M / 380 / 3600 = ~5.3h → finish at T+5.3h
  - Remaining 4+4 partitions on s1+s2+s3: 29.6M / (3 × 380) / 3600 = 7.2h (distributed) → or split evenly, each handles ~9.9M → ~7.2h on each
  - Total: max(T+5.3, T+5+7.2) = **T+12.2h** (vs 16h baseline, ~4h savings)

Plan assumes further distribution of partitions 2-5 across s1/s2/s3 once they finish.

---

## 2026-04-17 Afternoon: Multilingual V1 complete, SFT partition dispatch

### Multilingual V1 — all 12 partitions SUCCEEDED

By ~19:42 UTC, all 12 multilingual partition jobs had committed 221.84M rows (100% of the source), with throughput averaging ~420 rows/s per cluster. All Ray jobs transitioned to SUCCEEDED by ~20:00 UTC. The 12 multilingual output tables are at:

```
s3://ai-lumalabs-datasets-ap-se-2-lance/audio/pretrain/podcast_10m/metadata/
    speech_metadata_multilingual_v1_p{0..11}of12.lance
```

Ready to merge with `lax.scripts.infra.concat_tables` into `speech_metadata_multilingual_v1.lance` when convenient.

### SFT partition re-dispatch (8 partitions on 2 big tables)

After s0 (hours_140k) and s4 (podcast p17-20) were repartitioned at 08:47 with `partitions_range 0,6,2` (1/3 each), the plan was to dispatch the remaining 8 partitions (2, 3, 4, 5 of each table) to clusters as they freed up.

**First dispatch attempt (19:46 UTC, 5 jobs on vibevoice s0-s4)**: 3 of 5 failed with
`LSUFatalError: ... _rowid_mappings_tmp_.../original_row_id/*.index.json ... 404 NoSuchKey`.

**Root cause**: When multiple jobs share the same destination Lance table and all start within seconds of each other, LAX's resume logic builds `_rowid_mappings_tmp_*` folders in parallel — this causes a race where one job tries to read another job's partially-written manifest files.

**Fix**: serialize subsequent launches to the same dest with ~3 min spacing so each new job's resume phase completes before the next starts.

### Failed & relaunched jobs

| Cluster | Failed Job | Reason | Relaunch Job | Status |
|---------|-----------|--------|--------------|--------|
| vibevoice-s0 | `raysubmit_LWZxpUgPKKqnFHGt` | rowid_mappings race | `raysubmit_5hV5PLAEYPPBPS3E` | RUNNING |
| vibevoice-s2 | `raysubmit_11WRmFeAhcxtfXvG` | rowid_mappings race | `raysubmit_ApraG7urL9zUhm8f` | RUNNING |
| vibevoice-s3 | `raysubmit_VGyMv5KzpsD2MEHe` | rowid_mappings race | `raysubmit_cQSFxY9E3KUD9zgM` | RUNNING |

### Final 8-partition roster (SFT side)

| Cluster | Job ID | Table | Partition | Status |
|---------|--------|-------|-----------|--------|
| vibevoice-s0 | `raysubmit_5hV5PLAEYPPBPS3E` | hours_140k | `2,6,1` | RUNNING |
| vibevoice-s1 | `raysubmit_X2u43hgz8qSeXhjE` | hours_140k | `3,6,1` | ✅ SUCCEEDED |
| vibevoice-s2 | `raysubmit_ApraG7urL9zUhm8f` | podcast_p17to20 | `2,6,1` | RUNNING |
| vibevoice-s3 | `raysubmit_cQSFxY9E3KUD9zgM` | podcast_p17to20 | `3,6,1` | RUNNING |
| vibevoice-s4 | `raysubmit_PahNjgh64BxDdTXr` | podcast_p17to20 | `4,6,1` | RUNNING |
| metadata-s0 | `raysubmit_PDH1aB95TZbaPgvx` | hours_140k | `4,6,1` | RUNNING (launched 22:20) |
| metadata-s1 | `raysubmit_fr2hDgwWuP4zTGXd` | hours_140k | `5,6,1` | RUNNING (launched 22:24) |
| metadata-s2 | `raysubmit_ADLXTfAuab6thAY9` | podcast_p17to20 | `5,6,1` | RUNNING (launched 22:28) |

With partitions 0+1 from the initial `0,6,2` runs (SUCCEEDED), the 2 destination tables should have complete 6/6 coverage once these 8 jobs finish.

### Lesson learned

When multiple jobs write to the same Lance destination:

1. **Launch them in batches of 1-2 max per destination**, waiting ~3 min between batches.
2. **LAX's checkpoint/resume preserves prior work** across partition_range changes — no need to use separate destination tables per partition (which would waste already-committed rows).
3. Once a job is past its initial resume phase (`_rowid_mappings_tmp_*` folders built), concurrent writes from other jobs are fine.

A future improvement: the LAX framework could serialize resume-manifest construction across jobs sharing a destination, or fall back gracefully when a read 404s during the mapping phase (retry with backoff).

### Idle clusters (available for further work)

After the 8 SFT partition jobs are dispatched, the following multilingual clusters are idle and available:

- `dongguo-metadata-s3-c857b1`, `s4-720f5b`, `s5-d38cd4`, `s6-30970a`, `s7-ffd6fc` (5 clusters)
- `dongguo-vibevoice-omniva-s5-2e0b2e`, `s6-fecd96`, `s7-8c00e1`, `s8-23eb17` (4 clusters)

= **9 idle clusters / 72 H100 GPUs** available for next workload.

### Projected finish

At ~380 rows/s per cluster, each remaining partition is ~3.6M-3.8M rows → ~2.5-2.8h per partition. All 8 SFT partition jobs should finish by **~T+2.5-3h from now** (~01:00 UTC on 2026-04-18).

Total speech_metadata_v2 coverage when all done:
- **5 SFT tables** (americanrhetoric skipped, ~66M rows): COMPLETE
- **1 multilingual_v1 table** (221.8M rows): COMPLETE (12 partitions committed)

Grand total: **~288M rows** of audio enriched with gender/pitch/volume/speaking-rate/emotion/age in ~14h of wall time on 17 clusters.
