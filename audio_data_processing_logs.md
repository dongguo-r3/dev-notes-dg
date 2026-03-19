# Audio Data Processing Logs

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
- **Status**: FAILED → resubmitted as `raysubmit_GByqaBLBHyRF3jEX`
- **Failure**: Dataset has 54,905 fragments — randomized read with limit=3M was too slow. Resubmitted with `--no-randomized`.

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
ray job status raysubmit_GByqaBLBHyRF3jEX \
  --address "$RAY_ADDRESS" \
  --headers "{\"Authorization\": \"Bearer $RAY_PROXY_TOKEN\"}"
```

**Clean up when done**:

```bash
cd projects/kuma && source .venv/bin/activate
flytecli ray-cluster --name dongguo-fidelity --delete --yes
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
