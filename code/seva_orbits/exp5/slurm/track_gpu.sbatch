#!/bin/bash
#SBATCH --job-name=exp5-kvt
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp5/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp5/logs/%x-%j.out
# [STEP 1] KV-Tracker over the five SEVA 80-frame orbits (exp1), camera-only, headless,
#          exporting the keyframe point cloud; keyframe every 5 frames (16 kf) and every 10 (8 kf).
set -eo pipefail
export TMPDIR=/tmp; export HF_HOME=/home/isha.tariq/.cache/huggingface
E5=/home/isha.tariq/stable_virtual_camera/exp5
PY=/home/isha.tariq/.conda/envs/kvtracker/bin/python
echo "=== host $(hostname) ==="
if ! $PY -c "import sys,torch
try: torch.zeros(8,device='cuda'); print('GPU OK:',torch.cuda.get_device_name(0))
except Exception as e: print('GPU UNUSABLE:',e); sys.exit(1)"; then scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; exit 75; fi
cd /home/isha.tariq/kv_tracker_testing/third_party/kv_tracker     # SAM2 checkpoint path is relative to here
for cfg in "$E5"/configs/*.yaml; do
  name=$(basename "$cfg" .yaml); kf=${name##*__kf}
  echo; echo "### $name  (kf_auto $kf)"
  # timeout: a raise in the main loop leaves the data-stream child alive and the job hangs
  timeout 600 $PY main.py "$cfg" --cam_only --export_pcd --resize_dim 518 --kf_auto "$kf" 2>&1 | grep -v "Seg:" || echo "!! $name failed/timeout"
  ls "$E5/outputs/$name" | tr '\n' ' '; echo
done
echo; echo "### done"
