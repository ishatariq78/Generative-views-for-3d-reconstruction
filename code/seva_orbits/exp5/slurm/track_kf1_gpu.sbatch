#!/bin/bash
#SBATCH --job-name=exp5-kf1
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp5/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp5/logs/%x-%j.out
# [STEP 1b] KV-Tracker over the five SEVA orbits with a keyframe on EVERY frame (--kf_auto 1).
#           KVT_KEEP_LAST_PCD=1 keeps only the newest cumulative PLY; KVT_MAX_KF=80 lifts the
#           upstream 20-keyframe cap (both local patches in main.py).
set -eo pipefail
export TMPDIR=/tmp; export HF_HOME=/home/isha.tariq/.cache/huggingface; export KVT_KEEP_LAST_PCD=1; export KVT_MAX_KF=80
E5=/home/isha.tariq/stable_virtual_camera/exp5; PY=/home/isha.tariq/.conda/envs/kvtracker/bin/python
echo "=== host $(hostname) ==="
if ! $PY -c "import sys,torch
try: torch.zeros(8,device='cuda'); print('GPU OK:',torch.cuda.get_device_name(0))
except Exception as e: print('GPU UNUSABLE:',e); sys.exit(1)"; then scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; exit 75; fi
cd /home/isha.tariq/kv_tracker_testing/third_party/kv_tracker
for cfg in "$E5"/configs/*__kf1.yaml; do
  name=$(basename "$cfg" .yaml); out="$E5/outputs/$name"; rm -rf "$out"
  echo; echo "### $name  (kf_auto 1)"
  timeout 2400 $PY main.py "$cfg" --cam_only --export_pcd --resize_dim 518 --kf_auto 1 2>&1 | grep -v "Seg:" || echo "!! $name failed/timeout"
  ls "$out"/pcd_*.ply; du -sh "$out"
done
echo; echo "### done"
