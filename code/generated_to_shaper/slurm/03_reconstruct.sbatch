#!/bin/bash
#SBATCH --job-name=s2s-recon
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 3] Both reconstructors over the SEVA orbits, kvtracker env:
#   Pi3 batch (80 frames, ~7 s/object) and KV-Tracker online, keyframe every frame
#   (~4 min/object; KVT_MAX_KF lifts the upstream 20-keyframe cap, KVT_KEEP_LAST_PCD keeps
#   only the newest cumulative PLY -- both env-gated local patches in main.py).
#   sbatch slurm/03_reconstruct.sbatch
set -eo pipefail
export TMPDIR=/tmp; export HF_HOME=/home/isha.tariq/.cache/huggingface
export KVT_KEEP_LAST_PCD=1; export KVT_MAX_KF=80
E=/home/isha.tariq/seva_to_shaper; PY=/home/isha.tariq/.conda/envs/kvtracker/bin/python
echo "=== host $(hostname) ==="
if ! $PY -c "import sys,torch
try: torch.zeros(8,device='cuda'); print('GPU OK:',torch.cuda.get_device_name(0))
except Exception as e: print('GPU UNUSABLE:',e); sys.exit(1)"; then scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; exit 75; fi
cd "$E"
echo "### frame sets"; $PY scripts/make_frame_sets.py
echo "### Pi3 batch"; $PY scripts/run_pi3.py
echo "### KV-Tracker (kf_auto 1)"
cd /home/isha.tariq/kv_tracker_testing/third_party/kv_tracker    # SAM2 checkpoint path is repo-relative
for cfg in "$E"/outputs/kv_inputs/*.yaml; do
  name=$(basename "$cfg" .yaml); out="$E/outputs/kv/$name"; rm -rf "$out"
  echo; echo "### $name"
  timeout 2400 $PY main.py "$cfg" --cam_only --export_pcd --resize_dim 518 --kf_auto 1 2>&1 | grep -v "Seg:" || echo "!! $name failed/timeout"
  ls "$out"/pcd_*.ply 2>/dev/null; du -sh "$out" 2>/dev/null
done
cd "$E"
echo "### collect"; $PY scripts/collect_kv.py
echo "### done"
