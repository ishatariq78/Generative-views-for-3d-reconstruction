#!/bin/bash
#SBATCH --job-name=exp4-pi3
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp4/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp4/logs/%x-%j.out
# [STEP 1] Pi3 over the same 10 SEVA frame sets exp3 gave VGGT, then the shared
#          alignment sheets over everything (they iterate outputs/).
set -eo pipefail
export TMPDIR=/tmp; export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR"
export HF_HOME=/home/isha.tariq/.cache/huggingface        # Pi3 weights live here, on /home
KV=/home/isha.tariq/.conda/envs/kvtracker/bin/python
echo "=== host $(hostname) ==="
# gpu-54 also handed out a GPU CUDA could not initialise (2026-09-15); requeue on failure.
if ! $KV -c "import sys,torch
try: torch.zeros(8,device='cuda'); print('GPU OK:',torch.cuda.get_device_name(0))
except Exception as e: print('GPU UNUSABLE:',e); sys.exit(1)"; then scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; exit 75; fi
SETS=$($KV -c "import json;print(' '.join(json.load(open('/home/isha.tariq/stable_virtual_camera/exp4/sets.json'))))")
cd /home/isha.tariq/stable_virtual_camera/exp4
$KV scripts/run_pi3.py --only $SETS
echo; echo "### alignment sheets (shaper env)"
export PYOPENGL_PLATFORM=egl
/home/isha.tariq/.conda/envs/shaper/bin/python /home/isha.tariq/viewcrafter_orbits/experiment3/scripts/render_alignment.py
