#!/bin/bash
#SBATCH --job-name=exp6-down
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment6/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment6/logs/%x-%j.out

# [STEP 3] Sample the sparse videos, reconstruct them with VGGT, draw the fits.
#   sbatch slurm/downstream_gpu.sbatch
# The `sparse` sets join experiment 3's inputs as a fourth condition, so the same
# scripts score them alongside gen360, gen45 and real.
set -euo pipefail
P=/home/isha.tariq/viewcrafter_orbits
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
export PYOPENGL_PLATFORM=egl
export HF_HOME=/home/isha.tariq/shapeR_testing/cache/huggingface
mkdir -p "$XDG_RUNTIME_DIR"
PY=/home/isha.tariq/.conda/envs/shaper/bin/python

echo "=== host $(hostname) ==="
if ! $PY -c "
import sys, torch
try:
    torch.zeros(8, device='cuda'); print('GPU OK:', torch.cuda.get_device_name(0))
except Exception as e:
    print('GPU UNUSABLE:', type(e).__name__, e); sys.exit(1)
"; then scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true; exit 75; fi

echo; echo "### sampling the sparse videos"
$PY "$P/experiment6/scripts/extract_sparse_frames.py"
echo; echo "### VGGT over every frame set"
$PY "$P/experiment3/scripts/run_vggt.py"
echo; echo "### drawing the alignments"
$PY "$P/experiment3/scripts/render_alignment.py"
