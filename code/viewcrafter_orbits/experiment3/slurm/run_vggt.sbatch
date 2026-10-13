#!/bin/bash
#SBATCH --job-name=exp3-vggt
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment3/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment3/logs/%x-%j.out

# =============================================================================
# [STEP 2] Run VGGT over all nine frame sets.
#   sbatch slurm/run_vggt.sbatch
# VGGT is feed-forward, so this is fast -- seconds per set once the 1B model is
# loaded. Needs inputs/ from slurm/build_frames.sbatch.
#
# Uses the existing `shaper` env: it already carries torch 2.7.1+cu128, einops,
# safetensors and huggingface_hub, which is everything VGGT imports. The repo
# pins torch 2.3.1 but does not depend on anything removed since.
# =============================================================================
set -euo pipefail
PROJ=/home/isha.tariq/viewcrafter_orbits/experiment3
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)
export TMPDIR=/tmp
export HF_HOME=/home/isha.tariq/shapeR_testing/cache/huggingface
mkdir -p "$XDG_RUNTIME_DIR"

n=$(find "$PROJ/inputs" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$n" -ge 1 ] || { echo "ERROR: no frame sets in $PROJ/inputs"; \
                    echo "       run: sbatch slurm/build_frames.sbatch"; exit 1; }
echo "=== host $(hostname)  $n frame sets ==="

PY=/home/isha.tariq/.conda/envs/shaper/bin/python
if ! $PY -c "
import sys, torch
try:
    torch.zeros(8, device='cuda')
    print('GPU OK:', torch.cuda.get_device_name(0))
except Exception as e:
    print('GPU UNUSABLE:', type(e).__name__, e); sys.exit(1)
"; then
    nvidia-smi -L || true
    scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true
    exit 75
fi

time $PY "$PROJ/scripts/run_vggt.py"
