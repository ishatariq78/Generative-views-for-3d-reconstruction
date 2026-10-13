#!/bin/bash
#SBATCH --job-name=exp5-paths
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment5/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment5/logs/%x-%j.out

# =============================================================================
# [STEP 1] Orbit each object along six different camera paths.
#   sbatch slurm/run_paths.sbatch
# 18 runs (3 objects x 6 paths) sharing one loaded model, ~45 min.
# Reads experiment 2's cutout reference images; writes to outputs/.
# =============================================================================
set -euo pipefail
PROJ=/home/isha.tariq/viewcrafter_orbits/experiment5
VC=/home/isha.tariq/view_crafter/ViewCrafter

export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
mkdir -p "$XDG_RUNTIME_DIR"

n=$(ls /home/isha.tariq/viewcrafter_orbits/experiment2/inputs/*__cutout.png 2>/dev/null | wc -l)
[ "$n" -ge 1 ] || { echo "ERROR: experiment 2's cutout reference images are missing"; exit 1; }

set +u   # conda's activate hooks are not -u clean
source /apps/local/conda_init.sh
conda activate viewcrafter
set -u

cd "$VC"
echo "=== host $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi -L || true
if ! python -c "
import sys, torch
try:
    torch.zeros(8, device='cuda')
    print('GPU OK:', torch.cuda.get_device_name(0), '| torch', torch.__version__)
except Exception as e:
    print('GPU UNUSABLE:', type(e).__name__, e); sys.exit(1)
"; then
    scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true
    exit 75
fi

time python "$PROJ/scripts/run_camera_paths.py" --seed 123

echo
echo "=== videos ==="
find "$PROJ/outputs" -name 'diffusion0.mp4' -printf '%p\n' | sort
