#!/bin/bash
#SBATCH --job-name=exp10-sparse
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment10/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment10/logs/%x-%j.out
set -euo pipefail
PROJ=/home/isha.tariq/viewcrafter_orbits/experiment10
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
mkdir -p "$XDG_RUNTIME_DIR"
set +u; source /apps/local/conda_init.sh; conda activate viewcrafter; set -u
cd /home/isha.tariq/view_crafter/ViewCrafter
echo "=== host $(hostname) ==="
if ! python -c "
import sys, torch
try:
    torch.zeros(8, device=\"cuda\"); print(\"GPU OK:\", torch.cuda.get_device_name(0))
except Exception as e:
    print(\"GPU UNUSABLE:\", type(e).__name__, e); sys.exit(1)
"; then scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true; exit 75; fi
time python "$PROJ/scripts/run_sparse.py" --seed 123 "$@"
