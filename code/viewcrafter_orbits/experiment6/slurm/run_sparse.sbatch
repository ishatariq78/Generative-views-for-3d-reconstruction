#!/bin/bash
#SBATCH --job-name=exp6-sparse
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment6/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment6/logs/%x-%j.out

# [STEP 2] ViewCrafter in sparse-view mode, one run per object.
#   sbatch slurm/run_sparse.sbatch
# ~15 clips of diffusion per object, so roughly 12 min each. Needs
# checkpoints/model_sparse.ckpt (a different checkpoint from single-view mode).
set -euo pipefail
PROJ=/home/isha.tariq/viewcrafter_orbits/experiment6
VC=/home/isha.tariq/view_crafter/ViewCrafter
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
# The 10 GB diffusion model, DUSt3R and the fused point cloud all live on the card
# at once, and the failure was fragmentation as much as volume: 37.9 GB reserved
# against 34.6 GB actually allocated. torch 1.13 has no expandable_segments, so
# cap the split size instead.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
mkdir -p "$XDG_RUNTIME_DIR"

CKPT="$VC/checkpoints/model_sparse.ckpt"
sz=$(stat -c%s "$CKPT" 2>/dev/null || echo 0)
[ "$sz" -gt 9000000000 ] || { echo "ERROR: $CKPT missing or incomplete ($sz bytes)"; exit 1; }

set +u; source /apps/local/conda_init.sh; conda activate viewcrafter; set -u
cd "$VC"
echo "=== host $(hostname) ==="; nvidia-smi -L || true
if ! python -c "
import sys, torch
try:
    torch.zeros(8, device='cuda'); print('GPU OK:', torch.cuda.get_device_name(0))
except Exception as e:
    print('GPU UNUSABLE:', type(e).__name__, e); sys.exit(1)
"; then scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true; exit 75; fi

time python "$PROJ/scripts/run_sparse_viewcrafter.py" --seed 123
echo; echo "=== videos ==="; find "$PROJ/outputs" -name 'diffusion.mp4' -printf '%p\n' | sort
