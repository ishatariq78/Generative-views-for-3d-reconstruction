#!/bin/bash
#SBATCH --job-name=exp6-sparse1
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment6/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment6/logs/%x-%j.out
# Sparse mode for named objects only, in a fresh process — which is what the
# fifth object needs after four have fragmented the card.
#   sbatch slurm/run_sparse_one.sbatch BNB2727__bench
set -euo pipefail
PROJ=/home/isha.tariq/viewcrafter_orbits/experiment6
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
mkdir -p "$XDG_RUNTIME_DIR"
set +u; source /apps/local/conda_init.sh; conda activate viewcrafter; set -u
cd /home/isha.tariq/view_crafter/ViewCrafter
echo "=== host $(hostname) ==="
time python "$PROJ/scripts/run_sparse_viewcrafter.py" --seed 123 --only "$@"
