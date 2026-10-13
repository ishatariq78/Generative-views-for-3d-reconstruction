#!/bin/bash
#SBATCH --job-name=exp3-score
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment3/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment3/logs/%x-%j.out

# =============================================================================
# [STEP 3] Align each VGGT cloud to the object's reference mesh and score it.
#   sbatch slurm/score.sbatch
# CPU only -- this is ICP and KD-tree work, no GPU involved. ~20 min.
# =============================================================================
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)
export TMPDIR=/tmp
mkdir -p "$XDG_RUNTIME_DIR"

/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/viewcrafter_orbits/experiment3/scripts/align_and_score.py
