#!/bin/bash
#SBATCH --job-name=exp3-render
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment3/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment3/logs/%x-%j.out

# =============================================================================
# [STEP 4] Draw each aligned cloud against its reference mesh, to check the fit.
#   sbatch slurm/render_alignment.sbatch
# Needs a GPU for pyrender's EGL context. ~3 min.
# =============================================================================
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)
export TMPDIR=/tmp
export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR"

/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/viewcrafter_orbits/experiment3/scripts/render_alignment.py
