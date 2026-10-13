#!/bin/bash
#SBATCH --job-name=exp4-render
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment4/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment4/logs/%x-%j.out

# =============================================================================
# [STEP 3] Draw SLAM points and VGGT points over the same reference mesh.
#   sbatch slurm/render.sbatch
# Needs a GPU for pyrender's EGL context. ~5 min.
# =============================================================================
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR"
/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/viewcrafter_orbits/experiment4/scripts/render_comparison.py
