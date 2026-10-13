#!/bin/bash
#SBATCH --job-name=exp3-frames
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment3/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment3/logs/%x-%j.out

# =============================================================================
# [STEP 1] Build the three frame sets per object.
#   sbatch slurm/build_frames.sbatch
# Needs a GPU: the `real` condition rasterises the object's mesh into 25 frames
# through pyrender's EGL context. ~10 min.
# =============================================================================
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)
export TMPDIR=/tmp
export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR"

/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/viewcrafter_orbits/experiment3/scripts/build_frame_sets.py
