#!/bin/bash
#SBATCH --job-name=exp1-score
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp1/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp1/logs/%x-%j.out

# [STEP 3] Score each generated orbit against the object's true silhouette.
#   sbatch exp1/slurm/score.sbatch
# GPU needed: the ground truth comes from rasterising the reference mesh.
set -eo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR"
echo "host: $(hostname)"
/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/stable_virtual_camera/exp1/scripts/score_orbits.py
