#!/bin/bash
#SBATCH --job-name=tt-extra
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/shared/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/shared/logs/%x-%j.out
# Turntables for whichever samples are named after the script.
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR"
/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/viewcrafter_orbits/shared/scripts/render_shaper_turntables.py "$@"
