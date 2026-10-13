#!/bin/bash
#SBATCH --job-name=exp3-frames2
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment3/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment3/logs/%x-%j.out

# Frame sets for whichever objects are named after the script.
#   sbatch slurm/build_frames_extra.sbatch BNB2727__bench
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR"
/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/viewcrafter_orbits/experiment3/scripts/build_frame_sets.py --only "$@"
