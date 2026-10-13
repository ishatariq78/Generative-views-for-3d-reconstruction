#!/bin/bash
#SBATCH --job-name=exp2-views2
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment2/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment2/logs/%x-%j.out

# Object-only reference views for whichever samples are named on the command
# line, e.g.  sbatch slurm/build_views_extra.sbatch HWD0045__cabinet
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR"
/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/viewcrafter_orbits/experiment2/scripts/build_object_only_views.py "$@"
