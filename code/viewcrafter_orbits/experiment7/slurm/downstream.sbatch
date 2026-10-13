#!/bin/bash
#SBATCH --job-name=exp7-down
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment7/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment7/logs/%x-%j.out
set -euo pipefail
P=/home/isha.tariq/viewcrafter_orbits
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp; export PYOPENGL_PLATFORM=egl
export HF_HOME=/home/isha.tariq/shapeR_testing/cache/huggingface
mkdir -p "$XDG_RUNTIME_DIR"
PY=/home/isha.tariq/.conda/envs/shaper/bin/python
$PY "$P/experiment7/scripts/extract_frames.py"
$PY "$P/experiment3/scripts/run_vggt.py"
