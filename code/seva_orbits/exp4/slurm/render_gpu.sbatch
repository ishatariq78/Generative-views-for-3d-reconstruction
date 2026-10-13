#!/bin/bash
#SBATCH --job-name=exp4-render
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp4/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp4/logs/%x-%j.out
# [STEP 3] VGGT-vs-Pi3 comparison renders (pyrender/EGL).
set -eo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp; export PYOPENGL_PLATFORM=egl; mkdir -p "$XDG_RUNTIME_DIR"
cd /home/isha.tariq/stable_virtual_camera/exp4
# alignment sheets need points_aligned.npy, which align_and_score writes -- so after scoring
/home/isha.tariq/.conda/envs/shaper/bin/python /home/isha.tariq/viewcrafter_orbits/experiment3/scripts/render_alignment.py
/home/isha.tariq/.conda/envs/shaper/bin/python scripts/render_comparison_pi3.py
