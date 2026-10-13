#!/bin/bash
#SBATCH --job-name=vco-turntable
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/shared/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/shared/logs/%x-%j.out

# =============================================================================
# Render ShapeR's predicted and reference meshes as side-by-side 360 turntables,
# so ShapeR's answer can be watched next to ViewCrafter's.
#   sbatch slurm/render_turntables.sbatch
# Needs a GPU for pyrender's EGL context. ~2 min.
# =============================================================================
# cscc-gpu-p is the only partition this account can actually submit to: `long`
# advertises AllowAccounts=ALL but runs on QOS gpu-12, which the association does
# not carry, so `sbatch --partition=long` is rejected as an invalid QOS. When
# cscc-gpu-p sits in (Resources), the job simply waits.
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)
export TMPDIR=/tmp
export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR"

/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/viewcrafter_orbits/shared/scripts/render_shaper_turntables.py \
    ADT1292__chair BNB1045__sofa BNB1723__bed
