#!/bin/bash
#SBATCH --job-name=exp2-views
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment2/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment2/logs/%x-%j.out

# =============================================================================
# [STEP 1] Build object-only reference views.
#   sbatch slurm/build_views.sbatch
# Needs a GPU: the object silhouette comes from rasterising its reference mesh
# through pyrender's EGL context. ~5 min.
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
    /home/isha.tariq/viewcrafter_orbits/experiment2/scripts/build_object_only_views.py \
    ADT1292__chair BNB1045__sofa BNB1723__bed
