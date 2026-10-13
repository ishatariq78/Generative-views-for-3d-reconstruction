#!/bin/bash
#SBATCH --job-name=svc-smoke1
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/logs/%x-%j.out

# [STEP 2a] One image -> a full 360 deg orbit, via the `img2trajvid_s-prob` task.
#   sbatch slurm/smoke_single_view.sbatch
#
# The `orbit` preset really is a whole turn: get_arc_horizontal_w2cs defaults to
# degree=360.0 (seva/geometry.py:391), unlike ViewCrafter's documented +-45.
# Reads from assets/single_view/, which holds one frame lifted out of the demo
# scenes so the run needs nothing but the repo.
set -eo pipefail
P=/home/isha.tariq/stable_virtual_camera
PY=/home/isha.tariq/.conda/envs/svcam/bin/python
export TMPDIR=/tmp
# XDG_CACHE_HOME may point at node-local /scratch, which silently moves HF_HOME
# there: no cached weights, no token, a 401 that looks like revoked access.
# The cache lives on /home, the only shared filesystem.
export HF_HOME=/home/isha.tariq/.cache/huggingface
# Upstream's VAE repo (stabilityai/stable-diffusion-2-1-base) was pulled from HF
# and 404s. Point seva at the hash-verified local copy instead.
export SEVA_VAE_PATH="$P/assets/sd21base"
cd "$P/third_party/stable-virtual-camera"   # WORK_DIR is relative; work_dirs -> ../../outputs

echo "host: $(hostname)"
$PY demo.py \
    --data_path "$P/assets/single_view" \
    --task img2trajvid_s-prob \
    --replace_or_include_input True \
    --traj_prior orbit \
    --cfg 4.0,2.0 \
    --guider 1,2 \
    --num_targets 80 \
    --L_short 576 \
    --use_traj_prior True \
    --chunk_strategy interp \
    --save_subdir smoke
