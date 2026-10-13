#!/bin/bash
#SBATCH --job-name=svc-smoke8
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/logs/%x-%j.out

# [STEP 2b] Several posed views -> a target trajectory, via the `img2trajvid` task.
#   sbatch slurm/smoke_multi_view.sbatch
#
# This is the mode the experiments will actually use: input views + their poses
# + a target camera path, with no ordering assumed between the two sets. The
# 8 inputs and 225 targets come from garden_flythrough's train_test_split_8.json.
# P<=8 is the sparse-view regime, so cfg 3,2 and chunk_strategy interp-gt.
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
cd "$P/third_party/stable-virtual-camera"

echo "host: $(hostname)"
$PY demo.py \
    --data_path "$P/assets/assets_demo_cli" \
    --data_items garden_flythrough \
    --task img2trajvid \
    --num_inputs 8 \
    --cfg 3.0,2.0 \
    --L_short 576 \
    --use_traj_prior True \
    --chunk_strategy interp-gt \
    --save_subdir smoke
