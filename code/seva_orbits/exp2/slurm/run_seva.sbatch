#!/bin/bash
#SBATCH --job-name=exp2-seva
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp2/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp2/logs/%x-%j.out
# [STEP 2] generate the orbit for every occluded scene (10 runs, ~9 min each).
# Same flags and seed as exp1, so the only difference from exp1's clean run is
# the magenta ball -- anything that changes in the output is caused by it.
set -eo pipefail
P=/home/isha.tariq/stable_virtual_camera
PY=/home/isha.tariq/.conda/envs/svcam/bin/python
export TMPDIR=/tmp
# XDG_CACHE_HOME may point at node-local /scratch, which silently moves HF_HOME
# there: no cached weights, no token, a 401 that looks like revoked access.
# The cache lives on /home, the only shared filesystem.
export HF_HOME=/home/isha.tariq/.cache/huggingface
export SEVA_VAE_PATH="$P/assets/sd21base"
cd "$P/exp2"
SCENES=$($PY -c "import json;print(','.join(r['scene'] for r in json.load(open('inputs/manifest.json'))))")
echo "host: $(hostname)"; echo "scenes: $SCENES"
$PY "$P/third_party/stable-virtual-camera/demo.py" \
    --data_path "$P/exp2/inputs" --data_items "$SCENES" \
    --task img2trajvid --num_inputs 8 --cfg 3.0,2.0 \
    --use_traj_prior True --chunk_strategy interp-gt --video_save_fps 20
