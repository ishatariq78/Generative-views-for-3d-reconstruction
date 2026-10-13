#!/bin/bash
#SBATCH --job-name=exp1-seva
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp1/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp1/logs/%x-%j.out

# [STEP 2] Ask SEVA for the 360 deg orbit of each object.
#   sbatch exp1/slurm/run_seva.sbatch
#
# `img2trajvid` is the mode that takes P posed inputs AND our own target cameras.
# P=8 is the sparse-view regime, so cfg 3,2 and chunk_strategy interp-gt per the
# upstream guide. No --L_short: the images are already square 576, so SEVA's
# default centre crop is a no-op and nothing gets resampled twice.
set -eo pipefail
P=/home/isha.tariq/stable_virtual_camera
PY=/home/isha.tariq/.conda/envs/svcam/bin/python
export TMPDIR=/tmp
# XDG_CACHE_HOME may point at node-local /scratch, which silently moves HF_HOME
# there: no cached weights, no token, a 401 that looks like revoked access.
# The cache lives on /home, the only shared filesystem.
export HF_HOME=/home/isha.tariq/.cache/huggingface
# Upstream's VAE repo was pulled from HF; use the hash-verified local copy.
export SEVA_VAE_PATH="$P/assets/sd21base"
# demo.py's WORK_DIR is relative to the *working directory*, and it is invoked by
# absolute path, so running from exp1/ (where work_dirs -> outputs) keeps this
# experiment's results out of the top-level smoke-test outputs.
cd "$P/exp1"

OBJECTS=$(/home/isha.tariq/.conda/envs/svcam/bin/python -c "
import json, pathlib
m = json.loads(pathlib.Path('$P/exp1/inputs/manifest.json').read_text())
print(','.join(r['object'] for r in m))")

echo "host: $(hostname)"
echo "objects: $OBJECTS"
$PY "$P/third_party/stable-virtual-camera/demo.py" \
    --data_path "$P/exp1/inputs" \
    --data_items "$OBJECTS" \
    --task img2trajvid \
    --num_inputs 8 \
    --cfg 3.0,2.0 \
    --use_traj_prior True \
    --chunk_strategy interp-gt \
    --video_save_fps 20
