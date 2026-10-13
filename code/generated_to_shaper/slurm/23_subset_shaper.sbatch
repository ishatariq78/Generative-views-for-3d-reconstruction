#!/bin/bash
#SBATCH --job-name=s2s-sub
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 23] The subset result: 40 occlusion-prone objects, baseline vs generated views, at
# the 24-view budget the sweep chose. Views pkls are built in batches and deleted after
# each batch is scored -- 40 x ~120 MB would not fit on a /home already at 98%.
#   SHARD=0 N_SHARDS=2 sbatch --export=ALL,SHARD=0,N_SHARDS=2 slurm/23_subset_shaper.sbatch
set -uo pipefail
E=/home/isha.tariq/seva_to_shaper
export S2S_SCENES=$E/inputs_full S2S_ORBITS=$E/outputs_full/demo/img2trajvid
export S2S_PKLS=$E/outputs_sub/pkls S2S_SETS=${OBJLIST:-$E/objects_full_built.json}
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_assets; require_gpu; activate_shaper
echo "=== host $(hostname) shard ${SHARD:-0}/${N_SHARDS:-1} ==="
mkdir -p "$S2S_PKLS" "$E/outputs_sub"
cd "$E"
python scripts/subset_runner.py --shard "${SHARD:-0}" --n_shards "${N_SHARDS:-1}" --num_images 24 --objects "${OBJLIST:-objects_full_built.json}"
echo "### done"
