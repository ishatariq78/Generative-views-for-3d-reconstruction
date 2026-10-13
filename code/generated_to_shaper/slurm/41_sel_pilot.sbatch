#!/bin/bash
#SBATCH --job-name=s2s-pilot
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
# [STEP 41] The pilot: 16 development objects x 5 selection conditions x 3 seeds.
#   SHARD=0 N_SHARDS=2 sbatch --export=ALL,SHARD=0,N_SHARDS=2 slurm/41_sel_pilot.sbatch
set -uo pipefail
E=/home/isha.tariq/seva_to_shaper
export S2S_SCENES=$E/inputs_full S2S_ORBITS=$E/outputs_full/demo/img2trajvid
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_assets; require_gpu; activate_shaper
echo "=== host $(hostname) shard ${SHARD:-0}/${N_SHARDS:-1} ==="
cd "$E"
OBJS=$(python -c "import json;print(' '.join(r['name'] for r in json.load(open('sel/dev_objects.json'))))")
python scripts/sel_score.py --objects $OBJS
python scripts/sel_run.py --objects_json sel/dev_objects.json --out sel/pilot \
    --shard "${SHARD:-0}" --n_shards "${N_SHARDS:-1}"
echo "### done"
