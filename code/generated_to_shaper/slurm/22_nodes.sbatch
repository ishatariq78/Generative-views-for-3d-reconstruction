#!/bin/bash
#SBATCH --job-name=s2s-nodes
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 22] The per-stage figure: for each object, what every node of the pipeline outputs.
#   sbatch slurm/22_nodes.sbatch
set -eo pipefail
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_gpu; activate_shaper
echo "=== host $(hostname) ==="
cd /home/isha.tariq/seva_to_shaper
python scripts/render_pipeline.py --seed 0 --conds slam views \
    --objects BNB1182__stool BNB1045__barstool HWD0045__desktable BNB1045__wovenstool HWD0045__woodentable
echo "### done"
