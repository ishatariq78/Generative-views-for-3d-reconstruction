#!/bin/bash
#SBATCH --job-name=s2s-sevaB
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
# [STEP 21b] SEVA over the subset, driven from Python (see scripts/run_seva_subset.py).
#   SHARD=0 N_SHARDS=2 sbatch --export=ALL,SHARD=0,N_SHARDS=2 slurm/21b_seva_subset.sbatch
set -uo pipefail                 # deliberately NOT -e: the driver reports failures itself
PY=/home/isha.tariq/.conda/envs/svcam/bin/python
echo "=== host $(hostname) shard ${SHARD:-0}/${N_SHARDS:-1} ==="
$PY -c "import torch; torch.zeros(8,device='cuda'); print('GPU OK:',torch.cuda.get_device_name(0))" || {
  echo 'GPU unusable'; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null; exit 75; }
$PY /home/isha.tariq/seva_to_shaper/scripts/run_seva_subset.py --shard "${SHARD:-0}" --n_shards "${N_SHARDS:-1}" --objects "${OBJLIST:-objects_full_built.json}"
echo "### done"
