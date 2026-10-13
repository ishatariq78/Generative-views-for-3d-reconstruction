#!/bin/bash
#SBATCH --job-name=s2s-kvsub
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
# [STEP 30] KV-Tracker over 10 subset orbits, for the "cloud instead of SLAM points" test.
set -uo pipefail
PY=/home/isha.tariq/.conda/envs/kvtracker/bin/python
echo "=== host $(hostname) ==="
$PY -c "import torch; torch.zeros(8,device='cuda'); print('GPU OK:',torch.cuda.get_device_name(0))" || {
  scontrol requeue "$SLURM_JOB_ID" 2>/dev/null; exit 75; }
$PY /home/isha.tariq/seva_to_shaper/scripts/kv_subset_run.py
echo "### done"
