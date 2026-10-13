#!/bin/bash
#SBATCH --job-name=exp2-orbit2
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment2/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment2/logs/%x-%j.out

# Orbits for whichever objects are named after the script, cutout treatment only.
# The `context` treatment was a hedge against DUSt3R needing background texture;
# the first three objects showed it was not needed, so later objects skip it.
#   sbatch slurm/run_orbits_extra.sbatch BNB2727__bench ADT1292__woodentable
set -euo pipefail
PROJ=/home/isha.tariq/viewcrafter_orbits/experiment2
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
mkdir -p "$XDG_RUNTIME_DIR"

set +u; source /apps/local/conda_init.sh; conda activate viewcrafter; set -u
cd /home/isha.tariq/view_crafter/ViewCrafter
echo "=== host $(hostname) ==="; nvidia-smi -L || true
if ! python -c "
import sys, torch
try:
    torch.zeros(8, device='cuda'); print('GPU OK:', torch.cuda.get_device_name(0))
except Exception as e:
    print('GPU UNUSABLE:', type(e).__name__, e); sys.exit(1)
"; then scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true; exit 75; fi

time python "$PROJ/scripts/run_object_only_orbits.py" --seed 123 --treatments cutout --only "$@"
