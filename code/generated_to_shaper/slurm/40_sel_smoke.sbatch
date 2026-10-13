#!/bin/bash
#SBATCH --job-name=s2s-smoke
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
# [STEP 40] Two-object smoke test of the controlled-view selection pipeline before the pilot.
set -uo pipefail
E=/home/isha.tariq/seva_to_shaper
export S2S_SCENES=$E/inputs_full S2S_ORBITS=$E/outputs_full/demo/img2trajvid
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_assets; require_gpu; activate_shaper
echo "=== host $(hostname) ==="
cd "$E"
python scripts/sel_score.py --objects HWD0045__desktable BNB1045__nightstand_2
printf '[{"name":"HWD0045__desktable"},{"name":"BNB1045__nightstand_2"}]' > sel/smoke_objects.json
python scripts/sel_run.py --objects_json sel/smoke_objects.json --out sel/smoke --seeds 0
echo "### done"
