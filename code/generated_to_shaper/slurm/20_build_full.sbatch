#!/bin/bash
#SBATCH --job-name=s2s-bfull
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 20] Build SEVA scenes (8 cleaned posed views + an 80-camera orbit) for the WHOLE
# 178-object benchmark. Sharded so it can run 2-up under the QOS cap; --skip_done makes it
# resumable, and a per-object failure is caught so one bad object cannot stop the sweep.
#   SHARD=0 N_SHARDS=2 sbatch --export=ALL,SHARD=0,N_SHARDS=2 slurm/20_build_full.sbatch
set -eo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp; export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR"
PY=/home/isha.tariq/.conda/envs/shaper/bin/python
echo "=== host $(hostname)  shard ${SHARD:-0}/${N_SHARDS:-1} ==="
if ! $PY -c "import sys,torch
try: torch.zeros(8,device='cuda'); print('GPU OK:',torch.cuda.get_device_name(0))
except Exception as e: print('GPU UNUSABLE:',e); sys.exit(1)"; then scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; exit 75; fi
cd /home/isha.tariq/seva_to_shaper
$PY scripts/build_scenes.py --objects_json objects_full.json \
    --out_dir /home/isha.tariq/seva_to_shaper/inputs_full \
    --shard "${SHARD:-0}" --n_shards "${N_SHARDS:-1}" --skip_done
echo "### done"
