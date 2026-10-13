#!/bin/bash
#SBATCH --job-name=exp3-render
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp3/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp3/logs/%x-%j.out
# [STEP 1] VGGT over SEVA's orbits (10 new sets), then draw every alignment sheet.
#   sbatch exp3/slurm/vggt_gpu.sbatch
# Runs the ViewCrafter-orbit workspace's own scripts unchanged, so the clouds are made exactly
# the way every ViewCrafter condition's were.
set -eo pipefail
P=/home/isha.tariq/viewcrafter_orbits
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp; export PYOPENGL_PLATFORM=egl
export HF_HOME=/home/isha.tariq/shapeR_testing/cache/huggingface   # where VGGT's weights live
mkdir -p "$XDG_RUNTIME_DIR"
PY=/home/isha.tariq/.conda/envs/shaper/bin/python
echo "=== host $(hostname) ==="
if ! $PY -c "import sys,torch
try: torch.zeros(8,device='cuda'); print('GPU OK:',torch.cuda.get_device_name(0))
except Exception as e: print('GPU UNUSABLE:',e); sys.exit(1)"; then scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; exit 75; fi
SETS=$($PY -c "import json;print(' '.join(json.load(open('/home/isha.tariq/stable_virtual_camera/exp3/sets.json'))))")
echo; echo "### alignment sheets"
$PY "$P/experiment3/scripts/render_alignment.py"
