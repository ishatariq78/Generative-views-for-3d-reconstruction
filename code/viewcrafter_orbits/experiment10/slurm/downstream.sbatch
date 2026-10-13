#!/bin/bash
#SBATCH --job-name=exp10-down
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment10/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment10/logs/%x-%j.out
# frames -> VGGT -> mesh fit (experiment 3's scorer, --out_dir here) -> SLAM comparison
set -euo pipefail
P=/home/isha.tariq/viewcrafter_orbits; E=$P/experiment10
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp; export OMP_NUM_THREADS=8
export PYOPENGL_PLATFORM=egl; export HF_HOME=/home/isha.tariq/shapeR_testing/cache/huggingface
mkdir -p "$XDG_RUNTIME_DIR"
PY=/home/isha.tariq/.conda/envs/shaper/bin/python
echo "=== host $(hostname) ==="
if ! $PY -c "
import sys, torch
try:
    torch.zeros(8, device='cuda'); print('GPU OK:', torch.cuda.get_device_name(0))
except Exception as e:
    print('GPU UNUSABLE:', type(e).__name__, e); sys.exit(1)
"; then scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true; exit 75; fi
echo "### frames";  $PY "$E/scripts/extract_frames.py"
echo "### VGGT";    $PY "$P/experiment3/scripts/run_vggt.py" --in_dir "$E/frames" --out_dir "$E/vggt"
echo "### mesh fit (upright fitter)"; $PY "$P/experiment3/scripts/align_and_score.py" --out_dir "$E/vggt"
echo "### SLAM comparison"; $PY "$E/scripts/score_slam.py"
