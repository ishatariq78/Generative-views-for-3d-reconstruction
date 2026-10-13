#!/bin/bash
#SBATCH --job-name=s2s-seva
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 2] Stable Virtual Camera: 8 posed views + 80 target cameras -> 80 orbit frames per
#          object (same settings as ~/stable_virtual_camera/exp1: img2trajvid, cfg 3,2,
#          interp-gt). ~10 min per object.
#   sbatch slurm/02_run_seva.sbatch                 # every object in inputs/manifest.json
#   OBJECTS=BNB1045__wovenstool sbatch slurm/02_run_seva.sbatch
set -eo pipefail
P=/home/isha.tariq/stable_virtual_camera
E=/home/isha.tariq/seva_to_shaper
PY=/home/isha.tariq/.conda/envs/svcam/bin/python
export TMPDIR=/tmp
export HF_HOME=/home/isha.tariq/.cache/huggingface      # XDG_CACHE_HOME may point at node-local /scratch
export SEVA_VAE_PATH="$P/assets/sd21base"                # upstream's VAE repo is gone; hash-verified local copy
cd "$E"                                                  # demo.py's WORK_DIR is relative: work_dirs -> outputs
echo "=== host $(hostname) ==="
if ! $PY -c "import sys,torch
try: torch.zeros(8,device='cuda'); print('GPU OK:',torch.cuda.get_device_name(0))
except Exception as e: print('GPU UNUSABLE:',e); sys.exit(1)"; then scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; exit 75; fi
if [ -z "$OBJECTS" ]; then
  OBJECTS=$($PY -c "
import json, pathlib
m = json.loads(pathlib.Path('$E/inputs/manifest.json').read_text())
print(','.join(r['object'] for r in m))")
fi
echo "objects: $OBJECTS"
$PY "$P/third_party/stable-virtual-camera/demo.py" \
    --data_path "$E/inputs" \
    --data_items "$OBJECTS" \
    --task img2trajvid \
    --num_inputs 8 \
    --cfg 3.0,2.0 \
    --use_traj_prior True \
    --chunk_strategy interp-gt \
    --video_save_fps 20
echo "### copying videos"
for o in ${OBJECTS//,/ }; do
  d="$E/outputs/demo/img2trajvid/$o"
  [ -f "$d/samples-rgb.mp4" ] && cp "$d/samples-rgb.mp4" "$E/videos/${o}_orbit360.mp4"
  [ -f "$d/input.mp4" ] && cp "$d/input.mp4" "$E/videos/${o}_inputs.mp4"
done
ls -la "$E/videos"
echo "### done"
