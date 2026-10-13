#!/bin/bash
#SBATCH --job-name=s2s-sevaS
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 21] SEVA orbits for the occlusion-prone subset, sharded so two run at once under the
# QOS cap. ~11 min per object. Objects whose orbit already exists are skipped, so a requeue
# or a second pass costs nothing.
#   SHARD=0 N_SHARDS=2 sbatch --export=ALL,SHARD=0,N_SHARDS=2 slurm/21_seva_subset.sbatch
set -eo pipefail
P=/home/isha.tariq/stable_virtual_camera; E=/home/isha.tariq/seva_to_shaper
PY=/home/isha.tariq/.conda/envs/svcam/bin/python
export TMPDIR=/tmp HF_HOME=/home/isha.tariq/.cache/huggingface SEVA_VAE_PATH="$P/assets/sd21base"
cd "$E"
echo "=== host $(hostname)  shard ${SHARD:-0}/${N_SHARDS:-1} ==="
if ! $PY -c "import sys,torch
try: torch.zeros(8,device='cuda'); print('GPU OK:',torch.cuda.get_device_name(0))
except Exception as e: print('GPU UNUSABLE:',e); sys.exit(1)"; then scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; exit 75; fi
# A one-sided capture often yields fewer than 8 usable views, and build_scenes.py names the
# split file after the count it actually kept -- so `--num_inputs 8` raises KeyError on those.
# Run one SEVA pass per view-count group, each with its own --num_inputs.
mkdir -p "$E/outputs_full" "$E/run_full"
ln -sfn "$E/outputs_full" "$E/run_full/work_dirs"   # demo.py resolves WORK_DIR from the cwd
cd "$E/run_full"
# The helper prints the groups correctly but returns non-zero on this node, and under
# `set -e` the assignment then kills the job with no message. Keep the output, record the
# status, and let the emptiness test below decide whether there is work.
GROUPS=$($PY "$E/scripts/seva_groups.py" --shard "${SHARD:-0}" --n_shards "${N_SHARDS:-1}") || GRP_RC=$?
echo "group helper exit status: ${GRP_RC:-0}"
# An `[ test ] && { ... }` here would return non-zero whenever there IS work, and under
# `set -e` that kills the job silently right after the GPU check. Use a plain if.
if [ -z "$GROUPS" ]; then echo 'nothing left to do'; exit 0; fi
for G in $GROUPS; do
  NIN=${G%%:*}; OBJECTS=${G#*:}
  echo; echo "### num_inputs $NIN : $OBJECTS"
  $PY "$P/third_party/stable-virtual-camera/demo.py" \
      --data_path "$E/inputs_full" --data_items "$OBJECTS" \
      --task img2trajvid --num_inputs "$NIN" --cfg 3.0,2.0 \
      --use_traj_prior True --chunk_strategy interp-gt --video_save_fps 20 || echo "!! group $NIN failed"
done
echo "### done"
