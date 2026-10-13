#!/bin/bash
#SBATCH --job-name=s2s-nodes24
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
# [STEP 27] Re-render the per-stage figures with the mesh row taken from the SAME 24-view
# run the deck quotes, for the subset objects that have those meshes.
set -uo pipefail
E=/home/isha.tariq/seva_to_shaper
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_gpu; activate_shaper
cd "$E"
OBJS=$(for f in outputs_sub/meshes24/per_object/*__views.glb; do o=$(basename "$f" __views.glb);
  [ -f "outputs_sub/meshes24/per_object/GT__${o}__slam.glb" ] && echo "$o"; done | tr '\n' ' ')
echo "=== host $(hostname) objects: $OBJS"
python scripts/render_pipeline.py --objects $OBJS --seed 0 --conds slam views \
  --scene_root "$E/inputs_full" --orbit_root "$E/outputs_full/demo/img2trajvid" \
  --mesh_dir "$E/outputs_sub/meshes24/per_object"
echo "### done"
