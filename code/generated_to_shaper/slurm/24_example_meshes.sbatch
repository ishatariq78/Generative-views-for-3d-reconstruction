#!/bin/bash
#SBATCH --job-name=s2s-exmesh
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 24] Export meshes AT THE REPORTED CONFIGURATION (24 views) for the example objects.
# The pipeline figures previously drew their mesh row from the 16-view runs, because the
# 24-view jobs were never given --save_meshes: the pictures and the headline number came
# from different configurations. This fixes that.
#   sbatch slurm/24_example_meshes.sbatch
set -uo pipefail
E=/home/isha.tariq/seva_to_shaper
export S2S_SCENES=$E/inputs_full S2S_ORBITS=$E/outputs_full/demo/img2trajvid
export S2S_PKLS=$E/outputs_sub/pkls S2S_SETS=$E/objects_subset_built.json
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_assets; require_gpu; activate_shaper
echo "=== host $(hostname) ==="
mkdir -p "$S2S_PKLS"
cd "$E"
OBJS="BNB2909__armchair_2 BNB1045__nightstand_2 ADT1292__nightstand HWD0045__desktable HWD0045__ottoman"
for OBJ in $OBJS; do
  echo; echo "### $OBJ"
  python scripts/make_pkls_views.py --only "$OBJ" --conds views || { echo "!! pkl failed"; continue; }
  python scripts/run_shaper_nviews.py \
      --input_pkls "/home/isha.tariq/shapeR_testing/data/$OBJ.pkl" "$S2S_PKLS/${OBJ}__views.pkl" \
      --out_dir "$E/outputs_sub/meshes24" --seed 0 --num_images 24 --save_meshes
  mv "$E/outputs_sub/meshes24/per_object/$OBJ.json" "$E/outputs_sub/meshes24/per_object/${OBJ}__slam.json" 2>/dev/null
  mv "$E/outputs_sub/meshes24/per_object/$OBJ.glb"  "$E/outputs_sub/meshes24/per_object/${OBJ}__slam.glb" 2>/dev/null
  mv "$E/outputs_sub/meshes24/per_object/GT__$OBJ.glb" "$E/outputs_sub/meshes24/per_object/GT__${OBJ}__slam.glb" 2>/dev/null
  rm -f "$S2S_PKLS/${OBJ}__views.pkl"
done
echo "### done"
