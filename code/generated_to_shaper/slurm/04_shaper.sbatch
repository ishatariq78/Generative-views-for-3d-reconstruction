#!/bin/bash
#SBATCH --job-name=s2s-shaper
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 4] Align the clouds, write the per-condition pkls, run ShapeR on all of them.
#   sbatch slurm/04_shaper.sbatch
# ShapeR's rectified-flow sampler is unseeded, so every pkl is run REPS times
# (outputs/shaper/rep<k>/) and the comparison uses the mean +- std over repeats.
# Same protocol as the benchmark reproduction: --config balance, eval.py's own metrics.
set -eo pipefail
E=/home/isha.tariq/seva_to_shaper; REPS=${REPS:-3}
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_assets; require_gpu; activate_shaper
echo "=== host $(hostname) ==="
cd "$E"
echo "### align";  python scripts/align_clouds.py
echo "### pkls";   python scripts/make_pkls.py
mapfile -t PKLS < <(find "$E/outputs/pkls" -maxdepth 1 -name "*.pkl" | sort)
echo "### ShapeR on ${#PKLS[@]} pkls x $REPS repeats"
cd "$REPO"
for k in $(seq 1 "$REPS"); do
  R="$E/outputs/shaper/rep$k"; mkdir -p "$R/per_object"
  echo; echo "### repeat $k"
  python evaluation/eval.py --input_pkls "${PKLS[@]}" --is_local_path --config balance \
      --output_dir "$R/per_object" --eval_output "$R/eval_results.json" --save_meshes --save_visualization
done
cd "$E"
echo "### summarise"; python scripts/summarise.py
echo "### figures";   python scripts/render.py
echo "### done"
