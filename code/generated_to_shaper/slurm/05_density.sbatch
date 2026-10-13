#!/bin/bash
#SBATCH --job-name=s2s-dens
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 5] The density control: each cloud at exactly the object's SLAM point count
#          (conditions pi3n / kvn), same 3 repeats, results merged by scripts/summarise.py.
#   sbatch slurm/05_density.sbatch      # run after 04
set -eo pipefail
E=/home/isha.tariq/seva_to_shaper; REPS=${REPS:-3}
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_assets; require_gpu; activate_shaper
echo "=== host $(hostname) ==="
cd "$E"; echo "### pkls"; python scripts/make_pkls_density.py
mapfile -t PKLS < <(find "$E/outputs/pkls" -maxdepth 1 \( -name "*__pi3n.pkl" -o -name "*__kvn.pkl" \) | sort)
echo "### ShapeR on ${#PKLS[@]} density-matched pkls x $REPS repeats"
cd "$REPO"
for k in $(seq 1 "$REPS"); do
  R="$E/outputs/shaper/rep$k"; mkdir -p "$R/per_object"
  echo; echo "### repeat $k"
  python evaluation/eval.py --input_pkls "${PKLS[@]}" --is_local_path --config balance \
      --output_dir "$R/per_object" --eval_output "$R/eval_results_density.json" --save_meshes
done
cd "$E"; echo "### summarise"; python scripts/summarise.py
echo "### figures"; python scripts/render.py
echo "### done"
