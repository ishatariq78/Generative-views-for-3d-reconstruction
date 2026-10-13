#!/bin/bash
#SBATCH --job-name=s2s-pvfig
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --dependency=afterany:288628
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
# [STEP 29] The qualitative points-vs-images figure, once the points-conditioned meshes exist.
# Chained on the mesh job so it cannot run against missing inputs.
set -uo pipefail
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_gpu; activate_shaper
cd /home/isha.tariq/seva_to_shaper
OBJS=$(for f in outputs/seeded/s0/per_object/*__pi3.glb; do [ -e "$f" ] || continue; basename "$f" __pi3.glb; done | tr '\n' ' ')
echo "=== host $(hostname) objects: $OBJS"
[ -n "$OBJS" ] && python scripts/render_points_vs_views.py --objects $OBJS
echo "### done"
