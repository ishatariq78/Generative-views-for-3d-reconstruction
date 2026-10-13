#!/bin/bash
#SBATCH --job-name=exp2-orbit
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
# 4 CPUs / 48 GB is enough -- the model lives on the GPU and host RAM peaks
# only while the 12 GB of checkpoints load. Asking for 8/96G left the job
# sitting in (Resources) on a busy partition for no benefit.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment2/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment2/logs/%x-%j.out

# =============================================================================
# [STEP 2] Orbit the isolated objects.
#   sbatch slurm/run_orbits.sbatch
# 12 runs (3 objects x {cutout, context} x {orbit360, orbit45}) sharing one
# loaded model, ~30 min. Needs inputs/ from slurm/build_views.sbatch.
# =============================================================================
# cscc-gpu-p is the only partition this account can actually submit to: `long`
# advertises AllowAccounts=ALL but runs on QOS gpu-12, which the association does
# not carry, so `sbatch --partition=long` is rejected as an invalid QOS. When
# cscc-gpu-p sits in (Resources), the job simply waits.
set -euo pipefail
PROJ=/home/isha.tariq/viewcrafter_orbits/experiment2
VC=/home/isha.tariq/view_crafter/ViewCrafter

export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u)
export TMPDIR=/tmp
mkdir -p "$XDG_RUNTIME_DIR"

n=$(ls "$PROJ"/inputs/*__cutout.png 2>/dev/null | wc -l)
[ "$n" -ge 1 ] || { echo "ERROR: no object-only views in $PROJ/inputs"; \
                    echo "       run: sbatch slurm/build_views.sbatch"; exit 1; }

set +u   # conda's activate hooks are not -u clean
source /apps/local/conda_init.sh
conda activate viewcrafter
set -u

cd "$VC"

# Node health, checked the only way that is conclusive: allocate on the device.
# Two nodes so far have handed out a GPU index that CUDA cannot open while
# nvidia-smi looked healthy, and torch.cuda.device_count() reported 1 on both --
# so counting devices is not enough, and neither is nvidia-smi. The hostname is
# always logged, because the failure is per-node and that is what identifies it.
echo "=== host $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi -L || true
if ! python -c "
import sys, torch
try:
    torch.zeros(8, device='cuda')
    print('GPU OK:', torch.cuda.get_device_name(0), '| torch', torch.__version__)
except Exception as e:
    print('GPU UNUSABLE:', type(e).__name__, e); sys.exit(1)
"; then
    echo "       requeueing job ${SLURM_JOB_ID:-?} so it lands on another node."
    scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || \
        echo "       requeue refused -- resubmit by hand, adding this host to --exclude."
    exit 75
fi

time python "$PROJ/scripts/run_object_only_orbits.py" --seed 123

echo
echo "=== videos ==="
find "$PROJ/outputs" -name 'diffusion0.mp4' -printf '%p\n' | sort
