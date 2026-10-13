#!/bin/bash
#SBATCH --job-name=pi3_test
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --output=/home/isha.tariq/pi3/logs/test_pi3_%j.log
# Smoke test: does Pi3 / Pi3X produce point clouds end to end?
set -eo pipefail
ENV=/home/isha.tariq/.conda/envs/pi3
REPO=/home/isha.tariq/pi3/third_party/Pi3
OUT=/home/isha.tariq/pi3/outputs
export HF_HOME=/home/isha.tariq/.cache/huggingface   # XDG_CACHE_HOME may point at node-local /scratch
export PATH="$ENV/bin:$PATH"
echo "host: $(hostname)  job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

# gpu-05/gpu-51-style fault guard: allocate, don't just count devices.
python -c "import torch; torch.zeros(8, device='cuda'); print('cuda ok', torch.cuda.get_device_name())" \
    || { echo "CUDA allocation failed on $(hostname), requeueing"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }

mkdir -p "$OUT"
cd "$REPO"
# 1. Pi3 (original) on the bundled video, every 10th frame
timeout 20m python example.py    --data_path examples/skating.mp4 --save_path "$OUT/pi3_skating.ply"
# 2. Pi3X (recommended) on the same video
timeout 20m python example_mm.py --data_path examples/skating.mp4 --save_path "$OUT/pi3x_skating.ply"
# 3. Pi3X on an image folder (4 room frames), with and without the given conditions
timeout 20m python example_mm.py --data_path examples/room/rgb --save_path "$OUT/pi3x_room.ply"
timeout 20m python example_mm.py --data_path examples/room/rgb --conditions_path examples/room/condition.npz \
    --save_path "$OUT/pi3x_room_cond.ply"

echo "=== outputs ==="
ls -la "$OUT"
python /home/isha.tariq/pi3/scripts/check_ply.py "$OUT"/*.ply
