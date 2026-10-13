#!/bin/bash
#SBATCH --job-name=exp6-rescore
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment6/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment6/logs/%x-%j.out
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp; mkdir -p "$XDG_RUNTIME_DIR"
/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/viewcrafter_orbits/experiment6/scripts/rescore_tight_scale.py
