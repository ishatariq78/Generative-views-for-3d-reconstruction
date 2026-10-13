#!/bin/bash
#SBATCH --job-name=exp4-refit
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp4/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp4/logs/%x-%j.out
# Re-fit all seven deck conditions with the full-SO(3) search (exp5's fit_so3).
set -eo pipefail
export TMPDIR=/tmp; export OMP_NUM_THREADS=8
cd /home/isha.tariq/stable_virtual_camera/exp4
/home/isha.tariq/.conda/envs/shaper/bin/python scripts/refit_so3.py
