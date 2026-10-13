#!/bin/bash
# Build the `pi3` conda env for upstream Pi3 / Pi3X (yyfz/Pi3).
#
#   bash scripts/setup_env.sh
#
# Login node is fine, ~5 min. Unlike the KV-Tracker fork, upstream Pi3 has no
# CUDA extension (RoPE2D falls back to pure torch), so no in-env toolkit or
# compiler is needed -- just torch. requirements.txt pins torch 2.5.1, whose
# PyPI wheel bundles CUDA 12.4 and runs fine on the nodes' 570 driver.
set -eo pipefail

MAMBA=/home/isha.tariq/shapeR_testing/tools/miniforge3/bin/mamba
ENV=/home/isha.tariq/.conda/envs/pi3
REPO=/home/isha.tariq/pi3/third_party/Pi3

if [ ! -x "$ENV/bin/python" ]; then
    "$MAMBA" create -y -p "$ENV" -c conda-forge python=3.11 pip
fi
PY="$ENV/bin/python"

"$PY" -m pip install --upgrade pip
"$PY" -m pip install --no-cache-dir -r "$REPO/requirements.txt"
# pyproject packages=["pi3"] -> a plain editable install works here (no
# custom setup.py like the KV fork), so `import pi3` resolves from anywhere.
"$PY" -m pip install -e "$REPO"

echo
echo "=== versions ==="
"$PY" - <<'PYEOF'
import importlib.metadata as md, sys, torch
print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__, "cuda", torch.version.cuda)
for p in ("pi3", "numpy", "opencv-python", "plyfile", "huggingface_hub", "safetensors"):
    try: print(f"{p:16s}", md.version(p))
    except Exception as e: print(f"{p:16s} MISSING ({e})")
from pi3.models.pi3 import Pi3
from pi3.models.pi3x import Pi3X
print("pi3 imports ok")
PYEOF
echo "env ready: $ENV"
