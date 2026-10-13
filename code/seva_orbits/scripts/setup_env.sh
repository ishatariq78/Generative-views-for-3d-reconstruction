#!/bin/bash
# Build the `svcam` conda env for Stable Virtual Camera (SEVA).
#
#   bash scripts/setup_env.sh
#
# Runs on the login node — no GPU needed, nothing is compiled against CUDA.
# Takes ~10 min, almost all of it downloading the torch wheels.
set -eo pipefail

CONDA=/home/isha.tariq/shapeR_testing/tools/miniforge3/bin/conda
ENV=/home/isha.tariq/.conda/envs/svcam
REPO=/home/isha.tariq/stable_virtual_camera/third_party/stable-virtual-camera

# Python 3.11: numpy is pinned to 1.24.4 by the upstream pyproject and that
# release has no cp312 wheel, so 3.12 would try to build numpy from source.
if [ ! -x "$ENV/bin/python" ]; then
    "$CONDA" create -y -p "$ENV" python=3.11 pip
fi
PY="$ENV/bin/python"

# torch>=2.6.0 is a hard requirement (README). cu124 wheels ship their own CUDA
# runtime, so the cluster's 12.1-capped toolkit does not matter — nothing here
# compiles a CUDA extension. Driver 570.195 is 12.8-capable, so cu124 is safe.
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# The model itself.
"$PY" -m pip install -e "$REPO"

# scipy is missing from upstream's pyproject but seva/geometry.py imports it
# at module level, so every `import seva.sampling` fails without it.
"$PY" -m pip install "scipy<1.14"

# Only needed if a scene is given in COLMAP format; it is the pure-python
# SceneManager fork, not the C++ bindings, so it costs nothing to have.
"$PY" -m pip install \
    "git+https://github.com/jensenz-sai/pycolmap@543266bc316df2fe407b3a33d454b310b1641042"

echo
echo "=== versions ==="
"$PY" - <<'PYEOF'
import torch, numpy, importlib.metadata as md
print("python  ", __import__("sys").version.split()[0])
print("torch   ", torch.__version__, "cuda", torch.version.cuda)
print("numpy   ", numpy.__version__)
for p in ("seva", "diffusers", "open-clip-torch", "imageio"):
    try:
        print(f"{p:8s}", md.version(p))
    except Exception as e:
        print(f"{p:8s} MISSING ({e})")
PYEOF
echo
echo "env ready: $ENV"
