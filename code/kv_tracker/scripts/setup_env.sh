#!/bin/bash
# Build the `kvtracker` conda env for KV-Tracker (CVPR 2026) and its two submodules.
#
#   bash scripts/setup_env.sh
#
# Login node is fine. ~20 min: the CUDA toolkit is ~3 GB and both submodules
# compile a CUDA extension.
set -eo pipefail

MAMBA=/home/isha.tariq/shapeR_testing/tools/miniforge3/bin/mamba
ENV=/home/isha.tariq/.conda/envs/kvtracker
REPO=/home/isha.tariq/kv_tracker_testing/third_party/kv_tracker

# Two CUDA extensions have to be compiled: Pi3's `curope` (2D RoPE -- there is a
# pure-torch fallback, but a real-time tracker should not run it on the slow
# path) and SAM2's `_C`. Compiling needs nvcc matching the torch wheel, and this
# cluster ships nothing newer than 12.1. Same answer as for ShapeR: put the
# toolkit *in the env* from conda-forge and point CUDA_HOME at it. Upstream's own
# recommendation is torch cu128, so 12.8 makes the wheel and the toolkit agree.
# gcc 13 in the env too: the nodes' system g++ is 14.4, and nvcc 12.8 refuses
# anything >= 14 ("greater than the maximum required version by CUDA 12.8").
# CC/CXX are set explicitly rather than via `conda activate`, which dies under
# set -u on conda-forge's binutils.
if [ ! -x "$ENV/bin/nvcc" ]; then
    "$MAMBA" create -y -p "$ENV" -c conda-forge python=3.11 pip cuda-toolkit=12.8 \
        "gxx_linux-64=13" "gcc_linux-64=13"
fi
PY="$ENV/bin/python"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export CC="$ENV/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ENV/bin/x86_64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST="8.0"        # A100 only; SAM2 honours this, Pi3 hardcodes its list

"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Both pyprojects declare build-system.requires = ["setuptools==70.0.0", "torch"]
# and upstream installs with --no-build-isolation, so both must already be there.
"$PY" -m pip install "setuptools==70.0.0" wheel ninja plyfile   # plyfile: pi3.utils.basic needs it, Pi3 lists it only in requirements.txt

# Pi3 -- the geometry backbone, from the authors' fork with KV caching.
"$PY" -m pip install --no-build-isolation -e "$REPO/thirdparty/Pi3"
# The editable finder that install produces maps only the compiled
# `curope_backend`, not the `pi3` package -- setup.py's packages=["pi3"] and the
# pyproject's packages.find disagree and setuptools 70 sides with neither. A
# plain .pth entry makes `import pi3` resolve, which is all the editable was for.
echo "$REPO/thirdparty/Pi3" > "$ENV/lib/python3.11/site-packages/pi3_path.pth"

# SAM2 real-time fork -- needed even for --cam_only, because every loader calls
# init_models() unconditionally.
"$PY" -m pip install --no-build-isolation -e "$REPO/thirdparty/segment-anything-2-real-time"

# KV-Tracker. Its pyproject names Pi3 and sam-2 as deps resolved through
# uv-only [tool.uv.sources], which pip cannot read -- hence the two editable
# installs above come first and pip finds them satisfied.
"$PY" -m pip install --no-build-isolation -e "$REPO"

echo
echo "=== versions ==="
"$PY" - <<'PYEOF'
import importlib.metadata as md, subprocess, sys, torch
print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__, "cuda", torch.version.cuda)
print("nvcc    ", subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1])
for p in ("kv-tracker", "pi3", "SAM-2", "rerun-sdk", "open3d", "evo", "pyrealsense2", "numpy"):
    try: print(f"{p:12s}", md.version(p))
    except Exception as e: print(f"{p:12s} MISSING ({e})")
import curope_backend; print("curope   compiled")
import sam2._C; print("sam2._C  compiled")
PYEOF
echo "env ready: $ENV"
