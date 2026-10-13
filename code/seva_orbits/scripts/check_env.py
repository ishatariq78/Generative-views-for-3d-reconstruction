"""Prove the env can actually run SEVA on this node before spending a job on it.

Run it under srun/sbatch, never on the login node -- the point is the GPU check.
gpu-05 and gpu-51 hand out a GPU index CUDA cannot open while nvidia-smi looks
healthy, so this *allocates* rather than just counting devices.
"""

import os
import socket

import torch

print("host        ", socket.gethostname())
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("visible     ", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count", torch.cuda.device_count())

# Counting devices is not enough on the faulty nodes -- force an allocation.
x = torch.zeros(8, device="cuda")
print("alloc ok    ", x.sum().item() == 0.0)
print("gpu         ", torch.cuda.get_device_name(0))

# seva runs every attention under the flash backend explicitly, so a node where
# that backend is unavailable fails inside sampling rather than at import.
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

q = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
print("flash sdpa  ", tuple(out.shape))

import seva  # noqa: E402
from seva.model import Seva, SevaParams  # noqa: E402

print("seva        ", os.path.dirname(seva.__file__))
with torch.device("meta"):
    Seva(SevaParams())
print("model graph  builds")
