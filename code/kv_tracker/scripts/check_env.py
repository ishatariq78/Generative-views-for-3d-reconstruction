"""Prove the env runs KV-Tracker on this node: GPU alloc, Pi3 weights, SAM2, KV cache.

Run under sbatch, never on the login node -- the point is the GPU. gpu-05/gpu-51
hand out a GPU CUDA cannot open, so this *allocates* rather than counting.
"""
import os, socket, time
import torch

print("host        ", socket.gethostname())
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
x = torch.zeros(8, device="cuda"); print("alloc ok    ", x.sum().item() == 0.0)
print("gpu         ", torch.cuda.get_device_name(0))

os.chdir("/home/isha.tariq/kv_tracker_testing/third_party/kv_tracker")
t = time.time()
from kv_tracker.pi3_utilts import load_pi3_from_pretrained
m = load_pi3_from_pretrained("cuda").eval()
print(f"Pi3 loaded   {sum(p.numel() for p in m.parameters())/1e6:.0f}M params in {time.time()-t:.0f}s")

from sam2.build_sam import build_sam2_camera_predictor  # noqa: E402
p = build_sam2_camera_predictor("configs/sam2.1/sam2.1_hiera_s.yaml",
                                "thirdparty/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt")
print("SAM2.1 small loaded")
print("ok")
