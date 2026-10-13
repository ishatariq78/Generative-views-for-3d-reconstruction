# Environments

The models need mutually incompatible stacks (PyTorch 1.13 to 2.11, CUDA 11.7 to 12.8), so each lives in its own conda environment. **Do not try to merge them.** Versions below were read from the installed environments on the cluster.

| env | Python | PyTorch / CUDA | used for |
|---|---|---|---|
| `viewcrafter` | 3.9.16 | 1.13.1 / cu117 | ViewCrafter generation ([A1](../experiments/A1_viewcrafter_orbits.md)) |
| `svcam` | 3.11.16 | 2.6.0 / cu124 | SEVA generation ([A2](../experiments/A2_seva_orbits.md), [B1](../experiments/B1_false_content.md), D) |
| `shaper` | 3.10.21 | 2.7.1 / cu128 | ShapeR inference and metrics; VGGT; scene building (pyrender/EGL); all scoring and figures |
| `kvtracker` | 3.11.16 | 2.11.0 / cu128 | KV-Tracker **and π³** (π³ runs from KV-Tracker's bundled Pi3 fork) |
| `pi3` | 3.11.16 | 2.5.1 / cu124 | standalone π³ smoke test only (`code/pi3/`); not used for reported numbers |

Hardware: A100-SXM4-40GB (sm_80), SLURM. The cluster has no CUDA toolkit newer than 12.1, so `kvtracker` and `shaper` install `cuda-toolkit` into the env and set `CUDA_HOME=$CONDA_PREFIX`. Setup scripts are in `code/seva_orbits/scripts/setup_env.sh`, `code/kv_tracker/scripts/setup_env.sh` and `code/pi3/scripts/setup_env.sh`. The `shaper` env follows ShapeR's own install with the same in-env toolkit. Its install script lives in a separate ShapeR workspace and is **not** included here.

## Upstream versions

| project | repository | commit used |
|---|---|---|
| ShapeR | github.com/facebookresearch/ShapeR | `d4402f5` |
| ViewCrafter | github.com/Drexubery/ViewCrafter | `97cdb86` |
| Stable Virtual Camera | github.com/Stability-AI/stable-virtual-camera | `fe19948` (model v1.1) |
| VGGT | github.com/facebookresearch/vggt | `a288dd0` (weights `facebook/VGGT-1B`) |
| π³ | github.com/yyfz/Pi3 | `9fa3ddb` (weights `yyfz233/Pi3`) |
| KV-Tracker | github.com/Marwan99/kv_tracker | `0437079` |

## Local patches to upstream code

Upstream code is not redistributed here. Each licence governs its own code: the KV-Tracker licence agreement and the Stability AI Non-Commercial License are both restrictive. The changes are described in prose; the patched files exist only on the cluster.

**KV-Tracker** (`third_party/kv_tracker/main.py`; original kept as `kv_tracker_testing/logs/main.py.orig`). Every change is marked `LOCAL PATCH`.

1. Re-enable `import open3d as o3d`. It is commented out upstream, yet `--export_pcd` uses it, and without it the run hangs instead of failing.
2. Hoist the `pixels_to_viz` assignment out of `if args.rerun:`, so `--export_pcd` works headless.
3. `KVT_MAX_KF` env var overrides the hard-coded 20-keyframe cap in camera-only mode. Needed for keyframe-every-frame on 80 frames.
4. `KVT_KEEP_LAST_PCD=1` keeps only the newest cumulative PLY, plus `pcd_counts.npy`, so the disk does not fill.

Changes 3 and 4 do nothing unless their env vars are set. The SSH submodule URLs in `.gitmodules` were rewritten to https.

**Stable Virtual Camera** (`seva/modules/autoencoder.py`). The VAE repo `stabilityai/stable-diffusion-2-1-base` that upstream loads has been removed from Hugging Face. A `SEVA_VAE_PATH` env var points at a local copy of the same VAE, `vae/diffusion_pytorch_model.safetensors`, sha256 `a1d993488569e928…`, byte-identical across six independent mirrors. When the variable is unset, the code falls back to the original repo id. `scipy` must also be added to the env, because upstream's `pyproject.toml` omits it.

No change was made to ShapeR, ViewCrafter, VGGT or π³ source.

## Weights and data

- **ShapeR checkpoints and the evaluation benchmark:** obtained through ShapeR's own download instructions. The benchmark ships 178 `.pkl` samples (~25 GB) with Aria RGB frames, SLAM points, cameras, captions and reference meshes.
- **SEVA:** gated Hugging Face weights (`stabilityai/stable-virtual-camera`), plus CLIP ViT-H and the VAE above.
- **ViewCrafter:** `model.ckpt`, and `model_sparse.ckpt` (`Drexubery/ViewCrafter_25_sparse`) for sparse mode.

None of these are in this repository.
