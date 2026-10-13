"""Pull every weight SEVA needs, on the login node, before any GPU job runs.

    /home/isha.tariq/.conda/envs/svcam/bin/python scripts/fetch_weights.py

THREE things are needed, not one:

  stabilityai/stable-virtual-camera       the SEVA checkpoint    GATED (auto)
  the SD-2.1-base VAE                     seva/modules/autoencoder.py    SEE BELOW
  laion/CLIP-ViT-H-14-laion2B-s32B-b79K   the conditioner, via open_clip
                                          "laion2b_s32b_b79k"    public

**The VAE cannot be fetched the way upstream expects.** `AutoEncoder` loads it
from "stabilityai/stable-diffusion-2-1-base", and Stability has removed that
repo: it 404s even for an authenticated user, as does plain
"stable-diffusion-2-1", while their other repos still resolve. So this script
takes it from a mirror instead and verifies the bytes.

That is safe to do because the weights are pinned by consensus: six independent
uploaders carry a byte-identical vae/diffusion_pytorch_model.safetensors
(sha256 a1d993488569e928..., 334,643,276 b), and the 768-v "stable-diffusion-2-1"
mirrors carry the same file -- the expected result, since 2.1 and 2.1-base share
one VAE. The copy lands in assets/sd21base/vae/ and `SEVA_VAE_PATH` points seva
at it (see the LOCAL PATCH note in seva/modules/autoencoder.py).

Run this on the login node. Otherwise a compute job discovers the problem about
90 s in, having burnt an allocation. Everything lands under /home -- the only
shared filesystem on this cluster.
"""

import hashlib
import os
import shutil
import sys

SEVA_REPO = "stabilityai/stable-virtual-camera"
# v1.1 fixes foreground objects detaching from the background in v1.0; v1.0 is
# fetched too so the two can be compared later.
SEVA_FILES = ["config.yaml", "modelv1.1.safetensors", "model.safetensors"]

VAE_MIRROR = "sd2-community/stable-diffusion-2-1-base"
VAE_DEST = "/home/isha.tariq/stable_virtual_camera/assets/sd21base/vae"
# None = not content-addressed upstream (tiny json), verified by loading instead.
VAE_FILES = {
    "config.json": None,
    "diffusion_pytorch_model.safetensors": (
        "a1d993488569e928"
    ),
}


def main(skip_seva_v1_0: bool = False) -> int:
    from huggingface_hub import hf_hub_download

    failed = []

    print("=== SEVA checkpoint ===")
    for name in SEVA_FILES:
        if skip_seva_v1_0 and name == "model.safetensors":
            continue
        try:
            print(f"  {name} -> {hf_hub_download(repo_id=SEVA_REPO, filename=name)}")
        except Exception as e:
            failed.append((f"{SEVA_REPO}/{name}", e))
            print(f"  {name} FAILED: {type(e).__name__}")

    print(f"=== SD-2.1-base VAE (from {VAE_MIRROR}) ===")
    os.makedirs(VAE_DEST, exist_ok=True)
    for name, want in VAE_FILES.items():
        dst = os.path.join(VAE_DEST, name)
        try:
            src = hf_hub_download(repo_id=VAE_MIRROR, filename=f"vae/{name}")
            shutil.copyfile(src, dst)
            got = hashlib.sha256(open(dst, "rb").read()).hexdigest()
            if want is not None and not got.startswith(want):
                # A mirror that does not match consensus is not the same VAE.
                # Refuse it rather than silently generating with wrong weights.
                os.remove(dst)
                raise ValueError(f"sha256 {got[:16]} != expected {want}")
            print(f"  {name} ok ({os.path.getsize(dst):,} b, sha256 {got[:16]})")
        except Exception as e:
            failed.append((f"{VAE_MIRROR}/vae/{name}", e))
            print(f"  {name} FAILED: {type(e).__name__}")

    print("=== CLIP ViT-H-14 (open_clip) ===")
    try:
        import open_clip

        open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
        print("  ok")
    except Exception as e:
        failed.append(("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", e))
        print(f"  FAILED: {type(e).__name__}")

    if failed:
        print("\n--- not downloaded ---")
        for what, e in failed:
            print(f"  {what}\n    {e}")
        print(
            "\nA 401 on the SEVA repo means you are not authenticated or have not\n"
            "accepted its terms:\n"
            "  ~/.conda/envs/svcam/bin/hf auth login\n"
            "  https://huggingface.co/stabilityai/stable-virtual-camera  (accept)\n"
            "A 404 on the VAE mirror means that mirror is gone too -- pick another\n"
            "from the list in the module docstring and check the sha256 matches."
        )
        return 1

    print(
        "\nall weights ready\n"
        f"  export SEVA_VAE_PATH={os.path.dirname(VAE_DEST)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(skip_seva_v1_0="--skip-v1.0" in sys.argv))
