"""
Seeded ShapeR inference + benchmark metrics, with the view budget exposed.

A copy of ~/shapeR_testing/exp_object_aware/scripts/run_shaper_seeded.py with one addition:
--num_images overrides the preset's view count. The point is that ShapeR's `balance` preset
consumes 16 views, so generated orbit views *displace* real ones instead of adding to them.
Raising the budget lets the generated views be extra information rather than a substitution;
the baseline is run at the same budget so the comparison stays fair.

Same model, same `balance` preset and the same metric code as the benchmark run
(imported from ShapeR/evaluation/eval.py, which is left untouched), with one
difference: the RNG is reset to --seed right before every object, so

  * the rectified-flow noise x_0 for object X is identical in every condition,
  * it does not depend on which other objects are in the run or their order,
  * the 100k evaluation samples are drawn identically too.

A difference between two conditions is then caused by the inputs alone.

  python scripts/run_shaper_seeded.py --input_dir outputs/pkls/cluttered --out_dir outputs/runs/cluttered_s0 --seed 0
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import trimesh

sys.path.insert(0, "/home/isha.tariq/shapeR_testing/exp_object_aware/scripts")
from common import REPO, filter_sample_points, load_pkl, obj_name, save_pkl

os.chdir(REPO)                       # checkpoints/ and torch.hub paths are relative in upstream
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evaluation"))
from evaluation.eval import _eval_single_mesh, load_model  # noqa: E402
from dataset.shaper_dataset import InferenceDataset  # noqa: E402


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_dir")
    g.add_argument("--input_pkls", nargs="+")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels_dir", default=None,
                    help="object_aware.py diag dir: apply <name>.npz labels (1 = distractor) before inference")
    ap.add_argument("--config", default="balance")
    ap.add_argument("--num_images", type=int, default=None, help="override the preset view count")
    ap.add_argument("--n_points", type=int, default=100000)
    ap.add_argument("--save_meshes", action="store_true")
    ap.add_argument("--save_visualization", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    args = ap.parse_args()

    paths = args.input_pkls or sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    paths = [p for i, p in enumerate(paths) if i % args.n_shards == args.shard]
    per_obj = os.path.join(args.out_dir, "per_object")
    os.makedirs(per_obj, exist_ok=True)
    if args.skip_done:
        paths = [p for p in paths if not os.path.exists(os.path.join(per_obj, obj_name(p) + ".json"))]
    print(f"{len(paths)} objects, seed {args.seed}, config {args.config}, shard {args.shard}/{args.n_shards}")
    if not paths:
        return

    device = torch.device("cuda")
    (model, vae, text_feature_extractor, config, _, num_images, num_steps,
     token_count, embed_dim, use_shifted_sampling) = load_model(args.config, device=device)
    if args.num_images:
        num_images = args.num_images          # the view budget is the thing under test

    for p in paths:
        name = obj_name(p)
        t0 = time.time()
        seed_all(args.seed)                      # view selection (random.choices fallback) + noise
        run_path = p
        if args.labels_dir:
            lab_path = os.path.join(args.labels_dir, name + ".npz")
            if os.path.exists(lab_path):
                labels = np.load(lab_path)["labels"]
                run_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"oa_{os.getpid()}_{name}.pkl")
                save_pkl(filter_sample_points(load_pkl(p), labels != 1), run_path)
            else:
                print(f"  [WARN] no labels for {name} in {args.labels_dir}; running the input unchanged")
        ds = InferenceDataset(config, paths=[run_path], override_num_views=num_images)
        batch = ds.custom_collate([ds[0]])
        gt_verts = np.asarray(batch["vertices"][0], dtype=np.float32)
        gt_faces = np.asarray(batch["faces"][0])
        bounds = np.asarray(batch["bounds"][0], dtype=np.float32)
        batch = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)
        with torch.no_grad():
            seed_all(args.seed)                  # x_0 ~ N(0, I) drawn right after this
            latents = model.infer_latents(
                batch, token_shape=(1, token_count, embed_dim),
                text_feature_extractor=text_feature_extractor,
                num_steps=num_steps, use_shifted_sampling=use_shifted_sampling,
            )
            pred_mesh = vae.infer_mesh_from_latents(latents)[0]
        seed_all(args.seed)                      # surface sampling for the metrics
        metrics = _eval_single_mesh(
            np.asarray(pred_mesh.vertices, dtype=np.float32), np.asarray(pred_mesh.faces),
            gt_verts, gt_faces, args.n_points, bounds,
        )
        n_pts = int(batch["semi_dense_points_orig"][0].shape[0])
        rec = dict(name=name, seed=args.seed, num_images=int(num_images), metrics=metrics, n_points_used=n_pts,
                   n_faces=int(len(pred_mesh.faces)), seconds=float(time.time() - t0))
        json.dump(rec, open(os.path.join(per_obj, name + ".json"), "w"), indent=1)
        print(f"  {name:32s} CD(x100)={metrics['chamfer']*100:7.3f}  NC={metrics['normals']:.4f}  "
              f"F1={metrics['f-score']:.4f}  pts={n_pts:5d}  {rec['seconds']:.0f}s", flush=True)

        if args.save_meshes:
            m = pred_mesh.copy()
            m = ds.rescale_back(0, m, do_transform_to_world=False)
            m.export(os.path.join(per_obj, name + ".glb"), include_normals=True)
            gt = trimesh.Trimesh(vertices=gt_verts, faces=gt_faces)
            gt = ds.rescale_back(0, gt, do_transform_to_world=False)
            gt.export(os.path.join(per_obj, f"GT__{name}.glb"), include_normals=True)
            pair = trimesh.Scene()
            pv, gv = m.copy(), gt.copy()
            pv.visual.face_colors = [255, 192, 0, 180]
            gv.visual.face_colors = [34, 139, 34, 180]
            pair.add_geometry(pv, geom_name="pred")
            pair.add_geometry(gv, geom_name="gt")
            pair.export(os.path.join(per_obj, f"PAIR__{name}.glb"))
        if args.save_visualization:
            from postprocessing.helper import visualize_prediction_and_groundtruth

            gt = trimesh.Trimesh(vertices=gt_verts, faces=gt_faces)
            vis_images = batch["images"][0].float().cpu().numpy()
            vis_masks = batch["images"][0].float().cpu().clone().numpy()
            vis_masks[:, 1, :, :] = batch["masks_ingest"][0].float().cpu().numpy()
            visualize_prediction_and_groundtruth(
                pred_mesh.copy(), gt, batch["semi_dense_points_orig"][0], vis_images, vis_masks,
                batch["caption"][0], sample_name=name,
                save_path=os.path.join(per_obj, f"VIS__{name}.jpg"),
            )
        if run_path != p:
            os.remove(run_path)              # rescale_back re-reads the pkl, so only now


if __name__ == "__main__":
    main()
