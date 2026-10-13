# Methodology

## Data

All objects come from the **ShapeR evaluation benchmark**. Each of its 178 samples holds Project Aria RGB frames (Fisheye624 cameras), the per-frame projections of the object's points, semi-dense SLAM points in the object's model frame, a caption, the object box, and a reference mesh. Aria Gen1 RGB frames are stored a quarter turn off: the image roll is applied to K and to the pose together.

## Building object-centric generator inputs

Code: `code/seva_orbits/exp1/scripts/build_scenes.py`, copied into `code/generated_to_shaper/scripts/build_scenes.py` for the larger runs.

1. **Choose real views** spread around the object from the capture, up to 8 for SEVA (fewer when a capture is one-sided).
2. **Re-aim, don't crop.** A virtual camera rotated to face the object is an exact homography `K_new · R · K_old⁻¹` about the same optical centre. The principal point stays centred, which SEVA requires, and no parallax is introduced.
3. **Cut out** the object with the reference-mesh silhouette, composited onto a neutral backdrop with a contact shadow.
4. **Verify** by re-rasterising the mesh through the final K and pose. A view is refused below IoU 0.80.
5. **Occlusion cleaning** (`scripts/occlusion.py`). Every silhouette pixel is a known 3D point, so it is projected into the other views. Pixels that disagree with the median are repaired, and a view is dropped when its texture correlates with nothing **and** its outline lands on no real edge. Refill and repeat to convergence.
6. **Target cameras:** 80 on a circle in the model frame, at the inputs' elevation (SEVA). For ViewCrafter, trajectory files with azimuth / elevation / radius ([A1](../experiments/A1_viewcrafter_orbits.md)).

### Reference-assisted steps

Steps 3–6 use the **reference mesh**: the silhouette, the IoU check, the 3D points behind occlusion cleaning, and the frame that defines the orbit circle. The observability analysis ([A2](../experiments/A2_seva_orbits.md)) does too. A deployed system has no reference mesh, so every downstream result is conditional on this oracle-quality preprocessing. A reference-free version (e.g. masks from the capture's own points, or a segmenter) is **planned, not implemented**.

## Pathway 1: generated video → 3D

- VGGT (`facebook/VGGT-1B`): depth branch + `unproject_depth_map_to_point_map`.
- π³: batch over all frames.
- KV-Tracker: frames streamed as a TUM-format sequence, `--cam_only --export_pcd --resize_dim 518 --kf_auto k`.
- Shared subject mask (backdrop colour rule) and a 0.35 confidence-quantile cut.
- **Alignment to SLAM points:** upright, camera-anchored Sim(3) (`experiment3/scripts/align_and_score.py::fit_upright` / `realign_to_slam.py::fit_any`). The camera ring normal is taken as up and rotated to +z, the search runs over azimuth × scale, ICP follows, and tilts > 60° are rejected. Earlier fitters left clouds upside down, so their numbers are superseded ([discrepancies](discrepancies.md)).

## Bridge: reconstructed points into ShapeR

Placement in the model frame: Umeyama from estimated to requested camera centres, then a guarded Sim(3) ICP of the SLAM points onto the cloud ([C2](../experiments/C2_points_into_shaper.md)). The cloud replaces or augments `points_model`. Everything else in the sample is the benchmark's bytes.

## Pathway 2: generated images into ShapeR

Each generated frame becomes a ShapeR view with an **exact** camera ([D1](../experiments/D1_views_into_shaper.md#input-preparation-and-camera-handling)):

- OpenGL → OpenCV pose
- greyscale
- pinhole → equidistant remap, declared as zero-distortion Fisheye624
- per-view projections of the SLAM points (frustum test only)

Frames are appended to or replace the real views. `run_shaper_nviews.py` (a copy of a seeded ShapeR runner, plus `--num_images`) controls the view budget. `sel_build.py` freezes the exact view set for the selection study.

## What was not changed

ShapeR's weights, architecture, `balance` preset (except the view count where stated) and metric code. Generators and reconstructors are used as released, apart from the local fixes in [environments](environments.md#local-patches-to-upstream-code).
