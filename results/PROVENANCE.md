# Provenance

Every headline number, with the file it comes from. Paths are on the cluster (`~` = the original user's home). SLURM job IDs are given where the run log names them. "Rebuilt" means `scripts/build_results.py` recomputed the number from that file.

| finding | value | source file(s) | config / run | status |
|---|---|---|---|---|
| ViewCrafter 360°: frames > 1.5× subject area | 20–23 / 25 on 4 of 5 objects; 0 / 25 at ±45° | `~/viewcrafter_orbits/experiment2/outputs/subject_fraction.json` | `experiment2/slurm/run_orbits*.sbatch`, seed 123 | recomputed |
| ViewCrafter camera paths: worst-frame coverage | low 0.362 … dolly 0.226 (3 objects) | `experiment5/outputs/measurements.json` | `experiment5/trajs/*.txt` | recomputed |
| SEVA orbit IoU, share of ceiling | 0.751–0.851; 0.86–0.93 | `~/stable_virtual_camera/exp1/outputs/scores.json`, `input_arc_corrected.json` | `exp1/slurm/run_seva.sbatch` (cfg 3,2; interp-gt; 8 inputs) | rebuilt → `A_seva_orbit_fidelity.csv` |
| Surface ever photographed | 26–62% | `exp2/outputs/observability.json` | `exp2/scripts/observability.py` | rebuilt → `A_observed_vs_unseen.csv` |
| False content copied | one view 2 / 400; all views 374 / 400; invented 44 / 400 | `exp2/outputs/occlusion_scores.json` | `exp2/scripts/score_occlusion.py` (`RECON_T = 1.6`) | rebuilt → `B_false_content.csv` |
| Video → 3D, SLAM points reached | real 0.669 · VGGT-80 0.704 · π³-80 0.803 · π³-25 0.824 · KV kf1 0.831 | `experiment4/outputs/realigned_to_slam.json` (upright fit, job `exp9-score-248729`), `experiment10/outputs/realigned_to_slam.json`, `exp5/outputs/slam_scores.json` | `fit_upright` / `fit_any` | rebuilt → `C_video_to_3d_*.csv`; equals `experiment10/outputs/final_means_upright_with_table_fix.json` |
| SLAM-point accuracy and coverage of the reference | 0.35–0.73% of diag; within 2% of 17–68% of surface | `experiment4/outputs/baselines.json` | — | read directly |
| KV / π³ pose ATE | π³ 8.0 cm; KV kf1 10.3, kf5 10.8, kf10 11.7 cm | `exp5/outputs/poses.json` | evo Sim(3) vs requested cameras | recomputed |
| Generated points into ShapeR (6 objects) | π³ ΔCD +1.80, KV +2.71, gap1 +0.59 | `~/seva_to_shaper/outputs/seeded/s{0,1,2}/per_object/*.json` | `slurm/07_seeded.sbatch`, `09_combined.sbatch`, `balance` (16 views) | rebuilt → `C_points_into_shaper_6obj.csv` |
| KV points into ShapeR (9 objects) | ΔCD +1.63, 2 / 9 better | `outputs_sub/kvseeded/s{0,1,2}/per_object/*.json`, `outputs_sub/kvtest_placement.json` | `slurm/30_kv_subset`, `31_kv_score` (logs `s2s-kvsub-*`, `s2s-kvscore-*`), 24 views | rebuilt → `C_kv_points_into_shaper_subset.csv` |
| Views into ShapeR, 16 views (6 objects) | `views` ΔCD +0.11; `vonly` +1.02 | `outputs/seeded/s{0,1,2}/per_object/*.json` | `slurm/08_views.sbatch` | rebuilt → `D_views_into_shaper_6obj.csv` |
| View budget | `views` 8: +1.00 … 24: −0.19 … 32: +0.06 | `outputs/nviews{8,12,24,32}/s{0,1,2}/per_object/*.json` | `slurm/13_nviews.sbatch` (logs `logs/s2s-nv-*.out`) | rebuilt → `D_view_budget_6obj.csv` |
| Views at 24, 155 objects | ΔCD −0.106 (SE 0.046), 85 / 155; ΔF1 +0.011 | `outputs_sub/seeded/s{0,1,2}/per_object/*.json` (930 files) | `slurm/23_subset_shaper.sbatch`, `--num_images 24` | rebuilt → `D_views_into_shaper_155obj.csv` |
| Views at 24, 40 one-sided objects | ΔCD −0.207 (SE 0.094), 26 / 40 | same, filtered by `outputs_sub/traits.json` | `scripts/pick_subset.py --n 40` | rebuilt (`summary.json`) |
| 23 excluded objects and reasons | 17 / 5 / 1 | `logs/s2s-bfull-284635.out`, `s2s-bfull-284636.out` | `slurm/20_build_full.sbatch` | read directly |
| Frame-selection pilot (16 objects) | D vs B −0.077; D vs C −0.007 | `sel/pilot/s{0,1,2}/per_object/*.json` | `slurm/41_sel_pilot.sbatch` (logs `s2s-pilot-296618/296619.out`) | rebuilt → `D_frame_selection_pilot.csv` |

## Figures

| figure | built from | metric annotations |
|---|---|---|
| <a id="teaser"></a>`assets/teaser.jpg` | `seva_to_shaper/inputs/BNB1045__barstool/images`, `outputs/demo/img2trajvid/BNB1045__barstool`, meshes and JSONs in `outputs/seeded/s0/per_object/` | seed 0, 16 views, one run per mesh: `slam` 1.80 / 0.779, `pi3` 6.60 / 0.271, `views` 1.36 / 0.851. 3-seed means are in `C_points_into_shaper_6obj.csv` / `D_views_into_shaper_6obj.csv` (2.36 / 7.43 / 1.65 CD) |
| `assets/gifs/seva_orbits.gif` | `stable_virtual_camera/exp1/outputs/demo/img2trajvid/*/samples-rgb` (every 4th frame) | none |
| `assets/full_orbit_vs_sweep.jpg` | `viewcrafter_orbits/experiment2/outputs/BNB1045__sofa__cutout__orbit{360,45}/diffusion0.mp4`, `exp1/outputs/demo/img2trajvid/BNB1045__sofa` | none |
| `assets/observed_vs_unseen.jpg` | crop of `stable_virtual_camera/exp2/figs/observability_vs_output.png` (sofa, bed rows) | per-frame never-seen % from that figure |
| `assets/false_content.jpg` | `exp2/inputs/*__occ{1,all}/images`, `exp2/outputs/demo/img2trajvid/*/samples-rgb`, clean run from `exp1` | per-object counts from `occlusion_scores.json` |
| `assets/clouds_vggt_pi3_kv.png` | `exp5/outputs/<obj>__{seva360f,seva360fp,seva360fk1}__aligned_to_slam.npz`; SLAM points from the benchmark pkl | per object × reconstructor from `C_video_to_3d_per_object.csv` |
| `assets/chart_video_to_3d.png` | `results/C_video_to_3d_*.csv` | condition means, per-object dots |
| `assets/chart_view_budget.png`, `chart_155_paired.png` | `results/D_view_budget_6obj.csv`, `D_views_into_shaper_155obj.csv`, `summary.json` | aggregates labelled as such |
| `assets/gifs/viewcrafter_vs_seva.gif` | same sources as `full_orbit_vs_sweep.jpg`, all 25 ViewCrafter frames and 25 evenly spaced of SEVA's 80 | none |
| `assets/chart_conditions_6obj.png` | `results/C_points_into_shaper_6obj.csv`, `D_views_into_shaper_6obj.csv` | per-object 3-seed means |
| `assets/mesh_conditions.jpg` | `seva_to_shaper/outputs/seeded/s0/per_object/{GT__,}<obj>__<cond>.glb` + `.json` | each mesh's own seed-0 JSON |
| `assets/orbits_155_sample.jpg` | `seva_to_shaper/inputs_full/<obj>/images/000.png`, `outputs_full/demo/img2trajvid/<obj>/samples-rgb` | 3-seed mean from `D_views_into_shaper_155obj.csv` |
| `assets/chart_selection_pilot.png` | `results/D_frame_selection_pilot.csv` | 3-seed means |
| `assets/gallery_shaper_views.jpg` | `seva_to_shaper/outputs_sub/meshes24/per_object/` (seed-0 re-run with meshes) + `outputs_full/demo/img2trajvid` | each panel: that mesh's own JSON; row subtitle adds the 3-seed mean from `D_views_into_shaper_155obj.csv` |
