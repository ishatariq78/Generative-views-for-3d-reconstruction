# Experiments

Each page states its question, setup, exact objects and counts, commands, metrics, results with their source files, interpretation and limitations, and the experiment it led to.

| | page | pathway | objects | seeds |
|---|---|---|---:|---:|
| A1 | [ViewCrafter orbits: one, three and eight views; full orbit vs ±45°; six camera paths](A1_viewcrafter_orbits.md) | generation | 3–5 | 1 |
| A2 | [SEVA orbits, and observed vs never-photographed surface](A2_seva_orbits.md) | generation | 5 | 1 |
| B1 | [False content in one view vs every view](B1_false_content.md) | generation | 5 | 1 |
| C1 | [Generated video → 3D: VGGT, π³, KV-Tracker; frames, keyframes, alignment](C1_video_to_3d.md) | **1** | 5 | 1 |
| C2 | [Bridge: reconstructed orbit points as ShapeR's point input](C2_points_into_shaper.md) | 1 → ShapeR | 6 + 9 | 3 |
| D1 | [Real + generated images into ShapeR: camera handling, conditions, view budget](D1_views_into_shaper.md) | **2** | 6 | 3 |
| D2 | [Real + generated views at scale](D2_views_at_scale.md) | **2** | 40 ⊂ 155 | 3 |
| D3 | [Which generated frames? Selection pilot](D3_frame_selection.md) | **2** | 16 | 3 |

Object naming follows the benchmark: `<capture>__<object>`, e.g. `BNB1045__sofa`.

The chain of reasoning: A1 shows single-view generation cannot make a full turn → A2 gets one from a multi-view generator, but most of it is never-observed → B1 shows it trusts consistency, not truth → C1 shows the orbit adds coverage but not precision in 3D → C2 shows those points do not replace real SLAM points → D1/D2 show generated *images* give a small, uneven gain whose size depends on the view budget → D3 finds no frame-selection rule that matters.
