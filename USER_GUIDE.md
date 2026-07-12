# Qwen + Grounded-SAM-2 + SpatialTrackerV2 3D Trace Pipeline — User Guide

Goes from a raw video (+ optional narration) to per-entity (hand/object) 3D
trajectories and a combined visualization video. Built to prototype
Ego4D-style trace-data extraction for VLA training.

## Architecture

Three stages, two conda envs:

| Stage | Env | Script | What it does |
|---|---|---|---|
| 1 | `qwen_sam2` | `~/trace/pipeline/stage1_detect.py` | Qwen3-VL identifies task+objects from narration → periodic Grounded-SAM-2 detection+SAM2 tracking per label (hand + each named object) → farthest-point-sampled query points per instance → `bundle.json` |
| 2 | `trace_stv2` | `~/trace/SpaTrackerV2/stage2_lift3d.py` | SpatialTrackerV2 lifts every instance's query points to 3D (grid call for camera-pose stability + entity call for the actual points, points unprojected through dense depth + the grid call's bundle-adjusted camera poses) → per-instance `.npz` + `*_trace_summary.json` |
| 3 | `trace_stv2` | `~/trace/SpaTrackerV2/stage3_render.py` | Renders original + novel chase-cam angle side by side, all instances overlaid with trails |

Orchestrated by `~/trace/pipeline/run_pipeline.sh`, which spans both conda
envs via `conda run`.

## Quick start

```bash
~/trace/pipeline/run_pipeline.sh \
  --video /path/to/clip.mp4 \
  --out-dir ~/trace/pipeline_out/my_run \
  --narration "a person washing lettuce in the sink" \
  --box-threshold 0.2 --text-threshold 0.15 \
  --point-margin 20
```

Outputs land in `--out-dir`:
- `bundle.json` — stage 1's detections + query points (inspect this first if something looks wrong)
- `<clip>_trace_summary.json` — stage 2's per-instance metadata + which instances got skipped and why
- `<clip>_trace_<instance_id>.npz` — per-instance `points3d` (T,N,3), `points2d` (T,N,2), `visibility` (T,N bool), `frame_indices`, `seed_frame`
- `<clip>_scene.npz` — shared `video`, `depths`, `intrinsics`, `extrinsics` for the whole window
- `<clip>_trace_render.mp4` — the visualization

This exact command (on the `~/trace/SpaTrackerV2/assets/lettuce/lettuce.mp4`
clip) is the current best-known-good configuration — see "Recommended
defaults" below for why each flag is set this way, not left at its default.

## Full flag reference

| Flag | Default | Meaning |
|---|---|---|
| `--video` | required | input clip |
| `--out-dir` | required | output directory |
| `--narration` | "a person washing lettuce in the sink" | per-clip narration string fed to Qwen3-VL for task/object ID. **Always override this for a new clip** — the default is a leftover from the dev clip and will bias object identification toward lettuce/sink vocabulary. |
| `--point-budget` | 10 | query points sampled per instance |
| `--point-margin` | 3 | erode each instance's seed mask inward by this many px before sampling query points (see "point-margin" below) |
| `--max-frames` | 35 | **hard ceiling** on how many frames of the clip stage 2 ever looks at (see "max-frames" below) |
| `--check-interval` | 15 | how often (in frames) stage 1 re-runs GroundingDINO to catch entities entering/exiting frame |
| `--box-threshold` | 0.3 | GroundingDINO box confidence threshold |
| `--text-threshold` | 0.25 | GroundingDINO text-match threshold |
| `--grid-size` | 20 | background/VO query grid density for camera-pose stability (query count = grid_size²) |
| `--vo-points` | 2000 | bundle-adjustment point budget for the grid call |

## Recommended defaults, and why (read before changing hyperparameters)

### `--box-threshold` / `--text-threshold`: use 0.2 / 0.15, not the 0.3 / 0.25 defaults

The 0.3/0.25 defaults are too strict for two-simultaneous-similar-objects
scenes (e.g. two hands): GroundingDINO's *best* second-hand candidate is
often below 0.3 confidence, so raising these thresholds silently loses real
instances rather than producing a bad detection you could catch. Lowering to
0.2/0.15 surfaces more (sometimes duplicate/loose) candidates — this is safe
*only* because `entity_tracker.py`'s `filter_detections()` now does
containment-aware NMS (see below) to clean them up. Don't lower these
further without also re-checking the dedup step still keeps up.

### `--point-margin`: 3 is a safe minimum, 20 is validated and looks better

Query points sampled right at a mask's boundary are the least stable part of
a SAM2 mask frame-to-frame — an edge point is the most likely to flicker
foreground/background as the mask wobbles by a pixel or two between frames.
`--point-margin N` erodes the seed mask inward by N px before sampling, so
points land in the more stable interior. Tested up to 20px on ~400-500px-wide
hand masks with no ill effects (visually confirmed points stay well inside
the silhouette, not clustered uselessly at the centroid). If you're running
on much smaller objects/masks, note there's an automatic fallback: if a
margin would erode a mask to nothing (e.g. a thin sliver of hand visible
during heavy occlusion), it's silently reduced until some pixels survive — so
an aggressive margin degrades gracefully rather than crashing, but a margin
that's frequently hitting the fallback for a given clip means it's oversized
for that clip's object scale.

### `--max-frames`: 35 is a hardware ceiling, not a tuning knob — know its consequences

This is the single most consequential setting. Two things it controls:
1. Stage 2 **only ever looks at the first `max_frames` frames of the clip**,
   full stop — the video tensor itself is truncated. For a 313-frame clip,
   `--max-frames 35` means ~89% of the footage is never lifted to 3D at all.
2. Any detected instance whose first-visible frame falls beyond
   `max_frames` is **silently dropped** from the 3D output entirely (it's
   listed in `summary.json`'s `skipped_instances`, but produces no
   trajectory).

35 is not an arbitrary choice — it's the measured VRAM ceiling for this
pipeline's stage 2 (two sequential backend-tracker calls) on a 23GB A10G at
`grid_size=20, vo_points=2000, target_size=1288`; 38 frames OOMs. If you're
on a bigger GPU (e.g. the 96GB Blackwell setup documented in
[[project-spatialtrackerv2-blackwell-maxquality]]), you can likely raise this
substantially — but re-sweep it for your actual hardware/settings rather than
assuming; VRAM scales with both frame count and `grid_size²` in ways that
aren't obviously separable (see that memory for the actual sweep
methodology). There is currently **no chunking/windowing implemented** for
clips longer than one window fits — see "Known limitations" below.

### `--check-interval` and cross-checkpoint object matching

Lower `check_interval` catches entities entering/exiting frame sooner but
costs more GroundingDINO calls. More importantly: `entity_tracker.py` decides
whether a newly-detected box is an *existing* track that moved, or a
genuinely new object, using box IoU **or** center-proximity (relative to box
size) as a fallback for fast motion. That center-proximity fallback is now
gated to only apply to tracks that have survived at least one real
propagation cycle (see memory — this was a real bug that delayed a second
hand's first detection from frame 0 to frame 90 until fixed), so two
genuinely distinct objects detected in the *same* checkpoint's pass no longer
get incorrectly merged. This should need no further tuning for typical
scenes, but if you see two clearly-different objects of the same label
getting merged into one track, this is the logic to look at first.

### `--grid-size` / `--vo-points`: leave alone unless you're re-sweeping VRAM

These control the background camera-pose-stability point grid, not the
entities you actually care about — entity points are deliberately excluded
from this call (see architecture notes in stage2_lift3d.py's own docstring:
the grid call silently drops low-confidence points, which edge-of-object
points often are). Changing these mainly trades camera-pose accuracy for
VRAM headroom; 20/2000 is the validated A10G-safe setting paired with
`max_frames=35`.

## Known limitations (read before trusting output for anything beyond a demo)

1. **No chunking — `max_frames` is a hard, whole-pipeline ceiling.** Ego4D
   clips run far longer than a ~35-frame/1.5s window. A chunking design
   (overlapping windows + Sim(3) alignment between chunks using the overlap
   frames, or switching to camera-relative rather than world-frame
   coordinates to sidestep alignment entirely) has been proposed but not
   built — deprioritized for now in favor of getting single-window output
   solid first.
2. **The entity call's mixed-start-frame batched query is an unverified
   assumption.** Stage 2 batches every instance's query points into one
   tracker call, each with its own start frame — this is structurally
   supported by the query format but has only actually been exercised with
   all-points-at-t=0 before this pipeline. Not yet validated against a
   known-correct reference.
3. **Never run against real Ego4D footage.** This machine has no Ego4D
   license/AWS credentials; all validation so far is on one substitute
   clip (a Mixkit stock video of washing lettuce). Narration-seeding is
   designed around Ego4D's per-clip narration field specifically.
4. **Only validated on one clip.** Detection reliability on a
   busier/more-cluttered scene, more simultaneous entities, or faster motion
   is unknown.
5. **Same-class disambiguation and severe occlusion remain real ceilings** —
   not fixable by threshold tuning alone. Always spot-check output on
   several frames spread *densely* across the whole clip (not just
   endpoints) before trusting a result; recovery from a mid-clip failure can
   look identical to a permanent one if you only check a few sparse frames.
6. **`~/trace/SpaTrackerV2/stage2_lift3d.py` and `stage3_render.py` are not
   committed to git** (`~/trace/pipeline/` was never a git repo at all)  —
   this guide plus the memory record are currently the durable references.

## Troubleshooting

- **An instance is missing from the final video but was detected in stage
  1**: check `<clip>_trace_summary.json`'s `skipped_instances` — almost
  always means its `seed_frame >= max_frames`.
- **Two real, distinct objects merged into one track**: check
  `bundle.json` for whether both actually got separate `instance_id`s in
  stage 1. If not, see the `--check-interval` note above.
- **A mask looks like it's flickering between two different-colored blobs
  during heavy occlusion**: measure actual per-frame mask pixel overlap
  before assuming it's an occlusion artifact — it may be a genuine
  duplicate/overlapping detection from GroundingDINO that NMS didn't catch.
  See `~/trace/annot/debug_mask_overlap.py` for the measurement approach.
