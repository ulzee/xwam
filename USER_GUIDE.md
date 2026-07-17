# Ego4D 3D Trace Pipeline — User Guide

Goes from the Ego4D FHO annotation set to per-entity (hand/object) 3D
trajectories + optional visualization video, one short narration-anchored
segment at a time. Manifest-driven and range-processable (`--start`/`--end`
by row index) so a bulk run can be split, resumed, or parallelized across
machines.

## Architecture

Five scripts, two conda envs. Every stage after the manifest build
redownloads its own short trimmed segment from S3, uses it, and deletes it
— no raw or resampled video file is ever persisted to disk, only the small
JSON/`.npz` outputs and (optionally) a rendered visualization.

| # | Env | Script | What it does |
|---|---|---|---|
| 0 | any | `build_manifest_ego4d.py` | Joins `fho_main.json` narrations with `full_scale/manifest.csv` S3 locations → one JSONL row per (video, narration) segment, indexable by `--start`/`--end` |
| 1a | `qwen_sam2` | `stage1a.py` | Qwen3-VL identifies task+objects from narration, per manifest row → `identify.json` |
| 1b | `qwen_sam2` | `stage1b.py` | Periodic Grounded-SAM-2 detection+SAM2 tracking per label (hand + each named object) → farthest-point-sampled query points per instance → `bundle.json` |
| 2 | `trace_stv2` | `stage2_lift3d.py` | SpatialTrackerV2 lifts every instance's query points to 3D (grid call for camera-pose stability + entity call for the actual points, unprojected through dense depth + the grid call's bundle-adjusted camera poses) → per-instance `.npz` + `*_trace_summary.json` |
| 3 | `trace_stv2` | `stage3_render.py` | *(optional, one segment at a time)* Redownloads the segment, reruns just the front-end+grid pass for fresh depth/poses, renders a 2x2 grid (original / novel chase-cam / cumulative-trace reference / blank) using stage 2's already-saved trajectories |

Stage 1 is itself two separate processes (1a, then 1b), not one script with
two internal phases — Qwen3-VL (~17.5GB resident) and GroundingDINO+SAM2
don't fit in VRAM together on a 23GB card, so each needs its own process to
genuinely load once for a whole `--start:--end` range rather than swapping
in and out per row.

stage1a.py shares one download across nearby narrations of the same video
instead of downloading once per row: since manifest rows for the same
`video_uid` are always consecutive (guaranteed by how
`build_manifest_ego4d.py` writes them), it groups consecutive rows by
`video_uid`, then greedily splits that group into sub-groups capped at
`--max-group-span-sec` (default 30s) — narrations of the same video can be
minutes or hours apart, so grouping by `video_uid` alone risked a single
huge, slow, mostly-wasted download spanning a video's whole min-to-max
narration range. Within a sub-group, one shared segment covers every
member narration's window, and each narration's own window is carved out
of it with a local ffmpeg trim (no network) before being handed to Qwen.
The shared download is deleted once its sub-group is done. stage1b.py
doesn't get this optimization — it's a separate process and downloads once
per row, replaying the exact recipe stage1a.py recorded in
`identify.json`.

stage1a.py also overlaps downloading with inference via a producer/consumer
pair of threads: a background thread downloads sub-groups ahead of a bounded
`--prefetch` queue (default 2) while the main thread runs Qwen on whatever's
already ready, and only deletes a sub-group's shared download once it's
fully done with it. Downloading is CPU/network-bound (ffmpeg) and inference
is GPU-bound, so they don't compete for the same resource — real Ego4D
narration timing (median gap between consecutive narrations of the same
video: 1.8s; 98% within the 30s `--max-group-span-sec` default) means the
download side is almost always idle-waiting on the queue anyway, so this
hides most of the download+re-encode latency behind inference time for
free. `--prefetch N` keeps roughly N+1 shared segments on disk at once (N
waiting in the queue, plus the one actively downloading).

**Stage 2 requires stage 1 to have run first** for a row (it reads
`bundle.json`); running stage 2 over a wider `--start:--end` than stage 1
has covered is safe — uncovered rows are skipped with a log line, not an
error. All four stages **default to overwriting** existing output when
re-run — pass `--check-existing` to skip rows/chunks/renders whose output
already exists instead (resume mode), so re-running the same command
without the flag always redoes the work, and with it only fills in gaps.

`stage1a.py`'s `--end` is optional — omit it to process every row from
`--start` through the end of the manifest. The other stages still require
an explicit `--end`.

## Quick start (validated configuration)

All commands assume `~/ego4d/` holds the Ego4D download (`data/v2/...`) and
run from `~/trace/pipeline/`. This is the exact sequence + flags last run
successfully end-to-end (rows 0-3 of a live manifest, real Ego4D footage).

```bash
cd ~/trace/pipeline

# 0. Build the manifest once (~2GB fho_main.json load, takes a minute or two).
#    No GPU/conda env needed -- pure stdlib. Re-run only if the annotation
#    files change; --limit is for smoke-testing the join on a handful of rows.
python3 build_manifest_ego4d.py --out ~/ego4d/manifest.jsonl

# 1a. Qwen3-VL scene understanding, one process for the whole range.
conda run -n qwen_sam2 python stage1a.py \
  --manifest ~/ego4d/manifest.jsonl --start 0 --end 100 \
  --save-root ~/ego4d/points --tmp-dir ~/ego4d/scratch

# 1b. GroundingDINO+SAM2 tracking, one process for the whole range.
conda run -n qwen_sam2 python stage1b.py \
  --manifest ~/ego4d/manifest.jsonl --start 0 --end 100 \
  --save-root ~/ego4d/points --tmp-dir ~/ego4d/scratch

# 2. SpatialTrackerV2 3D lifting. Needs the SpaTrackerV2 checkout on
#    PYTHONPATH -- cd there first, same as stage3 below.
cd ~/trace/SpaTrackerV2
PYTHONPATH="$PWD:$PYTHONPATH" conda run -n trace_stv2 python \
  ~/trace/pipeline/stage2_lift3d.py \
  --manifest ~/ego4d/manifest.jsonl --start 0 --end 100 \
  --save-root ~/ego4d/points --tmp-dir ~/ego4d/scratch

# 3. (optional) Render one already-fully-processed segment to spot-check it.
PYTHONPATH="$PWD:$PYTHONPATH" conda run -n trace_stv2 python \
  ~/trace/pipeline/stage3_render.py \
  --manifest ~/ego4d/manifest.jsonl --index 0 \
  --save-root ~/ego4d/points --tmp-dir ~/ego4d/scratch
```

Every flag left unset above is at its validated default (see "Full flag
reference" and "Recommended defaults" below) — this is genuinely the
minimal working invocation, not an abbreviation of a longer recommended one.

Outputs land under `~/ego4d/points/<segment_id>/`:
- `identify.json` — stage 1a's task/objects + the exact download recipe (bucket/key/start_sec/duration_sec) every later stage replays
- `00000/bundle.json` — stage 1b's detections + query points for chunk 0 (inspect this first if something looks wrong; most segments are exactly one chunk, see `--target-fps`/`--max-frames` below)
- `00000/<segment_id>_trace_summary.json` — stage 2's per-instance metadata + which instances got skipped and why
- `00000/<segment_id>_trace_<instance_id>.npz` — per-instance `points3d` (T,N,3), `points2d` (T,N,2), `visibility` (T,N bool), `frame_indices`, `seed_frame`
- `00000/<segment_id>_trace_render.mp4` — only if stage 3 was run for this segment

Nothing else persists — no raw download, no resampled clip, no scene/depth
dump. `du -sh ~/ego4d/points` after a bulk run should be dominated by
whatever stage-3 renders you've kept, not by stage 1/2's own footprint.

## Manifest schema

One JSON object per line, from `build_manifest_ego4d.py`:
```json
{"index": 0, "segment_id": "<video_uid>_<local_narration_idx:03d>",
 "video_uid": "...", "bucket": "...", "key": "...",
 "native_fps": 30.0, "width": 1920, "height": 1440,
 "video_duration_sec": 958.1,
 "narration_text": "Embodiment picks the cat teaser wand from the toy box",
 "narration_timestamp_sec": 606.37}
```
`index` is a stable sort by `(video_uid, narration_timestamp_sec)` — every
`--start`/`--end` range refers to this. Multiple narrations of the same
video are just consecutive rows, not a nested structure.

Only camera-wearer-tagged narrations are kept — Ego4D annotators tag each
narration `#C` (the camera wearer, whose action the video actually shows)
or `#O` (another person in the scene); only `#C` ones make it into the
manifest. `narration_text` also has the raw annotator shorthand cleaned up
before it lands in the manifest: the leading `#C` tag and any other inline
`#tag` (e.g. `#Unsure`) are stripped, and the bare `C`/`CC` subject/object
pronoun annotators used for "the camera wearer" (e.g. `"#C C picks..."`,
or mid-sentence like `"...passes it to C"`) is replaced with `embodiment`,
so it reads as a normal sentence. Narrated actions
flagged `is_rejected`/`is_invalid_annotation` in `fho_main.json` are dropped
during the join.

## Full flag reference

Shared across stage1a/1b/2 bulk mode:

| Flag | Meaning |
|---|---|
| `--manifest` | JSONL from `build_manifest_ego4d.py` |
| `--start` / `--end` | row index range, `[start, end)`. `--end` is optional on stage1a.py only (omit to run through the end of the manifest); stage1b.py/stage2_lift3d.py still require both. |
| `--save-root` | output root, `<save-root>/<segment_id>/...` |
| `--tmp-dir` | scratch dir for downloaded/resampled clips (default `~/ego4d/scratch`) — safe to delete anytime, nothing durable lives there |
| `--check-existing` | off (default: overwrite) | skip a row/chunk whose output already exists instead of redoing it (resume mode) — present on all four scripts, see "Full flag reference" per-script tables below for exactly what existence check each one uses |

stage1a.py only:

| Flag | Default | Meaning |
|---|---|---|
| `--target-fps` | 10 | used only to size the downloaded segment's duration (see "segment duration" below) — must match stage1b.py's `--target-fps` |
| `--max-frames` | 35 | used only to size the downloaded segment's duration — must match stage1b.py's `--max-frames` |
| `--max-group-span-sec` | 30.0 | cap on how much wall-clock video one shared per-video download may cover (see "shared per-video downloads" below) |
| `--prefetch` | 2 | max sub-groups the download thread may keep downloaded-and-waiting ahead of inference (producer/consumer backpressure — see "Architecture" above) |
| `--check-existing` | off | skip a row if its `identify.json` already exists |

stage1b.py only:

| Flag | Default | Meaning |
|---|---|---|
| `--target-fps` | 10 | resample to this fps before chunking |
| `--max-frames` | 35 | hard ceiling on frames per SpatialTrackerV2 window (see "max-frames" below) |
| `--num-checks` | 3 | periodic re-detection passes within the window; `check_interval` derives as `max_frames // num_checks` unless overridden |
| `--check-interval` | derived | explicit frame interval between re-detection passes |
| `--box-threshold` | 0.3 | GroundingDINO box confidence threshold |
| `--text-threshold` | 0.25 | GroundingDINO text-match threshold |
| `--point-budget` | 10 | query points sampled per instance |
| `--point-margin` | 3 | erode each instance's seed mask inward by this many px before sampling query points |
| `--stale-frames` | 10 | frames a track can go undetected before being dropped |
| `--max-concurrent-per-label` | 4 | cap on simultaneously-tracked instances per label |
| `--check-existing` | off | skip a row if its chunk-0 `bundle.json` already exists |

stage2_lift3d.py bulk mode only:

| Flag | Default | Meaning |
|---|---|---|
| `--grid-size` | 20 | background/VO query grid density (query count = grid_size²) |
| `--vo-points` | 2000 | bundle-adjustment point budget for the grid call |
| `--target-size` | 1288 | front-end input resolution |
| `--track-mode` | offline | SpatialTrackerV2 backend variant |
| `--check-existing` | off | skip a chunk if its `*_trace_summary.json` already exists |

stage3_render.py:

| Flag | Default | Meaning |
|---|---|---|
| `--index` | required | single manifest row to render |
| `--chunk-index` | 0 | which chunk of that segment to render |
| `--out` | `<chunk_dir>/<segment_id>_trace_render.mp4` | output path |
| `--fixed` | off | hold the novel camera fixed (frame-0 pose) instead of chase-cam |
| `--fps` | source clip's fps | output playback fps |
| `--trail-len` | 7 | decaying-trail window (frames) on the top panels |
| `--ref-frame` | 0 | background frame for the bottom-left cumulative-trace panel |
| `--side-frac` / `--back-frac` / `--height-frac` / `--target-frac` | 1.0 / 1.0 / 0.4 / 1.5 | chase-cam rig offset, as fractions of median scene depth |
| `--check-existing` | off | skip rendering if the output mp4 already exists |

## Recommended defaults, and why (read before changing hyperparameters)

### Segment duration (stage1a.py): computed per row, not a fixed constant

stage1a.py doesn't take a `--duration-sec` flag — it derives how long a
segment to download from each row's own `native_fps` plus `--target-fps`/
`--max-frames`: `duration_sec = max_frames * stride / native_fps`, where
`stride = round(native_fps / target_fps)` (same stride math
`frame_sampling.resample_video` uses). This lands on *exactly* one full
`max_frames` chunk after resampling regardless of a video's native fps —
e.g. 3.5s at 30fps, 2.92s at 24fps, 3.50s at 59.94fps — rather than the
old fixed-duration approach, which assumed every video was ~30fps and
under/over-shot a full chunk on anything else. Every segment in the
validated run above ended up as exactly one `00000/` chunk. If you raise
`--max-frames` here, more chunks get created automatically downstream
(stage1b.py's chunking is general) — but re-check GPU headroom (see
`--max-frames` below) before doing so, and keep `--target-fps`/`--max-frames`
here matching what you pass to stage1b.py, since stage1a.py never talks to
stage1b.py directly — they only agree via the numbers baked into
`identify.json`/`bundle.json` (stage1a.py records the actual `duration_sec`
it used in `identify.json`, so stage1b.py replays that exact value rather
than recomputing it).

### `--max-group-span-sec` (stage1a.py): shared per-video downloads, bounded

30s is arbitrary but deliberately small relative to a full Ego4D video
(some run 30-40+ minutes) — narrations of the same `video_uid` are not
necessarily close together in time, so grouping strictly by `video_uid`
(an earlier version of this optimization did exactly that) hit a real
failure: two narrations 350+ seconds apart in the same video turned into
one shared download attempting to fetch that entire 350s span, which timed
out (ffmpeg's default 120s budget) and, worse, would have downloaded ~100x
more video than actually needed even had it succeeded. `make_subgroups()`
in stage1a.py greedily merges only consecutive narrations whose combined
span stays under this cap; anything farther apart falls into its own
sub-group (and its own download) instead. Raise it only if you've checked
your manifest's narration spacing is consistently tighter than that in
your `--start:--end` range and want fewer, larger shared downloads.

Earlier validation (on a single non-Ego4D stock clip) found the defaults
too strict for two-simultaneous-similar-objects scenes (e.g. two hands):
GroundingDINO's *best* second-hand candidate often lands below 0.3
confidence, silently losing a real instance rather than producing a bad
detection you could catch. This session's bulk runs against real Ego4D
clips (including genuine two-hand scenes) worked fine at the 0.3/0.25
defaults, so they're the documented default now — but if you spot-check a
render and a hand or object never gets picked up, try 0.2/0.15 first. This
is safe to lower because `entity_tracker.py`'s `filter_detections()` does
containment-aware NMS to clean up the resulting extra candidates; don't
lower further without re-checking the dedup step keeps up.

### `--point-margin`: 3 is a safe minimum, up to 20 is validated and looks better

Query points sampled right at a mask's boundary are the least stable part
of a SAM2 mask frame-to-frame — an edge point is the most likely to flicker
foreground/background as the mask wobbles by a pixel or two between frames.
`--point-margin N` erodes the seed mask inward by N px before sampling, so
points land in the more stable interior. Tested up to 20px on ~400-500px
hand masks with no ill effects. There's an automatic fallback if a margin
would erode a mask to nothing (e.g. a thin sliver of hand visible during
heavy occlusion): it's silently reduced until some pixels survive.

### `--max-frames`: 35 is a hardware ceiling, not a tuning knob — know its consequences

The single most consequential setting. Two things it controls:
1. Stage 2 only ever lifts the first `max_frames` frames of each chunk to
   3D — the video tensor itself is truncated there.
2. Any detected instance whose first-visible frame falls beyond
   `max_frames` within its chunk is dropped from the 3D output (listed in
   `*_trace_summary.json`'s `skipped_instances`, no trajectory produced).

35 is the measured VRAM ceiling for stage 2's two sequential backend-tracker
calls on a 23GB A10G at `grid_size=20, vo_points=2000, target_size=1288`
(peaked at 21.51GB in this session's runs); 38 frames OOMs. On a bigger GPU
you can likely raise this — but re-sweep for your actual hardware rather
than assuming; VRAM scales with both frame count and `grid_size²` in ways
that aren't obviously separable.

### The entity call's mixed-start-frame batched query: now empirically validated

Stage 2 batches every instance's query points into one tracker call, each
with its own start frame. This session's real runs had instances seeded at
both frame 0 and frame 11 within the same entity call, and both tracked and
rendered correctly (visually confirmed in `stage3_render.py` output) — this
was previously only a structural assumption about the query format, not a
confirmed behavior.

### `--grid-size` / `--vo-points`: leave alone unless you're re-sweeping VRAM

These control the background camera-pose-stability point grid, not the
entities you actually care about — entity points are deliberately excluded
from this call (the grid call silently drops low-confidence points, which
edge-of-object points often are). Changing these mainly trades camera-pose
accuracy for VRAM headroom; 20/2000 is the validated A10G-safe setting
paired with `max_frames=35`.

## Known limitations

1. **Chunking exists but defaults to one chunk per segment.** stage1a.py's
   computed segment duration is deliberately sized so each narration-anchored
   segment is short enough to fit one `--max-frames` window; stage1b.py's
   chunking machinery handles longer segments if you raise `--max-frames` (on
   both stage1a.py and stage1b.py), but that combination hasn't been
   exercised yet.
2. **Same-class disambiguation and severe occlusion remain real ceilings** —
   not fixable by threshold tuning alone. Spot-check `stage3_render.py`
   output on several frames spread *densely* across the whole clip (not
   just endpoints); recovery from a mid-clip failure can look identical to
   a permanent one if you only check a few sparse frames.
3. **`~/trace/pipeline/` is a git repo** (`git@github.com:ulzee/xwam.git`),
   but `build_manifest_ego4d.py`, `dataset_io.py`, `stage1a.py`, `stage1b.py`
   are currently untracked — `git add` them when you want this session's
   re-architecture actually committed. `~/trace/SpaTrackerV2/` is also its
   own git repo (currently on a `render` branch), separate from the
   pipeline repo.
4. **`run_pipeline.sh` is stale** — it called the now-deleted
   `stage1_detect.py`'s single-clip modes for ad-hoc, non-Ego4D-manifest
   debugging. Not currently maintained; use `stage1a.py`/`stage1b.py`
   against a one-row manifest slice for the same purpose, or call
   `scene_understanding.py`/`entity_tracker.py` directly.

## Troubleshooting

- **A segment silently does nothing under stage 2**: check whether
  `<save-root>/<segment_id>/00000/bundle.json` exists — stage 2 skips rows
  stage 1 hasn't reached yet, logging `SKIP: no chunk directories under ...`
  rather than erroring.
- **An instance is missing from the final render but was detected in stage
  1b**: check `*_trace_summary.json`'s `skipped_instances` — almost always
  means its `seed_frame >= max_frames` within its chunk.
- **Two real, distinct objects merged into one track**: check
  `bundle.json` for whether both actually got separate `instance_id`s in
  stage 1b. If not, look at `entity_tracker.py`'s box-IoU/center-proximity
  matching logic (`--check-interval` controls how often it re-checks).
- **A mask looks like it's flickering between two different-colored blobs
  during heavy occlusion**: measure actual per-frame mask pixel overlap
  before assuming it's an occlusion artifact — it may be a genuine
  duplicate/overlapping detection from GroundingDINO that NMS didn't catch.
  See `~/trace/annot/debug_mask_overlap.py` for the measurement approach.
- **`ModuleNotFoundError: No module named 'boto3'`**: install it into
  whichever conda env is missing it (`qwen_sam2` and `trace_stv2` both need
  it — every stage downloads directly now, there's no separate orchestrator
  process holding AWS credentials).
- **stage1a.py: `FFmpeg cannot edit existing files in-place` / a later row
  fails with `No such file or directory` on the SAME path / garbled
  `Invalid NAL unit size` errors reading a downloaded clip**: this was a
  real bug (fixed) where the shared per-sub-group download and a per-row
  local trim could land on the identical filename (both were
  `f"{video_uid}_{N:03d}_raw.mp4"`, just with `N` meaning two different
  counters that both start at 0) — corrupting the shared file via an
  in-place-edit refusal, or via a producer/consumer race once downloading
  moved to a background thread. The shared download is now named
  `f"{video_uid}_shared{N:03d}_raw.mp4"`, which can't collide, plus an
  `assert` in stage1a.py's main loop that fails loudly instead of silently
  corrupting data if this class of bug ever reappears. If you see this on
  a currently-checked-out version, you're on an old copy of stage1a.py.
