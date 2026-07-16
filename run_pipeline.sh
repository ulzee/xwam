#!/usr/bin/env bash
# End-to-end 3D trace extraction pipeline: 2D video -> per-instance 3D
# trajectories + a combined visualization video.
#
# Spans two conda envs because that's how they were set up across sessions:
#   qwen_sam2   -- Qwen3-VL scene understanding + GroundingDINO + SAM2 (stage 1)
#   trace_stv2  -- SpatialTrackerV2 3D lifting + rendering (stages 2-3)
#
# Two modes, mutually exclusive:
#
#   Single-clip mode (--out-dir): one 35-frame (--max-frames) window starting
#   at frame 0 of the clip, at native fps. Meant for spot-checking/dev.
#     ./run_pipeline.sh --video CLIP.mp4 --out-dir OUT/ \
#         [--narration "..."] [--point-budget 10] [--point-margin 3] [--max-frames 35] \
#         [--num-checks 3] [--check-interval N] [--box-threshold 0.3] [--text-threshold 0.25] \
#         [--grid-size 20] [--vo-points 2000] [--save-scene]
#
#   Chunked bulk-annotation mode (--save-root): resamples the clip to
#   --target-fps ONCE, then tiles it into consecutive --max-frames windows
#   ("chunks") from --start-frame to the end of the clip, producing one
#   trajectory set per chunk. Models load once per stage and are reused
#   across all chunks (not reloaded per chunk). Outputs land in
#   <save-root>/<video_stem>/<chunk_idx:05d>/.
#     ./run_pipeline.sh --video CLIP.mp4 --save-root OUT/ \
#         [--target-fps 10] [--start-frame 0] [--narration "..."] [--point-budget 10] \
#         [--point-margin 3] [--max-frames 35] [--num-checks 3] [--check-interval N] \
#         [--box-threshold 0.3] [--text-threshold 0.25] [--grid-size 20] [--vo-points 2000] \
#         [--save-scene]
#
# --max-frames is threaded to BOTH stage 1 (caps detection to the first N
# frames of each window, since anything discovered later can't be 3D-lifted
# anyway) and stage 2 (the SpatialTrackerV2 window ceiling) -- keeping them
# in sync is the whole point, so don't set them independently unless you
# know why.
#
# --check-interval defaults to unset, letting stage 1 derive it as
# max-frames / num-checks (i.e. "check num-checks times within the window")
# instead of a fixed interval that doesn't scale with --max-frames.
#
# --save-scene is OFF by default: stage 2's <clip>_scene.npz (raw video +
# dense depth + camera params) is ~99% of this pipeline's disk footprint
# (~500MB/window at 35 frames, 728x1288) and is only needed to render stage
# 3's visualization -- not for the trajectory data itself. Pass --save-scene
# when you want the render (e.g. spot-checking a handful of clips/chunks);
# leave it off for bulk dataset runs where only the per-instance trajectory
# npz's matter. Stage 3 is skipped automatically when --save-scene isn't
# passed, since it has nothing to render from otherwise. In chunked mode,
# stage 3 (if enabled) runs once per chunk.
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STV2_DIR="$HOME/trace/SpaTrackerV2"

VIDEO=""
OUT_DIR=""
SAVE_ROOT=""
TARGET_FPS=10
START_FRAME=0
NARRATION="a person washing lettuce in the sink"
POINT_BUDGET=10
POINT_MARGIN=3
MAX_FRAMES=35
NUM_CHECKS=3
CHECK_INTERVAL=""
BOX_THRESHOLD=0.3
TEXT_THRESHOLD=0.25
GRID_SIZE=20
VO_POINTS=2000
SAVE_SCENE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --save-root) SAVE_ROOT="$2"; shift 2 ;;
    --target-fps) TARGET_FPS="$2"; shift 2 ;;
    --start-frame) START_FRAME="$2"; shift 2 ;;
    --narration) NARRATION="$2"; shift 2 ;;
    --point-budget) POINT_BUDGET="$2"; shift 2 ;;
    --point-margin) POINT_MARGIN="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --num-checks) NUM_CHECKS="$2"; shift 2 ;;
    --check-interval) CHECK_INTERVAL="$2"; shift 2 ;;
    --box-threshold) BOX_THRESHOLD="$2"; shift 2 ;;
    --text-threshold) TEXT_THRESHOLD="$2"; shift 2 ;;
    --grid-size) GRID_SIZE="$2"; shift 2 ;;
    --vo-points) VO_POINTS="$2"; shift 2 ;;
    --save-scene) SAVE_SCENE=1; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$VIDEO" ]] || { [[ -z "$OUT_DIR" ]] && [[ -z "$SAVE_ROOT" ]]; } || { [[ -n "$OUT_DIR" ]] && [[ -n "$SAVE_ROOT" ]]; }; then
  echo "Usage: $0 --video CLIP.mp4 (--out-dir OUT/ | --save-root OUT/) [options]"
  echo "  --out-dir:   single-clip mode (one window, spot-checking/dev)"
  echo "  --save-root: chunked bulk-annotation mode (--target-fps, --start-frame apply)"
  exit 1
fi

CLIP_STEM="$(basename "${VIDEO%.*}")"

if [[ -n "$SAVE_ROOT" ]]; then
  # ============================== Chunked mode ==============================
  VIDEO_OUT_DIR="$SAVE_ROOT/$CLIP_STEM"

  echo "=== Stage 1: scene understanding + chunked entity detection/tracking (qwen_sam2 env) ==="
  STAGE1_ARGS=(
    --video "$VIDEO"
    --save-root "$SAVE_ROOT"
    --target-fps "$TARGET_FPS"
    --start-frame "$START_FRAME"
    --narration "$NARRATION"
    --point-budget "$POINT_BUDGET"
    --point-margin "$POINT_MARGIN"
    --max-frames "$MAX_FRAMES"
    --num-checks "$NUM_CHECKS"
    --box-threshold "$BOX_THRESHOLD"
    --text-threshold "$TEXT_THRESHOLD"
  )
  if [[ -n "$CHECK_INTERVAL" ]]; then
    STAGE1_ARGS+=(--check-interval "$CHECK_INTERVAL")
  fi
  (cd "$PIPELINE_DIR" && conda run -n qwen_sam2 python stage1_detect.py "${STAGE1_ARGS[@]}")

  echo "=== Stage 2: chunked 3D lifting with SpatialTrackerV2 (trace_stv2 env) ==="
  STAGE2_ARGS=(
    --video "$VIDEO"
    --save-root "$SAVE_ROOT"
    --max-frames "$MAX_FRAMES"
    --grid-size "$GRID_SIZE"
    --vo-points "$VO_POINTS"
  )
  if [[ "$SAVE_SCENE" -eq 1 ]]; then
    STAGE2_ARGS+=(--save-scene)
  fi
  (cd "$STV2_DIR" && PYTHONPATH="$STV2_DIR:${PYTHONPATH:-}" conda run -n trace_stv2 python "$PIPELINE_DIR/stage2_lift3d.py" "${STAGE2_ARGS[@]}")

  if [[ "$SAVE_SCENE" -eq 1 ]]; then
    echo "=== Stage 3: per-chunk trace visualization (trace_stv2 env) ==="
    for CHUNK_DIR in "$VIDEO_OUT_DIR"/[0-9][0-9][0-9][0-9][0-9]; do
      [[ -d "$CHUNK_DIR" ]] || continue
      CHUNK_SUMMARY="$CHUNK_DIR/${CLIP_STEM}_trace_summary.json"
      [[ -f "$CHUNK_SUMMARY" ]] || continue
      (cd "$STV2_DIR" && PYTHONPATH="$STV2_DIR:${PYTHONPATH:-}" conda run -n trace_stv2 python "$PIPELINE_DIR/stage3_render.py" \
        --summary "$CHUNK_SUMMARY" \
        --out "$CHUNK_DIR/${CLIP_STEM}_trace_render.mp4")
    done
  else
    echo "=== Stage 3: skipped (--save-scene not passed, no scene data to render from) ==="
  fi

  echo ""
  echo "Done. Outputs in $VIDEO_OUT_DIR/<chunk_idx:05d>/:"
  echo "  bundle (stage 1):        bundle.json"
  echo "  trace summary (stage 2): ${CLIP_STEM}_trace_summary.json"
  echo "  per-instance npz:        ${CLIP_STEM}_trace_<instance_id>.npz"
  if [[ "$SAVE_SCENE" -eq 1 ]]; then
    echo "  scene npz:               ${CLIP_STEM}_scene.npz"
    echo "  visualization (stage 3): ${CLIP_STEM}_trace_render.mp4"
  else
    echo "  scene npz:               (skipped, pass --save-scene to keep it)"
    echo "  visualization (stage 3): (skipped, pass --save-scene to render)"
  fi

else
  # ============================ Single-clip mode ============================
  mkdir -p "$OUT_DIR"
  BUNDLE="$OUT_DIR/bundle.json"
  SUMMARY="$OUT_DIR/${CLIP_STEM}_trace_summary.json"
  RENDER_OUT="$OUT_DIR/${CLIP_STEM}_trace_render.mp4"

  echo "=== Stage 1: scene understanding + entity detection/tracking (qwen_sam2 env) ==="
  STAGE1_ARGS=(
    --video "$VIDEO"
    --narration "$NARRATION"
    --point-budget "$POINT_BUDGET"
    --point-margin "$POINT_MARGIN"
    --max-frames "$MAX_FRAMES"
    --num-checks "$NUM_CHECKS"
    --box-threshold "$BOX_THRESHOLD"
    --text-threshold "$TEXT_THRESHOLD"
    --out "$BUNDLE"
  )
  if [[ -n "$CHECK_INTERVAL" ]]; then
    STAGE1_ARGS+=(--check-interval "$CHECK_INTERVAL")
  fi
  (cd "$PIPELINE_DIR" && conda run -n qwen_sam2 python stage1_detect.py "${STAGE1_ARGS[@]}")

  echo "=== Stage 2: 3D lifting with SpatialTrackerV2 (trace_stv2 env) ==="
  STAGE2_ARGS=(
    --bundle "$BUNDLE"
    --out-dir "$OUT_DIR"
    --max-frames "$MAX_FRAMES"
    --grid-size "$GRID_SIZE"
    --vo-points "$VO_POINTS"
  )
  if [[ "$SAVE_SCENE" -eq 1 ]]; then
    STAGE2_ARGS+=(--save-scene)
  fi
  (cd "$STV2_DIR" && PYTHONPATH="$STV2_DIR:${PYTHONPATH:-}" conda run -n trace_stv2 python "$PIPELINE_DIR/stage2_lift3d.py" "${STAGE2_ARGS[@]}")

  if [[ "$SAVE_SCENE" -eq 1 ]]; then
    echo "=== Stage 3: combined trace visualization (trace_stv2 env) ==="
    (cd "$STV2_DIR" && PYTHONPATH="$STV2_DIR:${PYTHONPATH:-}" conda run -n trace_stv2 python "$PIPELINE_DIR/stage3_render.py" \
      --summary "$SUMMARY" \
      --out "$RENDER_OUT")
  else
    echo "=== Stage 3: skipped (--save-scene not passed, no scene data to render from) ==="
  fi

  echo ""
  echo "Done. Outputs in $OUT_DIR:"
  echo "  bundle (stage 1):        $BUNDLE"
  echo "  trace summary (stage 2): $SUMMARY"
  echo "  per-instance npz:        ${OUT_DIR}/${CLIP_STEM}_trace_<instance_id>.npz"
  if [[ "$SAVE_SCENE" -eq 1 ]]; then
    echo "  scene npz:               ${OUT_DIR}/${CLIP_STEM}_scene.npz"
    echo "  visualization (stage 3): $RENDER_OUT"
  else
    echo "  scene npz:               (skipped, pass --save-scene to keep it)"
    echo "  visualization (stage 3): (skipped, pass --save-scene to render)"
  fi
fi
