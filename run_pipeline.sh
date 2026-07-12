#!/usr/bin/env bash
# End-to-end 3D trace extraction pipeline: 2D video -> per-instance 3D
# trajectories + a combined visualization video.
#
# Spans two conda envs because that's how they were set up across sessions:
#   qwen_sam2   -- Qwen3-VL scene understanding + GroundingDINO + SAM2 (stage 1)
#   trace_stv2  -- SpatialTrackerV2 3D lifting + rendering (stages 2-3)
#
# Usage:
#   ./run_pipeline.sh --video CLIP.mp4 --out-dir OUT/ \
#       [--narration "..."] [--point-budget 10] [--point-margin 3] [--max-frames 35] \
#       [--check-interval 15] [--box-threshold 0.3] [--text-threshold 0.25] \
#       [--grid-size 20] [--vo-points 2000]
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STV2_DIR="$HOME/trace/SpaTrackerV2"

VIDEO=""
OUT_DIR=""
NARRATION="a person washing lettuce in the sink"
POINT_BUDGET=10
POINT_MARGIN=3
MAX_FRAMES=35
CHECK_INTERVAL=15
BOX_THRESHOLD=0.3
TEXT_THRESHOLD=0.25
GRID_SIZE=20
VO_POINTS=2000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --narration) NARRATION="$2"; shift 2 ;;
    --point-budget) POINT_BUDGET="$2"; shift 2 ;;
    --point-margin) POINT_MARGIN="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --check-interval) CHECK_INTERVAL="$2"; shift 2 ;;
    --box-threshold) BOX_THRESHOLD="$2"; shift 2 ;;
    --text-threshold) TEXT_THRESHOLD="$2"; shift 2 ;;
    --grid-size) GRID_SIZE="$2"; shift 2 ;;
    --vo-points) VO_POINTS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$VIDEO" || -z "$OUT_DIR" ]]; then
  echo "Usage: $0 --video CLIP.mp4 --out-dir OUT/ [--narration '...'] [--point-budget 10] [--max-frames 35] ..."
  exit 1
fi

mkdir -p "$OUT_DIR"
CLIP_STEM="$(basename "${VIDEO%.*}")"
BUNDLE="$OUT_DIR/bundle.json"
SUMMARY="$OUT_DIR/${CLIP_STEM}_trace_summary.json"
RENDER_OUT="$OUT_DIR/${CLIP_STEM}_trace_render.mp4"

echo "=== Stage 1: scene understanding + entity detection/tracking (qwen_sam2 env) ==="
(cd "$PIPELINE_DIR" && conda run -n qwen_sam2 python stage1_detect.py \
  --video "$VIDEO" \
  --narration "$NARRATION" \
  --point-budget "$POINT_BUDGET" \
  --point-margin "$POINT_MARGIN" \
  --check-interval "$CHECK_INTERVAL" \
  --box-threshold "$BOX_THRESHOLD" \
  --text-threshold "$TEXT_THRESHOLD" \
  --out "$BUNDLE")

echo "=== Stage 2: 3D lifting with SpatialTrackerV2 (trace_stv2 env) ==="
(cd "$STV2_DIR" && conda run -n trace_stv2 python stage2_lift3d.py \
  --bundle "$BUNDLE" \
  --out-dir "$OUT_DIR" \
  --max-frames "$MAX_FRAMES" \
  --grid-size "$GRID_SIZE" \
  --vo-points "$VO_POINTS")

echo "=== Stage 3: combined trace visualization (trace_stv2 env) ==="
(cd "$STV2_DIR" && conda run -n trace_stv2 python stage3_render.py \
  --summary "$SUMMARY" \
  --out "$RENDER_OUT")

echo ""
echo "Done. Outputs in $OUT_DIR:"
echo "  bundle (stage 1):        $BUNDLE"
echo "  trace summary (stage 2): $SUMMARY"
echo "  per-instance npz:        ${OUT_DIR}/${CLIP_STEM}_trace_<instance_id>.npz"
echo "  scene npz:               ${OUT_DIR}/${CLIP_STEM}_scene.npz"
echo "  visualization (stage 3): $RENDER_OUT"
