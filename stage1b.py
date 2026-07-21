"""Stage 1b (runs in the `qwen_sam2` conda env): GroundingDINO+SAM2 tracking
only -- reads stage1a.py's identify.json (task/objects/background + the
exact download recipe it used), redownloads the same segment, resamples it,
tracks hands+every named "objects" item AND every named "background" item
across the clip in consecutive max-frames windows ("chunks"), and writes one
bundle.json per chunk: "instances" (hand + "objects") and "background" (the
identify.json "background" list), each segmented the same way. Manifest-
driven bulk mode only; for every row in [--start, --end) with an
identify.json already present, deletes the segment again once done.

Split out of a single interleaved stage1_detect.py into its own process
specifically so GroundingDINO+SAM2 only ever has to load ONCE for a whole
--start:--end range, without fighting Qwen3-VL (stage1a.py) for VRAM on a
23GB card -- see stage1a.py's docstring.

Like stage1a.py, the raw (and resampled) video is only ever a scratch file:
downloaded to --tmp-dir, used to produce this row's bundle.json(s), then
deleted in main()'s `finally` -- nothing durable ever holds onto the actual
video. --visualize is an opt-in escape hatch for eyeballing segmentation
quality without keeping the source video around: it saves one small JPG per
detected item (its seed frame with its mask + query points overlaid) under
save-root/<segment_id>/viz/objects/ or .../viz/background/, which -- unlike
the video -- IS kept. See run_track's docstring for the full breakdown.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse
import json
import time

import cv2
import decord
import numpy as np

from entity_tracker import track_prompted_entities_chunks, load_gdino, load_sam2_predictor
from fps_sample import farthest_point_sample
from frame_sampling import resample_video, compute_chunk_starts
from dataset_io import download_segment, load_manifest_range, cleanup

# Same palette as stage3_render.py's INSTANCE_COLORS, kept separate since the
# two modules don't otherwise share any code.
INSTANCE_COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (0, 200, 200), (240, 50, 230), (170, 110, 40), (128, 128, 0),
]


def draw_seed_visualization(frame_rgb, mask, points_px, label, color):
    """Overlays a translucent `color` mask + query points on frame_rgb (HxWx3
    uint8, RGB, as returned by decord). Returns a BGR uint8 image ready for
    cv2.imwrite."""
    img = frame_rgb.astype(np.float32)
    overlay = np.zeros_like(img)
    overlay[mask] = color
    alpha = 0.45
    blended = np.where(mask[..., None], img * (1 - alpha) + overlay * alpha, img)
    img_bgr = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_RGB2BGR)
    for x, y in points_px:
        pt = (int(round(x)), int(round(y)))
        cv2.circle(img_bgr, pt, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img_bgr, pt, 3, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.putText(img_bgr, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img_bgr


def run_track(video_path, save_root, segment_id, narration, task, objects, background_objects, resampled_path,
              download_recipe, point_budget=10, check_interval=None, num_checks=3, max_frames=35, target_fps=10,
              stale_frames=10, box_threshold=0.3, text_threshold=0.25, max_concurrent_per_label=4,
              point_margin=3, background_point_budget=None, visualize=False, verbose=True):
    """Resamples video_path once, tiles it into consecutive max_frames
    windows ("chunks") from frame 0 to the end of the clip, tracks every
    chunk, writes one bundle.json per chunk to
    save_root/<segment_id>/<chunk_idx:%05d>/bundle.json. A chunk with zero
    detected instances still gets a bundle.json (empty "instances": []) so
    stage2_lift3d.py can enumerate and skip it explicitly rather than
    silently missing a chunk index. `download_recipe`
    ({bucket, key, start_sec, duration_sec}) is embedded in every chunk's
    bundle.json so stage2/stage3 can replay the identical S3 trim without
    needing the manifest again. Assumes GroundingDINO+SAM2 are already
    loaded (caller's responsibility, so a bulk caller loads them once for a
    whole range instead of once per row).

    `background_objects` (identify.json's "background" list -- untouched
    scenery like "coffee table", "remote control") is detected+tracked the
    same way as `objects`, through the same GDINO+SAM2 pass, and written to
    the chunk's bundle.json as a "background" list parallel to "instances"
    (same per-entry shape: instance_id/label/category/.../query_points_px).
    Where a background item's mask overlaps a same-frame hand/object
    instance's mask, the overlap is subtracted from the background item's
    mask before query points are sampled -- "objects" (and hands) always
    win a collision, since those are the ones actually being manipulated.
    If a background item's mask is fully consumed by that subtraction, it's
    dropped for that chunk (logged, not an error) exactly like an instance
    that never produced a usable mask.

    visualize: if True, also writes, per detected item:
      save_root/<segment_id>/viz/objects/<chunk_idx>_<instance_id>.jpg
        -- for hand/object instances: seed frame + seed mask + query points.
      save_root/<segment_id>/viz/background/<chunk_idx>_<instance_id>.jpg
        -- for background items: seed frame + (collision-resolved) mask +
        query points.
    Reads frames from resampled_path (still on disk at this point, before
    the caller's cleanup), so it doesn't need the raw video kept around any
    longer than usual.
    """
    objects = objects or []
    background_objects = background_objects or []
    if background_point_budget is None:
        background_point_budget = point_budget
    t0 = time.time()
    def elapsed():
        return time.time() - t0

    video_out_dir = os.path.join(save_root, segment_id)
    os.makedirs(video_out_dir, exist_ok=True)
    obj_viz_dir, bg_viz_dir, viz_vr = None, None, None
    if visualize:
        obj_viz_dir = os.path.join(video_out_dir, "viz", "objects")
        bg_viz_dir = os.path.join(video_out_dir, "viz", "background")
        os.makedirs(obj_viz_dir, exist_ok=True)
        os.makedirs(bg_viz_dir, exist_ok=True)

    print(f"[{elapsed():7.1f}s] Resampling video...")
    rs_meta = resample_video(video_path, resampled_path, target_fps, verbose=verbose)
    stride = rs_meta["stride"]
    n_frames_resampled = rs_meta["n_frames_resampled"]
    print(f"[{elapsed():7.1f}s] Resample done.")

    if visualize:
        decord.bridge.set_bridge("native")
        viz_vr = decord.VideoReader(resampled_path)

    prompt_specs = [("hand", "hand", "a hand.")]
    prompt_specs += [(obj, "object", f"{obj}.") for obj in objects]
    prompt_specs += [(obj, "background", f"{obj}.") for obj in background_objects]

    if check_interval is None:
        check_interval = max(1, max_frames // num_checks)

    chunk_starts = compute_chunk_starts(n_frames_resampled, 0, max_frames)
    if verbose:
        print(f"{len(chunk_starts)} chunk(s) of up to {max_frames} frames each "
              f"(check_interval={check_interval})...")
    if not chunk_starts:
        raise RuntimeError(f"No chunks to process: resampled clip has only {n_frames_resampled} frame(s).")

    print(f"[{elapsed():7.1f}s] Starting GDINO+SAM2 chunk tracking ({len(chunk_starts)} chunk(s))...")
    result = track_prompted_entities_chunks(
        resampled_path, prompt_specs, chunk_starts, max_frames,
        check_interval=check_interval, stale_frames=stale_frames,
        box_threshold=box_threshold, text_threshold=text_threshold,
        max_concurrent_per_label=max_concurrent_per_label, verbose=verbose,
    )
    print(f"[{elapsed():7.1f}s] Chunk tracking done.")
    native_h, native_w = result["frame_size"]

    bundle_paths = []
    for chunk_idx, chunk in enumerate(result["chunks"]):
        chunk_start, chunk_end, meta = chunk["chunk_start"], chunk["chunk_end"], chunk["meta"]

        # Pass 1: resolve every detected item's seed_frame/instance_id and
        # build the per-frame union of hand/object masks, so pass 2 can
        # subtract same-frame foreground from any colliding background mask.
        resolved = []
        foreground_by_frame = {}
        for obj_id, info in meta.items():
            if info["seed_mask"] is None:
                if verbose:
                    print(f"  chunk {chunk_idx}: skipping obj {obj_id} ({info['label']}): never produced a usable mask")
                continue
            seed_frame = min(f for f, v in info["visible"].items() if v)
            resolved.append({
                "obj_id": obj_id, "info": info, "seed_frame": seed_frame,
                "instance_id": f"{info['label']}_{obj_id}",
            })
            if info["category"] in ("hand", "object"):
                if seed_frame not in foreground_by_frame:
                    foreground_by_frame[seed_frame] = np.zeros((native_h, native_w), dtype=bool)
                foreground_by_frame[seed_frame] |= info["seed_mask"]

        # Pass 2: build "instances" (hand/object, mask as-is) and
        # "background" (collision-resolved against same-frame foreground).
        instances, background = [], []
        for r in resolved:
            obj_id, info, seed_frame, instance_id = r["obj_id"], r["info"], r["seed_frame"], r["instance_id"]
            category = info["category"]
            mask = info["seed_mask"]
            if category == "background":
                fg_mask = foreground_by_frame.get(seed_frame)
                if fg_mask is not None:
                    mask = mask & ~fg_mask
                if not mask.any():
                    if verbose:
                        print(f"  chunk {chunk_idx}: dropping background obj {obj_id} ({info['label']}): "
                              f"fully overlapped by a same-frame object/hand mask")
                    continue
            budget = background_point_budget if category == "background" else point_budget
            pts = farthest_point_sample(mask, budget, margin=point_margin)
            n_visible = sum(1 for v in info["visible"].values() if v)
            record = {
                "instance_id": instance_id,
                "obj_id": obj_id,
                "label": info["label"],
                "category": category,
                "first_frame": info["first_frame"],
                "last_frame": info["last_frame"],
                "seed_frame": seed_frame,
                "num_visible_frames": n_visible,
                "gdino_score": info["gdino_score"],
                "query_points_px": pts.tolist(),
            }
            if visualize:
                frame = viz_vr[seed_frame].asnumpy()
                is_bg = category == "background"
                color = (180, 180, 180) if is_bg else INSTANCE_COLORS[obj_id % len(INSTANCE_COLORS)]
                img = draw_seed_visualization(frame, mask, pts, instance_id, color)
                viz_path = os.path.join(bg_viz_dir if is_bg else obj_viz_dir, f"{chunk_idx:05d}_{instance_id}.jpg")
                cv2.imwrite(viz_path, img)
                if verbose:
                    print(f"  chunk {chunk_idx}: wrote {viz_path}")
            (background if category == "background" else instances).append(record)

        chunk_dir = os.path.join(video_out_dir, f"{chunk_idx:05d}")
        os.makedirs(chunk_dir, exist_ok=True)
        bundle = {
            "segment_id": segment_id,
            "source_video": os.path.abspath(video_path),
            "video": resampled_path,
            "narration": narration,
            "task": task,
            "objects_identified": objects,
            "background_identified": background_objects,
            "native_fps": rs_meta["native_fps"],
            "target_fps": target_fps,
            "actual_fps": rs_meta["actual_fps"],
            "stride": stride,
            "num_frames_native": rs_meta["native_n_frames"],
            "num_frames_resampled": n_frames_resampled,
            "chunk_index": chunk_idx,
            "num_chunks": len(result["chunks"]),
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "check_interval": check_interval,
            "resolution_hw": [native_h, native_w],
            "point_budget": point_budget,
            "background_point_budget": background_point_budget,
            "instances": instances,
            "background": background,
            **download_recipe,
        }
        bundle_path = os.path.join(chunk_dir, "bundle.json")
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)
        bundle_paths.append(bundle_path)
        if verbose:
            print(f"  chunk {chunk_idx:05d} [{chunk_start},{chunk_end}): {len(instances)} instance(s), "
                  f"{len(background)} background item(s) -> {bundle_path}")

    print(f"[{elapsed():7.1f}s] Done. {len(bundle_paths)} bundle(s) written.")
    return bundle_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSONL manifest from build_manifest_ego4d.py, "
                     "same file/index used by stage1a.py")
    ap.add_argument("--start", type=int, required=True, help="first manifest row index (inclusive)")
    ap.add_argument("--end", type=int, required=True, help="last manifest row index (exclusive)")
    ap.add_argument("--save-root", required=True)
    ap.add_argument("--tmp-dir", default=os.path.expanduser("~/ego4d/scratch"),
                     help="scratch directory for downloaded/resampled clips.")
    ap.add_argument("--point-budget", type=int, default=10)
    ap.add_argument("--background-point-budget", type=int, default=None,
                     help="query points per background item (identify.json's \"background\" list -- detected/"
                          "segmented the same way as --point-budget's \"objects\", written to the chunk's "
                          "bundle.json under \"background\"). Defaults to --point-budget's value if unset.")
    ap.add_argument("--max-frames", type=int, default=35,
                     help="cap detection to the first N frames of each window, matching stage 2's "
                          "SpatialTrackerV2 window ceiling.")
    ap.add_argument("--num-checks", type=int, default=3,
                     help="number of periodic re-detection passes within the max-frames window; "
                          "check_interval is derived as max_frames // num_checks unless --check-interval "
                          "is given explicitly.")
    ap.add_argument("--check-interval", type=int, default=None,
                     help="explicit frame interval between re-detection passes, overriding the "
                          "--num-checks-derived default.")
    ap.add_argument("--target-fps", type=float, default=10,
                     help="resample to this fps before chunking. Must match stage1a.py's --duration-sec "
                          "sizing assumption if you change it.")
    ap.add_argument("--stale-frames", type=int, default=10)
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--max-concurrent-per-label", type=int, default=4)
    ap.add_argument("--point-margin", type=int, default=3,
                     help="erode each instance's seed mask inward by this many pixels before sampling query "
                          "points, so points land away from the mask boundary. 0 disables this.")
    ap.add_argument("--check-existing", action="store_true", default=False,
                     help="skip a row if its chunk-0 bundle.json already exists (resume mode). Default is to "
                          "overwrite/re-run every row in range.")
    ap.add_argument("--visualize", action="store_true", default=False,
                     help="for each tracked instance/background item, save a JPG of its seed frame with its "
                          "mask and query points overlaid, under save-root/<segment_id>/viz/objects/ and "
                          ".../viz/background/ respectively. Off by default since it adds per-item disk "
                          "writes; the raw/resampled video itself is always deleted after each row regardless "
                          "of this flag.")
    ap.add_argument("--quiet", dest="verbose", action="store_false", default=True)
    args = ap.parse_args()

    rows = load_manifest_range(args.manifest, args.start, args.end)
    print(f"Loaded {len(rows)} manifest row(s) in [{args.start}, {args.end})")
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    load_gdino()
    load_sam2_predictor()
    for row in rows:
        seg_id = row["segment_id"]
        out_dir = os.path.join(args.save_root, seg_id)
        identify_path = os.path.join(out_dir, "identify.json")
        bundle0_path = os.path.join(out_dir, "00000", "bundle.json")
        if args.check_existing and os.path.exists(bundle0_path):
            print(f"[{seg_id}] bundle.json exists, skipping")
            continue
        if not os.path.exists(identify_path):
            print(f"[{seg_id}] SKIP: no identify.json (run stage1a.py first)")
            continue
        with open(identify_path) as f:
            ident = json.load(f)
        seg_path = os.path.join(args.tmp_dir, f"{seg_id}_raw.mp4")
        resampled_path = os.path.join(args.tmp_dir, f"{seg_id}_resampled.mp4")
        try:
            download_segment(ident["bucket"], ident["key"], ident["start_sec"], ident["duration_sec"], seg_path)
            run_track(
                seg_path, args.save_root, seg_id, ident["narration_text"], ident["task"], ident["objects"],
                ident.get("background", []), resampled_path,
                download_recipe={"bucket": ident["bucket"], "key": ident["key"],
                                  "start_sec": ident["start_sec"], "duration_sec": ident["duration_sec"]},
                point_budget=args.point_budget, check_interval=args.check_interval,
                num_checks=args.num_checks, max_frames=args.max_frames, target_fps=args.target_fps,
                stale_frames=args.stale_frames, box_threshold=args.box_threshold,
                text_threshold=args.text_threshold, max_concurrent_per_label=args.max_concurrent_per_label,
                point_margin=args.point_margin, background_point_budget=args.background_point_budget,
                visualize=args.visualize, verbose=args.verbose)
        except Exception as e:
            print(f"[{seg_id}] FAILED: {e}")
        finally:
            cleanup(seg_path, resampled_path)

    print("Stage 1b done.")


if __name__ == "__main__":
    main()
