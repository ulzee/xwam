"""Stage 1 (runs in the `qwen_sam2` conda env): narration + keyframes -> object
list (Qwen3-VL) -> hands-always + every named object tracked via the
generalized periodic Grounded-SAM-2 detector (entity_tracker.py) -> a
farthest-point-sampled query-point budget per instance, saved as an
intermediate JSON bundle for stage2_lift3d.py to consume.

Each instance's query points are sampled at its own first-visible frame, NOT
universally at frame 0 -- an object first spotted mid-clip gets its points
from the frame it actually first appears at.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse
import gc
import json
import decord
import torch

import scene_understanding
from scene_understanding import identify_objects, DEFAULT_NARRATION
from entity_tracker import track_prompted_entities, track_prompted_entities_chunks
from fps_sample import farthest_point_sample
from frame_sampling import resample_video, compute_chunk_starts


def free_qwen():
    """Qwen3-VL-8B holds ~17.5GB resident; on a 23GB card there isn't room
    left for GroundingDINO+SAM2 in the same process unless it's freed first
    (hit a real OOM without this)."""
    scene_understanding._model = None
    scene_understanding._processor = None
    gc.collect()
    torch.cuda.empty_cache()


def run(video_path, narration=DEFAULT_NARRATION, point_budget=10,
        check_interval=None, num_checks=3, max_frames=35, stale_frames=10,
        box_threshold=0.3, text_threshold=0.25,
        max_concurrent_per_label=4, point_margin=3, verbose=True):
    """max_frames: cap detection/tracking to the first N frames of the clip,
    matching stage 2's SpatialTrackerV2 window ceiling -- an entity first
    seen beyond this point can never be 3D-lifted anyway (stage 2 truncates
    the video tensor to the same ceiling), so there's no point spending
    GroundingDINO/SAM2 budget discovering it. Keep this in sync with
    stage2_lift3d.py's --max-frames (run_pipeline.sh threads one value to
    both by default).

    check_interval: frame interval between re-detection passes. If None
    (default), derived as max(1, min(max_frames, n_frames) // num_checks) --
    i.e. "check num_checks times within the detection window" -- rather than
    a fixed absolute interval that doesn't scale with the window size. Pass
    an explicit value to override.
    """
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    n_frames = len(vr)
    frame0 = vr[0].asnumpy()
    frame_h, frame_w = frame0.shape[0], frame0.shape[1]
    video_fps = float(vr.get_avg_fps())

    if verbose:
        print(f"Video: {video_path} ({n_frames} frames, {frame_w}x{frame_h}, {video_fps:.2f}fps)")
        print(f"Narration: {narration!r}")
        print("Running Qwen3-VL scene understanding...")
    task, objects = identify_objects(video_path, narration=narration)
    if verbose:
        print(f"  task: {task}")
        print(f"  objects: {objects}")

    free_qwen()

    prompt_specs = [("hand", "hand", "a hand.")]
    prompt_specs += [(obj, "object", f"{obj}.") for obj in objects]

    detect_frame_limit = min(max_frames, n_frames)
    if check_interval is None:
        check_interval = max(1, detect_frame_limit // num_checks)

    if verbose:
        print(f"Tracking {len(prompt_specs)} label(s) with periodic Grounded-SAM-2 detection, "
              f"limited to the first {detect_frame_limit} frames (stage 2's max-frames={max_frames}), "
              f"check_interval={check_interval} ({num_checks} checks within that window)...")
    result = track_prompted_entities(
        video_path, prompt_specs,
        check_interval=check_interval, stale_frames=stale_frames,
        box_threshold=box_threshold, text_threshold=text_threshold,
        max_concurrent_per_label=max_concurrent_per_label,
        detect_frame_limit=detect_frame_limit, verbose=verbose,
    )
    meta = result["meta"]

    instances = []
    for obj_id, info in meta.items():
        if info["seed_mask"] is None:
            if verbose:
                print(f"  skipping obj {obj_id} ({info['label']}): never produced a usable mask")
            continue
        seed_frame = min(f for f, v in info["visible"].items() if v)
        pts = farthest_point_sample(info["seed_mask"], point_budget, margin=point_margin)
        n_visible = sum(1 for v in info["visible"].values() if v)
        instances.append({
            "instance_id": f"{info['label']}_{obj_id}",
            "obj_id": obj_id,
            "label": info["label"],
            "category": info["category"],
            "first_frame": info["first_frame"],
            "last_frame": info["last_frame"],
            "seed_frame": seed_frame,
            "num_visible_frames": n_visible,
            "gdino_score": info["gdino_score"],
            "query_points_px": pts.tolist(),
        })
        if verbose:
            print(f"  instance {info['label']}_{obj_id}: frames [{info['first_frame']},{info['last_frame']}], "
                  f"{n_visible} visible, seed_frame={seed_frame}, {len(pts)} query points")

    bundle = {
        "video": os.path.abspath(video_path),
        "narration": narration,
        "task": task,
        "objects_identified": objects,
        "fps": video_fps,
        "num_frames": n_frames,
        "detect_frame_limit": detect_frame_limit,
        "check_interval": check_interval,
        "resolution_hw": [frame_h, frame_w],
        "point_budget": point_budget,
        "instances": instances,
    }
    return bundle


def run_chunked(video_path, save_root, narration=DEFAULT_NARRATION, point_budget=10,
                 check_interval=None, num_checks=3, max_frames=35, target_fps=10, start_frame=0,
                 stale_frames=10, box_threshold=0.3, text_threshold=0.25,
                 max_concurrent_per_label=4, point_margin=3, verbose=True):
    """Bulk-annotation mode: instead of one 35-frame window starting at frame
    0, resample the whole clip to target_fps ONCE (stretching the same
    max_frames budget over a longer wall-clock span -- see
    frame_sampling.py's docstring for why resampling happens up front rather
    than threading a stride through the tracking loop), then tile it into
    consecutive max_frames windows ("chunks") from start_frame to the end of
    the clip. GDINO/SAM2 load once and are reused across every chunk
    (track_prompted_entities_chunks), matching stage2_lift3d.py's
    run_chunked's model-reuse. Qwen3-VL scene understanding also runs only
    ONCE, on the ORIGINAL (not resampled) video -- it already samples the
    whole clip internally (see scene_understanding.identify, fps=2.0), so one
    pass already sees everything; re-running it per chunk would multiply VLM
    calls for no benefit and risks the same object getting a different name
    in different chunks.

    start_frame is in ORIGINAL/native frame units (matches how you'd scrub
    the source video) and gets converted to the resampled video's frame
    units internally.

    Writes one bundle.json per chunk to
    save_root/<video_stem>/<chunk_idx:%05d>/bundle.json. A chunk with zero
    detected instances still gets a bundle.json (empty "instances": []) so
    stage2_lift3d.py's chunked mode can enumerate and skip it explicitly
    rather than silently missing a chunk index.
    """
    clip_stem = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(save_root, clip_stem)
    os.makedirs(video_out_dir, exist_ok=True)

    resampled_path = os.path.join(video_out_dir, f"_resampled_{target_fps}fps.mp4")
    rs_meta = resample_video(video_path, resampled_path, target_fps, verbose=verbose)
    stride = rs_meta["stride"]
    n_frames_resampled = rs_meta["n_frames_resampled"]

    if verbose:
        print(f"Narration: {narration!r}")
        print("Running Qwen3-VL scene understanding (once, on the original video)...")
    task, objects = identify_objects(video_path, narration=narration)
    if verbose:
        print(f"  task: {task}")
        print(f"  objects: {objects}")

    free_qwen()

    prompt_specs = [("hand", "hand", "a hand.")]
    prompt_specs += [(obj, "object", f"{obj}.") for obj in objects]

    if check_interval is None:
        check_interval = max(1, max_frames // num_checks)

    start_frame_resampled = round(start_frame / stride)
    chunk_starts = compute_chunk_starts(n_frames_resampled, start_frame_resampled, max_frames)
    if verbose:
        print(f"{len(chunk_starts)} chunk(s) of up to {max_frames} frames each "
              f"(check_interval={check_interval}), starting at resampled frame "
              f"{start_frame_resampled} (native frame {start_frame})...")
    if not chunk_starts:
        raise RuntimeError(
            f"No chunks to process: start_frame={start_frame} (resampled={start_frame_resampled}) "
            f"is at or past the end of the resampled clip ({n_frames_resampled} frames).")

    result = track_prompted_entities_chunks(
        resampled_path, prompt_specs, chunk_starts, max_frames,
        check_interval=check_interval, stale_frames=stale_frames,
        box_threshold=box_threshold, text_threshold=text_threshold,
        max_concurrent_per_label=max_concurrent_per_label, verbose=verbose,
    )
    native_h, native_w = result["frame_size"]

    bundle_paths = []
    for chunk_idx, chunk in enumerate(result["chunks"]):
        chunk_start, chunk_end, meta = chunk["chunk_start"], chunk["chunk_end"], chunk["meta"]
        instances = []
        for obj_id, info in meta.items():
            if info["seed_mask"] is None:
                if verbose:
                    print(f"  chunk {chunk_idx}: skipping obj {obj_id} ({info['label']}): never produced a usable mask")
                continue
            seed_frame = min(f for f, v in info["visible"].items() if v)
            pts = farthest_point_sample(info["seed_mask"], point_budget, margin=point_margin)
            n_visible = sum(1 for v in info["visible"].values() if v)
            instances.append({
                "instance_id": f"{info['label']}_{obj_id}",
                "obj_id": obj_id,
                "label": info["label"],
                "category": info["category"],
                "first_frame": info["first_frame"],
                "last_frame": info["last_frame"],
                "seed_frame": seed_frame,
                "num_visible_frames": n_visible,
                "gdino_score": info["gdino_score"],
                "query_points_px": pts.tolist(),
            })

        chunk_dir = os.path.join(video_out_dir, f"{chunk_idx:05d}")
        os.makedirs(chunk_dir, exist_ok=True)
        bundle = {
            "source_video": os.path.abspath(video_path),
            "video": resampled_path,
            "narration": narration,
            "task": task,
            "objects_identified": objects,
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
            "instances": instances,
        }
        bundle_path = os.path.join(chunk_dir, "bundle.json")
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)
        bundle_paths.append(bundle_path)
        if verbose:
            print(f"  chunk {chunk_idx:05d} [{chunk_start},{chunk_end}): {len(instances)} instance(s) -> {bundle_path}")

    return bundle_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--narration", default=DEFAULT_NARRATION)
    ap.add_argument("--point-budget", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=35,
                     help="cap detection to the first N frames of the clip, matching stage 2's "
                          "SpatialTrackerV2 window ceiling -- keep in sync with stage2_lift3d.py's "
                          "--max-frames (run_pipeline.sh does this automatically).")
    ap.add_argument("--num-checks", type=int, default=3,
                     help="number of periodic re-detection passes within the max-frames window; "
                          "check_interval is derived as max_frames // num_checks unless --check-interval "
                          "is given explicitly.")
    ap.add_argument("--check-interval", type=int, default=None,
                     help="explicit frame interval between re-detection passes, overriding the "
                          "--num-checks-derived default.")
    ap.add_argument("--stale-frames", type=int, default=10)
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--max-concurrent-per-label", type=int, default=4)
    ap.add_argument("--point-margin", type=int, default=3,
                     help="erode each instance's seed mask inward by this many pixels before sampling query "
                          "points, so points land away from the mask boundary (the noisiest part of a SAM2 "
                          "mask frame-to-frame -- an edge point is the most likely to flicker foreground/"
                          "background as the mask wobbles slightly between frames). 0 disables this.")
    ap.add_argument("--out", help="path to write the intermediate bundle JSON (single-clip mode)")
    ap.add_argument("--save-root", help="enables chunked bulk-annotation mode: resamples the clip to "
                     "--target-fps once, tiles it into consecutive --max-frames windows, and writes one "
                     "bundle.json per chunk to <save-root>/<video_stem>/<chunk_idx:05d>/bundle.json. "
                     "Mutually exclusive with --out.")
    ap.add_argument("--target-fps", type=float, default=10,
                     help="chunked mode only: resample the clip to this fps before chunking, so the same "
                          "--max-frames window spans more wall-clock time on faster-native-fps videos. "
                          "Actual fps may differ slightly (native_fps / round(native_fps/target_fps)).")
    ap.add_argument("--start-frame", type=int, default=0,
                     help="chunked mode only: first chunk starts here, in ORIGINAL/native video frame "
                          "units (converted internally to the resampled video's frame units).")
    args = ap.parse_args()

    if bool(args.save_root) == bool(args.out):
        raise SystemExit("Pass exactly one of --out (single-clip mode) or --save-root (chunked mode).")

    if args.save_root:
        run_chunked(args.video, args.save_root, narration=args.narration, point_budget=args.point_budget,
                    check_interval=args.check_interval, num_checks=args.num_checks, max_frames=args.max_frames,
                    target_fps=args.target_fps, start_frame=args.start_frame,
                    stale_frames=args.stale_frames,
                    box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                    max_concurrent_per_label=args.max_concurrent_per_label, point_margin=args.point_margin)
        return

    bundle = run(args.video, narration=args.narration, point_budget=args.point_budget,
                 check_interval=args.check_interval, num_checks=args.num_checks, max_frames=args.max_frames,
                 stale_frames=args.stale_frames,
                 box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                 max_concurrent_per_label=args.max_concurrent_per_label, point_margin=args.point_margin)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Saved bundle: {args.out} ({len(bundle['instances'])} instances)")


if __name__ == "__main__":
    main()
