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
from entity_tracker import track_prompted_entities
from fps_sample import farthest_point_sample


def free_qwen():
    """Qwen3-VL-8B holds ~17.5GB resident; on a 23GB card there isn't room
    left for GroundingDINO+SAM2 in the same process unless it's freed first
    (hit a real OOM without this)."""
    scene_understanding._model = None
    scene_understanding._processor = None
    gc.collect()
    torch.cuda.empty_cache()


def run(video_path, narration=DEFAULT_NARRATION, point_budget=10,
        check_interval=15, stale_frames=10, box_threshold=0.3, text_threshold=0.25,
        max_concurrent_per_label=4, point_margin=3, verbose=True):
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

    if verbose:
        print(f"Tracking {len(prompt_specs)} label(s) with periodic Grounded-SAM-2 detection "
              f"(check_interval={check_interval})...")
    result = track_prompted_entities(
        video_path, prompt_specs,
        check_interval=check_interval, stale_frames=stale_frames,
        box_threshold=box_threshold, text_threshold=text_threshold,
        max_concurrent_per_label=max_concurrent_per_label, verbose=verbose,
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
        "resolution_hw": [frame_h, frame_w],
        "point_budget": point_budget,
        "instances": instances,
    }
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--narration", default=DEFAULT_NARRATION)
    ap.add_argument("--point-budget", type=int, default=10)
    ap.add_argument("--check-interval", type=int, default=15)
    ap.add_argument("--stale-frames", type=int, default=10)
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--max-concurrent-per-label", type=int, default=4)
    ap.add_argument("--point-margin", type=int, default=3,
                     help="erode each instance's seed mask inward by this many pixels before sampling query "
                          "points, so points land away from the mask boundary (the noisiest part of a SAM2 "
                          "mask frame-to-frame -- an edge point is the most likely to flicker foreground/"
                          "background as the mask wobbles slightly between frames). 0 disables this.")
    ap.add_argument("--out", required=True, help="path to write the intermediate bundle JSON")
    args = ap.parse_args()

    bundle = run(args.video, narration=args.narration, point_budget=args.point_budget,
                 check_interval=args.check_interval, stale_frames=args.stale_frames,
                 box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                 max_concurrent_per_label=args.max_concurrent_per_label, point_margin=args.point_margin)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Saved bundle: {args.out} ({len(bundle['instances'])} instances)")


if __name__ == "__main__":
    main()
