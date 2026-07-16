"""Stage 2 (runs in the `trace_stv2` conda env): loads stage1_detect.py's
bundle JSON and lifts every instance's query points to 3D with
SpatialTrackerV2.

Lives in this pipeline repo, but `from models.SpaTrackV2...` below needs the
SpaTrackerV2 checkout (~/trace/SpaTrackerV2, not committed here -- it's a
clone of github.com/henry123-boy/SpaTrackerV2) on sys.path. run_pipeline.sh
handles this by cd-ing there and setting PYTHONPATH before invoking this
script by its path in the pipeline repo -- run standalone, do the same:
  cd ~/trace/SpaTrackerV2 && PYTHONPATH="$PWD" python /path/to/stage2_lift3d.py ...

Two separate backend tracker calls, mirroring patched_inference.py's existing
--query_point mechanism (generalized from 1 point to every instance's full
point budget at once):
  1. A background/VO grid call (full_point=False, get_points_on_a_grid) --
     purely for camera-pose (bundle-adjustment) stability. Entity points are
     deliberately NOT mixed into this call: full_point=False silently drops
     and replaces any query point whose local confidence is below 0.5, which
     happens easily for a point on a hand/object edge -- exactly the points
     we care about most (see SpaTrack.py:616-625).
  2. One combined entity call (full_point=True) carrying every instance's
     FPS-sampled query points at once, each with its OWN start frame ("t" in
     the (t,x,y) query format) -- an instance first seen mid-clip is queried
     from its own first-visible frame, not frame 0. NOTE: mixing different
     start frames within one batched query is assumed to work because the
     query format is structurally per-point (t,x,y), matching the standard
     TAP-style interface -- it has only been exercised with a single point at
     t=0 in this codebase before now, so treat this as a real assumption to
     verify against actual output, not a confirmed fact.

For entity points we do NOT trust the entity call's own inferred 3D position
(it has far less bundle-adjustment support than the grid). Instead we reuse
render_novel_view.py's own pattern: take the entity call's accurately-tracked
2D pixel per frame, then unproject it through the DENSE depth map (from the
shared front-end pass) and the GRID call's own bundle-adjusted camera pose --
the grid call is the one with real multi-point BA support, so its camera
trajectory is the trustworthy one.

query_no_BA is hardcoded to False for both calls (validated: ~22% less
jitter on tracked points at equal compute, see project_spatialtrackerv2_jitter
memory -- there is no scenario where the default True is preferable).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse
import gc
import glob
import json
import time

import numpy as np
import torch
import decord

from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.utils import get_points_on_a_grid
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def remap_point_to_preprocessed(x, y, W, H, target_size):
    """Mirrors preprocess_image(mode="crop") exactly: resize preserving aspect
    ratio to width=target_size, then center-crop vertically only if the
    reslarge height exceeds target_size (landscape video: usually no crop)."""
    scale = target_size / W
    new_h = round(H * scale / 14) * 14
    x2 = x * scale
    if new_h > target_size:
        start_y = (new_h - target_size) // 2
        y2 = y * scale - start_y
    else:
        y2 = y * scale
    return x2, y2


def unproject_tracked_pixel(depth, K, c2w, u, v, device, search=4):
    """Copied from render_novel_view.py: look up depth near pixel (u,v) with
    a small-neighborhood median fallback for masked-out/invalid depth
    (common right at a hand/object silhouette edge), then unproject to a
    world-space 3D point. Returns None if no valid depth is found nearby."""
    H, W = depth.shape
    iu, iv = int(round(u)), int(round(v))
    if not (0 <= iu < W and 0 <= iv < H):
        return None
    best = None
    for r in range(search + 1):
        lo_u, hi_u = max(0, iu - r), min(W, iu + r + 1)
        lo_v, hi_v = max(0, iv - r), min(H, iv + r + 1)
        patch = depth[lo_v:hi_v, lo_u:hi_u]
        valid = patch > 0
        if valid.any():
            best = patch[valid].median().item()
            break
    if best is None:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = best
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pt_cam = torch.tensor([x, y, z], dtype=torch.float32, device=device)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return (R @ pt_cam + t).cpu().numpy()


def lift_window(vr, chunk_start, chunk_end, native_h, native_w, instances, out_dir, clip_stem,
                 vggt4track_model, model, target_size, grid_size, vo_points, save_scene,
                 summary_extra=None):
    """Runs the front-end pass + the two backend calls + unprojection + save
    for ONE window [chunk_start, chunk_end) of frames read from the
    already-open `vr`, reusing the ALREADY-LOADED vggt4track_model/model so
    repeated calls (chunked mode) don't reload weights from HF each time --
    only this window's GPU tensors are freed at the end, not the models.

    `instances` must already have LOCAL (window-relative, 0-indexed)
    seed_frame/first_frame/last_frame values -- the caller is responsible for
    that conversion when instances came from a chunked bundle whose frame
    numbers are global into the shared resampled video (see
    stage1_detect.py's run_chunked).

    Returns the summary dict, or None if no instances fell inside the window
    (nothing to lift -- caller should skip/log, not treat as an error, since
    this is an expected outcome for some chunks in bulk mode).
    """
    max_frames = chunk_end - chunk_start
    kept, skipped = [], []
    for inst in instances:
        if inst["seed_frame"] < max_frames:
            kept.append(inst)
        else:
            skipped.append(inst)
    if skipped:
        log(f"Skipping {len(skipped)} instance(s) whose first appearance is beyond "
            f"this window ({max_frames} frames): {[i['instance_id'] for i in skipped]}")
    if not kept:
        log("No instances fall within this window -- nothing to lift.")
        return None

    video_tensor = torch.from_numpy(vr.get_batch(range(chunk_start, chunk_end)).asnumpy()).permute(0, 3, 1, 2).float()
    n_frames = video_tensor.shape[0]
    log(f"Loaded frames [{chunk_start},{chunk_end}) ({n_frames}f) at native {native_w}x{native_h}")

    video_tensor = preprocess_image(video_tensor, mode="crop", target_size=target_size)[None]
    log(f"Front-end input resolution: {video_tensor.shape[-2]}x{video_tensor.shape[-1]}")

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = vggt4track_model(video_tensor.cuda() / 255)
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
    log(f"Front-end done. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    intrs_frontend = intrs.copy()
    video_tensor = video_tensor.squeeze()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
    frame_h_pp, frame_w_pp = video_tensor.shape[2], video_tensor.shape[3]

    del predictions, extrinsic, intrinsic, depth_map, depth_conf
    gc.collect()
    torch.cuda.empty_cache()

    # --- Call 1: background/VO grid, for camera pose stability ---
    grid_pts = get_points_on_a_grid(grid_size, (frame_h_pp, frame_w_pp), device="cpu")
    query_xyt_grid = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    log(f"Tracking {query_xyt_grid.shape[0]} background/VO query points (grid_size={grid_size})...")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (c2w_traj, intrs_refined, point_map, conf_depth,
         _track3d, _track2d, _vis, _conf, video_out) = model.forward(
            video_tensor, depth=depth_tensor, intrs=intrs, extrs=extrs,
            queries=query_xyt_grid, fps=1, full_point=False, iters_track=4,
            query_no_BA=False, fixed_cam=False, stage=1, unc_metric=unc_metric,
            support_frame=n_frames - 1, replace_ratio=0.2)
    log(f"Grid call done. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

    depth_save = point_map[:, 2, ...].clone()
    depth_save[conf_depth < 0.5] = 0
    depth_save = depth_save.cpu()
    c2ws = c2w_traj.cpu()
    intrs_np = intrs_refined.cpu().numpy()
    video_np = (video_out.squeeze(0).cpu().numpy() / 255) if video_out.dim() == 5 else (video_out.cpu().numpy() / 255)

    # free the grid call's GPU tensors before the second (entity) backend
    # pass -- this process only has ~23GB total and needs both calls' peak
    # to fit sequentially, not just one
    del c2w_traj, intrs_refined, point_map, conf_depth, _track3d, _track2d, _vis, _conf, video_out
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Freed grid call tensors. VRAM in use: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    # --- Build the combined entity query set: every instance's own points, each at its own seed frame ---
    query_rows = []  # (t, x, y) in preprocessed-resolution pixel coords
    row_owner = []   # (instance_index, point_index)
    for i, inst in enumerate(kept):
        t = inst["seed_frame"]
        for j, (px, py) in enumerate(inst["query_points_px"]):
            x2, y2 = remap_point_to_preprocessed(px, py, native_w, native_h, target_size)
            query_rows.append([float(t), x2, y2])
            row_owner.append((i, j))

    query_xyt_entities = np.array(query_rows, dtype=np.float32)
    log(f"Tracking {len(query_rows)} entity query points across {len(kept)} instance(s) "
        f"(full_point=True, mixed start frames: {sorted(set(inst['seed_frame'] for inst in kept))})...")

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        custom_out = model.forward(
            video_tensor, depth=depth_tensor, intrs=intrs_frontend, extrs=extrs,
            queries=query_xyt_entities, fps=1, full_point=True, iters_track=4,
            query_no_BA=False, fixed_cam=False, stage=1, unc_metric=unc_metric,
            support_frame=n_frames - 1, replace_ratio=0.2)
    log(f"Entity call done. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    track2d_raw = custom_out[5]
    vis_raw = custom_out[6]
    if track2d_raw.dim() == 4:
        # (T, 1, N, C) -- batch/window dim of size 1 sits before the point dim
        entity_track2d = track2d_raw[:, 0, :, :2].cpu()
        entity_vis = vis_raw[:, 0, :, 0].cpu().numpy() if vis_raw.dim() == 4 else vis_raw[:, 0, :].cpu().numpy()
    else:
        # (T, N, C) -- no extra batch dim
        entity_track2d = track2d_raw[:, :, :2].cpu()
        entity_vis = vis_raw[:, :, 0].cpu().numpy() if vis_raw.dim() == 3 else vis_raw.cpu().numpy()
    assert entity_track2d.shape[1] == query_xyt_entities.shape[0], \
        f"point dim mismatch after reshape: got {entity_track2d.shape[1]}, expected {query_xyt_entities.shape[0]}"

    del custom_out
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda"
    os.makedirs(out_dir, exist_ok=True)

    # --- Unproject every entity point at every frame via dense depth + the grid call's camera poses ---
    n_pts_total = query_xyt_entities.shape[0]
    world = np.full((n_frames, n_pts_total, 3), np.nan, dtype=np.float32)
    pix = entity_track2d.numpy()  # (T, N, 2)
    vis_conf = entity_vis if entity_vis.ndim == 2 else entity_vis[:, None].repeat(n_pts_total, axis=1)

    depth_save_dev = depth_save.to(device)
    c2ws_dev = c2ws.to(device)
    for t in range(n_frames):
        K_t = torch.from_numpy(intrs_np[t]).to(device)
        c2w_t = c2ws_dev[t]
        d_t = depth_save_dev[t]
        for n in range(n_pts_total):
            u, v = pix[t, n]
            wpt = unproject_tracked_pixel(d_t, K_t, c2w_t, float(u), float(v), device)
            if wpt is not None:
                world[t, n] = wpt

    # --- Split back out per instance, save summary + per-instance npz ---
    instances_summary = []
    rows_by_instance = {}  # instance index -> list of row indices into query_xyt_entities/pix/vis_conf
    for row_i, (i, j) in enumerate(row_owner):
        rows_by_instance.setdefault(i, []).append(row_i)

    for i, inst in enumerate(kept):
        rows = rows_by_instance[i]
        t0 = inst["seed_frame"]
        pts3d = world[t0:, rows, :]                       # (T-t0, N_pts, 3)
        pts2d = pix[t0:, rows, :]                          # (T-t0, N_pts, 2)
        point_vis = vis_conf[t0:, rows] > 0.5              # (T-t0, N_pts) bool
        frame_indices = np.arange(t0, n_frames)
        has_pos = ~np.isnan(pts3d).any(axis=2)
        visibility = point_vis & has_pos

        traj_file = f"{clip_stem}_trace_{inst['instance_id']}.npz"
        np.savez(os.path.join(out_dir, traj_file),
                 points3d=pts3d, points2d=pts2d, visibility=visibility,
                 frame_indices=frame_indices, seed_frame=t0)

        instances_summary.append({
            "instance_id": inst["instance_id"], "label": inst["label"], "category": inst["category"],
            "first_frame": inst["first_frame"], "last_frame_sam2": inst["last_frame"],
            "seed_frame": t0, "num_points": len(rows),
            "num_frames_tracked": int(n_frames - t0),
            "num_frames_visible": int(visibility.any(axis=1).sum()),
            "gdino_score": inst["gdino_score"],
            "trajectory_file": traj_file,
        })
        log(f"  {inst['instance_id']}: {len(rows)} pts, tracked {n_frames - t0} frames, "
            f"{int(visibility.any(axis=1).sum())} with >=1 visible point -> {traj_file}")

    if save_scene:
        scene_file = f"{clip_stem}_scene.npz"
        np.savez(os.path.join(out_dir, scene_file),
                 video=video_np, depths=depth_save.numpy(), intrinsics=intrs_np,
                 extrinsics=torch.inverse(c2ws).numpy())
        log(f"Saved {scene_file}")
    else:
        scene_file = None
        log("Skipping scene.npz (pass --save-scene to keep it, e.g. for stage3_render.py)")

    summary = {
        "clip_stem": clip_stem,
        "num_frames_used": n_frames,
        "resolution_hw_native": [native_h, native_w],
        "resolution_hw_preprocessed": [frame_h_pp, frame_w_pp],
        "target_size": target_size, "grid_size": grid_size, "vo_points": vo_points,
        "scene_file": scene_file,
        "instances": instances_summary,
        "skipped_instances": [i["instance_id"] for i in skipped],
    }
    if summary_extra:
        summary.update(summary_extra)
    summary_path = os.path.join(out_dir, f"{clip_stem}_trace_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved {summary_path}")

    del depth_save, c2ws, depth_save_dev, c2ws_dev, video_tensor, entity_track2d, video_np
    gc.collect()
    torch.cuda.empty_cache()

    return summary


def load_models(track_mode, vo_points):
    log("Loading VGGT4Track (front-end)...")
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval().to("cuda")

    log(f"Loading Predictor (backend tracker, track_mode={track_mode})...")
    if track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model.spatrack.track_num = vo_points
    model.eval().to("cuda")
    return vggt4track_model, model


def run_single(args):
    with open(args.bundle) as f:
        bundle = json.load(f)

    video_path = bundle["video"]
    native_h, native_w = bundle["resolution_hw"]
    clip_stem = os.path.splitext(os.path.basename(video_path))[0]

    vggt4track_model, model = load_models(args.track_mode, args.vo_points)

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    chunk_end = min(len(vr), args.max_frames)

    lift_window(vr, 0, chunk_end, native_h, native_w, bundle["instances"], args.out_dir, clip_stem,
                vggt4track_model, model, args.target_size, args.grid_size, args.vo_points, args.save_scene,
                summary_extra={
                    "video": video_path, "narration": bundle.get("narration"), "task": bundle.get("task"),
                    "fps": bundle.get("fps"), "num_frames_total": bundle.get("num_frames"),
                    "max_frames": args.max_frames,
                })
    log("Done.")


def run_chunked(args):
    """Bulk-annotation mode: processes every chunk stage1_detect.py's
    run_chunked wrote under save_root/<video_stem>/<chunk_idx:05d>/bundle.json,
    reusing one loaded VGGT4Track + Predictor pair and one open decord reader
    (all chunks of a video share the same resampled video file) across all
    chunks instead of reloading per chunk.
    """
    clip_stem = os.path.splitext(os.path.basename(args.video))[0]
    video_dir = os.path.join(args.save_root, clip_stem)
    chunk_dirs = sorted(glob.glob(os.path.join(video_dir, "[0-9]" * 5)))
    if not chunk_dirs:
        raise RuntimeError(f"No chunk directories found under {video_dir} -- run stage1_detect.py's "
                            f"--save-root mode on this video first.")
    if args.chunk_index is not None:
        wanted = os.path.join(video_dir, f"{args.chunk_index:05d}")
        chunk_dirs = [d for d in chunk_dirs if d == wanted]
        if not chunk_dirs:
            raise RuntimeError(f"--chunk-index {args.chunk_index} not found under {video_dir}")
    log(f"Found {len(chunk_dirs)} chunk(s) under {video_dir}")

    vggt4track_model, model = load_models(args.track_mode, args.vo_points)

    vr = None
    n_processed, n_skipped_empty = 0, 0
    for chunk_dir in chunk_dirs:
        bundle_path = os.path.join(chunk_dir, "bundle.json")
        with open(bundle_path) as f:
            bundle = json.load(f)

        if vr is None:
            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(bundle["video"])

        chunk_start, chunk_end = bundle["chunk_start"], bundle["chunk_end"]
        native_h, native_w = bundle["resolution_hw"]
        log(f"=== chunk {bundle['chunk_index']:05d}/{bundle['num_chunks'] - 1} "
            f"[{chunk_start},{chunk_end}) ===")

        # bundle.json stores GLOBAL frame numbers into the shared resampled
        # video (so they stay meaningful/debuggable on their own); lift_window
        # needs them LOCAL to this window, 0-indexed from chunk_start.
        local_instances = []
        for inst in bundle["instances"]:
            local = dict(inst)
            local["seed_frame"] = inst["seed_frame"] - chunk_start
            local["first_frame"] = inst["first_frame"] - chunk_start
            local["last_frame"] = inst["last_frame"] - chunk_start
            local_instances.append(local)

        summary = lift_window(
            vr, chunk_start, chunk_end, native_h, native_w, local_instances, chunk_dir, clip_stem,
            vggt4track_model, model, args.target_size, args.grid_size, args.vo_points, args.save_scene,
            summary_extra={
                "video": bundle["video"], "source_video": bundle.get("source_video"),
                "narration": bundle.get("narration"), "task": bundle.get("task"),
                "native_fps": bundle.get("native_fps"), "target_fps": bundle.get("target_fps"),
                "actual_fps": bundle.get("actual_fps"), "stride": bundle.get("stride"),
                "chunk_index": bundle["chunk_index"], "num_chunks": bundle["num_chunks"],
                "chunk_start": chunk_start, "chunk_end": chunk_end,
                "max_frames": args.max_frames,
            })
        if summary is None:
            n_skipped_empty += 1
        else:
            n_processed += 1

    log(f"Done. {n_processed} chunk(s) lifted, {n_skipped_empty} skipped (no instances in window).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", help="stage1_detect.py output JSON (single-clip mode)")
    ap.add_argument("--out-dir", help="single-clip mode only")
    ap.add_argument("--save-root", help="chunked bulk-annotation mode: processes every chunk under "
                     "<save-root>/<video_stem>/<chunk_idx:05d>/bundle.json (as written by "
                     "stage1_detect.py's --save-root mode). Requires --video to derive video_stem. "
                     "Mutually exclusive with --bundle/--out-dir.")
    ap.add_argument("--video", help="chunked mode only: original source video, used only to derive "
                     "video_stem (matches stage1_detect.py's --video for the same run).")
    ap.add_argument("--chunk-index", type=int, default=None,
                     help="chunked mode only: process just this one chunk index instead of every "
                          "chunk under video_stem/ (e.g. to re-run a single chunk with --save-scene "
                          "for a spot-check without redoing/re-saving the rest).")
    ap.add_argument("--grid-size", type=int, default=20,
                     help="background/VO grid density (query count = grid_size^2). Default matches the "
                          "measured A10G ceiling for 35 frames at target_size=1288 (22.17GB peak for a single "
                          "call, see project_spatialtrackerv2_a10g memory) -- this pipeline runs a SECOND "
                          "backend call afterward for entity points, so there's less headroom than that "
                          "measurement assumed; raise only on a bigger GPU.")
    ap.add_argument("--vo-points", type=int, default=2000)
    ap.add_argument("--target-size", type=int, default=1288)
    ap.add_argument("--max-frames", type=int, default=35,
                     help="frame-count ceiling for a single STv2 window (default: measured A10G ceiling)")
    ap.add_argument("--track-mode", default="offline")
    ap.add_argument("--save-scene", action="store_true",
                     help="also save <clip>_scene.npz (raw video + dense depth + camera params, "
                          "~500MB for a 35-frame 728x1288 clip -- 99%%+ of this stage's disk footprint). "
                          "Only needed if you plan to run stage3_render.py on this clip; the per-instance "
                          "trajectory npz's (a few KB each) are saved regardless and are all that's needed "
                          "for downstream trajectory use. Off by default.")
    args = ap.parse_args()

    if bool(args.save_root) == bool(args.bundle or args.out_dir):
        raise SystemExit("Pass exactly one of --bundle/--out-dir (single-clip mode) or "
                          "--save-root/--video (chunked mode).")

    if args.save_root:
        if not args.video:
            raise SystemExit("--save-root mode requires --video (to derive video_stem).")
        run_chunked(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
