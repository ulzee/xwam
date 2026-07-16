"""Stage 3 (runs in the `trace_stv2` conda env -- no SpaTrackerV2 checkout
needed, unlike stage2_lift3d.py, since this only reads the npz's stage 2
already produced): combined trace visualization. Generalizes
render_novel_view.py's chase-cam rig +
draw_trail/draw_marker (previously used for exactly ONE tracked query point)
to every instance x every FPS-sampled point saved by stage2_lift3d.py, in a
2x2 grid: original view (top-left), novel chase-cam angle (top-right),
a fixed reference frame with the cumulative 2D trajectory drawn on it so far
(bottom-left), and a blank panel reserved for future use (bottom-right).
Each instance in its own color with a legend.

The chase-cam auto-aim rig (local_rig) and the vectorized z-buffer point
splat (render_novel) are copied verbatim from render_novel_view.py -- both
already validated (see project_spatialtrackerv2_blackwell_maxquality memory).
Per-point 3D positions for the novel-angle panel come directly from
stage2_lift3d.py's saved points3d (already unprojected there), not
re-derived from depth at render time.

The bottom-left reference panel is the direct test of "reprojecting a trace
back onto the original frame": it holds one frame's RGB fixed (--ref-frame,
default 0) and draws each instance's full points2d trail from its seed_frame
up to the current playback frame t, growing over the video's duration --
unlike the top-left panel's short decaying trail_len window, this shows the
whole path traced out so far against a static background.
"""
import argparse
import json
import os

import numpy as np
import torch
import cv2
import imageio.v2 as imageio
from tqdm import tqdm

INSTANCE_COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (0, 200, 200), (240, 50, 230), (170, 110, 40), (128, 128, 0),
]


def rodrigues(axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def local_rig(side, back, height, target_dist):
    pos = np.array([side, -height, -back], dtype=np.float32)
    target = np.array([0, 0, target_dist], dtype=np.float32)
    tx, ty, tz = target - pos
    horiz = np.sqrt(tx ** 2 + tz ** 2)
    yaw = np.arctan2(tx, tz)
    pitch = np.arctan2(-ty, horiz)
    R_local = rodrigues(np.array([0, 1, 0], dtype=np.float32), yaw) @ rodrigues(np.array([1, 0, 0], dtype=np.float32), pitch)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R_local
    M[:3, 3] = pos
    return M


def unproject(depth, K, c2w, device):
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy")
    z = depth
    valid = z > 0
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts_cam = torch.stack([x, y, z], dim=-1)
    R, t = c2w[:3, :3], c2w[:3, 3]
    return pts_cam @ R.T + t, valid


def project_point(pt_world, K, w2c, device):
    pt = torch.as_tensor(pt_world, dtype=torch.float32, device=device)
    R, t = w2c[:3, :3], w2c[:3, 3]
    p_cam = R @ pt + t
    z = p_cam[2].item()
    if z <= 1e-4:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = (p_cam[0] * fx / z + cx).item()
    v = (p_cam[1] * fy / z + cy).item()
    return u, v


def draw_marker(img, u, v, color, radius=6):
    img_u8 = np.ascontiguousarray((img * 255).astype(np.uint8))
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    center = (int(round(u)), int(round(v)))
    cv2.circle(img_u8, center, radius, bgr, thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img_u8, center, 2, bgr, thickness=-1, lineType=cv2.LINE_AA)
    return img_u8.astype(np.float32) / 255.0


def draw_trail(img, pts, color):
    n = len(pts)
    if n < 2:
        return img
    img_u8 = np.ascontiguousarray((img * 255).astype(np.uint8))
    for i in range(1, n):
        p0, p1 = pts[i - 1], pts[i]
        if p0 is None or p1 is None:
            continue
        alpha = i / (n - 1)
        thickness = max(1, int(round(1 + 2 * alpha)))
        bgr = (int(color[2] * alpha), int(color[1] * alpha), int(color[0] * alpha))
        pt0 = (int(round(p0[0])), int(round(p0[1])))
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        cv2.line(img_u8, pt0, pt1, bgr, thickness, lineType=cv2.LINE_AA)
    return img_u8.astype(np.float32) / 255.0


def draw_legend(img, instances, t):
    img_u8 = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    for i, inst in enumerate(instances):
        active = t >= inst["seed_frame"]
        label_txt = inst["label"] if active else f"{inst['label']} (not yet seen)"
        c = inst["color"]
        col_bgr = (int(c[2]), int(c[1]), int(c[0])) if active else (120, 120, 120)
        cv2.putText(img_u8, label_txt, (10, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_bgr, 2, cv2.LINE_AA)
    return img_u8.astype(np.float32) / 255.0


def draw_panel_title(img, text):
    img_u8 = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    h = img_u8.shape[0]
    cv2.putText(img_u8, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return img_u8.astype(np.float32) / 255.0


def render_novel(pts_world, colors, valid, K_novel, w2c_novel, H, W, device, splat=1):
    R, t = w2c_novel[:3, :3], w2c_novel[:3, 3]
    pts_cam = pts_world @ R.T + t
    z = pts_cam[..., 2]
    fx, fy, cx, cy = K_novel[0, 0], K_novel[1, 1], K_novel[0, 2], K_novel[1, 2]
    u = pts_cam[..., 0] * fx / z.clamp(min=1e-6) + cx
    v = pts_cam[..., 1] * fy / z.clamp(min=1e-6) + cy

    valid_full = valid & (z > 1e-4)
    u, v, z, col = u[valid_full], v[valid_full], z[valid_full], colors[valid_full]

    img = torch.zeros(H, W, 3, device=device)
    depth_buf = torch.full((H * W,), float("inf"), device=device)
    for du in range(-splat, splat + 1):
        for dv in range(-splat, splat + 1):
            uu = (u + du).round().long()
            vv = (v + dv).round().long()
            inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            if inb.sum() == 0:
                continue
            flat = (vv[inb] * W + uu[inb]).long()
            zz, cc = z[inb], col[inb]
            depth_buf.scatter_reduce_(0, flat, zz, reduce="amin", include_self=True)
            mask = zz <= (depth_buf[flat] + 1e-4)
            img.view(-1, 3).index_put_((flat[mask],), cc[mask], accumulate=False)
    return img.clamp(0, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="stage2_lift3d.py's <clip>_trace_summary.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--side-frac", type=float, default=1.0)
    ap.add_argument("--back-frac", type=float, default=1.0)
    ap.add_argument("--height-frac", type=float, default=0.4)
    ap.add_argument("--target-frac", type=float, default=1.5)
    ap.add_argument("--fixed", action="store_true", help="hold the novel camera fixed (frame-0 pose) instead of chase-cam")
    ap.add_argument("--fps", type=float, default=None,
                     help="output video fps. Defaults to the source clip's own fps (from the trace summary) "
                          "so playback speed matches real time; pass a value explicitly to override (e.g. "
                          "for deliberate slow-motion).")
    ap.add_argument("--splat", type=int, default=1)
    ap.add_argument("--trail-len", type=int, default=7)
    ap.add_argument("--ref-frame", type=int, default=0,
                     help="frame index to hold fixed as the background of the bottom-left reference panel, "
                          "onto which each instance's cumulative points2d trail (seed_frame..t) is drawn.")
    args = ap.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    out_dir = os.path.dirname(os.path.abspath(args.summary))

    if args.fps is None:
        args.fps = summary.get("fps") or 12
        print(f"--fps not given, using source clip fps from trace summary: {args.fps:.2f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not summary.get("scene_file"):
        raise RuntimeError(
            "No scene_file in trace summary -- stage2_lift3d.py was run without --save-scene, "
            "so there's no video/depth/camera data to render from. Re-run stage2_lift3d.py with "
            "--save-scene for this clip."
        )
    scene = np.load(os.path.join(out_dir, summary["scene_file"]))
    video, depths = scene["video"], scene["depths"]
    intrinsics, extrinsics = scene["intrinsics"], scene["extrinsics"]
    T, C, H, W = video.shape
    c2ws = np.linalg.inv(extrinsics)

    instances = []
    for i, inst in enumerate(summary["instances"]):
        traj = np.load(os.path.join(out_dir, inst["trajectory_file"]))
        instances.append({
            "label": inst["label"], "color": INSTANCE_COLORS[i % len(INSTANCE_COLORS)],
            "seed_frame": int(traj["seed_frame"]),
            "points3d": traj["points3d"], "points2d": traj["points2d"], "visibility": traj["visibility"],
        })
    print(f"Rendering {len(instances)} instance(s) over {T} frames: "
          f"{[inst['label'] for inst in instances]}")

    valid_depths = depths[depths > 0]
    median_depth = float(np.median(valid_depths)) if len(valid_depths) > 0 else 1.0
    K_novel = torch.from_numpy(intrinsics[0]).to(device)
    c2w0 = torch.from_numpy(c2ws[0]).to(device)

    rig = local_rig(side=median_depth * args.side_frac, back=median_depth * args.back_frac,
                     height=median_depth * args.height_frac, target_dist=median_depth * args.target_frac)
    rig = torch.from_numpy(rig).to(device)
    fixed_w2c_novel = torch.inverse(c2w0 @ rig)
    print(f"median scene depth: {median_depth:.3f}, rig offset: {rig[:3,3].cpu().numpy()}, fixed={args.fixed}")

    ref_frame = max(0, min(args.ref_frame, T - 1))
    ref_img_base = video[ref_frame].transpose(1, 2, 0).copy()
    blank_img = np.full_like(ref_img_base, 0.12)

    def pixel_of(inst, t):
        """(u,v) per point at frame t, or None per-point if not visible/not yet appeared."""
        lt = t - inst["seed_frame"]
        if lt < 0 or lt >= inst["points2d"].shape[0]:
            return [None] * inst["points2d"].shape[1]
        return [tuple(inst["points2d"][lt, n]) if inst["visibility"][lt, n] else None
                for n in range(inst["points2d"].shape[1])]

    def world_of(inst, t):
        lt = t - inst["seed_frame"]
        if lt < 0 or lt >= inst["points3d"].shape[0]:
            return [None] * inst["points3d"].shape[1]
        return [inst["points3d"][lt, n] if inst["visibility"][lt, n] else None
                for n in range(inst["points3d"].shape[1])]

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    for t in tqdm(range(T)):
        d = torch.from_numpy(depths[t]).to(device)
        K = torch.from_numpy(intrinsics[t]).to(device)
        c2w = torch.from_numpy(c2ws[t]).to(device)
        rgb = torch.from_numpy(video[t]).permute(1, 2, 0).to(device).float()

        w2c_novel = fixed_w2c_novel if args.fixed else torch.inverse(c2w @ rig)
        pts_world, valid = unproject(d, K, c2w, device)
        novel_img = render_novel(pts_world, rgb, valid, K_novel, w2c_novel, H, W, device, splat=args.splat)
        orig_img = video[t].transpose(1, 2, 0).copy()
        ref_img = ref_img_base.copy()

        for inst in instances:
            color = inst["color"]
            pix_now = pixel_of(inst, t)
            world_now = world_of(inst, t)
            n_pts = len(pix_now)
            for n in range(n_pts):
                trail_px = [pixel_of(inst, s)[n] for s in range(max(inst["seed_frame"], t - args.trail_len), t + 1)]
                orig_img = draw_trail(orig_img, trail_px, color)
                if pix_now[n] is not None:
                    orig_img = draw_marker(orig_img, *pix_now[n], color)

                trail_world = [world_of(inst, s)[n] for s in range(max(inst["seed_frame"], t - args.trail_len), t + 1)]
                trail_novel = [project_point(w, K_novel, w2c_novel, device) if w is not None else None for w in trail_world]
                trail_novel = [p if (p is not None and 0 <= p[0] < W and 0 <= p[1] < H) else None for p in trail_novel]
                novel_img = draw_trail(novel_img, trail_novel, color)
                if world_now[n] is not None:
                    p = project_point(world_now[n], K_novel, w2c_novel, device)
                    if p is not None and 0 <= p[0] < W and 0 <= p[1] < H:
                        novel_img = draw_marker(novel_img, *p, color)

                # Reference panel: cumulative trail from this instance's own seed_frame up to
                # t (not clipped to trail_len) drawn on the fixed --ref-frame background, so
                # the whole path traced out so far accumulates over the video's duration.
                cum_trail_px = [pixel_of(inst, s)[n] for s in range(inst["seed_frame"], t + 1)]
                ref_img = draw_trail(ref_img, cum_trail_px, color)
                if pix_now[n] is not None:
                    ref_img = draw_marker(ref_img, *pix_now[n], color)

        orig_img = draw_legend(orig_img, instances, t)
        ref_img = draw_legend(ref_img, instances, t)
        ref_img = draw_panel_title(ref_img, f"reference frame {ref_frame} + cumulative trace")

        top_row = np.concatenate([orig_img, novel_img], axis=1)
        bottom_row = np.concatenate([ref_img, blank_img], axis=1)
        combined = np.concatenate([top_row, bottom_row], axis=0)
        writer.append_data((np.clip(combined, 0, 1) * 255).astype(np.uint8))
    writer.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
