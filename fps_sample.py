"""Farthest-point sampling of query points from a boolean mask.

Replaces the official Grounded-SAM-2 repo's own `sample_points_from_masks`
(uniform random, see ~/trace/annot/grounded_sam2_official_demo.py) with a
spatially-spread selection: redundancy against occlusion/mask-boundary noise
plus enough spatial spread to recover an object's orientation, not just its
centroid -- discussed and agreed for the Ego4D 3D-trace pipeline design.
"""
import cv2
import numpy as np


def erode_mask(mask, margin):
    """Shrinks `mask` inward by `margin` pixels so sampled points land away
    from the mask boundary -- a SAM2 mask's edge pixels are the noisiest part
    frame-to-frame (this is where a point is most likely to flicker between
    being counted foreground vs. background as the mask wobbles slightly
    between frames), so keeping query points off the edge makes their
    visibility/position more stable across the clip. Falls back to a smaller
    margin (down to the unmodified mask) if erosion would eliminate the mask
    entirely -- e.g. a thin sliver of visible hand during heavy occlusion."""
    if margin <= 0 or mask.sum() == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    if eroded.sum() == 0:
        return erode_mask(mask, margin - 1)
    return eroded


def farthest_point_sample(mask, num_points, seed=0, margin=0):
    """mask: (H,W) bool array. margin: erode the mask inward by this many
    pixels before sampling (see erode_mask) so query points avoid noisy mask
    edges; 0 disables this and samples from the mask as given. Returns
    (num_points, 2) array of (x,y) pixel coords, spatially spread via greedy
    farthest-point sampling. If the mask has fewer than num_points pixels,
    samples are repeated (with replacement) to fill the budget, matching the
    official repo's own fallback behavior."""
    if margin > 0:
        mask = erode_mask(mask, margin)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)  # (M,2)
    m = len(pts)

    rng = np.random.default_rng(seed)
    if m <= num_points:
        idx = rng.choice(m, num_points, replace=True)
        return pts[idx]

    # greedy FPS: start from the point closest to the mask centroid (a
    # deterministic, representative seed rather than a random first pick)
    centroid = pts.mean(axis=0)
    first = int(np.argmin(((pts - centroid) ** 2).sum(axis=1)))
    chosen = [first]
    min_dist = ((pts - pts[first]) ** 2).sum(axis=1)

    for _ in range(num_points - 1):
        next_idx = int(np.argmax(min_dist))
        chosen.append(next_idx)
        d = ((pts - pts[next_idx]) ** 2).sum(axis=1)
        min_dist = np.minimum(min_dist, d)

    return pts[chosen]


if __name__ == "__main__":
    # quick smoke test: a hollow ring mask should get points spread around
    # the ring, not clustered near the centroid (which lies outside the mask)
    H, W = 200, 200
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    r = np.sqrt((xx - 100) ** 2 + (yy - 100) ** 2)
    ring = (r > 60) & (r < 80)
    pts = farthest_point_sample(ring, 8)
    print(pts)
    spread = np.arctan2(pts[:, 1] - 100, pts[:, 0] - 100)
    print("angles (deg):", np.sort(np.degrees(spread)))
