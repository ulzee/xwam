"""SAM2's own hole-filling post-process (fill_holes_in_mask_scores) requires
a compiled CUDA extension (sam2._C) that isn't built in this environment (no
nvcc available, only the PTX backend via the nvidia-cuda-nvcc-cu12 pip
package -- confirmed insufficient to build a torch CUDAExtension). This is a
pure numpy/OpenCV equivalent applied to every mask we extract, so tracking
isn't silently degraded by the missing kernel. (Copied verbatim from
~/trace/annot/mask_utils.py so this pipeline directory is self-contained.)"""
import cv2
import numpy as np


def clean_mask(mask, close_kernel=15, min_area_frac=0.02):
    """mask: HxW bool array. Morphological close to fill small internal gaps,
    then drop connected components smaller than min_area_frac of the largest
    one (removes stray specks, keeps multiple genuinely separate blobs e.g.
    visible knuckles on either side of a lettuce leaf)."""
    if mask.sum() == 0:
        return mask
    m = mask.astype(np.uint8)
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_labels <= 1:
        return m.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
    max_area = areas.max()
    keep = np.zeros(n_labels, dtype=bool)
    keep[0] = False
    for i, a in enumerate(areas, start=1):
        keep[i] = a >= min_area_frac * max_area
    return keep[labels]
