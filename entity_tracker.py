"""Periodic multi-label GroundedSAM2 detection + SAM2 tracking for a whole video.

Generalizes ~/trace/annot/hand_track_video.py (periodic MediaPipe-only hand
tracking) from a single hardcoded detector to arbitrary (label, text_prompt)
pairs run through the official Grounded-SAM-2 reference detection call
(validated in ~/trace/annot/grounded_sam2_official_demo.py: HF GroundingDINO
+ SAM2, confirmed to correctly separate two simultaneous hands for most of a
clip, with graceful -- not catastrophic -- degradation under heavy occlusion).

Runs GroundingDINO every `check_interval` frames (not just once) so entities
that enter/exit frame mid-video get picked up, same rationale as the original
hand tracker. Each label gets its own independent bookkeeping (so a "hand"
detection never merges with a "knife" detection) but every label's SAM2
objects share one predictor state and get propagated together per cycle.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import decord
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2_video_predictor

from mask_utils import clean_mask

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT = os.path.expanduser("~/trace/checkpoints/sam2/sam2.1_hiera_base_plus.pt")

_gdino_model = None
_gdino_processor = None
_sam2_predictor = None


def load_gdino():
    global _gdino_model, _gdino_processor
    if _gdino_model is None:
        _gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to("cuda")
    return _gdino_model, _gdino_processor


def load_sam2_predictor():
    global _sam2_predictor
    if _sam2_predictor is None:
        _sam2_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CKPT, device="cuda")
    return _sam2_predictor


def detect_boxes(frame_rgb, prompt, box_threshold, text_threshold):
    model, processor = load_gdino()
    img = Image.fromarray(frame_rgb)
    inputs = processor(images=img, text=prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[img.size[::-1]],
    )[0]
    boxes = [[float(v) for v in b] for b in results["boxes"]]
    scores = [float(s) for s in results["scores"]]
    return list(zip(boxes, scores))


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_containment(a, b):
    """intersection / area of the SMALLER box. Catches near-total nesting
    that plain IoU misses when two boxes differ a lot in size -- e.g. a tight
    box fully inside a much larger loose box can have IoU well under 0.5
    (since the union is dominated by the big box) while still being a near-
    -exact duplicate of the smaller box's content. See filter_detections."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    smaller = min(box_area(a), box_area(b))
    return inter / smaller if smaller > 0 else 0.0


def box_center_close(a, b, factor=0.4):
    """True if the two boxes' centers are within `factor` times the larger
    box's own diagonal -- robust to a fast-moving/deforming entity whose box
    shifts by more than its own size between checks (plain IoU would wrongly
    call that a new object). Copied from hand_track_video.py's validated logic."""
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    diag_a = ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5
    diag_b = ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5
    return dist < factor * max(diag_a, diag_b)


def mask_to_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def filter_detections(dets, frame_area, area_frac_max=0.75, nms_iou_thresh=0.5, containment_thresh=0.85):
    """De-dup GroundingDINO's raw output before it reaches SAM2. Failure
    modes seen when validating against the official reference pipeline (see
    project_grounded_sam2 memory):
      - near-exact duplicate boxes for the same instance at slightly
        different confidence (e.g. the "left hand. right hand." prompt
        returning 7 mutually-overlapping boxes for 2 real hands) -- caught by
        IoU-NMS.
      - one implausibly large box spanning multiple real instances at once.
        area_frac_max is set generously high (0.75) rather than aggressively
        low, since a loose-but-large box isn't automatically a duplicate.
      - a tight box nearly fully NESTED inside a much bigger, looser box for
        the same label. Plain IoU misses this: a small box entirely inside a
        3x-larger box can have IoU well under nms_iou_thresh (union is
        dominated by the big box), so both survive as separate SAM2 objects.
        CORRECTION (2026-07-10): an earlier note here claimed a loose ~59%
        -frame-area box "resolved into a correct second-hand track once
        handed to SAM2" on this exact clip -- that was never actually
        verified pixel-by-pixel and was wrong. Direct mask-intersection
        measurement (see project_qwen_gsam2_stv2_3d_pipeline memory) showed
        that exact box, seeded either as points-from-mask or as a raw box,
        produces a mask that overlaps the tight box's mask on ALL 313 frames
        of the clip at IoU 0.46-0.72 -- i.e. it's a duplicate track, not a
        genuine second hand, and the visualized "khaki" color at the
        heaviest-occlusion frames was this overlap being double alpha-
        -blended, not an occlusion-driven identity swap as previously
        assumed. box_containment (intersection / smaller-box-area) catches
        this nesting case that box_iou misses.
    """
    dets = [d for d in dets if box_area(d[0]) <= area_frac_max * frame_area]
    dets = sorted(dets, key=lambda d: -d[1])
    kept = []
    for box, score in dets:
        if all(box_iou(box, kb) < nms_iou_thresh and box_containment(box, kb) < containment_thresh
               for kb, _ in kept):
            kept.append((box, score))
    return kept


def track_prompted_entities(video_path, prompt_specs, check_interval=15, stale_frames=10,
                             match_iou_thresh=0.2, center_close_factor=0.4,
                             box_threshold=0.3, text_threshold=0.25,
                             max_concurrent_per_label=4, area_frac_max=0.75, verbose=True):
    """prompt_specs: list of (label, category, text_prompt), e.g.
    [("hand", "hand", "a hand."), ("lettuce", "object", "lettuce.")].
    box_threshold/text_threshold default to 0.3/0.25 (not the official repo's
    0.4/0.3 default) -- validated on the lettuce clip to be the difference
    between finding both hands vs. only the one clearly-visible one.

    Returns: dict obj_id -> {
        "label": str, "category": str,
        "first_frame": int, "last_frame": int (last frame with a non-empty mask),
        "seed_mask": HxW bool array at the first frame the mask was non-empty
                     (only this one frame's pixels are retained -- keeping a
                     full per-frame mask stack for a whole clip x many
                     instances would be too large; downstream point-picking
                     only ever needs the seed frame),
        "visible": {frame_idx: bool},
        "gdino_score": float (score at the seeding detection),
    }
    """
    predictor = load_sam2_predictor()
    state = predictor.init_state(video_path=video_path)

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    n_frames = len(vr)
    frame0 = vr[0].asnumpy()
    frame_h, frame_w = frame0.shape[0], frame0.shape[1]
    frame_area = frame_h * frame_w

    active = {}   # obj_id -> {"label", "empty_streak", "last_box"}
    meta = {}     # obj_id -> instance record (see docstring)
    next_obj_id = 1

    checkpoint = 0
    while checkpoint < n_frames:
        span = min(check_interval, n_frames - checkpoint)

        for obj_id in list(active.keys()):
            if active[obj_id]["empty_streak"] >= stale_frames:
                if verbose:
                    print(f"  [frame {checkpoint}] retiring obj {obj_id} ({active[obj_id]['label']}, empty {stale_frames}+ frames)")
                del active[obj_id]

        frame = vr[checkpoint].asnumpy()

        for label, category, text_prompt in prompt_specs:
            dets = detect_boxes(frame, text_prompt, box_threshold, text_threshold)
            dets = filter_detections(dets, frame_area, area_frac_max=area_frac_max)
            active_of_label = {oid: info for oid, info in active.items() if info["label"] == label}

            for box, score in dets:
                matched = any(
                    box_iou(box, info["last_box"]) > match_iou_thresh
                    or (info["confirmed"] and box_center_close(box, info["last_box"], factor=center_close_factor))
                    for info in active_of_label.values()
                )
                if matched:
                    continue
                if len(active_of_label) >= max_concurrent_per_label:
                    if verbose:
                        print(f"  [frame {checkpoint}] unmatched '{label}' but at capacity ({max_concurrent_per_label}), skipping")
                    continue
                obj_id = next_obj_id
                next_obj_id += 1
                box_arr = np.array(box, dtype=np.float32)
                predictor.add_new_points_or_box(state, frame_idx=checkpoint, obj_id=obj_id, box=box_arr)
                active[obj_id] = {"label": label, "empty_streak": 0, "last_box": box, "confirmed": False}
                active_of_label[obj_id] = active[obj_id]
                meta[obj_id] = {
                    "label": label, "category": category,
                    "first_frame": checkpoint, "last_frame": checkpoint,
                    "seed_mask": None, "visible": {}, "gdino_score": score,
                }
                if verbose:
                    print(f"  [frame {checkpoint}] seeding new obj {obj_id} ({label}, score={score:.2f}, box={[round(v, 1) for v in box]})")

        if active:
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                    state, start_frame_idx=checkpoint, max_frame_num_to_track=span):
                for obj_id, logits in zip(obj_ids, mask_logits):
                    mask = clean_mask((logits > 0.0).squeeze(0).cpu().numpy())
                    is_visible = bool(mask.sum() > 0)
                    if obj_id in meta:
                        meta[obj_id]["visible"][frame_idx] = is_visible
                        if is_visible:
                            meta[obj_id]["last_frame"] = frame_idx
                            if meta[obj_id]["seed_mask"] is None:
                                meta[obj_id]["seed_mask"] = mask
                    if obj_id not in active:
                        continue
                    if is_visible:
                        active[obj_id]["empty_streak"] = 0
                        b = mask_to_box(mask)
                        if b is not None:
                            active[obj_id]["last_box"] = b
                    else:
                        active[obj_id]["empty_streak"] += 1

        for obj_id in active:
            active[obj_id]["confirmed"] = True

        checkpoint += span

    predictor.reset_state(state)
    return {"meta": meta, "frame_size": (frame_h, frame_w), "n_frames": n_frames}


if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    prompt_specs = [("hand", "hand", "a hand.")]
    for obj in sys.argv[2:]:
        prompt_specs.append((obj, "object", f"{obj}."))
    result = track_prompted_entities(video_path, prompt_specs)
    for obj_id, info in result["meta"].items():
        n_visible = sum(1 for v in info["visible"].values() if v)
        print(f"obj {obj_id} ({info['label']}/{info['category']}): "
              f"frames [{info['first_frame']},{info['last_frame']}], {n_visible} visible frames, "
              f"seed_mask area={info['seed_mask'].sum() if info['seed_mask'] is not None else 0}")
