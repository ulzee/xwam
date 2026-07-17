"""Frame-rate resampling + chunk-window math shared by stage1b.py and
stage2_lift3d.py's chunked ("bulk annotation") run mode.

The resampling happens ONCE per source video, up front: a video's native fps
is usually much higher than needed for tracking (e.g. 24-60fps), and
SpatialTrackerV2's window is capped at a fixed FRAME COUNT (--max-frames,
default 35) for VRAM reasons -- so at native fps that window only covers a
second or two. Resampling to a lower --target-fps first stretches the same
35-frame budget over a proportionally longer span of wall-clock time, at the
cost of coarser temporal resolution.

Doing this once up front (rather than threading a stride through both
stages' frame-indexing logic) means both stage1's entity_tracker (SAM2's
video predictor has no native stride/skip-frames mechanism -- see
entity_tracker.py's module docstring) and stage2's decord-based windowing
can keep operating on plain contiguous frame ranges, just of the
already-resampled file instead of the original -- no per-stage stride-aware
reindexing needed.
"""
import os
import decord
import imageio.v2 as imageio


def compute_stride(native_fps, target_fps):
    """Nearest-integer stride, floored at 1 (never upsample)."""
    if target_fps is None or target_fps <= 0 or target_fps >= native_fps:
        return 1
    return max(1, round(native_fps / target_fps))


def resample_video(src_path, dst_path, target_fps, verbose=True):
    """Reads every `stride`-th frame of src_path and writes them to dst_path
    at the resulting actual fps (native_fps / stride -- not necessarily
    exactly target_fps, since stride is an integer). Returns a dict of the
    resampling metadata every downstream consumer needs.

    Skips the work and returns stride=1 metadata if dst_path already exists
    (chunked runs call this once per video; re-running the same video should
    be cheap on retry, not re-encode from scratch).
    """
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(src_path)
    native_n_frames = len(vr)
    native_fps = float(vr.get_avg_fps())
    stride = compute_stride(native_fps, target_fps)
    actual_fps = native_fps / stride
    indices = list(range(0, native_n_frames, stride))
    n_frames_resampled = len(indices)

    if os.path.exists(dst_path):
        if verbose:
            print(f"Resampled video already exists, skipping re-encode: {dst_path}")
    else:
        if verbose:
            print(f"Resampling {src_path} ({native_n_frames}f @ {native_fps:.2f}fps) -> "
                  f"{dst_path} ({n_frames_resampled}f @ {actual_fps:.2f}fps, stride={stride}, "
                  f"target_fps={target_fps})")
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        frames = vr.get_batch(indices).asnumpy()
        writer = imageio.get_writer(dst_path, fps=actual_fps, codec="libx264", quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()

    return {
        "native_fps": native_fps,
        "native_n_frames": native_n_frames,
        "target_fps": target_fps,
        "actual_fps": actual_fps,
        "stride": stride,
        "n_frames_resampled": n_frames_resampled,
        "resampled_video": os.path.abspath(dst_path),
    }


def compute_chunk_starts(n_frames_resampled, start_frame, max_frames, min_chunk_frames=2):
    """Non-overlapping chunk start offsets (in resampled-video frame units)
    covering [start_frame, n_frames_resampled). A trailing chunk shorter than
    min_chunk_frames is dropped rather than emitted as a near-empty chunk."""
    starts = []
    c = start_frame
    while c < n_frames_resampled:
        remaining = n_frames_resampled - c
        if remaining < min_chunk_frames:
            break
        starts.append(c)
        c += max_frames
    return starts
