"""Stage 1a (runs in the `qwen_sam2` conda env): Qwen3-VL scene
understanding only -- narration + keyframes -> task description + object
list to track. Manifest-driven bulk mode only; for every row in
[--start, --end), runs Qwen and writes identify.json.

Split out of a single interleaved stage1_detect.py into its own process
specifically so Qwen3-VL (~17.5GB resident) only ever has to load ONCE for
a whole --start:--end range: on a 23GB card there's no room left for
GroundingDINO+SAM2 (stage1b.py) in the same process. Since this is now its
own process, it doesn't need to free Qwen when done -- just exits.

Manifest rows for the same video_uid are always consecutive (guaranteed by
build_manifest_ego4d.py's build_rows(), which iterates videos in the outer
loop) -- so rows are grouped by video_uid via itertools.groupby, then
greedily split into SUB-groups (see make_subgroups) capped at
--max-group-span-sec: narrations of the same video can be minutes apart
(same video_uid does NOT mean "nearby"), so blindly spanning a video's
full min-to-max narration range risked a multi-hundred-second download
that both timed out and defeated the whole point of this optimization.
Within each sub-group, ONE shared segment is downloaded spanning every
member narration's window, and each narration's own window is carved out
of that shared local file with a local ffmpeg trim (no network) before
being handed to Qwen.

Downloading and inference run as a producer/consumer pair on two threads,
so the (CPU-bound) ffmpeg download+re-encode of the NEXT sub-group overlaps
the (GPU-bound) Qwen inference on the CURRENT one instead of the two
serializing:
  - download_worker (background thread): walks every sub-group in order,
    downloads it, and puts it on a bounded queue.Queue(maxsize=--prefetch)
    -- `Queue.put` blocks once the queue is full, which is what keeps the
    download side a fixed ~--prefetch sub-groups ahead instead of racing
    arbitrarily far ahead and piling up scratch disk usage.
  - main thread (consumer): pulls one ready sub-group at a time, runs Qwen
    on each of its rows, and ONLY THEN deletes that sub-group's shared
    download -- i.e. the inference side is what cleans up, once it's
    actually done with the file, not the download side.

stage1b.py reads identify.json (task/objects, plus the bucket/key/start_sec/
duration_sec used here, so it replays the identical PER-NARRATION download
-- it doesn't get the grouped-download or prefetch optimizations, since
it's a separate process and doesn't share stage1a.py's in-progress state).
"""
import os
import sys
# Checked via raw sys.argv, not argparse, because this has to run BEFORE
# `from scene_understanding import ...` below -- scene_understanding.py sets
# HF_HUB_OFFLINE=1 (via setdefault) at its own import time, and transformers
# reads that env var at import/call time too, so by the time argparse would
# normally run it's already too late to flip it back off.
_force_redownload_models = "--force-redownload-models" in sys.argv
if _force_redownload_models:
    os.environ["HF_HUB_OFFLINE"] = "0"
else:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse
import itertools
import json
import queue
import threading
import time

from scene_understanding import identify_objects, load_model
from frame_sampling import compute_stride
from dataset_io import download_segment, load_manifest_range, clamp_start, cleanup, trim_local


def format_eta(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def segment_duration_sec(native_fps, target_fps, max_frames):
    """How long a trimmed segment needs to be (at this row's own native_fps)
    to resample down to exactly one full max_frames chunk at target_fps --
    same stride math frame_sampling.resample_video uses, computed here up
    front so the download is sized right the first time instead of
    guessing a fixed duration for every video regardless of its native fps."""
    stride = compute_stride(native_fps, target_fps)
    return max_frames * stride / native_fps


def make_subgroups(pending, max_span_sec):
    """Greedily splits `pending` (rows of one video_uid, already in
    narration_timestamp_sec order) into runs whose combined
    [min_start, max_end) span stays within max_span_sec, so a shared
    download never has to cover more than that much wall-clock video
    regardless of how far apart two narrations of the same video actually
    are. A single row always forms its own subgroup if it doesn't fit with
    its predecessor, even if its own window alone exceeds max_span_sec."""
    subgroups = []
    current, cur_start, cur_end = [], None, None
    for p in pending:
        s, e = p["start_sec"], p["start_sec"] + p["duration_sec"]
        if current:
            new_start, new_end = min(cur_start, s), max(cur_end, e)
            if new_end - new_start <= max_span_sec:
                current.append(p)
                cur_start, cur_end = new_start, new_end
                continue
            subgroups.append(current)
        current, cur_start, cur_end = [p], s, e
    if current:
        subgroups.append(current)
    return subgroups


def download_worker(rows, args, work_queue):
    """Producer thread body: walks every sub-group across the whole row
    range in order, downloads it, and puts it on work_queue for the
    consumer (main thread) to run inference on. work_queue's maxsize is
    the backpressure mechanism -- `put()` blocks once the consumer has
    fallen --prefetch sub-groups behind, so this thread naturally stays a
    bounded distance ahead instead of downloading the entire range up
    front.

    Each item is a dict: {video_uid, subgroup, raw_path, group_start,
    n_seen, error}. `n_seen` is the cumulative count of manifest rows (both
    skipped-via---check-existing and pending) resolved by this thread up
    through the end of this sub-group -- the consumer uses it to keep
    printing an accurate "N row(s) left" without needing its own copy of
    the skip bookkeeping (which only this thread does). `error` is set
    (and raw_path already cleaned up) if the download itself failed --
    the consumer just logs and moves on rather than trying to process it.

    Always puts a final `None` sentinel, even on an unexpected exception,
    so the consumer's queue.get() can never deadlock waiting for work that
    will never arrive.
    """
    try:
        n_seen = 0
        for video_uid, group_iter in itertools.groupby(rows, key=lambda r: r["video_uid"]):
            group_rows = list(group_iter)

            pending = []
            for row in group_rows:
                seg_id = row["segment_id"]
                out_dir = os.path.join(args.save_root, seg_id)
                identify_path = os.path.join(out_dir, "identify.json")
                if args.check_existing and os.path.exists(identify_path):
                    print(f"[{seg_id}] identify.json exists, skipping")
                    n_seen += 1
                    continue
                duration_sec = segment_duration_sec(row["native_fps"], args.target_fps, args.max_frames)
                start_sec = clamp_start(row["narration_timestamp_sec"], row["video_duration_sec"], duration_sec)
                pending.append({"row": row, "out_dir": out_dir, "identify_path": identify_path,
                                 "start_sec": start_sec, "duration_sec": duration_sec})
            if not pending:
                continue

            for sub_idx, subgroup in enumerate(make_subgroups(pending, args.max_group_span_sec)):
                group_start = min(p["start_sec"] for p in subgroup)
                group_end = max(p["start_sec"] + p["duration_sec"] for p in subgroup)
                group_duration = group_end - group_start
                bucket, key = subgroup[0]["row"]["bucket"], subgroup[0]["row"]["key"]
                # "shared" makes this structurally impossible to collide with any
                # segment_id-derived path (segment_id is always f"{video_uid}_{NNN}",
                # so a per-row seg_path is f"{video_uid}_{NNN}_raw.mp4" -- if this
                # used the same f"{video_uid}_{NNN}_raw.mp4" shape, sub_idx and a
                # row's local narration index (two different, both-start-at-0
                # counters) would collide constantly, e.g. every video's first
                # sub-group (sub_idx=0) always contains local narration index 0.
                # That collision actually happened: it corrupted the shared
                # download via an in-place-edit ffmpeg refusal (same-thread) and,
                # separately, via a producer/consumer race where the download
                # thread was writing the NEXT sub-group's file while the consumer
                # thread was still trimming/reading FROM a same-named path in the
                # current one.
                raw_path = os.path.join(args.tmp_dir, f"{video_uid}_shared{sub_idx:03d}_raw.mp4")
                error = None
                try:
                    print(f"[{video_uid}] downloading shared segment [{group_start:.1f}s, {group_end:.1f}s) "
                          f"covering {len(subgroup)} narration(s)...")
                    download_segment(bucket, key, group_start, group_duration, raw_path,
                                      timeout=max(120, int(group_duration) + 60))
                except Exception as e:
                    error = str(e)
                    cleanup(raw_path)

                n_seen += len(subgroup)
                work_queue.put({"video_uid": video_uid, "subgroup": subgroup, "raw_path": raw_path,
                                 "group_start": group_start, "n_seen": n_seen, "error": error})
    finally:
        work_queue.put(None)


def run_identify(video_path, narration, verbose=True):
    if verbose:
        print(f"Narration: {narration!r}")
        print("Running Qwen3-VL scene understanding...")
    task, objects, background = identify_objects(video_path, narration=narration)
    if verbose:
        print(f"  task: {task}")
        print(f"  objects: {objects}")
        print(f"  background: {background}")
    return task, objects, background


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSONL manifest from build_manifest_ego4d.py")
    ap.add_argument("--start", type=int, required=True, help="first manifest row index (inclusive)")
    ap.add_argument("--end", type=int, default=None,
                     help="last manifest row index (exclusive); omit to process through the end of the manifest")
    ap.add_argument("--save-root", required=True)
    ap.add_argument("--tmp-dir", default=os.path.expanduser("~/ego4d/scratch"),
                     help="scratch directory for downloaded clips. Nothing durable lives here -- "
                          "safe to delete anytime between runs.")
    ap.add_argument("--target-fps", type=float, default=10,
                     help="resample-to fps used ONLY to size the download window -- must match stage1b.py's "
                          "--target-fps or the recorded duration_sec will over/under-shoot one full chunk.")
    ap.add_argument("--max-frames", type=int, default=35,
                     help="chunk frame ceiling used ONLY to size the download window -- must match "
                          "stage1b.py's --max-frames.")
    ap.add_argument("--max-group-span-sec", type=float, default=30.0,
                     help="cap on how much wall-clock video a single shared per-video download may cover. "
                          "Narrations of the same video_uid can be minutes or hours apart, so this bounds the "
                          "shared-download optimization to narrations that are actually close together -- "
                          "farther-apart narrations of the same video fall into separate sub-groups (and "
                          "separate downloads) instead of one huge, slow, mostly-wasted download.")
    ap.add_argument("--check-existing", action="store_true", default=False,
                     help="skip a row if its identify.json already exists (resume mode). Default is to "
                          "overwrite/re-run every row in range, so re-running with different Qwen prompting "
                          "or a different manifest doesn't silently keep stale output.")
    ap.add_argument("--prefetch", type=int, default=2,
                     help="how many sub-groups the background download thread is allowed to keep "
                          "downloaded-and-waiting ahead of Qwen inference (producer/consumer backpressure via "
                          "a bounded queue). Combined with the one it's actively downloading, this keeps "
                          "roughly --prefetch + 1 (~2-3 at the default) shared segments on disk at once.")
    ap.add_argument("--force-redownload-models", action="store_true", default=False,
                     help="bypass the local HuggingFace cache and re-fetch Qwen3-VL from the Hub, "
                          "overwriting any existing blobs/symlinks. Use if a resumed AWS instance's HF "
                          "cache looks present but is actually corrupt (stale/broken snapshot symlinks "
                          "after a disk swap) -- symptom is usually a load_model() crash or a model that "
                          "loads but behaves as if weights are missing/garbled. Also disables "
                          "HF_HUB_OFFLINE for this run, since the two are mutually exclusive.")
    ap.add_argument("--quiet", dest="verbose", action="store_false", default=True)
    args = ap.parse_args()
    assert args.force_redownload_models == _force_redownload_models, \
        "internal: argparse and the pre-import sys.argv check disagree on --force-redownload-models"

    rows = load_manifest_range(args.manifest, args.start, args.end)
    print(f"Loaded {len(rows)} manifest row(s) in [{args.start}, {args.end if args.end is not None else 'end'})")
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    load_model(force_download=args.force_redownload_models)

    work_queue = queue.Queue(maxsize=args.prefetch)
    producer = threading.Thread(target=download_worker, args=(rows, args, work_queue), daemon=True)
    producer.start()

    n_processed, total_row_time = 0, 0.0
    n_total_rows = len(rows)
    while True:
        item = work_queue.get()
        if item is None:
            break
        video_uid, subgroup, raw_path = item["video_uid"], item["subgroup"], item["raw_path"]
        group_start = item["group_start"]

        if item["error"] is not None:
            print(f"[{video_uid}] FAILED to download shared segment: {item['error']}")
            continue

        single = len(subgroup) == 1
        base_n_seen = item["n_seen"] - len(subgroup)
        for row_idx, p in enumerate(subgroup):
            row, out_dir, identify_path = p["row"], p["out_dir"], p["identify_path"]
            start_sec, duration_sec = p["start_sec"], p["duration_sec"]
            seg_id = row["segment_id"]
            os.makedirs(out_dir, exist_ok=True)
            seg_path = raw_path if single else os.path.join(args.tmp_dir, f"{seg_id}_raw.mp4")
            assert single or seg_path != raw_path, \
                f"filename collision: seg_path {seg_path!r} would alias shared download {raw_path!r}"
            row_t0 = time.time()
            try:
                if not single:
                    trim_local(raw_path, start_sec - group_start, duration_sec, seg_path)
                task, objects, background = run_identify(seg_path, row["narration_text"], verbose=args.verbose)
                with open(identify_path, "w") as f:
                    json.dump({
                        "task": task, "objects": objects, "background": background,
                        "narration_text": row["narration_text"],
                        "bucket": row["bucket"], "key": row["key"],
                        "start_sec": start_sec, "duration_sec": duration_sec,
                    }, f, indent=2)
                print(f"[{seg_id}] task={task!r} objects={objects}")
            except Exception as e:
                print(f"[{seg_id}] FAILED: {e}")
            finally:
                if not single:
                    cleanup(seg_path)

            n_processed += 1
            total_row_time += time.time() - row_t0
            avg = total_row_time / n_processed
            remaining = n_total_rows - (base_n_seen + row_idx + 1)
            print(f"[{seg_id}] {avg:.1f}s/row avg over {n_processed} row(s), "
                  f"{remaining} row(s) left, ETA {format_eta(avg * remaining)}")

        cleanup(raw_path)  # inference side is done with this sub-group -- delete it, move on

    producer.join()
    print("Stage 1a done.")


if __name__ == "__main__":
    main()
