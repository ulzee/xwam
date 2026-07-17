"""Stage 0 (dataset-specific): builds a flat, indexable manifest of every
(video, narration) segment in the Ego4D FHO annotation set, one JSON line
per segment. Downstream stages slice this manifest by --start/--end row
index -- multiple narrations of the same video just become multiple
consecutive rows, so "process videos --start to --end" and "account for
multiple narrations at varying timepoints per video" are the same
mechanism.

Joins two local files (both already downloaded via the Ego4D CLI):
  - annotations/fho_main.json: per-video narrated_actions (text +
    narration_timestamp_sec), keyed by video_uid. ~2GB, loaded whole with
    json.load -- verified to work fine on this box.
  - full_scale/manifest.csv: per-video S3 location (`path` column,
    `s3://bucket/key`), native fps/resolution/duration. This is the join
    key back to something stage1/2/3 can actually download.

A narrated_action is dropped if `is_rejected` or `is_invalid_annotation` is
true (annotator-flagged noise, e.g. "no human-object interaction").
"""
import argparse
import csv
import json
import re
from urllib.parse import urlparse

DEFAULT_FHO_MAIN = "/home/ubuntu/ego4d/data/v2/annotations/fho_main.json"
DEFAULT_FULL_SCALE_MANIFEST = "/home/ubuntu/ego4d/data/v2/full_scale/manifest.csv"
DEFAULT_OUT = "/home/ubuntu/ego4d/manifest.jsonl"


def load_video_locations(manifest_csv_path):
    """video_uid -> {bucket, key, canonical_fps, width, height, duration_sec}."""
    locations = {}
    with open(manifest_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            path = row.get("canonical_s3_location") or row.get("path")
            if not path:
                continue
            parsed = urlparse(path)
            if parsed.scheme != "s3":
                continue
            fps = row.get("canonical_fps") or row.get("fps")
            w = row.get("canonical_display_width") or row.get("display_resolution_width")
            h = row.get("canonical_display_height") or row.get("display_resolution_height")
            dur = row.get("canonical_mp4_duration_sec") or row.get("duration_sec")
            if not (fps and w and h and dur):
                continue
            locations[row["video_uid"]] = {
                "bucket": parsed.netloc,
                "key": parsed.path.lstrip("/"),
                "native_fps": float(fps),
                "width": int(float(w)),
                "height": int(float(h)),
                "video_duration_sec": float(dur),
            }
    return locations


CAMERA_WEARER_TAG_RE = re.compile(r"^#c+\b", re.IGNORECASE)
INLINE_TAG_RE = re.compile(r"#\w+")
BARE_SUBJECT_RE = re.compile(r"\bC+\b", re.IGNORECASE)


def is_camera_wearer_narration(text):
    """Ego4D narrations are annotator-tagged by whose action it is: '#C'
    (the camera wearer) vs. '#O' (another person in the scene) vs. an
    '#unsure'-only/untagged line. Only '#C' (allowing the 'CC'/'c' typo
    variants seen in the raw data) is the camera wearer's own action --
    that's the only one that matches "what this clip's video actually
    shows the camera wearer doing", which is what narration-seeding wants."""
    return bool(CAMERA_WEARER_TAG_RE.match(text))


def clean_camera_wearer_narration(text):
    """Strips the leading '#C' annotator tag and any other inline '#tag'
    annotation (e.g. '#Unsure'), then replaces the bare 'C'/'CC'/'c' subject
    pronoun the annotators used for "the camera wearer" with 'embodiment',
    so the text reads as a normal sentence instead of annotator shorthand.
    E.g. '#C C picks the cat teaser wand from the toy box' ->
    'Embodiment picks the cat teaser wand from the toy box'."""
    text = INLINE_TAG_RE.sub("", text)
    text = BARE_SUBJECT_RE.sub("embodiment", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def iter_narrations(fho_main_path):
    """Yields (video_uid, narration_text, narration_timestamp_sec) for every
    non-rejected, non-invalid, camera-wearer-tagged ('#C') narrated_action
    across the whole FHO set, in file order (stable within a video).
    narration_text has already been cleaned (see clean_camera_wearer_narration) --
    the raw annotator shorthand never makes it into the manifest."""
    with open(fho_main_path) as f:
        data = json.load(f)
    for video in data["videos"]:
        video_uid = video["video_uid"]
        for interval in video["annotated_intervals"]:
            for na in interval["narrated_actions"]:
                if na.get("is_rejected") or na.get("is_invalid_annotation"):
                    continue
                text = (na.get("narration_text") or "").strip()
                ts = na.get("narration_timestamp_sec")
                if not text or ts is None:
                    continue
                if not is_camera_wearer_narration(text):
                    continue
                cleaned = clean_camera_wearer_narration(text)
                if not cleaned:
                    continue
                yield video_uid, cleaned, ts


def build_rows(fho_main_path, manifest_csv_path, limit=None):
    locations = load_video_locations(manifest_csv_path)
    print(f"{len(locations)} videos with a resolvable S3 location in {manifest_csv_path}")

    per_video = {}  # video_uid -> list of (narration_text, timestamp_sec)
    n_total, n_no_location = 0, 0
    for video_uid, text, ts in iter_narrations(fho_main_path):
        n_total += 1
        if video_uid not in locations:
            n_no_location += 1
            continue
        per_video.setdefault(video_uid, []).append((text, ts))

    rows = []
    for video_uid in sorted(per_video):
        loc = locations[video_uid]
        narrations = sorted(per_video[video_uid], key=lambda x: x[1])
        for local_idx, (text, ts) in enumerate(narrations):
            rows.append({
                "index": len(rows),
                "segment_id": f"{video_uid}_{local_idx:03d}",
                "video_uid": video_uid,
                "bucket": loc["bucket"],
                "key": loc["key"],
                "native_fps": loc["native_fps"],
                "width": loc["width"],
                "height": loc["height"],
                "video_duration_sec": loc["video_duration_sec"],
                "narration_text": text,
                "narration_timestamp_sec": ts,
            })
            if limit and len(rows) >= limit:
                print(f"{n_total} camera-wearer (#C) narrated_actions seen, {n_no_location} dropped (no S3 location), "
                      f"stopping early at --limit {limit}")
                return rows

    print(f"{n_total} camera-wearer (#C) narrated_actions seen, {n_no_location} dropped (no S3 location), "
          f"{len(rows)} segment(s) across {len(per_video)} video(s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fho-main", default=DEFAULT_FHO_MAIN)
    ap.add_argument("--full-scale-manifest", default=DEFAULT_FULL_SCALE_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                     help="stop after this many segment rows (smoke-testing the join before a full build)")
    args = ap.parse_args()

    rows = build_rows(args.fho_main, args.full_scale_manifest, limit=args.limit)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
