"""Shared download/manifest/cleanup helpers for the manifest-driven bulk
pipeline (stage1a.py, stage1b.py, stage2_lift3d.py, stage3_render.py). Every stage
uses these to pull a short trimmed segment from S3 into scratch, use it, and
delete it -- no stage persists a raw or resampled video file.
"""
import json
import os
import re
import subprocess
import urllib.error
import urllib.request

import boto3
from botocore.client import Config

_region_cache = {}


def get_presigned_url(bucket, key, profile="default", expires=3600):
    """Region-retry presigned URL generation -- Ego4D buckets aren't all in
    the same region as the account's default, and S3 returns a 400 naming
    the correct region rather than redirecting."""
    region = _region_cache.get(bucket, "us-west-1")
    for _ in range(4):
        s3 = boto3.session.Session(profile_name=profile, region_name=region).client(
            "s3", config=Config(signature_version="s3v4"))
        url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        try:
            with urllib.request.urlopen(req, timeout=20):
                _region_cache[bucket] = region
                return url
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            m = re.search(r"expecting '([\w-]+)'", body)
            if m and m.group(1) != region:
                region = m.group(1)
                continue
            raise RuntimeError(f"unrecoverable S3 error for {bucket}/{key} region={region}: {body[:300]}")
    raise RuntimeError(f"region resolution gave up for {bucket}")


def clamp_start(narration_timestamp_sec, video_duration_sec, duration_sec):
    """Floor at 0, cap so the trim window never runs past the video's end."""
    start = max(0.0, narration_timestamp_sec)
    start = min(start, max(0.0, video_duration_sec - duration_sec))
    return start


def download_segment(bucket, key, start_sec, duration_sec, dst_path, profile="default", timeout=120):
    """Range-fetch-trims [start_sec, start_sec+duration_sec) directly from a
    presigned S3 URL via ffmpeg, re-encoding to H.264. Re-encoding (not
    `-c copy`) is required: these source videos are VP9-in-mp4, and decord's
    threaded decoder throws avcodec_send_packet errors on strided get_batch()
    reads of a stream-copied VP9 trim (root-caused during earlier profiling
    -- worked fine for sequential single-frame reads, failed specifically on
    frame_sampling.py's strided resample)."""
    url = get_presigned_url(bucket, key, profile=profile)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", url, "-t", str(duration_sec),
           "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", dst_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not os.path.exists(dst_path):
        raise RuntimeError(f"ffmpeg trim failed for {bucket}/{key} @ {start_sec}s: {r.stderr[-500:]}")
    return dst_path


def trim_local(src_path, start_sec, duration_sec, dst_path, timeout=60):
    """Cuts [start_sec, start_sec+duration_sec) out of an ALREADY-LOCAL video
    file via ffmpeg -- same re-encode reasoning as download_segment (an
    arbitrary non-keyframe -ss cut needs a re-encode to land cleanly, and
    downstream strided reads have historically been unhappy with stream
    copies). Used to carve multiple narrations' windows out of one shared
    per-video download instead of re-fetching from S3 per narration."""
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", src_path, "-t", str(duration_sec),
           "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", dst_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not os.path.exists(dst_path):
        raise RuntimeError(f"ffmpeg local trim failed for {src_path} @ {start_sec}s: {r.stderr[-500:]}")
    return dst_path


def load_manifest_range(path, start, end=None):
    """Reads JSONL manifest rows with index in [start, end). `start`/`end`
    are positional slice bounds into the manifest, not a field lookup --
    rows are read in file order and filtered by their own "index" field so
    a manifest can be split/concatenated without losing meaning. `end=None`
    means no upper bound -- every row from `start` to the end of the
    manifest."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if start <= row["index"] and (end is None or row["index"] < end):
                rows.append(row)
    rows.sort(key=lambda r: r["index"])
    return rows


def cleanup(*paths):
    """Best-effort delete -- called from `finally` blocks so a mid-row crash
    still doesn't leave a downloaded/resampled video on disk."""
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
