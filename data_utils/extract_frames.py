# data_utils/extract_frames.py
import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime

from data_utils.file_utils import prepare_output_dir

"""
Extract Image frames as given fps using ffmpeg utility
"""

IDX_SPEC_RE = re.compile(r'^(?P<prefix>.*)%(?:0(?P<width>\d+))?d(?P<suffix>.*)$')

def _pattern_regex(pattern: str) -> re.Pattern:
    """
    Turn printf pattern like 'frame_%06d.jpg' into a regex that captures the integer index.
    """
    m = IDX_SPEC_RE.match(pattern)
    if not m:
        # Fallback: capture trailing digits before extension
        return re.compile(r'^(.+?)(\d+)(\.[A-Za-z]+)$')
    prefix, suffix = m.group('prefix'), m.group('suffix')
    return re.compile(r'^' + re.escape(prefix) + r'(\d+)' + re.escape(suffix) + r'$')

def _list_matching(out_dir: str, pattern: str):
    """
    Return dict: {filename -> index_int} for files matching `pattern`.
    """
    rx = _pattern_regex(pattern)
    out = {}
    try:
        with os.scandir(out_dir) as it:
            for e in it:
                if not e.is_file():
                    continue
                m = rx.match(e.name)
                if m:
                    out[e.name] = int(m.group(1))
    except FileNotFoundError:
        pass
    return out

def _next_start_number(out_dir: str, pattern: str) -> int:
    files = _list_matching(out_dir, pattern)
    return (max(files.values()) + 1) if files else 1

def run(input_video, out_dir, fps=2.0, qscale=1, pattern="frame_%06d.jpg",
        start=None, end=None, meta_path=None,
        allow_append_to_existing=False, allow_overwrite=False,
        use_to=False):
    """
    - If allow_append_to_existing=True, we continue numbering using -start_number <next>.
    - `end`: duration (seconds or HH:MM:SS). If use_to=True, pass as -to (relative to -ss).
    """
    prepare_output_dir(out_dir, allow_overwrite, allow_append_to_existing, step_name="extract")

    # Snapshot BEFORE
    before = _list_matching(out_dir, pattern)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", input_video]

    if end:
        if use_to:
            cmd += ["-to", str(end)]   # intended as relative to -ss usage in this pipeline
        else:
            cmd += ["-t", str(end)]

    cmd += ["-vf", f"fps={fps}", "-qscale:v", str(qscale)]

    start_num_used = None
    if allow_append_to_existing:
        start_num_used = _next_start_number(out_dir, pattern)
        cmd += ["-start_number", str(start_num_used)]

    cmd += [os.path.join(out_dir, pattern)]

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    print("command: ", cmd)
    subprocess.check_call(cmd)

    # Snapshot AFTER
    after = _list_matching(out_dir, pattern)

    # Newly created in THIS run
    new_files = set(after.keys()) - set(before.keys())
    new_indices = sorted(after[f] for f in new_files)

    num_new = len(new_files)
    first_new = new_indices[0] if new_indices else None
    last_new  = new_indices[-1] if new_indices else None

    if meta_path:
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        meta = {
            "input_video": os.path.abspath(input_video),
            "out_dir": os.path.abspath(out_dir),
            "fps": fps,
            "qscale": qscale,
            "pattern": pattern,
            "start": start,
            "end": end,
            "use_to": use_to,
            "allow_append_to_existing": allow_append_to_existing,
            "start_number_used": start_num_used,
            "num_frames": num_new,                 # <-- accurate for this run
            "first_new_index": first_new,
            "last_new_index": last_new,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote {num_new} new frames "
              f"(first={first_new}, last={last_new}); fps {fps}, video {input_video}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--qscale", type=int, default=1)
    ap.add_argument("--pattern", default="frame_%06d.jpg")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None,
                    help="Duration (e.g., 2 or 00:00:02). Use --use_to to pass as -to relative to -ss.")
    ap.add_argument("--meta_path", default=None)
    ap.add_argument("--allow_append_to_existing", action="store_true")
    ap.add_argument("--allow_overwrite", action="store_true")
    ap.add_argument("--use_to", action="store_true",
                    help="Use -to (relative to -ss) instead of -t.")
    args = ap.parse_args()

    run(args.input, args.out_dir, args.fps, args.qscale, args.pattern,
        args.start, args.end, args.meta_path,
        args.allow_append_to_existing, args.allow_overwrite, args.use_to)

if __name__ == "__main__":
    main()
