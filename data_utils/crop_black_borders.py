import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from data_utils.file_utils import prepare_output_dir

"""
Preferred : Crop using masks as this is more accurate and easier to remove noise.

This file --> Cropping Driven from the Color Raw images : Prior to Supervisely annotation.



Crop black borders:
- HSV thresholding for non-black detection: uses the Value (V) channel to build a binary mask of “bright” pixels.
- Connected-component bounding box: find the largest connected component of the binary mask and compute the minimal enclosing rectangle.

- OpenCV tutorials – Color spaces and contour/connected component analysis
  https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
  https://docs.opencv.org/4.x/d3/d05/tutorial_py_table_of_contents_contours.html

"""

DEFAULT_BLACK_V = 18  # HSV V threshold (0-255)
DEFAULT_BLACK_RGB = 18  # min(R,G,B) threshold
MIN_NONBLACK_FRACTION = 0.02  # ignore tiny specks
COVERAGE_WARN = 0.95  # >95% of image ⇒ probably no crop


def iter_images(in_dir):
    for p in sorted(Path(in_dir).glob("*")):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            yield p


def non_black_mask(img, v_thresh=DEFAULT_BLACK_V, rgb_thresh=DEFAULT_BLACK_RGB):
    """Return binary mask of non-black pixels using HSV and RGB minima."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    rgb_min = img.min(axis=2)
    m = ((v > v_thresh) & (rgb_min > rgb_thresh)).astype(np.uint8)

    # remove tiny noise and fill small holes
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m


def bbox_by_black_bands(img, v_thresh=DEFAULT_BLACK_V, rgb_thresh=DEFAULT_BLACK_RGB, margin=0):
    """Trim solid dark bands from all sides by scanning columns/rows."""
    h, w = img.shape[:2]
    m = non_black_mask(img, v_thresh, rgb_thresh)

    col_nonblack = m.sum(axis=0) / float(h)
    row_nonblack = m.sum(axis=1) / float(w)

    # find first/last columns/rows where non-black fraction exceeds small cutoff
    cutoff = MIN_NONBLACK_FRACTION
    xs = np.where(col_nonblack > cutoff)[0]
    ys = np.where(row_nonblack > cutoff)[0]
    if xs.size == 0 or ys.size == 0:
        return (0, 0, w, h)  # give up

    x0, x1 = int(max(xs.min() - margin, 0)), int(min(xs.max() + 1 + margin, w))
    y0, y1 = int(max(ys.min() - margin, 0)), int(min(ys.max() + 1 + margin, h))
    return (x0, y0, x1, y1)


def per_video_roi(in_dir, sample_stride=10, margin=24):
    mins = [];
    maxs = []
    paths = list(iter_images(in_dir))
    if not paths:
        return None
    for idx, p in enumerate(paths):
        if idx % sample_stride != 0:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        m = non_black_mask(img)
        if m.mean() < MIN_NONBLACK_FRACTION:
            continue
        ys, xs = np.where(m > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        mins.append((x0, y0));
        maxs.append((x1, y1))
    if not mins:
        return None

    x0 = max(0, min([m[0] for m in mins]) - margin)
    y0 = max(0, min([m[1] for m in mins]) - margin)
    x1 = max([m[0] for m in maxs]) + margin
    y1 = max([m[1] for m in maxs]) + margin
    return (int(x0), int(y0), int(x1), int(y1))


def crop_and_write(img, bbox):
    x0, y0, x1, y1 = bbox
    h, w = img.shape[:2]
    x0 = max(0, min(x0, w - 1));
    x1 = max(1, min(x1, w))
    y0 = max(0, min(y0, h - 1));
    y1 = max(1, min(y1, h))
    return img[y0:y1, x0:x1]


def run(in_dir: str,
        out_dir: str,
        method: str = "largest_component",
        roi_margin_px: int = 24,
        sample_stride: int = 10,
        per_video_fixed_roi: bool = True,
        meta_path: str | None = None,
        allow_overwrite: bool = False,
        allow_append_to_existing=False
        ):
    """
    Detect and crop black borders from retinal surgery frames.

    This function scans all images in `in_dir`, determines the non-black
    region of interest (ROI), and writes cropped images to `out_dir`.
    Optionally logs crop statistics to a metadata JSON.

    Parameters
    ----------
    in_dir : str
        Directory of input frames to crop. All readable image files
        (e.g. .jpg, .png) will be processed.
    out_dir : str
        Destination directory for cropped frames. Will be created if it
        does not exist. If it exists:
          * If `allow_overwrite` is True, it will be cleared first.
          * If `allow_overwrite` is False and not empty, a RuntimeError
            is raised.
    method : str, default "largest_component"
        Cropping strategy:
          * "largest_component" – Convert each image to HSV, threshold on
            V channel to create a binary mask, then take the bounding box
            of the largest connected component.
          * (future) other methods such as "hough_circle" can be added.
    roi_margin_px : int, default 24
        Extra pixels of margin to include around the detected ROI on all
        sides to avoid tight crops.
    sample_stride : int, default 10
        For `per_video_fixed_roi=True`, analyze every Nth frame when
        computing a global ROI. Ignored if `per_video_fixed_roi=False`.
    per_video_fixed_roi : bool, default True
        Whether to compute a single fixed ROI per video (using sampled
        frames) and apply it to all frames. If False, the ROI is
        recomputed for each frame (slower but adapts to camera movement).
    meta_path : str or None, default None
        Optional path to a JSON file where crop metadata (ROI coordinates,
        input/output counts, etc.) will be written. If None, no file is
        written.
    allow_overwrite : bool, default False
        If True, removes `out_dir` and recreates it before writing cropped
        frames. If False, raises RuntimeError if `out_dir` exists and is
        non-empty.

    Returns
    -------
    None
        The function writes cropped images to `out_dir` and optionally a
        JSON metadata file to `meta_path`. Nothing is returned.

    Also
    ------------
    - Creates or clears the output directory.
    - Prints progress and summary statistics to stdout.

    Notes
    -----
    * The function uses :func:`data_utils.path_utils.prepare_output_dir`
      to safely create or clear the output directory.
    * Cropping is done with OpenCV (HSV threshold + connected components)
      for robust detection of non-black retinal areas.
    """
    print("THIS API IS DEPRECATED")

    prepare_output_dir(out_dir, allow_overwrite, allow_append_to_existing,step_name="crop")

    bbox = None
    kept = 0
    used_mode = "fixed_roi" if per_video_fixed_roi else "per_frame"
    fixed_bbox = None

    if per_video_fixed_roi:
        fixed_bbox = per_video_roi(in_dir, sample_stride=sample_stride, margin=roi_margin_px)

    for p in tqdm(list(iter_images(in_dir)), desc="crop"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        bb = fixed_bbox
        if bb is None:
            # try mask-based bbox
            m = non_black_mask(img)
            if m.mean() <= MIN_NONBLACK_FRACTION:
                # fallback to bands if mask is too small
                bb = bbox_by_black_bands(img, margin=roi_margin_px)
            else:
                ys, xs = np.where(m > 0)
                x0, x1 = xs.min(), xs.max()
                y0, y1 = ys.min(), ys.max()
                bb = (int(max(0, x0 - roi_margin_px)),
                      int(max(0, y0 - roi_margin_px)),
                      int(min(w, x1 + roi_margin_px)),
                      int(min(h, y1 + roi_margin_px)))
            used_mode = "per_frame"
        else:
            # Sanity-check: if bbox covers ~entire frame, try band-trim
            fx0, fy0, fx1, fy1 = fixed_bbox
            cov_w = (fx1 - fx0) / float(w)
            cov_h = (fy1 - fy0) / float(h)
            if cov_w >= COVERAGE_WARN and cov_h >= COVERAGE_WARN:
                bb = bbox_by_black_bands(img, margin=roi_margin_px)
                used_mode = "band_fallback"

        # Clamp and crop
        x0, y0, x1, y1 = bb
        x0 = int(max(0, min(x0, w - 1)));
        x1 = int(max(1, min(x1, w)))
        y0 = int(max(0, min(y0, h - 1)));
        y1 = int(max(1, min(y1, h)))
        cropped = img[y0:y1, x0:x1]
        cv2.imwrite(os.path.join(out_dir, p.name), cropped)
        kept += 1

    # ---- metadata (cast to plain ints for JSON) ----
    if meta_path:
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        meta_bbox = None
        if fixed_bbox is not None:
            meta_bbox = [int(fixed_bbox[0]), int(fixed_bbox[1]),
                         int(fixed_bbox[2]), int(fixed_bbox[3])]
        meta = {
            "in_dir": os.path.abspath(in_dir),
            "out_dir": os.path.abspath(out_dir),
            "method": method,
            "roi_margin_px": int(roi_margin_px),
            "sample_stride": int(sample_stride),
            "per_video_fixed_roi": bool(per_video_fixed_roi),
            "bbox_used": meta_bbox,
            "mode": used_mode,
            "kept": int(kept),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print("Crop: kept {}; bbox_used {}".format(kept, meta['bbox_used']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--method", default="largest_component")
    ap.add_argument("--roi_margin_px", type=int, default=24)
    ap.add_argument("--sample_stride", type=int, default=10)
    ap.add_argument("--per_video_fixed_roi", type=bool, default=True)
    ap.add_argument("--meta_path", default=None)
    args = ap.parse_args()
    run(args.in_dir, args.out_dir, args.method, args.roi_margin_px, args.sample_stride, args.per_video_fixed_roi,
        args.meta_path)


if __name__ == "__main__":
    main()

