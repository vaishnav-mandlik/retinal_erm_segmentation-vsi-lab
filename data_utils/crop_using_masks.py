# data_utils/crop_using_masks.py
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from data_utils.file_utils import prepare_output_dir

"""
# PREFERRED Cropping : Crops driven by the Supervisely masks
# 
# This module crops retinal surgery frames to a region of interest using
# binary masks exported from Supervisely. Foreground regions are merged
# via morphological dilation, expanded by pixel + fractional margins, to form a safe,
# training-ready bounding box (per-video fixed ROI or per-frame adaptive).
#
#  OpenCV Documentation:
#    - Image morphology (dilation, erosion, structuring elements)
#      https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
#    - Contour-based object bounding boxes
#      https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
#
#
#  Common medical-image preprocessing practice:
#    - Use of binary masks to determine regions of interest and compute
#      bounding rectangles is a standard pipeline step in medical image
#      segmentation. Example references:
#         Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
#         Image Segmentation", MICCAI 2015.
#         https://arxiv.org/abs/1505.04597
"""


def _read_mask(path):
    # Read as grayscale; masks may be palette PNGs—IMREAD_UNCHANGED preserves labels
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        # If Supervisely exported RGB mask, convert to single channel as "any non-zero is FG"
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m

def safe_bbox_from_mask(
    mask,
    min_size=(512, 512),
    fallback_center=True,
    margin_px=150,
    margin_frac=0.06,              # NEW: add ~6% of width/height per side
    dilate_kernel=41,              # NEW: stronger dilation (odd number)
    union_with_color=None,         # NEW: pass color image to union with non-black retina
    retina_v_thresh=18,
    retina_rgb_thresh=18
):
    """
    Return a safe, exclusive bbox (x0,y0,x1,y1) around foreground.

    - Dilates mask aggressively to merge small/fragmented blobs
    - Adds both pixel and % margins
    - Enforces a minimum width/height
    - Optional: unions with non-black retina bbox from the color frame
    """
    h, w = mask.shape[:2]
    fg = (mask > 0).astype(np.uint8)

    # 1) Merge thin tools/small blobs
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    fg = cv2.dilate(fg, k, iterations=1)

    ys, xs = np.where(fg > 0)
    if xs.size == 0 or ys.size == 0:
        if fallback_center:
            cx, cy = w // 2, h // 2
            mw, mh = min_size
            x0 = max(0, cx - mw // 2)
            y0 = max(0, cy - mh // 2)
            x1 = min(w, cx + mw // 2)
            y1 = min(h, cy + mh // 2)
            return (x0, y0, x1, y1)
        return (0, 0, w, h)

    # 2) Base bbox from mask (exclusive)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    # 3) Add margins (pixels + fraction of size)
    add_x = int(margin_px + margin_frac * w)
    add_y = int(margin_px + margin_frac * h)
    top_bias = int(0.10 * h)  # add 10% of image height only on top

    x0 -= add_x
    x1 += add_x
    y0 -= (add_y + top_bias)  # single combined upward expansion
    y1 += add_y  # normal downward expansion

    # 4) Enforce minimum size (centered growth)
    bw, bh = x1 - x0, y1 - y0
    if bw < min_size[0]:
        delta = (min_size[0] - bw) // 2
        x0 -= delta; x1 += delta
    if bh < min_size[1]:
        delta = (min_size[1] - bh) // 2
        y0 -= delta; y1 += delta

    # 5) Optional: union with retina non-black bbox (makes edge membranes safer)
    if union_with_color is not None:
        rx0, ry0, rx1, ry1 = _retina_nonblack_bbox(
            union_with_color, v_thresh=retina_v_thresh, rgb_thresh=retina_rgb_thresh
        )
        x0 = min(x0, rx0); y0 = min(y0, ry0)
        x1 = max(x1, rx1); y1 = max(y1, ry1)

    # 6) Clamp to image
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return (int(x0), int(y0), int(x1), int(y1))


def _retina_nonblack_bbox(color_img, v_thresh=18, rgb_thresh=18, min_frac=0.01):
    """Bounding box of non-black region in the color frame (exclusive)."""
    h, w = color_img.shape[:2]
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    rgb_min = color_img.min(axis=2)
    m = ((v > v_thresh) & (rgb_min > rgb_thresh)).astype(np.uint8)

    ys, xs = np.where(m > 0)
    if xs.size == 0 or ys.size == 0 or m.mean() < min_frac:
        return (0, 0, w, h)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # exclusive bounds
    return (int(x0), int(y0), int(x1 + 1), int(y1 + 1))


def _collect_bboxes(mask_dir, sample_stride=1, margin=12):
    bboxes = []
    paths = sorted([p for p in Path(mask_dir).glob("*")
                    if p.suffix.lower() in [".png", ".tif", ".tiff"]])
    for i, p in enumerate(paths):
        if i % sample_stride != 0:
            continue
        m = _read_mask(p)
        if m is None:
            continue
        bb = safe_bbox_from_mask(m, margin_px=margin)   # <— no min_area_frac here
        if bb is not None:
            bboxes.append(bb)
    return bboxes

def _merge_bboxes(bboxes, image_size):
    """Return one bbox that covers all, clamped to image_size."""
    if not bboxes:
        return None
    h, w = image_size
    x0 = min(bb[0] for bb in bboxes)
    y0 = min(bb[1] for bb in bboxes)
    x1 = max(bb[2] for bb in bboxes)
    y1 = max(bb[3] for bb in bboxes)
    x0 = max(0, min(x0, w-1))
    y0 = max(0, min(y0, h-1))
    x1 = max(x0+1, min(x1, w))
    y1 = max(y0+1, min(y1, h))
    return (x0, y0, x1, y1)

def run(
    in_color_dir: str,
    in_mask_dir: str,
    out_color_dir: str,
    out_mask_dir: str,
    per_video_fixed_roi: bool = True,
    sample_stride: int = 1,
    margin: int = 12,                 # passed to safe_bbox_from_mask as margin_px
    min_size: tuple[int, int] = (768, 768),  # (W,H) passed to safe_bbox_from_mask
    margin_frac: float = 0.08,         # +8% of width/height on each side (scales with image)
    dilate_kernel: int =31,          # diameter of ellipse for dilation (larger = safer union)
    use_color_union: bool = False,            # union with retina non-black bbox
    retina_v_thresh: int = 18,                # V-channel threshold for non-black detection
    retina_rgb_thresh: int = 18,              # RGB min-channel threshold for non-black detection
    meta_path: str | None = None,
    allow_overwrite: bool = False,
    debug_overlays: bool = True,              # write dbg_<name>.jpg to out_color_dir
    per_frame_meta_path: str | None = None,    # write detailed per-frame JSON
        allow_append_to_existing=False

):
    """
    Crop color + mask pairs using mask foreground (safer version with debug).

    Outputs
    -------
    - Cropped color frames to `out_color_dir`
    - Cropped masks to `out_mask_dir`
    - Run-level metadata JSON to `meta_path` (if set)
    - Per-frame JSON to `per_frame_meta_path` (if set)
    - Debug overlays (rectangles on color frames) if `debug_overlays=True`
    """
    prepare_output_dir(out_color_dir, allow_overwrite, allow_append_to_existing, step_name="maskcrop_color")
    prepare_output_dir(out_mask_dir,  allow_overwrite, allow_append_to_existing, step_name="maskcrop_mask")

    color_paths = {p.name: p for p in Path(in_color_dir).glob("*")
                   if p.suffix.lower() in [".jpg", ".jpeg", ".png"]}
    mask_paths  = {p.name: p for p in Path(in_mask_dir).glob("*")
                   if p.suffix.lower() in [".png", ".tif", ".tiff"]}

    def basestem(path: str | Path) -> str:
        return Path(path).stem

    stem_to_color = {basestem(k): v for k, v in color_paths.items()}
    stem_to_mask  = {basestem(k): v for k, v in mask_paths.items()}
    common_stems  = sorted(set(stem_to_color) & set(stem_to_mask))

    fixed_bb = None
    mode = "per_frame"
    H = W = None  # will set once we load a frame
    per_frame_meta = []

    # ---- Fixed-ROI estimation (if enabled) ----
    if per_video_fixed_roi and common_stems:
        # Determine size from the first available image
        first_img = cv2.imread(str(stem_to_color[common_stems[0]]), cv2.IMREAD_COLOR)
        if first_img is None:
            raise RuntimeError(f"Cannot read first image to determine size: {stem_to_color[common_stems[0]]}")
        H, W = first_img.shape[:2]

        # Collect generous bboxes from masks and merge
        bboxes = _collect_bboxes(in_mask_dir, sample_stride=sample_stride, margin=margin)
        fixed_bb = _merge_bboxes(bboxes, image_size=(H, W))
        mode = "fixed_roi" if fixed_bb is not None else "fallback_per_frame"
        if fixed_bb is not None:
            print(f"[maskcrop] fixed ROI (exclusive): {fixed_bb}")

    kept = 0
    failures = 0

    # ---- Process each matched pair ----
    for s in tqdm(common_stems, desc="maskcrop"):
        cpath = stem_to_color[s]
        mpath = stem_to_mask[s]

        img  = cv2.imread(str(cpath), cv2.IMREAD_COLOR)
        mask = _read_mask(mpath)
        if img is None or mask is None:
            failures += 1
            continue

        H, W = img.shape[:2]  # update in case sizes vary (shouldn't, but safe)

        # Choose bbox
        bb = fixed_bb
        if bb is None:
            # per-image safe bbox driven by mask (with optional color union)
            bb = safe_bbox_from_mask(
                mask,
                min_size=min_size if min_size else (512, 512),
                margin_px=margin,
                margin_frac=(margin_frac if margin_frac is not None else 0.08),
                dilate_kernel=(dilate_kernel if dilate_kernel is not None else 41),
                union_with_color=(img if use_color_union else None),
                retina_v_thresh=retina_v_thresh,
                retina_rgb_thresh=retina_rgb_thresh
            )
            mode = "per_frame"

        # Clamp & crop (exclusive coords)
        x0, y0, x1, y1 = bb
        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        x1 = max(x0 + 1, min(x1, W))
        y1 = max(y0 + 1, min(y1, H))

        bw, bh = x1 - x0, y1 - y0

        # ---- DEBUG: print & overlay ----
        # print(f"[maskcrop] {cpath.name}: bbox=({x0},{y0},{x1},{y1}) size={bw}x{bh} "
        #       f"orig={W}x{H} mode={mode}")

        if debug_overlays:
            dbg = img.copy()
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.imwrite(str(Path(out_color_dir) / f"dbg_{Path(cpath).stem}.jpg"), dbg)

        # Save crops
        img_c  = img[y0:y1, x0:x1]
        mask_c = mask[y0:y1, x0:x1]
        cv2.imwrite(str(Path(out_color_dir) / cpath.name), img_c)
        cv2.imwrite(str(Path(out_mask_dir)  / mpath.name),  mask_c)
        kept += 1

        # Collect per-frame meta
        per_frame_meta.append({
            "file": cpath.name,
            "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
            "crop_w": int(bw), "crop_h": int(bh),
            "orig_w": int(W), "orig_h": int(H),
            "mode": mode
        })

    # ---- Write run-level meta ----
    if meta_path:
        os.makedirs(Path(meta_path).parent, exist_ok=True)
        mb = [int(v) for v in fixed_bb] if fixed_bb is not None else None
        meta = {
            "in_color_dir": os.path.abspath(in_color_dir),
            "in_mask_dir":  os.path.abspath(in_mask_dir),
            "out_color_dir": os.path.abspath(out_color_dir),
            "out_mask_dir":  os.path.abspath(out_mask_dir),
            "per_video_fixed_roi": bool(per_video_fixed_roi),
            "sample_stride": int(sample_stride),
            "margin_px": int(margin),
            "min_size": list(min_size) if min_size else [512, 512],
            "margin_frac": float(margin_frac if margin_frac is not None else 0.08),
            "dilate_kernel": int(dilate_kernel if dilate_kernel is not None else 41),
            "use_color_union": bool(use_color_union),
            "retina_v_thresh": int(retina_v_thresh),
            "retina_rgb_thresh": int(retina_rgb_thresh),
            "mode": mode,
            "fixed_bbox_exclusive": mb,
            "kept_pairs": int(kept),
            "failed_pairs": int(failures),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ---- Write per-frame meta (optional) ----
    if per_frame_meta_path:
        os.makedirs(Path(per_frame_meta_path).parent, exist_ok=True)
        with open(per_frame_meta_path, "w") as f:
            json.dump(per_frame_meta, f, indent=2)

