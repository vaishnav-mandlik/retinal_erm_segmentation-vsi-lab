# data_utils/resize_pairs.py
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from data_utils.file_utils import prepare_output_dir


def _read_mask(path: Path):
    """
    Read a Supervisely-like mask. If it's RGB/palette, convert to single-channel.
    We keep labels as uint8.
    """
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m


def _interp(is_mask: bool):
    return cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA


def _resize_letterbox(img: np.ndarray, target_hw, pad_value, is_mask=False):
    """
    Keep aspect ratio, scale uniformly, then pad to target_hw=(H,W).
    Returns: canvas, info dict
    """
    th, tw = map(int, target_hw)
    h, w = img.shape[:2]

    scale = min(tw / w, th / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=_interp(is_mask))

    # Build canvas
    if is_mask:
        # masks are 1-channel (uint8)
        canvas = np.zeros((th, tw), dtype=resized.dtype)
        if isinstance(pad_value, (tuple, list)):
            pad_value = 0
        canvas[:] = pad_value
    else:
        canvas = np.full((th, tw, 3), pad_value, dtype=resized.dtype)

    top = (th - new_h) // 2
    left = (tw - new_w) // 2

    if is_mask:
        canvas[top:top+new_h, left:left+new_w] = resized
    else:
        canvas[top:top+new_h, left:left+new_w, ...] = resized

    info = {
        "method": "letterbox",
        "scale": float(scale),
        "top": int(top),
        "left": int(left),
        "new_h": int(new_h),
        "new_w": int(new_w),
        "orig_h": int(h),
        "orig_w": int(w)
    }
    return canvas, info


def _resize_center_crop(img: np.ndarray, target_hw, is_mask=False):
    """
    Uniform scale so target is fully covered, then center-crop to target size.
    """
    th, tw = map(int, target_hw)
    h, w = img.shape[:2]

    scale = max(tw / w, th / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=_interp(is_mask))

    y0 = max(0, (new_h - th) // 2)
    x0 = max(0, (new_w - tw) // 2)
    y1 = y0 + th
    x1 = x0 + tw
    cropped = resized[y0:y1, x0:x1].copy()

    info = {
        "method": "center-crop",
        "scale": float(scale),
        "crop_y0": int(y0),
        "crop_x0": int(x0),
        "new_h": int(new_h),
        "new_w": int(new_w),
        "orig_h": int(h),
        "orig_w": int(w)
    }
    return cropped, info


def _ensure_dir(path: str | Path):
    os.makedirs(path, exist_ok=True)


def run(
    in_color_dir: str,
    in_mask_dir: str,
    out_color_dir: str,
    out_mask_dir: str,
    target_size: tuple[int, int] = (1024, 1024),   # (H, W)
    method: str = "letterbox",                     # "letterbox" or "center-crop"
    pad_value_rgb: tuple[int, int, int] = (0, 0, 0),
    pad_value_mask: int = 0,
    meta_path: str | None = None,
    allow_overwrite: bool = False,
    allow_append_to_existing = False
):
    """
    Resize color+mask pairs without distortion, keeping alignment.

    Args
    ----
    in_color_dir:  folder with input color frames
    in_mask_dir:   folder with input masks (same basenames as frames)
    out_color_dir: output folder for resized color frames
    out_mask_dir:  output folder for resized masks
    target_size:   (H,W) final size (default 1024x1024)
    method:        "letterbox" (pad) or "center-crop"
    pad_value_rgb: (B,G,R) pad value for images when letterboxing
    pad_value_mask: pad value for masks when letterboxing
    meta_path:     optional JSON with per-frame resize info
    allow_overwrite: clear output dirs first if True
    """
    # Prepare outputs
    prepare_output_dir(out_color_dir, allow_overwrite, allow_append_to_existing, step_name="resize_color")
    prepare_output_dir(out_mask_dir,  allow_overwrite, allow_append_to_existing, step_name="resize_mask")

    # Gather files by stem
    color_paths = {Path(p).stem: Path(p) for p in Path(in_color_dir).glob("*") if Path(p).suffix.lower() in (".jpg",".jpeg",".png")}
    mask_paths  = {Path(p).stem: Path(p) for p in Path(in_mask_dir).glob("*")  if Path(p).suffix.lower() in (".png",".tif",".tiff")}
    stems = sorted(set(color_paths) & set(mask_paths))

    if not stems:
        print("[resize_pairs] No matching color+mask stems found.")
        return

    per_frame = []
    for s in tqdm(stems, desc="resize_pairs"):
        cpath = color_paths[s]
        mpath = mask_paths[s]

        img = cv2.imread(str(cpath), cv2.IMREAD_COLOR)
        msk = _read_mask(mpath)
        if img is None or msk is None:
            continue

        # Apply same method to both to keep alignment
        if method.lower() == "center-crop":
            img_r, info_i = _resize_center_crop(img, target_size, is_mask=False)
            msk_r, info_m = _resize_center_crop(msk, target_size, is_mask=True)
        else:
            img_r, info_i = _resize_letterbox(img, target_size, pad_value_rgb, is_mask=False)
            msk_r, info_m = _resize_letterbox(msk, target_size, pad_value_mask, is_mask=True)

        # Sanity: ensure identical HxW
        if img_r.shape[:2] != msk_r.shape[:2]:
            # Force mask to match exactly (rare)
            msk_r = cv2.resize(msk_r, (img_r.shape[1], img_r.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Save
        cv2.imwrite(str(Path(out_color_dir) / cpath.name), img_r)
        cv2.imwrite(str(Path(out_mask_dir)  / mpath.name),  msk_r)

        # Record
        per_frame.append({
            "file": cpath.name,
            "mask": mpath.name,
            "target_h": int(target_size[0]),
            "target_w": int(target_size[1]),
            "image_info": info_i,
            "mask_info": info_m,
        })

    # Metadata
    if meta_path:
        _ensure_dir(Path(meta_path).parent)
        meta = {
            "in_color_dir": os.path.abspath(in_color_dir),
            "in_mask_dir":  os.path.abspath(in_mask_dir),
            "out_color_dir": os.path.abspath(out_color_dir),
            "out_mask_dir":  os.path.abspath(out_mask_dir),
            "target_size": [int(target_size[0]), int(target_size[1])],
            "method": method,
            "pad_value_rgb": list(pad_value_rgb),
            "pad_value_mask": int(pad_value_mask),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "frames": per_frame
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
