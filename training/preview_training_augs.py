# training/preview_training_augs.py
# Lock-step preview of the *exact* training augmentations.
# - Imports LightAugmentor from training/augments.py
# - Respects data selection toggles in config.yaml
# - Deterministic per-op tiles; one row per op: [image | mask]
# - Saves either one tall strip or a multi-row grid (configurable)


"""
# To run
 python pipeline.py --config config.yaml --cmd preview_augs

# This will write panels into:
<work_root>/<run_id>/training/aug_previews/

Augmentation notes
	•	Rotate +6° → Rotates the image and mask slightly (6 degrees clockwise). Helps the model handle small orientation variations.
	•	Scale ×1.03 → Zooms in by 3%. Simulates slight magnification changes from the surgical camera.
	•	Translate 3% → Shifts the image and mask by ~3% of width/height. Mimics small movements of the surgical field.
	•	Color Jitter → Randomly perturbs brightness, contrast, saturation, and hue. Improves robustness to lighting variations.
	•	H-Flip → Flips image and mask horizontally. Helps generalize across mirrored surgical views.
"""

import os, cv2, random
import numpy as np
from glob import glob

# === Import the SAME augmentor used in training ===
from training.augments import LightAugmentor

from data_utils.file_utils import prepare_output_dir

# ---------------- Config helpers ----------------


def _cfg(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return default if cur is None else cur

def _pv(d, k, default=None):
    return _cfg(d, "preview_augs", k, default=default)

# ---------------- Data roots --------------------

PREF_FRAME_DIRS = ["frames_resized", "frames_maskcrop", ]
PREF_MASK_DIRS  = ["masks_resized", "masks_multiclass_crop"]

def _case_dir(cfg, case_id):
    return os.path.join(cfg["work_root"], cfg["run_id"], case_id)

def _pick_dir(root, candidates):
    for d in candidates:
        p = os.path.join(root, d)
        if os.path.isdir(p) and any(os.scandir(p)):
            return p
    return None

def _choose_roots(cfg, case_root):
    prefer_resized = bool(_pv(cfg, "prefer_resized", True))
    use_maskcrop   = bool(_cfg(cfg, "train", "use_maskcrop", default = True))

    # frames root
    frame_order = list(PREF_FRAME_DIRS)
    if use_maskcrop:
        # bias maskcrop ahead of generic cropped/dedup/raw
        frame_order = ["frames_maskcrop"] + [d for d in frame_order if d != "frames_maskcrop"]
    if prefer_resized:
        frame_order = ["frames_resized"] + [d for d in frame_order if d != "frames_resized"]
    frames_root = _pick_dir(case_root, frame_order)

    # masks root
    mask_order = list(PREF_MASK_DIRS)
    if use_maskcrop:
        mask_order = ["masks_multiclass_crop"] + [d for d in mask_order if d != "masks_multiclass_crop"]
    if prefer_resized:
        mask_order = ["masks_resized"] + [d for d in mask_order if d != "masks_resized"]
    masks_root = _pick_dir(case_root, mask_order)

    return frames_root, masks_root

def _guess_mask_path_from_roots(p_img, frames_root, masks_root):
    if not masks_root:
        return None
    base = os.path.splitext(os.path.basename(p_img))[0]
    if base.startswith("dbg_"):   # tolerate debug prefix
        base = base[4:]
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(masks_root, base + ext)
        if os.path.isfile(p):
            return p
    return None

# ---------------- Resize / letterbox ------------

def _letterbox(img, dst_hw, pad_color):
    """Keep aspect ratio; pad to dst size."""
    H, W = img.shape[:2]
    th, tw = dst_hw
    scale = min(tw / W, th / H)
    nw, nh = int(round(W * scale)), int(round(H * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (th - nh) // 2; bottom = th - nh - top
    left = (tw - nw) // 2; right  = tw - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=pad_color)

def _ensure_size_pair(cfg, img, msk):
    """If we didn't use pre-resized folders, mimic training's letterbox to target_size."""
    prefer_resized = bool(_pv(cfg, "prefer_resized", True))
    if prefer_resized:
        return img, msk  # already aligned by *_resized folders
    target = _cfg(cfg, "resize", "target_size", default=None)
    th, tw = int(target[0]), int(target[1])
    img2 = _letterbox(img, (th, tw), (0,0,0))
    ms2  = None
    if msk is not None:
        if msk.ndim == 2:  # single-channel labels
            ms2 = _letterbox(msk, (th, tw), 0)
        else:
            # multi-channel (rare for masks), pad each channel then stack
            chans = []
            for c in range(msk.shape[2]):
                chans.append(_letterbox(msk[...,c], (th, tw), 0))
            ms_np = np.stack(chans, axis=2)
            ms2 = ms_np
    return img2, ms2

# --------------- Mask visualization -----------

_DEFAULT_CLASS_COLORS = {
    # id -> BGR
    0: (0, 0, 0),          # background
    1: (255, 255, 255),    # ERM (white)
    2: (255, 255, 255),    # tool A (white)
    3: (255, 255, 255),    # tool B (white)
}

def _class_colors_from_cfg(cfg):
    # config can optionally provide class_colors: {id: [B,G,R]}
    cc = _pv(cfg, "class_colors", None)
    if not cc:
        return dict(_DEFAULT_CLASS_COLORS)
    out = {}
    for k, v in cc.items():
        try:
            kid = int(k)
            out[kid] = tuple(int(x) for x in v)
        except Exception:
            pass
    return out or dict(_DEFAULT_CLASS_COLORS)


def _mask_vis(msk, cfg=None):
    """Turn a (H,W) or (H,W,1/3) mask into a 3-ch preview image.

    If binarize=True → Otsu threshold. If False → keep levels; normalize to 0..255 if possible.
    Optional dilation for thicker preview edges.
    """
    cfg = cfg or {}
    pv = cfg.get("preview_augs", {})
    mp = pv.get("mask_preview", {})
    binarize = bool(mp.get("binarize", True))
    dilate_px = int(mp.get("dilate_px", 0))

    if msk is None:
        return np.zeros((10, 10, 3), np.uint8)

    if msk.ndim == 3 and msk.shape[2] == 3:
        gray = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    else:
        gray = msk if msk.ndim == 2 else msk[..., 0]

    gray = gray.astype(np.uint8)

    if binarize:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # preserve multi-level info if any; normalize if range > 0
        gmin, gmax = int(gray.min()), int(gray.max())
        if gmax > gmin:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        # else keep as-is (could be all zeros for empty mask)

    if dilate_px > 0:
        gray = cv2.dilate(gray, np.ones((dilate_px, dilate_px), np.uint8), iterations=1)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# --------------- Labeling & layout -----------

def _put_label(img, text, y=18):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return img


def _row(img, msk, label, cfg=None):
    """Compose one labeled row: [image | mask] with a small label band on top."""
    cfg = cfg or {}
    def _cfg(d, *keys, default=None):
        cur = d
        for k in keys:
            if cur is None or not isinstance(cur, dict):
                return default
            cur = cur.get(k)
        return default if cur is None else cur

    band_h = int(_cfg(cfg, "preview_augs", "label_band_px", default=28))
    gap_px = int(_cfg(cfg, "preview_augs", "label_gap_px", default=6))

    # image tile with label
    img_tile = img.copy()
    if band_h > 0:
        band = np.zeros((band_h, img_tile.shape[1], 3), np.uint8)
        cv2.putText(band, label, (8, band_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        img_tile = cv2.vconcat([band, img_tile])

    # mask tile with label (visualized)
    msk_vis = _mask_vis(msk, cfg)
    if msk_vis.shape[:2] != img.shape[:2]:
        msk_vis = cv2.resize(msk_vis, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    if band_h > 0:
        band2 = np.zeros((band_h, msk_vis.shape[1], 3), np.uint8)
        cv2.putText(band2, f"{label} (mask)", (8, band_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        msk_tile = cv2.vconcat([band2, msk_vis])
    else:
        msk_tile = msk_vis

    # equalize heights, preserve aspect
    H = max(img_tile.shape[0], msk_tile.shape[0])
    if img_tile.shape[0] != H:
        img_tile = cv2.resize(img_tile, (int(img_tile.shape[1] * H / img_tile.shape[0]), H),
                              interpolation=cv2.INTER_LINEAR)
    if msk_tile.shape[0] != H:
        msk_tile = cv2.resize(msk_tile, (int(msk_tile.shape[1] * H / msk_tile.shape[0]), H),
                              interpolation=cv2.INTER_NEAREST)

    gap = np.zeros((H, gap_px, 3), np.uint8)
    return cv2.hconcat([img_tile, gap, msk_tile])

# def _row(img, msk, label, cfg):
#     li = _put_label(img.copy(), label)
#     lm = _put_label(_mask_vis(msk, cfg), f"{label} (mask)")
#     # ensure same height for concat
#     if li.shape[0] != lm.shape[0]:
#         h = li.shape[0]
#         lm = cv2.resize(lm, (int(lm.shape[1]*h/lm.shape[0]), h), interpolation=cv2.INTER_NEAREST)
#     return cv2.hconcat([li, lm])

def _to_grid(panels, cols, gap=8, bg=(0,0,0)):
    if not panels:
        return None
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    row_w = cols * w + (cols - 1) * gap
    rows = int(np.ceil(len(panels) / cols))
    grid_h = rows * h + (rows - 1) * gap
    canvas = np.zeros((grid_h, row_w, 3), np.uint8)
    canvas[:] = bg
    for i, p in enumerate(panels):
        r = i // cols; c = i % cols
        y = r * (h + gap)
        x = c * (w + gap)
        if p.shape[0] != h or p.shape[1] != w:
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_NEAREST)
        canvas[y:y+h, x:x+w] = p
    return canvas

# --------------- Build ops from config -------

def _ops_from_config(cfg):
    """Create one LightAugmentor per row so the code path matches training."""
    rows = _pv(cfg, "rows", None)
    aug_cfg = _cfg(cfg, "train", "augment", default={})
    ops = []
    _j = _cfg(cfg, "train", "augment", "jitter", default=None)
    if _j is None:
        _j = [0.1, 0.1, 0.1, 0.05] if _cfg(cfg, "train", "do_color_jitter", default=False) else [0.0, 0.0, 0.0, 0.0]

    if rows:
        # Each row is a dict: {name, op, ...params}
        for row in rows:

            name = row.get("name", row.get("op", "aug"))
            op   = row.get("op", "").lower()
            # Build a LightAugmentor with only that op enabled
            if op == "rotate":
                aug = LightAugmentor(target_size=tuple(_cfg(cfg, "resize", "target_size", default = None)),
                                     rotate_deg=float(row.get("deg", aug_cfg.get("rotate_deg", 0))),
                                     scale_range=(1.0, 1.0),
                                     translate_frac=0.0,
                                     jitter=tuple(_j),
                                     hflip_prob=0.0)
            elif op == "scale":
                s = float(row.get("scale", 1.0))
                aug = LightAugmentor(target_size=tuple(_cfg(cfg, "resize", "target_size", default = None)),
                                     rotate_deg=0.0,
                                     scale_range=(s, s),
                                     translate_frac=0.0,
                                     jitter=tuple(_j),
                                     hflip_prob=0.0)
            elif op == "translate":
                f = float(row.get("frac", 0.0))
                aug = LightAugmentor(target_size=tuple(_cfg(cfg, "resize", "target_size", default = None)),
                                     rotate_deg=0.0,
                                     scale_range=(1.0, 1.0),
                                     translate_frac=f,
                                     jitter=tuple(_j),
                                     hflip_prob=0.0)
            elif op == "jitter":
                # LightAugmentor reads its own jitter deltas from constructor if supported
                aug = LightAugmentor(target_size=tuple(_cfg(cfg, "resize", "target_size", default = None)),
                                     rotate_deg=0.0,
                                     scale_range=(1.0, 1.0),
                                     translate_frac=0.0,
                                     jitter=tuple(_j),
                                     hflip_prob=0.0)
            elif op == "hflip":
                aug = LightAugmentor(target_size=tuple(_cfg(cfg, "resize", "target_size", default = None)),
                                     rotate_deg=0.0,
                                     scale_range=(1.0, 1.0),
                                     translate_frac=0.0,
                                     jitter=tuple(_j),
                                     hflip_prob=1.0)  # force flip
            else:
                continue
            ops.append((name, aug))
        return ops

    # Fallback: derive rows from training augment config
    R = float(aug_cfg.get("rotate_deg", 0))
    S_lo, S_hi = [float(x) for x in aug_cfg.get("scale_range", (1.0, 1.0))]
    T = float(aug_cfg.get("translate_frac", 0.0))
    jitter = aug_cfg.get("jitter", [0.1, 0.1, 0.1, 0.05])
    ops = []
    if R > 0:
        ops.append((f"Rotate ±{int(R)}°", LightAugmentor(target_size=tuple(_cfg(cfg, "resize", "target_size", default = None)),
                                                        rotate_deg=R, scale_range=(1.0,1.0),
                                                        translate_frac=0.0, jitter=tuple(_j), hflip_prob=0.0)))
    if S_lo != 1.0 or S_hi != 1.0:
        mid = 0.5*(S_lo + S_hi)
        ops.append((f"Scale ×{mid:.2f}", LightAugmentor(target_size=tuple(_cfg(cfg, "resize", "target_size", default = None)),
                                                        rotate_deg=0.0, scale_range=(mid, mid),
                                                        translate_frac=0.0, jitter=tuple(_j), hflip_prob=0.0)))
    if T > 0:
        ops.append((f"Translate {int(100*T)}%", LightAugmentor(target_size=tuple(_cfg(cfg, "resize", "target_size", default = None)),
                                                                rotate_deg=0.0, scale_range=(1.0,1.0),
                                                                translate_frac=T, jitter=tuple(_j), hflip_prob=0.0)))
    ops.append(("Color Jitter", LightAugmentor(target_size=tuple(_cfg(cfg, "resize", "target_size", default = None)),
                                               rotate_deg=0.0, scale_range=(1.0,1.0),
                                               translate_frac=0.0, jitter=tuple(_j),
                                                hflip_prob=0.0)))
    if float(aug_cfg.get("hflip_prob", 0.0)) > 0.0:
        ops.append(("H-Flip", LightAugmentor(target_size=tuple(_cfg(cfg, "resize", "target_size", default = None)),
                                             rotate_deg=0.0, scale_range=(1.0,1.0),
                                             translate_frac=0.0, jitter=tuple(_j), hflip_prob=1.0)))
    return ops

# --------------- Core preview ------------------

def _uniform_size(imgs, size=None):
    """Resize all tiles to the same (h, w) to satisfy h/vconcat."""
    if not imgs:
        return imgs, (0, 0)
    if size is None:
        # pick the smallest tile to avoid upscaling
        h = min(im.shape[0] for im in imgs)
        w = min(im.shape[1] for im in imgs)
    else:
        h, w = size
    out = []
    for im in imgs:
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(im)
    return out, (h, w)

def _grid_from_tiles(tiles, rows, cols):
    """Build a rows×cols grid using hconcat/vconcat."""
    tiles, (h, w) = _uniform_size(tiles)
    # pad to fill grid
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((h, w, 3), np.uint8))
    lines = []
    for r in range(rows):
        line = cv2.hconcat(tiles[r*cols:(r+1)*cols])
        lines.append(line)
    return cv2.vconcat(lines)

def _panel_for(cfg, img, msk):
    """Build rows: one LightAugmentor per op (deterministic)."""
    panels = []

    # Header (Original)
    base_img, base_msk = _ensure_size_pair(cfg, img, msk)
    panels.append(_row(base_img, base_msk, "Original", cfg))

    # Deterministic sequence
    random.seed(123); np.random.seed(123)
    for name, aug in _ops_from_config(cfg):
        # LightAugmentor is expected to be mask-safe (INTER_NEAREST inside)
        ai, am = aug(base_img.copy(), None if base_msk is None else base_msk.copy())
        panels.append(_row(ai, am, name, cfg))

    # Compose either a vertical strip or a grid
    if _pv(cfg, "one_strip", False):
        panels.append(_row(base_img.copy(), None, None, cfg))
        strip = cv2.vconcat(panels)
        return strip

    # return _to_grid(panels, cols=cols, gap=gap)
    return _compose_panel_from_tiles(cfg, panels)

def _make_grid(tiles, rows=2, cols=3, gap=6, bg=(0,0,0)):
    """
    Arrange HxWx3 BGR tiles into a rows×cols grid with 'gap' pixels of spacing.
    Tiles can have different widths; heights will be normalized.
    """
    import numpy as np, cv2
    if not tiles:
        return None

    # Normalize all tiles to the max height (keep aspect ratio)
    H = max(t.shape[0] for t in tiles)
    norm = []
    for t in tiles:
        if t.shape[0] != H:
            new_w = int(round(t.shape[1] * (H / t.shape[0])))
            t = cv2.resize(t, (new_w, H), interpolation=cv2.INTER_NEAREST)
        norm.append(t)

    # Pad to rows*cols with blanks so vconcat works
    need = rows * cols
    if len(norm) < need:
        blank_w = max(t.shape[1] for t in norm)
        norm += [np.full((H, blank_w, 3), bg, np.uint8)] * (need - len(norm))
    norm = norm[:need]

    # Build each row with horizontal gaps
    hsp = np.full((H, gap, 3), bg, np.uint8)
    row_strips, max_row_w = [], 0
    for r in range(rows):
        seg = norm[r*cols:(r+1)*cols]
        strip = seg[0]
        for t in seg[1:]:
            strip = cv2.hconcat([strip, hsp, t])
        row_strips.append(strip)
        max_row_w = max(max_row_w, strip.shape[1])

    # Right-pad rows so vconcat is valid
    padded = []
    for s in row_strips:
        if s.shape[1] < max_row_w:
            pad = np.full((H, max_row_w - s.shape[1], 3), bg, np.uint8)
            s = cv2.hconcat([s, pad])
        padded.append(s)

    vsp = np.full((gap, max_row_w, 3), bg, np.uint8)
    panel = padded[0]
    for s in padded[1:]:
        panel = cv2.vconcat([panel, vsp, s])
    return panel


def _compose_panel_from_tiles(cfg, tiles):
    """Read grid settings from YAML and compose a single grid image."""
    pv   = cfg.get("preview_augs", {})
    grid = pv.get("grid", {})
    rows = int(grid.get("rows", 2))
    cols = int(grid.get("cols", 3))
    gap  = int(grid.get("gap_px", 6))
    return _make_grid(tiles, rows=rows, cols=cols, gap=gap)


# --------------- Public entry ------------------

def run(cfg, cases_sel):
    save_root = os.path.join(cfg["work_root"], cfg["run_id"], "training", "aug_previews")
    prepare_output_dir(save_root, allow_overwrite=True, step_name="preview_augs")
    max_images_per_case = _cfg(cfg, "max_images_per_case", default=6)
    for case in cfg["cases"]:
        case_id = case["case_id"]
        if cases_sel and case_id not in cases_sel:
            continue
        case_root = _case_dir(cfg, case_id)
        frames_root, masks_root = _choose_roots(cfg, case_root)
        if not frames_root:
            continue

        # Collect images
        imgs = sorted(glob(os.path.join(frames_root, "*.jpg")) + glob(os.path.join(frames_root, "*.png")))
        imgs = imgs[:max_images_per_case] if max_images_per_case else imgs

        for p_img in imgs:
            img = cv2.imread(p_img, cv2.IMREAD_COLOR)
            if img is None:
                continue
            p_msk = _guess_mask_path_from_roots(p_img, frames_root, masks_root)
            msk = cv2.imread(p_msk, cv2.IMREAD_UNCHANGED) if p_msk else None

            panel = _panel_for(cfg, img, msk)

            # Optional upscaling for readability
            scale = float(_pv(cfg, "scale_up", 1.0))
            if scale != 1.0:
                panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            base = os.path.splitext(os.path.basename(p_img))[0]
            out_path = os.path.join(save_root, f"{case_id}_{base}.png")
            cv2.imwrite(out_path, panel)
    print(f"Preview folder: {save_root}")