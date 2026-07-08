from data_utils.file_utils import prepare_output_dir

"""
Annotator preview transforms for retinal surgery frames (ERM context).

This module generates side-by-side preview tiles to help *humans* annotate
membrane vs. retina. The goal is *visual discrimination* (not model pre-processing):
boost the cues that make the ERM flap and tools easier to see.

Included techniques (ranked by practical usefulness for annotators):

1) **CLAHE (color-preserving)**
   - **What**: Local contrast enhancement with clipping to prevent noise blow-up.  
   - **How**: Convert `BGR → LAB`; apply CLAHE to L (lightness) channel; merge back to BGR.  
   - **Effect**: Improves separation of stained vs. unstained tissue; preserves global color balance.  
   - **Implementation Reference**: `cv2.createCLAHE`, `cv2.cvtColor` (LAB ↔ BGR).  
   - **Paper Reference**: Pizer et al., *Adaptive histogram equalization and its variations*, Computer Vision, Graphics, and Image Processing, 1987.  

2) **Gamma Correction**
   - **What**: Non-linear brightness adjustment.  
   - **How**: Apply pixel-wise mapping `I_out = I_in^(1/γ)` after normalizing to [0,1].  
   - **Effect**: Brightens faint regions or suppresses overexposed areas; useful to reveal subtle tissue.  
   - **Implementation Reference**: NumPy elementwise power transform.  
   - **Paper Reference**: Gonzalez & Woods, *Digital Image Processing*, 2002 (standard gamma correction).  

3) **Channel Composites (R+B, Separation)**
   - **What**: Combine or isolate selected color channels (e.g. R+B composite).  
   - **How**: Extract chosen channels from BGR; merge into 3-channel image, zeros elsewhere.  
   - **Effect**: Accentuates structures visible in specific channels (e.g. stained tissue in R or B).  
   - **Implementation Reference**: `cv2.split`, `cv2.merge`.  
   - **Paper Reference**: Same as above; common practice in biomedical preprocessing (see Otsu 1979 for channel thresholding context).  

4) **Scharr Operator (Edge Extraction)**
   - **What**: Gradient-based operator optimized for rotational symmetry.  
   - **How**: Apply Scharr kernels in X and Y; magnitude = √(Gx² + Gy²).  
   - **Effect**: Produces sharper, isotropic edges compared to Sobel; emphasizes fine boundaries.  
   - **Implementation Reference**: `cv2.Scharr`, `cv2.magnitude`.  
   - **Paper Reference**: Jaehne et al., *Handbook of Computer Vision and Applications*, 1999.  

5) **Kirsch Operator (Edge Extraction)**
   - **What**: Compass edge detector using 8 oriented kernels (N, NE, E, SE, S, SW, W, NW).  
   - **How**: Convolve image with all 8 Kirsch kernels; take maximum response per pixel. Optionally map edges back to thin color overlays.  
   - **Effect**: Strong directional edges; more sensitive to membrane boundaries and tools than Sobel/Scharr.  
   - **Implementation Reference**: Custom kernel implementation (not in OpenCV). Example: [EternityCode/kirsch-edge-detector](https://github.com/EternityCode/kirsch-edge-detector).  
   - **Paper Reference**: Kirsch, R. A., *Computer determination of the constituent structure of biological images*, Computers and Biomedical Research, 1971.  

6) Unsharp Masking
   - What: Edge-accentuating sharpening (high-boost).
   - How: Gaussian blur then weighted sum (orig * 1.5 − blur * 0.5).
   - Effect: Crisper instrument tips and membrane folds/edges.
   - OpenCV: GaussianBlur, addWeighted
   - Refs: Bovik, Handbook of Image & Video Processing (2000)

7) LAB-L (Lightness only)
   - What: Illumination/contrast view without hue.
   - How: BGR→LAB; keep L channel; expand to 3-ch for tiling.
   - Effect: Highlights intensity boundaries (flaps/edges) without color bias.
   - OpenCV: cvtColor

8) False-color (HSV)
   - What: Pseudo-coloring of a single channel (e.g., Blue) into HSV space.
   - How: Normalize channel→Value; fix Hue (~30°) & moderate Saturation.
   - Effect: Adds separable color contrast without hallucinating edges.
   - OpenCV: cvtColor HSV<->BGR

9) PCA PC1 (unsupervised channel mixing)
   - What: First principal component of RGB per frame.
   - How: OpenCV PCACompute/PCAProject; normalize result to 0..255.
   - Effect: Data-driven grayscale maximizing variance; sometimes separates dye vs. tissue.
   - OpenCV: cv2.PCACompute, cv2.PCAProject
   - Refs: Jolliffe, Principal Component Analysis.

Design notes
------------
• All outputs are converted to 3-channel uint8 BGR and resized to the original H×W
  before tiling → robust hconcat/vconcat with labels on top.
• These views are intended as *annotation aids*. Keep raw color frames as the
  source of truth for model training unless clinical sign-off decides otherwise.
• Toggle which tiles to show via `config.yaml -> previews.include` (list of names).

Key OpenCV doc links
--------------------
- Image I/O               : https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
- Hist/CLAHE              : https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html
- Filtering (Gaussian etc): https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- LUT                     : https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga56dcd90d3cb3d7ef12a10f6d88af3ac3
- Gabor                   : https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga90a168eb964f969e4ef58f220fbe89ea
- PCA                     : https://docs.opencv.org/4.x/d1/dee/classcv_1_1PCA.html
"""

import os
from glob import glob
from typing import List, Tuple, Dict, Callable

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------- utils -----------------------------------------

def _label(img, text: str, pos=(10, 24), color=(0, 255, 0)):
    """Draw label directly on the tile → no big blank header."""
    if img is None:
        return None
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2, cv2.LINE_AA)
    return img

def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = np.min(x), np.max(x)
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

# ----------------------- transforms (color-preserving) -----------------------

def t_original(img):
    return img

def t_clahe_color(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    out = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)
    return out

def t_unsharp(img, ksize=(9, 9), sigma=10.0, amount=1.5):
    blur = cv2.GaussianBlur(img, ksize, sigma)
    out = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    return out

def t_gamma(img, gamma=1.25):
    inv = 1.0 / float(max(gamma, 1e-6))
    lut = (np.linspace(0, 1, 256) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(img, lut)

def t_twoch_rb(img):
    # simple 2-channel composite (R+B emphasized)
    b, g, r = cv2.split(img)
    return cv2.merge([b, np.zeros_like(g), r])

def t_g_only(img):
    # preview only; keep 3-ch
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def t_b_only(img):
    b = img[:, :, 0]
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

def t_pca_pc1(img):
    # RGB → first principal component (unsupervised channel mix), shown as gray 3-ch
    X = img.reshape(-1, 3).astype(np.float32)
    X -= X.mean(0, keepdims=True)
    cov = X.T @ X / max(len(X) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, np.argmax(eigvals)]  # (3,)
    proj = X @ pc1
    proj = _norm01(proj).reshape(img.shape[:2])
    proj_u8 = (proj * 255).astype(np.uint8)
    return cv2.cvtColor(proj_u8, cv2.COLOR_GRAY2BGR)

# --------------------------- edge transforms ---------------------------------

def _inside_mask(img_bgr):
    """Mask away the dark border so normalization ignores it."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v   = hsv[...,2]
    m   = cv2.threshold(v, 12, 255, cv2.THRESH_BINARY)[1]     # keep bright-ish
    m   = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    m   = cv2.morphologyEx(m, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))
    return m

def apply_scharr_mag(img_bgr):
    """
    Scharr edge magnitude with border suppression and percentile scaling.
    Produces a 3-channel grayscale BGR image with good contrast.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)                    # tame sensor noise

    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)                                # float32

    mask = _inside_mask(img_bgr).astype(bool)
    if mask.any():
        vals = mag[mask]
        lo, hi = np.percentile(vals, [2, 98])                  # robust window
        if hi > lo:
            mag = (mag - lo) / (hi - lo)
        else:
            mag = mag / (hi + 1e-6)
    else:
        mag = mag / (mag.max() + 1e-6)

    mag = np.clip(mag, 0, 1) ** 0.7                            # gentle gamma
    mag_u8 = (mag * 255).astype(np.uint8)
    return cv2.cvtColor(mag_u8, cv2.COLOR_GRAY2BGR)

def apply_kirsch_color_thin(img,
                            canny_lo=100, canny_hi=280,      # stricter edges
                            edge_erode=1,                    # 0..2 → thinner edges
                            sat=220,                         # color saturation on edges
                            alpha=0.30,                      # how strong the overlay looks
                            p_lo=10, p_hi=99.5):             # percentile norm inside retina
    """
    Colorized *thin* Kirsch edges overlaid on the original image.
    Tweaks to stop interior fill:
      - edges come only from Canny (strict thresholds)
      - single-pixel thinning via erosion
      - HSV V,S set to zero outside edges
    """

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 7)                 # denoise but keep edges
    gray = cv2.createCLAHE(2.0, (8,8)).apply(gray)

    # mask of the circular retina to keep normalization stable
    mask_ret = _detect_retina_mask(gray)

    # --- Kirsch responses (for orientation & magnitude coloring) ---
    K = [
        np.array([[-3,-3, 5],[-3, 0, 5],[-3,-3, 5]], np.float32),
        np.array([[-3, 5, 5],[-3, 0, 5],[-3,-3,-3]], np.float32),
        np.array([[ 5, 5, 5],[-3, 0,-3],[-3,-3,-3]], np.float32),
        np.array([[ 5, 5,-3],[ 5, 0,-3],[-3,-3,-3]], np.float32),
        np.array([[ 5,-3,-3],[ 5, 0,-3],[ 5,-3,-3]], np.float32),
        np.array([[-3,-3,-3],[ 5, 0,-3],[ 5, 5,-3]], np.float32),
        np.array([[-3,-3,-3],[-3, 0,-3],[ 5, 5, 5]], np.float32),
        np.array([[-3,-3,-3],[-3, 0, 5],[-3, 5, 5]], np.float32),
    ]
    f32  = gray.astype(np.float32)
    resp = [np.abs(cv2.filter2D(f32, cv2.CV_32F, k)) for k in K]
    resp = np.stack(resp, axis=-1)                   # (H,W,8)
    mag  = resp.max(axis=-1)                         # Kirsch magnitude
    dire = resp.argmax(axis=-1).astype(np.uint8)     # orientation 0..7

    # normalize mag inside retina only (avoid dark ring dominating)
    mag  = _norm_percentile(mag, mask_ret, p=(p_lo, p_hi))

    # --- build a THIN binary edge mask (no interiors) ---
    # use Canny to produce a 1px-ish ridge; restrict to retina
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    edges = cv2.bitwise_and(edges, edges, mask=mask_ret)

    if edge_erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_erode*2+1, edge_erode*2+1))
        # erode to *thin*, not to grow
        edges = cv2.erode(edges, k, iterations=1)

    # keep only top magnitudes along edges (optional refinement)
    # this trims weak, wide edge bands:
    mag_thresh = np.percentile(mag[edges > 0], 90) if np.any(edges) else 255
    edges = np.where((edges > 0) & (mag >= mag_thresh), 255, 0).astype(np.uint8)

    # --- color only the edge pixels ---
    hue_lut = (np.array([0, 20, 40, 70, 100, 130, 150, 170]) % 180).astype(np.uint8)
    hue = hue_lut[dire]
    val = mag.astype(np.uint8)

    # zero hue/value everywhere except edges (prevents interior fill)
    hue[edges == 0] = 0
    val[edges == 0] = 0
    hsv_edges = np.dstack([hue, np.full_like(hue, sat, np.uint8), val])
    color_edges = cv2.cvtColor(hsv_edges, cv2.COLOR_HSV2BGR)

    # overlay: only mix where edges==1 (alpha elsewhere = 0)
    m3 = (edges > 0).astype(np.uint8)
    m3 = np.repeat(m3[..., None], 3, axis=2)                 # (H,W,3)
    out = img.astype(np.float32)*(1 - alpha*m3) + color_edges.astype(np.float32)*(alpha*m3)
    return np.clip(out, 0, 255).astype(np.uint8)

def _detect_retina_mask(gray_or_bgr, thr=12):
    """Return 2-D uint8 mask: 255 inside retina, 0 elsewhere."""
    g = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY) if gray_or_bgr.ndim==3 else gray_or_bgr
    g = cv2.GaussianBlur(g, (11,11), 0)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    _, binm = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.full_like(g, 255, np.uint8)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(g, np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    return mask

def _norm_percentile(x, mask_u8=None, p=(1,99)):
    """Percentile normalization; if mask given, compute stats inside mask."""
    x = x.astype(np.float32)
    if mask_u8 is not None:
        vv = x[mask_u8==255]
        if vv.size:
            lo, hi = np.percentile(vv, list(p))
        else:
            lo, hi = np.percentile(x, list(p))
    else:
        lo, hi = np.percentile(x, list(p))
    if hi <= lo:
        return np.clip(x, 0, 255).astype(np.uint8)
    y = (x - lo) / (hi - lo)
    return (np.clip(y, 0, 1) * 255).astype(np.uint8)


# --- drop-in: smart auto-crop for round retina + letterbox back ---
def _autocrop_retina_letterbox(img, thr=12, margin=50):
    """
    Preview-only:
      1) detect bright retina area (robust to vignetting),
      2) crop to its bounding rect (+ margin),
      3) letterbox back to original size.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # normalize a bit, then threshold low intensities (keep brighter disc)
    g = cv2.GaussianBlur(gray, (11,11), 0)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    _, binm = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
    # keep the largest component
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img  # nothing detected; keep original
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    # expand a little
    x = max(0, x - margin); y = max(0, y - margin)
    w = min(W - x, w + 2*margin); h = min(H - y, h + 2*margin)
    crop = img[y:y+h, x:x+w]

    # letterbox back to (W,H) preserving aspect
    ch, cw = crop.shape[:2]
    scale = min(W / cw, H / ch)
    nw, nh = int(cw*scale), int(ch*scale)
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((H, W, 3), np.uint8)
    ox = (W - nw)//2; oy = (H - nh)//2
    canvas[oy:oy+nh, ox:ox+nw] = resized
    return canvas


def _letterbox(img, target_hw, pad_color=(0, 0, 0)):
    """Aspect-ratio preserving resize with padding to exactly (H,W)."""
    H, W = target_hw
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    scale = min(W / w, H / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((H, W, 3), pad_color, dtype=np.uint8)
    y0 = (H - nh) // 2
    x0 = (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

def _to_tile(img, target_hw):
    """Force BGR uint8, then letterbox to target tile size."""
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    return _letterbox(img, target_hw, pad_color=(0, 0, 0))

def _kirsch_responses(gray_f32: np.ndarray):
    # 8 Kirsch kernels (same as the repo)
    K = [
        np.array([[-3,-3, 5],[-3, 0, 5],[-3,-3, 5]], np.float32),  # 0  (east)
        np.array([[-3, 5, 5],[-3, 0, 5],[-3,-3,-3]], np.float32),  # 1
        np.array([[ 5, 5, 5],[-3, 0,-3],[-3,-3,-3]], np.float32),  # 2  (north)
        np.array([[ 5, 5,-3],[ 5, 0,-3],[-3,-3,-3]], np.float32),  # 3
        np.array([[ 5,-3,-3],[ 5, 0,-3],[ 5,-3,-3]], np.float32),  # 4  (west)
        np.array([[-3,-3,-3],[ 5, 0,-3],[ 5, 5,-3]], np.float32),  # 5
        np.array([[-3,-3,-3],[-3, 0,-3],[ 5, 5, 5]], np.float32),  # 6  (south)
        np.array([[-3,-3,-3],[-3, 0, 5],[-3, 5, 5]], np.float32),  # 7
    ]
    resp = [cv2.filter2D(gray_f32, cv2.CV_32F, k) for k in K]
    resp = np.stack(resp, axis=-1)             # (H,W,8)
    mag  = np.max(resp, axis=-1)               # magnitude
    ori  = np.argmax(resp, axis=-1).astype(np.uint8)  # 0..7
    mag  = np.abs(mag)
    return mag, ori

def _retina_mask(gray_u8: np.ndarray):
    """Loose circular mask to ignore black borders/rings."""
    blur = cv2.GaussianBlur(gray_u8, (9,9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=1)
    return th

def _percentile_norm(x: np.ndarray, mask: np.ndarray, lo=5, hi=99.5):
    """Normalize by percentiles inside mask to avoid halo dominance."""
    if mask is None or not np.any(mask):
        lo_v, hi_v = np.percentile(x, [lo, hi])
    else:
        sel = x[mask>0]
        if sel.size < 10:
            lo_v, hi_v = np.percentile(x, [lo, hi])
        else:
            lo_v, hi_v = np.percentile(sel, [lo, hi])
    hi_v = max(hi_v, lo_v+1e-6)
    y = (x - lo_v) / (hi_v - lo_v)
    return np.clip(y, 0, 1)

def apply_kirsch_mag(img_bgr: np.ndarray,
                     canny=(120, 220),
                     thin_erode=1,
                     pnorm=(5, 99.5)) -> np.ndarray:
    """
    Grayscale Kirsch magnitude with *thin* edges.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # light denoise to stabilize kernels
    gray = cv2.bilateralFilter(gray, 7, 50, 7)
    mask_ret = _retina_mask(gray)

    mag, _ = _kirsch_responses(gray.astype(np.float32))
    mag = _percentile_norm(mag, mask_ret, lo=pnorm[0], hi=pnorm[1])
    mag_u8 = (mag*255).astype(np.uint8)

    # thin edges via Canny mask
    e = cv2.Canny(gray, canny[0], canny[1])
    e = cv2.bitwise_and(e, e, mask=mask_ret)
    if thin_erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2*thin_erode+1, 2*thin_erode+1))
        e = cv2.erode(e, k, iterations=1)

    out = np.zeros((h, w), np.uint8)
    out[e>0] = mag_u8[e>0]
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)



# Name → (callable, pretty label)
TRANSFORMS: Dict[str, Tuple[Callable, str]] = {
    "Original":         (t_original,       "Original"),
    "CLAHE":            (t_clahe_color,    "CLAHE"),
    "Unsharp":          (t_unsharp,        "Unsharp"),
    "Gamma correct":    (t_gamma,          "Gamma"),
    "2-Ch (R+B)":       (t_twoch_rb,       "2-Ch (R+B)"),
    "G-only":           (t_g_only,         "G-only"),
    "B-only":           (t_b_only,         "B-only"),
    "PCA PC1":          (t_pca_pc1,        "PCA PC1"),
    "Scharr (mag)":     (apply_scharr_mag,     "Scharr (mag)"),
    "Kirsch (col)":     (apply_kirsch_color_thin,     "Kirsch (col)"),
}

# ----------------------------- main API --------------------------------------

def save_previews(
    in_dir: str,
    out_dir: str,
    include: List[str],
    allow_overwrite: bool = False,
    cols: int = 4,
    label_color=(0, 255, 0),
    auto_crop=True,
    allow_append_to_existing = False
):
    """
    Build a consistent grid panel for each input frame using the transforms
    listed in `include`. Unknown names are skipped.
    """
    prepare_output_dir(out_dir, allow_overwrite, allow_append_to_existing, step_name="annotator_previews")

    # gather frames
    frames = sorted([p for p in glob(os.path.join(in_dir, "*.jpg")) +
                           glob(os.path.join(in_dir, "*.png"))])
    if not frames:
        print(f"[annotator_previews] No images in {in_dir}")
        return

    # load one to determine target tile size
    probe = cv2.imread(frames[0])
    if probe is None:
        print(f"[annotator_previews] Could not read any image in {in_dir}")
        return
    base_h, base_w = probe.shape[:2]

    # build list of (name, fn) to apply in order
    ops = [(name, TRANSFORMS[name][0], TRANSFORMS[name][1])
           for name in include if name in TRANSFORMS]

    for fp in tqdm(frames, desc="annotator_previews"):
        img0 = cv2.imread(fp)
        if img0 is None:
            continue
        if auto_crop:
            img0 = _autocrop_retina_letterbox(img0)
        tiles = []
        for name, fn, pretty in ops:
            try:
                out = fn(img0.copy())
            except Exception as e:
                print(f"[annotator_previews] {name} failed on {os.path.basename(fp)}: {e}")
                out = np.zeros_like(img0)
            tile = _to_tile(out, (base_h, base_w))
            tile = _label(tile, pretty, pos=(20, 80), color=label_color)
            tiles.append(tile)

        if not tiles:
            continue

        # pad last row
        while len(tiles) % cols != 0:
            tiles.append(np.zeros((base_h, base_w, 3), dtype=np.uint8))

        # concat grid
        rows = []
        for i in range(0, len(tiles), cols):
            rows.append(cv2.hconcat(tiles[i:i+cols]))
        panel = cv2.vconcat(rows)

        out_path = os.path.join(out_dir, os.path.basename(fp))
        cv2.imwrite(out_path, panel)



def run_from_config(case_root: str, cfg: dict):
    """
    Picks the best available frames dir under the case and writes panels into
    <case_root>/annotator_previews/.
    """
    for d in ("frames_dedup", "frames_raw", "frames_maskcrop", "frames_cropped"):
        in_dir = os.path.join(case_root, d)
        if os.path.isdir(in_dir) and any(os.scandir(in_dir)):
            break
    else:
        print(f"[annotator_previews] No frames_* folder found under {case_root}")
        return

    out_dir = os.path.join(case_root, "annotator_previews")
    prv_cfg = cfg.get("previews", {})
    include = prv_cfg.get("include", ["Original", "CLAHE", "Unsharp", "Gamma correct"])
    cols = int(prv_cfg.get("cols", 4))
    allow_overwrite = bool(cfg.get("allow_overwrite", False))
    auto_crop = cfg.get("previews", {}).get("auto_crop", True)
    save_previews(
        in_dir=in_dir,
        out_dir=out_dir,
        include=include,
        allow_overwrite=allow_overwrite,
        cols=cols,
        label_color=(0, 255, 0),
        auto_crop=auto_crop,

    )
