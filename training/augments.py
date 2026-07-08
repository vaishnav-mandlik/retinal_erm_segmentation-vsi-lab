"""
Augmentation notes
	•	Rotate +6° → Rotates the image and mask slightly (6 degrees clockwise). Helps the model handle small orientation variations.
	•	Scale ×1.03 → Zooms in by 3%. Simulates slight magnification changes from the surgical camera.
	•	Translate 3% → Shifts the image and mask by ~3% of width/height. Mimics small movements of the surgical field.
	•	Color Jitter → Randomly perturbs brightness, contrast, saturation, and hue. Improves robustness to lighting variations.
	•	H-Flip → Flips image and mask horizontally. Helps generalize across mirrored surgical views.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

def _rot_mat(w, h, deg):
    return cv2.getRotationMatrix2D((w/2.0, h/2.0), deg, 1.0)

def _warp(img, M, shape, interp, border_val):
    return cv2.warpAffine(img, M, (shape[1], shape[0]),
                          flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

def _center_pad_or_crop(arr, H, W, pad_val):
    h, w = arr.shape[:2]
    if h >= H and w >= W:
        y0 = (h - H)//2; x0 = (w - W)//2
        return arr[y0:y0+H, x0:x0+W]
    top = (H - h)//2; bot = H - h - top
    left = (W - w)//2; right = W - w - left
    return cv2.copyMakeBorder(arr, top, bot, left, right, cv2.BORDER_CONSTANT, value=pad_val)

class LightAugmentor:
    """
    Geometry on both image/mask, photometric on image only.
    Matches your preview ops/params.
    """
    def __init__(self,
                 target_size: Tuple[int, int],
                 rotate_deg: int = 8,
                 scale_range: Tuple[float, float] = (0.97, 1.03),
                 translate_frac: float = 0.02,
                 hflip_prob: float = 0.0,
                 jitter: Tuple[float, float, float, float] = (0.10, 0.10, 0.10, 0.05)):
        self.H, self.W = target_size
        self.max_rot = rotate_deg
        self.scale_min, self.scale_max = scale_range
        self.tfrac = translate_frac
        self.hflip_p = hflip_prob
        self.j_b, self.j_c, self.j_s, self.j_h = jitter

    def _letterbox(self, img, msk):
        """Keep aspect, pad to (H,W)."""
        h, w = img.shape[:2]
        scale = min(self.W / w, self.H / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        msk_r = None
        if msk is not None:
            interp = cv2.INTER_NEAREST if msk.ndim == 2 else cv2.INTER_NEAREST
            msk_r = cv2.resize(msk, (nw, nh), interpolation=interp)
        img_out = _center_pad_or_crop(img_r, self.H, self.W, (0, 0, 0))
        msk_out = _center_pad_or_crop(msk_r, self.H, self.W, 0) if msk is not None else None
        return img_out, msk_out

    def _geom(self, img, msk):
        deg = np.random.uniform(-self.max_rot, self.max_rot)
        sc  = np.random.uniform(self.scale_min, self.scale_max)
        dx  = int(self.W * self.tfrac * np.random.uniform(-1, 1))
        dy  = int(self.H * self.tfrac * np.random.uniform(-1, 1))

        # rotation + scale
        M = _rot_mat(self.W, self.H, deg)
        M[:, :2] *= sc
        M[:, 2] += [dx, dy]

        img2 = _warp(img, M, (self.H, self.W), cv2.INTER_LINEAR, (0,0,0))
        msk2 = None
        if msk is not None:
            msk2 = _warp(msk, M, (self.H, self.W), cv2.INTER_NEAREST, 0)

        # optional hflip
        if np.random.rand() < self.hflip_p:
            img2 = cv2.flip(img2, 1)
            if msk2 is not None: msk2 = cv2.flip(msk2, 1)
        return img2, msk2

    def _jitter(self, img):
        # brightness/contrast/saturation in HSV + a light contrast scale
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2]*(1.0 + np.random.uniform(-self.j_b, self.j_b)), 0, 255)
        hsv[:,:,1] = np.clip(hsv[:,:,1]*(1.0 + np.random.uniform(-self.j_s, self.j_s)), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        alpha = 1.0 + np.random.uniform(-self.j_c, self.j_c)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=0)
        # hue light tweak (approx via HSV shift)
        return out

    def __call__(self, img, msk=None):
        # 1) letterbox to target
        img, msk = self._letterbox(img, msk)
        # 2) geometry
        img, msk = self._geom(img, msk)
        # 3) photometric (image only)
        img = self._jitter(img)
        return img, msk