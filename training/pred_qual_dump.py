import os

import cv2
import numpy as np
import torch

# ---------- QUAL VIS HELPERS ----------

def _class_palette(class_names):
    # BGR (cv2) colours chosen for contrast
    base = {
        "ERM": (40, 220, 40),           # green
        "Forceps": (0, 165, 255),       # orange
        "Light tool": (255, 200, 0),    # cyan-ish
    }
    # fallback colours if class name not in base
    extras = [(255, 0, 255), (0, 255, 255), (255, 0, 0)]
    pal = []
    for i, n in enumerate(class_names):
        pal.append(base.get(n, extras[i % len(extras)]))
    return pal  # list of BGR tuples

def _dice_np(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin & gt_bin).sum()
    return (2.0 * inter + eps) / (pred_bin.sum() + gt_bin.sum() + eps)


# ----------  _qual_dump (per-class) ----------
def _qual_dump(out_dir, step, logits, gts, rgbs, class_names,  thr_list=None, img_paths=None):
    """
    Writes a 4-up panel:
      [ RGB | GT overlay (per-class colours) | Pred overlay | Legend (per-class Dice) ]
    Shapes:
      logits: [B, C, H, W] (raw)
      gts:    [B, C, H, W] in {0,1}
      rgbs:   [B, 3, H, W] in [0,1] or [0,255]
    """
    os.makedirs(out_dir, exist_ok=True)


    def _nice_name(p):
        if not p: return ""
        base = os.path.basename(p)
        # try to include case id if present in path
        parts = p.split(os.sep)
        case = next((t for t in parts if t.lower().startswith("case")), None)
        return f"{case}-{base}" if case else base


    with torch.no_grad():
        B, C, H, W = logits.shape

        # -------- resolve thresholds robustly --------
        def _as_float_list(x, C, default=0.5):
            # None -> all default
            if x is None:
                return [float(default)] * C
            # cfg dict (common gotcha when caller passed the whole cfg)
            if isinstance(x, dict):
                tr = x.get("train", {}) if "train" in x else {}
                v = tr.get("pred_threshold", default)
                # could be scalar or list
                if isinstance(v, (list, tuple)):
                    try:
                        vals = [float(t) for t in v]
                    except Exception:
                        vals = [float(default)] * C
                    return vals + [float(default)] * max(0, C - len(vals))
                else:
                    try:
                        f = float(v)
                    except Exception:
                        f = float(default)
                    return [f] * C
            # scalar (incl. numpy/str scalars)
            try:
                if isinstance(x, (float, int, np.floating, np.integer)) or (isinstance(x, str) and x.strip() != ""):
                    return [float(x)] * C
            except Exception:
                pass
            # iterable (list/tuple of per-class thresholds)
            try:
                vals = [float(t) for t in x]
            except Exception:
                vals = [float(default)] * C
            if len(vals) < C:
                vals += [float(default)] * (C - len(vals))
            return vals

        thr_vals = _as_float_list(thr_list, C, default=0.5)

        # -------- tensors → cpu numpy --------
        pr = torch.sigmoid(logits).detach().cpu().numpy()   # (B,C,H,W)
        gt = gts.detach().cpu().numpy().astype(np.uint8)    # (B,C,H,W)
        rgb = rgbs.detach().cpu().numpy()                    # (B,3,H,W)
        if rgb.max() <= 1.01:
            rgb = (rgb * 255.0).clip(0, 255)
        rgb = rgb.astype(np.uint8).transpose(0, 2, 3, 1)     # (B,H,W,3)

        palette = _class_palette(class_names)

        for i in range(B):
            # img = rgb[i].copy()
            img_rgb = rgb[i].copy()  # (H,W,3) RGB
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # -> BGR for cv2
            gt_i = gt[i]
            pr_i = pr[i]
            name_text = _nice_name(img_paths[i]) if img_paths is not None else ""

            # --- coloured overlays (contours) for GT and Pred ---
            gt_ov = img.copy()
            pr_ov = img.copy()

            dices = []
            for c in range(C):
                thr = thr_vals[c]
                pred_bin = (pr_i[c] >= thr).astype(np.uint8)
                pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                gt_bin   = (gt_i[c] > 0).astype(np.uint8)

                gt_has = gt_bin.any()
                pr_has = pred_bin.any()
                if not gt_has and not pr_has:
                    d = None  # N/A for legend
                elif not gt_has and pr_has:
                    d = 0.0
                else:
                    d = float(_dice_np(pred_bin.astype(bool), gt_bin.astype(bool)))
                dices.append(d)

                color = palette[c]

                # contours: GT (2px), Pred (1px) — thinner than before
                for mask, canvas, thick in ((gt_bin, gt_ov, 1), (pred_bin, pr_ov, 1)):
                    if mask.any():
                        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            cv2.drawContours(canvas, cnts, -1, color, thickness=thick, lineType=cv2.LINE_AA)

            # --- legend tile ---
            legend = np.zeros_like(img) + 30  # dark background
            y = 30
            cv2.putText(legend, "Per-class Dice", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
            y += 24
            for c, (name, d) in enumerate(zip(class_names, dices)):
                swatch = np.zeros((18, 18, 3), np.uint8)
                swatch[:] = palette[c]
                legend[y-15:y+3, 20:38] = swatch
                txt =  f"{name}: {'N/A' if d is None else f'{d:.3f}'}"
                cv2.putText(legend, txt, (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
                y += 26

            # put filename on the LEFT image
            if name_text:
                cv2.putText(img, name_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(img, name_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # --- compose 4-up ---
            gap = 10
            tiles = [img, gt_ov, pr_ov, legend]
            h = max(t.shape[0] for t in tiles)
            tiles = [cv2.resize(t, (H, h)) if t.shape[1] != H else t for t in tiles]
            panel = np.zeros((h, sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1), 3), np.uint8)
            x = 0
            for t in tiles:
                w = t.shape[1]
                panel[:, x:x+w] = t
                x += w + gap

            cv2.imwrite(os.path.join(out_dir, f"step{step:06d}_idx{i:02d}_{name_text}.jpg"), panel)
