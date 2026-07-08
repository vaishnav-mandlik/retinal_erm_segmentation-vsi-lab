import os, json, glob
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils.file_utils import clean_or_make
from .dataset import SegPairDataset, build_augmentor_from_cfg
from models.models import _build_model_from_cfg
from .train import dice_per_channel, _select_device, _bfscore, _parse_topk_metric
from .pred_qual_dump import _qual_dump

"""
•	test_dice (all-frames)
        Soft Dice averaged over all images, per class (includes images where the class is absent). Useful for sanity, but can underestimate rare classes like ERM.
•	test_dice (present-only)
        Soft Dice computed only on images where the class is present in GT. This is the main quality number for each class (e.g., ERM Dice (present-only)).
•	Boundary-F1 @kpx (all | present-only)
        F1 on mask boundaries within a tolerance of k pixels (e.g., 3 px).
        All-frames treats empty-empty as 1.0; present-only ignores images without the class (preferred for ERM).
•	FP-rate (absent ≥ N px)
        On GT-absent images for a class, fraction where prediction produced ≥ N pixels. Lower is better (specificity proxy).
•	mmseg-style per-class Dice / IoU
        Dataset-level metrics from hard masks (argmax/thresholded): sums TP/FP/FN over the whole set, then computes Dice/IoU.
        mDice / mIoU (fg mean) = mean over foreground classes; good for cross-paper comparability.
•	Thresholds
        Per-class binarization thresholds used during evaluation (report them for reproducibility).
•	Boundary tolerance (px)
        Pixel tolerance used in Boundary-F1 (e.g., 3 px); report alongside BF1.
•	(If enabled) Mean pixel share (GT | Pred)
        Average class prevalence in GT and predictions; helpful to spot bias/over-prediction.


If running Top k 
test_dice(present-only) (XXX) as primary selector; 
then look at ERM present BF1 (0.XXX) of mmseg-style ERM or mDdice as tie-breakers


"""

def dice_per_channel_hard(pb, gb, eps=1e-6):
    # pb, gb: (N,C,H,W) in {0,1}
    num = 2.0 * (pb * gb).sum(dim=(0, 2, 3))
    den = pb.sum(dim=(0, 2, 3)) + gb.sum(dim=(0, 2, 3)) + eps
    return (num / den).detach().cpu().numpy()


def _device():
    if _select_device is not None:
        return _select_device("auto")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_best_ckpt(cfg: Dict) -> str:
    """
    Finds the last training runs by time and picks the best checkpoint in that run.
    """
    t = cfg["train"]
    base = t.get("out_dir", t.get("save_dir", "{work_root}/{run_id}/training")).format(**cfg)
    pats = [os.path.join(base, "**", "ckpts", "best.pt"), os.path.join(base, "ckpts", "best.pt")]
    cands = []
    for p in pats:
        cands.extend(glob.glob(p, recursive=True))
    if not cands:
        raise FileNotFoundError(f"No best.pt under {base}")
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def _fallback_dump(out_dir, idx, logits, gts, imgs, class_names, cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        pr = (torch.sigmoid(logits[0]) > 0.5).float().cpu().numpy()  # C,H,W
    gt = (gts[0].detach().cpu().numpy() > 0.5).astype(np.uint8)

    img = imgs[0].detach().float().cpu()
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    rgb = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    colors = {"ERM": (0, 255, 0), "Forceps": (0, 165, 255), "Light tool": (255, 255, 0)}

    def edges(mask01):
        m = (mask01.astype(np.uint8) * 255)
        e = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        return e

    def overlay_edges(base, mask01, col):
        out = base.copy()
        e = edges(mask01)
        c = np.zeros_like(out)
        c[:] = col
        out[e] = (0.85 * c[e] + 0.15 * out[e]).astype(np.uint8)
        return out

    t_gt = bgr.copy()
    t_pr = bgr.copy()
    for ci, name in enumerate(class_names):
        col = colors.get(name, (200, 200, 200))
        t_gt = overlay_edges(t_gt, gt[ci], col)
        t_pr = overlay_edges(t_pr, pr[ci], col)

    panel = np.concatenate([bgr, t_gt, t_pr], axis=1)
    cv2.imwrite(os.path.join(out_dir, f"sample_{idx:04d}.png"), panel)


def run(cfg: Dict, split: str = "test", save_previews: int = 20, chk_pt: str = None) -> Dict[str, float]:
    """
    Evaluate on a split. Returns dict of per-class Dice (all-frames).
    Prints and saves:
      • all-frames Dice,
      • present-only Dice and FP-rate,
      • Boundary-F1 (all & present) per class,
      • mmseg-style dataset-level Dice/IoU (binary-per-class),
      • pixel share per class (GT|Pred) and background.
    """
    # --- config & data ---
    t = cfg["train"]
    use_binary = bool(t.get("use_binary_masks", False))
    class_names = [t.get("binary_class_name", "ERM")] if use_binary else t.get("class_names")
    out_ch = 1 if use_binary else len(class_names)

    tol_px = int(cfg.get("metrics", {}).get("boundary_tol_px", 2))
    thr_list = cfg.get("metrics", {}).get("thresholds", [0.5] * out_ch)
    if len(thr_list) != out_ch:
        thr_list = (list(thr_list) + [0.5] * out_ch)[:out_ch]
    fp_min_pixels = int(cfg.get("metrics", {}).get("fp_min_pixels", 50))

    run_root = os.path.join(cfg["work_root"], cfg["run_id"])
    split_txt = os.path.join(run_root, "splits", f"{split}.txt")

    ds = SegPairDataset(
        cfg,
        split_txt=split_txt if os.path.isfile(split_txt) else None,
        transform=build_augmentor_from_cfg(cfg, for_val=True, target_sz=cfg["resize"]["target_size"]),
        for_val=True,
    )
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=int(t.get("num_workers", 4)))

    device = _device()
    model = _build_model_from_cfg(cfg, out_ch).to(device)

    ckpt = chk_pt if chk_pt else _find_best_ckpt(cfg)
    print(f"Using checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    base_out_dir = os.path.join(run_root, "eval", t.get("model", "model"))
    clean_or_make(base_out_dir)
    previews_dir = os.path.join(base_out_dir, "qual")
    os.makedirs(previews_dir, exist_ok=True)

    # accumulators
    dices_batches = []     # classic all-frames soft Dice per batch
    probs_list = []        # to build present-only & FP metrics (concatenate later)
    gts_list = []

    with torch.no_grad():
        s_idx = 0
        for batch in dl:
            imgs = batch["image"].to(device)
            gts = batch["mask"].to(device)

            out = model(imgs)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            # classic batch Dice (all frames)
            dices_batches.append(dice_per_channel(logits, gts))

            # stash for present-only/FP/BF/mmseg/pixel-share
            probs_list.append(torch.sigmoid(logits).cpu())
            gts_list.append(gts.cpu())

            # previews
            if save_previews and s_idx < save_previews:
                if _qual_dump is not None:
                    _qual_dump(previews_dir, s_idx, logits, gts, imgs, class_names, cfg, img_paths=batch.get("image_path"))
                else:
                    _fallback_dump(previews_dir, s_idx, logits, gts, imgs, class_names, cfg)
                s_idx += 1

    # === All-frames Dice (unchanged semantics) ===
    if dices_batches:
        dices_all = np.stack(dices_batches, 0).mean(0)  # per-class
        mean_dice_all = float(dices_all.mean())
    else:
        dices_all = np.zeros((out_ch,), dtype=np.float32)
        mean_dice_all = 0.0

    # === Present-only Dice + FP-rate + BF1 + mmseg-style + pixel shares ===
    if probs_list:
        probs_all = torch.cat(probs_list, dim=0)   # (N,C,H,W)
        gts_all = torch.cat(gts_list, dim=0)       # (N,C,H,W)

        # thresholds per class
        thr = torch.tensor(thr_list, device=probs_all.device, dtype=probs_all.dtype).view(1, -1, 1, 1)
        pr_bin = (probs_all > thr).to(torch.uint8)           # (N,C,H,W) {0,1}
        gt_bin = (gts_all > 0.5).to(torch.uint8)

        N, C, H, W = pr_bin.shape

        # present-only soft Dice (micro over present frames)
        def _dice_soft(pb, gb, eps: float = 1e-6) -> float:
            num = 2.0 * (pb * gb).sum()
            den = (pb * pb).sum() + (gb * gb).sum() + eps
            return float((num / den).cpu())

        present_dice = np.full((out_ch,), np.nan, dtype=np.float32)
        for c in range(out_ch):
            present_mask = gt_bin[:, c].any(dim=(1, 2))
            if present_mask.any():
                p = probs_all[present_mask, c]
                g = gts_all[present_mask, c]
                present_dice[c] = _dice_soft(p, g)
        present_mean = float(np.nanmean(present_dice)) if not np.all(np.isnan(present_dice)) else float("nan")

        # FP-rate on GT-absent frames (per class)
        fp_rate = np.full((out_ch,), np.nan, dtype=np.float32)
        for c in range(out_ch):
            neg_mask = ~gt_bin[:, c].any(dim=(1, 2))
            if neg_mask.any():
                pos_pix = pr_bin[neg_mask, c].sum(dim=(1, 2))  # per-image pixel count
                fp_rate[c] = float((pos_pix > fp_min_pixels).float().mean().cpu())

        # mmseg-style TP/FP/FN per class (binary-per-class)
        mm_tp = np.zeros(out_ch, dtype=np.int64)
        mm_fp = np.zeros(out_ch, dtype=np.int64)
        mm_fn = np.zeros(out_ch, dtype=np.int64)
        for c in range(out_ch):
            p = pr_bin[:, c].bool()
            g = gt_bin[:, c].bool()
            mm_tp[c] = int((p & g).sum().item())
            mm_fp[c] = int((p & ~g).sum().item())
            mm_fn[c] = int((~p & g).sum().item())

        dice_per_class_mm = {}
        iou_per_class_mm = {}
        for c in range(out_ch):
            tp, fp, fn = mm_tp[c], mm_fp[c], mm_fn[c]
            denom_d = 2 * tp + fp + fn
            denom_i = tp + fp + fn
            dice_per_class_mm[c] = (2.0 * tp / float(denom_d)) if denom_d > 0 else float("nan")
            iou_per_class_mm[c] = (tp / float(denom_i)) if denom_i > 0 else float("nan")

        valid_d = [v for v in dice_per_class_mm.values() if not np.isnan(v)]
        valid_i = [v for v in iou_per_class_mm.values() if not np.isnan(v)]
        mDice_fg = float(np.mean(valid_d)) if valid_d else float("nan")
        mIoU_fg = float(np.mean(valid_i)) if valid_i else float("nan")

        # Pixel share per class (GT | Pred), plus background = no class active
        gt_frac = {}
        pred_frac = {}
        for c in range(out_ch):
            gpf = gt_bin[:, c].float().mean(dim=(1, 2)).cpu().numpy()
            ppf = pr_bin[:, c].float().mean(dim=(1, 2)).cpu().numpy()
            gt_frac[c] = (float(gpf.mean()), float(gpf.std()))
            pred_frac[c] = (float(ppf.mean()), float(ppf.std()))

        gt_bg = (gt_bin.sum(dim=1) == 0).float().mean(dim=(1, 2)).cpu().numpy()
        pr_bg = (pr_bin.sum(dim=1) == 0).float().mean(dim=(1, 2)).cpu().numpy()
        bg_stats = {
            "gt": (float(gt_bg.mean()), float(gt_bg.std())),
            "pred": (float(pr_bg.mean()), float(pr_bg.std())),
        }

        # Per-class Boundary-F1 (all-frames & present-only)
        def _bf_all_definition(p_bin: np.ndarray, g_bin: np.ndarray, tol_px: int) -> float:
            has_p = bool(p_bin.any()); has_g = bool(g_bin.any())
            if not has_p and not has_g: return 1.0
            if not has_p and has_g:     return 0.0
            if has_p and not has_g:     return 0.0
            return _bfscore(p_bin, g_bin, tol_px)

        bf1_all = {}
        bf1_present = {}
        for c in range(out_ch):
            p_maps = pr_bin[:, c].cpu().numpy().astype(np.uint8)
            g_maps = gt_bin[:, c].cpu().numpy().astype(np.uint8)
            all_vals, pres_vals = [], []
            for i in range(N):
                bfa = _bf_all_definition(p_maps[i], g_maps[i], tol_px)
                if bfa is not None:
                    all_vals.append(bfa)
                if g_maps[i].any():
                    bfp = _bfscore(p_maps[i], g_maps[i], tol_px)
                    if bfp is not None:
                        pres_vals.append(bfp)
            bf1_all[c] = float(np.mean(all_vals)) if all_vals else float("nan")
            bf1_present[c] = float(np.mean(pres_vals)) if pres_vals else float("nan")

    else:
        present_dice = np.full((out_ch,), np.nan, dtype=np.float32)
        present_mean = float("nan")
        fp_rate = np.full((out_ch,), np.nan, dtype=np.float32)
        mDice_fg = float("nan")
        mIoU_fg = float("nan")
        dice_per_class_mm = {}
        iou_per_class_mm = {}
        gt_frac = {}
        pred_frac = {}
        bg_stats = {"gt": (float("nan"), float("nan")), "pred": (float("nan"), float("nan"))}
        bf1_all = {}
        bf1_present = {}

    # === Console ===
    pcs_all = "  ".join(f"{n}:{float(d):.3f}" for n, d in zip(class_names, dices_all))
    print(f"[eval] {split}_dice(all-frames)={mean_dice_all:.4f}  {pcs_all}")

    def _fmt_present(v):  # pretty-print NaN as N/A
        return "N/A" if (v != v) else f"{float(v):.3f}"

    pcs_present = "  ".join(f"{n}:{_fmt_present(present_dice[i])}" for i, n in enumerate(class_names))
    print(f"[eval] {split}_dice(present-only)={('nan' if present_mean!=present_mean else f'{present_mean:.4f}')}  {pcs_present}")

    # fps_line = "  ".join(f"{n}:{_fmt_present(fp_rate[i])}" for i, n in enumerate(class_names))
    # print(f"[eval] {split}_fp_rate(absent>={fp_min_pixels}px)  {fps_line}")

    print("\n-- Boundary-F1 (all-frames | present-only) --")
    for i, n in enumerate(class_names):
        a = bf1_all.get(i, float("nan")); p = bf1_present.get(i, float("nan"))
        a_s = ("nan" if np.isnan(a) else f"{a:.3f}")
        p_s = ("nan" if np.isnan(p) else f"{p:.3f}")
        print(f"{n:>18}: all={a_s}  present={p_s}")

    print("\n-- mmseg-style Dice / IoU (dataset-level, binary-per-class) --")
    for i, n in enumerate(class_names):
        d = dice_per_class_mm.get(i, float("nan")); j = iou_per_class_mm.get(i, float("nan"))
        if np.isnan(d):
            print(f"{n:>18}: Dice=nan   IoU=nan")
        else:
            print(f"{n:>18}: Dice={d:.4f}  IoU={j:.4f}")
    print(f"\nmmseg-style mDice (fg mean): {mDice_fg:.4f}")
    print(f"mmseg-style mIoU  (fg mean): {mIoU_fg:.4f}")

    print("\n-- Mean pixel share per class (GT | Pred) --")
    for i, n in enumerate(class_names):
        gm, gs = gt_frac.get(i, (float("nan"), float("nan")))
        pm, ps = pred_frac.get(i, (float("nan"), float("nan")))
        gm_s = "nan" if np.isnan(gm) else f"{gm:.3%}"
        gs_s = "nan" if np.isnan(gs) else f"{gs:.3%}"
        pm_s = "nan" if np.isnan(pm) else f"{pm:.3%}"
        ps_s = "nan" if np.isnan(ps) else f"{ps:.3%}"
        print(f"{n:>18}: {gm_s} ± {gs_s}  |  {pm_s} ± {ps_s}")
    bg_gt_m, bg_gt_s = bg_stats["gt"]
    bg_pr_m, bg_pr_s = bg_stats["pred"]
    if not np.isnan(bg_gt_m):
        print(f"{'background':>18}: {bg_gt_m:.3%} ± {bg_gt_s:.3%}  |  {bg_pr_m:.3%} ± {bg_pr_s:.3%}")
    else:
        print(f"{'background':>18}: nan ± nan  |  nan ± nan")

    # --- Pixel-share summary (means/stds) ---
    def _ms(arr):
        arr = list(arr)
        return (float(np.mean(arr)) if arr else float("nan"),
                float(np.std(arr)) if arr else float("nan"))

    pixel_share = {
        "background": {
            "gt_mean": float(bg_stats["gt"][0]),
            "gt_std": float(bg_stats["gt"][1]),
            "pred_mean": float(bg_stats["pred"][0]),
            "pred_std": float(bg_stats["pred"][1]),
        }
    }

    for n in class_names:
        # handle both dict-by-name and list/array-by-index gracefully
        if isinstance(gt_frac, dict):
            gm, gs = _ms(gt_frac.get(n, []))
            pm, ps = _ms(pred_frac.get(n, []))
        else:
            idx = class_names.index(n)
            gm, gs = _ms(gt_frac[idx])
            pm, ps = _ms(pred_frac[idx])

        pixel_share[n] = {
            "gt_mean": gm, "gt_std": gs,
            "pred_mean": pm, "pred_std": ps,
        }

    # === JSON (keep compatibility keys and add new sections) ===
    metrics = {"split": split, "thresholds": {n: thr_list[i] for i, n in enumerate(class_names)},
               "mean_dice": mean_dice_all, "dice": {n: float(dices_all[i]) for i, n in enumerate(class_names)},
               "miou": mIoU_fg, "all_frames": {
            "mean_dice": mean_dice_all,
            "per_class_dice": {n: float(dices_all[i]) for i, n in enumerate(class_names)},
            "boundary_tol_px": tol_px,
        }, "present_only": {
            "mean_dice": (None if (present_mean != present_mean) else float(present_mean)),
            "per_class_dice": {
                n: (None if (np.isnan(float(present_dice[i]))) else float(present_dice[i]))
                for i, n in enumerate(class_names)
            },
        }, "bf1": {
            "all_frames": {
                n: (None if (np.isnan(float(bf1_all.get(i, float("nan"))))) else float(bf1_all[i]))
                for i, n in enumerate(class_names)
            },
            "present_only": {
                n: (None if (np.isnan(float(bf1_present.get(i, float("nan"))))) else float(bf1_present[i]))
                for i, n in enumerate(class_names)
            },
        }, "mmseg": {
            "per_class": {
                n: {
                    "dice": (None if np.isnan(float(dice_per_class_mm.get(i, float("nan")))) else float(
                        dice_per_class_mm[i])),
                    "iou": (
                        None if np.isnan(float(iou_per_class_mm.get(i, float("nan")))) else float(iou_per_class_mm[i]))
                }
                for i, n in enumerate(class_names)
            },
            "mDice_fg": mDice_fg,
            "mIoU_fg": mIoU_fg
        }, "pixel_share": pixel_share
    }

    with open(os.path.join(base_out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nEVAL results in {base_out_dir}\n")
    # Return the classic all-frames per-class dict (kept for backward compatibility)
    return {n: float(dices_all[i]) for i, n in enumerate(class_names)}


def run_topk(cfg: Dict, split: str = "test", k: int = 5):
    """
    Finds topk_*.pt in the latest training ckpt folder and evaluates each.
    If none found, falls back to best.pt + newest epoch_*.pt files.
    """
    t = cfg["train"]
    base = t.get("out_dir", t.get("save_dir", "{work_root}/{run_id}/training")).format(**cfg)
    ckdir = None
    cands = [
        os.path.join(base, t.get("model", ""), "ckpts"),
        os.path.join(base, "ckpts"),
    ]
    for d in cands:
        if os.path.isdir(d):
            ckdir = d
    if ckdir is None:
        raise FileNotFoundError(f"No ckpts dir under {base}")

    def _paths(pattern):
        return sorted(glob.glob(os.path.join(ckdir, pattern)))

    # prefer topk
    topk = _paths("topk_*.pt")
    if topk:
        topk = sorted(topk, key=lambda p: _parse_topk_metric(os.path.basename(p)), reverse=True)[:k]
    else:
        best = _paths("best.pt")
        epks = sorted(_paths("epoch_*.pt"))[-k:]
        topk = best + epks

    print(f"[topk-eval] evaluating {len(topk)} checkpoints:")
    rows = []
    for cp in topk:
        print(f"\n=== EVAL: {os.path.basename(cp)} ===")
        _ = run(cfg, split=split, save_previews=0, chk_pt=cp)
        run_root = os.path.join(cfg["work_root"], cfg["run_id"])
        mpath = os.path.join(run_root, "eval", t.get("model", ""), "metrics.json")
        try:
            mj = json.load(open(mpath, "r"))
        except Exception:
            mj = {}

        rows.append({
            "ckpt": os.path.basename(cp),
            "mean_dice_all": mj.get("all_frames", {}).get("mean_dice"),
            "present_only_mean": mj.get("present_only", {}).get("mean_dice"),
            "mDice_fg": mj.get("mmseg", {}).get("mDice_fg"),
            "mIoU_fg": mj.get("mmseg", {}).get("mIoU_fg"),
        })

    print("\n[topk-eval] summary (pick your winner):")
    for r in rows:
        try:
            print(f"{r['ckpt']:>24}  all={r['mean_dice_all']:.4f}  present={r['present_only_mean']:.4f}  "
                  f"mDice={r['mDice_fg']:.4f}  mIoU={r['mIoU_fg']:.4f}")
        except Exception:
            print(r)