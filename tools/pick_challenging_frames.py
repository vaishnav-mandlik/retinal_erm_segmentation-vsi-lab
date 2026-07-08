#!/usr/bin/env python3
"""
pick_challenging_frames.py — auto-pick and export challenging ERM frames.
See the top-of-file docstring in the prior message for usage details.


python pick_challenging_frames.py --config config.yaml  --split train --k 5 --out ./work_dir/ERM_challenges_test

"""

import os, sys, csv, json, time, zipfile, argparse, shutil, yaml, numpy as np, torch
import cv2
from typing import Dict, List, Tuple

# Project imports
from training.train import _qual_dump, _align_logits_targets, _bfscore
from training.evaluate import _device, _find_best_ckpt
from training.dataset import SegPairDataset
from training.dataset import build_augmentor_from_cfg
from models.models import _build_model_from_cfg

def _load_cfg(path: str) -> Dict:
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)

def _grayscale_from_tensor(img_t: torch.Tensor) -> np.ndarray:
    if img_t.ndim != 3 or img_t.shape[0] != 3:
        arr = img_t.detach().cpu().float().numpy()
        return (arr.mean(0) if arr.ndim == 3 else arr).astype(np.float32)
    r, g, b = img_t[0].cpu().numpy(), img_t[1].cpu().numpy(), img_t[2].cpu().numpy()
    return (0.299*r + 0.587*g + 0.114*b).astype(np.float32)

def _sobel_mean(gray: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0: return float("inf")
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    return float(mag[mask > 0].mean())

def _overlap_frac(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum()); denom = float(a.sum() + 1e-6)
    return inter / denom

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def _zip_dir(src_dir: str, zip_path: str):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                full = os.path.join(root, fn); rel = os.path.relpath(full, src_dir); zf.write(full, rel)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val", choices=["val","test", "train"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    t = cfg["train"]
    use_binary = bool(t.get("use_binary_masks", False))
    class_names = [t.get("binary_class_name","ERM")] if use_binary else t.get("class_names")
    out_ch = 1 if use_binary else len(class_names)
    erm_idx = 0 if use_binary else class_names.index("ERM")

    thr_list = cfg.get("metrics", {}).get("thresholds", [0.5]*out_ch)
    if len(thr_list) != out_ch:
        thr_list = (list(thr_list) + [0.5]*out_ch)[:out_ch]
    fp_min_pixels = int(cfg.get("metrics", {}).get("fp_min_pixels", 50))

    run_root = os.path.join(cfg["work_root"], cfg["run_id"])
    split_txt = os.path.join(run_root, "splits", f"{args.split}.txt")

    ds = SegPairDataset(cfg, split_txt=split_txt if os.path.isfile(split_txt) else None,
                        transform=build_augmentor_from_cfg(cfg, for_val=True, target_sz=cfg["resize"]["target_size"]),
                        for_val=True)
    if len(ds) == 0:
        print(f"[WARN] Dataset split '{args.split}' is empty."); return

    device = _device()
    model = _build_model_from_cfg(cfg, out_ch).to(device)
    ckpt = _find_best_ckpt(cfg); model.load_state_dict(torch.load(ckpt, map_location="cpu")); model.eval()
    print(f"[INFO] {len(ds)} samples loaded from {ckpt}.")
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=int(t.get("num_workers", 4)))

    faint_items: List[Tuple[float,int,int,Dict]] = []
    overlap_items: List[Tuple[float,int,int,Dict]] = []
    blueconf_items: List[Tuple[float,int,int,Dict]] = []
    tiny_items: List[Tuple[float,int,int,Dict]] = []

    thr = torch.tensor(thr_list, dtype=torch.float32).view(1, -1, 1, 1)

    with torch.no_grad():
        for b_idx, batch in enumerate(dl):
            imgs = batch["image"].to(device)
            gts  = batch["mask"].to(device)
            out  = model(imgs)
            logits = out[0] if isinstance(out, (tuple,list)) else out
            seg_logits, gts = _align_logits_targets(logits, gts, out_ch, debug_once_flag=[])

            probs = torch.sigmoid(seg_logits)
            pr_bin = (probs > thr.to(device)).float()
            gt_bin = (gts > 0.5).float()

            B = imgs.shape[0]
            for i in range(B):
                case_id = batch.get("case_id", [""]*B)[i] if isinstance(batch.get("case_id"), list) else batch.get("case_id")
                if isinstance(case_id, torch.Tensor): case_id = str(case_id)
                img_path = batch.get("image_path", [""]*B)[i] if isinstance(batch.get("image_path"), list) else batch.get("image_path")
                if isinstance(img_path, torch.Tensor): img_path = str(img_path)

                gt_erm = (gt_bin[i, erm_idx].detach().cpu().numpy() > 0.5).astype(np.uint8)
                pr_erm = (pr_bin[i, erm_idx].detach().cpu().numpy() > 0.5).astype(np.uint8)
                gray = _grayscale_from_tensor(imgs[i])

                area = int(gt_erm.sum())
                if area > 0:
                    tiny_items.append((float(area), b_idx, i, {"case_id":case_id,"image_path":img_path,"erm_area":int(area)}))
                    sob = _sobel_mean(gray, gt_erm)
                    faint_items.append((float(sob), b_idx, i, {"case_id":case_id,"image_path":img_path,"sobel_mean":float(sob),"erm_area":int(area)}))

                    if out_ch >= 3:
                        gt_tools = ((gt_bin[i, class_names.index("Forceps")].cpu().numpy() > 0.5).astype(np.uint8) |
                                    (gt_bin[i, class_names.index("Light tool")].cpu().numpy() > 0.5).astype(np.uint8))
                        ov = _overlap_frac(gt_erm, gt_tools)
                        overlap_items.append((float(ov), b_idx, i, {"case_id":case_id,"image_path":img_path,"overlap_frac":float(ov),"erm_area":int(area)}))
                else:
                    pred_area = int(pr_erm.sum())
                    if pred_area >= max(fp_min_pixels, 200):
                        blueconf_items.append((float(pred_area), b_idx, i, {"case_id":case_id,"image_path":img_path,"pred_erm_area":int(pred_area)}))

    k = max(1, args.k)
    faint_items.sort(key=lambda x: x[0])
    tiny_items.sort(key=lambda x: x[0])
    overlap_items.sort(key=lambda x: x[0], reverse=True)
    blueconf_items.sort(key=lambda x: x[0], reverse=True)

    picks = {
        "01_faint_low_contrast": faint_items[:k],
        "02_tool_overlap": overlap_items[:k],
        "03_bg_blue_confusion": blueconf_items[:k],
        "04_tiny_erm": tiny_items[:k],
    }

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)
    index_csv = os.path.join(out_root, "index.csv")
    with open(index_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["category","case_id","image_path","note","panel_path"])

        # Re-run to dump panels for just selected items
        wanted = {}
        for cat, lst in picks.items():
            for _, b_idx, i, meta in lst:
                wanted.setdefault(b_idx, set()).add(i)

        with torch.no_grad():
            dl2 = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=int(t.get("num_workers", 4)))
            step = 0
            for b_idx, batch in enumerate(dl2):
                if b_idx not in wanted: continue
                imgs = batch["image"].to(device)
                gts  = batch["mask"].to(device)
                out  = model(imgs); logits = out[0] if isinstance(out,(tuple,list)) else out
                seg_logits, gts = _align_logits_targets(logits, gts, out_ch, debug_once_flag=[])
                for i in sorted(wanted[b_idx]):
                    log1, gt1, im1 = seg_logits[i:i+1], gts[i:i+1], imgs[i:i+1]
                    for cat, lst in picks.items():
                        hit = None
                        for tup in lst:
                            _, b0, i0, meta = tup
                            if b0==b_idx and i0==i: hit=(cat,meta); break
                        if hit is None: continue
                        cat_name, meta = hit
                        out_dir = os.path.join(out_root, cat_name); os.makedirs(out_dir, exist_ok=True)
                        _qual_dump(out_dir, step, log1, gt1, im1, class_names, cfg, img_paths=batch.get("image_path"))
                        panel_path = os.path.join(out_dir, f"step{step:06d}_idx00.jpg")
                        if cat_name=="01_faint_low_contrast":
                            note=f"faint sobel_mean={meta.get('sobel_mean'):0.4f}, area={meta.get('erm_area')}"
                        elif cat_name=="02_tool_overlap":
                            note=f"tool-overlap frac={meta.get('overlap_frac'):0.3f}, area={meta.get('erm_area')}"
                        elif cat_name=="03_bg_blue_confusion":
                            note=f"GT-absent FP pred_area={meta.get('pred_erm_area')}"
                        else:
                            note=f"tiny ERM area={meta.get('erm_area')}"
                        w.writerow([cat_name, meta.get('case_id',''), meta.get('image_path',''), note, panel_path])
                        step += 1

    with open(os.path.join(out_root,"README.md"),"w") as f:
        f.write("# ERM — Challenging Frames (Auto-picked)\n\n")
        f.write("- Categories: faint/unstained, tool overlap, background-blue confusion, tiny ERM\n")
        f.write("- See `index.csv` for case, path, and quick notes\n")
        f.write("- Panels are 4-up: RGB | GT | Pred | Dice legend\n")

    if args.zip:
        zip_path = out_root.rstrip('/\\') + ".zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_root):
                for fn in files:
                    full=os.path.join(root,fn); rel=os.path.relpath(full,out_root); zf.write(full,rel)
        print(f"[done] Zipped to: {zip_path}")
    else:
        print(f"[done] Exported to: {out_root}")
        print(f"       Index: {index_csv}")

if __name__ == "__main__":
    main()
