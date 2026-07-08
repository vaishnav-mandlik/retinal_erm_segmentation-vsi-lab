import csv
import json
import os
import re
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data_utils.file_utils import clean_or_make
from models.models import _build_model_from_cfg, MultiTaskSegWithPhase
from .dataset import SegPairDataset, build_augmentor_from_cfg, build_sample_weights_from_tags
from .pred_qual_dump import _qual_dump


def _fmt_hms(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------- Boundary-F1 helpers ----------
def _boundary_map(mask01: np.ndarray) -> np.ndarray:
    """Return a thin boundary for a binary mask (H,W) in {0,1}."""
    k = np.ones((3, 3), np.uint8)
    m = (mask01.astype(np.uint8) * 255)
    grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)  # edges ~1 px
    return (grad > 0).astype(np.uint8)


def _bfscore(pred01: np.ndarray, gt01: np.ndarray, tol_px: int) -> Optional[float]:
    """
    Boundary-F1 between predicted and GT binary masks with distance tolerance (pixels).
    Returns None if GT has no boundary (undefined); otherwise F1 in [0,1].
    """
    pb = _boundary_map(pred01)
    gb = _boundary_map(gt01)

    if gb.sum() == 0:
        # Undefined recall; skip this sample for BF aggregation
        return None

    # Distance transforms on inverse (so pixels inside boundary measure distance to nearest boundary)
    dt_gb = cv2.distanceTransform((1 - gb).astype(np.uint8), cv2.DIST_L2, 3)
    dt_pb = cv2.distanceTransform((1 - pb).astype(np.uint8), cv2.DIST_L2, 3)

    # A predicted boundary pixel is a TP if within tol of any GT boundary
    tp_pred = (pb > 0) & (dt_gb <= tol_px)
    # A GT boundary pixel is a TP if within tol of any predicted boundary
    tp_gt = (gb > 0) & (dt_pb <= tol_px)

    tp_p = tp_pred.sum()
    tp_g = tp_gt.sum()
    p_tot = max(1, int((pb > 0).sum()))
    g_tot = max(1, int((gb > 0).sum()))

    prec = tp_p / p_tot
    rec = tp_g / g_tot
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _bce_logits(input: torch.Tensor, target: torch.Tensor, pos_w_vec: torch.Tensor | None):
    """
    BCEWithLogits with channel-wise pos_weight for NCHW.
    Broadcast pos_weight correctly as (1,C,1,1).
    """
    pw = None
    if pos_w_vec is not None:
        pw = pos_w_vec.to(input.device, dtype=input.dtype).view(1, input.shape[1], 1, 1)
    return F.binary_cross_entropy_with_logits(input, target, pos_weight=pw)


# --- shape/format guards -----------------------------------------------------
def _maybe_to_nchw(x, want_ch: int):
    """
    If a 4D tensor looks NHWC (channels last == want_ch) but C in dim1 isn't want_ch,
    permute to NCHW. Always return a contiguous tensor.
    """
    if x is None or x.dim() != 4:
        return x
    if x.shape[1] != want_ch and x.shape[-1] == want_ch:
        return x.permute(0, 3, 1, 2).contiguous()
    return x.contiguous()


def _align_logits_targets(seg_logits, gts, out_ch: int, debug_once_flag: list):
    """
    Unwrap HF outputs, ensure (N,C,H,W) for both logits and targets,
    cast targets to float for BCE, and (optionally) print a one-time debug line.
    """
    # HuggingFace models sometimes return an object with `.logits`
    if hasattr(seg_logits, "logits"):
        seg_logits = seg_logits.logits

    # Binary targets might arrive as (N,H,W) → add channel
    if gts is not None and gts.dim() == 3:
        gts = gts.unsqueeze(1)

    # Heuristically enforce NCHW
    seg_logits = _maybe_to_nchw(seg_logits, out_ch)
    gts = _maybe_to_nchw(gts, out_ch)

    # Targets must be float for BCEWithLogits
    if gts is not None and not gts.is_floating_point():
        gts = gts.float()

    # Last resort: if shapes still mismatch in the (C vs last) dims, flip targets NHWC→NCHW
    if seg_logits is not None and gts is not None and seg_logits.dim() == 4 and gts.dim() == 4:
        if seg_logits.shape != gts.shape:
            if gts.shape[-1] == seg_logits.shape[1] and gts.shape[1] == gts.shape[-1]:
                gts = gts.permute(0, 3, 1, 2).contiguous()

    # One-time print for troubleshooting if cfg.train.debug_shapes = true
    if debug_once_flag and not debug_once_flag[0]:
        debug_once_flag[0] = True
        print(f"[debug] loss inputs -> logits {tuple(seg_logits.shape)} gts {tuple(gts.shape)}")

    return seg_logits, gts


def _log_csv_row(
    ep, phase, lr, loss, dice_vec, class_names, writer, file_handle,
    erm_bf=None, epoch_time_sec=None
):
    row = [ep, phase, f"{lr:.6f}"]
    row.append("" if (epoch_time_sec is None or phase != "train") else f"{epoch_time_sec:.2f}")
    row.append("" if loss is None else f"{loss:.5f}")

    if dice_vec is None:
        row.append("")  # dice_mean
        row += [""] * len(class_names)
    else:
        import numpy as np
        dice_arr = np.array(dice_vec, dtype=float)
        row.append(f"{dice_arr.mean():.5f}")  # dice_mean
        row += [f"{v:.5f}" for v in dice_arr]

    row.append("" if erm_bf is None else f"{erm_bf:.5f}")
    writer.writerow(row)
    file_handle.flush()


def _select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# ---- helpers: BCE with per-channel pos_weight ----
class ChannelWeightedBCE(torch.nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.clone().float().view(-1))
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        if self.pos_weight is None:
            return F.binary_cross_entropy_with_logits(logits, targets)
        w = self.pos_weight.view(1, -1, 1, 1).to(dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=w)


# ---------- losses ----------
class DiceLoss(torch.nn.Module):
    def __init__(self, weights: Optional[List[float]] = None, eps: float = 1e-6):
        super().__init__()
        self.weights = None if weights is None else torch.tensor(weights, dtype=torch.float32)
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum(dim=(0, 2, 3))
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(0, 2, 3)) + 1e-6
        dice_c = 1.0 - (num / den)  # per-channel loss (1 - dice)
        if self.weights is not None:
            w = self.weights.to(dice_c.device)
            dice = (dice_c * w).sum() / (w.sum() + self.eps)
        else:
            dice = dice_c.mean()
        return dice


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, weights: Optional[List[float]] = None, eps=1e-6):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.weights = None if weights is None else torch.tensor(weights, dtype=torch.float32)
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p * targets).sum(dim=(0, 2, 3))
        fp = (p * (1 - targets)).sum(dim=(0, 2, 3))
        fn = ((1 - p) * targets).sum(dim=(0, 2, 3))
        tversky_c = 1.0 - (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        if self.weights is not None:
            w = self.weights.to(tversky_c.device)
            return (tversky_c * w).sum() / (w.sum() + self.eps)
        return tversky_c.mean()


# ---------- metrics ----------
def dice_per_channel(logits, targets, eps=1e-6):
    """
    logits: (B, C, H, W) raw scores
    targets: (B, C, H, W) {0,1}
    Returns: np.ndarray shape (C,) with per-class Dice (soft, all-frames).
    """
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum((0, 2, 3))
    den = (probs * probs).sum((0, 2, 3)) + (targets * targets).sum((0, 2, 3)) + eps
    return (num / den).detach().cpu().numpy()


def _open_csv(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(header)
    f.flush()
    return f, w


def _accum_pos_counts(sum_pos: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
    # gts: (B,C,H,W) in {0,1}
    with torch.no_grad():
        csum = gts.sum(dim=(0, 2, 3)).detach().cpu()
        return sum_pos + csum


def _print_pixel_ratios(sum_pos: torch.Tensor, total_px: int, class_names: list[str], phase: str):
    ratios = (sum_pos / max(1, total_px)).numpy()
    pretty = {n: f"{r:.2%}" for n, r in zip(class_names, ratios)}
    print(f"[{phase}] class-pixel ratios:", pretty)


def _compute_present_only_and_fp_metrics(
    probs: torch.Tensor,   # (N,C,H,W) sigmoid already
    targets: torch.Tensor, # (N,C,H,W) 0/1 float
    fp_min_pixels: int = 50,
    thr: float | list = 0.5,
):
    """
    Returns:
      present_dice: np.ndarray (C,), NaN if no-present frames for that class
      fp_rate:      np.ndarray (C,), fraction of GT-absent frames with > fp_min_pixels predicted
      present_mean: float, mean of present-only dice over classes (ignores NaNs)
    """
    with torch.no_grad():
        N, C, H, W = probs.shape

        if isinstance(thr, (list, tuple, np.ndarray)):
            thr_t = torch.as_tensor(thr, device=probs.device, dtype=probs.dtype).view(1, C, 1, 1)
        else:
            thr_t = torch.full((1, C, 1, 1), float(thr), device=probs.device, dtype=probs.dtype)

        pr_bin = (probs > thr_t)

        present_dice = []
        fp_rate = []

        for c in range(C):
            gt_c = (targets[:, c] > 0.5)          # (N,H,W) bool
            pr_c = pr_bin[:, c]                   # (N,H,W) bool

            present_mask = gt_c.any(dim=(1, 2))   # (N,) bool  # GT-present frames
            neg_mask = ~present_mask              # GT-absent frames

            # present-only soft dice (micro across present frames)
            if present_mask.any().item():
                p = probs[present_mask, c]
                g = targets[present_mask, c]
                num = 2.0 * (p * g).sum()
                den = (p * p).sum() + (g * g).sum() + 1e-6
                present_dice.append((num / den).item())
            else:
                present_dice.append(float("nan"))

            # negative-frame FP rate
            if neg_mask.any().item():
                pos_pix = pr_c[neg_mask].sum(dim=(1, 2))  # per-image count
                fp = (pos_pix > fp_min_pixels).float().mean().item()
                fp_rate.append(fp)
            else:
                fp_rate.append(float("nan"))

        present_dice = np.array(present_dice, dtype=np.float32)
        fp_rate = np.array(fp_rate, dtype=np.float32)
        present_mean = float(np.nanmean(present_dice)) if not np.all(np.isnan(present_dice)) else float("nan")
        return present_dice, fp_rate, present_mean


def _case_from_path(p: str) -> str:
    parts = p.replace("\\", "/").split("/")
    for s in parts:
        if s.startswith("case"):
            return s
    return "unknown"


def _read_split_cases(txt_path: str) -> list[str]:
    cases = set()
    if os.path.isfile(txt_path):
        with open(txt_path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                cases.add(_case_from_path(ln))
    return sorted(cases)


def _parse_topk_metric(fn: str) -> float:
    # expects 'topk_<metric>_epXXXX.pt'
    try:
        m = re.search(r"topk_([0-9.]+)_ep\d+\.pt$", fn)
        return float(m.group(1)) if m else -1.0
    except Exception:
        return -1.0


def _save_topk_ckpt(ckdir: str, state_dict, metric: float, ep: int, top_k: int = 5):
    os.makedirs(ckdir, exist_ok=True)
    out = os.path.join(ckdir, f"topk_{metric:.5f}_ep{ep:04d}.pt")
    torch.save(state_dict, out)

    # keep only best top_k by metric (higher is better)
    files = [f for f in os.listdir(ckdir) if f.startswith("topk_") and f.endswith(".pt")]
    scored = [(_parse_topk_metric(f), f) for f in files]
    scored.sort(key=lambda x: x[0], reverse=True)
    for _, f in scored[top_k:]:
        try:
            os.remove(os.path.join(ckdir, f))
        except Exception:
            pass


def phase_eval(batch, device, phase_correct: int, phase_logits, phase_total: int, cm, label_counts_epoch, n_phases):
    ph = batch["phase"].to(device)
    valid = ph >= 0
    if valid.any():
        pred = phase_logits.argmax(dim=1)
        phase_correct += (pred[valid] == ph[valid]).sum().item()
        phase_total += valid.sum().item()
        # accumulate label counts and CM
        label_counts_epoch += torch.bincount(ph[valid].cpu(), minlength=n_phases)
        for t, p in zip(ph[valid].cpu().tolist(), pred[valid].cpu().tolist()):
            cm[t, p] += 1
    return phase_correct, phase_total, cm, label_counts_epoch


def run(cfg: Dict, cases_sel: Optional[List[str]] = None):
    t = cfg["train"]
    device = _select_device(t.get("device", "auto"))
    print(f"Using device: {device}")

    use_binary = bool(t.get("use_binary_masks", False))
    class_names = [t.get("binary_class_name", "ERM")] if use_binary else t.get("class_names")
    out_ch = 1 if use_binary else len(class_names)


    # Data
    run_root = os.path.join(cfg["work_root"], cfg["run_id"])
    splits_dir = os.path.join(run_root, "splits")
    tr_txt = os.path.join(splits_dir, "train.txt")
    va_txt = os.path.join(splits_dir, "val.txt")

    aug_tr = build_augmentor_from_cfg(cfg, for_val=False, target_sz=cfg["resize"]['target_size'])
    aug_va = build_augmentor_from_cfg(cfg, for_val=True, target_sz=cfg["resize"]['target_size'])

    ds_tr = SegPairDataset(cfg, split_txt=tr_txt if os.path.isfile(tr_txt) else None,
                           cases_sel=cases_sel, transform=aug_tr, for_val=False)

    sampler = None
    if t.get("oversample_membrane", False):
        w = build_sample_weights_from_tags(cfg, ds_tr.items)  # aligns with dataset items
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(t.get("batch_size", 4)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(t.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )
    ds_va = SegPairDataset(cfg, split_txt=va_txt if os.path.isfile(va_txt) else None,
                           cases_sel=cases_sel, transform=aug_va, for_val=True)

    #  Validate no overlap in train / val cases.
    tr_cases = _read_split_cases(tr_txt) if os.path.isfile(tr_txt) else sorted(
        {_case_from_path(p) for _, p, _ in ds_tr.items})
    va_cases = _read_split_cases(va_txt) if os.path.isfile(va_txt) else sorted(
        {_case_from_path(p) for _, p, _ in ds_va.items})
    print(f"[splits] train cases: {tr_cases}")
    print(f"[splits] val   cases: {va_cases}")
    overlap = set(tr_cases) & set(va_cases)
    if overlap:
        print("[WARNING] train/val case overlap:", sorted(overlap))

    dl_va = DataLoader(ds_va, batch_size=max(1, int(t.get("batch_size", 4))), shuffle=False,
                       num_workers=int(t.get("num_workers", 4)), pin_memory=(device.type == "cuda"))

    # Model
    model = _build_model_from_cfg(cfg, out_ch).to(device)

    # Losses
    tversky = TverskyLoss(alpha=0.3, beta=0.7, weights=t.get("class_weights"))
    lambda_tversky = float(t.get("lambda_tversky", 0.0))
    dice_loss = DiceLoss(weights=t.get("class_weights"))

    bce_pos_w = None
    if not use_binary and t.get("class_weights"):
        bce_pos_w = torch.tensor(t["class_weights"], dtype=torch.float32)
    bce = ChannelWeightedBCE(bce_pos_w)

    # Optim & sched
    opt = torch.optim.AdamW(model.parameters(), lr=float(t.get("lr", 1e-4)),
                            weight_decay=float(t.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)

    # Logs/dirs
    out_dir = t.get("out_dir", t.get("save_dir", "{work_root}/{run_id}/training")).format(**cfg)
    exp_dir = os.path.join(out_dir, t.get("model"))
    clean_or_make(exp_dir)
    qual_val_root = os.path.join(exp_dir, "qual", "val")
    os.makedirs(qual_val_root, exist_ok=True)

    tol = int(cfg.get("metrics", {}).get("boundary_tol_px", 2))
    with open(os.path.join(exp_dir, "config_snapshot.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    fcsv, wcsv = _open_csv(os.path.join(exp_dir, "logs", "train_log.csv"),
                           ["epoch", "phase", "lr", "epoch_time_sec", "loss", "dice_mean"]
                           + [f"dice_{n}" for n in class_names]
                           + [f"ERM Boundary F1 @{tol}px"])

    best_metric, best_path = -1.0, os.path.join(exp_dir, "ckpts", "best.pt")

    # AMP only on CUDA; plain FP on CPU/MPS
    use_amp = (device.type == "cuda") and bool(t.get("amp", False))
    scaler = (torch.amp.GradScaler('cuda') if use_amp else None)
    debug_once = [False] if bool(t.get("debug_shapes", False)) else []

    # NEW: monitor/early stop/top-K
    early_stop_enabled = bool(t.get("early_stop_enabled", False))  # default OFF
    patience = int(t.get("early_stop_patience", 10))
    min_epochs = int(t.get("early_stop_min_epochs", 15))
    epochs_no_improve = 0
    eps = 1e-4
    monitor = t.get("monitor", "present_only_mean")  # 'present_only_mean' | 'all_mean' | 'class:ERM'
    top_k = int(t.get("save_top_k", 5))

    epochs = int(t.get("epochs", 80))
    # monitor_class = t.get("monitor_class", "ERM")
    # print("Val monitor_class is {}".format(monitor_class))
    # mon_idx = None if monitor_class in (None, "mean", "avg") else (
    #     class_names.index(monitor_class) if monitor_class in class_names else None
    # )

    # Track GT-present counts on the val split to ensure gating is stable
    val_present_counts_ref = None

    step = 0
    for ep in tqdm(range(1, epochs + 1), ncols=70):
        ep_t0 = time.time()
        train_dice_batches = []

        # ---- train ----
        model.train()
        tloss = 0.0
        tnum = 0
        phase_correct = 0
        phase_total = 0
        n_phases = 4
        label_counts_epoch = torch.zeros(n_phases, dtype=torch.long)
        cm = torch.zeros(n_phases, n_phases, dtype=torch.long)

        for batch in dl_tr:
            imgs = batch["image"].to(device)
            gts = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)

            if isinstance(model, MultiTaskSegWithPhase):
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        seg_logits, phase_logits = model(imgs)
                        phase_correct, phase_total, cm, label_counts_epoch = phase_eval(
                            batch, device, phase_correct, phase_logits, phase_total, cm, label_counts_epoch, n_phases
                        )
                        seg_logits, gts = _align_logits_targets(seg_logits, gts, out_ch, debug_once)
                        with torch.no_grad():
                            train_dice_batches.append(dice_per_channel(seg_logits, gts))
                        loss_seg = dice_loss(seg_logits, gts) + bce(seg_logits, gts)
                        loss_seg = loss_seg + lambda_tversky * tversky(seg_logits, gts)

                        #

                        lam = float(t.get("phase_head", {}).get("lambda", 0.3))
                        loss = loss_seg
                        if lam > 0 and ("phase" in batch):
                            ph = batch["phase"].to(device)
                            valid = ph >= 0
                            if valid.any():
                                loss = loss + lam * F.cross_entropy(phase_logits[valid], ph[valid])
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    seg_logits, phase_logits = model(imgs)
                    phase_correct, phase_total, cm, label_counts_epoch = phase_eval(
                        batch, device, phase_correct, phase_logits, phase_total, cm, label_counts_epoch, n_phases
                    )
                    seg_logits, gts = _align_logits_targets(seg_logits, gts, out_ch, debug_once)
                    with torch.no_grad():
                        train_dice_batches.append(dice_per_channel(seg_logits, gts))
                    loss_seg = dice_loss(seg_logits, gts) + bce(seg_logits, gts)
                    loss_seg = loss_seg + lambda_tversky * tversky(seg_logits, gts)

                    lam = float(t.get("phase_head", {}).get("lambda", 0.3))
                    loss = loss_seg
                    if lam > 0 and ("phase" in batch):
                        ph = batch["phase"].to(device)
                        valid = ph >= 0
                        if valid.any():
                            loss = loss + lam * F.cross_entropy(phase_logits[valid], ph[valid])
                    loss.backward()
                    opt.step()
            else:
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        seg_logits = model(imgs)
                        seg_logits, gts = _align_logits_targets(seg_logits, gts, out_ch, debug_once)
                        with torch.no_grad():
                            train_dice_batches.append(dice_per_channel(seg_logits, gts))
                        loss_seg = dice_loss(seg_logits, gts) + bce(seg_logits, gts)
                        loss_seg = loss_seg + lambda_tversky * tversky(seg_logits, gts)


                        loss = loss_seg
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    seg_logits = model(imgs)
                    seg_logits, gts = _align_logits_targets(seg_logits, gts, out_ch, debug_once)
                    with torch.no_grad():
                        train_dice_batches.append(dice_per_channel(seg_logits, gts))
                    loss_seg = dice_loss(seg_logits, gts) + bce(seg_logits, gts)
                    loss_seg = loss_seg + lambda_tversky * tversky(seg_logits, gts)

                    loss = loss_seg
                    loss.backward()
                    opt.step()

            if not torch.isfinite(loss):
                print("[warn] non-finite loss; skipping step")
                continue

            tloss += float(loss.item())
            tnum += 1
            if (ep % max(1, int(t.get("qualitative_every", 1))) == 0) and (step % 50 == 0):
                _qual_dump(os.path.join(exp_dir, "qual", "train"), step, seg_logits, gts, imgs, class_names, None,
                           img_paths=batch.get("image_path"))
            step += 1

        tloss /= max(1, tnum)

        # --- summarise train dice for the epoch ---
        if train_dice_batches:
            tr_dices = np.stack(train_dice_batches, axis=0).mean(0)  # per-channel
            train_dice_mean = float(tr_dices.mean())
        else:
            tr_dices = np.zeros((out_ch,), dtype=np.float32)
            train_dice_mean = 0.0

        # ---- val ----
        model.eval()
        dices_batches = []     # classic (all-frames) per-batch dice_per_channel
        probs_list = []        # to build present-only & FP metrics
        gts_list = []

        tol = int(cfg.get("metrics", {}).get("boundary_tol_px", 2))
        fp_min_pixels = int(cfg.get("metrics", {}).get("fp_min_pixels", 50))
        erm_idx = 0 if use_binary else class_names.index("ERM")
        val_preview_max = int(t.get("qual_val_per_epoch", 2))  # configurable; defaults to 2
        val_dump_dir = os.path.join(qual_val_root, f"ep_{ep:03d}")
        os.makedirs(val_dump_dir, exist_ok=True)
        s_idx = 0
        with torch.no_grad():
            for batch in dl_va:
                imgs = batch["image"].to(device)
                gts = batch["mask"].to(device)

                out = model(imgs)
                seg_logits = out[0] if isinstance(out, tuple) else out
                seg_logits, gts = _align_logits_targets(seg_logits, gts, out_ch, debug_once_flag=[])

                # classic dice over this batch (all frames)
                dices_batches.append(dice_per_channel(seg_logits, gts))

                # stash for present-only & fp metrics
                probs_list.append(torch.sigmoid(seg_logits).cpu())
                gts_list.append(gts.cpu())

                if s_idx < val_preview_max:
                    if _qual_dump is not None:
                        _qual_dump(val_dump_dir, s_idx, seg_logits, gts, imgs, class_names, cfg,
                                   img_paths=batch.get("image_path"))
                    s_idx += 1

            # classic all-frames dice (as before)
            if dices_batches:
                dices = np.stack(dices_batches, axis=0).mean(0)
                dice_mean = float(dices.mean())
            else:
                dices = np.zeros((out_ch,), dtype=np.float32)
                dice_mean = 0.0

            # present-only Dice & negative-frame FP rate
            if probs_list:
                probs_all = torch.cat(probs_list, dim=0)   # (N,C,H,W)
                gts_all = torch.cat(gts_list, dim=0)       # (N,C,H,W)

                # ---- present-only GT gating counts ----
                present_mask_gt = (gts_all > 0.5).sum(dim=(2, 3)) > 0  # (N,C)
                n_present_per_class = present_mask_gt.sum(dim=0).tolist()
                counts_str = " ".join(f"{n}:{int(c)}" for n, c in zip(class_names, n_present_per_class))
                print(f"        [val GT-present counts] {counts_str}")

                if val_present_counts_ref is None:
                    val_present_counts_ref = n_present_per_class
                else:
                    if any(int(a) != int(b) for a, b in zip(val_present_counts_ref, n_present_per_class)):
                        warn_str = " ".join(f"{n}:{int(a)}→{int(b)}"
                                            for n, (a, b) in
                                            zip(class_names, zip(val_present_counts_ref, n_present_per_class)))
                        print(f"[warn] GT-present counts changed vs first epoch (val): {warn_str}")

                present_dice, fp_rate, present_mean = _compute_present_only_and_fp_metrics(
                    probs_all, gts_all, fp_min_pixels=fp_min_pixels,
                    thr=cfg["metrics"]["thresholds"]
                )

                # ERM Boundary-F1 on GT-present frames only
                pr_bin = (probs_all[:, erm_idx] > float(cfg["metrics"]["thresholds"][erm_idx])).float().numpy()
                gt_bin = (gts_all[:, erm_idx] > 0.5).float().numpy()
                bf_vals = []
                for i in range(pr_bin.shape[0]):
                    if gt_bin[i].any():  # only present frames
                        bf = _bfscore(pr_bin[i].astype(np.uint8), gt_bin[i].astype(np.uint8), tol)
                        if bf is not None:
                            bf_vals.append(bf)
                erm_bf = float(np.mean(bf_vals)) if bf_vals else None
            else:
                present_dice = np.full((out_ch,), np.nan, dtype=np.float32)
                fp_rate = np.full((out_ch,), np.nan, dtype=np.float32)
                present_mean = float("nan")
                erm_bf = None

        ep_secs = time.time() - ep_t0

        # 1) Log TRAIN
        _log_csv_row(
            ep=ep,
            phase="train",
            lr=opt.param_groups[0]["lr"],
            loss=tloss,
            dice_vec=tr_dices,
            class_names=class_names,
            writer=wcsv,
            file_handle=fcsv,
            epoch_time_sec=ep_secs
        )

        # 2) Log VAL
        _log_csv_row(
            ep=ep,
            phase="val",
            lr=opt.param_groups[0]["lr"],
            loss=None,
            dice_vec=dices if isinstance(dices, (list, tuple, np.ndarray)) else None,
            class_names=class_names,
            writer=wcsv,
            file_handle=fcsv,
            erm_bf=erm_bf,
            epoch_time_sec=ep_secs
        )

        # 3) Console print
        if dices is not None and len(dices):
            pcs = " ".join(f"{n}(val):{float(d):.3f}" for n, d in zip(class_names, dices))
            # extra_bl = f" λ_surf={_lambda_ramp(ep, bl_warm_s, bl_warm_e, bl_lambda_max):.2f}" if bl_enabled else ""


            # extra_bl = f" λ_surf={_lambda_ramp(ep, bl_warm_s, bl_warm_e, bl_lambda_max):.2f}" if bl_enabled else ""
            extra_bl = ""
            print(
                f"[ep {ep:03d}] train_loss={tloss:.4f} train_dice={train_dice_mean:.4f} "
                f"val_dice={float(np.mean(dices)):.4f}  {pcs}  lr={opt.param_groups[0]['lr']:.2e}   "
                f"{('ERM-BF @ ' + str(tol) + 'px=' + (f'{erm_bf:.3f}' if erm_bf is not None else 'NA'))}{extra_bl}"
            )
        else:
            print(f"[ep {ep:03d}] train_loss={tloss:.4f}  (no val set)  lr={opt.param_groups[0]['lr']:.2e}")

        # Extra console line: present-only & FP rates
        pcs_present = " ".join(
            f"{n}:{(np.nan if np.isnan(v) else float(v)):.3f}" for n, v in zip(class_names, present_dice)
        )
        print(
            "        [present in frames only val stats to report]: "
            f"dice_mean={present_mean if not np.isnan(present_mean) else float('nan'):.4f}  "
            f"[{pcs_present}] t={_fmt_hms(ep_secs)}\n "
        )

        # save ckpt(s)
        ckdir = os.path.join(exp_dir, "ckpts")
        os.makedirs(ckdir, exist_ok=True)

        # periodic checkpoint to keep tests deterministic
        if ep % int(t.get("ckpt_every", 1)) == 0:
            torch.save(model.state_dict(), os.path.join(ckdir, f"epoch_{ep:04d}.pt"))

        # ----- choose monitor metric -----
        # available: dice_mean (all-frames mean), dices (per-class all-frames), present_mean (present-only mean)
        if monitor == "present_only_mean":
            sel_metric = present_mean if not np.isnan(present_mean) else -1.0
        elif monitor == "all_mean":
            sel_metric = float(np.mean(dices)) if dices is not None and len(dices) else -1.0
        elif monitor.startswith("class:"):
            _cls = monitor.split(":", 1)[1].strip()
            if _cls in class_names and dices is not None and len(dices):
                sel_metric = float(dices[class_names.index(_cls)])
            else:
                sel_metric = -1.0
        else:
            sel_metric = float(np.mean(dices)) if dices is not None and len(dices) else -1.0

        # ----- best.pt (by selected monitor) -----
        if sel_metric > best_metric + eps:
            best_metric = sel_metric
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1

        # ----- top-K saving (by selected monitor) -----
        if top_k > 0 and np.isfinite(sel_metric) and sel_metric >= -1e8:
            _save_topk_ckpt(ckdir, model.state_dict(), float(sel_metric), ep, top_k=top_k)

        # scheduler on the same monitor
        sched.step(sel_metric)

        # optional early stop (disabled by default)
        if early_stop_enabled and (ep >= min_epochs) and (epochs_no_improve >= patience):
            print(f"[early-stop] no improvement for {patience} epochs on '{monitor}'. best={best_metric:.4f}")
            break

    fcsv.close()

    label = monitor
    print(f"[train] finished. best_{label}={best_metric:.4f} → {best_path}")
    print(f"\nTRAINING results in : {exp_dir}\n")
    return {"exp_dir": exp_dir, "best": best_path, "best_score": best_metric}