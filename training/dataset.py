# training/dataset.py
import logging
import os, cv2, numpy as np
from typing import List, Tuple, Dict, Optional
import torch, csv
from torch.utils.data import Dataset
from .augments import LightAugmentor



def _load_case_tags_map(run_root: str, case_id: str) -> dict[tuple[str,str], dict]:
    """
    Returns {(case_id, image_name) -> row_dict} for one case.
    Expects {run_root}/{case_id}/metadata/tags.csv
    """
    p = os.path.join(run_root, case_id, "metadata", "tags.csv")
    m = {}
    if os.path.isfile(p):
        with open(p, "r", newline="") as f:
            for row in csv.DictReader(f):
                m[(case_id, row["image_name"])] = row
    return m

def build_sample_weights_from_tags(cfg: dict, items: list[tuple[str,str,str]]) -> list[float]:
    """
    items: [(case_id, img_path, mask_path), ...]
    Weight >1.0 for rows where membrane_present==1.
    """
    run_root = os.path.join(cfg["work_root"], cfg["run_id"])
    case_ids = sorted({c for c, _, _ in items})
    idx_map = {}
    for c in case_ids:
        idx_map.update(_load_case_tags_map(run_root, c))

    w_pos = float(cfg.get("train",{}).get("oversample_membrane_weight", 4.0))
    weights = []
    for case_id, ip, _ in items:
        key = (case_id, os.path.basename(ip))
        row = idx_map.get(key)
        has = 1 if (row and str(row.get("membrane_present","0")) in ("1","true","True")) else 0
        weights.append(w_pos if has else 1.0)
    return weights


def _exists_nonempty(p: str) -> bool:
    return os.path.isdir(p) and any(os.scandir(p))

def _first_existing(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if _exists_nonempty(p):
            return p
    return None

def _case_dir(cfg: Dict, case_id: str) -> str:
    return os.path.join(cfg["work_root"], cfg["run_id"], case_id)

def _guess_frame_dir(case_root: str, prefer_resized: bool, use_maskcrop: bool) -> Optional[str]:
    candidates = []
    if prefer_resized:
        candidates += [os.path.join(case_root, "frames_resized")]
    if use_maskcrop:
        candidates += [os.path.join(case_root, "frames_maskcrop")]
    candidates += [os.path.join(case_root, "frames_cropped"),
                   os.path.join(case_root, "frames_dedup"),
                   os.path.join(case_root, "frames_raw")]
    return _first_existing(candidates)

def _guess_mask_dir(case_root: str, prefer_resized: bool, use_maskcrop: bool) -> Optional[str]:
    candidates = []
    if prefer_resized:
        candidates += [os.path.join(case_root, "masks_resized")]
    if use_maskcrop:
        candidates += [os.path.join(case_root, "masks_multiclass_crop")]
    candidates += [os.path.join(case_root, "masks_multiclass"),
                   os.path.join(case_root, "masks_binary")]
    return _first_existing(candidates)

def _to_one_hot(mask_idx: np.ndarray, pos_indices: List[int], H: int, W: int) -> np.ndarray:
    """Turn indexed mask → K-channel binary (background implicit)."""
    k = len(pos_indices)
    hot = np.zeros((H, W, k), np.uint8)
    for ci, cls_id in enumerate(pos_indices):
        hot[..., ci] = (mask_idx == cls_id).astype(np.uint8) * 255
    return hot

def _letterbox(img: np.ndarray, H: int, W: int, is_mask=False) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    img_r = cv2.resize(img, (nw, nh), interpolation=interp)
    pad_val = 0 if is_mask else (0,0,0)
    top  = (H - nh)//2; bot = H - nh - top
    left = (W - nw)//2; right = W - nw - left
    return cv2.copyMakeBorder(img_r, top, bot, left, right, cv2.BORDER_CONSTANT, value=pad_val)

class SegPairDataset(Dataset):
    """
    Loads RGB frames + masks with your toggles.
    - Multi-class → per-class sigmoid heads (no explicit background channel).
    - Binary → single-channel ERM.
    - Optionally returns phase index if available.
    """
    def __init__(self,
                 cfg: Dict,
                 split_txt: Optional[str] = None,
                 cases_sel: Optional[List[str]] = None,
                 transform: Optional[LightAugmentor] = None,
                 for_val: bool = False):
        self.cfg = cfg
        tcfg = cfg["train"]
        self.prefer_resized = bool(tcfg.get("prefer_resized", True))
        self.use_maskcrop = bool(tcfg.get("use_maskcrop", True))
        self.use_binary = bool(tcfg.get("use_binary_masks", False))
        self.target_size = tuple(cfg["resize"]['target_size'])
        if self.target_size is None:
            raise ValueError("Target size must be specified")
        self.transform = transform
        self.for_val = for_val

        # class map and channels
        cimap = tcfg.get("class_index_map")
        if (cimap is None): raise ValueError("Class index map must be specified")
        self.class_names = tcfg.get("class_names")
        self.class_ids = [int(cimap[n]) for n in self.class_names]
        if self.use_binary:
            bin_name = tcfg.get("binary_class_name", "ERM")
            self.class_names = [bin_name]
            self.class_ids = [cimap[bin_name]]

        # phase labels (optional)
        self.phase_labels = tcfg.get("phase_labels", [])
        self.phase_to_idx = {p:i for i,p in enumerate(self.phase_labels)}
        self.phase_map = {}  # "case_id/image_name" -> phase_idx
        phase_csv = tcfg.get("phase_csv_path", None)
        if phase_csv:
            phase_csv = phase_csv.format(**cfg)
        if tcfg.get("phase_head", {}).get("enabled", False) and phase_csv and os.path.isfile(phase_csv):
            # lazy CSV read (no pandas)
            with open(phase_csv, "r") as f:
                header = f.readline().strip().split(",")
                hidx = {h:i for i,h in enumerate(header)}
                for line in f:
                    cols = line.strip().split(",")
                    case_id = cols[hidx["case_id"]]
                    imgname = cols[hidx["image_name"]]
                    phase   = cols[hidx["phase"]]
                    key = f"{case_id}/{imgname}"
                    if phase in self.phase_to_idx:
                        self.phase_map[key] = self.phase_to_idx[phase]

        # build item list
        if split_txt and os.path.isfile(split_txt):
            with open(split_txt, "r") as f:
                img_paths = [ln.strip() for ln in f if ln.strip()]
        else:
            # collect from cases
            img_paths = []
            for case_id in (cases_sel or []):
                root = _case_dir(cfg, case_id)
                fdir = _guess_frame_dir(root, self.prefer_resized, self.use_maskcrop)
                if not fdir: continue
                for ext in ("*.jpg","*.png","*.jpeg"):
                    for p in sorted([os.path.join(fdir, q) for q in os.listdir(fdir) if q.endswith(ext.split("*")[-1][1:])]):
                        img_paths.append(p)
        self.items = []
        for p in img_paths:
            case_id = p.split(os.sep)[-3]  # .../<case_id>/frames_*/file
            case_root = _case_dir(cfg, case_id)
            mdir = _guess_mask_dir(case_root, self.prefer_resized, self.use_maskcrop)
            if not mdir: continue
            name = os.path.splitext(os.path.basename(p))[0]
            # try png first (masks usually PNG)
            cand = [os.path.join(mdir, name + ".png"), os.path.join(mdir, name + ".jpg")]
            mask_p = None
            for c in cand:
                if os.path.isfile(c):
                    mask_p = c
                    break
            if mask_p is None: continue
            self.items.append((case_id, p, mask_p))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        case_id, ip, mp = self.items[idx]
        img = cv2.imread(ip)
        if img is None: raise RuntimeError(f"Failed to read image: {ip}")
        msk_raw = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if msk_raw is None: raise RuntimeError(f"Failed to read mask: {mp}")

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # --- Enforce same HxW before anything else (in case disk pairs mismatch) ---
        if img.shape[:2] != msk_raw.shape[:2]:
            msk_raw = cv2.resize(msk_raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # --- Single source of truth for target_size/letterbox ---
        need_letterbox = not (("frames_resized" in ip) and ("masks_resized" in mp))
        if need_letterbox:
            th, tw = self.target_size  # single config entry
            # run letterbox once to get target shape; then force mask to that exact shape
            img = _letterbox(img, th, tw, is_mask=False)
            if msk_raw.shape[:2] != img.shape[:2]:
                msk_raw = cv2.resize(msk_raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # --- Reduce mask to 1 channel (palette-safe) ---
        if msk_raw.ndim == 3:
            # If mask is RGB/palette PNG, reduce to index map deterministically
            msk_idx = cv2.cvtColor(msk_raw, cv2.COLOR_BGR2GRAY)
            # (optional) log once per run instead of print-spamming:
            logging.warning("Mask not grayscale at %s; converted via cvtColor", mp)
        else:
            msk_idx = msk_raw

        # --- One-hot / binary ---
        if self.use_binary:
            hot = (msk_idx == self.class_ids[0]).astype(np.uint8) * 255
            msk = hot[..., None]
        else:
            H, W = img.shape[:2]
            msk = _to_one_hot(msk_idx, self.class_ids, H, W)  # ensures (H,W,K)

        # tiny guard to catch bad exports or wrong maps
        vals = set(np.unique(msk_idx).tolist())
        allowed = {0} | set(self.class_ids)
        if not vals.issubset(allowed):
            print(f"WARN: unexpected mask IDs {sorted(vals - allowed)} in {mp} (allowed {sorted(allowed)})")

        # --- Augment (must preserve alignment) ---
        if self.transform is not None:
            img, msk = self.transform(img, msk)

        # --- Final safety: assert exact alignment ---
        if img.shape[:2] != msk.shape[:2]:
            # last-resort fix (should be no-op):
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # --- To tensors ---
        img_t = torch.from_numpy(img[:, :, ::-1].copy()).float().permute(2, 0, 1) / 255.0  # BGR→RGB
        msk_t = torch.from_numpy(msk.copy()).float().permute(2, 0, 1) / 255.0

        key = f"{case_id}/{os.path.basename(ip)}"
        phase_idx = self.phase_map.get(key, -1)
        phase_t = torch.tensor(phase_idx, dtype=torch.long)

        return {
            "image": img_t, "mask": msk_t,
            "phase": phase_t, "case_id": case_id,
            "image_path": ip
        }

def build_augmentor_from_cfg(cfg: Dict, for_val: bool, target_sz: Tuple[int, int]) -> Optional[LightAugmentor]:
    tcfg = cfg["train"]
    if not tcfg.get("augment")['enable']:
        print("WARNING: augmentation not enabled")
        return None
    if (target_sz is None) or (target_sz[0] is None) or (target_sz[1] is None): raise ValueError("target_sz and target_sz[0] and target_sz[1] are required")
    acfg = tcfg.get("augment", {})
    if for_val:  # no aug for val/test; still need deterministic letterbox
        return LightAugmentor(target_sz,
                              rotate_deg=0, scale_range=(1.0,1.0),
                              translate_frac=0.0, hflip_prob=0.0, jitter=(0,0,0,0))
    return LightAugmentor(target_sz,
                          rotate_deg=int(acfg.get("rotate_deg",8)),
                          scale_range=tuple(acfg.get("scale_range",[0.97,1.03])),
                          translate_frac=float(acfg.get("translate_frac",0.02)),
                          hflip_prob=float(acfg.get("hflip_prob",0.0)),
                          jitter=tuple(acfg.get("jitter",[0.10,0.10,0.10,0.05])))