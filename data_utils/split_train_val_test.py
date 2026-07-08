# split_train_val_test.py
import os, json
from pathlib import Path
from typing import Dict, List, Optional

IMG_EXTS = (".jpg", ".jpeg", ".png")
CAND_FRAME_DIRS = ["frames_resized", "frames_maskcrop", "frames_cropped", "frames_dedup", "frames_raw"]

def _pick_frames_dir(case_path: Path, prefer_resized: bool = True) -> Optional[Path]:
    ordered = CAND_FRAME_DIRS if not prefer_resized else (["frames_resized"] + [d for d in CAND_FRAME_DIRS if d != "frames_resized"])
    for d in ordered:
        p = case_path / d
        if p.is_dir() and any(p.glob(f"*{ext}") for ext in IMG_EXTS):
            return p
    return None

def _list_images(dir_path: Path) -> List[str]:
    items: List[str] = []
    for ext in IMG_EXTS:
        items.extend(sorted(str(p.resolve()) for p in dir_path.glob(f"*{ext}")))
    return items

def run(run_root: str, splits_cfg: Dict, prefer_resized: bool = True) -> Dict[str, int]:
    """
    Write split files under <run_root>/splits/{train,val,test}.txt with absolute image paths.
    Returns per-split counts.
    """
    run_root_p = Path(run_root)
    splits_dir = run_root_p / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # If caller passed prefer_resized inside the same dict, strip it out
    pure_cfg = {k: v for k, v in splits_cfg.items() if isinstance(v, list)}

    counts: Dict[str, int] = {}
    for split_name, case_ids in pure_cfg.items():
        items: List[str] = []
        for case_id in (case_ids or []):
            case_path = run_root_p / case_id
            frames_dir = _pick_frames_dir(case_path, prefer_resized=prefer_resized)
            if not frames_dir:
                print(f"[split] WARNING: no frames found for {case_id} (looked in {CAND_FRAME_DIRS})")
                continue
            imgs = _list_images(frames_dir)
            if not imgs:
                print(f"[split] WARNING: {case_id} → {frames_dir} has no images")
            items.extend(imgs)

        out_file = splits_dir / f"{split_name}.txt"
        with out_file.open("w") as f:
            f.write("\n".join(items))
        counts[split_name] = len(items)
        print(f"[split] {split_name}: {len(items)} files → {out_file}")

    # Write/refresh meta (keeps your original behavior)
    meta = {
        "run_root": str(run_root_p.resolve()),
        "splits": pure_cfg,
    }
    (splits_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return counts