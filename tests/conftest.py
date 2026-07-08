# tests/conftest.py
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


def _write_config(work_root: Path, run_id: str, data_root: Path) -> dict:
    cfg = {
        "data_root": str(data_root),
        "work_root": str(work_root),
        "run_id": run_id,
        "allow_overwrite": True,
        "dry_run": False,

        # one synthetic case
        "cases": [
            {
                "case_id": "case1",
                "video_path": str(data_root / "raw_videos" / "dummy.mp4"),  # not used by tests
                "fps": 1.0
            }
        ],

        # minimal step configs (avoid KeyError in pipeline)
        "extract": {
            "fps": 1.0,
            "qscale": 2,
            "filename_pattern": "frame_%06d.jpg",
            "start": None,
            "end": None
        },
        "dedup": {
            "use_ssim": True,
            "ssim_threshold": 0.98,
            "blur_filter": False,
            "laplacian_var_min": 0.0
        },
        "crop": {
            "method": "largest_component",
            "roi_margin_px": 12,
            "sample_stride": 5,
            "per_video_fixed_roi": True
        },
        "maskcrop": {
            "per_video_fixed_roi": True,
            "sample_stride": 1,
            "margin_px": 12,
            "min_size": [48, 48],
            "margin_frac": 0.05,
            "dilate_kernel": 15,
            "use_color_union": False,
            "retina_v_thresh": 18,
            "retina_rgb_thresh": 18,
            "allow_overwrite": True
        },
        "splits": {
            "train": ["case1"],
            "val": [],
            "test": []
        },
        "previews": {
            # the preview code will auto-pick a frames folder; we seed frames_dedup
            "max_per_case": 4
        },
        "resize": {
            # the preview code will auto-pick a frames folder; we seed frames_dedup
            "target_size": [512, 512]
        },
        "train": {
            "use_maskcrop": False,
            "use_binary_masks": False,
            "class_names": [
                "ERM",
                "Forceps",
                "Light tool"
            ],
            "class_index_map": {
                "ERM": 1,
                "Light tool": 2,
                "Forceps": 3
            },
            "model": "unetpp",
            "phase_head": {
                "enabled": False,
                "lambda": 0.05,
                "enc_feats_hint": 512
            },
            "augment": {
                "enable": True,
                "rotate_deg": 8,
                "scale_range": [
                    0.97,
                    1.03
                ],
                "translate_frac": 0.02,
                "hflip_prob": 0.0,
                "jitter": [
                    0.1,
                    0.1,
                    0.1,
                    0.05
                ]
            },
            "batch_size": 1,
            "epochs": 1,
            "lr": 0.0001,
            "num_workers": 0,
            "class_weights": None,
            "save_dir": f"{work_root}/{run_id}/training/test_train",
            "ckpt_every": 1,
            "debug_shapes": False,
            "device": "cpu",
        },
        "metrics": {
            "boundary_tol_px": 3,
            "thresholds": [
                0.4,
                0.5,
                0.5
            ],
            "fp_min_pixels": 50
        },
    }
    with open(work_root.parent / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def _seed_synthetic_images(img_dir: Path, n=3, size=(64, 64)):
    img_dir.mkdir(parents=True, exist_ok=True)
    h, w = size
    for i in range(1, n + 1):
        # simple color gradient with a bright circle (rough “retina”)
        img = np.zeros((h, w, 3), np.uint8)
        cv2.circle(img, (w // 2, h // 2), min(h, w) // 3, (0, 80 + 40 * i, 180), -1)
        cv2.putText(img, str(i), (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        out = img_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(out), img)


def _seed_matching_masks(mask_dir: Path, n=3, size=(64, 64)):
    mask_dir.mkdir(parents=True, exist_ok=True)
    h, w = size
    for i in range(1, n + 1):
        m = np.zeros((h, w), np.uint8)
        # foreground ellipse (pretend ERM or instrument mask)
        cv2.ellipse(m, (w // 2, h // 2), (w // 4, h // 5), 0, 0, 360, 1, -1)  # label 1
        out = mask_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(out), m)


@pytest.fixture
def tmp_repo(tmp_path):
    """
    Creates a tiny synthetic repo structure and returns (cfg, case_id, paths_dict)
    """
    root = tmp_path  # repo root (pytest temp)
    # mimic your layout (pipeline at root, data_utils/ sibling)
    (root / "data_utils").mkdir(parents=True, exist_ok=True)

    work_root = root / "work_dir"
    data_root = root / "data_dir"
    run_id = "run_test"

    # video placeholder (not actually used)
    (data_root / "raw_videos").mkdir(parents=True, exist_ok=True)
    (data_root / "raw_videos" / "dummy.mp4").write_bytes(b"\x00")

    # build case dirs
    base = work_root / run_id / "case1"
    frames_raw = base / "frames_raw"
    frames_dedup = base / "frames_dedup"
    frames_cropped = base / "frames_cropped"
    frames_maskcrop = base / "frames_maskcrop"
    masks_mc = base / "masks_multiclass"
    masks_mc_crop = base / "masks_multiclass_crop"
    masks_bin = base / "masks_binary"
    sup_export = base / "supervisely_export"
    meta = base / "metadata"

    for d in [frames_raw, frames_dedup, frames_cropped, frames_maskcrop,
              masks_mc, masks_mc_crop, masks_bin, sup_export, meta]:
        d.mkdir(parents=True, exist_ok=True)

    # seed tiny images and masks
    _seed_synthetic_images(frames_dedup, n=3, size=(64, 64))
    _seed_matching_masks(masks_mc, n=3, size=(64, 64))

    # write minimal config.yaml at repo root
    cfg = _write_config(work_root=work_root, run_id=run_id, data_root=data_root)

    # paths dict mirroring pipeline.step_paths()
    paths = {
        "base": str(base),
        "frames_raw": str(frames_raw),
        "frames_dedup": str(frames_dedup),
        "frames_cropped": str(frames_cropped),
        "masks_multiclass": str(masks_mc),
        "masks_multiclass_crop": str(masks_mc_crop),
        "masks_binary": str(masks_bin),
        "supervisely_export": str(sup_export),
        "metadata": str(meta),
    }

    # change CWD so imports like `import pipeline` work from repo root
    os.chdir(root)

    return (cfg, "case1", paths)
