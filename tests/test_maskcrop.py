# tests/test_maskcrop.py
from pathlib import Path

import cv2

from data_utils.crop_using_masks import run as maskcrop_run


# tests/test_maskcrop.py
def test_maskcrop_contract(tmp_repo):
    cfg, case_id, p = tmp_repo
    out_color = Path(p["base"]) / "frames_maskcrop"
    out_mask  = Path(p["base"]) / "masks_multiclass_crop"

    maskcrop_run(
        in_color_dir=p["frames_dedup"],
        in_mask_dir=p["masks_multiclass"],
        out_color_dir=str(out_color),
        out_mask_dir=str(out_mask),
        per_video_fixed_roi=False,
        sample_stride=1,
        margin=24,
        min_size=(64, 64),          # ⟵ was (128,128)
        allow_overwrite=True,
        meta_path=str(Path(p["metadata"]) / "maskcrop.json"),
        per_frame_meta_path=str(Path(p["metadata"]) / "maskcrop_frames.json"),
        debug_overlays=False
    )

    img = cv2.imread(str(out_color / "frame_000001.jpg"))
    msk = cv2.imread(str(out_mask / "frame_000001.png"), cv2.IMREAD_UNCHANGED)
    assert img is not None and msk is not None
    assert img.shape[:2] == msk.shape[:2]
    h, w = img.shape[:2]
    assert h >= 64 and w >= 64          # ⟵ was 100