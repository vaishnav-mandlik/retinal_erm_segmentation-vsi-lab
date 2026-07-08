# tests/test_binmask.py
from pathlib import Path

import cv2
import numpy as np

from data_utils.generate_binary_masks import run as binmask_run


def test_generate_binary_masks(tmp_repo):
    cfg, case_id, p = tmp_repo
    out_dir = Path(p["masks_binary"])

    binmask_run(
        in_dir=p["masks_multiclass"],
        out_dir=str(out_dir),
        meta_path=str(Path(p["metadata"]) / "binary_masks.json"),
        allow_overwrite=True
    )

    out = cv2.imread(str(out_dir / "frame_000001.png"), cv2.IMREAD_UNCHANGED)
    assert out is not None
    uniq = set(np.unique(out).tolist())
    assert uniq.issubset({0, 255})
    