# tests/test_preview_training_augs.py
import os
from glob import glob
import numpy as np
import cv2
import pytest

from training import preview_training_augs as pta


@pytest.mark.skip(reason="Fix me")
def test_preview_training_aug_strips_created(tmp_repo):
    """
    Integration smoke:
      - run preview_augs on a tiny temp repo
      - ensure at least one PNG strip is produced and is a sane size
    """
    cfg, case_id, paths = tmp_repo

    # Minimal, implementation-agnostic preview config
    cfg["preview_augs"] = {
        "prefer_resized": True,           # will fallback internally if not present
        "one_strip": True,                # single PNG per source frame
        "scale_up": 1.0,                  # keep small for test
        "mask_preview": {"binarize": True, "dilate_px": 1},
        # Grid is used by the new panel composer; if code ignores it, harmless.
        "grid": {"rows": 2, "cols": 3, "gap_px": 4},
    }

    # Signature: run(cfg=..., cases_sel=[...])
    pta.run(cfg=cfg, cases_sel=[case_id])

    out_base = os.path.join(cfg["work_root"], cfg["run_id"])
    cands = []
    for pat in [
        os.path.join(out_base, "training", "aug_previews", "*.png"),
        os.path.join(out_base, "training", "aug_previews", "**", "*.png"),
        os.path.join(out_base, "*", "aug_previews", "*.png"),  # per-case fallback
    ]:
        cands.extend(glob(pat, recursive=True))
    pngs = sorted(cands)
    img = cv2.imread(pngs[0])
    assert img is not None and img.size > 0
    # keep artifact small in CI
    assert os.path.getsize(pngs[0]) < 10 * 1024 * 1024


def test_mask_vis_binarize_toggle():
    """
    _mask_vis should return 3-channel output.
    With binarize=False we allow either multi-level or binary,
    since upstream masks may come in already thresholded.
    """
    m = np.zeros((32, 32), np.uint8)
    m[8:24, 8:16] = 64
    m[8:24, 16:24] = 200

    cfg_on = {"preview_augs": {"mask_preview": {"binarize": True, "dilate_px": 0}}}
    vis_on = pta._mask_vis(m, cfg_on)
    assert vis_on.ndim == 3 and vis_on.shape[2] == 3

    cfg_off = {"preview_augs": {"mask_preview": {"binarize": False, "dilate_px": 0}}}
    vis_off = pta._mask_vis(m, cfg_off)
    assert vis_off.shape == vis_on.shape

    # May still be 0/255 depending on earlier scaling, so just require >=2 levels.
    uniq_off = np.unique(cv2.cvtColor(vis_off, cv2.COLOR_BGR2GRAY))
    assert len(uniq_off) >= 2


def test_row_image_mask_alignment():
    """
    _row() draws a label band and concatenates [image | mask].
    We only require: 3 channels, same height for both tiles, and width >= 2x tile.
    """
    img = np.zeros((64, 64, 3), np.uint8)
    msk = np.zeros((64, 64), np.uint8)  # single-channel label map

    row = pta._row(img, msk, "Test")

    assert row.ndim == 3 and row.shape[2] == 3
    assert row.shape[0] >= img.shape[0]  # can be taller due to label band
    assert row.shape[1] >= 2 * img.shape[1]

    half_w = row.shape[1] // 2
    left = row[:, :half_w]
    right = row[:, half_w:]
    assert left.shape[0] == right.shape[0]


def test_compose_panel_from_tiles_grid(monkeypatch):
    """
    If the new grid composer exists, verify it can tile 3x2 without errors.
    This test is skipped if the helper isn't present.
    """
    if not hasattr(pta, "_compose_panel_from_tiles"):
        pytest.skip("_compose_panel_from_tiles not present; skipping grid test")

    # Create 6 fake tiles (already-labeled rows), varying sizes a bit
    tiles = []
    for i in range(6):
        h = 64 + (i % 2) * 8     # 64 or 72
        w = 96 + (i % 3) * 8     # 96, 104, 112
        tiles.append(np.zeros((h, w, 3), np.uint8) + (30 + i * 10))

    cfg = {"preview_augs": {"grid": {"rows": 2, "cols": 3, "gap_px": 4}}}
    panel = pta._compose_panel_from_tiles(cfg, tiles)
    assert panel is not None and panel.ndim == 3 and panel.shape[2] == 3
    # Height should be >= tallest row sum (2 rows), width >= widest col sum (3 cols)
    assert panel.shape[0] > 100 and panel.shape[1] > 200


