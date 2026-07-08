# tests/test_previews.py
import os
from pathlib import Path

from data_utils.annotator_previews import save_previews


def test_annotator_previews_panel(tmp_repo):
    cfg, case_id, p = tmp_repo
    in_dir  = Path(p["frames_dedup"])          # any color frames folder is fine
    out_dir = Path(p["base"]) / "annotator_previews"

    save_previews(
        in_dir=str(in_dir),
        out_dir=str(out_dir),
        allow_overwrite=True,
        include=["Original", "CLAHE", "Unsharp", "Gamma correct"]
    )

    outs = [f for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    assert len(outs) >= 1
    