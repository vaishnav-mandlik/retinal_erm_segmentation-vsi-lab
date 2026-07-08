# tests/test_dedup.py
import os
from pathlib import Path

from data_utils.deduplicate_frames import run as dedup_run


def test_dedup_smoke(tmp_repo):
    cfg, case_id, p = tmp_repo

    out_dir = Path(p["base"]) / "frames_dedup_out"  # <— NEW
    out_dir.mkdir(exist_ok=True)

    dedup_run(
        in_dir=p["frames_dedup"],         # seeded by fixture
        out_dir=str(out_dir),             # <— different from in_dir
        use_ssim=True,
        ssim_threshold=0.9999,
        blur_filter=False,
        laplacian_var_min=0.0,
        meta_path=str(Path(p["metadata"]) / "dedup.json"),
        allow_overwrite=True
    )

    files = sorted([f for f in os.listdir(out_dir) if f.lower().endswith(".jpg")])
    assert len(files) >= 1