# tests/test_splits.py
from pathlib import Path

from data_utils.split_train_val_test import run as split_run


def test_split_files(tmp_repo):
    cfg, case_id, p = tmp_repo
    run_root = Path(cfg["work_root"]) / cfg["run_id"]

    split_run(run_root=str(run_root), splits_cfg=cfg["splits"])

    sdir = run_root / "splits"
    assert (sdir / "train.txt").exists()
    assert (sdir / "val.txt").exists()
    assert (sdir / "test.txt").exists()
    