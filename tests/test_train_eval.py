# tests/test_train_eval.py
import os
import json
from glob import glob

import training.train as tr
import training.evaluate as ev


def _one(hits):
    return hits[0] if hits else None


def _find_ckpt(run_root: str):
    patterns = [
        os.path.join(run_root, "**", "ckpts", "*.pt"),
        os.path.join(run_root, "training", "**", "ckpts", "*.pt"),
        os.path.join(run_root, "**", "*.pt"),
    ]
    for pat in patterns:
        hits = glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


def _find_metrics(run_root: str):
    patterns = [
        os.path.join(run_root, "**", "metrics*.json"),
        os.path.join(run_root, "training", "**", "metrics*.json"),
        os.path.join(run_root, "evaluation", "**", "metrics*.json"),
    ]
    for pat in patterns:
        hits = glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


class TestTrainEval:
    def test_train_smoke_and_ckpt(self, tmp_repo):
        """
        Smoke test: run a tiny train and verify a checkpoint exists.
        Uses cfg from tmp_repo as-is (no test-side plumbing).
        Gracefully skips if a known augmentor param bug is present.
        """
        cfg, case_id, _paths = tmp_repo
        run_root = os.path.join(cfg["work_root"], cfg["run_id"])
        tr.run(cfg=cfg, cases_sel=[case_id])
        ckpt = _find_ckpt(run_root)
        assert ckpt is not None, f"No checkpoint found under {run_root}"
        assert os.path.getsize(ckpt) > 0

    def test_eval_smoke_and_metrics(self, tmp_repo):
        """
        Train briefly (as above), then run evaluation and check metrics JSON exists & parses.
        """
        cfg, case_id, _paths = tmp_repo
        run_root = os.path.join(cfg["work_root"], cfg["run_id"])

        tr.run(cfg=cfg, cases_sel=[case_id])

        # Run evaluation (signature: run(cfg=..., cases_sel=[...]))
        ev.run(cfg=cfg)

        metrics = _find_metrics(run_root)
        assert metrics is not None, f"No metrics JSON produced under {run_root}"
        with open(metrics, "r") as f:
            mj = json.load(f)
        assert isinstance(mj, dict) and len(mj) > 0
        # replace the current "plausible" check with this

        plausible_legacy = {"mean_dice", "dice", "miou", "loss", "summary", "per_class"}
        plausible_new = {"all_frames", "present_only", "fp_rate_absent"}

        has_legacy = bool(plausible_legacy.intersection(mj.keys()))
        has_new = plausible_new.issubset(mj.keys())

        assert has_legacy or has_new, f"Unexpected metrics keys: {mj.keys()}"