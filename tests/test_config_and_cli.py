# tests/test_config_and_cli.py
import subprocess
import sys
from pathlib import Path

import yaml


def test_pipeline_imports():
    import pipeline  # noqa: F401

def test_config_placeholders_and_dry_run(tmp_repo, tmp_path):
    cfg, case_id, _ = tmp_repo
    cfg_path = tmp_path / "config.yaml"

    # Use placeholders; your pipeline.resolve_placeholders should handle these:
    cfg["data_root"] = "{data_root}"
    cfg["work_root"] = "{work_root}"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    # Dry-run a command that doesn’t touch external deps
    cmd = [sys.executable, str(Path("pipeline.py")), "--config", str(cfg_path), "--cmd", "split", "--dry_run"]
    subprocess.run(cmd, check=False)