# tests/test_paths_and_overwrite.py
import json
from pathlib import Path

def test_step_paths_and_metadata(tmp_repo):
    cfg, case_id, paths = tmp_repo
    meta_dir = Path(paths["metadata"])
    assert meta_dir.exists()

    meta_path = meta_dir / "probe.json"
    payload = {"ok": True}
    meta_path.write_text(json.dumps(payload))
    loaded = json.loads(meta_path.read_text())
    assert loaded["ok"] is True
    