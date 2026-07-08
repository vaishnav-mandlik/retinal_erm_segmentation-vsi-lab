# Tests

This folder contains lightweight tests for the **retinal_segmentation** repo.  
They aim to catch path/contract regressions in the preprocessing pipeline without requiring the full dataset.

## How to run locally


### Setup
```bash
# from repo root
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest pytest-cov

# make sure ffmpeg is available if you test extraction
# macOS: brew install ffmpeg
# Windows: choco install ffmpeg
```

### Running Tests
```
pytest -q
```



## What the tests cover

- **Config & CLI** (`tests/test_config_and_cli.py`)
  - Verifies placeholder resolution (`{work_root}`, `{data_root}`) and `--dry_run` path printing.

- **Paths & Overwrite** (`tests/test_paths_and_overwrite.py`)
  - Ensures `step_paths()` builds the expected case directory layout.
  - Confirms overwrite guards (won’t clobber non-empty dirs unless `--force` or `allow_overwrite:true`).

- **Dedup** (`tests/test_dedup.py`)
  - Smoke-check of SSIM/blur filtering on tiny synthetic frames.

- **Mask-driven cropping** (`tests/test_maskcrop.py`)
  - Contract test: color+mask pairs are cropped with **identical** boxes; metadata JSON is written.

- **Annotator Previews** (`tests/test_previews.py`)
  - Builds a preview panel and checks that the output image is created.

### Fixtures

- `tests/conftest.py` provides the **`tmp_repo`** fixture:
  - Creates a minimal repo-like temp structure.
  - Seeds a few small images into `frames_dedup/` so steps can run.
  - Writes a small `config.yaml` pointed at the temp structure.

> If pytest reports **`fixture 'tmp_repo' not found`**, ensure the file name is exactly `tests/conftest.py` and run `pytest` from the repository root.

### Tips

- Keep tests **fast** and **hermetic**:
  - Use tiny synthetic images (e.g., 64×64 JPEGs).
  - Avoid real video extraction in CI unless you need it (or gate with an env var).

- Visual checks:
  - For preview/mask tests, we currently assert existence/shape rather than pixel-perfect values.  
    If you want visual regression tests, add “golden” PNGs and compare with a tolerance.

- Common errors:
  - **In-place writes**: many steps delete the output dir when `allow_overwrite=true`.  
    In tests, write to a different output dir or use the “safe in-place” pattern.

### CI - Continuous Integration

A GitHub Actions workflow is included at `.github/workflows/ci.yml`. It:
- runs on push/PR  
- installs ffmpeg and Python deps  
- runs pytest with coverage  
- uploads coverage report artifacts