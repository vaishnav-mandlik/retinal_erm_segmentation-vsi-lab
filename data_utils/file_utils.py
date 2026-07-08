import os
import shutil

def prepare_output_dir(output_dir: str, allow_overwrite: bool = False,
                        allow_append_to_existing: bool = False, step_name: str = "", ):
    """
    Ensure output_dir is ready for writing.

    - If output_dir exists and allow_overwrite=True, delete and recreate it.
    - If output_dir exists and is not empty and allow_overwrite=False, raise an error.
    - If output_dir does not exist, create it.

    Parameters
    ----------
    output_dir : str
        Path to the directory that should be writable.
    allow_overwrite : bool, optional
        If True, remove any existing directory and recreate.
    step_name : str, optional
        Friendly label (e.g., 'dedup', 'crop') for logging.
    """
    if os.path.isdir(output_dir):
        if allow_overwrite:
            label = f"[{step_name}] " if step_name else ""
            print(f"{label}Removing existing files in {output_dir}")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        else:
            if allow_append_to_existing:
                return
            if os.listdir(output_dir):
                raise RuntimeError(
                    f"{step_name or 'Step'}: Output directory {output_dir} is not empty.\n"
                    f"Use --force or set allow_overwrite: true in config.yaml."
                )
    else:
        os.makedirs(output_dir, exist_ok=True)


def clean_or_make(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(out_dir):
        print(f"Removing existing files in {out_dir}")
        shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(out_dir, exist_ok=True)