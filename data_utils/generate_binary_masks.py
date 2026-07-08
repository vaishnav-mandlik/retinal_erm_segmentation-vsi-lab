# data_utils/generate_binary_masks.py
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from data_utils.file_utils import prepare_output_dir

"""
Multi class single channel (greyscale) to binary mark image construction.
- Class-collapsing (multi-label → binary): simple pixel-wise threshold (mask > 0 → 255) to merge all non-background classes.


Reference : 
- A common practice in semantic segmentation when moving from multi-class to binary tasks. 
  See, e.g., Fully Convolutional Networks for Semantic Segmentation (Long et al., CVPR 2015) for the concept of label maps and background class.
  https://arxiv.org/abs/1411.4038
"""


def iter_masks(in_dir):
    for p in sorted(Path(in_dir).glob("*")):
        if p.suffix.lower() in [".png",".tif",".tiff"]:
            yield p


def run(in_dir, out_dir, meta_path=None, allow_overwrite=False, allow_append_to_existing = False):
    prepare_output_dir(out_dir, allow_overwrite, allow_append_to_existing, step_name="dedup")
    count = 0
    for p in tqdm(list(iter_masks(in_dir)), desc="binmask"):
        mask = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if mask is None: 
            continue
        # collapse all non-zero labels into 255
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, p.name), binary)
        count += 1

    if meta_path:
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        meta = {
            "in_dir": os.path.abspath(in_dir),
            "out_dir": os.path.abspath(out_dir),
            "converted": count,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    print("Converted binary masks count {}".format(count))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--meta_path", default=None)
    args = ap.parse_args()
    run(args.in_dir, args.out_dir, args.meta_path)

if __name__ == "__main__":
    main()