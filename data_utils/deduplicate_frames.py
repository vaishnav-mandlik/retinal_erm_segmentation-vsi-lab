# data_utils/deduplicate_frames.py
import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
from skimage.metrics import structural_similarity
from tqdm import tqdm

from data_utils.file_utils import prepare_output_dir

"""
Blur Filtering
- Variance of Laplacian: a classic focus measure based on the spatial second derivative.

- OpenCV official documentation – Laplacian operator
https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html


Near-Duplicate Removal
- Structural Similarity Index (SSIM) to compare consecutive grayscale frames and discard duplicates above a similarity threshold.
- scikit-image implementation of structural_similarity (formerly compare_ssim)
https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

"""


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def iter_images(in_dir):
    for p in sorted(Path(in_dir).glob("*")):
        if p.suffix.lower() in [".jpg",".jpeg",".png",".tif",".tiff"]:
            yield p

def run(in_dir, out_dir, use_ssim=True, ssim_threshold=0.98, blur_filter=True, laplacian_var_min=30.0, meta_path=None,
        allow_append_to_existing=False, allow_overwrite=False):
    prepare_output_dir(out_dir, allow_overwrite, allow_append_to_existing, step_name="dedup")

    kept = 0; removed_dups = 0; removed_blur = 0
    prev_img_gray = None

    for p in tqdm(list(iter_images(in_dir)), desc="dedup"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None: 
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # blur filter
        if blur_filter:
            if variance_of_laplacian(gray) < laplacian_var_min:
                removed_blur += 1
                continue

        # ssim dedup
        if use_ssim and prev_img_gray is not None:
            score = structural_similarity(prev_img_gray, gray)
            if score >= ssim_threshold:
                removed_dups += 1
                continue

        # keep
        out_path = os.path.join(out_dir, p.name)
        shutil.copy2(str(p), out_path)
        kept += 1
        prev_img_gray = gray

    if meta_path:
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        meta = {
            "in_dir": os.path.abspath(in_dir),
            "out_dir": os.path.abspath(out_dir),
            "use_ssim": use_ssim,
            "ssim_threshold": ssim_threshold,
            "blur_filter": blur_filter,
            "laplacian_var_min": laplacian_var_min,
            "kept": kept,
            "removed_dups": removed_dups,
            "removed_blur": removed_blur,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print("removed_dups {}; removed_blurs {}".format(removed_dups, removed_blur))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--use_ssim", type=bool, default=True)
    ap.add_argument("--ssim_threshold", type=float, default=0.98)
    ap.add_argument("--blur_filter", type=bool, default=True)
    ap.add_argument("--laplacian_var_min", type=float, default=30.0)
    ap.add_argument("--meta_path", default=None)
    args = ap.parse_args()
    run(args.in_dir, args.out_dir, args.use_ssim, args.ssim_threshold, args.blur_filter, args.laplacian_var_min, args.meta_path)

if __name__ == "__main__":
    main()