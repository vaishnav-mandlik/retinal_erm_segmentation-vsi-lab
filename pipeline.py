import argparse
import os
from typing import Dict, Any, List

import yaml

from data_utils.annotator_previews import run_from_config as previews_run
# Import step - Data Preprocessing
from data_utils.crop_black_borders import run as step_crop
from data_utils.crop_using_masks import run as step_maskcrop
from data_utils.deduplicate_frames import run as step_dedup
from data_utils.extract_frames import run as step_extract
from data_utils.generate_binary_masks import run as step_bin
from data_utils.resize_pairs import run as step_resize
from data_utils.split_train_val_test import run as step_split
from training.preview_training_augs import run as step_preview_augs
from data_utils.generate_tags_csv import run as step_tags_csv
from data_utils.compute_stats import run as step_compute_stats
from training.evaluate import run as step_eval
from training.evaluate import run_topk as step_eval_topk


# Training steps
from training.train import run as step_train


def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def resolve_placeholders(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # recursively replace {data_root} and {work_root} occurrences
    def repl(v):
        if isinstance(v, str):
            # Only replace the two known placeholders; leave any other braces alone
            v = v.replace("{data_root}", cfg["data_root"])
            v = v.replace("{work_root}", cfg["work_root"])
            return v
        elif isinstance(v, list):
            return [repl(x) for x in v]
        elif isinstance(v, dict):
            return {k: repl(val) for k, val in v.items()}
        else:
            return v
    return repl(cfg)

def case_dir(cfg, case_id):
    return os.path.join(cfg["work_root"], cfg["run_id"], case_id)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def nonempty(path):
    return os.path.isdir(path) and any(os.scandir(path))

def guard_overwrite(cfg, out_dir):
    if cfg.get("allow_overwrite", False):
        return
    if nonempty(out_dir):
        if (cfg.get("allow_append_to_existing", False)):
            return
        raise RuntimeError(f"Refusing to overwrite non-empty output dir: {out_dir}. Set allow_overwrite: true or pass --force.")

def print_dry(cfg, msg):
    if cfg.get("dry_run", False):
        print(f"[DRY-RUN] {msg}")

def list_cases(cfg, only: List[str] = None):
    all_cases = [c["case_id"] for c in cfg["cases"]]
    return [c for c in all_cases if (only is None or c in only)]

def step_paths(cfg, case_id):
    base = case_dir(cfg, case_id)
    return {
        "base": base,
        "frames_raw": os.path.join(base, "frames_raw"),
        "frames_dedup": os.path.join(base, "frames_dedup"),
        "frames_cropped": os.path.join(base, "frames_cropped"),
        "frames_maskcrop": os.path.join(base, "frames_maskcrop"),
        "masks_multiclass": os.path.join(base, "masks_multiclass"),
        "masks_multiclass_crop": os.path.join(base, "masks_multiclass_crop"),
        "masks_binary": os.path.join(base, "masks_binary"),
        "supervisely_export": os.path.join(base, "supervisely_export"),
        "metadata": os.path.join(base, "metadata"),
        "frames_resized": os.path.join(base, "frames_resized"),
        "masks_resized": os.path.join(base, "masks_resized"),
    }

def run_extract(cfg, cases_sel):
    for c in cfg["cases"]:
        if c["case_id"] not in cases_sel: continue
        paths = step_paths(cfg, c["case_id"])
        ensure_dir(paths["frames_raw"])
        guard_overwrite(cfg, paths["frames_raw"])
        print(f">> [extract] {c['case_id']} -> {paths['frames_raw']}")
        if not cfg.get("dry_run", False):
            step_extract(
                input_video=c["video_path"],
                out_dir=paths["frames_raw"],
                fps=float(c.get("fps", cfg["extract"]["fps"])),
                qscale=int(cfg["extract"]["qscale"]),
                pattern=cfg["extract"]["filename_pattern"],
                start=cfg["extract"].get("start"),
                end=cfg["extract"].get("end"),
                meta_path=os.path.join(paths["metadata"], "extract.json"),
                allow_overwrite=cfg.get("allow_overwrite", False),
                allow_append_to_existing=cfg.get("allow_append_to_existing", False),
            )

def run_dedup(cfg, cases_sel):
    for c in cfg["cases"]:
        if c["case_id"] not in cases_sel: continue
        paths = step_paths(cfg, c["case_id"])
        ensure_dir(paths["frames_dedup"])
        guard_overwrite(cfg, paths["frames_dedup"])
        print(f">> [dedup] {c['case_id']} {paths['frames_raw']} -> {paths['frames_dedup']}")
        if not cfg.get("dry_run", False):
            step_dedup(
                in_dir=paths["frames_raw"],
                out_dir=paths["frames_dedup"],
                use_ssim=cfg["dedup"]["use_ssim"],
                ssim_threshold=float(cfg["dedup"]["ssim_threshold"]),
                blur_filter=cfg["dedup"]["blur_filter"],
                laplacian_var_min=float(cfg["dedup"]["laplacian_var_min"]),
                meta_path=os.path.join(paths["metadata"], "dedup.json"),
                allow_overwrite=cfg.get("allow_overwrite", False),
                allow_append_to_existing=cfg.get("allow_append_to_existing", False),

            )


def run_stats(cfg):
    run_root = os.path.join(cfg["work_root"], cfg["run_id"])
    print(f">> [stats] {run_root}")
    step_compute_stats(cfg)


def run_tags(cfg, cases_sel=None):
    run_root = os.path.join(cfg["work_root"], cfg["run_id"])
    step_tags_csv(run_root, cfg)

def run_crop(cfg, cases_sel):
    for c in cfg["cases"]:
        if c["case_id"] not in cases_sel: continue
        paths = step_paths(cfg, c["case_id"])
        ensure_dir(paths["frames_cropped"])
        guard_overwrite(cfg, paths["frames_cropped"])
        print(f">> [crop] {c['case_id']} {paths['frames_dedup']} -> {paths['frames_cropped']}")
        if not cfg.get("dry_run", False):
            step_crop(
                in_dir=paths["frames_dedup"],
                out_dir=paths["frames_cropped"],
                method=cfg["crop"]["method"],
                roi_margin_px=int(cfg["crop"]["roi_margin_px"]),
                sample_stride=int(cfg["crop"]["sample_stride"]),
                per_video_fixed_roi=bool(cfg["crop"]["per_video_fixed_roi"]),
                meta_path=os.path.join(paths["metadata"], "crop.json"),
                allow_overwrite=cfg.get("allow_overwrite", False),
                allow_append_to_existing = cfg.get("allow_append_to_existing", False),
            )

def run_maskcrop(cfg, cases_sel):
    for c in cfg["cases"]:
        if c["case_id"] not in cases_sel: continue
        p = step_paths(cfg, c["case_id"])

        in_masks = p["masks_multiclass"]    # Supervisely masks live here

        os.makedirs(p["metadata"], exist_ok=True)
        print(f">> [maskcrop] {c['case_id']} using {in_masks}")
        if not cfg.get("dry_run", False):
            step_maskcrop(
                in_color_dir=p["frames_dedup"],  # or frames_cropped
                in_mask_dir=p["masks_multiclass"],  # or masks_binary
                out_color_dir=os.path.join(p["base"], "frames_maskcrop"),
                out_mask_dir=os.path.join(p["base"], "masks_multiclass_crop"),
                per_video_fixed_roi=cfg["maskcrop"]["per_video_fixed_roi"],
                sample_stride=cfg["maskcrop"]["sample_stride"],
                margin=cfg["maskcrop"]["margin_px"],
                use_color_union=cfg["maskcrop"]["use_color_union"],
                meta_path=os.path.join(p["metadata"], "maskcrop.json"),
                per_frame_meta_path=os.path.join(p["metadata"], "maskcrop_per_frame.json"),
                allow_overwrite=cfg.get("maskcrop", {}).get("allow_overwrite", False),
                allow_append_to_existing=cfg.get("maskcrop", {}).get("allow_append_to_existing", False),
            )


def run_bin(cfg, cases_sel):
    for c in cfg["cases"]:
        if c["case_id"] not in cases_sel: continue
        paths = step_paths(cfg, c["case_id"])
        ensure_dir(paths["masks_binary"])
        guard_overwrite(cfg, paths["masks_binary"])
        if cfg.get("crop_using_masks", False):
            multiclass_dir = paths['masks_multiclass_crop']
        else:
            multiclass_dir = paths['masks_multiclass']
        print(f">> [binmask] {c['case_id']} {multiclass_dir} -> {paths['masks_binary']}")
        if not cfg.get("dry_run", False):
            step_bin(
                in_dir=multiclass_dir,
                out_dir=paths["masks_binary"],
                meta_path=os.path.join(paths["metadata"], "binary_masks.json"),
                allow_overwrite=cfg.get("allow_overwrite", False),
                allow_append_to_existing = cfg.get("binmask", {}).get("allow_append_to_existing", False)
            )


def run_split(cfg):
    base = os.path.join(cfg["work_root"], cfg["run_id"])
    print(f">> [split] writing train/val/test to {base}/splits")
    if not cfg.get("dry_run", False):
        step_split(
            run_root=base,
            splits_cfg=cfg["splits"],
        )


def _pick_preview_source(case_root: str) -> str:
    for name in ["frames_maskcrop", "frames_cropped", "frames_dedup", "frames_raw"]:
        cand = os.path.join(case_root, name)
        if os.path.isdir(cand) and any(os.scandir(cand)):
            return cand
    # fallback
    return os.path.join(case_root, "frames_dedup")



def run_previews(cfg, cases_sel):
    if not cfg.get("previews").get("enabled", "False"):
        print(">> [previews] skipping previews")
        return
    for c in cfg["cases"]:
        if c["case_id"] not in cases_sel:
            continue
        case_root = step_paths(cfg, c["case_id"])["base"]
        print(f">> [previews] {c['case_id']} -> annotator_previews/")
        if not cfg.get("dry_run", False):
            previews_run(case_root, cfg)

def run_resize(cfg, cases_sel):
    for c in cfg["cases"]:
        if c["case_id"] not in cases_sel:
            continue
        p = step_paths(cfg, c["case_id"])
        # choose source: after maskcrop so pairs align
        src_img = p.get("frames_maskcrop", os.path.join(p["base"], "frames_maskcrop"))
        src_mask = p.get("masks_multiclass_crop", os.path.join(p["base"], "masks_multiclass_crop"))

        out_img = p["frames_resized"]
        out_msk = p["masks_resized"]
        os.makedirs(p["metadata"], exist_ok=True)

        print(f">> [resize] {c['case_id']} {src_img} -> {out_img}  (masks -> {out_msk})")
        if not cfg.get("dry_run", False):
            step_resize(
                in_color_dir=src_img,
                in_mask_dir=src_mask,
                out_color_dir=out_img,
                out_mask_dir=out_msk,
                target_size=cfg["resize"]["target_size"],
                method=cfg["resize"]["method"],
                pad_value_rgb=cfg["resize"]["pad_value_rgb"],
                pad_value_mask=cfg["resize"]["pad_value_mask"],
                meta_path=os.path.join(p["metadata"], "resize_pairs.json"),
                allow_overwrite=cfg["resize"]["allow_overwrite"],
                allow_append_to_existing=cfg.get("resize", {}).get("allow_append_to_existing", False)
            )

def run_training_previews(cfg, cases_sel):
    print(">> [preview_augs]")
    if not cfg.get("dry_run", False):
        step_preview_augs(cfg, cases_sel)

def run_train(cfg, cases_sel):
    print(">> [train] starting…")
    if not cfg.get("dry_run", False):
        step_train(cfg, cases_sel)

def run_eval(cfg, cases_sel, split="test", chk_pt=None):
    if cfg.get("train")['eval_topk']:
        print(">> [eval] topk")
        return step_eval_topk(cfg, split=split)
    else:
        print(">> [eval] best")
        return step_eval(cfg, split=split, chk_pt=chk_pt)

def main():
    ap = argparse.ArgumentParser(description="Retinal segmentation preprocessing pipeline")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--cmd", choices=["extract", "dedup", "crop", "maskcrop", "binmask", "resize", "split", "tags", "stats", "previews","preview_augs","train", "eval","all"], required=True)
    ap.add_argument("--case", action="append", help="Process only these case_id(s). Can repeat.")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--work_root", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.work_root: cfg["work_root"] = args.work_root
    if args.data_root: cfg["data_root"] = args.data_root
    if args.run_id:    cfg["run_id"]    = args.run_id
    if args.force:     cfg["allow_overwrite"] = True
    if args.dry_run:   cfg["dry_run"] = True
    cfg = resolve_placeholders(cfg)

    cases_sel = list_cases(cfg, only=args.case)
    if not cases_sel:
        cases_sel = list_cases(cfg)

    if args.cmd in ("extract","all"):
        run_extract(cfg, cases_sel)
    if args.cmd in ("dedup","all"):
        run_dedup(cfg, cases_sel)
    # if args.cmd in ("crop","all"):
    #     run_crop(cfg, cases_sel)

    # binmask and maskcrop depends on manual Supervisely export into masks_multiclass/
    if args.cmd in ("maskcrop","all"):
        run_maskcrop(cfg, cases_sel)
    if args.cmd in ("binmask","all"):
        run_bin(cfg, cases_sel)
    if args.cmd in ("tags", "all"):
        run_tags(cfg, cases_sel)
    if args.cmd in ("split","all"):
        run_split(cfg)
    if args.cmd in ("previews","all"):
        run_previews(cfg, cases_sel)
    if args.cmd in ("resize","all"):
        run_resize(cfg, cases_sel)
    if args.cmd in ("stats","all"):
        run_stats(cfg)
    if args.cmd == "preview_augs":
        run_training_previews(cfg, cases_sel)
    if args.cmd == "train":
        run_train(cfg, cases_sel)
    if args.cmd == "eval":
        run_eval(cfg, cases_sel, split="test", chk_pt=args.ckpt)

if __name__ == "__main__":
    main()