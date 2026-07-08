import os
import csv
from collections import defaultdict
from typing import Dict, Any, List, Tuple


def _get_run_root(cfg: Dict[str, Any]) -> str:
    return os.path.join(cfg["work_root"], cfg["run_id"])


def _load_all_tags(run_root: str, cfg: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    tags_cfg = cfg.get("tags", {})
    stats_cfg = tags_cfg.get("stats", {})
    all_tags_name = stats_cfg.get("all_tags_csv", "all_tags.csv")
    path = os.path.join(run_root, "metadata", all_tags_name)
    rows: List[Dict[str, Any]] = []

    if not os.path.isfile(path):
        print(f"[stats] WARN: all_tags.csv not found at {path}")
        return path, rows

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return path, rows


def _to_int(row: Dict[str, Any], key: str) -> int:
    v = row.get(key, "")
    if v == "" or v is None:
        return 0
    try:
        return int(v)
    except ValueError:
        try:
            return int(float(v))
        except ValueError:
            return 0


def _phase_cols(cfg: Dict[str, Any]) -> List[str]:
    vals = cfg.get("tags", {}).get("valid_phase_values", [])
    # convert to safe column names, e.g. "dye injection" -> "phase_dye_injection"
    cols = []
    for v in vals:
        c = "phase_" + v.lower().replace(" ", "_")
        cols.append(c)
    return cols


def _class_bool_cols(cfg: Dict[str, Any]) -> List[str]:
    return list(cfg.get("tags", {}).get("class_to_boolean", {}).values())


def _aggregate_case_stats(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Returns: {case_id: stats_dict}
    Stats columns:
      num_frames, membrane_frames, instrument_frames, per-phase counts, per-class-bool counts.
    """
    phase_cols = _phase_cols(cfg)
    bool_cols = _class_bool_cols(cfg)

    per_case: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(int))

    for r in rows:
        case_id = r.get("case_id", "UNKNOWN")
        pc = per_case[case_id]

        pc["num_frames"] += 1

        # instrument_frames: any instrument_count > 0
        if _to_int(r, "instrument_count") > 0:
            pc["instrument_frames"] += 1

        # if we have a membrane_present (or other bools)
        for bc in bool_cols:
            if _to_int(r, bc) > 0:
                pc[f"{bc}_frames"] += 1

        # phase columns from canonical values
        phase_val = (r.get("phase") or "").strip().lower()
        if phase_val:
            for v in cfg.get("tags", {}).get("valid_phase_values", []):
                if phase_val == v.lower():
                    col = "phase_" + v.lower().replace(" ", "_")
                    pc[col] += 1

    # Ensure missing keys are present with 0
    for cid, s in per_case.items():
        s.setdefault("num_frames", 0)
        s.setdefault("instrument_frames", 0)
        for bc in bool_cols:
            s.setdefault(f"{bc}_frames", 0)
        for pc_col in phase_cols:
            s.setdefault(pc_col, 0)

    return per_case


def _write_case_stats(run_root: str, cfg: Dict[str, Any], per_case: Dict[str, Dict[str, Any]]) -> str:
    tags_cfg = cfg.get("tags", {})
    stats_cfg = tags_cfg.get("stats", {})
    fname = stats_cfg.get("case_stats_csv", "case_stats.csv")
    out_path = os.path.join(run_root, "metadata", fname)

    bool_cols = _class_bool_cols(cfg)
    phase_cols = _phase_cols(cfg)

    fieldnames = (
        ["case_id", "num_frames", "instrument_frames"]
        + [f"{bc}_frames" for bc in bool_cols]
        + phase_cols
    )

    # Totals accumulator
    totals = defaultdict(int)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cid, stats in sorted(per_case.items()):
            row = {"case_id": cid}
            for col in fieldnames[1:]:
                v = int(stats.get(col, 0))
                row[col] = v
                totals[col] += v
            w.writerow(row)

        # Total row
        total_row = {"case_id": "Total"}
        for col in fieldnames[1:]:
            total_row[col] = totals[col]
        w.writerow(total_row)

        # TotalPct row (percent of total frames)
        total_frames = totals["num_frames"] if totals["num_frames"] > 0 else 1
        pct_row = {"case_id": "TotalPct"}
        for col in fieldnames[1:]:
            pct_row[col] = round(100.0 * totals[col] / total_frames, 2)
        w.writerow(pct_row)

    print(f"[stats] wrote case-level stats to {out_path}")
    return out_path


def _aggregate_split_stats(per_case: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Summarize statistics per split using cfg['splits'].
    """
    splits_cfg = cfg.get("splits", {})
    bool_cols = _class_bool_cols(cfg)
    phase_cols = _phase_cols(cfg)

    per_split: Dict[str, Dict[str, Any]] = {}

    for split_name, case_ids in splits_cfg.items():
        agg = defaultdict(int)
        for cid in case_ids or []:
            s = per_case.get(cid)
            if not s:
                continue
            agg["num_frames"] += int(s.get("num_frames", 0))
            agg["instrument_frames"] += int(s.get("instrument_frames", 0))
            for bc in bool_cols:
                agg[f"{bc}_frames"] += int(s.get(f"{bc}_frames", 0))
            for pc_col in phase_cols:
                agg[pc_col] += int(s.get(pc_col, 0))
        per_split[split_name] = agg

    return per_split


def _write_split_stats(run_root: str, cfg: Dict[str, Any], per_split: Dict[str, Dict[str, Any]]) -> str:
    tags_cfg = cfg.get("tags", {})
    stats_cfg = tags_cfg.get("stats", {})
    fname = stats_cfg.get("split_stats_csv", "split_stats.csv")
    out_path = os.path.join(run_root, "metadata", fname)

    bool_cols = _class_bool_cols(cfg)
    phase_cols = _phase_cols(cfg)

    fieldnames = (
        ["split", "num_frames", "instrument_frames"]
        + [f"{bc}_frames" for bc in bool_cols]
        + phase_cols
    )

    totals = defaultdict(int)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for split_name, stats in sorted(per_split.items()):
            row = {"split": split_name}
            for col in fieldnames[1:]:
                v = int(stats.get(col, 0))
                row[col] = v
                totals[col] += v
            w.writerow(row)

        total_row = {"split": "Total"}
        for col in fieldnames[1:]:
            total_row[col] = totals[col]
        w.writerow(total_row)

        total_frames = totals["num_frames"] if totals["num_frames"] > 0 else 1
        pct_row = {"split": "TotalPct"}
        for col in fieldnames[1:]:
            pct_row[col] = round(100.0 * totals[col] / total_frames, 2)
        w.writerow(pct_row)

    print(f"[stats] wrote split-level stats to {out_path}")
    return out_path


def run(cfg: Dict[str, Any]) -> None:
    """
    Main entry: load all_tags.csv, compute per-case and per-split statistics.
    """
    run_root = _get_run_root(cfg)
    all_path, rows = _load_all_tags(run_root, cfg)
    if not rows:
        print(f"[stats] No rows found in {all_path}; nothing to summarize.")
        return

    per_case = _aggregate_case_stats(rows, cfg)
    _write_case_stats(run_root, cfg, per_case)

    per_split = _aggregate_split_stats(per_case, cfg)
    if per_split:
        _write_split_stats(run_root, cfg, per_split)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Compute case- and split-level stats from all_tags.csv")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run(cfg)