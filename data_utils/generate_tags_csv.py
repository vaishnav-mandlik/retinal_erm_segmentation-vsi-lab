# data_utils/generate_tags_csv.py
import os, json, csv, re
from pathlib import Path
from typing import Dict, List, Any, Tuple


# ------------------------- helpers -------------------------

_NORM_RE = re.compile(r"[^a-z0-9]+")

def _norm(s: str) -> str:
    """Lowercase; replace non-alnum with single space; collapse spaces; trim."""
    if s is None:
        return ""
    s = s.lower()
    s = _NORM_RE.sub(" ", s)
    s = " ".join(s.split())
    return s

def _iter_case_dirs(run_root: str) -> List[Path]:
    root = Path(run_root)
    return [
        p for p in sorted(root.iterdir())
        if p.is_dir() and p.name.lower().startswith("case")
    ]

def _iter_ann_files(ann_dir: Path) -> List[Path]:
    return sorted(ann_dir.glob("*.json")) if ann_dir.is_dir() else []

def _extract_phase(ann: Dict[str, Any],
                   phase_tag_name: str,
                   valid_phase_values: List[str]) -> Tuple[str, int]:
    phase_val = ""
    phase_valid = 0
    want = phase_tag_name
    for t in ann.get("tags", []) or []:
        if t.get("name") == want:
            phase_val = t.get("value") or ""
            if valid_phase_values:
                phase_valid = int(phase_val in valid_phase_values)
            else:
                phase_valid = 1
            break
    return phase_val, phase_valid

def _prepare_class_maps(class_to_boolean: Dict[str, str],
                        class_aliases: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns (norm_key -> bool_col, unmatched_seen dict).
    Also expands aliases: if aliases map 'epiretinal membrane' -> 'ERM',
    we route that alias to the bool column for 'ERM' (if present).
    """
    norm_map: Dict[str, str] = {}
    # canonical keys
    for k, bool_col in (class_to_boolean or {}).items():
        norm_map[_norm(k)] = bool_col

    # aliases (optional)
    for alias, canonical in (class_aliases or {}).items():
        canon_norm = _norm(canonical)
        alias_norm = _norm(alias)
        if canon_norm in norm_map:
            norm_map[alias_norm] = norm_map[canon_norm]

    return norm_map, {}  # second is placeholder for type symmetry

def _match_bool_col(norm_title: str, norm_map: Dict[str, str]) -> str:
    """
    1) exact normalized match
    2) substring either way (contains / contained-by)
    Returns bool_col or "" if not matched.
    """
    if norm_title in norm_map:
        return norm_map[norm_title]
    # substring passes
    for k_norm, bcol in norm_map.items():
        if k_norm in norm_title or norm_title in k_norm:
            return bcol
    return ""

def _extract_object_flags(
    ann: Dict[str, Any],
    norm_map: Dict[str, str],
    instrument_classes_norm: List[str],
    unmatched_once: set
) -> Tuple[int, int, Dict[str, int]]:
    objs = ann.get("objects", []) or []
    object_count = len(objs)

    # init all bools to 0 (cover exactly columns in norm_map values)
    bool_cols = {bcol: 0 for bcol in set(norm_map.values())}

    instrument_count = 0
    for obj in objs:
        title = obj.get("classTitle") or ""
        nt = _norm(title)

        # flip per-class boolean if mappable
        bcol = _match_bool_col(nt, norm_map)
        if bcol:
            bool_cols[bcol] = 1
        else:
            # record unseen titles once
            if nt and nt not in unmatched_once:
                print(f"[tags][WARN] Unmatched classTitle seen: '{title}' (norm='{nt}')")
                unmatched_once.add(nt)

        # instrument_count heuristic:
        # treat any object whose normalized title matches any instrument class key
        # (from mapping keys except 'ERM') as an instrument
        if instrument_classes_norm:
            # if nt matches any instrument class (exact or substring)
            if any(ic == nt or ic in nt or nt in ic for ic in instrument_classes_norm):
                instrument_count += 1
        else:
            # fallback: everything except 'erm' counted as instrument
            if nt != "erm":
                instrument_count += 1

    return object_count, instrument_count, bool_cols


def _write_csv(path: Path,
               rows: List[Dict[str, Any]],
               bool_cols_order: List[str]) -> None:
    if not rows:
        return
    fieldnames = [
        "case_id", "image_name", "phase", "phase_valid",
        "object_count", "instrument_count",
    ] + bool_cols_order
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ------------------------- main entry -------------------------

def run(run_root: str, cfg: Dict, **kwargs) -> None:
    """
    Builds per-case tags.csv and a run-level tags_all_cases.csv.

    Config used:
      tags:
        phase_tag_name: "surgery phase"
        valid_phase_values: ["dye injection", "flap initiation", "peeling progression", "completion"]
        class_to_boolean:
          "ERM": "membrane_present"
          "Forceps": "forceps_present"
          "Light tool": "light_tool_present"
        # optional synonyms (alias -> canonical_key_from_above)
        class_aliases:
          "epiretinal membrane": "ERM"
          "forceps 1": "Forceps"
          "light": "Light tool"
    """
    tags_cfg = cfg.get("tags", {})
    phase_tag_name = tags_cfg.get("phase_tag_name", "surgery phase")
    valid_phase_values = tags_cfg.get("valid_phase_values", []) or []
    class_to_boolean: Dict[str, str] = tags_cfg.get("class_to_boolean", {}) or {}
    class_aliases: Dict[str, str] = tags_cfg.get("class_aliases", {}) or {}

    # Build normalized lookups
    norm_map, _ = _prepare_class_maps(class_to_boolean, class_aliases)
    bool_cols_order = list(set(norm_map.values()))  # stable order not guaranteed; fine for CSV
    # instrument classes = everything in map EXCEPT whatever maps from 'ERM' (if present)
    # identify which norm keys map to which bool cols
    erm_norm_keys = {k for k, v in norm_map.items() if v == class_to_boolean.get("ERM", "")}
    instrument_classes_norm = [k for k in norm_map.keys() if k not in erm_norm_keys]

    all_rows: List[Dict[str, Any]] = []
    unmatched_once: set = set()

    for case_dir in _iter_case_dirs(run_root):
        case_id = case_dir.name
        ann_dir = case_dir / "supervisely_export" / "ann"
        if not ann_dir.is_dir():
            print(f"[tags] Skip {case_id}: missing {ann_dir}")
            continue

        out_case_csv = case_dir / "metadata" / "tags.csv"
        rows: List[Dict[str, Any]] = []

        for ann_path in _iter_ann_files(ann_dir):
            with open(ann_path, "r") as f:
                ann = json.load(f)

            image_name = ann.get("name") or ann_path.stem

            phase_val, phase_valid = _extract_phase(ann, phase_tag_name, valid_phase_values)
            object_count, instrument_count, bool_cols = _extract_object_flags(
                ann, norm_map, instrument_classes_norm, unmatched_once
            )

            row = {
                "case_id": case_id,
                "image_name": image_name,
                "phase": phase_val,
                "phase_valid": phase_valid,
                "object_count": object_count,
                "instrument_count": instrument_count,
            }
            row.update(bool_cols)
            rows.append(row)
            all_rows.append(row)

        _write_csv(out_case_csv, rows, bool_cols_order)
        print(f"[tags] wrote {len(rows)} rows for {case_id} → {out_case_csv}")

    # run-level concatenation
    if all_rows:
        out_all = Path(run_root) / "metadata" / "all_tags.csv"
        _write_csv(out_all, all_rows, bool_cols_order)
        print(f"[tags] wrote {len(all_rows)} total rows → {out_all}")