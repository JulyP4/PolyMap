"""Prepare analysis_indices.xlsx for the multidimensional actin pipeline.

This script converts a raw Excel/CSV table into a per-cell CSV compatible with
``actin_multidim_analysis.py`` and optionally emits a JSON schema describing
feature metadata (exclusions, non-normalized indices, feature groups).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

FOLDER_KEY_MAP = {
    "Lifeact-cotyleden": "PC",
    "Lifeact-hypocotyl/Lifeact-DMSO 12h base": "Btm",
    "Lifeact-hypocotyl/Lifeact-DMSO 12h mid": "Mid",
    "Lifeact-hypocotyl/Lifeact-DMSO 12h upper": "Top",
    "Lifeact-root/differentation": "DZ",
    "Lifeact-root/elongation": "EZ",
    "Lifeact-root/meristem": "MZ",
    "Lifeact-root/transition": "TZ",
    "Lifeact-roothair": "RH",
}

METRIC_GROUPS = {
    "Density": ["occupancy", "linear_density"],
    "Bundling": [
        "skewness",
        "cv",
        "diameter_tdt",
        "diameter_sdt",
    ],
    "Connectivity": ["segment_density"],
    "Branching": ["branching_activity"],
    "Directionality": ["anisotropy"],
}

NON_NORMALIZED = [
    "total_length",
    "total_element",
    "total_node",
    "total_graph_theoretic_branch",
]

EXCLUDE_COLUMNS = {"k1", "k2", "file_path"}
DEFAULT_RENAME_MAP = {
    "linear_density (PU/PU^2)": "linear_density",
    "diameter_tdt (PU)": "diameter_tdt",
    "diameter_sdt (PU)": "diameter_sdt",
    "segment_density (/PU of filament)": "segment_density",
    "branching_act(/PU of filament)": "branching_activity",
    "nsi_total_length": "total_length",
    "nsi_total_element": "total_element",
    "nsi_total_node": "total_node",
    "nsi_total_graph_theoretic_branch": "total_graph_theoretic_branch"
}


def _split_list_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_json(path: Optional[str]) -> Optional[Dict[str, object]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON at {path} must be an object.")
    return data


def _map_labels(series: pd.Series, label_map: Dict[str, str]) -> pd.Series:
    mapped = series.map(label_map)
    return mapped.fillna(series)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare analysis_indices.xlsx for multidimensional analysis."
    )
    parser.add_argument("input", type=str, help="Input .xlsx or .csv file")
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_indices_prepared.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="0",
        help="Sheet name or index for Excel input (default: 0)",
    )
    parser.add_argument(
        "--label-source",
        type=str,
        default="folder_key",
        help="Column to use as label source (default: folder_key)",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        default="label",
        help="Label column name in output CSV (default: label)",
    )
    parser.add_argument(
        "--id-source",
        type=str,
        default="file_name",
        help="Column to use as id source (default: file_name)",
    )
    parser.add_argument(
        "--id-name",
        type=str,
        default="id",
        help="ID column name in output CSV (default: id)",
    )
    parser.add_argument(
        "--exclude-cols",
        type=str,
        default="k1,k2,file_path",
        help="Comma-separated columns to exclude (default: k1,k2,file_path)",
    )
    parser.add_argument(
        "--non-normalized-cols",
        type=str,
        default="nsi_total_element,nsi_total_node,nsi_total_graph_theoretic_branch",
        help="Comma-separated non-normalized index columns",
    )
    parser.add_argument(
        "--rename-map",
        type=str,
        default=None,
        help="Path to JSON mapping of column renames (merged with defaults)",
    )
    parser.add_argument(
        "--feature-groups",
        type=str,
        default=None,
        help="Path to JSON file defining feature groups",
    )
    parser.add_argument(
        "--folder-key-map",
        type=str,
        default=None,
        help="Path to JSON mapping of folder_key to English labels",
    )
    parser.add_argument(
        "--schema-out",
        type=str,
        default=None,
        help="Optional output path for JSON schema",
    )
    parser.add_argument(
        "--include-non-normalized",
        action="store_true",
        help="Set include_non_normalized=true in the schema output",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        sheet_value: object = int(args.sheet) if args.sheet.isdigit() else args.sheet
        df = pd.read_excel(input_path, sheet_name=sheet_value)
    else:
        df = pd.read_csv(input_path)

    df = df.dropna(axis=1, how="all")

    rename_map = DEFAULT_RENAME_MAP.copy()
    rename_map.update(_load_json(args.rename_map) or {})
    if rename_map:
        df = df.rename(columns=rename_map)

    if args.label_source in df.columns and args.label_name not in df.columns:
        label_map = _load_json(args.folder_key_map) or FOLDER_KEY_MAP
        df[args.label_name] = _map_labels(df[args.label_source], label_map)
    if args.id_source in df.columns and args.id_name not in df.columns:
        df[args.id_name] = df[args.id_source]

    exclude_cols = set(_split_list_arg(args.exclude_cols)) | EXCLUDE_COLUMNS
    df = df.drop(columns=[col for col in df.columns if col in exclude_cols])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if args.schema_out:
        exclude_features = sorted(set(_split_list_arg(args.exclude_cols)) | EXCLUDE_COLUMNS)
        schema: Dict[str, object] = {
            "label_col": args.label_name,
            "id_col": args.id_name,
            "exclude_features": exclude_features,
            "non_normalized_features": _split_list_arg(args.non_normalized_cols),
            "include_non_normalized": args.include_non_normalized,
        }

        if rename_map:
            schema["feature_rename"] = rename_map

        groups = _load_json(args.feature_groups) or METRIC_GROUPS
        if groups:
            schema["feature_groups"] = groups

        label_map = _load_json(args.folder_key_map) or FOLDER_KEY_MAP
        if label_map:
            schema["label_map"] = label_map

        schema_path = Path(args.schema_out)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Saved prepared CSV to {output_path}")
    if args.schema_out:
        print(f"Saved schema JSON to {args.schema_out}")


if __name__ == "__main__":
    main()
