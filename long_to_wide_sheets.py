#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
long_to_wide_sheets.py

Convert a "long" table (rows=observations, one column=label/region) into
an Excel workbook where each sheet is one parameter in "wide" format:
columns=labels, rows=replicates (observations).

- Auto-detect label column (case-insensitive, common names)
- Auto-detect numeric parameter columns
- Keep varying replicate counts (columns can have different lengths)
- Add a "manifest" sheet summarizing counts per label per parameter

Usage:
  python long_to_wide_sheets.py --input analysis_indices_prepared.csv --output wide_by_param.xlsx
Optional:
  --label-col label
  --include-cols anisotropy skewness
  --exclude-cols some_col1 some_col2
  --label-order PC GC Top Mid Btm RH DZ EZ TZ MZ
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Input long-format CSV/TSV/Excel")
    p.add_argument("--output", type=Path, required=True, help="Output Excel .xlsx")
    p.add_argument("--label-col", type=str, default=None,
                   help="Label column name. If not set, auto-detect.")
    p.add_argument("--label-order", nargs="+", default=None,
                   help="Optional explicit label order for columns.")
    p.add_argument("--include-cols", nargs="+", default=None,
                   help="Only export these parameter columns (besides label).")
    p.add_argument("--exclude-cols", nargs="+", default=None,
                   help="Exclude these columns from parameters.")
    p.add_argument("--dropna-label", action="store_true", help="Drop rows with missing label")
    p.add_argument("--sheet-name-maxlen", type=int, default=31,
                   help="Excel sheet name max length (default 31)")
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suf in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t")
    # default csv
    return pd.read_csv(path)


def autodetect_label_col(df: pd.DataFrame) -> str:
    # Common candidates (case-insensitive)
    candidates = ["label", "region", "group", "zone", "class"]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]

    # Try fuzzy match: any col containing these tokens
    for c in df.columns:
        cl = c.lower()
        if any(tok in cl for tok in candidates):
            return c

    raise ValueError(
        "Could not auto-detect label column. Use --label-col to specify it."
    )


def clean_sheet_name(name: str, maxlen: int = 31) -> str:
    # Excel forbids: : \ / ? * [ ]
    s = re.sub(r"[:\\/?*\[\]]", "_", str(name)).strip()
    if not s:
        s = "sheet"
    return s[:maxlen]


def choose_parameter_columns(
    df: pd.DataFrame,
    label_col: str,
    include_cols: Optional[List[str]],
    exclude_cols: Optional[List[str]],
) -> List[str]:
    cols = [c for c in df.columns if c != label_col]

    if include_cols:
        # keep only those that exist
        wanted = []
        inc_lower = {x.lower() for x in include_cols}
        for c in cols:
            if c.lower() in inc_lower:
                wanted.append(c)
        cols = wanted

    if exclude_cols:
        exc_lower = {x.lower() for x in exclude_cols}
        cols = [c for c in cols if c.lower() not in exc_lower]

    # Keep numeric-like columns:
    # convertable to numeric for at least some rows.
    numeric_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(s).any():
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError("No numeric parameter columns detected.")
    return numeric_cols


def long_to_wide(df: pd.DataFrame, label_col: str, value_col: str, label_order: Optional[List[str]]) -> pd.DataFrame:
    """
    Return wide table:
    - columns: labels
    - rows: replicate index 1..N (max within labels)
    """
    sub = df[[label_col, value_col]].copy()
    sub[label_col] = sub[label_col].astype(str).str.strip()

    # numeric coercion
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[value_col])

    # group -> list
    grouped = sub.groupby(label_col)[value_col].apply(list).to_dict()

    # order labels
    if label_order:
        labels = [lab for lab in label_order if lab in grouped]
        # append any missing labels not specified
        labels += [lab for lab in grouped.keys() if lab not in labels]
    else:
        labels = sorted(grouped.keys())

    max_len = max((len(grouped[lab]) for lab in labels), default=0)
    data: Dict[str, List[float]] = {}
    for lab in labels:
        vals = grouped.get(lab, [])
        # pad with NaN to max_len (Excel-friendly)
        padded = vals + [np.nan] * (max_len - len(vals))
        data[lab] = padded

    wide = pd.DataFrame(data)
    wide.index = np.arange(1, max_len + 1)
    wide.index.name = "replicate"
    return wide


def build_manifest(df: pd.DataFrame, label_col: str, param_cols: List[str]) -> pd.DataFrame:
    """
    Manifest: for each parameter and label, how many finite values exist.
    """
    rows = []
    labels = sorted(df[label_col].dropna().astype(str).str.strip().unique().tolist())

    for p in param_cols:
        s = pd.to_numeric(df[p], errors="coerce")
        tmp = df[[label_col]].copy()
        tmp["_v"] = s
        tmp[label_col] = tmp[label_col].astype(str).str.strip()
        counts = tmp.groupby(label_col)["_v"].apply(lambda x: int(np.isfinite(x).sum())).to_dict()

        row = {"parameter": p}
        for lab in labels:
            row[lab] = counts.get(lab, 0)
        row["total_n"] = int(np.isfinite(s).sum())
        rows.append(row)

    manifest = pd.DataFrame(rows)
    # put total_n near front
    cols = ["parameter", "total_n"] + [c for c in manifest.columns if c not in ("parameter", "total_n")]
    return manifest[cols]


def main() -> None:
    args = parse_args()

    df = read_table(args.input)

    label_col = args.label_col or autodetect_label_col(df)
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in columns: {list(df.columns)}")

    if args.dropna_label:
        df = df.dropna(subset=[label_col])

    # normalize labels
    df[label_col] = df[label_col].astype(str).str.strip()

    param_cols = choose_parameter_columns(df, label_col, args.include_cols, args.exclude_cols)

    # Write workbook
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        # manifest first
        manifest = build_manifest(df, label_col, param_cols)
        manifest.to_excel(writer, sheet_name="manifest", index=False)

        used_sheet_names = set(["manifest"])
        for pcol in param_cols:
            wide = long_to_wide(df, label_col, pcol, args.label_order)
            sheet = clean_sheet_name(pcol, maxlen=args.sheet_name_maxlen)

            # avoid duplicate sheet names (after truncation)
            base = sheet
            k = 1
            while sheet in used_sheet_names:
                suffix = f"_{k}"
                sheet = (base[: args.sheet_name_maxlen - len(suffix)] + suffix) if len(base) + len(suffix) > args.sheet_name_maxlen else base + suffix
                k += 1

            wide.to_excel(writer, sheet_name=sheet, index=True)
            used_sheet_names.add(sheet)

    print(f"Done.\nLabel column: {label_col}\nParameters exported: {len(param_cols)}\nOutput: {args.output}")


if __name__ == "__main__":
    main()
