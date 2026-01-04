#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_root_heatmap.py

Modes:
- stats != 'real': read summary_per_label.csv and color each SVG element by label summary value
- stat == 'real': read raw observation CSV (per-row observation), grouped by label; for each SVG sub-shape
  assigned to that label, sample a REAL observed value (optionally filtered) and color it

Key constraints:
- Preserve original SVG stroke/stroke-width exactly:
  -> ONLY update fill inside 'style' string; DO NOT overwrite other style fields; DO NOT set attribute 'fill' directly.

- Use ImageJ Fire LUT EXACTLY (256 entries as provided by user), no invert.
- Fill mapping can be restricted to colormap subrange [cmap_range_lo, cmap_range_hi] (visual-only).
- Colorbar:
  -> placed INSIDE the figure at bottom-right (default), with black frame (optional),
  -> shows FULL colormap 0..1 (not cropped to cmap-range),
  -> scalable via --colorbar-scale (width/height/frame/ticks/font scale together).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import matplotlib
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------
# ImageJ Fire LUT (exact 256xRGB as user provided)
# ---------------------------------------------------------------------
IMAGEJ_FIRE_LUT_256: List[Tuple[int, int, int]] = [
    tuple(x) for x in [[0, 0, 0], [0, 0, 7], [0, 0, 15], [0, 0, 22], [0, 0, 30], [0, 0, 38], [0, 0, 45],
                       [0, 0, 53], [0, 0, 61], [0, 0, 65], [0, 0, 69], [0, 0, 74], [0, 0, 78], [0, 0, 82],
                       [0, 0, 87], [0, 0, 91], [1, 0, 96], [4, 0, 100], [7, 0, 104], [10, 0, 108],
                       [13, 0, 113], [16, 0, 117], [19, 0, 121], [22, 0, 125], [25, 0, 130], [28, 0, 134],
                       [31, 0, 138], [34, 0, 143], [37, 0, 147], [40, 0, 151], [43, 0, 156], [46, 0, 160],
                       [49, 0, 165], [52, 0, 168], [55, 0, 171], [58, 0, 175], [61, 0, 178], [64, 0, 181],
                       [67, 0, 185], [70, 0, 188], [73, 0, 192], [76, 0, 195], [79, 0, 199], [82, 0, 202],
                       [85, 0, 206], [88, 0, 209], [91, 0, 213], [94, 0, 216], [98, 0, 220], [101, 0, 220],
                       [104, 0, 221], [107, 0, 222], [110, 0, 223], [113, 0, 224], [116, 0, 225],
                       [119, 0, 226], [122, 0, 227], [125, 0, 224], [128, 0, 222], [131, 0, 220],
                       [134, 0, 218], [137, 0, 216], [140, 0, 214], [143, 0, 212], [146, 0, 210],
                       [148, 0, 206], [150, 0, 202], [152, 0, 199], [154, 0, 195], [156, 0, 191],
                       [158, 0, 188], [160, 0, 184], [162, 0, 181], [163, 0, 177], [164, 0, 173],
                       [166, 0, 169], [167, 0, 166], [168, 0, 162], [170, 0, 158], [171, 0, 154],
                       [173, 0, 151], [174, 0, 147], [175, 0, 143], [177, 0, 140], [178, 0, 136],
                       [179, 0, 132], [181, 0, 129], [182, 0, 125], [184, 0, 122], [185, 0, 118],
                       [186, 0, 114], [188, 0, 111], [189, 0, 107], [190, 0, 103], [192, 0, 100],
                       [193, 0, 96], [195, 0, 93], [196, 1, 89], [198, 3, 85], [199, 5, 82], [201, 7, 78],
                       [202, 8, 74], [204, 10, 71], [205, 12, 67], [207, 14, 64], [208, 16, 60],
                       [209, 19, 56], [210, 21, 53], [212, 24, 49], [213, 27, 45], [214, 29, 42],
                       [215, 32, 38], [217, 35, 35], [218, 37, 31], [220, 40, 27], [221, 43, 23],
                       [223, 46, 20], [224, 48, 16], [226, 51, 12], [227, 54, 8], [229, 57, 5],
                       [230, 59, 4], [231, 62, 3], [233, 65, 3], [234, 68, 2], [235, 70, 1], [237, 73, 1],
                       [238, 76, 0], [240, 79, 0], [241, 81, 0], [243, 84, 0], [244, 87, 0], [246, 90, 0],
                       [247, 92, 0], [249, 95, 0], [250, 98, 0], [252, 101, 0], [252, 103, 0], [252, 105, 0],
                       [253, 107, 0], [253, 109, 0], [253, 111, 0], [254, 113, 0], [254, 115, 0], [255, 117, 0],
                       [255, 119, 0], [255, 121, 0], [255, 123, 0], [255, 125, 0], [255, 127, 0], [255, 129, 0],
                       [255, 131, 0], [255, 133, 0], [255, 134, 0], [255, 136, 0], [255, 138, 0], [255, 140, 0],
                       [255, 141, 0], [255, 143, 0], [255, 145, 0], [255, 147, 0], [255, 148, 0], [255, 150, 0],
                       [255, 152, 0], [255, 154, 0], [255, 155, 0], [255, 157, 0], [255, 159, 0], [255, 161, 0],
                       [255, 162, 0], [255, 164, 0], [255, 166, 0], [255, 168, 0], [255, 169, 0], [255, 171, 0],
                       [255, 173, 0], [255, 175, 0], [255, 176, 0], [255, 178, 0], [255, 180, 0], [255, 182, 0],
                       [255, 184, 0], [255, 186, 0], [255, 188, 0], [255, 190, 0], [255, 191, 0], [255, 193, 0],
                       [255, 195, 0], [255, 197, 0], [255, 199, 0], [255, 201, 0], [255, 203, 0], [255, 205, 0],
                       [255, 206, 0], [255, 208, 0], [255, 210, 0], [255, 212, 0], [255, 213, 0], [255, 215, 0],
                       [255, 217, 0], [255, 219, 0], [255, 220, 0], [255, 222, 0], [255, 224, 0], [255, 226, 0],
                       [255, 228, 0], [255, 230, 0], [255, 232, 0], [255, 234, 0], [255, 235, 4], [255, 237, 8],
                       [255, 239, 13], [255, 241, 17], [255, 242, 21], [255, 244, 26], [255, 246, 30],
                       [255, 248, 35], [255, 248, 42], [255, 249, 50], [255, 250, 58], [255, 251, 66],
                       [255, 252, 74], [255, 253, 82], [255, 254, 90], [255, 255, 98], [255, 255, 105],
                       [255, 255, 113], [255, 255, 121], [255, 255, 129], [255, 255, 136], [255, 255, 144],
                       [255, 255, 152], [255, 255, 160], [255, 255, 167], [255, 255, 175], [255, 255, 183],
                       [255, 255, 191], [255, 255, 199], [255, 255, 207], [255, 255, 215], [255, 255, 223],
                       [255, 255, 227], [255, 255, 231], [255, 255, 235], [255, 255, 239], [255, 255, 243],
                       [255, 255, 247], [255, 255, 251], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                       [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]
]
assert len(IMAGEJ_FIRE_LUT_256) == 256


def imagej_fire_colormap() -> mcolors.ListedColormap:
    arr = np.array(IMAGEJ_FIRE_LUT_256, dtype=float) / 255.0
    return mcolors.ListedColormap(arr, name="imagej_fire")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill labeled SVG regions with heatmap colors.")
    p.add_argument("--summary", type=Path, default=None,
                   help="summary_per_label.csv (required if using stats other than 'real').")
    p.add_argument("--real-data", type=Path, default=None,
                   help="Raw observation CSV (required if any stat is 'real').")
    p.add_argument("--labels", type=Path, required=True, help="root_svg_labels.json")
    p.add_argument("--svg", type=Path, required=True, help="Base SVG template")
    p.add_argument("--output-dir", type=Path, default=Path("heatmap_outputs"), help="Output directory")
    p.add_argument("--features", nargs="+", required=True, help="Feature columns to visualize")
    p.add_argument("--stats", nargs="+", required=True, help="Stats to visualize (e.g., mean std ... or real)")
    p.add_argument("--colormap", default="imagej_fire",
                   help="Matplotlib colormap name, or 'imagej_fire' to use ImageJ Fire LUT")

    # Fill mapping: restrict colormap sampling range (visual-only)
    p.add_argument("--cmap-range", nargs=2, type=float, default=[0.10, 0.90],
                   metavar=("LO", "HI"),
                   help="Restrict FILL colormap sampling to [LO, HI] in [0,1], purely visual. "
                        "Example: --cmap-range 0.1 0.9. (Colorbar still shows FULL 0..1)")

    # Real data filtering for sampling (avoid extreme outliers)
    p.add_argument("--real-filter", choices=["none", "zscore", "mad"], default="mad",
                   help="Filter extreme observations within each label before sampling.")
    p.add_argument("--real-z-thresh", type=float, default=1.5,
                   help="For real-filter=zscore: keep |x-mean| <= z*std")
    p.add_argument("--real-mad-thresh", type=float, default=1.5,
                   help="For real-filter=mad: keep |x-median| <= k*MAD (MAD=median(|x-median|)).")
    p.add_argument("--real-min-keep-frac", type=float, default=0.25,
                   help="If filtering leaves less than this fraction, fall back to unfiltered values.")

    p.add_argument("--missing-color", default="#d1d5db", help="Hex color for missing values")
    p.add_argument("--vmin", type=float, default=None, help="Override vmin (else data min)")
    p.add_argument("--vmax", type=float, default=None, help="Override vmax (else data max)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for stat='real' sampling")

    # Colorbar / combined SVG
    p.add_argument("--colorbar-orientation", choices=["vertical", "horizontal"], default="vertical")
    p.add_argument("--no-colorbar", action="store_true", help="Do not generate combined SVG with colorbar")

    p.add_argument("--colorbar-anchor", choices=["outside-right", "bottom-right"], default="bottom-right",
                   help="Where to place colorbar. bottom-right puts it inside the figure like the example.")
    p.add_argument("--colorbar-inset-x", type=float, default=68.0,
                   help="Inset from the right edge (px) when anchor=bottom-right")
    p.add_argument("--colorbar-inset-y", type=float, default=24.0,
                   help="Inset from the bottom edge (px) when anchor=bottom-right")

    p.add_argument("--colorbar-width", type=float, default=10.0, help="Base bar width (px) before scaling.")
    p.add_argument("--colorbar-frac-height", type=float, default=0.42,
                   help="Base bar height as fraction of figure height before scaling.")
    p.add_argument("--colorbar-margin", type=float, default=28.0, help="Used only for outside-right anchor.")
    p.add_argument("--tick-count", type=int, default=2)
    p.add_argument("--colorbar-frame", action="store_true",
                   help="Draw a black frame around the colorbar (like the example).")
    p.add_argument("--colorbar-scale", type=float, default=0.38,
                   help="Global scale factor for colorbar size (width/height/frame/ticks/font). "
                        "Example: 0.65 makes a smaller inset bar.")

    # Optional export to PNG
    p.add_argument("--export-png", action="store_true",
                   help="Export combined PNG with cairosvg (optional; requires working cairosvg/cairo).")
    return p.parse_args()


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def read_summary(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, header=[0, 1], index_col=0)
    except Exception:
        return pd.read_csv(path, index_col=0)


def normalize_label_name(name: str) -> str:
    return str(name).strip().lower()


def load_labels_json(path: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    label_id_to_name = {lab["id"]: lab["name"] for lab in payload["labels"]}
    label_name_to_id = {normalize_label_name(lab["name"]): lab["id"] for lab in payload["labels"]}
    assignments = payload["assignments"]  # svg_element_id -> label_id
    return label_id_to_name, label_name_to_id, assignments


def build_metric_series(summary_df: pd.DataFrame, feature: str, stat: str) -> pd.Series:
    if isinstance(summary_df.columns, pd.MultiIndex):
        key = (feature, stat)
        if key not in summary_df.columns:
            raise KeyError(f"Missing column {key} in summary.")
        return summary_df[key]

    col1 = f"{feature}_{stat}"
    if col1 in summary_df.columns:
        return summary_df[col1]
    if feature in summary_df.columns:
        return summary_df[feature]
    raise KeyError(f"Missing column for feature={feature}, stat={stat} in summary.")


def resolve_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "Label", "LABEL", "region", "Region", "REGION"]
    for c in candidates:
        if c in df.columns:
            return c
    cols_lower = {c.lower(): c for c in df.columns}
    for c in ["label", "region"]:
        if c in cols_lower:
            return cols_lower[c]
    raise KeyError("Real CSV missing label column. Expected label/region (case-insensitive).")


def resolve_feature_column(df: pd.DataFrame, feature: str) -> str:
    if feature in df.columns:
        return feature
    cols_lower = {c.lower(): c for c in df.columns}
    if feature.lower() in cols_lower:
        return cols_lower[feature.lower()]
    raise KeyError(f"Real CSV missing feature column: '{feature}' (case-insensitive).")


# ---------------------------------------------------------------------
# SVG fill update (PRESERVE stroke!)
# ---------------------------------------------------------------------
_FILL_RE = re.compile(r'((^|;)\s*fill\s*:\s*)[^;]*')


def update_style_preserve_stroke(style: Optional[str], fill_color: str) -> str:
    """Only touch fill inside style; preserve stroke/stroke-width/others."""
    if not style:
        return f"fill:{fill_color}"
    s = style.strip()
    if re.search(r'(^|;)\s*fill\s*:', s):
        return _FILL_RE.sub(r'\1' + fill_color, s)
    if s.endswith(";") or len(s) == 0:
        return s + f"fill:{fill_color}"
    return s + f";fill:{fill_color}"


def color_svg_by_element(svg_path: Path, element_fill: Dict[str, str], output_path: Path) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    for el in root.iter():
        el_id = el.attrib.get("id")
        if not el_id:
            continue
        fill = element_fill.get(el_id)
        if not fill:
            continue
        # ONLY modify style fill; do not set el.attrib["fill"]
        el.attrib["style"] = update_style_preserve_stroke(el.attrib.get("style"), fill)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------
# Colormap mapping
# ---------------------------------------------------------------------
def get_cmap(name: str) -> matplotlib.colors.Colormap:
    if name.lower() in {"imagej_fire", "fire", "ij_fire", "imagej-fire"}:
        return imagej_fire_colormap()
    return matplotlib.colormaps.get_cmap(name)


def apply_cmap_range(t: float, lo: float, hi: float) -> float:
    """Map t in [0,1] into [lo,hi] (visual-only compression), then clamp."""
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    if hi < lo:
        lo, hi = hi, lo
    t2 = lo + (hi - lo) * float(np.clip(t, 0.0, 1.0))
    return float(np.clip(t2, 0.0, 1.0))


def color_from_value_fill(val: float,
                          cmap: matplotlib.colors.Colormap,
                          norm: matplotlib.colors.Normalize,
                          cmap_lo: float,
                          cmap_hi: float) -> str:
    """Color for FILL: apply norm then compress into cmap-range."""
    t = float(norm(val))  # 0..1
    t2 = apply_cmap_range(t, cmap_lo, cmap_hi)
    rgba = cmap(t2)
    return mcolors.to_hex(rgba, keep_alpha=False)


def color_from_value_bar_full(t: float,
                              cmap: matplotlib.colors.Colormap) -> str:
    """Color for COLORBAR: always full 0..1 colormap (ignore cmap-range)."""
    rgba = cmap(float(np.clip(t, 0.0, 1.0)))
    return mcolors.to_hex(rgba, keep_alpha=False)


# ---------------------------------------------------------------------
# SVG combined + inset colorbar
# ---------------------------------------------------------------------
def _parse_svg_length(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    v = value.strip()
    for suf in ("px", "pt", "mm", "cm", "in"):
        if v.lower().endswith(suf):
            v = v[: -len(suf)].strip()
            break
    try:
        return float(v)
    except ValueError:
        return None


def create_combined_svg(
    heatmap_svg: Path,
    cmap: matplotlib.colors.Colormap,
    norm: matplotlib.colors.Normalize,
    vmin: float,
    vmax: float,
    orientation: str,
    output_svg: Path,
    bar_width: float,
    bar_frac_height: float,
    margin: float,
    tick_count: int,
    anchor: str,
    inset_x: float,
    inset_y: float,
    frame: bool,
    scale: float,
) -> None:
    tree = ET.parse(heatmap_svg)
    root = tree.getroot()

    # namespace
    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}", 1)[0].strip("{")
    else:
        ns_uri = "http://www.w3.org/2000/svg"
    ns = {"svg": ns_uri}
    ET.register_namespace("", ns_uri)

    width = _parse_svg_length(root.attrib.get("width"))
    height = _parse_svg_length(root.attrib.get("height"))
    view_box = root.attrib.get("viewBox")

    if (width is None or height is None) and view_box:
        parts = [p for p in view_box.replace(",", " ").split() if p]
        if len(parts) == 4:
            try:
                width = float(parts[2])
                height = float(parts[3])
            except ValueError:
                pass
    if width is None or height is None:
        width, height = 1000.0, 1000.0

    # defs
    defs = root.find("svg:defs", ns)
    if defs is None:
        defs = ET.Element(f"{{{ns_uri}}}defs")
        root.insert(0, defs)

    grad_id = "heatmapColorbarGradientFull"
    for g in list(defs.findall("svg:linearGradient", ns)):
        if g.attrib.get("id") == grad_id:
            defs.remove(g)

    linear = ET.SubElement(defs, f"{{{ns_uri}}}linearGradient", {"id": grad_id})
    if orientation == "vertical":
        linear.attrib.update({"x1": "0", "y1": "1", "x2": "0", "y2": "0"})
    else:
        linear.attrib.update({"x1": "0", "y1": "0", "x2": "1", "y2": "0"})

    # ✅ colorbar always shows full 0..1 colormap
    n_stops = 256
    for i in range(n_stops):
        t = i / (n_stops - 1)  # 0..1
        col = color_from_value_bar_full(t, cmap)
        ET.SubElement(linear, f"{{{ns_uri}}}stop",
                      {"offset": f"{t*100:.4f}%", "stop-color": col})

    # remove old colorbar group if any
    cb_group_id = "heatmap_colorbar"
    for el in list(root.findall(f".//svg:g[@id='{cb_group_id}']", ns)):
        parent = root
        for p in root.iter():
            if el in list(p):
                parent = p
                break
        parent.remove(el)

    cb_group = ET.SubElement(root, f"{{{ns_uri}}}g", {"id": cb_group_id})

    # scale all size-related parameters together
    scale = float(scale)
    bar_w = float(bar_width) * scale
    bar_h = float(height) * float(bar_frac_height) * scale

    tick_len = 7.0 * scale
    tick_gap = 6.0 * scale
    font_size = 11.0 * scale
    frame_width = 0.8 * scale
    label_space = 65.0 * scale  # used only for outside-right

    # position
    if anchor == "bottom-right":
        # inset inside the existing canvas (do NOT change viewBox/size)
        bar_x = float(width) - float(inset_x) - bar_w
        bar_y = float(height) - float(inset_y) - bar_h
    else:
        # outside-right: keep old behavior (expand canvas)
        bar_x = float(width) + float(margin)
        bar_y = (float(height) - bar_h) / 2.0

        new_width = float(width) + float(margin) + bar_w + tick_gap + tick_len + label_space + float(margin)
        if view_box:
            parts = [p for p in view_box.replace(",", " ").split() if p]
            if len(parts) == 4:
                try:
                    x0, y0 = float(parts[0]), float(parts[1])
                    root.attrib["viewBox"] = f"{x0} {y0} {new_width} {height}"
                except ValueError:
                    pass
        root.attrib["width"] = f"{new_width}px"
        root.attrib["height"] = f"{height}px"

    # bar rect with optional black frame
    rect_attrib = {
        "x": f"{bar_x}",
        "y": f"{bar_y}",
        "width": f"{bar_w}",
        "height": f"{bar_h}",
        "fill": f"url(#{grad_id})",
    }
    if frame:
        rect_attrib.update({"stroke": "#000000", "stroke-width": f"{frame_width}"})
    ET.SubElement(cb_group, f"{{{ns_uri}}}rect", rect_attrib)

    # ticks (data-space) positioned by norm(val)
    if tick_count < 2:
        tick_count = 2
    ticks = np.linspace(vmin, vmax, tick_count)

    def fmt_2sig(x: float) -> str:
        return f"{x:.2g}"  # 2 significant digits

    # only show min/max
    center_x = bar_x + bar_w / 2.0

    if orientation == "vertical":
        # text positions: above top and below bottom
        top_y = bar_y - (4.0 * scale)  # small gap above bar
        bot_y = bar_y + bar_h + (font_size + 4.0 * scale)  # below bar

        # Top label = vmax
        ET.SubElement(cb_group, f"{{{ns_uri}}}text",
                      {"x": f"{center_x}", "y": f"{top_y}",
                       "font-size": f"{font_size}",
                       "font-family": "Arial, Helvetica, sans-serif",
                       "fill": "#000",
                       "text-anchor": "middle"}).text = fmt_2sig(vmax)

        # Bottom label = vmin
        ET.SubElement(cb_group, f"{{{ns_uri}}}text",
                      {"x": f"{center_x}", "y": f"{bot_y}",
                       "font-size": f"{font_size}",
                       "font-family": "Arial, Helvetica, sans-serif",
                       "fill": "#000",
                       "text-anchor": "middle"}).text = fmt_2sig(vmin)

    else:
        # horizontal (if you ever use it): left=min, right=max
        left_x = bar_x
        right_x = bar_x + bar_w
        text_y = bar_y - (4.0 * scale)

        ET.SubElement(cb_group, f"{{{ns_uri}}}text",
                      {"x": f"{left_x}", "y": f"{text_y}",
                       "font-size": f"{font_size}",
                       "font-family": "Arial, Helvetica, sans-serif",
                       "fill": "#000",
                       "text-anchor": "start"}).text = fmt_2sig(vmin)

        ET.SubElement(cb_group, f"{{{ns_uri}}}text",
                      {"x": f"{right_x}", "y": f"{text_y}",
                       "font-size": f"{font_size}",
                       "font-family": "Arial, Helvetica, sans-serif",
                       "fill": "#000",
                       "text-anchor": "end"}).text = fmt_2sig(vmax)


    output_svg.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_svg, encoding="utf-8", xml_declaration=True)


def export_png(svg_path: Path, png_path: Path) -> bool:
    try:
        import cairosvg  # type: ignore
    except Exception:
        return False
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
    return True


# ---------------------------------------------------------------------
# Real data loading + filtering
# ---------------------------------------------------------------------
@dataclass
class RealData:
    by_label_id: Dict[str, np.ndarray]  # label_id -> filtered values


def filter_values(vals: np.ndarray, method: str, z_thresh: float, mad_thresh: float, min_keep_frac: float) -> np.ndarray:
    vals = vals.astype(float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or method == "none":
        return vals

    original_n = vals.size

    if method == "zscore":
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=0))
        if sd <= 0:
            kept = vals
        else:
            kept = vals[np.abs(vals - mu) <= z_thresh * sd]

    elif method == "mad":
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        if mad <= 0:
            kept = vals
        else:
            kept = vals[np.abs(vals - med) <= mad_thresh * mad]

    else:
        kept = vals

    # fallback if too aggressive
    if kept.size < max(1, int(np.ceil(min_keep_frac * original_n))):
        return vals
    return kept


def load_real_data(path: Path, label_name_to_id: Dict[str, str], feature: str,
                   real_filter: str, z_thresh: float, mad_thresh: float, min_keep_frac: float) -> RealData:
    df = pd.read_csv(path)
    lab_col = resolve_label_column(df)
    feat_col = resolve_feature_column(df, feature)

    label_ids: List[Optional[str]] = []
    for raw in df[lab_col].astype(str).tolist():
        lid = label_name_to_id.get(normalize_label_name(raw))
        label_ids.append(lid)

    df = df.copy()
    df["_label_id"] = label_ids
    df = df[df["_label_id"].notna()]

    vals = pd.to_numeric(df[feat_col], errors="coerce")
    df["_val"] = vals
    df = df[df["_val"].notna()]

    by_label_raw: Dict[str, List[float]] = {}
    for lid, v in zip(df["_label_id"].astype(str).tolist(), df["_val"].astype(float).tolist()):
        by_label_raw.setdefault(lid, []).append(float(v))

    by_label_filtered: Dict[str, np.ndarray] = {}
    for lid, vs in by_label_raw.items():
        arr = np.array(vs, dtype=float)
        arr_f = filter_values(arr, real_filter, z_thresh, mad_thresh, min_keep_frac)
        by_label_filtered[lid] = arr_f

    return RealData(by_label_id=by_label_filtered)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # cmap-range sanity
    cmap_lo, cmap_hi = args.cmap_range
    cmap_lo = float(np.clip(cmap_lo, 0.0, 1.0))
    cmap_hi = float(np.clip(cmap_hi, 0.0, 1.0))
    if cmap_hi < cmap_lo:
        cmap_lo, cmap_hi = cmap_hi, cmap_lo

    _, label_name_to_id, assignments = load_labels_json(args.labels)
    cmap = get_cmap(args.colormap)

    summary_df: Optional[pd.DataFrame] = None
    if any(s.lower() != "real" for s in args.stats):
        if args.summary is None:
            raise SystemExit("--summary is required when using stats other than 'real'.")
        summary_df = read_summary(args.summary)

    # reproducible RNG for real sampling
    if args.seed is None:
        seed_str = "|".join(args.features) + "||" + "|".join(args.stats)
        args.seed = (abs(hash(seed_str)) % (2**31 - 1))
    rng = np.random.default_rng(args.seed)

    manifest_rows: List[List[str]] = []

    for feature in args.features:
        real_data: Optional[RealData] = None
        if any(s.lower() == "real" for s in args.stats):
            if args.real_data is None:
                raise SystemExit("--real-data is required when using stat='real'.")
            real_data = load_real_data(
                args.real_data,
                label_name_to_id,
                feature,
                real_filter=args.real_filter,
                z_thresh=args.real_z_thresh,
                mad_thresh=args.real_mad_thresh,
                min_keep_frac=args.real_min_keep_frac,
            )

        for stat in args.stats:
            stat_l = stat.lower()

            element_values: Dict[str, float] = {}  # svg_element_id -> value
            used_values: List[float] = []

            if stat_l == "real":
                assert real_data is not None
                for el_id, label_id in assignments.items():
                    vals = real_data.by_label_id.get(label_id)
                    if vals is None or vals.size == 0:
                        continue
                    v = float(rng.choice(vals))
                    element_values[el_id] = v
                    used_values.append(v)

            else:
                assert summary_df is not None
                series = build_metric_series(summary_df, feature, stat)

                label_value_by_id: Dict[str, float] = {}
                for label_name, value in series.items():
                    lid = label_name_to_id.get(normalize_label_name(label_name))
                    if lid is None or pd.isna(value):
                        continue
                    label_value_by_id[lid] = float(value)

                for el_id, lid in assignments.items():
                    v = label_value_by_id.get(lid)
                    if v is None:
                        continue
                    element_values[el_id] = v
                    used_values.append(float(v))

            if len(used_values) == 0:
                raise RuntimeError(f"No values available for feature={feature}, stat={stat} after label matching.")

            vmin = float(np.min(used_values)) if args.vmin is None else float(args.vmin)
            vmax = float(np.max(used_values)) if args.vmax is None else float(args.vmax)
            if vmax == vmin:
                vmax = vmin + 1e-9

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

            # element_id -> fill color (apply cmap-range compression ONLY for fill)
            element_fill: Dict[str, str] = {
                el_id: color_from_value_fill(v, cmap, norm, cmap_lo, cmap_hi) for el_id, v in element_values.items()
            }

            svg_out = args.output_dir / f"heatmap_{feature}_{stat}.svg"
            color_svg_by_element(args.svg, element_fill, svg_out)

            combined_svg_name = ""
            combined_png_name = ""

            if not args.no_colorbar:
                combined_svg = args.output_dir / f"heatmap_{feature}_{stat}_combined.svg"
                create_combined_svg(
                    heatmap_svg=svg_out,
                    cmap=cmap,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    orientation=args.colorbar_orientation,
                    output_svg=combined_svg,
                    bar_width=args.colorbar_width,
                    bar_frac_height=args.colorbar_frac_height,
                    margin=args.colorbar_margin,
                    tick_count=args.tick_count,
                    anchor=args.colorbar_anchor,
                    inset_x=args.colorbar_inset_x,
                    inset_y=args.colorbar_inset_y,
                    frame=args.colorbar_frame,
                    scale=args.colorbar_scale,
                )
                combined_svg_name = combined_svg.name

                if args.export_png:
                    png_path = args.output_dir / f"heatmap_{feature}_{stat}_combined.png"
                    if not export_png(combined_svg, png_path):
                        raise RuntimeError("PNG export requested but cairosvg is not available/configured.")
                    combined_png_name = png_path.name

            manifest_rows.append([
                feature, stat, svg_out.name, combined_svg_name, combined_png_name,
                str(args.seed), f"{vmin}", f"{vmax}",
                f"{cmap_lo}", f"{cmap_hi}",
                args.real_filter, f"{args.real_z_thresh}", f"{args.real_mad_thresh}", f"{args.real_min_keep_frac}",
                args.colorbar_anchor, f"{args.colorbar_inset_x}", f"{args.colorbar_inset_y}",
                f"{args.colorbar_width}", f"{args.colorbar_frac_height}", f"{args.colorbar_scale}",
                "1" if args.colorbar_frame else "0",
            ])

    report_path = args.output_dir / "heatmap_manifest.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "feature", "stat", "svg", "combined_svg", "combined_png",
            "seed", "vmin", "vmax",
            "cmap_lo", "cmap_hi",
            "real_filter", "real_z_thresh", "real_mad_thresh", "real_min_keep_frac",
            "colorbar_anchor", "colorbar_inset_x", "colorbar_inset_y",
            "colorbar_width", "colorbar_frac_height", "colorbar_scale", "colorbar_frame",
        ])
        w.writerows(manifest_rows)

    print(f"Done. Outputs in: {args.output_dir}")
    print(f"Manifest: {report_path}")


if __name__ == "__main__":
    main()
