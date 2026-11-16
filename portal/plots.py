"""Plotly figures. Palette follows plot_utils.R from the original AML_cell_lines analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COL_A, COL_B = "#2a78d6", "#e34948"       # group A (blue) / group B (red)
TINT_A, TINT_B = "#9fc2ee", "#f2a8a6"
COL_NS = "#c9c8c3"
SEQ = ["#2a78d6", "#e34948", "#1baf7a", "#eda100", "#8a5cd0", "#52514e", "#d65aa0", "#3fb3c4"]


def volcano(res: pd.DataFrame, labels: pd.Series, name_a: str, name_b: str,
            lower_is_stronger: bool, fdr: float = 0.05, n_label: int = 10) -> go.Figure:
    """res: welch() output. Colour = the group a feature favours; for dependency/AUC a
    lower value is the stronger effect, so the negative side belongs to group A."""
    r = res[np.isfinite(res["p_welch"])].copy()
    r["y"] = -np.log10(r["p_welch"])
    r["label"] = r["feature"].map(labels).fillna(r["feature"])
    favours_a = (r["diff_A_minus_B"] < 0) if lower_is_stronger else (r["diff_A_minus_B"] > 0)
    r["cls"] = np.where(r["fdr_welch"] < fdr, np.where(favours_a, "A", "B"),
                np.where(r["fdr_welch"] < 0.25, np.where(favours_a, "a", "b"), "ns"))
    style = {"A": (COL_A, f"{name_a}, FDR < {fdr}"), "a": (TINT_A, f"{name_a}, FDR < 0.25"),
             "B": (COL_B, f"{name_b}, FDR < {fdr}"), "b": (TINT_B, f"{name_b}, FDR < 0.25"),
             "ns": (COL_NS, "n.s.")}
    fig = go.Figure()
    for cls in ["ns", "b", "a", "B", "A"]:
        s = r[r["cls"] == cls]
        if s.empty:
            continue
        col, nm = style[cls]
        fig.add_trace(go.Scattergl(
            x=s["diff_A_minus_B"], y=s["y"], mode="markers", name=nm,
            marker=dict(color=col, size=6 if cls != "ns" else 4, line=dict(width=0)),
            customdata=np.c_[s["label"], s["fdr_welch"], s["feature"]],
            hovertemplate="<b>%{customdata[0]}</b><br>diff %{x:.3f}<br>p %{y:.2f} (−log10)"
                          "<br>FDR %{customdata[1]:.3g}<br>%{customdata[2]}<extra></extra>"))
    top = r.nsmallest(n_label, "p_welch")
    for _, t in top.iterrows():
        fig.add_annotation(x=t["diff_A_minus_B"], y=t["y"], text=t["label"], showarrow=False,
                           yshift=9, font=dict(size=10))
    fig.add_vline(x=0, line_width=1, line_color="#52514e")
    fig.update_layout(xaxis_title=f"mean {name_a} − mean {name_b}", yaxis_title="−log10 p (Welch)",
                      height=520, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def strip(df: pd.DataFrame, value_col: str, group_col: str, order: list[str],
          title: str = "", y_title: str = "", colors: dict | None = None) -> go.Figure:
    """Box + jittered points per group; hover shows the cell line."""
    colors = colors or {g: SEQ[i % len(SEQ)] for i, g in enumerate(order)}
    fig = go.Figure()
    for g in order:
        s = df[df[group_col] == g].dropna(subset=[value_col])
        if s.empty:
            continue
        fig.add_trace(go.Box(
            y=s[value_col], name=f"{g} (n={len(s)})", boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(color=colors.get(g, COL_NS), size=7), line=dict(color=colors.get(g, COL_NS)),
            fillcolor="rgba(0,0,0,0)", text=s["cell_line"],
            hovertemplate="%{text}: %{y:.3f}<extra></extra>"))
    fig.update_layout(title=title, yaxis_title=y_title, showlegend=False, height=380,
                      margin=dict(l=10, r=10, t=40 if title else 10, b=10))
    return fig


NA_COLOR = "rgba(140,140,140,0.22)"   # reads as light gray on light and dark themes
TRUE_COLOR, FALSE_COLOR = "#2a78d6", "#cfe0f5"
AVAIL_COLOR = "#1baf7a"
# categorical tracks avoid the blues (binary TRUE/FALSE) and the green (data available)
CAT = ["#eda100", "#8a5cd0", "#e34948", "#3fb3c4", "#d65aa0", "#8c6d31", "#52514e", "#b5b23a"]


def track_colors(levels: list[str], binary: tuple[str, str] | None = None) -> dict[str, str]:
    """Binary tracks: dark = TRUE label, light = FALSE label. Categorical: qualitative palette."""
    if binary:
        return {binary[0]: TRUE_COLOR, binary[1]: FALSE_COLOR}
    return {lv: CAT[i % len(CAT)] for i, lv in enumerate(levels)}


def _discrete_scale(colors: list[str]) -> list:
    """Plotly colorscale mapping integer codes 0..k-1 to the given colors as flat bands."""
    k = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale += [[i / k, c], [(i + 1) / k, c]]
    return scale


def availability_heatmap(lines: list[str], tracks: list[dict], avail: pd.DataFrame) -> go.Figure:
    """Oncoprint-style figure: one column per cell line, drawn as a single heatmap.

    tracks: [{"label": str, "values": Series indexed by line (NA allowed), "colors": {level: color}}]
            annotation rows at the top.
    avail:  lines x layers boolean (columns = layer labels), rows below the tracks.
    Every (row, level) gets its own integer code so one trace can carry all the colours.
    """
    colors = [NA_COLOR, AVAIL_COLOR]                           # code 0 = NA / no data, 1 = available
    z_rows, text_rows, labels = [], [], []
    for t in tracks:
        v = t["values"].reindex(lines)
        codes = {}
        for lv, col in t["colors"].items():
            codes[lv] = len(colors)
            colors.append(col)
        z_rows.append(v.map(codes).fillna(0).to_numpy())
        text_rows.append(v.fillna("not annotated").astype(str).to_numpy())
        labels.append(t["label"])
    a = avail.reindex(lines).fillna(False).astype(bool)
    for c in a.columns:
        z_rows.append(a[c].astype(int).to_numpy())
        text_rows.append(np.where(a[c], "available", "no data"))
        labels.append(c)
    fig = go.Figure(go.Heatmap(
        x=lines, y=labels, z=np.vstack(z_rows), text=np.vstack(text_rows),
        zmin=-0.5, zmax=len(colors) - 0.5, colorscale=_discrete_scale(colors), showscale=False,
        xgap=1, ygap=2, hovertemplate="%{x}<br>%{y}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11), automargin=True)
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), type="category", automargin=True)
    if tracks:                                                 # separate tracks from layers
        fig.add_hline(y=len(tracks) - 0.5, line_width=4, line_color="rgba(128,128,128,0.6)")
    fig.update_layout(height=150 + 20 * len(labels), margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def platform_scatter(df: pd.DataFrame, x_title: str, y_title: str, style: dict[str, tuple[str, str]],
                     n_label: int = 12, zero_lines: tuple = (0, 0), thresholds: tuple | None = None) -> go.Figure:
    """Gene-level CRISPR vs RNAi scatter. df: gene, x, y, cls, score (lowest = labelled first).
    style: cls -> (colour, legend name), drawn in the given order (last on top).
    customdata = gene symbol, so a Streamlit selection returns the genes picked."""
    fig = go.Figure()
    for cls, (col, name) in style.items():
        s = df[df["cls"] == cls]
        if s.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=s["x"], y=s["y"], mode="markers", name=f"{name} ({len(s):,})",
            marker=dict(color=col, size=4 if cls == "ns" else 6, line=dict(width=0)),
            customdata=s["gene"], selected=dict(marker=dict(size=10)),
            hovertemplate="<b>%{customdata}</b><br>CRISPR %{x:.3f}<br>RNAi %{y:.3f}<extra></extra>"))
    for _, t in df.nsmallest(n_label, "score").iterrows():
        fig.add_annotation(x=t["x"], y=t["y"], text=t["gene"], showarrow=False, yshift=9, font=dict(size=10))
    fig.add_vline(x=zero_lines[0], line_width=1, line_color="#52514e")
    fig.add_hline(y=zero_lines[1], line_width=1, line_color="#52514e")
    if thresholds:
        fig.add_vline(x=thresholds[0], line_width=1, line_dash="dot", line_color="#52514e")
        fig.add_hline(y=thresholds[1], line_width=1, line_dash="dot", line_color="#52514e")
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, height=560, dragmode="select",
                      clickmode="event+select",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.15))
    return fig


def line_scatter(df: pd.DataFrame, group_col: str, order: list[str], x_title: str, y_title: str,
                 title: str = "") -> go.Figure:
    """One point per cell line (x = CRISPR, y = RNAi), coloured by group, with a least-squares line."""
    fig = go.Figure()
    for i, g in enumerate(order):
        s = df[df[group_col] == g]
        if s.empty:
            continue
        col = COL_NS if g in ("not annotated", "not compared") else SEQ[i % len(SEQ)]
        fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="markers+text", name=f"{g} (n={len(s)})",
                                 text=s["cell_line"], textposition="top center", textfont=dict(size=9),
                                 marker=dict(color=col, size=9),
                                 hovertemplate="%{text}<br>CRISPR %{x:.3f}<br>RNAi %{y:.3f}<extra></extra>"))
    ok = df[["x", "y"]].dropna()
    if len(ok) >= 3 and ok["x"].std() > 0:
        b, a = np.polyfit(ok["x"], ok["y"], 1)
        xs = np.array([ok["x"].min(), ok["x"].max()])
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color="#52514e", width=1, dash="dash")))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, height=440,
                      margin=dict(l=10, r=10, t=40 if title else 10, b=10), legend=dict(orientation="h", y=-0.2))
    return fig
