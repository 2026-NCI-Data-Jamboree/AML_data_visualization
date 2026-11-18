"""Plotly figures for the Surfaceome page (kept separate from plots.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from portal.plots import COL_A, COL_B, COL_NS, SEQ

HL = "#111111"
NA_FILL = "rgba(140,140,140,0.10)"


def _alpha(color: str, a: float) -> str:
    if color.startswith("#") and len(color) == 7:
        r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r},{g},{b},{a})"
    return color


def group_box(df: pd.DataFrame, order: list[str], title: str, y_title: str = "", subtitle: str = "",
              colors: dict | None = None) -> go.Figure:
    """Box + points per group (df: cell_line, value, group); horizontal labels with n."""
    colors = colors or {}
    fig, ticks = go.Figure(), []
    for i, g in enumerate(order):
        s = df[df["group"] == g].dropna(subset=["value"])
        if s.empty:
            continue
        col = colors.get(g, SEQ[i % len(SEQ)])
        tick = f"{_wrap(g)}<br><span style='font-size:10px'>n={len(s)}</span>"
        ticks.append(tick)
        fig.add_trace(go.Box(x=[tick] * len(s), y=s["value"], boxpoints="all", jitter=0.45, pointpos=0, width=0.55,
                             marker=dict(color=col, size=6, opacity=0.85, line=dict(width=0.8, color="white")),
                             line=dict(color=col, width=1.5), fillcolor=_alpha(col, 0.12), text=s["cell_line"],
                             hovertemplate="%{text}: %{y:.2f}<extra></extra>"))
    head = f"<b>{title}</b>" + (f"<br><span style='font-size:11px;color:#8a8984'>{subtitle}</span>" if subtitle else "")
    fig.update_layout(title=dict(text=head, x=0, xanchor="left", y=0.97, font=dict(size=15)), yaxis_title=y_title,
                      showlegend=False, height=340, margin=dict(l=10, r=10, t=58 if subtitle else 40, b=10))
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=ticks, tickangle=0)
    return fig


def volcano(res: pd.DataFrame, highlight: set[str], name_a: str, name_b: str, sig_by: str = "q",
            fdr: float = 0.1, alpha: float = 0.01) -> go.Figure:
    """res: feature, diff_A_minus_B, p_welch, fdr_welch. Red = higher in A, blue = lower (at the threshold)."""
    r = res.assign(y=-np.log10(res["p_welch"].clip(lower=1e-300)))
    sig = (r["fdr_welch"] < fdr) if sig_by == "q" else (r["p_welch"] < alpha)
    col = np.where(sig & (r["diff_A_minus_B"] > 0), COL_B, np.where(sig, COL_A, COL_NS))
    fig = go.Figure(go.Scattergl(x=r["diff_A_minus_B"], y=r["y"], mode="markers", customdata=r["feature"],
                                 marker=dict(color=col, size=np.where(sig, 7, 4), line=dict(width=0)),
                                 hovertemplate="<b>%{customdata}</b><br>log2FC %{x:.2f}<br>−log10 p %{y:.2f}"
                                               "<extra></extra>"))
    h = r[r["feature"].isin(highlight)]
    if len(h):
        fig.add_trace(go.Scatter(x=h["diff_A_minus_B"], y=h["y"], mode="markers+text", text=h["feature"],
                                 textposition="top center", textfont=dict(size=11, color=HL), hoverinfo="skip",
                                 marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(width=2, color=HL))))
    if sig_by == "p":
        fig.add_hline(y=-np.log10(alpha), line_dash="dot", line_width=1, line_color="#52514e")
    elif sig.any():
        fig.add_hline(y=r.loc[sig, "y"].min(), line_dash="dot", line_width=1, line_color="#52514e")
    fig.add_vline(x=0, line_width=1, line_color="#52514e")
    fig.update_layout(xaxis_title=f"log2 fold change ({name_a} − {name_b})", yaxis_title="−log10(p-value)",
                      height=380, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


