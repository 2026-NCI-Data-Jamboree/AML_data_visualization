"""Plotly figures for the Surfaceome page (kept separate from plots.py)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from portal.plots import SEQ


def _alpha(color: str, a: float) -> str:
    if color.startswith("#") and len(color) == 7:
        r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r},{g},{b},{a})"
    return color


def _wrap(label: str, width: int = 14) -> str:
    """Break a long group label once (at a space, dash or slash) so box-plot tick labels stay readable."""
    if len(label) <= width:
        return label
    cut = max((i for i, ch in enumerate(label) if ch in " -/" and i <= len(label) * 0.6), default=-1)
    return label if cut <= 0 else label[:cut + 1].rstrip() + "<br>" + label[cut + 1:].lstrip()


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


