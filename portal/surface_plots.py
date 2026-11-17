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


def _wrap(label: str, width: int = 14) -> str:
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


def rank_bars(df: pd.DataFrame, y_title: str, order: list[str]) -> go.Figure:
    """Lines ranked by value (df: cell_line, value, group), one bar each, coloured by group."""
    d = df.sort_values("value", ascending=False)
    fig = go.Figure()
    for i, g in enumerate(order):
        s = d[d["group"] == g]
        if len(s):
            fig.add_trace(go.Bar(x=s["cell_line"], y=s["value"], name=f"{g} (n={len(s)})",
                                 marker_color=COL_NS if g == "not annotated" else SEQ[i % len(SEQ)],
                                 hovertemplate="%{x}: %{y:.2f}<extra>" + g + "</extra>"))
    fig.add_hline(y=d["value"].median(), line_dash="dot", line_width=1, line_color="#52514e",
                  annotation_text=f"median {d['value'].median():.2f}", annotation_position="top right")
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=list(d["cell_line"]), tickangle=-90)
    fig.update_layout(barmode="overlay", yaxis_title=y_title, height=400, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1))
    return fig


def panel_heatmap(Z: pd.DataFrame, groups: pd.Series) -> go.Figure:
    """Markers x lines z-scores (blank = not detected); group names above the columns."""
    fig = go.Figure(go.Heatmap(x=list(Z.columns), y=list(Z.index), z=Z.clip(-2.5, 2.5).to_numpy(), zmin=-2.5, zmax=2.5,
                               zmid=0, colorscale=[[0, COL_A], [0.5, NA_FILL], [1, COL_B]], xgap=1, ygap=1,
                               colorbar=dict(title=dict(text="z-score", side="right"), len=0.8), hoverongaps=False,
                               hovertemplate="%{y} · %{x}<br>z %{z:.2f}<extra></extra>"))
    g = groups.reindex(Z.columns).fillna("not annotated").astype(str)
    start = 0
    for i in range(1, len(g) + 1):
        if i == len(g) or g.iloc[i] != g.iloc[start]:
            fig.add_annotation(x=(start + i - 1) / 2, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                               text=g.iloc[start], font=dict(size=11))
            if i < len(g):
                fig.add_vline(x=i - 0.5, line_width=2, line_color="#52514e")
            start = i
    fig.update_yaxes(autorange="reversed", type="category", automargin=True)
    fig.update_xaxes(type="category", tickangle=-90, automargin=True)
    fig.update_layout(height=180 + 22 * len(Z), margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def frequency_bars(S: pd.DataFrame) -> go.Figure:
    """% of lines positive per marker (S index = marker), coloured by SD (heterogeneity)."""
    d = S.sort_values("pct_positive", ascending=False)
    fig = go.Figure(go.Bar(x=list(d.index), y=d["pct_positive"],
                           marker=dict(color=d["sd"], colorscale=[[0, "#9fc2ee"], [1, COL_B]],
                                       colorbar=dict(title=dict(text="SD", side="right"), len=0.7)),
                           customdata=np.c_[d["n_positive"], d["n_measured"], d["sd"]],
                           hovertemplate="%{x}: %{y:.0f}% positive (%{customdata[0]} lines)<br>measured in "
                                         "%{customdata[1]} · SD %{customdata[2]:.2f}<extra></extra>"))
    fig.update_yaxes(title="% of lines positive", range=[0, 105])
    fig.update_xaxes(type="category", tickangle=-45, automargin=True)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def freq_vs_het(S: pd.DataFrame, highlight: set[str]) -> go.Figure:
    lab = [m if m in highlight else "" for m in S.index]
    fig = go.Figure(go.Scatter(x=S["pct_positive"], y=S["sd"], mode="markers+text", text=lab, textposition="top center",
                               textfont=dict(size=10, color=HL), customdata=S.index,
                               marker=dict(size=9, color=np.where(S.index.isin(list(highlight)), COL_B, COL_NS),
                                           line=dict(width=1, color="white")),
                               hovertemplate="<b>%{customdata}</b><br>%{x:.0f}% positive · SD %{y:.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="% of lines positive (frequency)", yaxis_title="SD of log2 level (heterogeneity)",
                      height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[-3, 103]))
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


def upset(combos: pd.DataFrame, members: list[str], sizes: pd.Series) -> go.Figure:
    """UpSet-style bars of observed antigen combinations (combos: pattern, count, name)."""
    from plotly.subplots import make_subplots
    x = [f"c{i}" for i in range(len(combos))]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.4])
    n_on = combos["pattern"].map(sum)
    fig.add_trace(go.Bar(x=x, y=combos["count"], text=combos["count"], textposition="outside", hovertext=combos["name"],
                         marker_color=[COL_NS if k == 0 else COL_A if k == 1 else COL_B for k in n_on],
                         hovertemplate="%{hovertext}<br>%{y} lines<extra></extra>"), row=1, col=1)
    ylab = [f"{m} ({int(sizes[m])})" for m in members]
    for xi, pat in zip(x, combos["pattern"]):
        on = [ylab[i] for i, v in enumerate(pat) if v]
        fig.add_trace(go.Scatter(x=[xi] * len(ylab), y=ylab, mode="markers", hoverinfo="skip",
                                 marker=dict(size=11, color="rgba(140,140,140,0.25)")), row=2, col=1)
        if on:
            fig.add_trace(go.Scatter(x=[xi] * len(on), y=on, mode="markers+lines", hoverinfo="skip",
                                     marker=dict(size=11, color="#52514e"), line=dict(color="#52514e", width=2)),
                          row=2, col=1)
    fig.update_xaxes(showticklabels=False, type="category")
    fig.update_yaxes(title_text="cell lines", row=1, col=1)
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=ylab[::-1], automargin=True, row=2, col=1)
    fig.update_layout(height=300 + 26 * len(members), showlegend=False, margin=dict(l=10, r=10, t=20, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
