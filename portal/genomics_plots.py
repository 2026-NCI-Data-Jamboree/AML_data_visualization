"""Plotly figures for the genomic cell-line tabs (kept separate from plots.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from portal.plots import COL_A, COL_B, COL_NS, SEQ

HL = "#111111"


def pair_matrix(res: pd.DataFrame, order: list[str], labels: dict[str, str] | None = None,
                fdr: float = 0.1, clip: float = 3.0) -> go.Figure:
    """Lower-triangle matrix of log2 odds ratios: blue = mutual exclusivity, red = co-occurrence.
    ** q < fdr, * p < 0.05; untestable pairs are blank."""
    labels = labels or {}
    idx = {a: i for i, a in enumerate(order)}
    n = len(order)
    z = np.full((n, n), np.nan)
    text = np.full((n, n), "", dtype=object)
    hover = np.full((n, n), "", dtype=object)
    for _, r in res.iterrows():
        i, j = sorted((idx[r["A"]], idx[r["B"]]), reverse=True)    # row > column: lower triangle
        z[i, j] = np.clip(r["log2_odds_ratio"], -clip, clip)
        text[i, j] = "**" if r["q"] < fdr else ("*" if r["p"] < 0.05 else "")
        hover[i, j] = (f"{labels.get(r['A'], r['A'])} × {labels.get(r['B'], r['B'])}<br>{r['tendency']}"
                       f"<br>log2 OR {r['log2_odds_ratio']:.2f}<br>p {r['p']:.3g} · q {r['q']:.3g}"
                       f"<br>both {r['both']} · A only {r['A_not_B']} · B only {r['B_not_A']}"
                       f" · neither {r['neither']}")
    names = [labels.get(a, a) for a in order]
    fig = go.Figure(go.Heatmap(
        x=names, y=names, z=z, text=text, customdata=hover, texttemplate="%{text}",
        textfont=dict(size=14), zmin=-clip, zmax=clip, zmid=0, xgap=2, ygap=2,
        colorscale=[[0, COL_A], [0.5, "rgba(140,140,140,0.12)"], [1, COL_B]],
        colorbar=dict(title=dict(text="log2 OR", side="right"), tickvals=[-clip, 0, clip],
                      ticktext=[f"≤ −{clip:g} exclusive", "0", f"≥ {clip:g} co-occur"], len=0.8),
        hovertemplate="%{customdata}<extra></extra>", hoverongaps=False))
    fig.update_yaxes(autorange="reversed", automargin=True, type="category", showgrid=False)
    fig.update_xaxes(tickangle=-45, automargin=True, type="category", showgrid=False)
    side = 160 + 34 * n
    fig.update_layout(height=min(max(side, 320), 900), margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def upset(combos: pd.DataFrame, members: list[str], labels: dict[str, str], sizes: pd.Series) -> go.Figure:
    """UpSet-style plot. combos: one row per observed combination with 'count', 'pattern' (tuple of
    bools aligned with members) and 'id'. Bars carry the id as customdata so a click selects it."""
    from plotly.subplots import make_subplots
    x = [f"c{i}" for i in range(len(combos))]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.4])
    n_on = combos["pattern"].map(sum)
    col = [COL_NS if k == 0 else (COL_A if k == 1 else COL_B) for k in n_on]
    fig.add_trace(go.Bar(x=x, y=combos["count"], marker_color=col, text=combos["count"], textposition="outside",
                         customdata=combos["id"], hovertext=combos["name"],
                         hovertemplate="%{hovertext}<br>%{y} lines<extra></extra>"), row=1, col=1)
    ylab = [f"{labels.get(m, m)} ({int(sizes[m])})" for m in members]
    for xi, pat in zip(x, combos["pattern"]):
        on = [ylab[i] for i, v in enumerate(pat) if v]
        fig.add_trace(go.Scatter(x=[xi] * len(ylab), y=ylab, mode="markers", hoverinfo="skip",
                                 marker=dict(size=11, color="rgba(140,140,140,0.25)")), row=2, col=1)
        if on:
            fig.add_trace(go.Scatter(x=[xi] * len(on), y=on, mode="markers+lines", hoverinfo="skip",
                                     marker=dict(size=11, color="#52514e"), line=dict(color="#52514e", width=2)),
                          row=2, col=1)
    fig.update_xaxes(showticklabels=False, type="category", row=1, col=1)
    fig.update_xaxes(showticklabels=False, type="category", row=2, col=1)
    fig.update_yaxes(title_text="cell lines", row=1, col=1)
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=ylab[::-1], automargin=True, row=2, col=1)
    fig.update_layout(height=300 + 26 * len(members), showlegend=False, clickmode="event+select",
                      margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def frequency_bars(res: pd.DataFrame, labels: dict[str, str], name_a: str, name_b: str,
                   fdr: float = 0.1) -> go.Figure:
    """Grouped bars: % of lines altered per alteration in group A (blue) and B (red), in res order.
    ** q < fdr, * p < 0.05 above each pair."""
    x = [labels.get(a, a) for a in res["alteration"]]
    fig = go.Figure()
    for g, name, col in (("A", name_a, COL_A), ("B", name_b, COL_B)):
        fig.add_trace(go.Bar(
            x=x, y=100 * res[f"freq_{g}"], name=name, marker_color=col,
            customdata=np.c_[res[f"altered_{g}"], res[f"n_{g}"]],
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.0f}}% (%{{customdata[0]}}/%{{customdata[1]}})"
                          "<extra></extra>"))
    top = 100 * res[["freq_A", "freq_B"]].max(axis=1)
    for xi, yi, p, q in zip(x, top, res["p"], res["q"]):
        mark = "**" if q < fdr else ("*" if p < 0.05 else "")
        if mark:
            fig.add_annotation(x=xi, y=yi, text=mark, showarrow=False, yshift=10, font=dict(size=14))
    fig.update_layout(barmode="group", bargap=0.25, yaxis_title="% of lines altered", height=440,
                      yaxis=dict(range=[0, min(100, top.max() * 1.15 + 5)] if len(top) else None),
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(tickangle=-45, automargin=True, type="category")
    return fig


def cnv_heatmap(L: pd.DataFrame, pct_amp: pd.Series, pct_del: pd.Series, clip: float = 1.5) -> go.Figure:
    """Top: % of lines amplified (red, up) / deleted (blue, down) per region.
    Bottom: lines x regions heatmap of log2 relative CN (red = gain, blue = loss)."""
    from plotly.subplots import make_subplots
    regions = list(L.columns)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.22, 0.78])
    fig.add_trace(go.Bar(x=regions, y=pct_amp, name="amplified", marker_color=COL_B,
                         hovertemplate="%{x}: %{y:.0f}% of lines amplified<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=regions, y=-pct_del, name="deleted", marker_color=COL_A, customdata=pct_del,
                         hovertemplate="%{x}: %{customdata:.0f}% of lines deleted<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(
        x=regions, y=list(L.index), z=L.clip(-clip, clip).to_numpy(), zmin=-clip, zmax=clip, zmid=0,
        colorscale=[[0, COL_A], [0.5, "rgba(140,140,140,0.10)"], [1, COL_B]], customdata=L.to_numpy(),
        colorbar=dict(title=dict(text="log2 rel. CN", side="right"), len=0.7, y=0.38),
        hovertemplate="%{y} · %{x}<br>log2 relative CN %{customdata:.2f}<extra></extra>",
        hoverongaps=False), row=2, col=1)
    fig.update_yaxes(title_text="% lines", row=1, col=1, zeroline=True, zerolinecolor="#52514e")
    fig.update_yaxes(autorange="reversed", type="category", tickfont=dict(size=9), automargin=True, row=2, col=1)
    fig.update_xaxes(type="category", tickangle=-90, tickfont=dict(size=9), automargin=True, row=2, col=1)
    fig.update_layout(barmode="relative", height=260 + 13 * len(L), showlegend=False,
                      margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def assoc_heatmap(res: pd.DataFrame, rows: list[str], cols: list[str], labels: dict[str, str],
                  fdr: float = 0.1, cap: float = 4.0, what: str = "") -> go.Figure:
    """rows (mutations) x cols (regions): signed −log10 p, red = higher in mutated lines,
    blue = lower. ** q < fdr, * p < 0.05; untested cells blank."""
    ri, ci = {r: i for i, r in enumerate(rows)}, {c: j for j, c in enumerate(cols)}
    z = np.full((len(rows), len(cols)), np.nan)
    text = np.full(z.shape, "", dtype=object)
    hover = np.full(z.shape, "", dtype=object)
    for _, r in res.iterrows():
        i, j = ri[r["mutation"]], ci[r["region"]]
        z[i, j] = np.clip(np.sign(r["effect"]) * -np.log10(max(r["p"], 1e-300)), -cap, cap)
        text[i, j] = "**" if r["q"] < fdr else ("*" if r["p"] < 0.05 else "")
        hover[i, j] = (f"{labels.get(r['mutation'], r['mutation'])} · {r['region']}<br>{r['detail']}"
                       f"<br>p {r['p']:.3g} · q {r['q']:.3g}")
    fig = go.Figure(go.Heatmap(
        x=cols, y=[labels.get(r, r) for r in rows], z=z, text=text, customdata=hover, texttemplate="%{text}",
        zmin=-cap, zmax=cap, zmid=0, xgap=1, ygap=2, hoverongaps=False,
        colorscale=[[0, COL_A], [0.5, "rgba(140,140,140,0.10)"], [1, COL_B]],
        colorbar=dict(title=dict(text="signed −log10 p", side="right"), tickvals=[-cap, 0, cap],
                      ticktext=[f"lower {what}", "0", f"higher {what}"], len=0.9),
        hovertemplate="%{customdata}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", type="category", automargin=True)
    fig.update_xaxes(type="category", tickangle=-90, tickfont=dict(size=9), automargin=True)
    fig.update_layout(height=160 + 26 * len(rows), margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def similarity_heatmap(S: pd.DataFrame, hover: np.ndarray | None = None) -> go.Figure:
    """Square lines x lines similarity heatmap (already ordered, e.g. by clustering)."""
    fig = go.Figure(go.Heatmap(
        x=list(S.columns), y=list(S.index), z=S.to_numpy(), zmin=0, zmax=1, xgap=1, ygap=1,
        colorscale=[[0, "rgba(140,140,140,0.08)"], [1, COL_B]], hoverongaps=False,
        customdata=hover, colorbar=dict(title=dict(text="Jaccard", side="right"), len=0.7),
        hovertemplate=("%{y} × %{x}<br>similarity %{z:.2f}" + ("<br>%{customdata}" if hover is not None else "")
                       + "<extra></extra>")))
    fig.update_yaxes(autorange="reversed", type="category", tickfont=dict(size=9), automargin=True)
    fig.update_xaxes(type="category", tickangle=-90, tickfont=dict(size=9), automargin=True)
    side = 200 + 12 * len(S)
    fig.update_layout(height=min(max(side, 400), 1100), margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def neighbour_bars(nb: pd.DataFrame) -> go.Figure:
    """Similarity of the nearest lines; bar label = number of shared aberrations."""
    fig = go.Figure(go.Bar(x=nb["cell_line"], y=nb["similarity"], marker_color=COL_B,
                           customdata=nb["shared"], text=nb["n_shared"], textposition="outside",
                           hovertemplate="%{x}: %{y:.2f}<br>shared: %{customdata}<extra></extra>"))
    fig.update_layout(yaxis=dict(title="Jaccard similarity", range=[0, 1.1]), height=320,
                      margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(type="category", tickangle=-45))
    return fig


def drug_volcano(res: pd.DataFrame, hit: pd.Series, n_label: int = 6) -> go.Figure:
    """Aberration x drug tests: x = ΔAUC (with − without; < 0 = more sensitive with), y = −log10 p.
    Blue = significant and more sensitive, red = significant and more resistant; top n labelled."""
    r = res.assign(y=-np.log10(res["p"].clip(lower=1e-300)))
    col = np.where(hit & (r["dAUC"] < 0), COL_A, np.where(hit, COL_B, "rgba(140,140,140,0.35)"))
    txt = r["label"] + " · " + r["compound"].astype(str)
    fig = go.Figure(go.Scattergl(x=r["dAUC"], y=r["y"], mode="markers", text=txt,
                                 marker=dict(color=col, size=np.where(hit, 7, 5), line=dict(width=0)),
                                 hovertemplate="%{text}<br>ΔAUC %{x:+.3f}<br>−log10 p %{y:.2f}<extra></extra>"))
    for k, (_, t) in enumerate(r.nsmallest(n_label, "p").iterrows()):
        fig.add_annotation(x=t["dAUC"], y=t["y"], text=f"{t['label']} · {t['compound']}", showarrow=True, arrowhead=0,
                           arrowwidth=1, arrowcolor="#8a8984", ax=(60 if t["dAUC"] < 0 else -60), ay=-18 - 14 * (k % 3),
                           font=dict(size=11), bgcolor="rgba(255,255,255,0.75)")
    fig.add_vline(x=0, line_width=1, line_color="#52514e")
    fig.add_annotation(x=0, xref="paper", y=1.02, yref="paper", text="← more sensitive with aberration", showarrow=False,
                       xanchor="left", font=dict(size=11, color=COL_A))
    fig.add_annotation(x=1, xref="paper", y=1.02, yref="paper", text="more resistant with aberration →", showarrow=False,
                       xanchor="right", font=dict(size=11, color=COL_B))
    fig.update_layout(xaxis_title="ΔAUC (with − without)", yaxis_title="−log10(p-value)", height=440,
                      margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(zeroline=False, gridcolor="rgba(140,140,140,0.15)")
    fig.update_yaxes(gridcolor="rgba(140,140,140,0.15)")
    return fig


def target_bars(top: pd.DataFrame, unit: str) -> go.Figure:
    """Horizontal bars: median expression of the most highly expressed drug targets."""
    d = top.sort_values("median")
    fig = go.Figure(go.Bar(y=d["gene"], x=d["median"], orientation="h", marker_color=COL_A,
                           customdata=np.c_[d["n_drugs"], d["drugs"].str.slice(0, 80)],
                           hovertemplate="%{y}: %{x:.2f}<br>%{customdata[0]} drugs: %{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title=f"median across AML lines · {unit}")
    fig.update_yaxes(type="category", automargin=True, tickfont=dict(size=10))
    fig.update_layout(height=140 + 18 * len(d), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def group_box(df: pd.DataFrame, value_col: str, group_col: str, order: list[str], title: str,
              y_title: str = "", subtitle: str = "", colors: dict | None = None) -> go.Figure:
    """Tidy box + points per group for small multiples: horizontal wrapped labels with n,
    light box fill, outlined points, bold title with an optional stats subtitle."""
    colors = colors or {}
    fig = go.Figure()
    ticks = []
    for i, g in enumerate(order):
        s = df[df[group_col] == g].dropna(subset=[value_col])
        if s.empty:
            continue
        col = colors.get(g, COL_NS if g == "not annotated" else SEQ[i % len(SEQ)])
        tick = f"{_wrap(g)}<br><span style='font-size:10px'>n={len(s)}</span>"
        ticks.append(tick)
        fig.add_trace(go.Box(
            x=[tick] * len(s), y=s[value_col], name=g, boxpoints="all", jitter=0.45, pointpos=0, width=0.55,
            marker=dict(color=col, size=6, opacity=0.85, line=dict(width=0.8, color="white")),
            line=dict(color=col, width=1.5), fillcolor=_alpha(col, 0.12), text=s["cell_line"],
            hovertemplate="%{text}: %{y:.2f}<extra></extra>"))
    head = f"<b>{title}</b>" + (f"<br><span style='font-size:11px;color:#8a8984'>{subtitle}</span>" if subtitle else "")
    fig.update_layout(title=dict(text=head, x=0, xanchor="left", y=0.97, font=dict(size=15)),
                      yaxis_title=y_title, showlegend=False, height=340, boxgap=0.35,
                      margin=dict(l=10, r=10, t=58 if subtitle else 40, b=10))
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=ticks, tickangle=0, tickfont=dict(size=11))
    fig.update_yaxes(zeroline=False, gridcolor="rgba(140,140,140,0.2)")
    return fig


def _wrap(label: str, width: int = 14) -> str:
    """Break a group label onto two lines at a space / hyphen near the middle, for horizontal ticks."""
    if len(label) <= width:
        return label
    cut = max((i for i, ch in enumerate(label) if ch in " -/" and i <= len(label) * 0.6), default=-1)
    return label if cut <= 0 else label[:cut + 1].rstrip() + "<br>" + label[cut + 1:].lstrip()


def _alpha(color: str, a: float) -> str:
    """'#rrggbb' -> 'rgba(r,g,b,a)'; other colour strings are returned unchanged."""
    if color.startswith("#") and len(color) == 7:
        r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r},{g},{b},{a})"
    return color
