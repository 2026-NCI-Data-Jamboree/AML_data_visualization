"""Kaplan-Meier estimates and the log-rank test (same results as R survival::survfit / survdiff)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def kaplan_meier(time, event) -> pd.DataFrame:
    """Step function: time, n_risk, n_event, n_censor, surv, lower, upper (95% log-log CI).
    Starts at time 0 with surv = 1."""
    t = np.asarray(time, float)
    e = np.asarray(event, bool)
    ok = np.isfinite(t)
    t, e = t[ok], e[ok]
    rows = [(0.0, len(t), 0, 0, 1.0, 1.0, 1.0)]
    s, var = 1.0, 0.0
    for u in np.unique(t):
        n = int((t >= u).sum())
        d = int(((t == u) & e).sum())
        c = int(((t == u) & ~e).sum())
        if d:
            s *= 1 - d / n
            var += d / (n * (n - d)) if n > d else np.inf
        lo = hi = s
        if 0 < s < 1 and np.isfinite(var):
            se = np.sqrt(var) / abs(np.log(s))            # log(-log S) scale, as R's conf.type="log-log"
            lo, hi = s ** np.exp(1.96 * se), s ** np.exp(-1.96 * se)
        rows.append((u, n, d, c, s, lo, hi))
    return pd.DataFrame(rows, columns=["time", "n_risk", "n_event", "n_censor", "surv", "lower", "upper"])


def median_survival(km: pd.DataFrame) -> float:
    """First time at which survival drops to 0.5 or below (NaN if never)."""
    hit = km[km["surv"] <= 0.5]
    return float(hit["time"].iloc[0]) if len(hit) else np.nan


def logrank(time, event, group, order: list | None = None) -> dict:
    """k-group log-rank test. Returns chi2, df, p, and per-group observed / expected events.
    For two groups also the hazard ratio (first group vs second, in `order` if given, else in order
    of appearance) as (O1/E1)/(O2/E2)."""
    d = pd.DataFrame({"t": np.asarray(time, float), "e": np.asarray(event, bool),
                      "g": np.asarray(group, object)}).dropna()
    present = set(d["g"])
    levels = [g for g in order if g in present] if order is not None else list(dict.fromkeys(d["g"]))
    k = len(levels)
    if k < 2:
        return {"chi2": np.nan, "df": 0, "p": np.nan, "observed": {}, "expected": {}, "hr": np.nan}
    O, E = np.zeros(k), np.zeros(k)
    V = np.zeros((k, k))
    for u in np.unique(d.loc[d["e"], "t"]):
        at_risk = d["t"] >= u
        n_j = np.array([(at_risk & (d["g"] == g)).sum() for g in levels], float)
        d_j = np.array([(at_risk & (d["t"] == u) & d["e"] & (d["g"] == g)).sum() for g in levels], float)
        n, dd = n_j.sum(), d_j.sum()
        O += d_j
        E += dd * n_j / n
        if n > 1:
            f = dd * (n - dd) / (n * n * (n - 1))
            V += f * (np.diag(n_j) * n - np.outer(n_j, n_j))
    z = (O - E)[:-1]
    try:
        chi2 = float(z @ np.linalg.solve(V[:-1, :-1], z))
    except np.linalg.LinAlgError:
        chi2 = np.nan
    hr = (O[0] / E[0]) / (O[1] / E[1]) if k == 2 and E[0] > 0 and E[1] > 0 and O[1] > 0 else np.nan
    return {"chi2": chi2, "df": k - 1, "p": float(stats.chi2.sf(chi2, k - 1)) if np.isfinite(chi2) else np.nan,
            "observed": dict(zip(levels, O)), "expected": dict(zip(levels, E)), "hr": hr}


def at_risk(time, group, ticks) -> pd.DataFrame:
    """Number at risk per group (rows) at each tick (columns)."""
    d = pd.DataFrame({"t": np.asarray(time, float), "g": np.asarray(group, object)}).dropna()
    return pd.DataFrame({f"{x:g}": [(d.loc[d["g"] == g, "t"] >= x).sum() for g in dict.fromkeys(d["g"])]
                         for x in ticks}, index=list(dict.fromkeys(d["g"])))
