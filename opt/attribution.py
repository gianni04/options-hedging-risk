"""Daily P&L attribution by greek: delta, gamma, theta, vega, vanna, volga, rho."""

import numpy as np
import pandas as pd

from .legs import DAYS_PER_YEAR

TERMS = (
    "delta_pnl",
    "gamma_pnl",
    "theta_pnl",
    "vega_pnl",
    "volga_pnl",
    "vanna_pnl",
    "rho_pnl",
    "residual",
)
STATE = ("dS", "dsigma", "dt", "dr")
COLUMNS = TERMS + ("total",) + STATE

_OPTION_KINDS = ("call", "put")
_ZERO_VEGA = 1e-12


def vega_weighted_dsigma(position, m0, m1):
    """Change in implied vol seen by the position, weighted by leg vega at m0.

    Each leg's vol under m1 is read at the maturity left under m1, not at the
    maturity it had under m0. Returns nan when the position carries no vega.
    """
    num = 0.0
    den = 0.0
    for leg in position.legs:
        if leg.kind not in _OPTION_KINDS:
            continue
        t0 = m0.years_to(leg.expiry)
        t1 = m1.years_to(leg.expiry)
        v = leg.greeks(m0)["vega"]
        num += v * (m1.sigma(leg.strike, t1) - m0.sigma(leg.strike, t0))
        den += v
    if abs(den) < _ZERO_VEGA:
        return float("nan")
    return num / den


def attribute(position, m0, m1):
    """Split the value change from m0 to m1 into second-order greek terms.

    Greeks are read at m0. Whatever the expansion misses lands in 'residual'.
    """
    g = position.greeks(m0)
    ds = float(m1.spot - m0.spot)
    dr = float(m1.r - m0.r)
    dt = (m1.asof - m0.asof).days / DAYS_PER_YEAR
    dsigma = vega_weighted_dsigma(position, m0, m1)
    dv = 0.0 if np.isnan(dsigma) else dsigma

    out = {
        "delta_pnl": g["delta"] * ds,
        "gamma_pnl": 0.5 * g["gamma"] * ds * ds,
        "theta_pnl": g["theta"] * dt,
        "vega_pnl": g["vega"] * dv,
        "volga_pnl": 0.5 * g["volga"] * dv * dv,
        "vanna_pnl": g["vanna"] * ds * dv,
        "rho_pnl": g["rho"] * dr,
    }
    total = position.value(m1) - position.value(m0)
    out["residual"] = total - sum(out.values())
    out["total"] = total
    out["dS"] = ds
    out["dsigma"] = dsigma
    out["dt"] = dt
    out["dr"] = dr
    return out


def attribute_path(position, markets):
    """Attribution of every consecutive step of a market path.

    Indexed by the closing date of each step, with '_cum' columns carrying the
    running contribution of each term.
    """
    markets = list(markets)
    if len(markets) < 2:
        raise ValueError("attribute_path needs at least two markets")
    rows = [attribute(position, a, b) for a, b in zip(markets[:-1], markets[1:])]
    index = pd.Index([m.asof for m in markets[1:]], name="asof")
    df = pd.DataFrame(rows, index=index, columns=list(COLUMNS))

    endpoints = position.value(markets[-1]) - position.value(markets[0])
    drift = float(df["total"].sum() - endpoints)
    if abs(drift) > 1e-8:
        raise ValueError(f"path does not telescope, off by {drift:.3e}")

    for c in TERMS + ("total",):
        df[c + "_cum"] = df[c].cumsum()
    return df


def summary(df):
    """Contribution of each term over the whole path, and its share of gross P&L.

    The share is taken against the sum of absolute contributions, so terms that
    cancel each other stay visible instead of netting to nothing.
    """
    cols = [c for c in TERMS if c in df.columns]
    totals = df[cols].sum()
    gross = float(totals.abs().sum())
    out = {}
    for c in cols:
        out[c] = float(totals[c])
        out[c + "_pct"] = 100.0 * float(totals[c]) / gross if gross else float("nan")
    out["gross"] = gross
    out["total"] = float(df["total"].sum()) if "total" in df.columns else float("nan")
    return pd.Series(out)


def plot_attribution(df, path, title="P&L attribution"):
    """Stacked area of the cumulative contributions, written to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [c for c in TERMS if c + "_cum" in df.columns]
    x = np.arange(len(df.index))
    pos = np.zeros(len(df.index))
    neg = np.zeros(len(df.index))
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, c in enumerate(cols):
        y = df[c + "_cum"].to_numpy(dtype=float)
        up = np.maximum(y, 0.0)
        down = np.minimum(y, 0.0)
        ax.fill_between(x, pos, pos + up, color=colors(i % 10), label=c, linewidth=0)
        ax.fill_between(x, neg, neg + down, color=colors(i % 10), linewidth=0)
        pos = pos + up
        neg = neg + down

    if "total_cum" in df.columns:
        ax.plot(x, df["total_cum"].to_numpy(dtype=float), color="black", lw=2.0, label="total")

    step = max(1, len(x) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(d) for d in df.index[::step]], rotation=45, ha="right")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("cumulative P&L")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=3, fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
