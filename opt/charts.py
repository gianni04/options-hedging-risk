"""Figures for option positions: payoff diagrams, greek profiles, vol surfaces.

P&L curves are net of the opening cost, so zero is the breakeven line.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import bs  # noqa: E402

INK = "#1a1a1a"
MUTED = "#8a8a8a"
ACCENT = "#c0392b"
BLUE = "#2c5f8a"
GREEN = "#2d7a4f"
GRID = "#e3e3e3"
PALETTE = (BLUE, ACCENT, GREEN, "#8e44ad", "#d4820a", "#0f7f8f")

_YEAR = 365.0

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    }
)


def finish(fig, path: Path, title: str, subtitle: str = "") -> None:
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.01, ha="left", y=1.0)
    if subtitle:
        fig.text(0.01, 0.955, subtitle, fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90 if subtitle else 0.95))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _spot_grid(spot: float, spot_range: float, n: int = 400):
    return np.linspace(spot * (1 - spot_range), spot * (1 + spot_range), n)


def _strikes(position):
    return sorted({leg.strike for leg in position.legs if leg.kind in ("call", "put")})


def _value_curve(position, market, spots):
    return np.array([position.value(replace(market, spot=float(s))) for s in spots])


def _greek_curve(position, market, spots, greek):
    return np.array(
        [position.greeks(replace(market, spot=float(s)))[greek] for s in spots]
    )


def _is_capped(curve, side):
    span = curve.max() - curve.min()
    if span < 1e-9:
        return True
    window = max(int(len(curve) * 0.03), 2)
    segment = curve[-window:] if side == "right" else curve[:window]
    drift = abs(segment[-1] - segment[0])
    return drift < 1e-3 * span


def _breakevens(spots, curve):
    sign = np.sign(curve)
    crossings = []
    for i in np.where(np.diff(sign) != 0)[0]:
        x0, x1, y0, y1 = spots[i], spots[i + 1], curve[i], curve[i + 1]
        if y1 != y0:
            crossings.append(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
    return crossings


def _annotate_payoff(ax, spots, curve):
    idx_max, idx_min = int(np.argmax(curve)), int(np.argmin(curve))
    if 0 < idx_max < len(curve) - 1 or _is_capped(curve, "right"):
        ax.scatter([spots[idx_max]], [curve[idx_max]], color=GREEN, zorder=5, s=22)
        ax.annotate(
            f"max profit {curve[idx_max]:.2f}", xy=(spots[idx_max], curve[idx_max]),
            xytext=(8, 8), textcoords="offset points", fontsize=8, color=GREEN,
        )
    if 0 < idx_min < len(curve) - 1 or _is_capped(curve, "left"):
        ax.scatter([spots[idx_min]], [curve[idx_min]], color=ACCENT, zorder=5, s=22)
        ax.annotate(
            f"max loss {curve[idx_min]:.2f}", xy=(spots[idx_min], curve[idx_min]),
            xytext=(8, -12), textcoords="offset points", fontsize=8, color=ACCENT,
        )
    for x in _breakevens(spots, curve):
        ax.axvline(x, color=MUTED, linewidth=0.7, linestyle=":")
        ax.annotate(
            f"breakeven {x:.2f}", xy=(x, 0), xytext=(4, 10),
            textcoords="offset points", fontsize=7.5, color=MUTED, rotation=90,
        )


def payoff(position, market, path: Path, spot_range: float = 0.30, title: str = None) -> None:
    """Payoff at expiry against value today, both net of the cost to open."""
    spots = _spot_grid(market.spot, spot_range)
    cost = position.cost(market)
    at_expiry = position.payoff(spots, market) - cost
    today = _value_curve(position, market, spots) - cost

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(spots, at_expiry, color=INK, linewidth=1.8, label="at expiry")
    ax.plot(spots, today, color=BLUE, linewidth=1.4, linestyle="--", label="today")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(market.spot, color=MUTED, linewidth=0.8, linestyle=":")
    for strike in _strikes(position):
        ax.axvline(strike, color=GRID, linewidth=1.0, zorder=0)
    _annotate_payoff(ax, spots, at_expiry)
    ax.set_xlabel("spot")
    ax.set_ylabel("P&L, net of cost")
    ax.legend(loc="best")

    label = title or f"{position.name or 'position'} payoff"
    side = "debit" if cost >= 0 else "credit"
    finish(fig, path, label, f"net of a {side} of {abs(cost):.2f}, spot at {market.spot:g}")


def greek_profile(
    position, market, path: Path, greek: str = "delta",
    spot_range: float = 0.30, days=(0, 7, 30),
) -> None:
    """One greek against spot, one curve per days remaining to the first expiry."""
    expiries = position.expiries()
    anchor = expiries[0] if expiries else market.asof
    spots = _spot_grid(market.spot, spot_range)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for colour, d in zip(PALETTE, days):
        m_d = replace(market, asof=anchor - timedelta(days=int(d)))
        curve = _greek_curve(position, m_d, spots, greek)
        ax.plot(spots, curve, color=colour, linewidth=1.6, label=f"{d}d to expiry")

    ax.axvline(market.spot, color=MUTED, linewidth=0.8, linestyle=":")
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_xlabel("spot")
    ax.set_ylabel(greek)
    ax.legend(loc="best")

    finish(
        fig, path, f"{position.name or 'position'} {greek}",
        "curves at decreasing days to the first expiry",
    )


def greek_grid(position, market, path: Path, spot_range: float = 0.30) -> None:
    """Delta, gamma, vega and theta against spot, four panels."""
    spots = _spot_grid(market.spot, spot_range)
    panels = (("delta", BLUE), ("gamma", ACCENT), ("vega", GREEN), ("theta", "#8e44ad"))

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (greek, colour) in zip(axes.flat, panels):
        curve = _greek_curve(position, market, spots, greek)
        ax.plot(spots, curve, color=colour, linewidth=1.6)
        ax.axhline(0, color=INK, linewidth=0.6)
        ax.axvline(market.spot, color=MUTED, linewidth=0.7, linestyle=":")
        ax.set_title(greek, fontsize=9, loc="left")
        ax.set_xlabel("spot")

    finish(fig, path, f"{position.name or 'position'} greeks")


def smile(market, path: Path, expiries_days=(9, 30, 90, 180), moneyness=(0.7, 1.3)) -> None:
    """Implied vol against strike, one curve per expiry."""
    strikes = np.linspace(market.spot * moneyness[0], market.spot * moneyness[1], 200)
    flat = not callable(market.vol)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for colour, d in zip(PALETTE, expiries_days):
        T = d / _YEAR
        iv = np.array([market.sigma(k, T) for k in strikes])
        ax.plot(strikes, iv, color=colour, linewidth=1.6, label=f"{d}d")

    ax.axvline(market.spot, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("strike")
    ax.set_ylabel("implied vol")
    ax.legend(loc="best")

    title = "Implied volatility smile"
    if flat:
        title += " (flat input vol, same at every strike)"
    finish(fig, path, title)


def term_structure(market, path: Path, days=range(7, 365, 7)) -> None:
    """Implied vol at the forward strike against maturity."""
    days_arr = np.asarray(list(days), dtype=float)
    T = days_arr / _YEAR
    forward_strikes = bs.forward(market.spot, T, market.r, market.q)
    iv = np.array([market.sigma(k, t) for k, t in zip(forward_strikes, T)])

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(days_arr, iv, color=BLUE, linewidth=1.8)
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.set_xlabel("days to maturity")
    ax.set_ylabel("implied vol at the forward")

    finish(fig, path, "Volatility term structure")


def equity(df: pd.DataFrame, path: Path, columns=None, title: str = None) -> None:
    """Cumulative P&L with the drawdown underneath, never without it."""
    cols = list(columns) if columns is not None else list(df.columns)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    for colour, name in zip(PALETTE, cols):
        series = df[name]
        top.plot(series.index, series.to_numpy(), color=colour, linewidth=1.6, label=name)
        drawdown = series - series.cummax()
        bottom.fill_between(series.index, 0, drawdown.to_numpy(), color=colour, alpha=0.30)

    top.axhline(0, color=INK, linewidth=0.8)
    top.set_ylabel("equity")
    top.legend(loc="upper left")
    bottom.set_ylabel("drawdown")

    finish(fig, path, title or "Equity curve")


def comparison_bars(series: pd.Series, path: Path, title: str = None, xlabel: str = None) -> None:
    """Sorted horizontal bars, for comparing strategies side by side."""
    ordered = series.sort_values()
    height = max(3.0, 0.4 * len(ordered) + 1.2)
    fig, ax = plt.subplots(figsize=(7, height))

    values = ordered.to_numpy(dtype=float)
    colours = np.where(values < 0, ACCENT, BLUE)
    y = np.arange(len(ordered))
    ax.barh(y, values, color=colours)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered.index)
    ax.axvline(0, color=INK, linewidth=0.8)
    if xlabel:
        ax.set_xlabel(xlabel)

    finish(fig, path, title or "Strategy comparison")
