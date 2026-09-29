"""Discrete delta hedging of short options.

Each step a delta-hedged short option earns
    -1/2 Gamma S^2 ((dS/S)^2 - sigma^2 dt)
i.e. theta collects implied variance and gamma pays realized variance.
Checked on simulated GBM paths (Derman-Kamal hedging error, choice of hedging
vol, Leland transaction costs), then on SPY straddles 2011-2026.
"""

import numpy as np
import pandas as pd

from . import bs
from .surface import vix_consistent

DAYS_PER_YEAR = 365.0
STRADDLE = ("call", "put")


def gbm(S0, sigma, T, n, paths, r=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n
    steps = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((paths, n))
    logS = np.concatenate([np.zeros((paths, 1)), np.cumsum(steps, axis=1)], axis=1)
    return S0 * np.exp(logS), np.linspace(0.0, T, n + 1)


def _legs(right):
    return (right,) if isinstance(right, str) else tuple(right)


def delta_hedge(S, t, K, r, sigma_price, sigma_hedge, right="call", cost=0.0):
    """Short one option, or one of each leg in right, on every row of S.

    Rebalanced at every column but the last, which is expiry. t holds the
    dates in years, one row per path or one row shared. K, r and the vols are
    scalars or one per row. cost is charged on the notional of every stock
    trade, the opening hedge included; the unwind at expiry is free, as in
    Leland. Returns the final P&L and its forecast: the premium gap between the
    two vols plus the gamma-theta sum at the hedging vol.
    """
    legs = _legs(right)
    S = np.atleast_2d(np.asarray(S, dtype=float))
    t = np.broadcast_to(np.asarray(t, dtype=float), S.shape)
    T = t[:, -1]
    premium = lambda sigma: sum(bs.price(S[:, 0], K, T, r, 0.0, sigma, leg) for leg in legs)
    cash = premium(sigma_price) * np.ones(S.shape[0])
    forecast = premium(sigma_price) - premium(sigma_hedge)
    held = np.zeros(S.shape[0])
    for i in range(S.shape[1] - 1):
        g = [bs.greeks(S[:, i], K, T - t[:, i], r, 0.0, sigma_hedge, leg) for leg in legs]
        delta = sum(x["delta"] for x in g)
        gamma = sum(x["gamma"] for x in g)
        dt = t[:, i + 1] - t[:, i]
        trade = delta - held
        cash = (cash - trade * S[:, i] - cost * np.abs(trade) * S[:, i]) * np.exp(r * dt)
        held = delta
        ret = S[:, i + 1] / S[:, i] - 1.0
        forecast = forecast - 0.5 * gamma * S[:, i] ** 2 * (ret**2 - sigma_hedge**2 * dt)
    payoff = sum(np.maximum(bs.omega(leg) * (S[:, -1] - K), 0.0) for leg in legs)
    return cash + held * S[:, -1] - payoff, forecast


def derman_kamal(S, K, T, sigma, n, r=0.0):
    """Standard deviation of the hedging error with n rebalances, to leading order."""
    vega = float(bs.greeks(S, K, T, r, 0.0, sigma, "call")["vega"])
    return np.sqrt(np.pi / 4.0) * vega * sigma / np.sqrt(n)


def leland_vol(sigma, cost, dt):
    """Vol that prices in the rebalancing cost; cost is per side, Leland's k is round trip."""
    return sigma * np.sqrt(1.0 + np.sqrt(2.0 / np.pi) * 2.0 * cost / (sigma * np.sqrt(dt)))


def paths(panel, dte):
    """Every trading day's path to its expiry, padded at the front.

    Expiry is the last trading day on or before d0 + dte, so no trade holds
    time value over a weekend it cannot hedge. Padding repeats the first close
    at t = 0: a padded step has no move, no time and no trade. Returns S, t in
    years, and the number of trades.
    """
    dates = panel.index
    close = panel["close"].to_numpy(float)
    end = dates.searchsorted(dates + pd.Timedelta(days=dte), side="right")
    n = int((end < len(dates)).sum())
    k = np.arange(n)[:, None]
    last = end[:n, None] - k - 1
    width = int(last.max()) + 1 if n else 1
    idx = k + np.clip(np.arange(width) - (width - 1 - last), 0, None)
    days = (dates.values[idx] - dates.values[k]) / np.timedelta64(1, "D")
    return close[idx], days / DAYS_PER_YEAR, n


def replay(panel, dte=30, cost=0.0, vol_slippage=0.0, surfaces=None):
    """Short an ATM straddle every trading day, hedge daily at the close, hold to expiry.

    Priced and hedged at the ATM vol of that day's smile, rescaled so the
    smile's variance swap is VIX. cost is charged on stock trades,
    vol_slippage is the half-spread paid in vol points through the straddle's
    vega. One row per trade, P&L as a fraction of the strike, costs included.
    """
    if surfaces is None:
        surfaces, _ = vix_consistent(panel)
    S, t, n = paths(panel, dte)
    T = t[:, -1]
    K = S[:, 0]
    r = panel["rate"].to_numpy(float)[:n]
    iv = np.array([float(surfaces[i].vol(K[i], T[i], K[i])) for i in range(n)])
    pnl, forecast = delta_hedge(S, t, K, r, iv, iv, STRADDLE, cost)
    pnl = pnl - 2.0 * bs.greeks(K, K, T, r, 0.0, iv, "call")["vega"] * vol_slippage
    rv = np.sqrt(np.sum(np.diff(np.log(S), axis=1) ** 2, axis=1) / T)
    return pd.DataFrame(
        {
            "days": np.round(T * DAYS_PER_YEAR).astype(int),
            "strike": K,
            "vix": panel["vix"].to_numpy(float)[:n],
            "iv": iv,
            "rv": rv,
            "pnl": pnl / K,
            "forecast": forecast / K,
        },
        index=pd.Index(panel.index[:n], name="date"),
    )
