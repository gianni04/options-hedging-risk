"""One-day 99% VaR of a delta-hedged short straddle, five methods, backtested.

The book is rebuilt daily: short one 30-day ATM straddle, hedged to zero delta.
Realized next-day P&L is repriced on the next day's smile at the old strike.
A second series moves the strike's vol by the change in ATM vol (parallel
shift), which is what most VaR models assume; it is used as a comparison.

All methods use the same 500-day window of (log return, change in ATM vol).
Backtests: Kupiec (exception count) and Christoffersen (independence).
"""

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm

from . import bs
from .hedge import DAYS_PER_YEAR, STRADDLE
from .surface import vix_consistent

METHODS = ("delta_normal", "delta_gamma", "delta_gamma_vega", "full_parallel", "full_smile")
ACTUALS = ("realized", "parallel")
T_BOOK = 30.0 / DAYS_PER_YEAR
VOL_FLOOR = 1e-4


def _straddle(S, K, T, r, sigma):
    return sum(bs.price(S, K, T, r, 0.0, sigma, leg) for leg in STRADDLE)


def _greeks(S, K, T, r, sigma):
    g = [bs.greeks(S, K, T, r, 0.0, sigma, leg) for leg in STRADDLE]
    return {k: float(sum(x[k] for x in g)) for k in ("delta", "gamma", "vega", "theta")}


def backtest(panel, window=500, alpha=0.99):
    """Per day: realized P&L of the book and the VaR each method forecast for it.

    Everything is a fraction of spot. The row dated d holds the P&L from the
    previous close to d and the VaRs known at that previous close. smile_flag
    marks days whose smile, on either close, had no variance swap to rescale
    onto VIX.
    """
    surfaces, lam = vix_consistent(panel)
    S = panel["close"].to_numpy(float)
    x = np.diff(np.log(S))
    dsig = np.diff([float(s.atm(T_BOOK)) for s in surfaces])
    gap = np.diff(panel.index).astype("timedelta64[D]").astype(float) / DAYS_PER_YEAR
    flag = lam.isna().to_numpy()
    z, q = norm.ppf(alpha), 1.0 - alpha
    rows = []
    for i in range(window, len(S) - 1):
        s0, K, r, T1 = S[i], S[i], float(panel["rate"].iloc[i]), T_BOOK - gap[i]
        iv = float(surfaces[i].vol(K, T_BOOK, s0))
        g = _greeks(s0, K, T_BOOK, r, iv)
        v0 = float(_straddle(s0, K, T_BOOK, r, iv))
        hedge = g["delta"]
        book_delta = hedge - g["delta"]

        dS1 = S[i + 1] - s0
        realized = float(_straddle(S[i + 1], K, T1, r, surfaces[i + 1].vol(K, T1, S[i + 1])))
        parallel = float(_straddle(S[i + 1], K, T1, r, max(iv + dsig[i], VOL_FLOOR)))

        hx, hs = x[i - window:i], dsig[i - window:i]
        dS = s0 * np.expm1(hx)
        dg = -(0.5 * g["gamma"] * dS**2 + g["theta"] * gap[i])
        dgv = dg - g["vega"] * hs
        full_parallel = -(_straddle(s0 + dS, K, T1, r, np.maximum(iv + hs, VOL_FLOOR)) - v0) + hedge * dS
        full_smile = -(_straddle(s0 + dS, K, T1, r, surfaces[i].vol(K, T1, s0 + dS, shift=hs)) - v0) + hedge * dS
        rows.append((
            panel.index[i + 1],
            (-(realized - v0) + hedge * dS1) / s0,
            (-(parallel - v0) + hedge * dS1) / s0,
            z * abs(book_delta) * hx.std(),
            -np.quantile(dg, q) / s0,
            -np.quantile(dgv, q) / s0,
            -np.quantile(full_parallel, q) / s0,
            -np.quantile(full_smile, q) / s0,
            bool(flag[i] or flag[i + 1]),
        ))
    return pd.DataFrame(rows, columns=["date", *ACTUALS, *METHODS, "smile_flag"]).set_index("date")


def kupiec(exceptions, n, p):
    """Proportion-of-failures likelihood ratio and its chi-square(1) p-value."""
    x = int(exceptions)
    loglik = lambda pi: xlogy(n - x, 1.0 - pi) + xlogy(x, pi)
    lr = float(-2.0 * (loglik(p) - loglik(x / n)))
    return lr, float(chi2.sf(lr, 1))


def christoffersen(hits):
    """Independence likelihood ratio: does an exception make the next one likelier?"""
    h = np.asarray(hits, dtype=bool)
    a, b = h[:-1], h[1:]
    n00, n01 = int((~a & ~b).sum()), int((~a & b).sum())
    n10, n11 = int((a & ~b).sum()), int((a & b).sum())
    loglik = lambda n0, n1: xlogy(n0, n0 / max(n0 + n1, 1)) + xlogy(n1, n1 / max(n0 + n1, 1))
    lr = float(-2.0 * (loglik(n00 + n10, n01 + n11) - loglik(n00, n01) - loglik(n10, n11)))
    return lr, float(chi2.sf(lr, 1))


def summary(bt, actual="realized", alpha=0.99):
    rows = {}
    for m in METHODS:
        d = bt[[actual, m]].dropna()
        hits = (d[actual] < -d[m]).to_numpy()
        lr, pv = kupiec(hits.sum(), len(d), 1.0 - alpha)
        lr_ind, pv_ind = christoffersen(hits)
        rows[m] = {
            "mean_var": d[m].mean(),
            "exceptions": int(hits.sum()),
            "rate": hits.mean(),
            "kupiec_p": pv,
            "independence_p": pv_ind,
        }
    return pd.DataFrame.from_dict(rows, orient="index")
