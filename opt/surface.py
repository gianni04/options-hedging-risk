"""Implied volatility surface rebuilt from CBOE indices.

- VIX9D, VIX, VIX3M: at-the-money term structure (interpolated in total variance)
- SKEW: slope of the smile, by matching the risk-neutral skewness
- VVIX: curvature

The smile is calibrated with a no-arbitrage constraint (non-negative density).
When SKEW asks for more skew than that allows, calibration stops at the bound
and the shortfall is recorded. The at-the-money level is rescaled so that the
smile's variance swap equals VIX.
"""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import bs

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

ANCHORS = (9.0 / 365.0, 30.0 / 365.0, 90.0 / 365.0)
Z_SAT = 3.0
CALIB_T = 30.0 / 365.0
GAMMA_BASE = 0.040
VVIX_REF = 0.86
GAMMA_BOUNDS = (0.010, 0.090)
BETA_BOUNDS = (-1.20, -1e-4)
CHECK_T = (9.0 / 365.0, 30.0 / 365.0, 90.0 / 365.0, 182.0 / 365.0, 1.0)
GRID_N, GRID_SD = 1500, 6.0


def atm_vol(T, vix9d, vix, vix3m):
    """ATM volatility at maturity T, interpolated in total variance."""
    T = np.asarray(T, dtype=float)
    t = np.array(ANCHORS)
    anchors = np.array([vix9d, vix, vix3m], dtype=float)
    anchors = np.where(np.isfinite(anchors), anchors, float(vix))
    w = np.maximum.accumulate(anchors**2 * t)
    wT = np.interp(T, t, w)
    wT = np.where(T < t[0], w[0] / t[0] * T, wT)
    wT = np.where(T > t[2], w[2] + (w[2] - w[1]) / (t[2] - t[1]) * (T - t[2]), wT)
    return np.sqrt(np.maximum(wT, 1e-10) / np.maximum(T, 1e-10))


def saturate(z):
    """Smooth saturation of standardised moneyness.

    A hard clip leaves a kink. Its second derivative is a spike, and that spike
    shows up as a negative density. tanh is smooth everywhere, so it does not.
    """
    return Z_SAT * np.tanh(np.asarray(z, dtype=float) / Z_SAT)


@dataclass(frozen=True)
class Surface:
    vix9d: float
    vix: float
    vix3m: float
    beta: float
    gamma: float
    r: float = 0.0
    q: float = 0.0

    def atm(self, T):
        return atm_vol(T, self.vix9d, self.vix, self.vix3m)

    def forward(self, spot, T):
        return np.asarray(spot, dtype=float) * np.exp(
            (self.r - self.q) * np.asarray(T, dtype=float)
        )

    def vol(self, strike, T, spot, shift=0.0):
        """shift moves the at-the-money level; the smile keeps its shape in z."""
        T = np.maximum(np.asarray(T, dtype=float), 1.0 / 365.0)
        sa = np.maximum(self.atm(T) + shift, 1e-4)
        z = saturate(
            np.log(np.asarray(strike, dtype=float) / self.forward(spot, T))
            / (sa * np.sqrt(T))
        )
        return sa * np.exp(self.beta * z + 0.5 * self.gamma * z * z)

    def bind(self, spot):
        """A plain (strike, T) -> vol callable, for legs.Market."""
        return lambda k, t: float(self.vol(k, t, spot))


def density(surface, spot, T, n=GRID_N, sd=GRID_SD):
    """Risk-neutral density of the terminal price, differentiated analytically.

    f(K) = e^{rT} vega [ 1/(K^2 s T) + 2 d1 s'/(K sqrt(T) s)
                         + d1 d2 s'^2 / s + s'' ]

    Only the volatility function is differenced numerically. It is smooth and of
    order one, unlike the call price, which spans fifteen decades and turns a
    double numerical derivative into noise.

    The strike grid is asymmetric. A skewed smile puts far more volatility in
    the left wing than at the money, so a grid sized on the at-the-money
    volatility truncates the tail that carries the skewness.
    """
    T = float(T)
    F = float(surface.forward(spot, T))
    sa = float(surface.atm(T))
    wing = float(np.exp(-surface.beta * Z_SAT + 0.5 * surface.gamma * Z_SAT**2))
    lo = sd * sa * wing * np.sqrt(T)
    hi = sd * sa * np.sqrt(T)
    K = F * np.exp(np.linspace(-lo, hi, n))
    h = K * 1e-4
    s = surface.vol(K, T, spot)
    up = surface.vol(K + h, T, spot)
    dn = surface.vol(K - h, T, spot)
    sp = (up - dn) / (2.0 * h)
    spp = (up - 2.0 * s + dn) / (h * h)
    sqT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s * T) / (s * sqT)
    d2 = d1 - s * sqT
    vega = (
        spot * np.exp(-surface.q * T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) * sqT
    )
    f = np.exp(surface.r * T) * vega * (
        1.0 / (K * K * s * T)
        + 2.0 * d1 * sp / (K * sqT * s)
        + d1 * d2 * sp * sp / s
        + spp
    )
    return K, f


def arbitrage_ratio(surface, spot, T):
    """min(f) / max(f). Negative means a butterfly arbitrage exists."""
    _, f = density(surface, spot, T)
    peak = float(f.max())
    return float(f.min() / peak) if peak > 0 else -1.0


def log_moments(surface, spot, T):
    """Mean, variance, skewness and excess kurtosis of the log return."""
    K, f = density(surface, spot, T)
    f = np.maximum(f, 0.0)
    mass = float(_trapezoid(f, K))
    if mass <= 0:
        return (np.nan,) * 4
    x = np.log(K / float(surface.forward(spot, T)))
    m1 = float(_trapezoid(x * f, K)) / mass
    c = x - m1
    m2 = float(_trapezoid(c**2 * f, K)) / mass
    m3 = float(_trapezoid(c**3 * f, K)) / mass
    m4 = float(_trapezoid(c**4 * f, K)) / mass
    if m2 <= 0:
        return (np.nan,) * 4
    return m1, m2, m3 / m2**1.5, m4 / m2**2 - 3.0


def _probe(beta, gamma, vix9d, vix, vix3m):
    return Surface(vix9d, vix, vix3m, beta, gamma)


def worst_arbitrage(surface, spot=100.0, maturities=CHECK_T):
    """The most negative density ratio across the whole term structure."""
    return min(arbitrage_ratio(surface, spot, T) for T in maturities)


@lru_cache(maxsize=8192)
def _bound_cached(gamma, vix9d, vix, vix3m, tol=1e-9):
    g = lambda b: worst_arbitrage(_probe(b, gamma, vix9d, vix, vix3m)) + tol
    lo, hi = BETA_BOUNDS
    if g(hi) < 0:
        return hi
    if g(lo) > 0:
        return lo
    return brentq(g, lo, hi, xtol=1e-6)


def arbfree_beta_bound(gamma, vix9d, vix, vix3m, tol=1e-9):
    """Steepest slope with a non-negative density at every maturity checked.

    The bound is computed on the day's own term structure, not on a flat one:
    a smile that is arbitrage-free at the thirty-day volatility need not be at
    the nine-day volatility, and the nine-day point is where it breaks first.

    Rounded before caching, since the bound moves far more slowly than its
    inputs and thousands of days collapse onto a few hundred calibrations.
    """
    return _bound_cached(
        round(float(gamma), 3), round(float(vix9d), 3),
        round(float(vix), 3), round(float(vix3m), 3), tol,
    )


def gamma_from_vvix(vvix):
    """Smile curvature scaled by the volatility of volatility."""
    return float(np.clip(GAMMA_BASE * float(vvix) / VVIX_REF, *GAMMA_BOUNDS))


def calibrate_beta(target_skewness, gamma, vix9d, vix, vix3m, T=CALIB_T):
    """Slope reproducing SKEW's skewness, capped at the no-arbitrage boundary.

    Returns (beta, achieved_skewness, constrained).
    """
    skew = lambda b: log_moments(_probe(b, gamma, vix9d, vix, vix3m), 100.0, T)[2]
    bound = arbfree_beta_bound(gamma, vix9d, vix, vix3m)
    at_bound = skew(bound)
    if not np.isfinite(at_bound) or at_bound >= target_skewness:
        return bound, float(at_bound), True
    hi = BETA_BOUNDS[1]
    if skew(hi) - target_skewness < 0:
        return hi, float(skew(hi)), True
    beta = brentq(lambda b: skew(b) - target_skewness, bound, hi, xtol=1e-6)
    return beta, float(skew(beta)), False


def build(row):
    """Calibrate one day. `row` needs vix9d, vix, vix3m, skewness, vvix, rate."""
    vix = float(row["vix"])
    v9 = float(row.get("vix9d", np.nan))
    v3 = float(row.get("vix3m", np.nan))
    v9 = v9 if np.isfinite(v9) else vix
    v3 = v3 if np.isfinite(v3) else vix
    gamma = gamma_from_vvix(row["vvix"])
    beta, achieved, constrained = calibrate_beta(
        float(row["skewness"]), gamma, v9, vix, v3
    )
    s = Surface(v9, vix, v3, beta, gamma, r=float(row.get("rate", 0.0)))
    return s, {
        "beta": beta,
        "gamma": gamma,
        "skew_target": float(row["skewness"]),
        "skew_achieved": achieved,
        "constrained": bool(constrained),
    }


def build_all(df, progress_every=1000):
    """Calibrate every day. Returns a diagnostics frame indexed like df."""
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        rows.append(build(row)[1])
        if progress_every and i and i % progress_every == 0:
            print(f"  {i}/{len(df)}", flush=True)
    out = pd.DataFrame(rows, index=df.index)
    out["skew_shortfall"] = out["skew_target"] - out["skew_achieved"]
    return out


def surfaces_from(df, diag):
    """Rebuild Surface objects from a calibrated diagnostics frame."""
    return {
        ts: Surface(
            float(df.at[ts, "vix9d"]),
            float(df.at[ts, "vix"]),
            float(df.at[ts, "vix3m"]),
            float(diag.at[ts, "beta"]),
            float(diag.at[ts, "gamma"]),
            r=float(df.at[ts, "rate"]),
        )
        for ts in df.index
    }


def from_row(row):
    """The day's surface from a market_data row joined to its calibration."""
    return Surface(
        float(row["vix9d"]), float(row["vix"]), float(row["vix3m"]),
        float(row["beta"]), float(row["gamma"]), r=float(row["rate"]),
    )


def scaled(surface, lam):
    """The same smile with the whole at-the-money term structure multiplied by lam."""
    return replace(surface, vix9d=surface.vix9d * lam, vix=surface.vix * lam, vix3m=surface.vix3m * lam)


def variance_swap(surface, T, n=GRID_N, sd=GRID_SD):
    """Fair variance-swap volatility of the smile: the number VIX publishes.

    2 e^{rT} / T times the integral of out-of-the-money prices over K^2, on the
    same asymmetric strike grid as the density.
    """
    T = float(T)
    F = float(surface.forward(1.0, T))
    sa = float(surface.atm(T))
    wing = float(np.exp(-surface.beta * Z_SAT + 0.5 * surface.gamma * Z_SAT**2))
    K = F * np.exp(np.linspace(-sd * sa * wing, sd * sa, n) * np.sqrt(T))
    v = surface.vol(K, T, 1.0)
    right = np.where(K < F, "put", "call")
    otm = bs.price(1.0, K, T, surface.r, surface.q, v, right)
    return float(np.sqrt(2.0 * np.exp(surface.r * T) / T * _trapezoid(otm / K**2, K)))


def atm_scale(surface, T=CALIB_T):
    """Factor on the at-the-money level that makes the smile's variance swap equal VIX.

    VIX is a variance swap, not an at-the-money vol: a negative skew puts it
    above, by 6% to 35% of VIX on 3,920 of 3,934 days. On the other fourteen,
    all in March 2020 but one, the slope sits at or near its box bound, the
    saturated put wing carries the integral to forty-five times VIX or more, and
    the answer depends on where the grid stops. Those return nan.
    """
    if variance_swap(surface, T) > 2.0 * surface.vix:
        return np.nan
    return brentq(lambda lam: variance_swap(scaled(surface, lam), T) - surface.vix, 0.2, 2.0, xtol=1e-10)


def vix_consistent(panel, T=CALIB_T):
    """One surface per day with its at-the-money level rescaled onto VIX.

    Days where atm_scale has no answer carry the previous day's factor.
    Returns the list of surfaces and the factors as a Series, nan kept where
    the factor was carried.
    """
    raw = [from_row(row) for _, row in panel.iterrows()]
    lam = pd.Series([atm_scale(s, T) for s in raw], index=panel.index)
    carried = lam.ffill().bfill()
    return [scaled(s, l) for s, l in zip(raw, carried)], lam
