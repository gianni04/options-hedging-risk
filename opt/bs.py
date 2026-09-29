"""Black-Scholes-Merton pricing and greeks, vectorised over any argument."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_TINY = 1e-12


def omega(right):
    """+1 for a call, -1 for a put. Accepts strings, numbers or arrays."""
    if isinstance(right, str):
        r = right.strip().lower()
        if r.startswith("c"):
            return 1.0
        if r.startswith("p"):
            return -1.0
        raise ValueError(f"unknown right: {right!r}")
    arr = np.asarray(right)
    if arr.dtype.kind in "US":
        return np.where(np.char.lower(arr.astype(str)).astype("<U1") == "c", 1.0, -1.0)
    return np.asarray(arr, dtype=float)


def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _d1_d2(S, K, T, r, q, sigma):
    v = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return d1, d1 - v


def _degenerate(T, sigma):
    return (np.asarray(T) <= _TINY) | (np.asarray(sigma) <= _TINY)


def price(S, K, T, r, q, sigma, right):
    """Option price. T in years, r/q/sigma annualised and continuous."""
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma)]
    )
    w = omega(right)
    intrinsic = np.maximum(w * (S * np.exp(-q * T) - K * np.exp(-r * T)), 0.0)
    bad = _degenerate(T, sigma)
    safe_T = np.where(bad, 1.0, T)
    safe_s = np.where(bad, 0.2, sigma)
    d1, d2 = _d1_d2(S, K, safe_T, r, q, safe_s)
    out = w * (
        S * np.exp(-q * safe_T) * ndtr(w * d1) - K * np.exp(-r * safe_T) * ndtr(w * d2)
    )
    return np.where(bad, intrinsic, out)


def greeks(S, K, T, r, q, sigma, right):
    """delta, gamma, vega, theta, rho, vanna, volga.

    vega and volga are per 1.00 of vol, theta per year, rho per 1.00 of rate.
    """
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma)]
    )
    w = omega(right)
    bad = _degenerate(T, sigma)
    safe_T = np.where(bad, 1.0, T)
    safe_s = np.where(bad, 0.2, sigma)
    d1, d2 = _d1_d2(S, K, safe_T, r, q, safe_s)
    pdf = _norm_pdf(d1)
    dfq = np.exp(-q * safe_T)
    dfr = np.exp(-r * safe_T)
    sqT = np.sqrt(safe_T)

    delta = w * dfq * ndtr(w * d1)
    gamma = dfq * pdf / (S * safe_s * sqT)
    vega = S * dfq * pdf * sqT
    theta = (
        -S * dfq * pdf * safe_s / (2.0 * sqT)
        - w * r * K * dfr * ndtr(w * d2)
        + w * q * S * dfq * ndtr(w * d1)
    )
    rho = w * K * safe_T * dfr * ndtr(w * d2)
    vanna = -dfq * pdf * d2 / safe_s
    volga = vega * d1 * d2 / safe_s

    expired_delta = w * dfq * (w * (S - K) > 0)
    zero = np.zeros_like(S)
    return {
        "delta": np.where(bad, expired_delta, delta),
        "gamma": np.where(bad, zero, gamma),
        "vega": np.where(bad, zero, vega),
        "theta": np.where(bad, zero, theta),
        "rho": np.where(bad, zero, rho),
        "vanna": np.where(bad, zero, vanna),
        "volga": np.where(bad, zero, volga),
    }


def forward(S, T, r, q):
    return S * np.exp((r - q) * T)


def implied_vol(target, S, K, T, r, q, right, lo=1e-6, hi=5.0):
    """Invert price for sigma. Returns nan outside the no-arbitrage bounds."""
    w = omega(right)
    lower = max(float(w * (S * np.exp(-q * T) - K * np.exp(-r * T))), 0.0)
    upper = float(S * np.exp(-q * T)) if w > 0 else float(K * np.exp(-r * T))
    if not (lower - 1e-10 <= target <= upper + 1e-10) or T <= _TINY:
        return np.nan
    f = lambda s: float(price(S, K, T, r, q, s, right)) - target
    if f(lo) > 0 or f(hi) < 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-10, rtol=1e-12, maxiter=200)
