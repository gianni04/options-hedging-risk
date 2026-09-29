"""Put-call parity checks: implied forward, financing rate, conversions, box spreads."""

import numpy as np
import pandas as pd

_TINY = 1e-12


def implied_forward(call, put, strike, T, r):
    """F = K + (C - P) e^{rT}. Independent of any volatility assumption."""
    return np.asarray(strike, dtype=float) + (
        np.asarray(call, dtype=float) - np.asarray(put, dtype=float)
    ) * np.exp(np.asarray(r, dtype=float) * np.asarray(T, dtype=float))


def implied_rate(call, put, strike, T, spot, q=0.0):
    """The financing rate the option market is quoting, from one strike pair."""
    T = np.asarray(T, dtype=float)
    x = np.asarray(spot, dtype=float) * np.exp(-np.asarray(q, dtype=float) * T) - (
        np.asarray(call, dtype=float) - np.asarray(put, dtype=float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.log(x / np.asarray(strike, dtype=float)) / T
    return np.where((x > 0) & (T > _TINY), out, np.nan)


def implied_carry(call, put, strike, T, spot, r):
    """The dividend yield implied by parity, given a financing rate."""
    T = np.asarray(T, dtype=float)
    x = np.asarray(strike, dtype=float) * np.exp(-np.asarray(r, dtype=float) * T) + (
        np.asarray(call, dtype=float) - np.asarray(put, dtype=float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.log(x / np.asarray(spot, dtype=float)) / T
    return np.where((x > 0) & (T > _TINY), out, np.nan)


def box_value(call_lo, call_hi, put_lo, put_hi):
    """Cost of a long box: it pays (K_hi - K_lo) at expiry whatever the spot does."""
    return (call_lo - call_hi) + (put_hi - put_lo)


def box_rate(call_lo, call_hi, put_lo, put_hi, k_lo, k_hi, T):
    """The zero-coupon rate synthesised by four options."""
    v = box_value(call_lo, call_hi, put_lo, put_hi)
    width = np.asarray(k_hi, dtype=float) - np.asarray(k_lo, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.log(v / width) / T
    return np.where((v > 0) & (width > 0) & (T > _TINY), out, np.nan)


def conversion_edge(call, put, strike, spot, T, r, q=0.0):
    """Profit of long stock + long put + short call against the risk-free rate.

    Positive means the structure earns more than financing, per share.
    """
    T = np.asarray(T, dtype=float)
    paid = np.asarray(spot, dtype=float) + np.asarray(put, dtype=float) - np.asarray(
        call, dtype=float
    )
    received = np.asarray(strike, dtype=float) * np.exp(-np.asarray(r, dtype=float) * T)
    carry = np.asarray(spot, dtype=float) * (
        1.0 - np.exp(-np.asarray(q, dtype=float) * T)
    )
    return received + carry - paid


def parity_scan(strikes, calls, puts, spot, T, r, q=0.0):
    """Per-strike parity diagnostics for one expiry of a real chain.

    The implied forward must be the same at every strike. The spread across
    strikes is the arbitrage signal, and it is measured, not modelled.
    """
    df = pd.DataFrame(
        {
            "strike": np.asarray(strikes, dtype=float),
            "call": np.asarray(calls, dtype=float),
            "put": np.asarray(puts, dtype=float),
        }
    ).sort_values("strike", ignore_index=True)
    df["forward"] = implied_forward(df["call"], df["put"], df["strike"], T, r)
    df["rate"] = implied_rate(df["call"], df["put"], df["strike"], T, spot, q)
    df["edge"] = conversion_edge(df["call"], df["put"], df["strike"], spot, T, r, q)
    ref = df["forward"].median()
    df["forward_dev"] = df["forward"] - ref
    return df


def monotone_violations(strikes, calls):
    """Strikes where call prices rise with strike, which cannot happen."""
    k = np.asarray(strikes, dtype=float)
    c = np.asarray(calls, dtype=float)
    order = np.argsort(k)
    k, c = k[order], c[order]
    bad = np.diff(c) > 0
    return [(float(k[i]), float(k[i + 1])) for i in np.flatnonzero(bad)]


def butterfly_violations(strikes, calls):
    """Strikes where the call price is not convex, which implies a negative density."""
    k = np.asarray(strikes, dtype=float)
    c = np.asarray(calls, dtype=float)
    order = np.argsort(k)
    k, c = k[order], c[order]
    out = []
    for i in range(1, len(k) - 1):
        w = (k[i + 1] - k[i]) / (k[i + 1] - k[i - 1])
        interp = w * c[i - 1] + (1.0 - w) * c[i + 1]
        if c[i] > interp + 1e-10:
            out.append((float(k[i - 1]), float(k[i]), float(k[i + 1]), float(c[i] - interp)))
    return out
