"""Catalog of standard option structures built from legs.Position."""

from scipy.optimize import brentq

from . import bs
from .legs import Position, call, put, stock, cash


def long_call(strike, expiry, qty=1):
    """Buy calls."""
    return Position([call(strike, expiry, qty)], "long_call")


def long_put(strike, expiry, qty=1):
    """Buy puts."""
    return Position([put(strike, expiry, qty)], "long_put")


def covered_call(strike, expiry, shares=100, qty=1):
    """Long stock against short calls."""
    return Position([stock(qty=shares), call(strike, expiry, -qty)], "covered_call")


def cash_secured_put(strike, expiry, qty=1):
    """Short put fully collateralised with cash."""
    return Position([put(strike, expiry, -qty), cash(strike * 100.0 * qty)], "cash_secured_put")


def bull_call_spread(lo, hi, expiry, qty=1):
    """Long call at lo, short call at hi."""
    return Position([call(lo, expiry, qty), call(hi, expiry, -qty)], "bull_call_spread")


def bear_put_spread(hi, lo, expiry, qty=1):
    """Long put at hi, short put at lo."""
    return Position([put(hi, expiry, qty), put(lo, expiry, -qty)], "bear_put_spread")


def straddle(strike, expiry, qty=1):
    """Call and put at the same strike."""
    return Position([call(strike, expiry, qty), put(strike, expiry, qty)], "straddle")


def strangle(put_strike, call_strike, expiry, qty=1):
    """Out-of-the-money put and call."""
    return Position([put(put_strike, expiry, qty), call(call_strike, expiry, qty)], "strangle")


def iron_condor(put_lo, put_hi, call_lo, call_hi, expiry, qty=1):
    """Short the body, long the wings."""
    return Position([
        put(put_lo, expiry, qty),
        put(put_hi, expiry, -qty),
        call(call_lo, expiry, -qty),
        call(call_hi, expiry, qty),
    ], "iron_condor")


def butterfly(lo, mid, hi, expiry, qty=1, right="call"):
    """1x long, 2x short, 1x long on the same right."""
    leg_fn = call if right == "call" else put
    return Position([
        leg_fn(lo, expiry, qty),
        leg_fn(mid, expiry, -2 * qty),
        leg_fn(hi, expiry, qty),
    ], "butterfly")


def broken_wing_butterfly(lo, mid, hi, expiry, qty=1, right="put"):
    """Butterfly with unevenly spaced wings."""
    leg_fn = call if right == "call" else put
    return Position([
        leg_fn(lo, expiry, qty),
        leg_fn(mid, expiry, -2 * qty),
        leg_fn(hi, expiry, qty),
    ], "broken_wing_butterfly")


def calendar_spread(strike, near_expiry, far_expiry, qty=1, right="call"):
    """Short the near leg, long the far leg."""
    leg_fn = call if right == "call" else put
    return Position([
        leg_fn(strike, near_expiry, -qty),
        leg_fn(strike, far_expiry, qty),
    ], "calendar_spread")


def diagonal_spread(near_strike, far_strike, near_expiry, far_expiry, qty=1, right="call"):
    """Short near-term leg, long far-term leg at a different strike."""
    leg_fn = call if right == "call" else put
    return Position([
        leg_fn(near_strike, near_expiry, -qty),
        leg_fn(far_strike, far_expiry, qty),
    ], "diagonal_spread")


def risk_reversal(put_strike, call_strike, expiry, qty=1):
    """Short put, long call."""
    return Position([put(put_strike, expiry, -qty), call(call_strike, expiry, qty)], "risk_reversal")


def collar(put_strike, call_strike, expiry, shares=100):
    """Long stock hedged with a bought put and a sold call."""
    qty = shares / 100.0
    return Position([
        stock(qty=shares),
        put(put_strike, expiry, qty),
        call(call_strike, expiry, -qty),
    ], "collar")


def conversion(strike, expiry, shares=100):
    """Long stock, long put, short call at the same strike."""
    qty = shares / 100.0
    return Position([
        stock(qty=shares),
        put(strike, expiry, qty),
        call(strike, expiry, -qty),
    ], "conversion")


def box_spread(lo, hi, expiry, qty=1):
    """Bull call spread combined with a bear put spread."""
    pos = bull_call_spread(lo, hi, expiry, qty) + bear_put_spread(hi, lo, expiry, qty)
    return Position(pos.legs, "box_spread")


CATALOG = {
    "long_call": long_call,
    "long_put": long_put,
    "covered_call": covered_call,
    "cash_secured_put": cash_secured_put,
    "bull_call_spread": bull_call_spread,
    "bear_put_spread": bear_put_spread,
    "straddle": straddle,
    "strangle": strangle,
    "iron_condor": iron_condor,
    "butterfly": butterfly,
    "broken_wing_butterfly": broken_wing_butterfly,
    "calendar_spread": calendar_spread,
    "diagonal_spread": diagonal_spread,
    "risk_reversal": risk_reversal,
    "collar": collar,
    "conversion": conversion,
    "box_spread": box_spread,
}


def strike_from_delta(market, expiry, target_delta, right):
    """Invert Black-Scholes delta for the strike matching an absolute target delta."""
    T = market.years_to(expiry)

    def f(K):
        sigma = market.sigma(K, T)
        d = bs.greeks(market.spot, K, T, market.r, market.q, sigma, right)["delta"]
        return abs(float(d)) - target_delta

    return brentq(f, 0.2 * market.spot, 3.0 * market.spot, xtol=1e-8)


def atm_strike(market, expiry, step=1.0):
    """Strike nearest the forward, rounded to the given step."""
    T = market.years_to(expiry)
    fwd = bs.forward(market.spot, T, market.r, market.q)
    return round(fwd / step) * step
