"""Property-based tests for opt/strategies.py."""

import inspect
from datetime import date, timedelta

import numpy as np

from opt import bs, strategies
from opt.legs import Market, Position

ASOF = date(2026, 1, 1)
EXPIRY = ASOF + timedelta(days=90)
NEAR_EXPIRY = ASOF + timedelta(days=30)
FAR_EXPIRY = ASOF + timedelta(days=180)
SPOT = 100.0


def flat_market(vol=0.20, r=0.03, q=0.0):
    return Market(spot=SPOT, asof=ASOF, r=r, vol=vol, q=q)


def smile_market(r=0.03, q=0.0):
    def vol(strike, T):
        return 0.15 + 0.10 * np.exp(-((strike - SPOT) / 25.0) ** 2)

    return Market(spot=SPOT, asof=ASOF, r=r, vol=vol, q=q)


def test_box_spread_is_riskless_regardless_of_vol():
    lo, hi, qty = 90.0, 110.0, 2.0
    expected_per_unit = (hi - lo) * 100.0
    for market in (flat_market(0.10), flat_market(0.60), smile_market()):
        pos = strategies.box_spread(lo, hi, EXPIRY, qty)
        T = market.years_to(EXPIRY)
        expected = expected_per_unit * qty * np.exp(-market.r * T)
        assert abs(pos.value(market) - expected) < 1e-8


def test_conversion_is_delta_and_vega_neutral():
    for market in (flat_market(0.10), flat_market(0.50), smile_market()):
        pos = strategies.conversion(100.0, EXPIRY, shares=100)
        g = pos.greeks(market)
        assert abs(g["delta"]) < 1e-8
        assert abs(g["vega"]) < 1e-8


def test_iron_condor_payoff_is_bounded():
    pos = strategies.iron_condor(70.0, 90.0, 110.0, 130.0, EXPIRY, qty=1)
    market = flat_market()
    spots = np.linspace(1.0, 5.0 * SPOT, 400)
    payoff = pos.payoff(spots, market)
    assert np.isfinite(payoff).all()
    assert np.isclose(payoff[0], payoff[3], atol=1e-6)
    assert np.isclose(payoff[-1], payoff[-4], atol=1e-6)
    span = (90.0 - 70.0) * 100.0 + (130.0 - 110.0) * 100.0
    assert payoff.max() - payoff.min() <= span + 1e-6


def test_butterfly_payoff_is_bounded():
    pos = strategies.butterfly(90.0, 100.0, 110.0, EXPIRY, qty=1, right="call")
    market = flat_market()
    spots = np.linspace(1.0, 5.0 * SPOT, 400)
    payoff = pos.payoff(spots, market)
    assert np.isfinite(payoff).all()
    assert np.isclose(payoff[0], payoff[3], atol=1e-6)
    assert np.isclose(payoff[-1], payoff[-4], atol=1e-6)


def test_butterfly_cost_and_max_payoff():
    lo, mid, hi = 90.0, 100.0, 110.0
    pos = strategies.butterfly(lo, mid, hi, EXPIRY, qty=1, right="call")
    market = flat_market()
    assert pos.cost(market) > 0.0
    payoff_at_mid = pos.payoff(np.array([mid]), market)[0]
    assert abs(payoff_at_mid - (mid - lo) * 100.0) < 1e-6


def test_straddle_vega_sign_follows_qty():
    market = flat_market()
    long_vega = strategies.straddle(100.0, EXPIRY, qty=1).greeks(market)["vega"]
    short_vega = strategies.straddle(100.0, EXPIRY, qty=-1).greeks(market)["vega"]
    assert long_vega > 0.0
    assert short_vega < 0.0


def test_calendar_spread_has_positive_vega_under_flat_skew():
    market = flat_market()
    pos = strategies.calendar_spread(SPOT, NEAR_EXPIRY, FAR_EXPIRY, qty=1, right="call")
    assert pos.greeks(market)["vega"] > 0.0


def test_risk_reversal_has_positive_delta():
    market = flat_market()
    pos = strategies.risk_reversal(90.0, 110.0, EXPIRY, qty=1)
    assert pos.greeks(market)["delta"] > 0.0


def test_covered_call_delta_between_zero_and_shares():
    market = flat_market()
    shares = 100
    pos = strategies.covered_call(110.0, EXPIRY, shares=shares, qty=1)
    delta = pos.greeks(market)["delta"]
    assert -1e-9 <= delta <= shares + 1e-9


_PARAM_POOL = {
    "strike": 100.0,
    "expiry": EXPIRY,
    "near_expiry": NEAR_EXPIRY,
    "far_expiry": FAR_EXPIRY,
    "near_strike": 95.0,
    "far_strike": 105.0,
    "lo": 90.0,
    "mid": 100.0,
    "hi": 110.0,
    "put_strike": 90.0,
    "call_strike": 110.0,
    "put_lo": 70.0,
    "put_hi": 90.0,
    "call_lo": 110.0,
    "call_hi": 130.0,
    "shares": 100,
}


def test_catalog_entries_return_nonempty_positions():
    for name, fn in strategies.CATALOG.items():
        params = inspect.signature(fn).parameters
        kwargs = {p: _PARAM_POOL[p] for p in params if p in _PARAM_POOL}
        pos = fn(**kwargs)
        assert isinstance(pos, Position), name
        assert len(pos) > 0, name


def test_strike_from_delta_recovers_target_delta():
    market = flat_market()
    for right, target in (("call", 0.25), ("put", 0.25)):
        strike = strategies.strike_from_delta(market, EXPIRY, target, right)
        T = market.years_to(EXPIRY)
        sigma = market.sigma(strike, T)
        delta = bs.greeks(market.spot, strike, T, market.r, market.q, sigma, right)["delta"]
        assert abs(abs(delta) - target) < 1e-6


def test_atm_strike_is_nearest_forward():
    market = flat_market(r=0.05, q=0.0)
    strike = strategies.atm_strike(market, EXPIRY, step=1.0)
    T = market.years_to(EXPIRY)
    fwd = bs.forward(market.spot, T, market.r, market.q)
    assert abs(strike - fwd) <= 0.5 + 1e-9


def test_the_daily_market_applies_the_atm_scale():
    import pandas as pd
    from opt import sim

    row = pd.Series(dict(vix9d=0.15, vix=0.16, vix3m=0.18, beta=-0.3, gamma=0.04, rate=0.02, close=SPOT))
    T = 30 / 365
    forward = SPOT * np.exp(0.02 * T)
    base = sim._market(row, SPOT, ASOF)
    rescaled = sim._market(pd.concat([row, pd.Series({"atm_scale": 0.8})]), SPOT, ASOF)
    assert np.isclose(rescaled.sigma(forward, T), 0.8 * base.sigma(forward, T), rtol=1e-12)
    assert rescaled.sigma(0.9 * SPOT, T) < base.sigma(0.9 * SPOT, T)
