"""The decomposition has to answer why the money moved, not just how much."""

from datetime import date, timedelta

import numpy as np
import pytest

from opt import legs
from opt.attribution import TERMS, attribute, attribute_path, plot_attribution, summary

ASOF = date(2024, 1, 2)
EXPIRY = date(2025, 1, 2)
SPOT = 100.0
STRIKE = 100.0
RATE = 0.02
VOL = 0.20


def market(spot=SPOT, asof=ASOF, r=RATE, vol=VOL):
    return legs.Market(spot=spot, asof=asof, r=r, vol=vol)


def straddle(qty=1.0):
    return legs.Position(
        [legs.call(STRIKE, EXPIRY, qty), legs.put(STRIKE, EXPIRY, qty)], "straddle"
    )


def delta_neutral(qty=1.0, m=None):
    m = m or market()
    p = straddle(qty)
    return p + [legs.stock(-p.greeks(m)["delta"])]


def test_small_move_leaves_almost_no_residual():
    m0 = market()
    m1 = market(spot=SPOT * 1.001, asof=ASOF + timedelta(days=1), vol=VOL + 0.001)
    a = attribute(straddle(), m0, m1)
    assert abs(a["residual"]) < 0.01 * abs(a["total"])


def test_short_straddle_makes_money_on_time_alone():
    m0 = market()
    m1 = market(asof=ASOF + timedelta(days=1))
    a = attribute(straddle(-1.0), m0, m1)
    assert a["dS"] == 0.0
    assert a["dsigma"] == 0.0
    assert a["total"] > 0.0
    assert a["theta_pnl"] / a["total"] > 0.99


def test_delta_neutral_straddle_lives_off_gamma():
    m0 = market()
    m1 = market(spot=SPOT * 1.05)
    for qty, sign in ((1.0, 1.0), (-1.0, -1.0)):
        a = attribute(delta_neutral(qty, m0), m0, m1)
        assert a["dt"] == 0.0
        assert a["dsigma"] == 0.0
        assert sign * a["total"] > 0.0
        assert abs(a["gamma_pnl"]) > abs(a["delta_pnl"])
        assert sign * a["gamma_pnl"] > 0.9 * abs(a["total"])


def test_vol_move_shows_up_as_vega():
    m0 = market()
    m1 = market(vol=VOL + 0.02)
    a = attribute(straddle(), m0, m1)
    assert a["dsigma"] == pytest.approx(0.02)
    assert a["vega_pnl"] / a["total"] > 0.95


def test_surface_is_read_at_the_remaining_maturity():
    surface = lambda strike, T: 0.20 + 0.10 * T
    m0 = legs.Market(SPOT, ASOF, RATE, surface)
    m1 = legs.Market(SPOT, ASOF + timedelta(days=180), RATE, surface)
    a = attribute(straddle(), m0, m1)
    dT = m1.years_to(EXPIRY) - m0.years_to(EXPIRY)
    assert a["dsigma"] == pytest.approx(0.10 * dT)


def test_stock_only_position_reports_nan_dsigma():
    m0 = market()
    m1 = market(spot=SPOT + 1.0)
    a = attribute(legs.Position([legs.stock(100.0)]), m0, m1)
    assert np.isnan(a["dsigma"])
    assert a["delta_pnl"] == pytest.approx(100.0)
    assert a["vega_pnl"] == 0.0
    assert abs(a["residual"]) < 1e-9
    assert a["total"] == pytest.approx(100.0)


def path(n=13, seed=11):
    rng = np.random.default_rng(seed)
    spot, vol = SPOT, VOL
    out = []
    for i in range(n):
        spot *= float(np.exp(rng.normal(0.0, 0.015)))
        vol = float(np.clip(vol + rng.normal(0.0, 0.006), 0.06, 0.60))
        rate = RATE + float(rng.normal(0.0, 0.0005))
        out.append(legs.Market(spot, ASOF + timedelta(days=7 * i), rate, vol))
    return out


def test_path_conserves_the_end_to_end_value_change():
    ms = path()
    p = straddle()
    df = attribute_path(p, ms)
    assert len(df) == len(ms) - 1
    assert abs(df["total"].sum() - (p.value(ms[-1]) - p.value(ms[0]))) < 1e-8
    assert df["total_cum"].iloc[-1] == pytest.approx(df["total"].sum())
    for c in TERMS:
        assert df[c + "_cum"].iloc[-1] == pytest.approx(df[c].sum())


def test_summary_shares_add_up_to_one_hundred():
    df = attribute_path(straddle(-1.0), path())
    s = summary(df)
    assert sum(abs(s[c + "_pct"]) for c in TERMS) == pytest.approx(100.0)
    assert sum(s[c] for c in TERMS) == pytest.approx(s["total"])
    assert s["total"] == pytest.approx(df["total"].sum())
    assert s["gross"] >= abs(s["total"])


def test_plot_writes_a_png(tmp_path):
    out = tmp_path / "attribution.png"
    plot_attribution(attribute_path(straddle(), path()), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_path_needs_two_markets():
    with pytest.raises(ValueError):
        attribute_path(straddle(), [market()])
