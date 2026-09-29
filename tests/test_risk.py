"""Tests for opt.risk: bounded-loss detection, portfolio limits, kill switch."""

import math
from datetime import date

import pytest

from opt import bs, legs, risk

SPOT = 100.0
ASOF = date(2024, 1, 1)
EXPIRY = date(2024, 4, 1)
VOL = 0.20
R = 0.01


def market():
    return legs.Market(SPOT, ASOF, R, VOL)


def test_short_call_naked_is_unbounded():
    pos = legs.Position([legs.call(110, EXPIRY, qty=-1)], "short call")
    assert risk.max_loss(pos, market()) == float("inf")


def test_short_put_naked_max_loss_equals_strike_minus_premium():
    m = market()
    K = 90.0
    pos = legs.Position([legs.put(K, EXPIRY, qty=-1)], "short put")
    T = m.years_to(EXPIRY)
    premium = float(bs.price(SPOT, K, T, R, 0.0, VOL, "put")) * 100.0
    expected = K * 100.0 - premium
    got = risk.max_loss(pos, m)
    assert math.isfinite(got)
    assert got == pytest.approx(expected, rel=1e-6)


def test_iron_condor_max_loss_equals_wing_width_minus_credit():
    m = market()
    put_long, put_short = 85.0, 90.0
    call_short, call_long = 110.0, 115.0
    pos = legs.Position(
        [
            legs.put(put_long, EXPIRY, qty=1),
            legs.put(put_short, EXPIRY, qty=-1),
            legs.call(call_short, EXPIRY, qty=-1),
            legs.call(call_long, EXPIRY, qty=1),
        ],
        "iron condor",
    )
    credit = -pos.cost(m)
    wing_width = (put_short - put_long) * 100.0
    expected = wing_width - credit
    got = risk.max_loss(pos, m)
    assert math.isfinite(got)
    assert got == pytest.approx(expected, rel=0.01)


def test_check_new_refuses_undefined_risk_by_default():
    m = market()
    strangle = legs.Position(
        [legs.call(115, EXPIRY, qty=-1), legs.put(85, EXPIRY, qty=-1)], "short strangle"
    )
    violations = risk.check_new(strangle, [], m, risk.Limits())
    assert any("undefined" in v.lower() for v in violations)


def test_check_new_allows_undefined_risk_when_flag_set():
    m = market()
    strangle = legs.Position(
        [legs.call(115, EXPIRY, qty=-1), legs.put(85, EXPIRY, qty=-1)], "short strangle"
    )
    limits = risk.Limits(allow_undefined_risk=True, max_net_delta=1e6, max_net_vega=1e6)
    violations = risk.check_new(strangle, [], m, limits)
    assert violations == []


def test_check_new_refuses_on_net_delta_breach_with_existing():
    m = market()
    existing = legs.Position([legs.stock(qty=450)], "existing stock")
    new_pos = legs.Position([legs.stock(qty=100)], "new stock")
    limits = risk.Limits(
        max_net_delta=500.0,
        max_loss_per_trade=1e9,
        max_total_loss=1e9,
        max_net_vega=1e9,
        max_positions=100,
    )
    alone = risk.check_new(new_pos, [], m, limits)
    with_existing = risk.check_new(new_pos, [existing], m, limits)
    assert not any("delta" in v.lower() for v in alone)
    assert any("delta" in v.lower() for v in with_existing)


def test_check_new_refuses_on_net_vega_breach_with_existing():
    m = market()
    existing_pos = legs.Position([legs.call(100, EXPIRY, qty=10)], "existing calls")
    new_pos = legs.Position([legs.call(105, EXPIRY, qty=3)], "new calls")
    existing_vega = existing_pos.greeks(m)["vega"]
    new_alone_vega = new_pos.greeks(m)["vega"]
    threshold = existing_vega + new_alone_vega * 0.5
    assert new_alone_vega < threshold < existing_vega + new_alone_vega

    limits = risk.Limits(
        max_net_vega=threshold,
        max_net_delta=1e9,
        max_loss_per_trade=1e9,
        max_total_loss=1e9,
        max_positions=100,
    )
    alone = risk.check_new(new_pos, [], m, limits)
    with_existing = risk.check_new(new_pos, [existing_pos], m, limits)
    assert not any("vega" in v.lower() for v in alone)
    assert any("vega" in v.lower() for v in with_existing)


def test_check_new_refuses_when_too_many_positions():
    m = market()
    existing = [legs.Position([legs.stock(qty=1)]) for _ in range(10)]
    new_pos = legs.Position([legs.stock(qty=1)])
    limits = risk.Limits(
        max_positions=10,
        max_loss_per_trade=1e9,
        max_total_loss=1e9,
        max_net_delta=1e9,
        max_net_vega=1e9,
    )
    violations = risk.check_new(new_pos, existing, m, limits)
    assert any("position" in v.lower() for v in violations)


def test_portfolio_greeks_sums_positions():
    m = market()
    p1 = legs.Position([legs.call(100, EXPIRY, qty=1)])
    p2 = legs.Position([legs.put(100, EXPIRY, qty=1)])
    combined = risk.portfolio_greeks([p1, p2], m)
    expected_delta = p1.greeks(m)["delta"] + p2.greeks(m)["delta"]
    assert combined["delta"] == pytest.approx(expected_delta)


def test_kill_switch_trips_on_drawdown_from_peak_not_start():
    ks = risk.KillSwitch()
    ks.update(100.0)
    ks.update(150.0)
    tripped = ks.update(119.0)
    assert tripped is True
    assert ks.tripped is True
    assert ks.reason is not None


def test_kill_switch_no_trip_on_small_starting_loss():
    ks = risk.KillSwitch()
    ks.update(100.0)
    tripped = ks.update(95.0)
    assert tripped is False
    assert ks.tripped is False


def test_kill_switch_stays_tripped_until_reset():
    ks = risk.KillSwitch()
    ks.update(100.0)
    ks.update(150.0)
    ks.update(119.0)
    assert ks.tripped is True
    ks.update(200.0)
    assert ks.tripped is True
    ks.reset()
    assert ks.tripped is False


def test_size_for_risk_zero_for_infinite_loss():
    pos = legs.Position([legs.call(110, EXPIRY, qty=-1)], "naked call")
    assert risk.size_for_risk(pos, market(), budget=1000.0) == 0


def test_size_for_risk_vertical_spread_is_coherent():
    m = market()
    pos = legs.Position(
        [legs.call(100, EXPIRY, qty=1), legs.call(110, EXPIRY, qty=-1)], "bull call spread"
    )
    unit_loss = risk.max_loss(pos, m)
    assert math.isfinite(unit_loss)
    assert unit_loss > 0
    assert risk.size_for_risk(pos, m, budget=unit_loss * 3.4) == 3
    assert risk.size_for_risk(pos, m, budget=unit_loss * 0.5) == 0
