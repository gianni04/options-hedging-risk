import numpy as np
import pandas as pd
import pytest

from opt import bs, hedge

S0, K, T = 100.0, 100.0, 30 / 365


def flat_panel(days, vol=0.2, freq="D"):
    idx = pd.date_range("2020-01-06", periods=days, freq=freq)
    return pd.DataFrame(
        {"close": 100.0, "vix9d": vol, "vix": vol, "vix3m": vol, "rate": 0.0, "beta": 0.0, "gamma": 0.0},
        index=idx,
    )


def straddle(S, K, T, sigma, r=0.0):
    return float(sum(bs.price(S, K, T, r, 0.0, sigma, leg) for leg in hedge.STRADDLE))


def test_hedging_error_matches_derman_kamal():
    S, t = hedge.gbm(S0, 0.2, T, 100, 20_000, seed=1)
    pnl, _ = hedge.delta_hedge(S, t, K, 0.0, 0.2, 0.2)
    assert abs(pnl.mean()) < 0.01
    assert pnl.std() == pytest.approx(hedge.derman_kamal(S0, K, T, 0.2, 100), rel=0.05)


def test_error_halves_when_rebalancing_four_times_more():
    stds = []
    for n in (25, 100):
        S, t = hedge.gbm(S0, 0.2, T, n, 20_000, seed=2)
        stds.append(hedge.delta_hedge(S, t, K, 0.0, 0.2, 0.2)[0].std())
    assert stds[0] / stds[1] == pytest.approx(2.0, rel=0.05)


def test_cash_earns_the_rate():
    S, t = hedge.gbm(S0, 0.2, T, 200, 20_000, r=0.05, seed=6)
    pnl, _ = hedge.delta_hedge(S, t, K, 0.05, 0.2, 0.2)
    assert abs(pnl.mean()) < 0.01


def test_hedging_at_realized_vol_locks_the_price_difference():
    S, t = hedge.gbm(S0, 0.25, T, 400, 5_000, seed=3)
    exact = float(bs.price(S0, K, T, 0.0, 0.0, 0.20, "call") - bs.price(S0, K, T, 0.0, 0.0, 0.25, "call"))
    at_implied, _ = hedge.delta_hedge(S, t, K, 0.0, 0.20, 0.20)
    at_realized, _ = hedge.delta_hedge(S, t, K, 0.0, 0.20, 0.25)
    assert at_realized.mean() == pytest.approx(exact, abs=0.01)
    assert at_realized.std() == pytest.approx(hedge.derman_kamal(S0, K, T, 0.25, 400), rel=0.05)
    assert at_implied.std() > 2 * at_realized.std()


def test_leland_vol_pays_for_rebalancing_but_not_the_first_trade():
    cost, n = 0.001, 100
    sl = hedge.leland_vol(0.2, cost, T / n)
    S, t = hedge.gbm(S0, 0.2, T, n, 20_000, seed=4)
    pnl, _ = hedge.delta_hedge(S, t, K, 0.0, sl, sl, cost=cost)
    first_trade = cost * S0 * float(bs.greeks(S0, K, T, 0.0, 0.0, sl, "call")["delta"])
    assert pnl.mean() == pytest.approx(-first_trade, abs=0.01)


@pytest.mark.parametrize("hedge_vol", [0.20, 0.25])
def test_forecast_explains_the_pnl_whichever_vol_hedges(hedge_vol):
    S, t = hedge.gbm(S0, 0.25, T, 200, 5_000, seed=5)
    pnl, forecast = hedge.delta_hedge(S, t, K, 0.0, 0.20, hedge_vol, "put")
    assert np.corrcoef(pnl, forecast)[0, 1] > 0.95
    assert (pnl - forecast).std() < 0.05
    assert abs((pnl - forecast).mean()) < 0.01
    assert pnl.mean() < 0


def test_a_straddle_is_its_legs_but_trades_once():
    S, t = hedge.gbm(S0, 0.2, T, 50, 2_000, seed=8)
    legs = sum(hedge.delta_hedge(S, t, K, 0.0, 0.2, 0.2, leg)[0] for leg in hedge.STRADDLE)
    both = hedge.delta_hedge(S, t, K, 0.0, 0.2, 0.2, hedge.STRADDLE)[0]
    np.testing.assert_allclose(both, legs, atol=1e-10)
    cost = 0.001
    legs = sum(hedge.delta_hedge(S, t, K, 0.0, 0.2, 0.2, leg, cost)[0] for leg in hedge.STRADDLE)
    both = hedge.delta_hedge(S, t, K, 0.0, 0.2, 0.2, hedge.STRADDLE, cost)[0]
    call_delta = float(bs.greeks(S0, K, T, 0.0, 0.0, 0.2, "call")["delta"])
    np.testing.assert_allclose(both - legs, cost * S0 * (1 - abs(2 * call_delta - 1)), atol=1e-10)


def test_replay_on_a_frozen_spot_keeps_the_premium():
    p = flat_panel(70)
    r = hedge.replay(p, dte=30)
    assert (r["days"] == 30).all()
    assert np.allclose(r["iv"], 0.2, atol=1e-5)
    premium = [straddle(100, 100, 30 / 365, v) / 100 for v in r["iv"]]
    np.testing.assert_allclose(r["pnl"], premium, rtol=1e-12)
    assert (r["rv"] == 0).all()
    slip = hedge.replay(p, dte=30, vol_slippage=0.01)
    vega = bs.greeks(100, 100, 30 / 365, 0, 0, r["iv"].to_numpy(), "call")["vega"]
    np.testing.assert_allclose(r["pnl"] - slip["pnl"], 2 * vega * 0.01 / 100, rtol=1e-12)


def test_replay_expires_on_the_last_trading_day():
    p = flat_panel(80, freq="B")
    r = hedge.replay(p, dte=30)
    assert r["days"].max() == 30 and r["days"].min() == 28
    expiry = r.index + pd.to_timedelta(r["days"], unit="D")
    assert expiry.isin(p.index).all()
    premium = [straddle(100, 100, d / 365, v) / 100 for d, v in zip(r["days"], r["iv"])]
    np.testing.assert_allclose(r["pnl"], premium, rtol=1e-12)


def test_replay_realized_vol_is_annualised_on_the_calendar():
    p = flat_panel(90)
    p["close"] = 100.0 * np.exp(0.2 / np.sqrt(365) * (np.arange(90) % 2))
    r = hedge.replay(p, dte=30)
    assert np.allclose(r["rv"], 0.2)
    assert np.corrcoef(r["pnl"], r["forecast"])[0, 1] > 0.9
