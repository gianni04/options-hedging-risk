import numpy as np
import pandas as pd
import pytest

from opt import hedge, var


def flat_panel(days=560, vol=0.2, seed=0, leverage=0.0):
    S, _ = hedge.gbm(100.0, vol, days / 365, days - 1, 1, seed=seed)
    v = vol * (S[0] / 100.0) ** -leverage
    idx = pd.date_range("2020-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {"close": S[0], "vix9d": v, "vix": v, "vix3m": v, "rate": 0.0, "beta": 0.0, "gamma": 0.0},
        index=idx,
    )


def test_kupiec():
    assert var.kupiec(10, 1000, 0.01) == pytest.approx((0.0, 1.0))
    assert var.kupiec(0, 250, 0.01)[0] == pytest.approx(-500 * np.log(0.99))


def test_christoffersen_sees_clusters_kupiec_cannot():
    spread = np.zeros(1000, bool)
    spread[::100] = True
    clustered = np.zeros(1000, bool)
    clustered[500:510] = True
    assert var.kupiec(spread.sum(), 1000, 0.01) == var.kupiec(clustered.sum(), 1000, 0.01)
    assert var.christoffersen(spread)[1] > 0.5
    assert var.christoffersen(clustered)[1] < 1e-6


def test_flat_constant_vol_world():
    bt = var.backtest(flat_panel(), window=500)
    assert (bt["delta_normal"] == 0).all()
    assert np.allclose(bt["delta_gamma"], bt["delta_gamma_vega"])
    assert np.allclose(bt["realized"], bt["parallel"])
    assert np.allclose(bt["full_parallel"], bt["full_smile"])
    assert (bt["full_parallel"] > 0).all()
    assert not bt["smile_flag"].any()


def test_full_var_is_the_quantile_of_realized_pnl_in_a_flat_world():
    window = 100
    bt = var.backtest(flat_panel(days=400), window=window)
    for k in range(window, len(bt)):
        assert bt["full_parallel"].iloc[k] == pytest.approx(
            -np.quantile(bt["parallel"].iloc[k - window:k], 0.01), rel=1e-9
        )


def test_var_uses_nothing_after_the_previous_close():
    p = flat_panel(days=300)
    base = var.backtest(p, window=200)
    k = 250
    q = p.copy()
    q.iloc[k:, q.columns.get_loc("close")] *= 1.05
    q.iloc[k:, q.columns.get_loc("vix")] *= 1.5
    moved = var.backtest(q, window=200)
    known = base.index <= p.index[k]
    pd.testing.assert_frame_equal(base.loc[known, list(var.METHODS)], moved.loc[known, list(var.METHODS)])


def test_greek_vars_track_full_reprice_when_vol_moves():
    bt = var.backtest(flat_panel(days=400, leverage=5.0), window=200)
    assert np.allclose(bt["delta_gamma_vega"], bt["full_parallel"], rtol=0.15)
    assert (bt["delta_gamma_vega"] > 1.2 * bt["delta_gamma"]).all()
    assert np.allclose(bt["realized"], bt["parallel"])


def test_summary_counts_losses_beyond_var():
    bt = pd.DataFrame({"realized": [-0.02, -0.005, 0.01, -0.011, np.nan], **dict.fromkeys(var.METHODS, 0.01)})
    s = var.summary(bt)
    assert (s["exceptions"] == 2).all()
    assert s["rate"].iloc[0] == pytest.approx(0.5)


def test_smile_revaluation_moves_the_money_by_the_scenario():
    bt = var.backtest(flat_panel(days=400, leverage=5.0), window=200)
    assert np.allclose(bt["full_smile"], bt["full_parallel"], rtol=1e-9)


def test_christoffersen_is_zero_when_transitions_are_independent():
    hits = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    assert var.christoffersen(hits)[0] == pytest.approx(0.0, abs=1e-12)


def test_weekend_theta_is_three_days():
    S, _ = hedge.gbm(100.0, 0.2, 1.6, 419, 1, seed=0)
    idx = pd.bdate_range("2020-01-06", periods=420)
    p = pd.DataFrame({"close": S[0], "vix9d": 0.2, "vix": 0.2, "vix3m": 0.2, "rate": 0.0, "beta": 0.0, "gamma": 0.0},
                     index=idx)
    bt = var.backtest(p, window=200)
    ratio = bt["delta_gamma"] / bt["full_parallel"]
    assert ratio.between(0.9, 1.1).all()
