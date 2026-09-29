"""Tests for opt.vol: estimator accuracy, sampling efficiency and absence of look-ahead."""

import numpy as np
import pandas as pd
import pytest

from opt import vol

SIGMA = 0.20
N_DAYS = 5000
SUB_STEPS = 400
SEED = 7


def simulate_ohlc(n_days=N_DAYS, sigma=SIGMA, steps=SUB_STEPS, seed=SEED, s0=100.0):
    """Geometric random walk of known volatility, aggregated into daily OHLC bars."""
    rng = np.random.default_rng(seed)
    step_sd = sigma * np.sqrt(1.0 / (vol.TRADING_DAYS * steps))
    intraday = np.cumsum(rng.normal(0.0, step_sd, size=(n_days, steps)), axis=1)
    opens = np.log(s0) + np.concatenate([[0.0], np.cumsum(intraday[:, -1])[:-1]])
    paths = opens[:, None] + intraday
    return pd.DataFrame(
        {
            "open": np.exp(opens),
            "high": np.exp(np.maximum(opens, paths.max(axis=1))),
            "low": np.exp(np.minimum(opens, paths.min(axis=1))),
            "close": np.exp(paths[:, -1]),
        },
        index=pd.bdate_range("2000-01-03", periods=n_days),
    )


@pytest.fixture(scope="module")
def ohlc():
    return simulate_ohlc()


@pytest.fixture(scope="module")
def estimates(ohlc):
    return {
        "close_to_close": vol.close_to_close(ohlc["close"]),
        "parkinson": vol.parkinson(ohlc["high"], ohlc["low"]),
        "garman_klass": vol.garman_klass(ohlc),
        "rogers_satchell": vol.rogers_satchell(ohlc),
        "yang_zhang": vol.yang_zhang(ohlc),
    }


@pytest.mark.parametrize(
    "name",
    ["close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang"],
)
def test_estimators_recover_known_sigma(estimates, name):
    series = estimates[name].dropna()
    assert len(series) > N_DAYS - 30
    assert abs(series.mean() - SIGMA) < 0.02


@pytest.mark.parametrize("name", ["parkinson", "yang_zhang"])
def test_range_estimators_are_more_efficient(estimates, name):
    baseline = estimates["close_to_close"].dropna().std()
    candidate = estimates[name].dropna().std()
    assert candidate < 0.7 * baseline


def test_estimators_return_aligned_decimal_series(ohlc, estimates):
    for series in estimates.values():
        assert isinstance(series, pd.Series)
        assert series.index.equals(ohlc.index)
        assert series.dropna().between(0.0, 2.0).all()


@pytest.mark.parametrize("cut", [480, 610, 823])
def test_no_lookahead_in_forecasts(cut):
    close = simulate_ohlc(n_days=900, steps=40, seed=3)["close"]
    for forecast, kwargs in (
        (vol.garch_forecast, {"horizon": 21, "refit_every": 25, "min_obs": 300}),
        (vol.har_forecast, {"horizon": 21, "min_obs": 300}),
    ):
        full = forecast(close, **kwargs).iloc[:cut].dropna()
        truncated = forecast(close.iloc[:cut], **kwargs).dropna()
        assert len(truncated) > 150
        assert full.index.equals(truncated.index)
        assert np.abs(full.to_numpy() - truncated.to_numpy()).max() < 1e-10


def test_forecasts_are_nan_before_min_obs():
    close = simulate_ohlc(n_days=900, steps=40, seed=3)["close"]
    for series in (
        vol.garch_forecast(close, refit_every=25, min_obs=300),
        vol.har_forecast(close, min_obs=300),
    ):
        assert series.iloc[:299].isna().all()
        assert series.iloc[400:].notna().any()


def test_variance_risk_premium_detects_scale():
    index = pd.bdate_range("2020-01-01", periods=300)
    rv = pd.Series(0.1234, index=index)
    points = pd.Series(np.linspace(12.0, 22.0, 300), index=index)
    decimals = points / 100.0
    from_points = vol.variance_risk_premium(points, rv)
    from_decimals = vol.variance_risk_premium(decimals, rv)
    assert np.allclose(from_points.to_numpy(), from_decimals.to_numpy())
    assert np.isclose(from_points.iloc[0], 0.12 - 0.1234)


def test_vrp_zscore_uses_only_past_data():
    index = pd.bdate_range("2015-01-01", periods=800)
    rng = np.random.default_rng(5)
    vrp = pd.Series(rng.normal(0.02, 0.03, 800), index=index)
    full = vol.vrp_zscore(vrp, window=252)
    truncated = vol.vrp_zscore(vrp.iloc[:600], window=252)
    assert np.abs(full.iloc[:600].dropna() - truncated.dropna()).max() < 1e-12
    assert full.iloc[:251].isna().all()


def test_ewma_reacts_faster_than_a_21_day_window():
    calm, shocked, jump = 400, 40, 6.0
    daily = SIGMA / np.sqrt(vol.TRADING_DAYS)
    n = calm + shocked
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    magnitudes = np.concatenate([np.full(calm, daily), np.full(shocked, jump * daily)])
    index = pd.bdate_range("2010-01-04", periods=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(signs * magnitudes)), index=index)
    fast = vol.ewma(close)
    slow = vol.close_to_close(close, window=21)
    assert abs(fast.iloc[calm - 1] - SIGMA) < 0.01
    for lag in range(0, 6):
        assert fast.iloc[calm + lag] > slow.iloc[calm + lag]
    target = 2.5 * SIGMA
    reached_fast = np.argmax(fast.iloc[calm:].to_numpy() > target)
    reached_slow = np.argmax(slow.iloc[calm:].to_numpy() > target)
    assert reached_fast < reached_slow
