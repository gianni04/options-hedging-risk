"""The surface is only usable if the density it implies is a real density."""

import numpy as np
import pytest

from opt import surface as sf

trapezoid = getattr(np, "trapezoid", None) or np.trapz

SPOT = 400.0
MATURITIES = (9 / 365, 30 / 365, 90 / 365, 182 / 365, 1.0)
STATES = [
    dict(vix9d=0.10, vix=0.12, vix3m=0.15, skewness=-1.5, vvix=0.75, rate=0.01),
    dict(vix9d=0.14, vix=0.16, vix3m=0.18, skewness=-3.0, vvix=0.90, rate=0.04),
    dict(vix9d=0.55, vix=0.60, vix3m=0.45, skewness=-5.0, vvix=1.80, rate=0.001),
    dict(vix9d=0.09, vix=0.09, vix3m=0.12, skewness=-8.0, vvix=0.62, rate=0.053),
]


def test_flat_smile_reproduces_the_lognormal_density_exactly():
    s = sf.Surface(0.20, 0.20, 0.20, beta=0.0, gamma=0.0, r=0.04)
    T = 30 / 365
    K, f = sf.density(s, SPOT, T)
    F = SPOT * np.exp(0.04 * T)
    d2 = (np.log(F / K) - 0.5 * 0.04 * T) / (0.20 * np.sqrt(T))
    exact = np.exp(-0.5 * d2 * d2) / np.sqrt(2 * np.pi) / (K * 0.20 * np.sqrt(T))
    assert np.abs(f - exact).max() < 1e-12
    assert abs(trapezoid(f, K) - 1.0) < 1e-6


def test_atm_vol_reproduces_its_anchors():
    v9, v30, v90 = 0.1398, 0.1561, 0.1819
    for T, want in zip(sf.ANCHORS, (v9, v30, v90)):
        assert abs(float(sf.atm_vol(T, v9, v30, v90)) - want) < 1e-12


def test_total_variance_never_decreases():
    for st in STATES:
        T = np.linspace(1 / 365, 3.0, 900)
        w = sf.atm_vol(T, st["vix9d"], st["vix"], st["vix3m"]) ** 2 * T
        assert np.diff(w).min() > -1e-12


def test_atm_vol_falls_back_when_an_anchor_is_missing():
    assert abs(float(sf.atm_vol(30 / 365, np.nan, 0.20, np.nan)) - 0.20) < 1e-12


def test_saturation_is_smooth_where_a_clip_would_kink():
    z = np.linspace(-8, 8, 20001)
    h2 = (z[1] - z[0]) ** 2
    smooth = np.abs(np.diff(sf.saturate(z), 2)).max() / h2
    kinked = np.abs(np.diff(np.clip(z, -sf.Z_SAT, sf.Z_SAT), 2)).max() / h2
    assert smooth < 1.0
    assert kinked > 100 * smooth


@pytest.mark.parametrize("state", STATES)
def test_calibrated_surface_has_no_butterfly_arbitrage(state):
    s, _ = sf.build(state)
    for T in MATURITIES:
        assert sf.arbitrage_ratio(s, SPOT, T) >= -1e-6, f"arbitrage at {T * 365:.0f} days"


@pytest.mark.parametrize("state", STATES)
def test_density_integrates_to_one(state):
    s, _ = sf.build(state)
    for T in MATURITIES:
        K, f = sf.density(s, SPOT, T)
        assert abs(trapezoid(np.maximum(f, 0.0), K) - 1.0) < 5e-3


def test_a_reachable_skewness_is_reached_exactly():
    beta, achieved, constrained = sf.calibrate_beta(-1.0, 0.04, 0.14, 0.16, 0.18)
    assert not constrained
    assert abs(achieved - (-1.0)) < 1e-3


def test_an_unreachable_skewness_is_reported_not_faked():
    beta, achieved, constrained = sf.calibrate_beta(-8.0, 0.04, 0.14, 0.16, 0.18)
    assert constrained
    assert achieved > -8.0
    assert sf.arbitrage_ratio(sf._probe(beta, 0.04, 0.14, 0.16, 0.18), 100.0, sf.CALIB_T) >= -1e-6


def test_steeper_slope_means_more_negative_skewness():
    skews = [sf.log_moments(sf._probe(b, 0.04, 0.14, 0.16, 0.18), 100.0, sf.CALIB_T)[2]
             for b in (-0.05, -0.10, -0.15, -0.20)]
    assert all(a > b for a, b in zip(skews, skews[1:]))


def test_gamma_from_vvix_is_monotone_and_bounded():
    vals = [sf.gamma_from_vvix(v) for v in (0.4, 0.8, 1.2, 2.5)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))
    assert all(sf.GAMMA_BOUNDS[0] <= v <= sf.GAMMA_BOUNDS[1] for v in vals)


def test_puts_are_dearer_than_calls_at_equal_distance():
    s, _ = sf.build(STATES[1])
    T = 30 / 365
    F = float(s.forward(SPOT, T))
    assert float(s.vol(F * 0.90, T, SPOT)) > float(s.vol(F * 1.10, T, SPOT))


def test_bind_matches_the_unbound_call():
    s, _ = sf.build(STATES[1])
    f = s.bind(SPOT)
    assert abs(f(420.0, 0.25) - float(s.vol(420.0, 0.25, SPOT))) < 1e-12


def test_variance_swap_of_a_flat_smile_is_its_vol():
    flat = sf.Surface(0.2, 0.2, 0.2, -1e-4, 0.0)
    assert sf.variance_swap(flat, 30 / 365) == pytest.approx(0.2, abs=1e-6)
    assert sf.atm_scale(flat) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("state", STATES[:2], ids=["calm", "normal"])
def test_skew_puts_vix_above_the_money_and_rescaling_closes_the_gap(state):
    s, _ = sf.build(state)
    lam = sf.atm_scale(s)
    assert 0.6 < lam < 1.0
    assert sf.variance_swap(sf.scaled(s, lam), sf.CALIB_T) == pytest.approx(state["vix"], rel=1e-8)
    assert sf.worst_arbitrage(sf.scaled(s, lam), SPOT) > -1e-3


def test_a_wing_on_the_box_bound_has_no_variance_swap():
    s = sf.Surface(0.6, 0.6, 0.45, sf.BETA_BOUNDS[0], 0.08)
    assert np.isnan(sf.atm_scale(s))


def test_shift_moves_the_money_and_keeps_the_shape():
    s = sf.Surface(0.15, 0.16, 0.18, -0.3, 0.04)
    T, spot = 30 / 365, SPOT
    sa = float(s.atm(T))
    up = sf.Surface(0.15, 0.16, 0.18, -0.3, 0.04).vol(spot, T, spot, shift=0.02)
    assert float(up) == pytest.approx(sa + 0.02, rel=1e-12)
    K = spot * np.exp(-0.1)
    assert float(s.vol(K, T, spot, shift=0.0)) == pytest.approx(float(s.vol(K, T, spot)))


def test_a_degenerate_day_carries_the_previous_factor():
    import pandas as pd
    normal = dict(vix9d=0.15, vix=0.16, vix3m=0.18, beta=-0.3, gamma=0.04, rate=0.02)
    broken = dict(vix9d=0.60, vix=0.60, vix3m=0.45, beta=sf.BETA_BOUNDS[0], gamma=0.08, rate=0.02)
    panel = pd.DataFrame([normal, broken, normal], index=pd.date_range("2020-03-10", periods=3))
    surfaces, lam = sf.vix_consistent(panel)
    assert lam.isna().tolist() == [False, True, False]
    assert surfaces[1].vix == pytest.approx(0.60 * lam.iloc[0])
    assert sf.variance_swap(surfaces[0], sf.CALIB_T) == pytest.approx(0.16, rel=1e-8)
