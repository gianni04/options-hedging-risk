"""Contract tests for opt/charts.py: files exist, data is right, figures close."""

from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opt import charts, strategies
from opt.legs import Market

ASOF = date(2026, 1, 1)
EXPIRY = ASOF + timedelta(days=90)
SPOT = 100.0


def flat_market(vol=0.20, r=0.03, q=0.0):
    return Market(spot=SPOT, asof=ASOF, r=r, vol=vol, q=q)


def smile_market(r=0.03, q=0.0):
    def vol(strike, T):
        return 0.15 + 0.10 * np.exp(-((strike - SPOT) / 25.0) ** 2)

    return Market(spot=SPOT, asof=ASOF, r=r, vol=vol, q=q)


def _capture_close(monkeypatch):
    """Intercept plt.close to keep a reference to the figure it closes.

    A closed Figure is dropped from pyplot's registry, not destroyed: the
    reference grabbed here still exposes its axes and lines after the call.
    """
    captured = {}
    real_close = plt.close

    def spy(fig=None):
        target = fig if fig is not None else plt.gcf()
        captured["fig"] = target
        real_close(target)

    monkeypatch.setattr(charts.plt, "close", spy)
    return captured


def _png_ok(path):
    assert path.exists()
    assert path.stat().st_size > 1000


def test_payoff_long_call_is_monotone_and_capped_below(tmp_path, monkeypatch):
    captured = _capture_close(monkeypatch)
    pos = strategies.long_call(SPOT, EXPIRY)
    market = flat_market()
    path = tmp_path / "long_call.png"

    charts.payoff(pos, market, path)

    _png_ok(path)
    line = captured["fig"].axes[0].lines[0]
    ydata = np.asarray(line.get_ydata())
    assert np.all(np.diff(ydata) >= -1e-9)
    cost = pos.cost(market)
    assert np.isclose(ydata.min(), -cost, atol=1e-6)
    assert len(plt.get_fignums()) == 0


def test_payoff_iron_condor_is_bounded_and_peaks_between_sold_strikes(tmp_path, monkeypatch):
    captured = _capture_close(monkeypatch)
    put_lo, put_hi, call_lo, call_hi = 70.0, 90.0, 110.0, 130.0
    pos = strategies.iron_condor(put_lo, put_hi, call_lo, call_hi, EXPIRY)
    market = flat_market()
    path = tmp_path / "iron_condor.png"

    charts.payoff(pos, market, path)

    _png_ok(path)
    line = captured["fig"].axes[0].lines[0]
    xdata = np.asarray(line.get_xdata())
    ydata = np.asarray(line.get_ydata())
    assert np.isfinite(ydata).all()
    idx_max = int(np.argmax(ydata))
    assert put_hi <= xdata[idx_max] <= call_lo
    assert len(plt.get_fignums()) == 0


def test_greek_profile_draws_one_line_per_day(tmp_path, monkeypatch):
    captured = _capture_close(monkeypatch)
    pos = strategies.long_call(SPOT, EXPIRY)
    market = flat_market()
    path = tmp_path / "greek_profile.png"

    charts.greek_profile(pos, market, path, greek="delta", days=(0, 7, 30))

    _png_ok(path)
    ax = captured["fig"].axes[0]
    handles, _ = ax.get_legend_handles_labels()
    assert len(handles) == 3
    assert len(plt.get_fignums()) == 0


def test_greek_grid_has_four_axes(tmp_path, monkeypatch):
    captured = _capture_close(monkeypatch)
    pos = strategies.straddle(SPOT, EXPIRY)
    market = flat_market()
    path = tmp_path / "greek_grid.png"

    charts.greek_grid(pos, market, path)

    _png_ok(path)
    assert len(captured["fig"].axes) == 4
    assert len(plt.get_fignums()) == 0


def test_smile_with_constant_vol_does_not_crash(tmp_path, monkeypatch):
    captured = _capture_close(monkeypatch)
    market = flat_market()
    path = tmp_path / "smile_flat.png"

    charts.smile(market, path)

    _png_ok(path)
    assert len(captured["fig"].axes) == 1
    assert len(plt.get_fignums()) == 0


def test_smile_with_callable_vol(tmp_path, monkeypatch):
    captured = _capture_close(monkeypatch)
    market = smile_market()
    path = tmp_path / "smile_surface.png"

    charts.smile(market, path)

    _png_ok(path)
    assert len(plt.get_fignums()) == 0


def test_term_structure(tmp_path, monkeypatch):
    captured = _capture_close(monkeypatch)
    market = smile_market()
    path = tmp_path / "term_structure.png"

    charts.term_structure(market, path)

    _png_ok(path)
    assert len(captured["fig"].axes) == 1
    assert len(plt.get_fignums()) == 0


def test_equity_produces_two_axes_sharing_x(tmp_path, monkeypatch):
    captured = _capture_close(monkeypatch)
    idx = pd.date_range("2026-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {"portfolio": np.cumsum(np.random.default_rng(0).normal(size=50))}, index=idx
    )
    path = tmp_path / "equity.png"

    charts.equity(df, path)

    _png_ok(path)
    axes = captured["fig"].axes
    assert len(axes) == 2
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[1])
    assert len(plt.get_fignums()) == 0


def test_comparison_bars_splits_positive_and_negative_colours(tmp_path, monkeypatch):
    captured = _capture_close(monkeypatch)
    series = pd.Series({"alpha": 12.0, "beta": -5.0, "gamma": 3.5})
    path = tmp_path / "comparison.png"

    charts.comparison_bars(series, path)

    _png_ok(path)
    ax = captured["fig"].axes[0]
    colours = {bar.get_facecolor() for bar in ax.patches}
    assert len(colours) == 2
    assert len(plt.get_fignums()) == 0
