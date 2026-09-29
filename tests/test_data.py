"""Tests for opt.data: fetch, cache and the assembled market dataset."""

import os
import socket
import time

import pandas as pd
import pytest

from opt import data


def _network_available():
    try:
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


needs_network = pytest.mark.skipif(not _network_available(), reason="no network connection")


@pytest.fixture(scope="module")
def market():
    return data.market_data()


@needs_network
def test_market_data_columns(market):
    assert list(market.columns) == data.MARKET_DATA_COLUMNS


@needs_network
def test_no_nan_in_critical_columns(market):
    assert not market[data.CRITICAL_COLUMNS].isna().any().any()


@needs_network
def test_vol_indices_are_decimal(market):
    for col in ["vix", "vix9d", "vix3m", "vvix"]:
        assert market[col].dropna().between(0.02, 3.0).all()


@needs_network
def test_skewness_mostly_negative(market):
    assert market["skewness"].median() < 0


@needs_network
def test_rate_within_bounds(market):
    assert market["rate"].between(0.0, 0.20).all()


@needs_network
def test_index_strictly_increasing_no_duplicates(market):
    assert market.index.is_monotonic_increasing
    assert not market.index.duplicated().any()


@needs_network
def test_series_starts_after_vvix_floor(market):
    assert market.index.min() >= pd.Timestamp("2007-01-03")


@needs_network
def test_cache_avoids_refetch(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    data.load("^VIX9D")

    def _boom(*args, **kwargs):
        raise AssertionError("fetch should not run again on a warm cache")

    monkeypatch.setattr(data, "fetch", _boom)
    start = time.perf_counter()
    result = data.load("^VIX9D")
    elapsed = time.perf_counter() - start

    assert not result.empty
    assert elapsed < 1.0


def test_load_falls_back_to_stale_cache_on_fetch_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    idx = pd.date_range("2020-01-01", periods=3, freq="D", name="date")
    cached = pd.DataFrame(
        {"open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0], "low": [1.0, 2.0, 3.0],
         "close": [1.0, 2.0, 3.0], "volume": [0, 0, 0]},
        index=idx,
    )
    path = data._cache_path("^FAKE")
    cached.to_csv(path)
    stale = time.time() - 999 * 3600
    os.utime(path, (stale, stale))

    def _boom(*args, **kwargs):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(data, "fetch", _boom)
    result = data.load("^FAKE", start="2020-01-01")
    assert len(result) == 3


def test_load_raises_when_fetch_and_cache_both_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)

    def _boom(*args, **kwargs):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(data, "fetch", _boom)
    with pytest.raises(RuntimeError):
        data.load("^NOPE")
