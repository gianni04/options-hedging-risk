"""Underlying and CBOE volatility-surface data: fetch, cache and assemble."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OHLCV = ["open", "high", "low", "close", "volume"]
INDEX_TICKERS = {
    "vix": "^VIX",
    "vix9d": "^VIX9D",
    "vix3m": "^VIX3M",
    "skew": "^SKEW",
    "vvix": "^VVIX",
}
CRITICAL_COLUMNS = ["close", "vix", "rate"]
MARKET_DATA_COLUMNS = [
    "open", "high", "low", "close",
    "vix", "vix9d", "vix3m", "skew", "skewness", "vvix", "rate",
]


CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
CBOE_INDICES = {
    "^VIX": "VIX", "^VIX9D": "VIX9D", "^VIX3M": "VIX3M",
    "^VVIX": "VVIX", "^SKEW": "SKEW",
}


def fetch_cboe(ticker):
    """Volatility index history from CBOE, which computes and publishes it.

    Yahoo mirrors these series and intermittently answers with a single row.
    CBOE is the source, and its VIX3M starts in September 2009 while Yahoo
    silently splices in VXV, the predecessor index, before that date.
    """
    df = pd.read_csv(CBOE_URL.format(CBOE_INDICES[ticker]))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.set_index("date").sort_index()
    if "close" not in df.columns:
        df["close"] = df[df.columns[0]]
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
    df["volume"] = 0
    return df[OHLCV].astype(float)


def fetch(ticker, start="1990-01-01"):
    """Download OHLCV history for ticker: lowercase columns, naive sorted DatetimeIndex.

    Pulls the full history and slices locally. Yahoo answers a dated request on
    ^VIX9D and ^VIX3M with a single row, and answers period="max" on the same
    tickers with the whole series.
    """
    if ticker in CBOE_INDICES:
        df = fetch_cboe(ticker)
        return df[df.index >= pd.Timestamp(start)]
    df = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r}")
    df = df.rename(columns=str.lower)[OHLCV]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "date"
    df = df.sort_index()
    return df[df.index >= pd.Timestamp(start)]


def _cache_path(ticker):
    return DATA_DIR / f"{ticker.lstrip('^')}.csv"


def _read_cache(path):
    return pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()


def load(ticker, start="1990-01-01", max_age_hours=12):
    """Load ticker history from the local CSV cache, refetching when stale or missing.

    Falls back to a stale cache when the network fetch fails, and raises only
    when neither a live fetch nor a cache is available. A fetch that comes back
    shorter than the cache it would replace is discarded: the provider truncates
    a series far more often than a series genuinely loses history.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker)
    age_hours = (time.time() - path.stat().st_mtime) / 3600 if path.exists() else None
    if age_hours is not None and age_hours <= max_age_hours:
        df = _read_cache(path)
    else:
        try:
            df = fetch(ticker, start=start)
            if path.exists() and len(df) < len(_read_cache(path)):
                df = _read_cache(path)
            else:
                df.to_csv(path)
        except Exception as exc:
            if path.exists():
                df = _read_cache(path)
            else:
                raise RuntimeError(
                    f"cannot load {ticker!r}: fetch failed and no cache exists"
                ) from exc
    return df[df.index >= pd.Timestamp(start)]


def market_data(start="2011-01-04", underlying="SPY"):
    """Assemble the daily options-lab dataset: underlying OHLC plus the CBOE vol surface.

    rate is the continuously-compounded equivalent of the ^IRX discount yield:
    r = ln(1 + y/100 * 91/360) / (91/365), for y the quoted 13-week T-bill
    yield in percent. skewness = (100 - skew) / 10 is the risk-neutral 30-day
    skew implied by the CBOE SKEW index. Everything is aligned onto the
    underlying's trading dates; index columns are forward-filled at most 5
    sessions to bridge exchange-calendar gaps, rate is forward-filled without
    limit and clipped to [0, 0.20] since it must never be NaN, and the
    underlying price itself is never forward-filled.
    """
    spot = load(underlying, start=start)[["open", "high", "low", "close"]]
    idx = spot.index

    raw = {name: load(ticker, start=start)["close"] for name, ticker in INDEX_TICKERS.items()}
    vix = raw["vix"].reindex(idx).ffill(limit=5) / 100.0
    vix9d = raw["vix9d"].reindex(idx).ffill(limit=5) / 100.0
    vix3m = raw["vix3m"].reindex(idx).ffill(limit=5) / 100.0
    skew = raw["skew"].reindex(idx).ffill(limit=5)
    vvix = raw["vvix"].reindex(idx).ffill(limit=5) / 100.0
    skewness = (100.0 - skew) / 10.0

    irx = load("^IRX", start=start)["close"].reindex(idx).ffill(limit=5)
    y = irx / 100.0
    rate = np.log(1.0 + y * 91.0 / 360.0) / (91.0 / 365.0)
    rate = rate.ffill().clip(0.0, 0.20)

    out = spot.copy()
    out["vix"] = vix
    out["vix9d"] = vix9d
    out["vix3m"] = vix3m
    out["skew"] = skew
    out["skewness"] = skewness
    out["vvix"] = vvix
    out["rate"] = rate

    out = out.dropna(subset=CRITICAL_COLUMNS)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[MARKET_DATA_COLUMNS]


def latest(underlying="SPY"):
    """Most recent market_data row as a plain dict, keyed by column name plus 'date'."""
    df = market_data(underlying=underlying)
    row = df.iloc[-1].to_dict()
    row["date"] = df.index[-1]
    return row
