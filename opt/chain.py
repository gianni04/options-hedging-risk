"""One real SPX option chain, to hold the reconstructed surface against quotes.

CBOE publishes its full chain with a delay, free, one day at a time. Every
snapshot fetched is cached, so a check run once stays reproducible offline.
"""

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import bs
from .parity import implied_forward

CHAIN_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_{}.json"
QUOTE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{}.json"
CHAIN_DIR = Path(__file__).resolve().parent.parent / "data" / "chains"
STATE = {"vix": "VIX", "vix9d": "VIX9D", "vix3m": "VIX3M", "skew": "SKEW", "vvix": "VVIX", "irx": "IRX"}


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)["data"]


def parse(data, root="SPXW"):
    """Two-sided quotes of one option root: expiry, right, strike, bid, ask, mid."""
    o = pd.DataFrame(data["options"])
    parts = o["option"].str.extract(rf"^{root}(\d{{6}})([CP])(\d{{8}})$").dropna()
    o = o.loc[parts.index]
    out = pd.DataFrame({
        "expiry": pd.to_datetime(parts[0], format="%y%m%d"),
        "right": parts[1].map({"C": "call", "P": "put"}),
        "strike": parts[2].astype(float) / 1000.0,
        "bid": o["bid"].astype(float),
        "ask": o["ask"].astype(float),
    })
    out = out[(out["bid"] > 0) & (out["ask"] >= out["bid"])]
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    return out.reset_index(drop=True)


def fetch(symbol="SPX", root="SPXW"):
    """Today's delayed chain and the index closes that rebuild the surface, cached by date."""
    data = _get(CHAIN_URL.format(symbol))
    asof = str(data["last_trade_time"])[:10]
    state = {k: float(_get(QUOTE_URL.format(v))["close"]) for k, v in STATE.items()}
    state["spot"], state["asof"], state["iv30"] = float(data["close"]), asof, float(data.get("iv30", np.nan))
    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    parse(data, root).to_csv(CHAIN_DIR / f"{root}_{asof}.csv", index=False)
    (CHAIN_DIR / f"{root}_{asof}.json").write_text(json.dumps(state, indent=1))
    return load(asof, root)


def load(asof=None, root="SPXW"):
    """A cached snapshot, the latest when asof is None: (chain, state)."""
    if asof is None:
        asof = max(p.stem.split("_")[1] for p in CHAIN_DIR.glob(f"{root}_*.csv"))
    chain = pd.read_csv(CHAIN_DIR / f"{root}_{asof}.csv", parse_dates=["expiry"])
    return chain, json.loads((CHAIN_DIR / f"{root}_{asof}.json").read_text())


def rate(irx):
    """CBOE quotes ^IRX in tenths of a percent; same conversion as data.market_data."""
    y = irx / 1000.0
    return float(np.log(1.0 + y * 91.0 / 360.0) / (91.0 / 365.0))


def atm_term(chain, spot, asof, r, days=(7, 90)):
    """Forward from put-call parity and ATM vol per expiry, interpolated to the forward."""
    rows = []
    for expiry, g in chain.groupby("expiry"):
        n = (expiry - pd.Timestamp(asof)).days
        if not days[0] <= n <= days[1]:
            continue
        T = n / 365.0
        c = g[g["right"] == "call"].set_index("strike")["mid"]
        p = g[g["right"] == "put"].set_index("strike")["mid"]
        k = c.index.intersection(p.index).sort_values()
        k0 = (c[k] - p[k]).abs().idxmin()
        F = float(implied_forward(c[k0], p[k0], k0, T, r))
        q = r - np.log(F / spot) / T
        lo, hi = k[k <= F].max(), k[k > F].min()
        iv = [bs.implied_vol(float(c[x]), spot, x, T, r, q, "call") for x in (lo, hi)]
        rows.append((n, F, q, float(np.interp(F, [lo, hi], iv))))
    return pd.DataFrame(rows, columns=["days", "forward", "q", "atm"]).set_index("days")


def atm_at(term, days):
    """ATM vol at a calendar-day maturity, interpolated in total variance."""
    w = np.interp(days, term.index, term["atm"] ** 2 * term.index)
    return float(np.sqrt(w / days))


def smile(chain, term, spot, asof, r, days, moneyness):
    """Out-of-the-money implied vols on the expiry nearest `days`, at the listed strike nearest each K/F."""
    n = int(term.index[np.abs(term.index - days).argmin()])
    expiry = pd.Timestamp(asof) + pd.Timedelta(days=n)
    g = chain[chain["expiry"] == expiry]
    F, q, T = term.at[n, "forward"], term.at[n, "q"], n / 365.0
    rows = []
    for m in moneyness:
        right = "put" if m < 1.0 else "call"
        side = g[g["right"] == right]
        row = side.iloc[(side["strike"] - m * F).abs().argmin()]
        iv = bs.implied_vol(float(row["mid"]), spot, float(row["strike"]), T, r, q, right)
        rows.append((float(row["strike"]), float(row["strike"]) / F, iv))
    return n, pd.DataFrame(rows, columns=["strike", "k_over_f", "iv"])
