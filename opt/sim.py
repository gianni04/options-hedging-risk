"""Replay a strategy day by day on the rebuilt surface.

Chooses strikes by delta, opens at 45 days and rolls at 7, and charges costs.
Slippage is charged in vol points through vega (0.5 vol pt per leg) rather
than as a % of premium, which would overcharge cheap OTM options.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from . import strategies as st
from .legs import Market
from .surface import from_row, scaled

TRADING_DAYS = 252
VOL_SLIPPAGE = 0.005
COMMISSION = 0.65
STOCK_SLIPPAGE = 1e-4


def _expiry(asof, dte):
    return asof + timedelta(days=int(dte))


def _atm(market, expiry):
    return st.atm_strike(market, expiry, step=1.0)


def _k(market, expiry, delta, right):
    return round(st.strike_from_delta(market, expiry, delta, right))


PLAYBOOK = {
    "long_call_25d": lambda m, e: st.long_call(_k(m, e, 0.25, "call"), e),
    "long_put_25d": lambda m, e: st.long_put(_k(m, e, 0.25, "put"), e),
    "covered_call_25d": lambda m, e: st.covered_call(_k(m, e, 0.25, "call"), e),
    "cash_secured_put_25d": lambda m, e: st.cash_secured_put(_k(m, e, 0.25, "put"), e),
    "bull_call_spread": lambda m, e: st.bull_call_spread(
        _atm(m, e), _k(m, e, 0.25, "call"), e
    ),
    "bear_put_spread": lambda m, e: st.bear_put_spread(
        _atm(m, e), _k(m, e, 0.25, "put"), e
    ),
    "long_straddle": lambda m, e: st.straddle(_atm(m, e), e, qty=1),
    "short_straddle": lambda m, e: st.straddle(_atm(m, e), e, qty=-1),
    "long_strangle_25d": lambda m, e: st.strangle(
        _k(m, e, 0.25, "put"), _k(m, e, 0.25, "call"), e, qty=1
    ),
    "short_strangle_16d": lambda m, e: st.strangle(
        _k(m, e, 0.16, "put"), _k(m, e, 0.16, "call"), e, qty=-1
    ),
    "iron_condor_16d": lambda m, e: st.iron_condor(
        _k(m, e, 0.08, "put"), _k(m, e, 0.16, "put"),
        _k(m, e, 0.16, "call"), _k(m, e, 0.08, "call"), e,
    ),
    "butterfly_atm": lambda m, e: st.butterfly(
        _k(m, e, 0.25, "put"), _atm(m, e), _k(m, e, 0.25, "call"), e
    ),
    "risk_reversal_25d": lambda m, e: st.risk_reversal(
        _k(m, e, 0.25, "put"), _k(m, e, 0.25, "call"), e
    ),
    "collar_25d": lambda m, e: st.collar(
        _k(m, e, 0.25, "put"), _k(m, e, 0.25, "call"), e
    ),
    "conversion": lambda m, e: st.conversion(_atm(m, e), e),
    "box_spread": lambda m, e: st.box_spread(
        _k(m, e, 0.25, "put"), _k(m, e, 0.25, "call"), e
    ),
}


def trade_cost(position, market, vol_slippage=VOL_SLIPPAGE, commission=COMMISSION):
    """Slippage in vol points priced through vega, plus commission per contract."""
    total = 0.0
    for leg in position.legs:
        if leg.kind in ("call", "put"):
            total += abs(leg.greeks(market)["vega"]) * vol_slippage
            total += commission * abs(leg.qty)
        elif leg.kind == "stock":
            total += abs(leg.notional) * market.spot * STOCK_SLIPPAGE
    return total


def _market(row, spot, asof):
    """The day's market; an atm_scale column rescales the surface onto VIX."""
    s = from_row(row)
    if "atm_scale" in row:
        s = scaled(s, float(row["atm_scale"]))
    return Market(spot=spot, asof=asof, r=float(row["rate"]), vol=s.bind(spot))


def simulate(panel, name, dte=45, roll_dte=7, capital=100_000.0,
             vol_slippage=VOL_SLIPPAGE, commission=COMMISSION):
    """Roll one structure through the whole history. Returns a daily frame."""
    build = PLAYBOOK[name]
    rows = []
    pos, expiry, cash = None, None, capital

    for ts, row in panel.iterrows():
        asof = ts.date()
        spot = float(row["close"])
        m = _market(row, spot, asof)

        if pos is not None and (expiry - asof).days <= roll_dte:
            cash += pos.value(m) - trade_cost(pos, m, vol_slippage, commission)
            pos, expiry = None, None

        if pos is None:
            expiry = _expiry(asof, dte)
            try:
                pos = build(m, expiry)
            except Exception:
                pos, expiry = None, None
            if pos is not None:
                cash -= pos.value(m) + trade_cost(pos, m, vol_slippage, commission)

        value = pos.value(m) if pos is not None else 0.0
        g = pos.greeks(m) if pos is not None else dict.fromkeys(
            ("delta", "gamma", "vega", "theta"), 0.0
        )
        rows.append({
            "equity": cash + value,
            "position_value": value,
            "cash": cash,
            "delta": g["delta"], "gamma": g["gamma"],
            "vega": g["vega"], "theta": g["theta"],
            "open": pos is not None,
        })

    out = pd.DataFrame(rows, index=panel.index)
    out["pnl"] = out["equity"].diff().fillna(0.0)
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    return out


def metrics(sim, capital=100_000.0):
    """Return, risk, and the shape of the loss tail."""
    eq = sim["equity"]
    pnl = sim["pnl"]
    years = len(eq) / TRADING_DAYS
    daily = pnl / capital
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    mean = daily.mean() * TRADING_DAYS
    n = len(daily)
    final = eq.iloc[-1] / capital
    return {
        "total_return": final - 1.0,
        "cagr": final ** (1 / years) - 1.0 if years > 0 and final > 0 else np.nan,
        "vol": vol,
        "sharpe": mean / vol if vol > 0 else np.nan,
        "t_stat": daily.mean() / daily.std() * np.sqrt(n) if daily.std() > 0 else np.nan,
        "max_drawdown": sim["drawdown"].min(),
        "worst_day": daily.min(),
        "best_day": daily.max(),
        "skew": daily.skew(),
        "kurtosis": daily.kurtosis(),
        "win_rate": (pnl > 0).mean(),
        "days": n,
    }


def compare(panel, names=None, **kwargs):
    """Run every structure over the same history and rank them."""
    names = names or list(PLAYBOOK)
    capital = kwargs.get("capital", 100_000.0)
    rows = {}
    for name in names:
        print(f"  {name}", flush=True)
        rows[name] = metrics(simulate(panel, name, **kwargs), capital)
    return pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)
