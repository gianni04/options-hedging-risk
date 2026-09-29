"""Interactive Brokers bridge: Leg/Position <-> ib_async contracts and orders.

Connections default to paper trading and delayed data. Live ports and market
orders are refused unless explicitly opted into.
"""

import functools
import math
import time
from datetime import date, datetime

import pandas as pd
from ib_async import IB, Bag, ComboLeg, LimitOrder, Option, Stock

PAPER_PORTS = {7497: "TWS paper", 4002: "Gateway paper"}
LIVE_PORTS = {7496: "TWS live", 4001: "Gateway live"}


def connect(host="127.0.0.1", port=7497, client_id=17, allow_live=False, realtime=False):
    """Connect to TWS/Gateway. Refuses live ports unless allow_live is True."""
    if port in LIVE_PORTS and not allow_live:
        raise ValueError(
            f"refusing to connect to {LIVE_PORTS[port]} (port {port}) with allow_live=False"
        )
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    ib.reqMarketDataType(1 if realtime else 3)
    return ib


def to_contract(leg, symbol, exchange="SMART", currency="USD"):
    """Translate a Leg into an ib_async Contract. Raises on a cash leg."""
    if leg.kind in ("call", "put"):
        if leg.expiry is None:
            raise ValueError(f"leg of kind {leg.kind!r} has no expiry")
        return Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=leg.expiry.strftime("%Y%m%d"),
            strike=leg.strike,
            right="C" if leg.kind == "call" else "P",
            exchange=exchange,
            multiplier=str(int(leg.multiplier)),
            currency=currency,
        )
    if leg.kind == "stock":
        return Stock(symbol=symbol, exchange=exchange, currency=currency)
    raise ValueError(f"cannot build an IB contract for leg kind {leg.kind!r}")


def _tradable_legs(position):
    legs = [l for l in position.legs if l.kind != "cash"]
    if not legs:
        raise ValueError("position has no tradable legs")
    return legs


def _ratios_and_actions(legs):
    qtys = [int(round(l.qty)) for l in legs]
    divisor = functools.reduce(math.gcd, (abs(q) for q in qtys))
    if divisor == 0:
        raise ValueError("all leg quantities are zero")
    ratios = [abs(q) // divisor for q in qtys]
    actions = ["BUY" if q > 0 else "SELL" for q in qtys]
    return ratios, actions


def combo(ib, position, symbol, exchange="SMART", currency="USD"):
    """Build a BAG contract from a Position, with minimal integer ratios."""
    legs = _tradable_legs(position)
    ratios, actions = _ratios_and_actions(legs)
    contracts = [to_contract(l, symbol, exchange, currency) for l in legs]
    ib.qualifyContracts(*contracts)
    combo_legs = [
        ComboLeg(conId=c.conId, ratio=r, action=a, exchange=exchange)
        for c, r, a in zip(contracts, ratios, actions)
    ]
    return Bag(symbol=symbol, exchange=exchange, currency=currency, comboLegs=combo_legs)


def place(ib, position, symbol, limit_price, qty=1, dry_run=True):
    """Describe or send a LIMIT order for a combo. dry_run=True touches nothing."""
    legs = _tradable_legs(position)
    ratios, actions = _ratios_and_actions(legs)
    order_legs = [
        {
            "kind": l.kind,
            "strike": l.strike,
            "expiry": l.expiry.isoformat() if l.expiry else None,
            "ratio": r,
            "action": a,
        }
        for l, r, a in zip(legs, ratios, actions)
    ]
    description = {
        "symbol": symbol,
        "order_type": "LMT",
        "action": "BUY",
        "quantity": qty,
        "limit_price": limit_price,
        "legs": order_legs,
        "dry_run": dry_run,
    }
    if dry_run:
        return description
    if ib.client.port not in PAPER_PORTS:
        raise ValueError(
            f"refusing to send a live order: port {ib.client.port} is not a paper port"
        )
    combo_contract = combo(ib, position, symbol)
    order = LimitOrder("BUY", qty, limit_price)
    description["trade"] = ib.placeOrder(combo_contract, order)
    return description


def _spot_price(ib, contract, timeout):
    ticker = ib.reqMktData(contract, "", False, False)
    deadline = time.time() + timeout
    while time.time() < deadline and math.isnan(ticker.marketPrice()):
        ib.sleep(0.2)
    price = ticker.marketPrice()
    ib.cancelMktData(contract)
    if math.isnan(price):
        raise TimeoutError(f"no spot price for {contract.symbol} within {timeout}s")
    return price


def _option_params(ib, symbol, exchange, currency):
    underlying = Stock(symbol, exchange, currency)
    ib.qualifyContracts(underlying)
    params = ib.reqSecDefOptParams(underlying.symbol, "", underlying.secType, underlying.conId)
    if not params:
        raise RuntimeError(f"no option chain found for {symbol}")
    return underlying, next((p for p in params if p.exchange == exchange), params[0])


def expiries(ib, symbol, exchange="SMART", currency="USD"):
    """Sorted list of available option expiries for symbol."""
    _, match = _option_params(ib, symbol, exchange, currency)
    return sorted(datetime.strptime(e, "%Y%m%d").date() for e in match.expirations)


def _ticker_ready(ticker):
    return not math.isnan(ticker.bid) and not math.isnan(ticker.ask) and ticker.modelGreeks is not None


def chain(ib, symbol, expiry: date, n_strikes=20, exchange="SMART", currency="USD", timeout=10.0):
    """Option chain quotes around spot: strikes, bid/ask, iv and delta per side."""
    underlying, match = _option_params(ib, symbol, exchange, currency)
    spot = _spot_price(ib, underlying, timeout)
    strikes = sorted(sorted(match.strikes, key=lambda k: abs(k - spot))[:n_strikes])
    expiry_str = expiry.strftime("%Y%m%d")
    multiplier = str(match.multiplier)
    calls = {
        k: Option(symbol, expiry_str, k, "C", exchange, multiplier, currency) for k in strikes
    }
    puts = {
        k: Option(symbol, expiry_str, k, "P", exchange, multiplier, currency) for k in strikes
    }
    contracts = list(calls.values()) + list(puts.values())
    ib.qualifyContracts(*contracts)
    tickers = {c.conId: ib.reqMktData(c, "106", False, False) for c in contracts}

    deadline = time.time() + timeout
    while time.time() < deadline and not all(_ticker_ready(t) for t in tickers.values()):
        ib.sleep(0.2)

    rows = []
    for k in strikes:
        ct, pt = tickers[calls[k].conId], tickers[puts[k].conId]
        cg, pg = ct.modelGreeks, pt.modelGreeks
        rows.append(
            {
                "strike": k,
                "call_bid": ct.bid,
                "call_ask": ct.ask,
                "call_iv": cg.impliedVol if cg else float("nan"),
                "call_delta": cg.delta if cg else float("nan"),
                "put_bid": pt.bid,
                "put_ask": pt.ask,
                "put_iv": pg.impliedVol if pg else float("nan"),
                "put_delta": pg.delta if pg else float("nan"),
            }
        )
    for c in contracts:
        ib.cancelMktData(c)
    return pd.DataFrame(rows)


def mid(bid, ask):
    """Midpoint of a bid/ask quote."""
    return (bid + ask) / 2.0


def worst(bid, ask, side):
    """The unfavourable side of a quote: ask to buy, bid to sell."""
    side = side.strip().lower()
    if side == "buy":
        return ask
    if side == "sell":
        return bid
    raise ValueError(f"unknown side: {side!r}")


def account_summary(ib):
    """Net liquidation, available margin and buying power."""
    wanted = {
        "NetLiquidation": "net_liquidation",
        "AvailableFunds": "available_margin",
        "BuyingPower": "buying_power",
    }
    out = dict.fromkeys(wanted.values(), None)
    for item in ib.accountSummary():
        if item.tag in wanted:
            out[wanted[item.tag]] = float(item.value)
    return out
