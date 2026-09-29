"""Offline tests for opt.ib. No test opens a real TWS/Gateway connection."""

from datetime import date

import pytest

from opt import ib as ibmod
from opt import legs

EXPIRY = date(2026, 9, 18)


def test_to_contract_call():
    leg = legs.call(100.0, EXPIRY, qty=2)
    c = ibmod.to_contract(leg, "SPY")
    assert c.secType == "OPT"
    assert c.right == "C"
    assert c.strike == 100.0
    assert c.lastTradeDateOrContractMonth == "20260918"
    assert c.multiplier == "100"
    assert c.symbol == "SPY"


def test_to_contract_put():
    leg = legs.put(95.0, EXPIRY)
    c = ibmod.to_contract(leg, "SPY")
    assert c.secType == "OPT"
    assert c.right == "P"


def test_to_contract_stock():
    leg = legs.stock(qty=100)
    c = ibmod.to_contract(leg, "SPY")
    assert c.secType == "STK"
    assert c.symbol == "SPY"


def test_to_contract_cash_raises():
    leg = legs.cash(500.0)
    with pytest.raises(ValueError):
        ibmod.to_contract(leg, "SPY")


def test_connect_refuses_live_port_before_network():
    with pytest.raises(ValueError):
        ibmod.connect(port=7496, allow_live=False)


def test_connect_refuses_live_gateway_port():
    with pytest.raises(ValueError):
        ibmod.connect(port=4001, allow_live=False)


class ExplodingIB:
    def __getattr__(self, name):
        raise AssertionError(f"ib.{name} was accessed during a dry run")


def _iron_condor():
    return legs.Position(
        [
            legs.put(90.0, EXPIRY, qty=1),
            legs.put(95.0, EXPIRY, qty=-1),
            legs.call(105.0, EXPIRY, qty=-1),
            legs.call(110.0, EXPIRY, qty=1),
        ],
        name="iron condor",
    )


def test_place_dry_run_never_touches_ib():
    pos = _iron_condor()
    result = ibmod.place(ExplodingIB(), pos, "SPY", limit_price=1.5, qty=1, dry_run=True)
    assert result["dry_run"] is True
    assert result["order_type"] == "LMT"
    assert result["action"] == "BUY"
    assert result["limit_price"] == 1.5
    assert [l["action"] for l in result["legs"]] == ["BUY", "SELL", "SELL", "BUY"]
    assert [l["ratio"] for l in result["legs"]] == [1, 1, 1, 1]


def test_place_live_denied_without_paper_port():
    class FakeClient:
        port = 7496

    class FakeIB:
        client = FakeClient()

    pos = _iron_condor()
    with pytest.raises(ValueError):
        ibmod.place(FakeIB(), pos, "SPY", limit_price=1.5, qty=1, dry_run=False)


class FakeQualifyIB:
    def qualifyContracts(self, *contracts):
        for i, c in enumerate(contracts):
            c.conId = 1000 + i
        return list(contracts)


def test_combo_ratios_and_actions_iron_condor():
    pos = _iron_condor()
    bag = ibmod.combo(FakeQualifyIB(), pos, "SPY")
    assert bag.secType == "BAG"
    assert [l.ratio for l in bag.comboLegs] == [1, 1, 1, 1]
    assert [l.action for l in bag.comboLegs] == ["BUY", "SELL", "SELL", "BUY"]


def test_combo_ratios_reduced_by_gcd():
    pos = legs.Position(
        [
            legs.call(100.0, EXPIRY, qty=2),
            legs.call(110.0, EXPIRY, qty=-4),
        ]
    )
    bag = ibmod.combo(FakeQualifyIB(), pos, "SPY")
    assert [l.ratio for l in bag.comboLegs] == [1, 2]
    assert [l.action for l in bag.comboLegs] == ["BUY", "SELL"]


def test_worst_buy_takes_ask():
    assert ibmod.worst(bid=1.0, ask=1.2, side="buy") == 1.2


def test_worst_sell_takes_bid():
    assert ibmod.worst(bid=1.0, ask=1.2, side="sell") == 1.0


def test_worst_unknown_side_raises():
    with pytest.raises(ValueError):
        ibmod.worst(1.0, 1.2, "hold")


def test_mid():
    assert ibmod.mid(1.0, 1.2) == pytest.approx(1.1)


class AccountValue:
    def __init__(self, tag, value):
        self.tag = tag
        self.value = value


class FakeAccountIB:
    def accountSummary(self):
        return [
            AccountValue("NetLiquidation", "10000.0"),
            AccountValue("AvailableFunds", "8000.0"),
            AccountValue("BuyingPower", "16000.0"),
            AccountValue("SomeOtherTag", "1.0"),
        ]


def test_account_summary():
    out = ibmod.account_summary(FakeAccountIB())
    assert out == {
        "net_liquidation": 10000.0,
        "available_margin": 8000.0,
        "buying_power": 16000.0,
    }
