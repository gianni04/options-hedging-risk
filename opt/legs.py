"""Legs and positions: a strategy is a list of legs, valued generically."""

from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np

from . import bs

GREEKS = ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga")
DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class Market:
    """Everything needed to value a position on one day."""

    spot: float
    asof: date
    r: float
    vol: object = 0.20
    q: float = 0.0

    def sigma(self, strike, T):
        if callable(self.vol):
            return float(self.vol(strike, T))
        return float(self.vol)

    def years_to(self, expiry):
        if expiry is None:
            return 0.0
        return max((expiry - self.asof).days / DAYS_PER_YEAR, 0.0)


@dataclass(frozen=True)
class Leg:
    kind: str
    qty: float
    strike: float = 0.0
    expiry: date | None = None
    multiplier: float = 100.0

    @property
    def notional(self):
        return self.qty * self.multiplier

    def value(self, market):
        T = market.years_to(self.expiry)
        if self.kind == "stock":
            return market.spot * self.notional
        if self.kind == "cash":
            return self.qty * np.exp(-market.r * T)
        sigma = market.sigma(self.strike, T)
        px = bs.price(market.spot, self.strike, T, market.r, market.q, sigma, self.kind)
        return float(px) * self.notional

    def greeks(self, market):
        out = dict.fromkeys(GREEKS, 0.0)
        T = market.years_to(self.expiry)
        if self.kind == "stock":
            out["delta"] = self.notional
            return out
        if self.kind == "cash":
            out["rho"] = -T * self.qty * float(np.exp(-market.r * T))
            return out
        sigma = market.sigma(self.strike, T)
        g = bs.greeks(market.spot, self.strike, T, market.r, market.q, sigma, self.kind)
        for k in GREEKS:
            out[k] = float(g[k]) * self.notional
        return out


def call(strike, expiry, qty=1.0, multiplier=100.0):
    return Leg("call", qty, strike, expiry, multiplier)


def put(strike, expiry, qty=1.0, multiplier=100.0):
    return Leg("put", qty, strike, expiry, multiplier)


def stock(qty=1.0, multiplier=1.0):
    return Leg("stock", qty, 0.0, None, multiplier)


def cash(amount, expiry=None):
    return Leg("cash", amount, 0.0, expiry, 1.0)


@dataclass
class Position:
    legs: list = field(default_factory=list)
    name: str = ""

    def __add__(self, other):
        legs = other.legs if isinstance(other, Position) else list(other)
        return Position(self.legs + legs, self.name)

    def __neg__(self):
        return Position([replace(l, qty=-l.qty) for l in self.legs], f"-{self.name}")

    def __len__(self):
        return len(self.legs)

    def value(self, market):
        return sum(l.value(market) for l in self.legs)

    def greeks(self, market):
        total = dict.fromkeys(GREEKS, 0.0)
        for l in self.legs:
            for k, v in l.greeks(market).items():
                total[k] += v
        return total

    def expiries(self):
        return sorted({l.expiry for l in self.legs if l.expiry is not None})

    def payoff(self, spots, market, at=None):
        """Value across a spot grid at the nearest expiry.

        Legs that outlive that date are marked with the model at the same
        volatility, which is what a calendar spread's diagram actually shows.
        """
        exps = self.expiries()
        at = at or (exps[0] if exps else market.asof)
        out = np.zeros(len(np.atleast_1d(spots)), dtype=float)
        for i, s in enumerate(np.atleast_1d(spots)):
            m = replace(market, spot=float(s), asof=at)
            out[i] = self.value(m)
        return out

    def cost(self, market):
        """Cash paid to open. Positive means a debit."""
        return self.value(market)
