"""Pre-trade risk gate: bounded-loss checks, portfolio greeks, and a kill switch."""

import math
from dataclasses import dataclass

import numpy as np

from .legs import GREEKS, Position

_EPS = 1e-9


def max_loss(position, market, lo_mult=0.01, hi_mult=5.0, n=2001):
    """Worst-case loss of `position` at its first expiry, inf if unbounded.

    A naked short call, or short stock not covered by long calls, makes the
    upside slope negative: the loss grows without bound as spot rises. Any
    other combination of calls, puts, stock and cash has a loss bounded by
    the position's value as spot falls to zero, found on a spot grid.
    """
    net_up = sum(l.notional for l in position.legs if l.kind in ("call", "stock"))
    if net_up < -_EPS:
        return float("inf")

    spot0 = market.spot
    grid = np.linspace(lo_mult * spot0, hi_mult * spot0, n)
    strikes = [l.strike for l in position.legs if l.kind in ("call", "put")]
    spots = np.unique(np.concatenate([grid, np.array([0.0, *strikes], dtype=float)]))
    spots = spots[spots >= 0.0]

    pnl = position.payoff(spots, market) - position.cost(market)
    worst = float(np.min(pnl)) if len(pnl) else 0.0
    return max(0.0, -worst)


def portfolio_greeks(positions, market):
    """Sum of greeks across a list of positions."""
    total = dict.fromkeys(GREEKS, 0.0)
    for p in positions:
        for k, v in p.greeks(market).items():
            total[k] += v
    return total


@dataclass
class Limits:
    max_loss_per_trade: float = 2000.0
    max_total_loss: float = 10000.0
    max_net_delta: float = 500.0
    max_net_vega: float = 2000.0
    max_positions: int = 10
    allow_undefined_risk: bool = False


def check_new(position, existing, market, limits):
    """Violations if `position` were added to `existing`. Empty means allowed."""
    violations = []
    combined = Position(list(position.legs) + [l for p in existing for l in p.legs])

    trade_loss = max_loss(position, market)
    total_loss = max_loss(combined, market)

    if not math.isfinite(total_loss):
        if not limits.allow_undefined_risk:
            violations.append("undefined risk not allowed: max_loss is unbounded")
    else:
        if total_loss > limits.max_total_loss:
            violations.append(
                f"total max loss {total_loss:.2f} exceeds max_total_loss "
                f"{limits.max_total_loss:.2f}"
            )

    if math.isfinite(trade_loss) and trade_loss > limits.max_loss_per_trade:
        violations.append(
            f"trade max loss {trade_loss:.2f} exceeds max_loss_per_trade "
            f"{limits.max_loss_per_trade:.2f}"
        )

    greeks = portfolio_greeks([combined], market)
    if abs(greeks["delta"]) > limits.max_net_delta:
        violations.append(
            f"net delta {greeks['delta']:.2f} exceeds max_net_delta "
            f"{limits.max_net_delta:.2f}"
        )
    if abs(greeks["vega"]) > limits.max_net_vega:
        violations.append(
            f"net vega {greeks['vega']:.2f} exceeds max_net_vega "
            f"{limits.max_net_vega:.2f}"
        )

    count = len(existing) + 1
    if count > limits.max_positions:
        violations.append(f"position count {count} exceeds max_positions {limits.max_positions}")

    return violations


class KillSwitch:
    """Trips on drawdown from peak equity or on a single-step loss, and stays tripped."""

    def __init__(self, max_drawdown=0.20, max_daily_loss=0.05):
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self._peak = None
        self._last = None
        self._tripped = False
        self._reason = None

    @property
    def tripped(self):
        return self._tripped

    @property
    def reason(self):
        return self._reason

    def update(self, equity):
        """Feed a new equity mark. Returns whether the switch is tripped."""
        if self._peak is None:
            self._peak = equity
        if self._last is None:
            self._last = equity

        if not self._tripped:
            drawdown = (self._peak - equity) / self._peak if self._peak > 0 else 0.0
            daily = (self._last - equity) / self._last if self._last > 0 else 0.0
            if drawdown > self.max_drawdown:
                self._tripped = True
                self._reason = f"drawdown {drawdown:.2%} exceeds {self.max_drawdown:.2%}"
            elif daily > self.max_daily_loss:
                self._tripped = True
                self._reason = f"daily loss {daily:.2%} exceeds {self.max_daily_loss:.2%}"

        self._peak = max(self._peak, equity)
        self._last = equity
        return self._tripped

    def reset(self):
        """Clear the tripped state. Tracked peak and last equity are kept."""
        self._tripped = False
        self._reason = None


def size_for_risk(position, market, budget):
    """Max integer repeats of `position` keeping max_loss within `budget`."""
    unit_loss = max_loss(position, market)
    if not math.isfinite(unit_loss):
        return 0
    if unit_loss <= 0:
        return int(1e9)
    return max(int(budget // unit_loss), 0)
