"""Delta hedging: three predictions checked on simulated paths, then SPY 2011-2026."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from opt import bs, charts, data, hedge, sim, surface
from run_compare import OUT, calibration

S0, K, T, SIG = 100.0, 100.0, 30 / 365, 0.20
FIG = OUT / "figures" / "hedge"


def derman_kamal():
    print("1. hedging error vs rebalancing count, sigma 20%, 30-day ATM call, 20,000 paths")
    print(f"   {'N':>4} {'measured':>9} {'predicted':>9} {'gap':>7}")
    ns, measured, predicted = (5, 10, 21, 50, 100, 250), [], []
    for n in ns:
        S, t = hedge.gbm(S0, SIG, T, n, 20_000, seed=n)
        measured.append(hedge.delta_hedge(S, t, K, 0.0, SIG, SIG)[0].std())
        predicted.append(hedge.derman_kamal(S0, K, T, SIG, n))
        print(f"   {n:>4} {measured[-1]:>9.4f} {predicted[-1]:>9.4f} {measured[-1] / predicted[-1] - 1:>+7.1%}")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.loglog(ns, predicted, color=charts.MUTED, linewidth=1.2, label="sqrt(pi/4) vega sigma / sqrt(N)")
    ax.loglog(ns, measured, "o", color=charts.ACCENT, label="Monte Carlo, 20,000 paths")
    ax.set_xlabel("rebalances to expiry")
    ax.set_ylabel("std of hedged P&L")
    ax.legend()
    charts.finish(fig, FIG / "derman_kamal.png", "Hedging error falls as 1/sqrt(N)",
                  "Four times the trades to halve the error; the formula is leading order, so small N sits below it")


def which_vol():
    print("\n2. sold at 20%, realized 25%, 400 rebalances, 5,000 paths: hedge with which vol?")
    S, t = hedge.gbm(S0, 0.25, T, 400, 5_000, seed=7)
    exact = float(bs.price(S0, K, T, 0, 0, 0.20, "call") - bs.price(S0, K, T, 0, 0, 0.25, "call"))
    for label, h in (("implied 20%", 0.20), ("realized 25%", 0.25)):
        pnl, forecast = hedge.delta_hedge(S, t, K, 0.0, 0.20, h)
        print(f"   hedge at {label:<13} mean {pnl.mean():+.4f}  std {pnl.std():.4f}"
              f"  std of P&L - forecast {(pnl - forecast).std():.4f}")
    print(f"   V(20%) - V(25%) = {exact:+.4f}, Derman-Kamal at 25% and N=400 = "
          f"{hedge.derman_kamal(S0, K, T, 0.25, 400):.4f}")


def leland():
    print("\n3. 10 bp per stock trade (k = 20 bp round trip): sell and hedge at Leland's vol, 20,000 paths")
    cost = 0.001
    for n in (21, 100):
        sl = hedge.leland_vol(SIG, cost, T / n)
        S, t = hedge.gbm(S0, SIG, T, n, 20_000, seed=n + 1)
        naive = hedge.delta_hedge(S, t, K, 0.0, SIG, SIG, cost=cost)[0].mean()
        pnl = hedge.delta_hedge(S, t, K, 0.0, sl, sl, cost=cost)[0]
        first = cost * S0 * float(bs.greeks(S0, K, T, 0, 0, sl, "call")["delta"])
        print(f"   N={n:<4} leland vol {sl:.4f}  mean P&L {pnl.mean():+.4f} +/- {pnl.std() / np.sqrt(pnl.size):.4f}"
              f"  opening trade {-first:+.4f}  sold at 20%: {naive:+.4f}")


def non_overlapping(r):
    """Greedy chain of trades, each opened the day after the previous expired."""
    keep, free = [], r.index[0]
    for d, days in r["days"].items():
        if d >= free:
            keep.append(d)
            free = d + np.timedelta64(int(days) + 1, "D")
    return r.loc[keep]


def spy():
    df = data.market_data()
    panel = df.join(calibration(df))
    surfaces, lam = surface.vix_consistent(panel)
    raw = [surface.from_row(row) for _, row in panel.iterrows()]
    costs = dict(cost=sim.STOCK_SLIPPAGE, vol_slippage=sim.VOL_SLIPPAGE)
    r = hedge.replay(panel, surfaces=surfaces, **costs)
    at_vix = hedge.replay(panel, surfaces=raw, **costs)
    r.to_csv(OUT / "hedge_replay.csv")

    fit = np.corrcoef(r.pnl, r.forecast)[0, 1] ** 2
    var_gap = np.corrcoef(r.pnl, r.iv**2 - r.rv**2)[0, 1] ** 2
    vol_gap = np.corrcoef(r.pnl, r.iv - r.rv)[0, 1] ** 2
    solo = non_overlapping(r)
    offsets = [r.iloc[k::21] for k in range(21)]
    print(f"\n4. SPY: short ATM straddle opened every day {r.index[0].date()} to {r.index[-1].date()},"
          f" 28-30 days, hedged at the close")
    print(f"   ATM rescaled so the smile's variance swap is VIX: factor median {lam.median():.3f},"
          f" VIX - ATM median {(r.vix - r.iv).median() * 100:.1f} vol pts; {lam.isna().sum()} days carried")
    print(f"   trades {len(r)} overlapping, net of costs")
    print(f"   mean P&L {r.pnl.mean() * 1e4:+.0f} bp of strike, win rate {(r.pnl > 0).mean():.1%}")
    print(f"   same trades sold at the VIX level: {at_vix.pnl.mean() * 1e4:+.0f} bp, win rate {(at_vix.pnl > 0).mean():.1%}")
    print(f"   {len(solo)} non-overlapping trades: mean {solo.pnl.mean() * 1e4:+.0f} bp, win rate {(solo.pnl > 0).mean():.1%}")
    print(f"   every 21st trade, 21 offsets: mean {min(o.pnl.mean() for o in offsets) * 1e4:+.0f} to"
          f" {max(o.pnl.mean() for o in offsets) * 1e4:+.0f} bp")
    print(f"   implied {r.iv.mean():.1%} vs realized {r.rv.mean():.1%}")
    print(f"   R2 of P&L on the gamma-theta sum   {fit:.3f}"
          f"  ({min(np.corrcoef(o.pnl, o.forecast)[0, 1] ** 2 for o in offsets):.3f} to"
          f" {max(np.corrcoef(o.pnl, o.forecast)[0, 1] ** 2 for o in offsets):.3f} across offsets)")
    print(f"   R2 of P&L on iv - rv               {vol_gap:.3f}")
    print(f"   R2 of P&L on iv^2 - rv^2           {var_gap:.3f}")
    for d, row in r.nsmallest(3, "pnl").iterrows():
        print(f"   worst {d.date()}  iv {row.iv:.0%}  rv {row.rv:.0%}  P&L {row.pnl:+.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    axes[0].scatter(r.forecast * 100, r.pnl * 100, s=4, alpha=0.35, color=charts.BLUE)
    axes[0].set_xlabel("sum -1/2 Gamma S^2 (r^2 - sigma^2 dt), % of strike")
    axes[0].set_ylabel("realized hedged P&L, % of strike")
    axes[0].set_title(f"gamma-weighted: R2 {fit:.2f}")
    axes[1].scatter((r.iv - r.rv) * 100, r.pnl * 100, s=4, alpha=0.35, color=charts.ACCENT)
    axes[1].set_xlabel("implied - realized vol, points")
    axes[1].set_title(f"unweighted: R2 {vol_gap:.2f}")
    charts.finish(fig, FIG / "spy_forecast.png", "Where the straddle's money came from",
                  f"{len(r)} short SPY straddles, one opened every day, delta-hedged at the close")


if __name__ == "__main__":
    derman_kamal()
    which_vol()
    leland()
    spy()
    print(f"\nwritten to {FIG}")
