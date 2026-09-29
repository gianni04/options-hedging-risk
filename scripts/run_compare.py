"""Roll every structure through the same history and rank them."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt import charts, data, sim, surface

OUT = Path(__file__).resolve().parents[1] / "output"
CALIB = OUT / "calibration.csv"


def calibration(df, refresh=False):
    """Calibrate the smile for every day, cached on disk."""
    if CALIB.exists() and not refresh:
        diag = pd.read_csv(CALIB, index_col=0, parse_dates=True)
        if len(diag) == len(df) and diag.index.equals(df.index):
            return diag
    print(f"calibrating {len(df)} days")
    diag = surface.build_all(df)
    OUT.mkdir(parents=True, exist_ok=True)
    diag.to_csv(CALIB)
    return diag


def main():
    df = data.market_data()
    diag = calibration(df, refresh="--refresh" in sys.argv)

    print(f"\ncalibration over {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  bound by no-arbitrage : {diag['constrained'].mean() * 100:.1f}% of days")
    print(f"  skewness asked  median: {diag['skew_target'].median():+.3f}")
    print(f"  skewness reached median: {diag['skew_achieved'].median():+.3f}")
    print(f"  slope beta       median: {diag['beta'].median():+.4f}")

    panel = df.join(diag)
    _, lam = surface.vix_consistent(panel)
    panel["atm_scale"] = lam.ffill().bfill()
    print(f"  ATM factor so the variance swap is VIX, median: {lam.median():.3f}")
    print(f"\nsimulating {len(sim.PLAYBOOK)} structures")
    table = sim.compare(panel)
    table.to_csv(OUT / "comparison.csv")

    show = ["cagr", "vol", "sharpe", "t_stat", "max_drawdown", "worst_day", "win_rate"]
    out = table[show].copy()
    for c in ("cagr", "vol", "max_drawdown", "worst_day", "win_rate"):
        out[c] = (out[c] * 100).round(2)
    out["sharpe"] = out["sharpe"].round(2)
    out["t_stat"] = out["t_stat"].round(2)
    print("\n" + out.to_string())

    charts.comparison_bars(table["sharpe"], OUT / "figures" / "sharpe.png",
                           title="Sharpe by structure", xlabel="Sharpe")
    charts.comparison_bars(table["max_drawdown"] * 100,
                           OUT / "figures" / "drawdown.png",
                           title="Worst drawdown by structure", xlabel="percent")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
