"""One-day 99% VaR of a delta-hedged short straddle, five methods, backtested."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from opt import charts, data, var
from run_compare import OUT, calibration


def show(bt, actual, title):
    table = var.summary(bt, actual)
    table["mean_var"] = (table["mean_var"] * 1e4).round(0)
    table["rate"] = (table["rate"] * 100).round(2)
    table[["kupiec_p", "independence_p"]] = table[["kupiec_p", "independence_p"]].round(3)
    print(f"\n{title} ({len(bt)} days; mean_var in bp of spot, rate in %)")
    print(table.to_string())


def main():
    df = data.market_data()
    bt = var.backtest(df.join(calibration(df)))
    bt.to_csv(OUT / "var_backtest.csv")

    flagged = bt[bt["smile_flag"]]
    print(f"{len(bt)} days, {bt.index[0].date()} to {bt.index[-1].date()}, 99% one-day, 500-day window")
    show(bt, "realized", "realized P&L, repriced on the next day's smile")
    show(bt[~bt["smile_flag"]], "realized", f"same, without the {len(flagged)} days touching a degenerate smile")
    show(bt, "parallel", "counterfactual: strike vol moved by the change in ATM vol")
    print(f"\ncorrelation of realized and parallel P&L: {bt[list(var.ACTUALS)].corr().iloc[0, 1]:.2f}")
    print("worst realized days:")
    for d, row in bt.nsmallest(5, "realized").iterrows():
        print(f"  {d.date()}  realized {row.realized:+.2%}  parallel {row.parallel:+.2%}"
              f"  full_smile VaR {row.full_smile:.2%}{'  degenerate smile' if row.smile_flag else ''}")

    w = bt.loc["2020-02-15":"2020-04-30"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    clean = w[~w["smile_flag"]]
    ax.bar(clean.index, clean["realized"] * 100, color=charts.MUTED, width=0.8, label="realized P&L")
    ax.bar(w.index[w["smile_flag"]], w.loc[w["smile_flag"], "realized"] * 100, color=charts.GRID,
           edgecolor=charts.MUTED, width=0.8, label="realized, degenerate smile")
    for m, colour in zip(("delta_gamma", "delta_gamma_vega", "full_smile"), (charts.ACCENT, charts.GREEN, charts.BLUE)):
        ax.plot(w.index, -w[m] * 100, color=colour, linewidth=1.4, label=f"-VaR {m.replace('_', '-')}")
    ax.axhline(0, color=charts.INK, linewidth=0.8)
    ax.set_ylabel("% of spot")
    ax.legend(loc="lower left")
    charts.finish(fig, OUT / "figures" / "hedge" / "var_2020.png", "Delta-hedged is not riskless",
                  "Short 30-day SPY straddle, delta-hedged: delta-normal VaR is zero every day")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    main()
