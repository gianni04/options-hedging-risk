"""Payoff and greek charts for every structure."""

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt import charts, data, risk, sim, surface
from opt.legs import Market

OUT = Path(__file__).resolve().parents[1] / "output" / "figures" / "lab"
DTE = 45


def today_market():
    """The market as it stands on the last day of the panel."""
    df = data.market_data()
    row = df.iloc[-1]
    s, diag = surface.build(row)
    s = surface.scaled(s, surface.atm_scale(s))
    spot = float(row["close"])
    m = Market(spot=spot, asof=df.index[-1].date(), r=float(row["rate"]),
               vol=s.bind(spot))
    return m, s, diag, df.index[-1].date()


def main():
    m, s, diag, asof = today_market()
    expiry = asof + timedelta(days=DTE)
    print(f"{asof}  spot {m.spot:.2f}  r {m.r * 100:.2f}%  "
          f"ATM {DTE}d {float(s.atm(DTE / 365)) * 100:.2f}%")
    print(f"smile slope {diag['beta']:+.4f}  curvature {diag['gamma']:.4f}  "
          f"skewness asked {diag['skew_target']:+.2f} reached {diag['skew_achieved']:+.2f}")

    charts.smile(m, OUT / "00_smile.png")
    charts.term_structure(m, OUT / "00_term_structure.png")

    rows = {}
    for name, build in sim.PLAYBOOK.items():
        pos = build(m, expiry)
        g = pos.greeks(m)
        loss = risk.max_loss(pos, m)
        rows[name] = {
            "legs": len(pos),
            "cost": pos.cost(m),
            "max_loss": loss,
            "delta": g["delta"], "gamma": g["gamma"],
            "vega": g["vega"], "theta": g["theta"] / 365.0,
        }
        charts.payoff(pos, m, OUT / f"{name}_payoff.png", title=name)
        charts.greek_grid(pos, m, OUT / f"{name}_greeks.png")

    table = pd.DataFrame(rows).T
    print(f"\n{DTE} days to expiry, one unit, theta per day\n")
    print(table.round(2).to_string())
    table.to_csv(OUT.parents[1] / "structures.csv")
    print(f"\n{2 * len(rows) + 2} figures written to {OUT}")


if __name__ == "__main__":
    main()
