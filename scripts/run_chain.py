"""The reconstructed surface against one real SPX chain: level and shape."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from opt import chain as ch
from opt import charts, surface
from run_compare import OUT

MONEYNESS = (0.85, 0.90, 0.95, 0.975, 1.0, 1.025, 1.05)


def main():
    try:
        c, s = ch.fetch()
    except Exception as exc:
        print(f"fetch failed ({exc}); using the latest cached chain")
        c, s = ch.load()
    r = ch.rate(s["irx"])
    term = ch.atm_term(c, s["spot"], s["asof"], r)
    real = ch.atm_at(term, 30)

    state = dict(vix9d=s["vix9d"] / 100, vix=s["vix"] / 100, vix3m=s["vix3m"] / 100,
                 skewness=(100.0 - s["skew"]) / 10.0, vvix=s["vvix"] / 100, rate=r)
    at_vix, _ = surface.build(state)
    lam = surface.atm_scale(at_vix)
    model = surface.scaled(at_vix, lam)
    a_vix, a_model = float(at_vix.atm(30 / 365)), float(model.atm(30 / 365))

    print(f"SPXW chain {s['asof']}, {len(c)} two-sided quotes, spot {s['spot']:.2f}")
    print(f"30-day ATM: chain {real:.2%}, CBOE iv30 {s['iv30'] / 100:.2%}, VIX {s['vix'] / 100:.2%}")
    print(f"surface at the VIX level   {a_vix:.2%}  ({(a_vix - real) * 100:+.2f} pts)")
    print(f"surface rescaled onto VIX  {a_model:.2%}  ({(a_model - real) * 100:+.2f} pts), factor {lam:.3f}")

    n, sm = ch.smile(c, term, s["spot"], s["asof"], r, 30, MONEYNESS)
    T = n / 365.0
    spot_at_forward = term.at[n, "forward"] * np.exp(-r * T)
    sm["rescaled"] = [float(model.vol(k, T, spot_at_forward)) for k in sm["strike"]]
    sm["at_vix"] = [float(at_vix.vol(k, T, spot_at_forward)) for k in sm["strike"]]
    print(f"\n{n}-day smile, implied vol")
    print(sm.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sm["k_over_f"], sm["iv"] * 100, "o-", color=charts.INK, label=f"SPXW quotes {s['asof']}")
    ax.plot(sm["k_over_f"], sm["rescaled"] * 100, color=charts.BLUE, label="surface rescaled onto VIX")
    ax.plot(sm["k_over_f"], sm["at_vix"] * 100, "--", color=charts.ACCENT, label="surface at the VIX level")
    ax.set_xlabel("strike / forward")
    ax.set_ylabel("implied vol, %")
    ax.legend()
    charts.finish(fig, OUT / "figures" / "hedge" / "chain_smile.png", f"{n}-day SPX smile: quotes against the surface",
                  "VIX is a variance swap; read as an ATM vol it prices every strike too high")


if __name__ == "__main__":
    main()
