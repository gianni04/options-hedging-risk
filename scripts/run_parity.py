"""Recover the financing rate from option prices with put-call parity (no vol model)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt import bs, parity

SPOT, T, RATE, CARRY, VOL = 764.11, 0.25, 0.0370, 0.0125, 0.1780


def main():
    strikes = np.arange(SPOT * 0.85, SPOT * 1.16, SPOT * 0.025)
    calls = np.array([float(bs.price(SPOT, k, T, RATE, CARRY, VOL, "call")) for k in strikes])
    puts = np.array([float(bs.price(SPOT, k, T, RATE, CARRY, VOL, "put")) for k in strikes])

    scan = parity.parity_scan(strikes, calls, puts, SPOT, T, RATE, CARRY)
    print(f"spot {SPOT}  {T * 365:.0f} days  true rate {RATE:.4f}  true carry {CARRY:.4f}\n")
    print(scan.round(8).to_string(index=False))

    print(f"\nforward, true       {float(bs.forward(SPOT, T, RATE, CARRY)):.10f}")
    print(f"forward, spread     {np.ptp(scan['forward']):.3e}")
    print(f"rate, worst error   {np.nanmax(np.abs(scan['rate'] - RATE)):.3e}")
    print(f"carry, worst error  "
          f"{np.nanmax(np.abs(parity.implied_carry(calls, puts, strikes, T, SPOT, RATE) - CARRY)):.3e}")

    lo, hi = 2, len(strikes) - 3
    box = parity.box_rate(calls[lo], calls[hi], puts[lo], puts[hi],
                          strikes[lo], strikes[hi], T)
    print(f"\nbox {strikes[lo]:.0f}/{strikes[hi]:.0f} pays "
          f"{strikes[hi] - strikes[lo]:.0f} at expiry whatever happens")
    print(f"  zero-coupon rate it synthesises  {float(box):.10f}")
    print(f"  error against the true rate      {abs(float(box) - RATE):.3e}")

    print("\nno-arbitrage checks on the clean chain")
    print(f"  calls rising with strike : {parity.monotone_violations(strikes, calls)}")
    print(f"  convexity violations     : {parity.butterfly_violations(strikes, calls)}")

    i = 6
    fly = calls[i - 1] - 2 * calls[i] + calls[i + 1]
    print("")
    print(f"butterfly {strikes[i-1]:.0f}/{strikes[i]:.0f}/{strikes[i+1]:.0f} is worth "
          f"{fly:.4f} per share")
    print(f"convexity breaks once the middle call is marked up past half of that, "
          f"{fly / 2:.4f}")
    for markup in (0.90 * fly / 2, 1.01 * fly / 2, 1.5 * fly / 2, 3.0 * fly / 2):
        bad = calls.copy()
        bad[i] += markup
        hits = parity.butterfly_violations(strikes, bad)
        edge = f"{hits[0][3]:.4f} per share" if hits else "none"
        print(f"  middle call marked up by {markup:7.4f}  ->  arbitrage {edge}")


if __name__ == "__main__":
    main()
