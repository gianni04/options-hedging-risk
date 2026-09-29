import numpy as np
import pandas as pd
import pytest

from opt import bs
from opt import chain as ch

SPOT, R, Q, VOL = 5000.0, 0.04, 0.013, 0.18
ASOF = "2026-01-05"


def synthetic(days=(14, 30, 45)):
    rows = []
    for n in days:
        expiry = pd.Timestamp(ASOF) + pd.Timedelta(days=n)
        for k in np.arange(4000.0, 6005.0, 25.0):
            for right in ("call", "put"):
                px = float(bs.price(SPOT, k, n / 365, R, Q, VOL, right))
                rows.append((expiry, right, k, px - 0.05, px + 0.05, px))
    return pd.DataFrame(rows, columns=["expiry", "right", "strike", "bid", "ask", "mid"])


def test_parse_keeps_one_root_and_two_sided_quotes():
    data = {"options": [
        {"option": "SPXW261016C07800000", "bid": 10.0, "ask": 11.0},
        {"option": "SPXW261016P07800000", "bid": 0.0, "ask": 0.05},
        {"option": "SPX261016C07800000", "bid": 10.0, "ask": 11.0},
        {"option": "SPXW261016C078000001", "bid": 10.0, "ask": 11.0},
    ]}
    out = ch.parse(data)
    assert len(out) == 1
    row = out.iloc[0]
    assert (row.expiry, row.right, row.strike, row.mid) == (pd.Timestamp("2026-10-16"), "call", 7800.0, 10.5)


def test_rate_matches_the_panel_convention():
    assert ch.rate(40.05) == pytest.approx(np.log(1 + 0.04005 * 91 / 360) / (91 / 365))


def test_atm_term_recovers_forward_and_vol():
    term = ch.atm_term(synthetic(), SPOT, ASOF, R)
    for n, row in term.iterrows():
        assert row.forward == pytest.approx(SPOT * np.exp((R - Q) * n / 365), rel=1e-9)
        assert row.atm == pytest.approx(VOL, abs=1e-6)
    assert ch.atm_at(term, 20) == pytest.approx(VOL, abs=1e-6)


def test_smile_of_a_flat_chain_is_flat():
    c = synthetic()
    term = ch.atm_term(c, SPOT, ASOF, R)
    n, sm = ch.smile(c, term, SPOT, ASOF, R, 30, (0.9, 1.0, 1.1))
    assert n == 30
    assert np.allclose(sm["iv"], VOL, atol=1e-6)
