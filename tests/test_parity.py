"""Parity holds by arithmetic, so these tests are exact, not approximate."""

import numpy as np
import pytest

from opt import bs, parity

S, T, R, Q, SIGMA = 100.0, 0.5, 0.045, 0.015, 0.22
STRIKES = np.arange(70.0, 131.0, 5.0)


@pytest.fixture
def chain():
    c = np.array([float(bs.price(S, k, T, R, Q, SIGMA, "call")) for k in STRIKES])
    p = np.array([float(bs.price(S, k, T, R, Q, SIGMA, "put")) for k in STRIKES])
    return c, p


def test_implied_forward_is_identical_at_every_strike(chain):
    c, p = chain
    f = parity.implied_forward(c, p, STRIKES, T, R)
    assert np.ptp(f) < 1e-9
    assert abs(f[0] - float(bs.forward(S, T, R, Q))) < 1e-9


def test_implied_rate_recovers_the_rate(chain):
    c, p = chain
    assert np.nanmax(np.abs(parity.implied_rate(c, p, STRIKES, T, S, Q) - R)) < 1e-12


def test_implied_carry_recovers_the_dividend(chain):
    c, p = chain
    assert np.nanmax(np.abs(parity.implied_carry(c, p, STRIKES, T, S, R) - Q)) < 1e-12


def test_box_recovers_the_rate_from_any_strike_pair(chain):
    c, p = chain
    for i in range(len(STRIKES) - 2):
        for j in range(i + 2, len(STRIKES)):
            r = parity.box_rate(c[i], c[j], p[i], p[j], STRIKES[i], STRIKES[j], T)
            assert abs(float(r) - R) < 1e-9


def test_box_value_is_independent_of_volatility():
    values = []
    for sig in (0.05, 0.20, 0.60, 1.50):
        c = [float(bs.price(S, k, T, R, Q, sig, "call")) for k in (90.0, 110.0)]
        p = [float(bs.price(S, k, T, R, Q, sig, "put")) for k in (90.0, 110.0)]
        values.append(float(parity.box_value(c[0], c[1], p[0], p[1])))
    assert np.ptp(values) < 1e-9
    assert abs(values[0] - 20.0 * np.exp(-R * T)) < 1e-9


def test_conversion_has_no_edge_against_a_fair_chain(chain):
    c, p = chain
    assert np.max(np.abs(parity.conversion_edge(c, p, STRIKES, S, T, R, Q))) < 1e-10


def test_a_clean_chain_has_no_violations(chain):
    c, _ = chain
    assert parity.monotone_violations(STRIKES, c) == []
    assert parity.butterfly_violations(STRIKES, c) == []


def test_an_overpriced_call_breaks_convexity(chain):
    c, _ = chain
    bad = c.copy()
    bad[5] += 1.5
    hits = parity.butterfly_violations(STRIKES, bad)
    assert len(hits) == 1
    assert hits[0][1] == STRIKES[5]
    assert hits[0][3] > 1.0


def test_a_call_rising_with_strike_is_caught(chain):
    c, _ = chain
    bad = c.copy()
    bad[8] = bad[7] + 1.0
    assert (STRIKES[7], STRIKES[8]) in parity.monotone_violations(STRIKES, bad)


def test_parity_scan_reports_a_flat_forward(chain):
    c, p = chain
    df = parity.parity_scan(STRIKES, c, p, S, T, R, Q)
    assert list(df.columns) == ["strike", "call", "put", "forward", "rate", "edge", "forward_dev"]
    assert df["forward_dev"].abs().max() < 1e-9
