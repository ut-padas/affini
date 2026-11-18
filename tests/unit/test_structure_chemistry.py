"""Chemistry oracles (src/structure/chemistry.py): rates, Jacobians and conservation."""

import numpy as np

from src.structure import chemistry as c


def _primordial_state(nH=1e3, T=1000.0, xe=1e-4, xh2=1e-6):
    y = np.full(10, 1e-20 * nH)
    y[0], y[1], y[3], y[5], y[6], y[-1] = nH * (1 - xe - 2 * xh2), nH * xe, nH * xh2, nH * xe, 0.079 * nH, T
    return y


def test_case_a_recombination_matches_the_standard_value():
    assert abs(c.primordial_rates(1e4)["k2"] - 4.17e-13) < 0.01e-13


def test_analytic_jacobians_match_finite_differences():
    for net, y in ((c.ROBERTSON, np.array([0.9, 2e-5, 0.1])), (c.PRIMORDIAL, _primordial_state())):
        J = net.jac(y)
        h = 1e-6 * np.maximum(np.abs(y), 1e-10)
        Jn = np.stack([(net.rhs(y + h[i] * np.eye(len(y))[i]) - net.rhs(y - h[i] * np.eye(len(y))[i])) / (2 * h[i])
                       for i in range(len(y))], axis=1)  # fmt: skip
        n = len(net.species) - (1 if net.species[-1] == "T" else 0)
        assert np.max(np.abs(J[:, :n] - Jn[:, :n])) <= 1e-6 * np.max(np.abs(J))


def test_a_primordial_step_conserves_nuclei_and_charge():
    y = _primordial_state()
    y1 = c.PRIMORDIAL.step(y[None], 3.156e12)[0]

    def totals(v):
        return np.array([v[0] + v[1] + v[2] + 2 * v[3] + 2 * v[4], v[6] + v[7] + v[8], v[1] - v[2] + v[4] - v[5] + v[7] + 2 * v[8]])

    assert np.allclose(totals(y1), totals(y), rtol=1e-6, atol=1e-6)
    assert y1[-1] == y[-1] and y1[3] > y[3]  # isothermal; H2 forms


def test_the_stiff_test_problems_match_their_published_reference_solutions():
    from scipy.integrate import solve_ivp

    for net, y0, tend, ref in ((c.HIRES, c.HIRES_Y0, c.HIRES_TEND, c.HIRES_REF), (c.POLLU, c.POLLU_Y0, c.POLLU_TEND, c.POLLU_REF),
                               (c.CHEMAKZO, c.CHEMAKZO_Y0, c.CHEMAKZO_TEND, c.CHEMAKZO_REF)):  # fmt: skip
        s = solve_ivp(lambda t, y, net=net: net.rhs(y), (0.0, tend), y0, method="Radau", jac=lambda t, y, net=net: net.jac(y),
                      rtol=1e-10, atol=net.atol)  # fmt: skip
        big = np.abs(ref) > 1e-6 * np.max(np.abs(ref))
        assert np.max(np.abs(s.y[big, -1] - ref[big]) / np.abs(ref[big])) < 1e-6, net.name
