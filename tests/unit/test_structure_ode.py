"""ODE oracles (src/structure/ode): rates, Jacobians, conservation, integrators, regimes and the ladder."""

import numpy as np

from src.structure import ode
from src.structure.ode import chemistry as c


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
        n = len(net.components) - (1 if net.components[-1] == "T" else 0)
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


def test_the_stiff_families_have_analytic_jacobians_and_stiffness_that_grows_with_the_grid():
    for fam, (make, initial, sizes) in ode.FAMILIES.items():
        if fam == "lorenz96":
            continue
        stiff = []
        for n in sizes[:4]:
            net, y = make(n), initial(n)(1, 3)[0]
            J = net.jac(y)
            h = 1e-6
            Jn = np.stack([(net.rhs(y + h * e) - net.rhs(y - h * e)) / (2 * h) for e in np.eye(len(y))], axis=1)
            assert np.max(np.abs(J - Jn)) <= 1e-6 * np.max(np.abs(J)), fam
            stiff.append(np.max(np.abs(np.linalg.eigvals(J))))
        assert stiff[-1] > 4 * stiff[0], fam


def test_a_ladder_step_stays_in_its_coordinates_and_moves_towards_the_stable_state():
    net = ode.fisher_kpp(5)
    y = ode.fisher_kpp_initial(5)(4, 1)
    z1 = net.coords_step(net.coords(y), 0.1)
    z0 = net.coords(y)
    assert np.all(np.isfinite(z1)) and np.all(z1 <= 1e-9)  # 0 < u <= 1 stays, in log10
    assert np.all(z1.min(axis=1) > z0.min(axis=1))  # growth lifts the smallest value (diffusion may lower a peak)
    ac = ode.allen_cahn(5)
    u = ode.allen_cahn_initial(5)(4, 1)
    assert np.array_equal(ac.coords(u), u)  # linear coordinates


def test_invariants_are_found_and_the_projection_restores_them_exactly():
    from src.structure.ode import bench as cb

    p = cb.PROBLEMS["robertson"]
    Y = c.robertson_initial(30, 4)
    Z = np.log10(Y)
    W = cb.invariants_of(p, Z[:, None, :].repeat(2, axis=1))
    assert W.shape == (1, 3) and np.allclose(np.abs(W), 1 / np.sqrt(3))  # total mass

    class Biased:
        def predict(self, Q):
            return np.full(Q.shape, 0.01)  # every abundance up by 2.3%: the total is off

        def refuses(self, Q):
            return np.zeros(len(Q), dtype=bool)

    Z1 = Z + cb.Projected(Biased(), p, W).predict(Z)
    assert np.allclose((10.0**Z1).sum(axis=1), Y.sum(axis=1), rtol=1e-12)
    assert np.all(np.abs(Z1[:, 1] - Z[:, 1] - 0.01) < 1e-6)  # the trace species keeps its change


def test_lorenz96_has_an_analytic_jacobian_and_an_explicit_step_that_matches_a_tight_implicit_one():
    from scipy.integrate import solve_ivp

    for n in (4, 5, 10):
        net, y = ode.lorenz96(n), ode.lorenz96_initial(n)(1, 2)[0]
        h = 1e-6
        Jn = np.stack([(net.rhs(y + h * e) - net.rhs(y - h * e)) / (2 * h) for e in np.eye(n)], axis=1)
        assert np.max(np.abs(net.jac(y) - Jn)) <= 1e-8
        assert net.method == "RK45" and not net.log
        ref = solve_ivp(lambda t, v, net=net: net.rhs(v), (0.0, 0.25), y, method="Radau", jac=lambda t, v, net=net: net.jac(v), rtol=1e-12, atol=1e-14)
        assert np.max(np.abs(net.step(y[None], 0.25)[0] - ref.y[:, -1])) < 1e-7


def test_problems_declare_a_regime_and_their_baseline_follows_their_integrator():
    from src.structure.ode import bench as cb

    assert cb.PROBLEMS["lorenz96_d10"].regime == "nonstiff" and cb.PROBLEMS["lorenz96_d10"].baseline == "scipy-rk45-cpu1"
    assert cb.PROBLEMS["bruss_d10"].regime == "stiff" and cb.PROBLEMS["robertson"].baseline == "scipy-radau-cpu1"
    assert all(p.regime in cb.REGIMES and p.baseline in cb.BASELINES for p in cb.PROBLEMS.values())


def test_the_fill_runs_the_first_families_before_the_rest_each_by_dimension():
    from src.structure.ode.fill import family, ladder

    names = ladder(20, first=("lorenz96",))
    k = sum(family(n) == "lorenz96" for n in names)
    assert k == 4 and all(family(n) == "lorenz96" for n in names[:k])
    assert names[:k] == ["lorenz96_d4", "lorenz96_d5", "lorenz96_d10", "lorenz96_d20"]
    assert names[k:] == [n for n in ladder(20) if family(n) != "lorenz96"]


def test_a_row_label_names_the_parameters_that_set_the_regime():
    from src.structure.ode import bench as cb

    assert cb._row_label("lorenz96_d10").startswith("lorenz96_d10 (10; F = 1")
    assert cb._row_label("bruss_d10").startswith("bruss_d10 (10")
