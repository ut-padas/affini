"""The conservation laws a KIDApy network declares: parsed from its species and checked against every reaction."""

import numpy as np
import pytest

from src.problems.ode import astro

KIDA_SPECIES = astro.KIDAPY / "networks" / "kida.uva.2024" / "gas_species_kida.uva.2024.in"


def test_composition_reads_formulas_ions_isomers_and_lumped_species():
    assert astro.composition("C10H+") == {"charge": 1, "C": 10, "H": 1}
    assert astro.composition("l-C3H2") == {"C": 3, "H": 2}
    assert astro.composition("e-") == {"charge": -1}
    assert astro.composition("HeH+") == {"charge": 1, "He": 1, "H": 1}
    assert astro.composition("GRAIN-") == {"grain": 1, "charge": -1}
    assert astro.composition("M+") == {"charge": 1, "M": 1}
    assert astro.composition("CHx") == {"H": None, "C": 1}
    with pytest.raises(ValueError):
        astro.composition("Xy2")


@pytest.mark.skipif(not KIDA_SPECIES.exists(), reason="KIDApy is not checked out")
def test_composition_matches_every_species_of_the_kida_species_file():
    cols = ("charge", *astro.ELEMENTS)
    for line in KIDA_SPECIES.read_text().splitlines():
        f = line.split()
        if len(f) < 15 or f[0].startswith("GRAIN"):  # the pseudo-grains carry a grain count the file has no column for
            continue
        assert astro.composition(f[0]) == {c: v for c, v in zip(cols, map(int, f[1:15]), strict=True) if v}, f[0]


def test_a_quantity_is_a_law_only_when_every_reaction_conserves_it():
    species = ["C", "C+", "e-", "CO", "O", "OHx"]
    reactions = [
        {"reactants": ["C+", "e-"], "products": ["C"]},
        {"reactants": ["C", "O"], "products": ["CO"]},
        {"reactants": ["O"], "products": ["OHx"]},
        {"reactants": ["C+", "OHx"], "products": ["CO", "H+"]},  # H+ is not a species of the network: charge leaves it
    ]
    W = astro.conserved_totals(species, reactions)
    # C and O are conserved; charge is not (the untracked H+ takes it), and H is untracked (OHx), so neither is a law
    assert W.shape == (2, len(species))
    assert set(map(tuple, W)) == {(1, 1, 0, 1, 0, 0), (0, 0, 0, 1, 1, 1)}


def test_a_declared_law_is_what_invariants_of_returns():
    from dataclasses import replace

    from src.benchmark import step_maps as cb

    p = cb.STEP_MAPS["robertson"]
    W = np.array([[1.0, 1.0, 1.0]])
    q = replace(p, system=replace(p.system, conserved=W))
    Z = np.log10(cb.STEP_MAPS["robertson"].initial(5, 1))
    out = cb.invariants_of(q, Z[:, None, :].repeat(2, axis=1))
    assert out.shape == (1, 3) and np.allclose(np.abs(out), 1 / np.sqrt(3))


@pytest.mark.skipif(astro.NELSON is None, reason="KIDApy is not checked out")
def test_nelson_declares_exactly_the_laws_its_right_hand_side_conserves():
    W = np.asarray(astro.NELSON.conserved)
    Y = astro.nelson_initial(8, 3)
    F = np.array([astro.NELSON.rhs(y) for y in Y])
    assert W.shape[0] == 4  # C, O, He and the lumped metal; charge and H leave through untracked species
    assert np.abs(F @ W.T).max() <= 1e-12 * np.abs(F).max()
