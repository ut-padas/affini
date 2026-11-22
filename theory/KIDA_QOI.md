# KIDA scoring: the key species

The primary error metric for KIDA uva 2024 (578 species, `libs/KIDApy`) is the worst error over **13 key species**, at 0.01 dex,
wherever the species' true abundance exceeds 1e-12. All 578 species stay in the state the surrogate predicts. The strict metric of the
benchmark (every species above 1e-12, 43–223 species per state) is reported beside it as a secondary number.

| Role | Species | Why (sources) |
|---|---|---|
| Cooling (couples chemistry to the gas's thermal state) | H2, CO, C, C+, O, OH, H2O | C+, C, O and CO are the dominant fine-structure and rotational coolants; OH and H2O dominate molecular cooling in dense, low-metallicity gas ([arXiv:2610.17577](https://arxiv.org/pdf/2610.17577), [arXiv:2206.06393](https://arxiv.org/pdf/2206.06393), [A&A 2018](https://www.aanda.org/10.1051/0004-6361/201834033), [arXiv:1402.0867](https://arxiv.org/pdf/1402.0867)); reduced networks for hydrodynamics keep H, C+, C, CO for the same reason |
| Ionisation (sets the magnetic coupling through ambipolar diffusion) | e-, H3+, HCO+, N2H+, S+ | the electron fraction controls neutral–field coupling; HCO+ and N2H+ are the dominant molecular ions used to measure it, with H3+ upstream of both ([arXiv:2610.08981](https://arxiv.org/pdf/2610.08981), [arXiv:astro-ph/0109023](https://arxiv.org/pdf/astro-ph/0109023), [arXiv:0805.4018](https://arxiv.org/pdf/0805.4018)); the electron budget at low density is tied to sulphur ([arXiv:2406.02387](https://arxiv.org/html/2406.02387v1)) and the role of metal ions is debated ([arXiv:0705.3765](https://arxiv.org/pdf/0705.3765)), so S+ stands in for the metal ions |
| Dense-gas tracer | HCN | among the species emulator benchmarks score ([arXiv:2412.00758](https://pdf.arxiv.org/pdf/2412.00758)) |

Also consulted: KIDApy's own examples (`examples/kida_uva_2024_point.py`: CO, H2O, OH, HCO+, N2H+, H2; `examples/nelson_shielding.py`:
CO, C, C+), and an emulator comparison that stresses rare catastrophic errors in coupled runs ([arXiv:2603.08567](https://arxiv.org/html/2603.08567v1)).
No standard list of "KIDA quantities of interest" was found; studies choose by purpose, and the three purposes above are the ones that
matter for coupling to MHD.

**Dropped:** CS, a common tracer, never exceeds 1e-12 in the dense range (median 10^-13.5), so it would never be scored.

**Coverage** is to be measured on the dark range (the dense range's numbers reflected its charge imbalance).

**The dense range's chemistry was a setup bug, now fixed.** In the dense range (`theory/recipe/r3_kida.py`) the median HCO+ was
about 1e-4 and electrons about 3e-12. The cause: KIDApy's abundances file lists no electrons and KIDApy leaves them to the caller
(charge neutrality), but the loader left them at zero, so every state carried a net positive charge of 1.7e-4 that the network
conserves, and positive ions piled up with almost nothing to recombine with. The **dark** range sets the electrons from charge
neutrality and uses the standard dark-cloud conditions of KIDA/Nautilus models: n_H = 2e4 cm^-3, T = 10 K, zeta = 1.3e-17 s^-1, A_V
10–30, atomic initial abundances with molecular H2 ([arXiv:2410.09212](https://arxiv.org/pdf/2410.09212),
[arXiv:1307.6868](https://arxiv.org/pdf/1307.6868), [arXiv:1311.3865](https://arxiv.org/pdf/1311.3865),
[arXiv:1104.1530](https://arxiv.org/pdf/1104.1530), [arXiv:2504.18138](https://arxiv.org/pdf/2504.18138)). Its ranges: n_H 1e4–1e5,
T 10–20 K, A_V 5–20, UV 1–3 Draine, driven along each tracer as before; zeta stays KIDApy's default 1.6e-17. After one step its states
are neutral and plausible: HCO+ 7.6e-10, electrons 1.0e-8, CO 1.3e-4, C+ 1.6e-10, H3+ 1.8e-10, N2H+ 2.9e-12. The dark range is the one
to read physically; the dense records are kept as the record of the charged setup.

**The benchmark's Nelson problem** starts the same way (net charge 2.0e-4, no electrons), but Nelson's reduced network does not conserve
charge (some reactions create electrons) and its states end with excess electrons. Changing it changes benchmark data, so it is left
for the final rerun and flagged here.
