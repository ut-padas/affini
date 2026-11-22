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

**Coverage in the dense range** (200 training tracers; share of states above 1e-12): every key species 0.98–1.00 except e- (0.72).

**To check before reading KIDA results physically.** In the dense range as set up (`theory/recipe/r3_kida.py`: n_H 1e3–1e5, T 10–50 K,
A_v 3–10, UV 1–10 Draine, KIDApy with its pseudo-grains, burn-in 1e4 yr from KIDApy's initial abundances) the median HCO+ abundance is
about 1e-4, comparable to CO, and the median electron abundance about 3e-12. Dense clouds typically have HCO+ near 1e-9 and electrons
well above that, so the grain treatment, the environment range or the burn-in may be unrepresentative. The surrogate comparisons
measure agreement with the solver whatever its chemistry, so they stand; the physical reading of KIDA results waits for this check.
