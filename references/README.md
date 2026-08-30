# Literature Foundation — lamXsim GDS Delamination Correlation Engine

For the engineering interpretation of these papers - including the causal
chain, current code coverage, GDS-only limits and the work a human owner must
take over - read the
[first-principles engineering guide](../docs/first_principles_engineering.md).

Generated 2026-08-29. Availability verified against OpenAlex, Unpaywall and
Semantic Scholar. **No paywall was circumvented.**

## Acquisition status

| # | Paper | Access | Local PDF |
|---|-------|--------|-----------|
| 1 | Vanstreels et al. 2020, *Impact of BEOL architecture on CPI in advanced interconnects*, Microelectronics Reliability | **Closed** (Elsevier) | — |
| 2 | Zahedmanesh & Vanstreels 2019, *Mechanical integrity of nano-interconnects; the impact of metallization density*, Micro and Nano Engineering | **Gold OA, CC-BY-NC-ND** — legally free, but the publisher blocks scripted download | — (open in a browser) |
| 3 | Yoo et al. 2004, *Characterization of patterned low-k film delamination during CMP…*, IITC | **Closed** (IEEE Xplore) | — |
| 4 | Tan et al. 2008, *Delamination-induced dielectric breakdown in Cu/low-k interconnects*, JMR | **Closed** (Cambridge/Springer) | — |
| 5 | Rabie et al. 2018, *BEoL Layout Design Considerations to Mitigate CPI Risk*, IITC | **Closed** (IEEE Xplore) | — |

Four of five are closed access. **Zahedmanesh 2019 is genuinely open access**
(CC-BY-NC-ND) — the only obstacle is ScienceDirect's bot protection, so it
opens fine in a normal browser.

### Downloaded (open access, in `pdf/`)

The two MDPI articles are **CC BY 4.0** and are redistributed here under that
licence, with attribution as given below. The Gambino paper is **IEEE
copyrighted** (`978-1-4244-4072-6/09/$25.00 2009 IEEE`) and carries no
redistribution licence, so it is deliberately **not tracked in this
repository** — download it directly from the IEEE SSCS Denver chapter
reference page:
<https://ewh.ieee.org/r5/denver/sscs/References/2009_09_Gambino.pdf>

| File | Pages | Why it is here |
|---|---|---|
| `OA_BEoL-failure-locations-20nm-CPI-semielliptical-cracks.pdf` | 14 | Where in the BEoL stack and where on the die CPI cracks initiate — this is the **PACKAGE_POSITION prior** the engine must control for |
| `OA_Micromachines2025_BEOL-design-factors-thermal-reliability-FCCSP.pdf` | 14 | Freely readable modern stand-in for Vanstreels 2020 on BEOL design factors vs thermo-mechanical reliability |
| *(not in this repository)* `OA_Gambino_Cu-interconnect-32nm-node-and-beyond.pdf` | 8 | ULK integration / CMP delamination / CPI background at the node Yoo 2004 targets |

## DOI correction

The Vanstreels 2020 DOI is **`10.1016/j.microrel.2020.113825`**.
A plausible-looking `10.1016/j.microrel.2020.113767` resolves to an unrelated
paper on TRIAC ageing — do not cite it.

Zahedmanesh & Vanstreels is in **Micro and Nano Engineering**
(`10.1016/j.mne.2018.12.001`), *not* Materials Today Nano.

## How to obtain the four closed papers

1. **Institutional access** — an IEEE Xplore + ScienceDirect + Cambridge Core
   subscription covers all four. This is the intended route.
2. **Author request** — `mailto:` the corresponding authors; imec authors
   (Vanstreels, Zahedmanesh) routinely share reprints.
3. **Interlibrary loan** — for the two 2004/2008 papers.

Place the PDFs in `pdf/` using the BibTeX keys as filenames:
`vanstreels2020beol.pdf`, `yoo2004perimeter.pdf`, `tan2008delamination.pdf`,
`rabie2018cpi.pdf`, `zahedmanesh2019metallization.pdf`.

## Files

- `delam_gds_features.bib` — BibTeX for all five core papers plus the three OA PDFs, with per-entry notes on which spec feature each supports.
- `feature_evidence_map.csv` — machine-readable feature ↔ evidence mapping (30 rows), keyed to the spec sections.

## What the literature actually constrains in the design

**1. `perimeter_density` is not a secondary diagnostic.** Yoo 2004 found pattern
perimeter more decisive than pattern density for patterned ULK delamination.
This is the single strongest justification for the spec's §26 experiment
(same density, different perimeter).

**2. The metal-density relationship is non-monotonic.** Zahedmanesh 2019 shows a
stiff top Z-group can *lower* ULK pre-crack ERR through elastic stress
shielding. So:
- do not encode any monotone "denser → worse" prior anywhere;
- do not use one-sided statistical tests;
- expect and allow **sign reversal across layers** — the same feature can point
  opposite ways on a top vs an intermediate layer. Association tables must
  report signed effect size per layer, never a pooled magnitude.

**3. Corners and line ends are the mechanistic hotspots, not comb lines.**
Tan 2008 observed delamination at terminated tips and corners. This raises the
priority of getting a rigorous morphological definition of `line_end_density`
before Phase 2 — with a merged, flattened `Region` the concept does not exist
unless it is defined explicitly.

**4. Die corner and bump neighborhood are first-order confounders.** Rabie 2018's
levers are all die-corner and bump-relative (corner metal tiles, diagonal Al cap
under corner bumps, pad geometry, PSPI opening, crackstop). This confirms the
review point that a **position-only baseline model is mandatory** and that a
**bump/C4 map input is missing from spec §2** — three rows in
`feature_evidence_map.csv` are marked `tier1_confounder` for this reason.

**5. Connectivity / largest-polygon features have no direct delamination
evidence.** They are derived descriptors. In `feature_evidence_map.csv` they are
marked `exploratory` — report effect sizes without FDR-based significance
claims, per the tiered-hypothesis approach.

**6. Tier split for §12.** `feature_evidence_map.csv` supplies it: 17 `tier1`
+ 3 `tier1_confounder` (FDR-corrected primary hypotheses) vs 10 `exploratory`
(effect size + CI only). This keeps the corrected hypothesis count near ~20
families instead of ~8,000, which is what makes the multiple-comparison problem
tractable.
