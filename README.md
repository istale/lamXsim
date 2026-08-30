# lamXsim — GDS Spatial Delamination Correlation Engine

**Status: a GDS-first, literature-grounded tool for generating and testing
failure hypotheses.** Not a physics solver, and not a reliability workflow that
can be signed off unattended.

The real-data path is complete end to end: a study manifest declares the layer
semantics GDS does not contain, registration is fitted and its error gates
which analysis scales are admissible, the multi-layer stack is extracted with
vias and package context, held-out-die validation runs when more than one die
is present, and results are partitioned by what each row is allowed to claim.

Every feature family with direct delamination evidence in
`references/feature_evidence_map.csv` is implemented. What remains is not
missing features -- see [Known gaps](#known-gaps) for what is, and
[What GDS cannot answer](#what-gds-cannot-answer) for what no amount of work
here will fix.

## Engineering interpretation

lamXsim is deliberately GDS-first. It uses literature to turn layout geometry
into testable engineering hypotheses; it does not pretend that GDS contains
package material properties, thermal history, interface toughness or
inspection sensitivity. Read the
[first-principles engineering guide](docs/first_principles_engineering.md)
before applying the pipeline to measured failures. It maps each literature
claim to its GDS proxy, current code coverage, required human input and the
conclusions the evidence does not yet support.

## Why this shape

The spec's phase plan builds every feature (Phases 2–3) before any statistics
run (Phase 5). That order risks months of feature engineering before
discovering the analysis cannot be performed with the data available. This
build inverts it: the statistical machinery is finished and validated first,
on a synthetic die whose ground truth is known, so that broadening the feature
catalogue is a mechanical extension of something already proven to work.

## Quick start

```bash
PYTHONPATH=src python3 -m lamxsim phase0
```

```bash
PYTHONPATH=src python3 -m lamxsim thinslice --outdir results
```

```bash
PYTHONPATH=src python3 -m pytest tests -q
```

## Phase 0: is the study feasible?

`phase0` sizes the hypothesis space and converts it into a data requirement.
With 25 features × 12 layers × 6 scales plus gradients and cross-layer terms,
the full grid is **9,576 hypotheses**. Correcting all of them at once, and
allowing for spatial autocorrelation (design effect 5.8 at Moran's I = 0.6):

| target ROC-AUC | uncorrected | tiered FDR (20 literature-backed) | full-grid FDR (9,576) |
|---|---|---|---|
| 0.70 | 122 | 226 | 441 |
| 0.75 | 75 | 145 | 278 |
| 0.80 | 52 | 99 | 191 |

Read as: **detecting an AUC-0.75 effect needs ~145 measured failure sites**
under the tiered scheme, or ~278 if every combination is corrected together.
This is the number to take into the discussion about how much failure data to
collect, and it is why `references/feature_evidence_map.csv` splits features
into `tier1` (corrected) and `exploratory` (effect size only).

`phase0` also converts measurement accuracy into a scale floor. At a
positional uncertainty of 50 µm, the 25/50/100 µm scales are rejected and only
250 µm and above can be analysed — deciding this before extraction avoids
"discovering" that coarse scales work best when the finer ones were only ever
registration noise.

## What the thin slice established

The validation die (`layout/synth.py:validation_die`) drives metal density and
perimeter density from two uncorrelated random fields. For parallel lines of
pitch *p* and width *w*, density is *w/p* while perimeter density is ~*2/p*,
so the two decouple. Failures are then simulated from perimeter density alone.

**1. The pipeline recovers the true driver.** Perimeter density reaches
AUC 0.71–0.88; metal density 0.51–0.71. A pipeline that had collapsed into a
metal-density detector could not produce this ordering.

**2. Metal density still looks significant (q = 1e-4) although it drives
nothing.** It correlates r ≈ 0.44 with the real driver. Univariate rankings
alone cannot separate a cause from its correlate, which is why feature
clustering belongs in the reporting layer before any ranking is published.

**3. Failures are assigned to a cell by its bounds, not by a radius.** Cells
are squares; testing a radius against the cell centre inscribes a circle in
each one and silently discards everything in the corners — 21 % of the die
(1 − π/4), arranged on a regular lattice rather than scattered. An explicit
`radius_um` still selects the circular test, for deliberately crediting a
failure to every cell near it.

**4. The spatial null model is not optional.** On a die with *no* package-position
effect at all:

| | naive Mann-Whitney | block permutation |
|---|---|---|
| PACKAGE_POSITION features called significant | **11 / 12** | **0 / 12** |
| GDS_GEOMETRY features called significant | 7 / 8 | 7 / 8 |

Every position association was an artifact of spatial autocorrelation. Block
permutation removed all of them while keeping the real geometry signal. Without
spec section 15, this engine would confidently report that distance-to-die-edge
predicts delamination on a die where it does not.

**5. Effective sample size is far below cell count.** At 25 µm the grid has
6,320 cells but ~489 independent observations; at 250 µm, 56 cells and ~29.
Both numbers are reported on every row.

**6. "Best scale" needs a confidence interval.** Perimeter density peaks at
250 µm (AUC 0.883) but its CI is [0.79, 0.96], overlapping the 100 µm CI of
[0.70, 0.81]. The point estimates alone would name a best scale that the data
does not support.

**7. Perimeter cannot stand in for line terminations.** Chopping lines into
segments removes long-edge length as fast as it adds end-cap length, so
perimeter density moves only ~3 % while the termination count rises tenfold.
`line_end_density` is therefore its own feature, and with it the section 26
pair separates properly:

| | metal_density | perimeter_density | line_end_density |
|---|---|---|---|
| continuous lines | 0.4000 | 0.4080 | 0.0020 |
| segmented, same density | 0.4000 | 0.4200 | 0.0400 |
| ratio | 1.00x | 1.03x | **20x** |

## The observation unit is (cell, die)

The failure schema has always required `lot_id`, `wafer_id`, `die_x`, `die_y`,
for the held-out-die validation of spec section 17. The analysis then pooled
every die onto one grid and asked "did anything ever fail here", which is not
a rescaled version of the single-die question but a different and wrong one:

| dies | failures | pooled prevalence |
|---|---|---|
| 1 | 30 | 0.24 |
| 3 | 90 | 0.57 |
| 10 | 300 | **0.98** |

Prevalence climbs towards 1 with the number of dies, a cell that failed on one
die of ten becomes indistinguishable from one that failed on all ten, and die
identity — the thing section 17 wants to hold out — is gone before any fold can
be built from it. Requiring those columns and then discarding them is worse
than not requiring them, because a reader reasonably assumes they were used.

Each die now contributes its own labels over the same layout, features repeat
rather than labels being collapsed, and `run` uses **leave-one-die-out** folds
whenever more than one die is present. With a single die the run says plainly
that nothing here can be shown to generalise. Permutation blocks combine the
spatial block with the die, so a permutation never moves a label between dies —
and the block size still comes from the feature's own autocorrelation, because
fixing it at one cell turns the block permutation back into the naive shuffle
it exists to replace.

## Failure modes are not pooled by assumption

`failure_type` was required at import and never consulted. Li et al. (2023)
found the largest energy release rate at one particular upper BEOL interface,
with bottom interconnect interfaces more critical than sidewalls, so two
failures at the same coordinates on different interfaces are not two
observations of the same thing. `failed_layer` and `failed_interface` are now
part of the schema, a file mixing modes is refused, and pooling can be asserted
— `--allow-pooling-modes` — with the assertion recorded in the run metadata as
the operator's judgement rather than a property of the data.

## Pooling mechanisms can cancel two real effects into none

`--stratify-by failed_interface` analyses each failure population separately
and reports how far they agree. Li et al. (2023) found the largest energy
release rate at one particular upper BEOL interface, with bottom interconnect
interfaces more critical than sidewalls, and Zahedmanesh & Vanstreels (2019)
show a stiff top group *lowering* the crack driving force beneath it -- so the
same geometry helping on one interface and hurting on another is the expected
shape, not a contrived one.

On a die carrying both:

| | ROC-AUC | effect | q |
|---|---|---|---|
| pooled | 0.587 | +0.175 | **0.234 — no finding** |
| M8/ULK | 0.809 | **+0.619** | 2e-11 |
| M8/CAP | 0.310 | **−0.381** | 8e-05 |

Two strong opposite effects, pooled into nothing. The consistency table puts
sign disagreement first, because a feature pointing one way on one mechanism
and the other way on another is not one effect measured twice — and telling
those apart is what separates a mechanism from a proxy for one.

## What a primary claim rests on

The pipeline computed a within-die block permutation and then corrected the
Mann-Whitney p-value instead, leaving the spatial result in a side table. With
`n_permutations: 0` the primary table still produced FDR q-values — so the
spatial null was not a precondition for a finding, in a repository whose own
README shows that test calling 11 of 12 position associations significant
where the permutation called none.

`spatial_q_value` now comes from the block permutation and is what a primary
row rests on; `fdr_q_value` is kept as the naive diagnostic and the contrast.
They differ materially: on the validation die `perimeter_density` moves from
q = 0.00000 to **q = 0.01524**, three orders of magnitude weaker and the
honest number. Without permutations there is no primary evidence at all — the
rows land in `not_spatially_corrected` and the run says why.

## Traceability is enforced, not requested

`references/feature_evidence_map.csv` carries, per feature family, the
physical hypothesis, the supporting paper and evidence type, the exact GDS
observable and unit, the expected confounders, a named discrimination test, a
falsification condition, where it is implemented, whether it is primary or
exploratory, and what further evidence would promote it from an association to
an engineering rule.

Every feature the pipeline reports is matched against that registry, and a
feature whose entry is missing or incomplete **cannot be a primary result** --
it lands in `not_traceable`. Auditing a gap and printing the row anyway is not
enforcement. The check found four families added two commits earlier that had
no entry at all.

Package and process conditions get the same treatment from the other side.
EMC thickness, underfill CTE and modulus, reflow and thermal-cycle profile,
the dielectric stack, inspection method and sensitivity — none is in the GDS,
and the literature shows each changing the crack driving force. The manifest
requires every one to be declared fixed, stratified, or a baseline covariate;
anything left out is reported as an undeclared condition. The software cannot
check these, only refuse to let them go unmade.

## One population per die, and one layout for all of them

Inspection footprints are declared per die, with a default for the rest. A
campaign rarely inspects every die the same way -- one gets a full acoustic
scan, another three cross-sections chosen after it -- and collapsing that to a
single footprint either discards the dies inspected more thoroughly or credits
the ones inspected less with controls nobody earned. Eligibility is therefore
per (cell, die): a cell inspected on one die and not on another is a control on
the first and missing data on the second. A die with no footprint and no
default is refused rather than treated as fully inspected.

A failure that lands outside its die's footprint by less than three times its
own reported positional uncertainty is treated as the same failure seen
through its own error, not as a contradiction. Without that, the check fires
on the near-edge failures of every real campaign and gets overridden as a
matter of routine -- and a check everyone overrides is not a check. Beyond the
tolerance, measurement error stops being an explanation. With no reported
sigma the boundary is strict: no stated uncertainty, no room to grant.

Failures may carry a `layout_revision`, and one that disagrees with the
manifest, or a file spanning two of them, is refused. Every feature comes from
one GDS and every die's labels are mapped onto it, so a failure from another
revision would be scored against geometry that was never on its silicon -- and
a revision usually changes exactly the metal a study is about. Absent the
column, the run records that the assumption is unverified. The SHA-256 of the
layout is written into the run metadata, so a result names the file that
produced it.

## Routing direction is a layout lever, and needs the bump frame to see

Rabie et al. (2018) recommend running the final metal *diagonally* under the
corner bumps. No scalar distance can express that: two cells the same distance
from the same bump, one routed radially and one diagonally, are identical in
every other feature.

Orientation is measured as a length-weighted axial tensor -- angles doubled
before averaging, since a line at 179 degrees and one at 1 degree differ by 2,
not 178 -- giving a dominant direction and a coherence. The direction is then
resolved against the bump radial direction:

| routing vs radial | `routing_radial_alignment` | `routing_diagonality` |
|---|---|---|
| radial | +1 | 0 |
| **45 degrees** | 0 | **1** |
| tangential | −1 | 0 |

The recommendation gets a feature of its own rather than being the midpoint of
one. A window with no dominant direction returns NaN instead of a spurious 45
degrees, because isotropic and deliberately-diagonal sit at the same place on
the alignment axis and only coherence separates them.

On a die where every block has identical density and pitch and only the routing
direction varies:

| feature | ROC-AUC | q |
|---|---|---|
| **`routing_diagonality`** | **0.770** | **<0.0001** |
| `metal_density` | 0.500 | 1.000 |
| `perimeter_density` | 0.475 | 0.912 |
| `corner_density` | 0.500 | 1.000 |
| `orientation_anisotropy` | 0.524 | 0.912 |

Without it the run finds nothing in geometry, and `distance_to_nearest_corner`
takes the top of the table instead — the layout effect attributed to die
position.

These are GDS_GEOMETRY features even though they need a bump map to compute.
Classifying them as package position would put the designer's lever into the
baseline the lever is supposed to beat.

## Registration: what the measurement accuracy allows

Spec section 10 assumes failures have already been brought into layout
coordinates. That is a fitted transform with its own uncertainty, and the
uncertainty decides which analysis scales mean anything, so `lamxsim register`
fits it and reports the floor.

```bash
PYTHONPATH=src python3 -m lamxsim register fiducials.csv
```

**The residual of a fit is not the accuracy of the mapping.** Each fiducial
supplies two equations while the model consumes `dof` of them. Fit four
fiducials with a six-parameter affine and only two residual degrees of freedom
remain; over 200 draws with 8 um of true noise:

| | mean |
|---|---|
| in-fit RMS | **4.70 um** (theory: `sigma*sqrt(2/8)` = 4.00) |
| leave-one-out RMS | **46.5 um** |

The in-fit number would certify the 25 um scale. The honest floor is 140 um.
With three fiducials and an affine model the in-fit RMS is *exactly zero* — the
fit passes through every point by construction — so `register` refuses to
certify any scale from an under-determined fit rather than reporting that zero.
`position_sigma_um` therefore always comes from leave-one-out, and combines in
quadrature with whatever uncertainty the measurement itself reported.

**Model choice is made on prediction error, not on fit.** A richer model always
fits the fiducials better and does not always place an unseen point better. On
10 fiducials with 12 um noise and a 0.15-degree rotation:

| model | in-fit RMS | leave-one-out RMS |
|---|---|---|
| translation | 17.27 | 19.73 |
| **rigid** | 10.42 | **12.30** |
| similarity | 10.18 | 12.67 |
| affine | 10.09 | 13.83 |

In-fit picks `affine`; prediction error picks `rigid`, and recovers the true
12 um noise. Choosing on in-fit residual lets real registration error be
absorbed into shear and scale where it stops being visible.

Also handled: **reflection is reported, not absorbed**. Backside acoustic
imaging mirrors the frame, and a fit that quietly takes the flip as a negative
scale still lands every fiducial. Outlying fiducials are found by leave-one-out
error and dropped before the model is re-selected — on a set with one
mis-identified mark, that took the floor from 191 um to 19 um.

## Phase 6: does geometry add anything?

```bash
PYTHONPATH=src python3 -m lamxsim phase6 --outdir results
```

**Spatial separation first.** Blocking alone does not separate train from test:
a training cell can sit one stride from a test cell across the block boundary.
On a 1,600-cell grid at 50 um:

| scheme | min train/test separation | train cells |
|---|---|---|
| random cell split | 50 um | 1,280 |
| 250 um blocks, no buffer | **50 um** | 1,280 |
| 250 um blocks + 250 um buffer | 250 um | 513 |
| 250 um blocks + 500 um buffer | 500 um | 87 |

The cost of real separation is most of the training data, which is the concrete
reason held-out dies are worth more than any within-die scheme.

**Every model is scored against a position-only baseline.** An absolute AUC
answers "can something predict this", which is not the question. `ablation.run`
refuses to proceed without PACKAGE_POSITION columns present.

On the validation die, where failures are driven by perimeter density and metal
density is spatially decoupled from it:

| model | AUC | ΔAUC vs position | 95 % CI | adds information |
|---|---|---|---|---|
| position only | 0.493 | — | — | — |
| A metal density | 0.569 | +0.076 | [+0.011, +0.136] | yes |
| B + via | 0.569 | +0.076 | [+0.011, +0.136] | yes |
| **C + perimeter** | **0.773** | **+0.280** | **[+0.231, +0.318]** | **yes** |
| D + terminations, corners | 0.768 | +0.275 | [+0.225, +0.315] | yes |
| E–G + orientation, gradients, cross-layer | 0.764 | +0.271 | [+0.218, +0.312] | yes |

That is the section 18 question answered, and it is answered by the size of
each step rather than by which steps reach significance. Metal density is not
the driver but correlates with it (r ~ 0.44 on this die), so it carries a
small real effect; the perimeter step is roughly four times larger, and nothing
added after it improves on it. An ablation read only by significance would have
called metal density a finding.

The interval is a paired block bootstrap over the same out-of-fold predictions
— a per-observation bootstrap would return one several times too narrow.

Run with `--null` and every family reports `adds_information = False`, with all
seven intervals spanning zero.

## Gradients and cross-layer architecture

Spec sections 5 and 7 exist because a layout is not a set of independent
per-layer scalar maps. Two dies were built where that reduction provably
destroys the driver.

**A gradient driver the value cannot see.** Metal density varies sinusoidally,
so its value and its gradient are orthogonal (measured r = 0.000). Failures
are driven by the gradient alone:

| feature @100um | ROC-AUC | q |
|---|---|---|
| `metal_density_grad_mag` | **0.767** | <0.0001 |
| `metal_density` | 0.482 | 0.95 |
| `metal_density_dx` / `_dy` | 0.499 / 0.500 | 1.0 |

A pipeline that only scored absolute values would report nothing here. The
signed components are null on their own because the sine's gradient alternates
sign — the magnitude is what carries the effect. At 250 um everything collapses,
which is correct: the driving wavelength is 500 um.

**Gradients need their boundary ring dropped.** Interior cells get a centred
difference, die-edge cells can only get a one-sided one, and one-sided
differences are systematically larger. On a field of pure noise the boundary
ring's `|grad|` runs 1.55x the interior mean, producing a *significant*
correlation with `distance_to_die_edge` — Spearman −0.115, p = 0.021 — from
numerics alone. That is a PACKAGE_POSITION confounder manufactured by the
differencing scheme. The boundary ring is set to NaN rather than filled.

**Cross-layer differences are emitted signed *and* as magnitudes.** The
shielding result requires the sign; a mismatch-driven effect requires the
magnitude, and neither substitutes for the other. On a die whose two layers
draw their orientation from independent random fields — so no single layer
predicts the mismatch — the same underlying quantity gives:

| feature @100um | ROC-AUC |
|---|---|
| `orientation_mismatch_M8_M7` (magnitude) | **0.819** |
| `orientation_difference_M8_M7` (signed) | 0.396 |
| best per-layer feature | 0.391 |

The signed form collapses because both directions of disagreement sit at
opposite ends of its scale. Layer identity stays in the name throughout:
`density_difference_M8_M7`, never a pooled index.

**Trimming the layer-pair set saves compute, not sample size.** Going from all
66 pairs to the 21 the literature motivates cuts cross-layer hypotheses from
3,168 to 1,008 — and moves the required failure count from 278 to 278. Sample
size scales with log(m), so a 21 % reduction is nothing. Only the tiered scheme
matters there, because it is a 500x reduction rather than a 1.26x one.

## Choosing the line-end definition

Line ends are the one tier-1 feature with no self-evident definition on merged
geometry, so the definition was picked by scoring candidates against eight
patterns whose termination count follows from their construction — continuous
lines, segmented lines, a solid plate, a dummy-fill array, a closed ring, a
comb, staircase lines, T junctions.

| definition | total error | note |
|---|---|---|
| **D1** short edge, convex corner at both ends | **144** | every side of a fill square qualifies |
| **D2** D1 + flanks at least `aspect` x the cap | **0** | recommended |
| **D3** D2 + flanks must run antiparallel | **0** | extra SVRF condition, no gain |

D3 was written to reject staircase jogs, but it never disagreed with D2 — on
300 random Manhattan layouts carrying 7,579 terminations the two matched every
time, because on Manhattan rings the antiparallel condition is already implied
by convex-convex. **D2 is the recommendation.**

Parameter behaviour, measured rather than assumed:

* `aspect` is the knob that matters, and it has a plateau: exact from 1.2 to
  2.0. At 1.0 dummy fill floods the result (a square's flanks equal its cap,
  so its aspect is exactly 1); from 3.0 upward real short stubs start being
  dropped. Default 1.5, mid-plateau.
* `w_max` is a step, not a dial. Identical results from 1 to 20 um on 1-um
  lines, then it flips once it reaches the width of a wide structure. It is
  what separates "a routing line terminated" from "a power strap edge", so set
  it between the routing width and the strap width and leave it alone.

One caution the benchmark itself produced: on the comb pattern the detectors
returned 9 where the hand-written ground truth said 10. The detectors were
right — the first tooth sits flush with the spine's left end, so those two
faces merge into an L-bend rather than remaining two free ends.

## Calibre / SVRF route

Calibre's `DENSITY ... WINDOW ... STEP ...` is already a moving-window scanner,
so the productive split is Calibre for exact geometry and Python for spatial
statistics. Every length- or count-density feature becomes an area density and
rides the native scanner:

```
METAL -> marker layer -> DENSITY WINDOW/STEP -> per-window value -> Python
```

`lamxsim.calibre.svrf` generates the decks; `lamxsim.calibre.ingest` reads
the results back onto the same grid the KLayout path uses, so the two are
interchangeable downstream. Three findings from measuring the approximation
against exact edge lengths are built in.

**Use the inside band, not a straddling one.** `METAL NOT (SIZE METAL BY -eps)`
with `P = area/eps`, rather than `SIZE(+eps) NOT SIZE(-eps)` with `P = area/2eps`.
A straddling band sits half outside the metal, so any metal edge lying on a
window border loses the half of its band that falls in the next window. On a
test bar whose edges coincide with the window boundary the straddling band
reports 20 um against a true 40 um; the inside band reports 39.96 um. Power
rails aligned to round coordinates make this a routine case, not a corner one.

**`eps` has a cliff at half the minimum width, and it is silent.** Below it the
band is accurate; at or past it the negative size erases the conductor and the
whole line counts as band. Measured on 0.5-um-pitch lines: exact at
`eps = 0.05 um`, **-38 %** at `eps = 0.2 um`. Nothing in the output indicates
it. The generated deck carries an `INTERNAL` guard per layer that flags any
geometry narrower than the assumed minimum width, and `eps` is snapped to the
database unit -- dividing by a nominal `eps` that Calibre rounded costs up to
4 % on its own.

**The residual bias is exactly recoverable, for free.** An inside band loses
`eps^2` at every convex corner and gains it at every concave one, so

```
P = band_area/eps + eps * (n_convex - n_concave)
```

is exact on Manhattan geometry -- 0.0000 % against edge lengths on line arrays,
staircases and segmented patterns, including a case where the raw band was
5.7 % low. The corner counts come from the corner marker layers the same deck
already produces for their own sake, so the correction adds no runtime.
Implemented in `features/corners.py` and `calibre/ingest.py`, locked by
`tests/test_calibre_band.py`.

Two things stay out of SVRF deliberately. Gradients and cross-layer terms are
computed in Python, where the multi-scale tensor already exists. And
`line_end_density` gets pattern matching rather than stacked LENGTH/ANGLE
heuristics -- the measurement below shows why it cannot be derived from
perimeter at all.

Note that `STEP < WINDOW` produces overlapping windows. That is fine, but the
ingest records the ratio so the statistics layer reports effective sample size
rather than window count.

## Implementation notes

**Window-local perimeter clips edges, not polygons.** Intersecting the region
with a window and taking its perimeter counts the window cut as metal
boundary: a 30×10 µm bar cut at x = 15 µm reports 50 µm instead of the true
40 µm, a 25 % overstatement of the project's most important feature. Guarded by
`tests/test_perimeter_clipping.py`.

**Regions are materialised, not lazily bound.** A `db.Region` built directly
from a `RecursiveShapeIterator` stays tied to its `Layout` and silently empties
once that layout is collected — so `LayoutReader(path).region(spec)` written as
a one-liner returns zero-valued features with no error anywhere, depending on
GC timing. The reader copies shapes in with `insert()` instead. Guarded by a
regression test.

**Extraction runs a grid row at a time** against a pre-clipped strip. Against
the full-layer region it costs ~35× more (7.7 vs 0.22 ms/cell on the 2 mm test
die) and gets worse with die area.

**Tests are two-sided throughout.** Zahedmanesh & Vanstreels (2019) show a
stiff top metal group can *lower* the crack driving force beneath it via
elastic stress shielding, so the same feature may associate with failure in
opposite directions on different layers. Effect sizes are signed and reported
per layer, never pooled as magnitude.

## Layout

```
src/lamxsim/
  evidence.py            evidence classes (spec section 30) + mixing guard
  units.py               dbu <-> um, isolated at the layout boundary
  pipeline.py            thin slice: layout -> features -> labels -> association
  cli.py                 phase0 / thinslice / run
  layout/reader.py       KLayout reader; cached merged Regions and Edges
  layout/synth.py        section 25/26 pattern pairs + validation die
  features/grid.py       multi-scale physical grids
  features/geometry.py   metal_density (4A), perimeter_density (4C),
                         line_end_density (4D)
  features/orientation.py length-weighted orientation (4F)
  features/gradient.py   dQ_dx, dQ_dy, |grad Q| (5), boundary-ring handling
  features/crosslayer.py layer-pair and stack features (7, 8)
  features/corners.py    convex/concave classification, corner markers
  features/lineends.py   candidate line-end definitions and the scored choice
  calibre/svrf.py        SVRF deck generation for the marker -> DENSITY route
  calibre/ingest.py      Calibre RDB/CSV back onto the analysis grid
  labels/failure.py      failure import, value validation, die and mode identity
  labels/inspection.py   inspection footprint and control opportunity
  labels/package_context.py bump, pad, PI-opening and crackstop context
  study.py               study manifest: the semantics GDS does not contain
  report.py              results partitioned by what each row may claim
  labels/position.py     PACKAGE_POSITION features (section 9)
  labels/simulate.py     simulated labels, for validation only
  stats/univariate.py    effect size, AUC, PR-AUC, enrichment, effective N, CI
  stats/fdr.py           tiered Benjamini-Hochberg
  stats/permutation.py   Moran's I, autocorrelation range, block permutation
  stats/power.py         Phase 0 feasibility
  stats/cv.py            block, buffered-block and grouped folds + leakage report
  stats/baseline.py      regularised logistic regression, calibration, ΔAUC
  stats/ablation.py      nested feature families against the position baseline
  registration/          transform fitting, honest error, scale gating
```

## Deviations from the spec, and why

**Failure CSV requires `lot_id, wafer_id, die_x, die_y`.** The spec's example
carries only `sample_id`, but spec section 17 asks for held-out-die validation.
Grouping identity cannot be recovered after import, so it is required up front
rather than optional.

**`position_sigma_um` gates which scales are analysed.** The spec says not to
convert uncertain locations into exact points; this is the operational form of
that rule.

**FDR is applied within hypothesis tiers, not across all combinations.**
Correcting 9,576 tests together leaves nothing significant at realistic failure
counts. Tiers come from `references/feature_evidence_map.csv` and are set by
literature evidence, not by looking at results.

## Known gaps

Accurate as of the current commit. Verified against the code rather than
recalled.

**Package geometry is nearly complete.** Bump distance and radial/tangential
decomposition, the routing direction resolved against the bump radial
direction, PI-opening edge and corner distance, crackstop rail distance, pad
edge distance, and wide metal, slotting and declared dummy fill as their own
family. Not yet expressed: pad overlap fraction and PI-opening shape
descriptors.

**Deliberately not built.** Sections 4G-4I (connectivity, largest structures,
empty area) are `exploratory` tier with no direct delamination evidence, so
they are the lowest-value additions. Sections 19-23 (ConvNeXt, VLM, hotspot
generation, GDS back-annotation) stay closed until a real study shows
deterministic geometry carries signal. An effective-stiffness proxy was
evaluated and rejected as provably redundant -- see above.

## Reproducing the test result

`pyproject.toml` declares the bounds; `constraints-verified.txt` records the
versions the suite has actually been run against.

```bash
pip install -e . -c constraints-verified.txt && python -m pytest tests -q
```

A clean install used to fail at collection: the multivariate baseline and the
ablation import scikit-learn, which was never declared, and `ndarray.ptp()`
was removed in NumPy 2.0 — reached by every synthetic die, so one line
cascaded into dozens of errors. Both are fixed, and the numbers quoted here
belong to the pinned environment rather than to whatever a resolver picks.

## What GDS cannot answer

No amount of work in this repository recovers these, and the software records
them as study strata rather than fabricating proxies: elastic moduli, CTE and
their temperature dependence; interface adhesion and fracture toughness;
residual stress, cure and reflow history; package warpage, bump stiffness,
mould compound and underfill; the initial defect population; fabrication bias
relative to drawn geometry; and inspection sensitivity and false-negative rate.

If those vary across the samples in a study, an apparent geometry effect may be
standing in for the variation. The minimum remedy is to hold them fixed or
record them as strata -- which is a decision about the measurement campaign,
not something the analysis can settle.

**Still needed for the real-data path:** bump/C4, pad and PI-opening semantics
when those layers are available, plus a CLI workflow that applies the fitted
registration to a failure CSV before analysis. Every layout lever in Rabie et
al. (2018) is defined relative to die corners and bumps, so without bump context
the engine cannot reproduce effects the literature already documents. The
registration library now fits a transform and estimates uncertainty honestly,
but `lamxsim register` currently writes the fit report rather than a registered
failure table.
