# First-principles engineering guide

This document defines what lamXsim can learn from GDS, how the literature
constrains that analysis, what the current implementation covers, and what a
human engineer must supply before the output can support real work.

The scope is deliberately **GDS-first**, not physics-complete. Package finite
element models, material cards and process histories are not required inputs.
They remain part of the causal system, however, so the analysis must name them
as latent conditions instead of silently attributing their effects to layout.

## The engineering question

The defensible question is:

> Within a defined package, process, inspection method and population of dies,
> which deterministic GDS features are spatially associated with measured
> delamination after controlling for the package-position information available
> from the layout?

The current evidence does **not** justify the stronger questions:

> Does this GDS feature cause delamination? What is the absolute probability of
> failure on a new package? Will changing this feature pass qualification?

Those require controlled layout experiments, held-out hardware and/or a
mechanics model carrying the missing boundary conditions.

## The causal chain that must remain visible

The literature describes a chain, not a direct GDS-to-failure law:

```text
package geometry + material properties + thermal/process history
                              |
                              v
            bump-relative displacement and stress transfer
                              |
                              v
       layer-specific BEOL stiffness and local stress concentration
                              |
                              v
       crack driving force (ERR) versus interface fracture resistance
                              |
                              v
          crack initiation/propagation and measurement visibility
                              |
                              v
                    recorded failure locations
```

GDS observes part of the middle of this chain: metal/via geometry, their
spatial transitions, layer relationships and, when the relevant mask layers
are present and identified, bump/pad/PI/crackstop context. GDS does not contain
material properties, adhesion, residual stress, thermal history, initial
defects or inspection sensitivity.

The missing quantities are not reasons to abandon GDS analysis. They are
reasons to constrain the study population, stratify results and call the output
an association or engineering hypothesis rather than a universal risk law.

## Literature-to-code map

| Literature constraint | Physical interpretation | GDS-observable proxy | Current code coverage | Human handoff |
|---|---|---|---|---|
| Critical bumps are near die corners/farthest from die centre; BEOL stress concentrates near the PI opening. EMC thickness, underfill CTE and PI opening change ERR. [Li et al. 2023](https://doi.org/10.3390/mi14101953), [Li et al. 2025](https://doi.org/10.3390/mi16020121), Rabie 2018 | Layout is loaded through a package-position-dependent boundary condition. Local geometry cannot be interpreted independently of bump context. | Distance to die edge/corner/centre; bump-relative distance/orientation; pad and PI-opening geometry if those layers exist in the GDS | Die edge/corner/centre features exist, measured from the **declared** die outline rather than the geometry bounding box. Bump distance, radial/tangential decomposition, local pitch and under-bump indicator are extracted, as are PI-opening edge and corner distance, crackstop rail distance and pad edge distance. Routing orientation is resolved against the bump radial direction, so the diagonal final-metal lever has a feature of its own. Bump, pad and PI-opening shape are extracted per object before any gridding -- area, equivalent diameter, Feret widths, aspect ratio from the minimum-area rotated rectangle, circularity, plan-view corner angles, principal axis (or the reason it is undefined), placement angle -- with pad/bump and PI/pad matching under a declared rule, their overlap fraction and their offsets resolved into the radial frame. `package_objects.csv` carries one row per object. The crackstop is measured as a structure: narrowest rail width, rail count, continuity, gap count, and the same resolved at each die corner. All of it is drawn plan-view geometry: not the post-reflow bump, not the printed opening, not the assembled overlay, and no sidewall or taper angle, which a layout cannot contain at all. What is still missing is explicit corner-tile morphology, a scored bump-geometry channel and the connectivity graph inside a corner window; see `unimplemented_gds_observables.csv`. | Identify the true die outline and, where present, bump/pad/PI layers. Keep package construction, EMC, underfill and thermal condition fixed or record them as strata. |
| Failure is layer- and interface-specific; the 20 nm study found the largest ERR and observed delamination at a particular upper BEOL interface, with bottom interconnect interfaces more critical than sidewalls. [Li et al. 2023](https://doi.org/10.3390/mi14101953) | A two-dimensional failure coordinate is incomplete without the failed layer/interface and failure mode. | Per-layer features, ordered layer-pair features and failure type/layer labels | Pipeline APIs preserve layer identity and implement layer-pair features, signed and as magnitudes. The real CLI reads the ordered stack from the manifest. `failed_layer` and `failed_interface` are in the schema, a file mixing them is refused, and `--stratify-by` analyses each population separately and reports where their effects disagree in sign. The interface strings are free text, not a controlled vocabulary; a row whose stratifying value is missing is refused rather than bucketed, because an unknown interface is not another known interface. | Supply the ordered BEOL layer map and preserve FIB/SAM failure classification. Do not pool failures from different interfaces without an explicit model. |
| Metal and via density affect BEOL fracture, while a stiff top group can shield an underlying weak layer; the sign is not universally “denser is worse.” [Vanstreels 2020](https://doi.org/10.1016/j.microrel.2020.113825); [Zahedmanesh and Vanstreels 2019](https://doi.org/10.1016/j.mne.2018.12.001) | Effective stiffness and load transfer depend on which layer is dense and on its relationship to neighbouring layers. | Signed per-layer density effects; via area/count density; signed and magnitude layer-pair differences | Two-sided tests, signed effects and metal cross-layer features exist. Via area density, via count density and mean via area are extracted and carry the identity of the metal layer they sit under. | Identify metal/via layers and their physical order. Freeze pair selection before looking at failures. Interpret signs per layer, never as a pooled density rule. |
| Pattern perimeter can be more decisive than area density for patterned ULK delamination. [Yoo 2004](https://doi.org/10.1109/IITC.2004.1345761) | Equal metal area can create very different Cu/low-k boundary length and therefore different opportunity for interfacial damage. | Metal/dielectric perimeter per analysis area | Implemented with true clipped edges; synthetic same-density/different-perimeter tests protect the definition. Calibre band correction is also implemented. | Confirm the mask represents the physical interface of interest and that hierarchy/biasing/slotting semantics match the fabricated layer. |
| Terminated tips, corners and orientation are mechanistic hotspots rather than interchangeable versions of perimeter. [Tan 2008](https://doi.org/10.1557/JMR.2008.0222); [Rabie 2018](https://doi.org/10.1109/IITC.2018.8430440) | Local stress concentration and load direction depend on topology and orientation. | Line-end density, convex/concave corner density, length-weighted orientation | Line ends, orientation and convex/concave corner density are all in the pipeline. The line-end width cutoff comes from the manifest's PDK rules, applied per layer, and `min_width_um` opens the layer before detection. | Provide PDK-informed routing width versus strap/fill width. Review line-end detections on representative real clips. Wire and validate corner density before treating the ablation family as complete. |
| Layout transitions and top-to-underlying architecture can matter. [Rabie 2018](https://doi.org/10.1109/IITC.2018.8430440); [Vanstreels 2020](https://doi.org/10.1016/j.microrel.2020.113825) | Abrupt stiffness changes and layer mismatch may concentrate load even when either local value alone is ordinary. | Spatial gradients, adjacent-layer differences, top-versus-underlying summaries | Gradient and cross-layer functions are in the real pipeline, with the die-edge ring dropped so one-sided differences cannot manufacture a position effect. The real CLI configures the multi-layer stack from the manifest. | Supply layer order and mechanically meaningful layer pairs; reject post-hoc pair selection. Validate the gradient scale against registration accuracy and layout pitch. |

The bibliographic inventory and evidence tiers live in
[`references/README.md`](../references/README.md),
[`references/core_references.csv`](../references/core_references.csv) and
[`references/feature_evidence_map.csv`](../references/feature_evidence_map.csv).

## The GDS-only deliverable

Before any failure data exists, `lamxsim characterize` produces a literature
exposure atlas: seven channels, one per documented mechanism, each scoring
where this layout departs from the lever its paper describes. Every candidate
carries the citation, the exact GDS observable, and the package or material
quantities that would be needed to turn the departure into a driving force.

It sits at levels 1 and 3 of the four below -- deterministic geometry, and a
mechanistic hypothesis worth a cross-section -- and deliberately not at level
2, which requires measured failure, or level 4.

Channels are never summed. Combining them requires weights, the weights could
only come from the data this stage does not have, and the result would be the
arbitrary weighted probability spec section 1 forbids.

## What is a fact, an association and a hypothesis?

Every result should be read at one of four levels:

1. **Deterministic GDS fact** - for example, metal density is 0.42 in a 100 um
   window. This can be checked against KLayout/Calibre and does not depend on
   failure data.
2. **Statistical association** - failure-labelled cells have a different
   feature distribution in this study population, after the stated spatial
   correction. This is what the current pipeline can establish.
3. **Mechanistic engineering hypothesis** - literature supplies a plausible
   physical path connecting the GDS feature to crack driving force. This is a
   reason to prioritize a controlled experiment, not proof of causality.
4. **Qualified design rule or predictor** - the effect survives held-out dies,
   lots and relevant package/process strata, and a layout change improves
   hardware outcome. The repo does not currently establish this level.

The report must never promote a level-2 result to level 4 merely because its
p-value or AUC is strong.

## Human inputs that give GDS physical meaning

GDS layer numbers are identifiers, not engineering semantics. Before a real
run, the owner of the technology/test vehicle must provide or approve the
following study manifest.

### Layout semantics

- Top cell and explicit die outline. The bounding box of arbitrary top-cell
  geometry is not automatically the physical die boundary.
- Ordered metal stack, via layers and datatype purpose. State whether fill,
  slotting, seal ring, crackstop, redistribution and top pad metal are included.
- Which GDS geometry corresponds to the interface implicated by failure
  analysis. Drawn metal is only a proxy for fabricated geometry.
- PDK-informed minimum width, routing-line maximum width and the distinction
  between routing, dummy fill and power straps. Do not let the shortest edge in
  an arbitrary design silently define a physical line end.
- Bump/C4, pad and PI-opening layers when present. If they are absent from the
  delivered GDS, record that bump-relative confounding remains uncontrolled.

### Failure and inspection semantics

- Coordinate frame, registration fiducials and positional uncertainty.
- Inspection method, resolution and coverage. An unlabelled grid cell is a
  valid control only if it had a real opportunity to be inspected and called.
- Failure type, failed BEOL layer/interface, extent and confidence. Pool only
  modes that share a defensible mechanism.
- The sampling denominator: all inspected dies/cells, not only a CSV of positive
  findings. Case-only locations do not define failure probability.
- Lot, wafer, die and package/test-condition identity for held-out validation
  and stratification.

### Analysis commitments made before seeing results

- Primary feature families and exploratory families.
- Layer-pair set and spatial scales.
- Minimum trustworthy scale from registration/measurement uncertainty.
- Control definition, exclusion mask and inspection footprint.
- Held-out unit: die first, wafer/lot where data permit.
- Negative controls and falsification tests.

## First-principles gaps

The gaps fall into three different classes and should not be mixed together.

### A. Available from GDS in principle

These are the highest-value software tasks because they extend the stated input
without pretending to know hidden physics. Status verified against the code.

1. **Done.** Multi-layer and via configuration in `lamxsim run`, from the study
   manifest.
2. **Done.** Via area density, count density and mean via area, keyed to the
   metal layer they sit under.
3. **Done.** Convex/concave corner density, with hole rings classified from the
   metal side rather than from their own winding.
4. **Done.** Explicit die-outline and top-cell configuration, and the declared
   outline now sets the position origin. Geometry bounding box, declared die and
   inspection footprint are three separate frames.
5. **Done.** Bump, PI-opening, crackstop and pad context are extracted, with
   distances measured to the boundary of each shape. Wide metal, slotting and
   declared dummy fill are their own feature family.
6. **Done.** Bump-relative radial and tangential offsets, plus the routing
   direction resolved against the bump radial direction. Rabie's diagonal
   final-metal lever has a feature of its own -- `routing_diagonality` peaks
   at 45 degrees, where both the radial and tangential cases sit at zero.
7. **Done.** Line rules are applied per layer, and `min_width_um` opens the
   layer before detection so a cap narrower than the drawn minimum is not read
   as a tip.
8. **Done.** `lamxsim run` fits registration, applies it to the failure set,
   propagates the leave-one-out error into the scale gate and analyses only the
   scales that survive.
9. **Done.** `lamxsim run --ablation` fits the position baseline and the nested
   feature families under leave-one-die-out folds when more than one die is
   present, and says so when there is only one.
10. **Done.** Results are partitioned into supported, primary, confounders,
    exploratory, unsupported_scale, underpowered, not_spatially_corrected and
    not_traceable. `supported` is the findings; `primary` is the
    pre-specified hypothesis set that was corrected over and contains rows at
    q = 1 by construction. `best_features.csv` no longer exists.

### A'. Remaining, in the order they change a conclusion

1. PI-opening shape descriptors and pad overlap fraction.
3. `failed_layer` and `failed_interface` used to stratify rather than only to
   refuse pooling.
4. A traceability matrix from literature to mechanism to GDS observable to
   unidentifiable parameter to statistical test.

### B. Not recoverable from ordinary GDS

The software should record these as study metadata or strata, not fabricate
proxies and not label their absence a geometry-extraction bug.

- Elastic moduli, CTE, plasticity and temperature dependence of BEOL/package
  materials.
- Interface adhesion/fracture toughness and its process/aging dependence.
- Residual stress, cure history, reflow profile and thermal-cycle history.
- Package warpage, bump stiffness, EMC thickness and underfill properties.
- Initial defect population and crack geometry.
- Fabrication bias relative to drawn geometry.
- Inspection sensitivity, false-negative rate and destructive-analysis
  selection.

If these vary across samples, an apparent GDS effect may be a proxy for the
variation. The minimum remedy is to stratify or hold out those conditions.

### C. Evidence gaps that require human experiments

- The core synthetic tests prove that the code can recover a planted driver;
  they do not validate a feature against hardware.
- Several core papers are closed access. Feature definitions derived from their
  summaries should remain hypotheses until an engineer reviews the full paper.
- Results from one node, stack or package do not create a universal sign or
  threshold for another.
- Correlated GDS features require controlled pattern pairs or layout changes to
  distinguish mechanism from proxy.
- A useful design recommendation requires prospective or held-out validation,
  ideally with a layout split that changes the target feature while holding
  bump/package context constant.

## Acceptance gates for realistic engineering use

### Gate 1 - extraction is numerically credible

- Unit conversions, die bounds and layer maps are reviewed.
- Representative clips agree with an independent KLayout/Calibre measurement.
- Line ends/corners are visually audited against real routing, fill and straps.
- Same-density/different-perimeter and other constructed discrimination tests
  remain green.

### Gate 2 - labels are physically credible

- Registration error is measured, not asserted.
- Unsupported scales are excluded from primary conclusions.
- Controls come from inspected opportunity, not merely absence from a case CSV.
- Failure modes and interfaces are not inappropriately pooled.

### Gate 3 - association is not position/package leakage

- Die-corner and, when available, bump-relative baselines are included, along
  with every package/process condition declared a covariate.
- Significance is corrected from the within-die block permutation
  (`spatial_q_value`), not from a test that assumes independent cells.
- The permutation count can resolve the correction: with a family of m tests
  the smallest reachable q is `m / (n_permutations + 1)`, so 999 permutations
  cannot clear alpha on a family of 240.
- Spatial null tests and negative controls pass.
- Geometry adds information beyond the baseline under the same folds.

### Gate 4 - the result generalizes

- Entire dies are held out; wafer/lot/package conditions are held out or
  stratified where possible.
- Effect direction and useful scale are stable, with intervals, across groups.
- A result that only exists on one die remains a local diagnostic.

### Gate 5 - a design action is justified

- The proposed change is tied to a literature mechanism.
- A falsifying alternative is stated.
- The target feature can be changed without simultaneously changing the known
  package-position confounders.
- New hardware or an independent cohort confirms the improvement.

## Required traceability for every new feature

Any feature added to the repo should document:

1. the physical hypothesis;
2. the supporting paper and evidence type (observation, experiment, FEM or
   derived analogy);
3. the exact GDS observable and unit;
4. layer and scale semantics;
5. expected confounders;
6. a synthetic or geometric discrimination test;
7. a negative control or falsification condition;
8. whether it is implemented in library code, real CLI and report output;
9. whether it is primary, confounder or exploratory;
10. what additional human evidence would promote it from association to an
    engineering rule.

This traceability is the mechanism by which the repo can approach maximum
coverage of the information available in GDS without overstating what GDS can
prove. It is enforced: `collective.foundation`'s registry matches every reported feature
against `references/feature_evidence_map.csv`, and the run metadata names any
that have no entry or whose entry is incomplete.
