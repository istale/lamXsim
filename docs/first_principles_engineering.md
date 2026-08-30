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
| Critical bumps are near die corners/farthest from die centre; BEOL stress concentrates near the PI opening. EMC thickness, underfill CTE and PI opening change ERR. [Li et al. 2023](https://doi.org/10.3390/mi14101953), [Li et al. 2025](https://doi.org/10.3390/mi16020121), Rabie 2018 | Layout is loaded through a package-position-dependent boundary condition. Local geometry cannot be interpreted independently of bump context. | Distance to die edge/corner/centre; bump-relative distance/orientation; pad and PI-opening geometry if those layers exist in the GDS | Die edge/corner/centre features exist, measured from the **declared** die outline rather than the geometry bounding box. Bump distance, radial/tangential decomposition, local pitch and under-bump indicator are extracted, as are PI-opening edge and corner distance, crackstop rail distance and pad edge distance. Routing orientation relative to the bump radial direction is not. | Identify the true die outline and, where present, bump/pad/PI layers. Keep package construction, EMC, underfill and thermal condition fixed or record them as strata. |
| Failure is layer- and interface-specific; the 20 nm study found the largest ERR and observed delamination at a particular upper BEOL interface, with bottom interconnect interfaces more critical than sidewalls. [Li et al. 2023](https://doi.org/10.3390/mi14101953) | A two-dimensional failure coordinate is incomplete without the failed layer/interface and failure mode. | Per-layer features, ordered layer-pair features and failure type/layer labels | Pipeline APIs preserve layer identity and implement layer-pair features, signed and as magnitudes. The real CLI reads the ordered stack from the manifest. `failed_layer` and `failed_interface` are in the schema and a file mixing them is refused; they are not yet used to stratify a single analysis. | Supply the ordered BEOL layer map and preserve FIB/SAM failure classification. Do not pool failures from different interfaces without an explicit model. |
| Metal and via density affect BEOL fracture, while a stiff top group can shield an underlying weak layer; the sign is not universally “denser is worse.” [Vanstreels 2020](https://doi.org/10.1016/j.microrel.2020.113825); [Zahedmanesh and Vanstreels 2019](https://doi.org/10.1016/j.mne.2018.12.001) | Effective stiffness and load transfer depend on which layer is dense and on its relationship to neighbouring layers. | Signed per-layer density effects; via area/count density; signed and magnitude layer-pair differences | Two-sided tests, signed effects and metal cross-layer features exist. Via area density, via count density and mean via area are extracted and carry the identity of the metal layer they sit under. | Identify metal/via layers and their physical order. Freeze pair selection before looking at failures. Interpret signs per layer, never as a pooled density rule. |
| Pattern perimeter can be more decisive than area density for patterned ULK delamination. [Yoo 2004](https://doi.org/10.1109/IITC.2004.1345761) | Equal metal area can create very different Cu/low-k boundary length and therefore different opportunity for interfacial damage. | Metal/dielectric perimeter per analysis area | Implemented with true clipped edges; synthetic same-density/different-perimeter tests protect the definition. Calibre band correction is also implemented. | Confirm the mask represents the physical interface of interest and that hierarchy/biasing/slotting semantics match the fabricated layer. |
| Terminated tips, corners and orientation are mechanistic hotspots rather than interchangeable versions of perimeter. [Tan 2008](https://doi.org/10.1557/JMR.2008.0222); [Rabie 2018](https://doi.org/10.1109/IITC.2018.8430440) | Local stress concentration and load direction depend on topology and orientation. | Line-end density, convex/concave corner density, length-weighted orientation | Line ends, orientation and convex/concave corner density are all in the pipeline. The line-end width cutoff comes from the manifest's PDK rules, though the CLI currently applies the widest rule in the stack to every layer rather than each layer's own. | Provide PDK-informed routing width versus strap/fill width. Review line-end detections on representative real clips. Wire and validate corner density before treating the ablation family as complete. |
| Layout transitions and top-to-underlying architecture can matter. [Rabie 2018](https://doi.org/10.1109/IITC.2018.8430440); [Vanstreels 2020](https://doi.org/10.1016/j.microrel.2020.113825) | Abrupt stiffness changes and layer mismatch may concentrate load even when either local value alone is ordinary. | Spatial gradients, adjacent-layer differences, top-versus-underlying summaries | Gradient and cross-layer functions are in the real pipeline, with the die-edge ring dropped so one-sided differences cannot manufacture a position effect. The real CLI configures the multi-layer stack from the manifest. | Supply layer order and mechanically meaningful layer pairs; reject post-hoc pair selection. Validate the gradient scale against registration accuracy and layout pitch. |

The bibliographic inventory and evidence tiers live in
[`references/README.md`](../references/README.md),
[`references/core_references.csv`](../references/core_references.csv) and
[`references/feature_evidence_map.csv`](../references/feature_evidence_map.csv).

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
5. **Partly.** Bump, PI-opening, crackstop and pad context are extracted, with
   distances measured to the boundary of each shape. Seal ring, slotting,
   dummy fill and wide-metal discontinuity are not feature families yet.
6. **Partly.** Bump-relative radial and tangential offsets exist. Routing
   *orientation* relative to the bump radial direction does not, so Rabie's
   diagonal final-metal lever cannot be tested directly.
7. **Partly.** The line-end width comes from the manifest rather than the
   shortest edge, but the CLI applies the widest rule in the stack to every
   layer, and `min_width_um` does not reach extraction.
8. **Done.** `lamxsim run` fits registration, applies it to the failure set,
   propagates the leave-one-out error into the scale gate and analyses only the
   scales that survive.
9. **Done.** `lamxsim run --ablation` fits the position baseline and the nested
   feature families under leave-one-die-out folds when more than one die is
   present, and says so when there is only one.
10. **Done.** Results are partitioned into primary, confounders, exploratory,
    unsupported_scale and underpowered. `best_features.csv` no longer exists.

### A'. Remaining, in the order they change a conclusion

1. Per-layer PDK line rules, and `min_width_um` reaching extraction (item 7
   above).
2. A failure outside the inspected footprint should stop the run rather than
   warn: it disproves the population definition.
3. Routing orientation relative to the bump radial direction, PI-opening shape
   descriptors, pad overlap fraction.
4. Seal ring, slotting, dummy fill and wide-metal discontinuity as families.
5. Per-die inspection footprints, and an explicit assertion that every die
   shares the layout revision being analysed.
6. `failed_layer` and `failed_interface` used to stratify rather than only to
   refuse pooling.
7. A traceability matrix from literature to mechanism to GDS observable to
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

- Die-corner and, when available, bump-relative baselines are included.
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
prove.
