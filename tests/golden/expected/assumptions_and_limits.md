# What this atlas is, and what it is not

## What it is

Every feature map is a **deterministic geometry fact** -- checkable against KLayout or Calibre, independent of any failure data.

Every one of the 137 candidate records is a **mechanistic engineering hypothesis**: a location where this layout departs from a lever the literature documents, with the citation attached. It is a reason to look there first.

## What it is not

- Not a statistical association. No measured failure was involved.
- Not a probability. Nothing here is calibrated, so candidates are ranked by percentile **within this die** and by nothing else. The same layout on another package or process would rank the same and mean something different.
- Not a design rule. That needs held-out hardware.
- Not a combined risk score. Channels are reported separately because combining them requires weights, and the weights could only come from data this study does not have. A location flagged on three channels is three records with three citations, not a score of three.

## Channels reported

**perimeter_at_matched_density** -- 28 candidate(s), one-sided
  - mechanism: Cu/low-k boundary length is the interface available for interfacial damage, and was more decisive than pattern density in a dummy-pattern CMP experiment
  - references: yoo2004perimeter
  - observable: perimeter density, with the part explained by metal density removed
  - note: The residual, not the raw value: Yoo's result is that perimeter beats density, so a channel reading perimeter directly would rank the densest regions and reproduce a density map.

**termination** -- 30 candidate(s), one-sided
  - mechanism: delamination observed at terminated tips and corners rather than along parallel comb lines
  - references: tan2008delamination
  - observable: line-end density and re-entrant corner density

**via_architecture** -- 14 candidate(s), two-sided
  - mechanism: metal and via density set the layer's effective stiffness; a stiff top group can shield the layer beneath it, so the sign is not universally 'denser is worse'
  - references: vanstreels2020beol, zahedmanesh2019metallization
  - observable: via area and count density
  - note: Two-sided because the shielding result denies a fixed direction.

**layout_transition** -- 20 candidate(s), one-sided
  - mechanism: an abrupt change in metallisation concentrates load even where either local value is ordinary
  - references: rabie2018cpi, vanstreels2020beol
  - observable: spatial gradient magnitude of density and perimeter

**cross_layer_mismatch** -- 16 candidate(s), two-sided
  - mechanism: BEOL architecture, not any single layer, correlates with observed fracture; the topmost group's cross-sectional metal area stood out
  - references: vanstreels2020beol, zahedmanesh2019metallization
  - observable: top-to-underlying density and orientation mismatch

**routing_in_bump_frame** -- 28 candidate(s), one-sided
  - mechanism: the package loads the layout through the bumps, and diagonal final metal under the corner bumps is one of the documented levers, so routing that is radial or tangential there is the departure
  - references: rabie2018cpi
  - observable: routing direction resolved against the bump radial direction
  - note: Inverted, not two-sided. Diagonality is already folded: radial and tangential both sit at 0 and diagonal at 1, so a two-sided rank would score Rabie's recommendation as the departure. Conditioned on die-corner proximity, because the recommendation is about the corner bumps.

**pi_opening_proximity** -- 1 candidate(s), one-sided
  - mechanism: the BEOL stress concentration sits near the PI opening of the bumps farthest from the die centre
  - references: rabie2018cpi, li2023beol_failure_locations, li2025beol_design_factors
  - observable: distance to the nearest PI-opening edge and corner, within the outermost-bump region
  - note: Scored die-wide: the opening is a package feature, the same for every metal layer, so scoring it per layer would report one candidate as several. Conditioned on the nearest bump being one of the outermost, which is where the 20 nm study places the global loading before it compares anything beneath.

## Channels that could not be scored

- `via_architecture` on M7: none of ['via_density', 'via_count_density'] is available

## Declared gaps in the manifest

- no fill_layers: dummy fill cannot be separated from functional geometry, so it contributes to every density and sets the shortest edge on the layer
- no layout_revision: nothing checks that every die analysed was built from the layout in this file
- 11 package/process condition(s) undeclared (emc_thickness_um, underfill_cte_ppm_k, underfill_modulus_gpa, underfill_tg_c...): none can be read from GDS, and if any varies across the study an apparent geometry effect may be standing in for it
- no registration fiducials: positional uncertainty must come from the failure file, and no scale can be certified from measured registration

## GDS overlay layers

- layer 200: `cross_layer_mismatch`
- layer 201: `layout_transition`
- layer 202: `perimeter_at_matched_density`
- layer 203: `pi_opening_proximity`
- layer 204: `routing_in_bump_frame`
- layer 205: `termination`
- layer 206: `via_architecture`

There is deliberately no combined hotspot layer.

## Where the feature maps came from

All feature maps were extracted with KLayout from the GDS directly. `lamxsim characterize --features-from DIR` reads the density and count maps from a Calibre deck run instead; on the regression die the two paths produce the same candidate set.

## What would turn this into evidence

Measured failure locations in the same coordinate frame, with lot/wafer/die identity, a registration fiducial set, an inspected footprint, and the failed layer or interface. `unsupported_non_gds_physics.csv` lists the package and material quantities that no GDS contains and that a study must hold fixed, stratify, or measure; `unimplemented_gds_observables.csv` lists the ones that are in the layout and simply are not extracted yet, which is a different kind of gap and a much cheaper one to close.

Run `lamxsim phase0` for how many failure sites the association analysis would need before it could say anything at all.
