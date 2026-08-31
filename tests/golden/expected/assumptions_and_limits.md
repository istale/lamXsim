# What this atlas is, and what it is not

## What it is

Every feature map is a **deterministic geometry fact** -- checkable against KLayout or Calibre, independent of any failure data.

Every one of the 120 candidate records is a **mechanistic engineering hypothesis**: a location where this layout departs from a lever the literature documents, with the citation attached. It is a reason to look there first.

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

**wide_metal_slotting** -- 3 candidate(s), one-sided
  - mechanism: a continuous span of wide metal carries the stiffness mismatch across its whole extent; slotting breaks the span, and Rabie lists wide-metal slotting among the layout levers
  - references: rabie2018cpi
  - observable: wide-metal area fraction that is not slotted, from a morphological opening at the declared wide-metal width
  - note: Unslotted wide metal, not wide metal. Ranking wide-metal fraction alone would flag a correctly slotted plate exactly as hard as an unbroken one, which inverts the lever: slotting is the recommended state, so its presence must lower the score. Can co-fire with corner_metal_tiles on top-layer geometry near a die corner; that is one piece of geometry seen through two of Rabie's levers, not two independent observations.

**corner_metal_tiles** -- 0 candidate(s), one-sided
  - mechanism: corner metal tiling is the first of Rabie's die-corner levers: unbroken top metal at the die corner couples the package corner load straight into the stack
  - references: rabie2018cpi
  - observable: unslotted wide-metal fraction on the topmost metal layer, inside the die-corner region
  - note: Top layer only and corner only, because that is the lever as stated. Scored on every layer it would assert something about the layers beneath that the reference does not; scored die-wide it would be the wide_metal_slotting channel under a second citation.

**pad_geometry_departure** -- 0 candidate(s), one-sided
  - mechanism: pad geometry is one of Rabie's five layout levers; a pad that departs from the recommended shape, or that sits off the bump it carries, changes how the package load enters the stack at that site
  - references: rabie2018cpi
  - observable: departure of the drawn pad's plan-view corner angles from the declared target, and the pad-to-bump centroid offset
  - note: Departure from a declared target, not risk. The target is the manifest's, because nothing in a GDS says which pad shape a process recommends. Drawn geometry only: assembly overlay and the manufactured pad are not in a layout, so a concentric drawn pair says nothing about the assembled one. Where every pad is identical the ranking reports no candidate rather than picking among equals.

**pi_opening_shape** -- 0 candidate(s), two-sided
  - mechanism: Li et al. vary the PI opening directly and locate the critical BEOL stress at its edge, so the opening's size and elongation are levers in their own right, separately from how close a cell is to one
  - references: li2023beol_failure_locations, li2025beol_design_factors
  - observable: drawn opening area, equivalent diameter, aspect ratio and plan-view corner-angle departure
  - note: Two-sided: the studies vary the opening and report the response without fixing a direction that holds for every stack, so flagging only large openings would invent one. Plan view only. A sidewall or taper angle is not derivable from a GDS by any means -- there is no Z information in a layout -- and the manifest refuses to accept one under that name.

**crackstop_structure** -- 0 candidate(s), one-sided
  - mechanism: the crackstop lever Rabie reports is about the ring itself -- how wide it is, whether there are two, and whether it is continuous -- not about how far a cell is from it
  - references: rabie2018cpi
  - observable: the seal ring's local drawn width, per analysis window, measured where the ring actually runs
  - note: A local width map, because two coarser versions of this channel could not report anything at all. A whole-ring number broadcast to every cell has no variation to rank; a per-quadrant corner width puts a quarter of the die on one value, and a quarter of the cells tied sit at the 88th percentile, below the 95th the atlas selects at, however narrow that corner is. The width where the ring actually runs ranks ring against ring and points at the pinch. Inverted: narrow is the departure. Rail count, continuity, gap count and the per-corner figures are still extracted -- they compare die rather than locate within one, and they are in package_objects.csv for that. Distance to the crackstop is a different feature and stays separate.

**routing_in_bump_frame** -- 8 candidate(s), one-sided
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

- `via_architecture` on M7: none of ['via_density', 'via_count_density'] is available [M7 @ 100um]
- `via_architecture` on M7: none of ['via_density', 'via_count_density'] is available [M7 @ 250um]

## Declared gaps in the manifest

- no shape_targets.pad_corner_angle_deg: pad shape can be described but not scored against a recommendation, because nothing says which shape is recommended
- no shape_targets.pi_plan_view_corner_angle_deg: the same, for the PI opening. If the figure being matched is a sidewall or taper angle rather than a plan-view corner, it is not obtainable from a GDS at all -- a layout holds no Z information -- and no value here can stand in for it
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
- layer 207: `wide_metal_slotting`

There is deliberately no combined hotspot layer.

## The die frame

A die outline is declared in the manifest, so distance to a corner, offset from the die centre and bump radial direction are measured from a frame the manifest vouches for.

## Object-level shape, and what drawn geometry is

Bump, pad, PI-opening and crackstop descriptors are computed per object and only then projected onto the grid; `package_objects.csv` holds one row per object with its id, source layer, the definition each descriptor was computed with, and any doubt about which object it was matched to. A window mean would have destroyed all of that: two pads of equal area and opposite elongation produce the same mean.

All of it is **drawn** geometry in plan view. A GDS says what was drawn, not what was manufactured, so none of these is the post-reflow bump, the printed opening after lithography, or the assembled overlay -- a drawn pad concentric with its drawn bump says nothing about the assembled pair. No sidewall or taper angle is derivable at all: a layout holds no Z information, and the manifest refuses a sidewall angle offered under a plan-view key rather than silently treating one as the other.

The outermost bump ring is flagged as a geometric fact. Which bump carries the largest driving force depends on package loading and on the stiffness of everything above it, none of which is in a layout.

Descriptors are rounded to the database unit, and corner angles to 0.1 degrees. Below that there is no geometry, only the arithmetic of computing a centroid from snapped vertices -- left in, it gets ranked, and on a die of identical pads it manufactures a candidate out of rounding.

## Literature conditioning

- `corner_metal_tiles` is ranked **inside** the top 25% of `distance_to_nearest_corner` (low end), because that is the region its citation is about. Cells outside it are not ranked and cannot be candidates.
- `routing_in_bump_frame` is ranked **inside** the top 25% of `distance_to_nearest_corner` (low end), because that is the region its citation is about. Cells outside it are not ranked and cannot be candidates.
- `pi_opening_proximity` is ranked **inside** the top 25% of `nearest_bump_distance_from_die_center`, because that is the region its citation is about. Cells outside it are not ranked and cannot be candidates.

## Where the feature maps came from

All feature maps were extracted with KLayout from the GDS directly. `lamxsim characterize --features-from DIR` reads the density and count maps from a Calibre deck run instead; on the regression die the two paths produce the same candidate set.

## What would turn this into evidence

Measured failure locations in the same coordinate frame, with lot/wafer/die identity, a registration fiducial set, an inspected footprint, and the failed layer or interface. `unsupported_non_gds_physics.csv` lists the package and material quantities that no GDS contains and that a study must hold fixed, stratify, or measure. `unimplemented_gds_observables.csv` carries everything else this atlas does not cover, with a status on every row: `absent` when nothing of it is implemented, `partial` when a channel covers part of it -- the row names that channel and says which part is still missing -- and `not_recoverable` when no layout can supply it at all. The last are listed anyway, because each has a GDS-derived proxy nearby that is easy to mistake for it.

Run `lamxsim phase0` for how many failure sites the association analysis would need before it could say anything at all.
