"""Position, package context, failure files and inspection footprints.

Consolidated from ``labels/position.py``, ``labels/package_context.py``, ``labels/failure.py``, ``labels/inspection.py``, ``labels/simulate.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import klayout.db as db
import numpy as np
import pandas as pd
from .foundation import EvidenceClass
from .layout import BBox, LayerSpec, LayoutReader


# ----------------------------------------------------------------------
# labels/position.py
# ----------------------------------------------------------------------
"""Package-position features (spec section 9).

These are PACKAGE_POSITION evidence, deliberately not GDS_GEOMETRY. They
exist so that the position-only baseline model can be built, because
"geometry predicts delamination" is only meaningful as a claim relative to
"die position already predicts delamination".
"""
POSITION_FEATURES = (
    "distance_to_die_edge",
    "distance_to_nearest_corner",
    "normalized_distance_from_die_center",
)
EVIDENCE_CLASS = EvidenceClass.PACKAGE_POSITION


def position_extract(grid, die_bbox) -> dict[str, np.ndarray]:
    x = np.array([c.x_center for c in grid.cells])
    y = np.array([c.y_center for c in grid.cells])
    b = die_bbox

    d_edge = np.minimum.reduce([x - b.xmin, b.xmax - x, y - b.ymin, b.ymax - y])

    corners = [(b.xmin, b.ymin), (b.xmin, b.ymax), (b.xmax, b.ymin), (b.xmax, b.ymax)]
    d_corner = np.min([np.hypot(x - cx, y - cy) for cx, cy in corners], axis=0)

    mx, my = (b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2
    half = np.hypot(b.width / 2, b.height / 2)
    d_center = np.hypot(x - mx, y - my) / max(half, 1e-12)

    return {
        "distance_to_die_edge": d_edge,
        "distance_to_nearest_corner": d_corner,
        "normalized_distance_from_die_center": d_center,
    }

# ----------------------------------------------------------------------
# labels/package_context.py
# ----------------------------------------------------------------------
"""Bump, pad and PI-opening context (spec section 9, extended).

Every layout lever Rabie et al. (2018) report is defined relative to die
corners and bumps: corner metal tiles, diagonal final-metal routing *under
corner bumps*, pad geometry, the PI opening angle, crackstop width. Li et al.
(2023, 2025) locate the critical BEOL stress near the PI opening of the bumps
farthest from the die centre.

So bump context is not an optional refinement. It is the boundary condition
through which the package loads the layout, and without it any association
found for a layout feature may simply be that feature's correlation with
bump proximity.

These are **PACKAGE_POSITION** evidence, not GDS_GEOMETRY, even though they
are extracted from GDS layers. They belong in the position-only baseline that
every geometry model is scored against; putting them in the geometry model
would let the baseline be beaten by the confounder it exists to control.

When the delivered layout has no bump layer, `absent_context_note` states
plainly that bump-relative confounding is uncontrolled, so the omission
appears in the run metadata instead of being forgotten.
"""
PACKAGE_CONTEXT_FEATURES = (
    "distance_to_nearest_bump",
    "bump_radial_offset",
    "bump_tangential_offset",
    "bump_radial_direction_rad",
    "nearest_bump_distance_from_die_center",
    "local_bump_pitch",
    "bump_count_density",
    "under_bump_indicator",
    "distance_to_nearest_pi_opening",
    "distance_to_pi_opening_corner",
    "distance_to_crackstop",
    "distance_to_pad_edge",
    "under_pad_indicator",
)


@dataclass
class PackageLayers:
    """Which GDS layers carry package context, when they are delivered at all."""
    bump: LayerSpec | None = None
    pad: LayerSpec | None = None
    pi_opening: LayerSpec | None = None
    crackstop: LayerSpec | None = None

    @property
    def any_present(self) -> bool:
        return any(v is not None for v in
                   (self.bump, self.pad, self.pi_opening, self.crackstop))


def _edge_segments(reader: LayoutReader, spec: LayerSpec | None) -> np.ndarray:
    """Boundary segments of a layer, as (x0, y0, x1, y1) rows in um.

    Distances to a PI opening or a crackstop have to be measured to the shape,
    not to a bounding-box centre. A crackstop is a ring around the die: its
    bounding box is the die, so a centre-based distance to it is numerically
    identical to distance-to-die-centre -- a feature that already exists,
    arriving under a name that suggests crackstop proximity was measured.
    """
    if spec is None:
        return np.empty((0, 4))
    edges = reader.edges(spec)
    dbu = reader.units.dbu
    rows = [(e.x1 * dbu, e.y1 * dbu, e.x2 * dbu, e.y2 * dbu)
            for e in edges.each()]
    return np.array(rows, dtype=float).reshape(-1, 4)


def _distance_to_segments(x: np.ndarray, y: np.ndarray, seg: np.ndarray,
                          block: int = 2048) -> np.ndarray:
    """Shortest distance from each point to any boundary segment."""
    if len(seg) == 0:
        return np.full(len(x), np.nan)
    ax, ay, bx, by = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    length2 = np.where(length2 > 0, length2, 1.0)

    out = np.empty(len(x))
    for s in range(0, len(x), block):
        e = min(s + block, len(x))
        px = x[s:e, None] - ax[None, :]
        py = y[s:e, None] - ay[None, :]
        t = np.clip((px * dx[None, :] + py * dy[None, :]) / length2[None, :], 0.0, 1.0)
        out[s:e] = np.hypot(px - t * dx[None, :], py - t * dy[None, :]).min(axis=1)
    return out


def _centroids(reader: LayoutReader, spec: LayerSpec | None) -> np.ndarray:
    if spec is None:
        return np.empty((0, 2))
    region = reader.region(spec)
    dbu = reader.units.dbu
    pts = []
    for poly in region.each():
        b = poly.bbox()
        pts.append(((b.left + b.right) / 2 * dbu, (b.bottom + b.top) / 2 * dbu))
    return np.array(pts, dtype=float).reshape(-1, 2)


def _nearest_distance(x, y, targets: np.ndarray, block: int = 4096) -> np.ndarray:
    if len(targets) == 0:
        return np.full(len(x), np.nan)
    out = np.empty(len(x))
    for s in range(0, len(x), block):
        e = min(s + block, len(x))
        d = np.hypot(x[s:e, None] - targets[None, :, 0],
                     y[s:e, None] - targets[None, :, 1])
        out[s:e] = d.min(axis=1)
    return out


def _nearest_index(x, y, targets: np.ndarray, block: int = 4096) -> np.ndarray:
    out = np.empty(len(x), dtype=int)
    for s in range(0, len(x), block):
        e = min(s + block, len(x))
        d = np.hypot(x[s:e, None] - targets[None, :, 0],
                     y[s:e, None] - targets[None, :, 1])
        out[s:e] = d.argmin(axis=1)
    return out


def package_context_extract(grid, die_bbox: BBox, reader: LayoutReader,
            layers: PackageLayers) -> dict[str, np.ndarray]:
    """Bump-relative and package-feature distances for every grid cell.

    ``bump_radial_offset`` and ``bump_tangential_offset`` resolve the cell's
    displacement from its nearest bump into components along and across the
    die-centre direction. Rabie's diagonal final-metal recommendation is
    directional, so a scalar distance alone cannot express it.
    """
    x = np.array([c.x_center for c in grid.cells])
    y = np.array([c.y_center for c in grid.cells])
    n = len(grid)
    out: dict[str, np.ndarray] = {}

    bumps = _centroids(reader, layers.bump)
    if len(bumps):
        idx = _nearest_index(x, y, bumps)
        bx, by = bumps[idx, 0], bumps[idx, 1]
        dx, dy = x - bx, y - by
        out["distance_to_nearest_bump"] = np.hypot(dx, dy)

        # Radial direction points away from the die centre, which is the
        # direction package-induced shear grows along.
        mx = (die_bbox.xmin + die_bbox.xmax) / 2
        my = (die_bbox.ymin + die_bbox.ymax) / 2
        rx, ry = bx - mx, by - my
        norm = np.hypot(rx, ry)
        safe = norm > 0
        ux = np.where(safe, rx / np.where(safe, norm, 1.0), 1.0)
        uy = np.where(safe, ry / np.where(safe, norm, 1.0), 0.0)
        out["bump_radial_offset"] = dx * ux + dy * uy
        out["bump_tangential_offset"] = -dx * uy + dy * ux
        # The direction package-induced shear grows along at this location,
        # kept as an angle so layout orientation can be resolved against it.
        out["bump_radial_direction_rad"] = np.arctan2(uy, ux) % np.pi
        # How far out this cell's nearest bump sits. Li et al. (2023) locate
        # the global loading at the bumps farthest from the die centre and
        # only then compare the layers beneath them, so "which bump" has to be
        # available before that conditioning can be applied.
        out["nearest_bump_distance_from_die_center"] = norm

        # Local pitch from the nearest-neighbour spacing of the bump this cell
        # belongs to, so a die with a non-uniform bump map is described
        # correctly rather than by one global pitch.
        out["local_bump_pitch"] = _bump_pitch(bumps)[idx]
        # Counted within one local bump pitch rather than within the analysis
        # window: at any scale finer than the bump pitch a window-sized radius
        # contains no bump at all, and the feature would be identically zero.
        radius = float(np.nanmedian(_bump_pitch(bumps))) if len(bumps) > 1 \
            else grid.scale_um
        out["bump_count_density"] = _count_within(x, y, bumps, radius) / (
            np.pi * radius ** 2)
        out["under_bump_indicator"] = _inside_any(
            x, y, reader, layers.bump).astype(float)
    else:
        for k in ("distance_to_nearest_bump", "bump_radial_offset",
                  "bump_tangential_offset", "bump_radial_direction_rad",
                  "nearest_bump_distance_from_die_center",
                  "local_bump_pitch", "bump_count_density",
                  "under_bump_indicator"):
            out[k] = np.full(n, np.nan)

    # Measured to the boundary of the shape. Li et al. (2023, 2025) locate the
    # BEOL stress concentration at the PI *opening edge*, not at the centre of
    # the opening, and a crackstop is a rail whose distance is meaningful only
    # to the rail itself.
    out["distance_to_nearest_pi_opening"] = _distance_to_segments(
        x, y, _edge_segments(reader, layers.pi_opening))
    out["distance_to_pi_opening_corner"] = _nearest_distance(
        x, y, _corner_points(reader, layers.pi_opening))
    out["distance_to_crackstop"] = _distance_to_segments(
        x, y, _edge_segments(reader, layers.crackstop))
    out["distance_to_pad_edge"] = _distance_to_segments(
        x, y, _edge_segments(reader, layers.pad))
    out["under_pad_indicator"] = _inside_any(
        x, y, reader, layers.pad).astype(float)
    return out


def _corner_points(reader: LayoutReader, spec: LayerSpec | None) -> np.ndarray:
    """Convex corners of a layer, in um.

    An opening's corners concentrate stress differently from its straight
    edges, which is why the corner distance is kept separate from the edge
    distance rather than folded into it.
    """
    if spec is None:
        return np.empty((0, 2))
    from .geometry import classify
    convex, _ = classify(reader.region(spec))
    d = reader.units.dbu
    return np.array([[p.x * d, p.y * d] for p in convex], float).reshape(-1, 2)


def _bump_pitch(bumps: np.ndarray) -> np.ndarray:
    """Nearest-neighbour spacing for each bump."""
    if len(bumps) < 2:
        return np.full(len(bumps), np.nan)
    d = np.hypot(bumps[:, None, 0] - bumps[None, :, 0],
                 bumps[:, None, 1] - bumps[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def _count_within(x, y, targets: np.ndarray, radius: float,
                  block: int = 4096) -> np.ndarray:
    out = np.zeros(len(x))
    for s in range(0, len(x), block):
        e = min(s + block, len(x))
        d = np.hypot(x[s:e, None] - targets[None, :, 0],
                     y[s:e, None] - targets[None, :, 1])
        out[s:e] = (d <= radius).sum(axis=1)
    return out


def _inside_any(x, y, reader: LayoutReader, spec: LayerSpec | None) -> np.ndarray:
    """Whether each cell centre falls inside a shape on *spec*."""
    if spec is None:
        return np.zeros(len(x), dtype=bool)
    import klayout.db as db
    region = reader.region(spec)
    u = reader.units
    out = np.zeros(len(x), dtype=bool)
    # A zero-area box intersects nothing, so the probe has area; it is centred
    # on the point so that a coordinate exactly on a shape boundary is not
    # reported as outside it.
    for i, (xi, yi) in enumerate(zip(x, y)):
        px, py = u.um_to_dbu(xi), u.um_to_dbu(yi)
        probe = db.Region(db.Box(px - 1, py - 1, px + 1, py + 1))
        out[i] = not (region & probe).is_empty()
    return out


def absent_context_note(layers: PackageLayers) -> list[str]:
    """Record uncontrolled confounding rather than letting it go unmentioned."""
    notes = []
    if layers.bump is None:
        notes.append(
            "no bump/C4 layer supplied: bump-relative confounding is "
            "UNCONTROLLED. Every layout lever reported by Rabie et al. (2018) "
            "is defined relative to bumps, so an association found here may be "
            "that feature's correlation with bump proximity.")
    if layers.pi_opening is None:
        notes.append(
            "no PI-opening layer supplied: the location Li et al. (2023, 2025) "
            "identify as the BEOL stress concentration is not represented.")
    if layers.crackstop is None:
        notes.append("no crackstop layer supplied: die-edge structure is not "
                     "distinguished from ordinary routing.")
    if layers.pad is None:
        notes.append("no pad layer supplied: pad geometry, which Rabie et al. "
                     "(2018) list among the layout levers, is not represented.")
    return notes


#: Per-object shape features, produced by the object table rather than by a
#: window scan. The distinction matters: a pad's aspect ratio belongs to the
#: pad, and a window mean over several pads is a different quantity that
#: cannot be inverted back into any of them.
SHAPE_FEATURES = (
    "bump_object_count", "bump_area_um2", "bump_equivalent_diameter_um",
    "bump_aspect_ratio", "bump_placement_angle_rad", "bump_circularity",
    "bump_is_outermost",
    "pad_object_count", "pad_area_um2", "pad_aspect_ratio",
    "pad_circularity", "pad_corner_angle_departure_deg",
    "pad_target_corner_fraction", "pad_bump_centroid_offset_um",
    "pad_bump_radial_offset_um", "pad_bump_overlap_fraction",
    "pi_object_count", "pi_area_um2", "pi_equivalent_diameter_um",
    "pi_aspect_ratio", "pi_principal_axis_rad",
    "pi_corner_angle_departure_deg", "pi_pad_centroid_offset_um",
    "crackstop_rail_width_min_um", "crackstop_rail_count",
    "crackstop_continuity_ratio", "crackstop_n_gaps",
    "crackstop_corner_narrowest_um", "crackstop_corner_asymmetry",
    "crackstop_local_width_um", "crackstop_local_gap_um",
)


def object_table(reader: LayoutReader, layers: PackageLayers,
                 semantics, die_bbox: BBox | None):
    """Every package object, described individually, before any gridding.

    Returned as objects and matches rather than as arrays, so a caller can
    write the table out: an object's id, its source layer, the definition each
    descriptor was computed with, and any matching ambiguity all survive to
    the report. Once these are averaged into windows none of that is
    recoverable.
    """
    from . import objects as obj

    kinds = {"bump": layers.bump, "pad": layers.pad,
             "pi_opening": layers.pi_opening}
    table, polys = {}, {}
    for kind, spec in kinds.items():
        polarity = semantics.polarity_of(kind)
        table[kind] = obj.objects_for(reader, spec, kind=kind,
                                      polarity=polarity, die_bbox=die_bbox)
        polys[kind] = (list(obj.region_for(reader, spec, polarity).each())
                       if spec is not None else [])
    # The crackstop is a structure rather than an array of objects. Its
    # polygons are described like any other, and the structure facts -- rail
    # count, continuity, gap count, per-corner width -- are returned beside
    # them so the writer can attach them. Returning only the polygons put a
    # crackstop row in the table with none of the numbers the channel and the
    # docs said were there.
    table["crackstop"] = obj.objects_for(
        reader, layers.crackstop, kind="crackstop",
        polarity=semantics.polarity_of("crackstop"), die_bbox=die_bbox)
    structure = obj.crackstop_structure(reader, layers.crackstop, die_bbox)
    topology = obj.corner_topology(reader, layers.crackstop, die_bbox)

    matches = {}
    rule = semantics.object_matching
    tol = semantics.match_tolerance_um
    for primary, secondary in (("pad", "bump"), ("pi_opening", "pad")):
        # The polygons themselves, not the layer regions: containment is
        # polygon containment, and an overlap fraction taken against a box
        # around the primary lets a neighbouring pad into the denominator.
        both = (polys[primary], polys[secondary])
        matches[(primary, secondary)] = obj.match(
            table[primary], table[secondary], rule=rule, die_bbox=die_bbox,
            tolerance_um=tol,
            polygons=both if all(both) else None, dbu=reader.units.dbu)
    return table, matches, {"structure": structure, "corner_topology": topology}


def _outermost_flag(bumps, tolerance_um: float = 1.0) -> np.ndarray:
    """1.0 for the bumps at the greatest radius, within a tolerance.

    Li et al. place the global loading at the bumps farthest from the die
    centre, so the outermost ring is worth naming. It is named as a
    *geometric* fact and nothing more: which bump is mechanically critical
    depends on the package loading and the stiffness of everything above it,
    none of which is in a layout. Ties are kept as ties -- a bump array has a
    whole outer ring at the same radius, and picking one of them would be an
    artefact of ordering.
    """
    if not bumps:
        return np.empty(0)
    radii = np.array([b.radial_distance_um for b in bumps])
    if not np.isfinite(radii).any():
        return np.full(len(bumps), np.nan)
    return (radii >= np.nanmax(radii) - tolerance_um).astype(float)


def extract_shapes(grid, die_bbox: BBox | None, reader: LayoutReader,
                   layers: PackageLayers, semantics) -> dict[str, np.ndarray]:
    """Object-level shape descriptors, rasterised onto *grid*.

    Cells with no object of a kind get NaN, not zero: "no pad here" and "a pad
    with an aspect ratio of zero" are different statements, and only the
    second belongs in a ranking of pads.
    """
    from . import objects as obj

    n = len(grid)
    # Target-dependent features are left absent, not filled with NaN, when no
    # target is declared. An all-NaN column makes the channel look available
    # and then quietly rank nothing; an absent one makes it say which input it
    # wanted and did not get.
    target_dependent = {"pad_corner_angle_departure_deg",
                        "pad_target_corner_fraction",
                        "pi_corner_angle_departure_deg"}
    declared = set()
    if semantics.pad_corner_angle_deg is not None:
        declared |= {"pad_corner_angle_departure_deg",
                     "pad_target_corner_fraction"}
    if semantics.pi_plan_view_corner_angle_deg is not None:
        declared.add("pi_corner_angle_departure_deg")
    out = {name: np.full(n, np.nan) for name in SHAPE_FEATURES
           if name not in target_dependent or name in declared}
    if not layers.any_present:
        return out

    table, matches, _ = object_table(reader, layers, semantics, die_bbox)

    bumps = table["bump"]
    if bumps:
        outermost = _outermost_flag(bumps)
        out.update(obj.rasterise(bumps, grid, {
            "area_um2": np.array([b.area_um2 for b in bumps]),
            "equivalent_diameter_um": np.array(
                [b.equivalent_diameter_um for b in bumps]),
            "aspect_ratio": np.array([b.aspect_ratio for b in bumps]),
            "placement_angle_rad": np.array(
                [b.placement_angle_rad for b in bumps]),
            "circularity": np.array([b.circularity for b in bumps]),
            "is_outermost": outermost,
        }, prefix="bump"))
        out["bump_object_count"] = out.pop("bump_count")

    pads = table["pad"]
    if pads:
        target = semantics.pad_corner_angle_deg
        tol = semantics.corner_angle_tolerance_deg
        values = {
            "area_um2": np.array([p.area_um2 for p in pads]),
            "aspect_ratio": np.array([p.aspect_ratio for p in pads]),
            "circularity": np.array([p.circularity for p in pads]),
        }
        if target is not None:
            values["corner_angle_departure_deg"] = np.array(
                [obj.corner_angle_departure(p, target, tolerance_deg=tol)
                 for p in pads])
            values["target_corner_fraction"] = np.array(
                [obj.target_corner_fraction(p, target, tolerance_deg=tol)
                 for p in pads])
        by_id = {p.object_id: i for i, p in enumerate(pads)}
        for name, attr in (("bump_centroid_offset_um", "centroid_offset_um"),
                           ("bump_radial_offset_um", "radial_offset_um"),
                           ("bump_overlap_fraction", "overlap_fraction")):
            series = np.full(len(pads), np.nan)
            for m in matches[("pad", "bump")]:
                series[by_id[m.primary_id]] = getattr(m, attr)
            values[name] = series
        out.update(obj.rasterise(pads, grid, values, prefix="pad"))
        out["pad_object_count"] = out.pop("pad_count")

    openings = table["pi_opening"]
    if openings:
        target = semantics.pi_plan_view_corner_angle_deg
        tol = semantics.corner_angle_tolerance_deg
        values = {
            "area_um2": np.array([o.area_um2 for o in openings]),
            "equivalent_diameter_um": np.array(
                [o.equivalent_diameter_um for o in openings]),
            "aspect_ratio": np.array([o.aspect_ratio for o in openings]),
            "principal_axis_rad": np.array(
                [o.principal_axis_rad for o in openings]),
        }
        if target is not None:
            values["corner_angle_departure_deg"] = np.array(
                [obj.corner_angle_departure(o, target, tolerance_deg=tol)
                 for o in openings])
        by_id = {o.object_id: i for i, o in enumerate(openings)}
        series = np.full(len(openings), np.nan)
        for m in matches[("pi_opening", "pad")]:
            series[by_id[m.primary_id]] = m.centroid_offset_um
        values["pad_centroid_offset_um"] = series
        out.update(obj.rasterise(openings, grid, values, prefix="pi"))
        out["pi_object_count"] = out.pop("pi_count")

    structure = obj.crackstop_structure(reader, layers.crackstop, die_bbox)
    if structure is not None:
        # Whole-ring facts, broadcast. They cannot be ranked within one die --
        # every cell carries the same value -- and they are not meant to be:
        # they are the right shape for comparing die, and they travel in the
        # feature maps and in package_objects.csv for that.
        for name, value in (
                ("crackstop_rail_width_min_um", structure.rail_width_min_um),
                ("crackstop_rail_count", float(structure.n_rails)),
                ("crackstop_continuity_ratio", structure.continuity_ratio),
                ("crackstop_n_gaps", float(structure.n_gaps))):
            out[name] = np.full(n, value, dtype=float)

    # Corner-resolved figures, reported per cell of the quadrant they belong
    # to. These compare die rather than locate within one -- a quarter of the
    # cells sharing a value sits at the 88th percentile whatever the value is
    # -- so no channel ranks them; they travel as features.
    topology = obj.corner_topology(reader, layers.crackstop, die_bbox)
    if topology and not topology["undefined_reason"]:
        per_corner = topology["per_corner"]
        mx = (die_bbox.xmin + die_bbox.xmax) / 2 if die_bbox else 0.0
        my = (die_bbox.ymin + die_bbox.ymax) / 2 if die_bbox else 0.0
        narrowest = np.full(n, np.nan)
        for cell in grid.cells:
            key = ("l" if cell.y_center < my else "u") + \
                  ("l" if cell.x_center < mx else "r")
            narrowest[cell.cell_id] = per_corner[key]["narrowest_um"]
        out["crackstop_corner_narrowest_um"] = narrowest
        out["crackstop_corner_asymmetry"] = np.full(
            n, topology["corner_asymmetry"], dtype=float)

    # The local width where the ring actually runs. This is the one that can
    # be ranked within a die: NaN off the ring, so the comparison is ring
    # against ring, and a pinch is located where it is.
    local = obj.crackstop_width_map(reader, layers.crackstop, grid)
    if local is not None:
        out["crackstop_local_width_um"] = local

    # And where it is interrupted. A break cannot be found by the width map:
    # where the ring is absent there is nothing to measure, the cell is NaN,
    # and NaN is not an extreme -- so a cut ring produced whole-ring numbers
    # and no locatable candidate.
    gaps = obj.crackstop_gap_map(reader, layers.crackstop, grid, support=local)
    if gaps is not None:
        out["crackstop_local_gap_um"] = gaps
    return out

# ----------------------------------------------------------------------
# labels/failure.py
# ----------------------------------------------------------------------
"""Measured failure import and mapping (spec section 10).

Failure data is MEASURED_FAILURE evidence and is kept in its own table with
its own coordinate frame and its own uncertainty. It is joined to grid cells
only through :func:`map_to_grid`, which never converts an uncertain location
into an exact one -- the reported ``position_sigma_um`` travels with every
record and gates which analysis scales are admissible.
"""
#: Columns required by spec section 2, extended for the grouped-split needs of
#: spec section 17. lot/wafer/die identity cannot be recovered later, so it is
#: required at import rather than optional.
REQUIRED_COLUMNS = ("sample_id", "x_um", "y_um", "failure_type")
GROUPING_COLUMNS = ("lot_id", "wafer_id", "die_x", "die_y")
OPTIONAL_COLUMNS = ("confidence", "position_sigma_um", "extent_um", "coord_frame",
                    "failed_layer", "failed_interface", "layout_revision")

#: Columns whose distinct values define separate failure populations. Li et al.
#: (2023) found the largest energy release rate at one particular upper BEOL
#: interface, with bottom interconnect interfaces more critical than sidewalls,
#: so two failures at the same (x, y) on different interfaces are not two
#: observations of the same thing.
MODE_COLUMNS = ("failure_type", "failed_layer", "failed_interface")


@dataclass
class FailureSet:
    """A set of measured (or simulated) failure locations."""
    table: pd.DataFrame
    evidence_class: EvidenceClass = EvidenceClass.MEASURED_FAILURE
    simulated: bool = False
    source: str = ""
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.table)

    @property
    def position_sigma_um(self) -> float:
        """Worst-case reported positional uncertainty, in um."""
        if "position_sigma_um" not in self.table:
            return float("nan")
        sigma = float(self.table["position_sigma_um"].max())
        if np.isfinite(sigma) and sigma < 0:
            raise ValueError(
                f"position_sigma_um is negative ({sigma}); a negative "
                "uncertainty would certify every analysis scale")
        return sigma

    def die_keys(self) -> "pd.Series":
        """One identifier per physical die, from lot/wafer/die coordinates."""
        cols = [c for c in GROUPING_COLUMNS if c in self.table]
        if not cols:
            return pd.Series(["<unknown>"] * len(self.table),
                             index=self.table.index)
        return (self.table[cols].astype(str)
                .agg("|".join, axis=1).rename("die_key"))

    def n_dies(self) -> int:
        return int(self.die_keys().nunique())

    def layout_revisions(self) -> list[str]:
        """Distinct layout revisions the failures claim to come from."""
        if "layout_revision" not in self.table:
            return []
        return sorted(self.table["layout_revision"].dropna().astype(str).unique())

    def assert_single_layout_revision(self, declared: str | None) -> list[str]:
        """Refuse failures drawn from a layout other than the one being analysed.

        Every feature is extracted from one GDS, and every die's labels are
        mapped onto that one layout. If a lot spans a revision, some failures
        are being scored against geometry that was not on the silicon they
        came from -- which is not a small error, because a revision usually
        changes exactly the metal the study is about.
        """
        found = self.layout_revisions()
        if not found:
            return ["failures carry no layout_revision: the assumption that "
                    "every die shares the layout being analysed is unverified"]
        if len(found) > 1:
            raise ValueError(
                f"the failure set spans layout revisions {found}. Features "
                "come from one GDS, so failures from another revision would be "
                "scored against geometry that was not on their silicon. Split "
                "the study by revision.")
        if declared is not None and found[0] != declared:
            raise ValueError(
                f"the failures are from layout revision {found[0]!r} but the "
                f"manifest declares {declared!r}.")
        return []

    def modes(self) -> dict[str, list]:
        """Distinct values of every column that defines a failure population."""
        out = {}
        for col in MODE_COLUMNS:
            if col in self.table:
                vals = sorted(self.table[col].dropna().astype(str).unique())
                if vals:
                    out[col] = vals
        return out

    def assert_single_mode(self, *, allow_pooling: bool = False) -> list[str]:
        """Refuse to pool failure modes that were not declared poolable.

        A mode column with more than one value means the file mixes
        populations. They may share a mechanism, but that is an engineering
        judgement about the physics, not something the counts can settle, so
        it has to be asserted rather than assumed.
        """
        mixed = {k: v for k, v in self.modes().items() if len(v) > 1}
        if not mixed:
            return []
        summary = "; ".join(f"{k}: {v}" for k, v in mixed.items())
        if not allow_pooling:
            raise ValueError(
                f"the failure set mixes populations ({summary}). Analyse them "
                "separately, or pass allow_pooling=True to assert that these "
                "modes share a defensible mechanism -- the assertion is "
                "recorded in the run metadata.")
        return [f"pooling failure modes across {summary}, asserted by the "
                "operator rather than established by the data"]

    def min_trustworthy_scale_um(self, factor: float = 3.0) -> float:
        """Smallest analysis scale the registration accuracy can support.

        Below roughly 3x the positional uncertainty a window no longer
        reliably contains the failure it is credited with, so association at
        that scale measures registration noise rather than layout.
        """
        s = self.position_sigma_um
        return float("nan") if np.isnan(s) else factor * s


def _validate_values(df: pd.DataFrame, path) -> None:
    """Reject values that would corrupt the analysis rather than fail loudly.

    Each of these has a specific downstream consequence, so none is tolerated
    and none is silently dropped -- discarding a measured failure is exactly
    the kind of quiet loss this module exists to prevent.
    """
    problems: list[str] = []

    for col in ("x_um", "y_um"):
        bad = ~np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy(float))
        if bad.any():
            rows = list(np.where(bad)[0][:5])
            problems.append(
                f"{col}: {int(bad.sum())} non-finite value(s) at row(s) {rows}. "
                "A single NaN coordinate makes distance_to_nearest_failure NaN "
                "for every cell on the die")

    if "position_sigma_um" in df:
        sigma = pd.to_numeric(df["position_sigma_um"], errors="coerce").to_numpy(float)
        bad = np.isfinite(sigma) & (sigma < 0)
        if bad.any():
            problems.append(
                f"position_sigma_um: {int(bad.sum())} negative value(s) at "
                f"row(s) {list(np.where(bad)[0][:5])}. A negative uncertainty "
                "produces a negative scale floor, which certifies every "
                "analysis scale instead of rejecting the small ones")

    if "confidence" in df:
        conf = pd.to_numeric(df["confidence"], errors="coerce").to_numpy(float)
        bad = np.isfinite(conf) & ((conf < 0) | (conf > 1))
        if bad.any():
            problems.append(
                f"confidence: {int(bad.sum())} value(s) outside [0, 1] at "
                f"row(s) {list(np.where(bad)[0][:5])}")

    ids = df["sample_id"].astype("string")
    bad = ids.isna() | (ids.str.strip() == "")
    if bad.any():
        problems.append(
            f"sample_id: {int(bad.sum())} empty value(s) at row(s) "
            f"{list(np.where(bad.to_numpy())[0][:5])}; a failure that cannot be "
            "named cannot be traced back to its measurement")

    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))


def load_failures(path: str | Path, *, require_grouping: bool = True) -> FailureSet:
    """Read a failure CSV, validating both the schema and the values."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    notes = []
    if require_grouping:
        miss_g = [c for c in GROUPING_COLUMNS if c not in df.columns]
        if miss_g:
            raise ValueError(
                f"{path}: missing grouping columns {miss_g}. Without lot/wafer/die "
                "identity the held-out-die validation of spec section 17 cannot be "
                "performed, and any reported AUC would be un-generalisable."
            )
    if "confidence" not in df:
        df["confidence"] = 1.0
        notes.append("confidence absent; defaulted to 1.0")
    if "position_sigma_um" not in df:
        df["position_sigma_um"] = np.nan
        notes.append(
            "position_sigma_um absent: registration accuracy unknown, so no "
            "analysis scale can be certified trustworthy"
        )
    for col in ("failed_layer", "failed_interface"):
        if col not in df:
            notes.append(
                f"{col} absent: the failed layer/interface is not recorded, so "
                "failures on mechanically different interfaces cannot be "
                "separated and are being analysed as one population")

    _validate_values(df, path)
    return FailureSet(table=df, source=str(path), notes=notes)


def map_to_grid(failures: FailureSet, grid, *, radius_um: float | None = None
                ) -> dict[str, np.ndarray]:
    """Derive per-cell failure labels (spec section 10).

    By default a failure belongs to the cell whose *bounds* contain it. Cells
    are squares, so testing a radius against the cell centre instead inscribes
    a circle in each one and silently discards everything in the corners --
    21% of the die (1 - pi/4), arranged on a regular lattice rather than
    scattered, which biases the labels wherever the layout has structure on
    the grid pitch.

    ``radius_um`` switches to a circular test around the cell centre, for the
    case where a failure is deliberately being credited to every cell within
    some distance of it -- an uncertainty-aware assignment, not containment.

    ``distance_to_nearest_failure`` is Euclidean in both modes.

    Returns failure_present, failure_count and distance_to_nearest_failure for
    every cell. A cell may legitimately hold more than one failure, and with
    an overlapping grid one failure legitimately lands in several cells.
    """
    cx = np.array([c.x_center for c in grid.cells])
    cy = np.array([c.y_center for c in grid.cells])
    fx = failures.table["x_um"].to_numpy(float)
    fy = failures.table["y_um"].to_numpy(float)

    n = len(grid)
    if len(fx) == 0:
        return {"failure_present": np.zeros(n, np.int8),
                "failure_count": np.zeros(n, np.int32),
                "distance_to_nearest_failure": np.full(n, np.inf)}

    x0 = np.array([c.x0 for c in grid.cells])
    y0 = np.array([c.y0 for c in grid.cells])
    x1 = np.array([c.x1 for c in grid.cells])
    y1 = np.array([c.y1 for c in grid.cells])
    # Bounds are half-open so that a failure on a shared edge is counted once,
    # except at the outer edge of the grid, where closing the interval is what
    # keeps a failure exactly on the die boundary from belonging to no cell.
    close_x = x1 >= grid.bbox.xmax - 1e-9
    close_y = y1 >= grid.bbox.ymax - 1e-9

    nearest = np.empty(n)
    count = np.zeros(n, np.int32)
    block = 4096
    for s in range(0, n, block):
        e = min(s + block, n)
        d = np.hypot(cx[s:e, None] - fx[None, :], cy[s:e, None] - fy[None, :])
        nearest[s:e] = d.min(axis=1)
        if radius_um is None:
            inside = (
                (fx[None, :] >= x0[s:e, None])
                & (np.where(close_x[s:e, None], fx[None, :] <= x1[s:e, None],
                            fx[None, :] < x1[s:e, None]))
                & (fy[None, :] >= y0[s:e, None])
                & (np.where(close_y[s:e, None], fy[None, :] <= y1[s:e, None],
                            fy[None, :] < y1[s:e, None]))
            )
        else:
            inside = d <= float(radius_um)
        count[s:e] = inside.sum(axis=1)

    return {"failure_present": (count > 0).astype(np.int8),
            "failure_count": count,
            "distance_to_nearest_failure": nearest}


def map_to_grid_per_die(failures: FailureSet, grid, *,
                        radius_um: float | None = None
                        ) -> dict[str, dict[str, np.ndarray]]:
    """Labels for each die separately, keyed by die identity.

    Pooling several dies of the same design onto one grid and asking "did
    anything ever fail here" is not a rescaling of the single-die case, it is
    a different and wrong question: prevalence grows with the number of dies
    (0.24 for one die becomes 0.98 for ten in a uniform simulation), a cell
    that failed on one die of ten becomes indistinguishable from one that
    failed on all ten, and die identity -- the thing spec section 17 wants to
    hold out -- is gone before any fold can be built from it.

    The observation unit is therefore (cell, die), which is also what spec
    section 11 means by a case: a location on a piece of silicon.
    """
    from dataclasses import replace

    keys = failures.die_keys()
    out = {}
    for key in keys.unique():
        subset = replace(failures,
                         table=failures.table[keys == key].reset_index(drop=True))
        out[str(key)] = map_to_grid(subset, grid, radius_um=radius_um)
    return out


def stratify(failures: FailureSet, by=("failed_interface",)
             ) -> dict[str, FailureSet]:
    """Split a failure set into the populations a mode column defines.

    Li et al. (2023) found the largest energy release rate at one particular
    upper BEOL interface, with bottom interconnect interfaces more critical
    than sidewalls. Analysing those together asks whether a feature associates
    with "failure" in general; analysing them apart asks whether it associates
    with each mechanism, and whether it does so in the same direction -- which
    is the question that distinguishes a mechanism from a proxy.
    """
    from dataclasses import replace as _replace

    present = [c for c in by if c in failures.table]
    if not present:
        return {"<all>": failures}

    # A missing value in a stratifying column is refused, not bucketed.
    # Joining it in produced a TypeError here (a float NaN survives the
    # astype(str) on a nullable string column), and the obvious repair -- a
    # "nan" or "<missing>" stratum -- is worse than the crash: it presents
    # "we do not know which interface this was" as a mechanism alongside
    # M8/ULK, and every per-stratum effect, direction and q-value would then
    # be reported for it. An unknown interface is not another known interface,
    # and the analysis cannot decide which one it was.
    blank = failures.table[present].isna().any(axis=1)
    for column in present:
        values = failures.table[column]
        if values.dtype == object or str(values.dtype).startswith("str"):
            blank |= values.astype("string").fillna("").str.strip() == ""
    if blank.any():
        counts = {c: int(failures.table[c].isna().sum()) for c in present}
        raise ValueError(
            f"{int(blank.sum())} of {len(failures.table)} failure(s) have no "
            f"value in the stratifying column(s) {present} "
            f"({', '.join(f'{k}: {v} missing' for k, v in counts.items())}). "
            "Stratifying by mechanism asks whether a feature associates with "
            "each mechanism separately; a row whose mechanism is unknown "
            "belongs to no stratum, and putting it in one of its own would "
            "report 'unknown' as a mechanism with its own effect size and "
            "q-value. Classify these rows, or drop them from the file and say "
            "in the study record that they were dropped and why.")

    keys = (failures.table[present].astype(str)
            .agg("|".join, axis=1).rename("stratum"))
    out = {}
    for key in sorted(keys.unique()):
        out[str(key)] = _replace(
            failures,
            table=failures.table[keys == key].reset_index(drop=True),
            notes=list(failures.notes) + [f"stratum {key} of {present}"])
    return out

# ----------------------------------------------------------------------
# labels/inspection.py
# ----------------------------------------------------------------------
"""Inspection footprint and control opportunity.

A case-control design needs a denominator. The pipeline's default -- every
cell without a recorded failure is a control -- is only valid where inspection
actually had a chance to find one. Acoustic scans cover regions; FIB
cross-sections are selected, and selected towards die corners and known-risk
structures. So uninspected area silently becomes control area, and any layout
feature correlated with where engineers chose to look acquires an association
that has nothing to do with mechanics.

That bias is not fixable downstream. Block permutation, FDR and the
position-only baseline all operate on whatever population they are handed; if
the population is wrong they are wrong together.

Cells outside the footprint are therefore excluded from **both** cases and
controls. Full-die coverage remains available but has to be asserted
deliberately, with a justification recorded in the run metadata, rather than
being what happens when nobody supplies a footprint.
"""
@dataclass
class InspectionFootprint:
    """Where inspection had a real opportunity to find a failure."""
    region: db.Region
    source: str
    assumed_full_coverage: bool = False
    justification: str = ""
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_gds_layer(cls, reader: LayoutReader, spec: LayerSpec
                       ) -> "InspectionFootprint":
        """Footprint drawn as polygons on a GDS layer."""
        region = reader.region(spec)
        if region.is_empty():
            raise ValueError(
                f"inspection footprint layer {spec} is empty; supply the "
                "inspected area or assert full coverage explicitly")
        return cls(region=region, source=f"gds_layer:{spec}")

    @classmethod
    def from_rectangles(cls, rects, *, dbu: float = 0.001, source: str = "rectangles"
                        ) -> "InspectionFootprint":
        """Footprint as (x0, y0, x1, y1) boxes in um -- e.g. scan frames."""
        region = db.Region()
        for x0, y0, x1, y1 in rects:
            region.insert(db.Box(int(round(x0 / dbu)), int(round(y0 / dbu)),
                                 int(round(x1 / dbu)), int(round(y1 / dbu))))
        region.merge()
        if region.is_empty():
            raise ValueError("no inspection rectangles supplied")
        return cls(region=region, source=source)

    @classmethod
    def full_die(cls, bbox: BBox, justification: str, *, dbu: float = 0.001
                 ) -> "InspectionFootprint":
        """Assert that the whole die was inspected.

        The justification is required and travels into the run metadata,
        because "we inspected everything" is a claim about the measurement
        campaign that someone has to own.
        """
        if not justification.strip():
            raise ValueError(
                "full-die coverage must be justified: state how the whole die "
                "was inspected and called, or supply the real footprint")
        region = db.Region(db.Box(int(round(bbox.xmin / dbu)),
                                  int(round(bbox.ymin / dbu)),
                                  int(round(bbox.xmax / dbu)),
                                  int(round(bbox.ymax / dbu))))
        return cls(region=region, source="full_die",
                   assumed_full_coverage=True, justification=justification)

    def area_um2(self, dbu: float = 0.001) -> float:
        return self.region.area() * dbu * dbu

    def report(self, dbu: float = 0.001) -> dict:
        return {"source": self.source,
                "assumed_full_coverage": self.assumed_full_coverage,
                "justification": self.justification,
                "area_um2": round(self.area_um2(dbu), 3),
                "notes": self.notes}


def coverage(footprint: InspectionFootprint, grid, *, dbu: float = 0.001
             ) -> np.ndarray:
    """Fraction of each cell that lies inside the inspected footprint."""
    out = np.zeros(len(grid))
    rb = footprint.region.bbox()
    rows: dict[int, list] = {}
    for c in grid.cells:
        rows.setdefault(c.row, []).append(c)

    def u(v):
        return int(round(v / dbu))

    for cells in rows.values():
        y0, y1 = u(cells[0].y0), u(cells[0].y1)
        if y1 <= rb.bottom or y0 >= rb.top:
            continue
        strip = footprint.region & db.Region(
            db.Box(rb.left - 1, y0, rb.right + 1, y1))
        if strip.is_empty():
            continue
        for c in cells:
            win = db.Region(db.Box(u(c.x0), u(c.y0), u(c.x1), u(c.y1)))
            out[c.cell_id] = ((strip & win).area() * dbu * dbu) / c.area_um2
    return np.clip(out, 0.0, 1.0)


def eligibility(footprint: InspectionFootprint, grid, *,
                min_coverage: float = 0.5, dbu: float = 0.001
                ) -> tuple[np.ndarray, np.ndarray]:
    """Which cells may take part in the analysis, and their coverage.

    A cell only partly inspected is a weaker control than a fully inspected
    one, and there is no way to express "half a control" in a binary label, so
    the threshold excludes it rather than pretending.
    """
    frac = coverage(footprint, grid, dbu=dbu)
    return frac >= min_coverage, frac


#: A failure may sit outside the footprint by this many standard deviations of
#: its own positional uncertainty without that being evidence of a wrong frame.
#: It matches the factor the scale floor uses, so "close enough to be the same
#: place" means the same thing in both.
TOLERANCE_SIGMAS = 3.0


def audit_failures(footprint: InspectionFootprint, failures, *,
                   dbu: float = 0.001, tolerance_um: float | None = None
                   ) -> dict:
    """Check that every recorded failure lies inside the inspected footprint.

    A failure genuinely outside it is a contradiction -- something was found
    where nothing was looked at -- and means the footprint, the registration
    or the coordinate frame is wrong. Each of those invalidates a different
    part of the analysis, so it is reported rather than tolerated.

    But a failure a few micrometres outside a boundary, measured with a
    positional uncertainty larger than that, is not a contradiction: it is the
    same failure seen through its own error. Treating it as one would make the
    check fire on every real campaign, and a check everyone overrides is not a
    check. The tolerance defaults to the failure set's own reported sigma
    times :data:`TOLERANCE_SIGMAS`; beyond it, measurement error is no longer
    an explanation.
    """
    if tolerance_um is None:
        sigma = failures.position_sigma_um
        tolerance_um = (TOLERANCE_SIGMAS * sigma
                        if np.isfinite(sigma) and sigma > 0 else 0.0)

    x = failures.table["x_um"].to_numpy(float)
    y = failures.table["y_um"].to_numpy(float)
    strict = footprint.region
    tolerant = (strict if tolerance_um <= 0
                else strict.sized(int(round(tolerance_um / dbu))))

    inside = np.zeros(len(x), dtype=bool)
    within_tolerance = np.zeros(len(x), dtype=bool)
    for i, (xi, yi) in enumerate(zip(x, y)):
        px, py = int(round(xi / dbu)), int(round(yi / dbu))
        # Centred on the point, not extending from it. A one-sided probe at a
        # coordinate lying exactly on the footprint boundary meets it only
        # along a line, which has no area, and the failure is reported as
        # outside -- a contradiction manufactured by the probe.
        probe = db.Region(db.Box(px - 1, py - 1, px + 1, py + 1))
        inside[i] = not (strict & probe).is_empty()
        if not inside[i]:
            within_tolerance[i] = not (tolerant & probe).is_empty()

    beyond = np.where(~inside & ~within_tolerance)[0]
    ids = failures.table["sample_id"].astype(str).to_numpy()
    return {
        "n_failures": len(x),
        "n_inside_footprint": int(inside.sum()),
        "n_within_tolerance": int(within_tolerance.sum()),
        "n_outside_footprint": int(len(beyond)),
        "tolerance_um": float(tolerance_um),
        "outside_sample_ids": list(ids[beyond][:10]),
        "consistent": bool(len(beyond) == 0),
    }


@dataclass
class FootprintSet:
    """One inspected footprint per die, with a fallback for the rest.

    A campaign rarely inspects every die the same way: one die gets a full
    acoustic scan, another gets three FIB cross-sections chosen after the
    scan. Collapsing that to a single footprint either discards the dies that
    were inspected more, or credits the ones inspected less with controls they
    never earned.
    """
    default: InspectionFootprint | None = None
    per_die: dict[str, InspectionFootprint] = field(default_factory=dict)

    def for_die(self, die_key: str) -> InspectionFootprint | None:
        return self.per_die.get(die_key, self.default)

    @property
    def is_uniform(self) -> bool:
        return not self.per_die

    def report(self, dbu: float = 0.001) -> dict:
        return {
            "uniform": self.is_uniform,
            "default": self.default.report(dbu) if self.default else None,
            "per_die": {k: v.report(dbu) for k, v in self.per_die.items()},
        }


def audit_failures_per_die(footprints: FootprintSet, failures, *,
                           dbu: float = 0.001) -> dict:
    """Check every failure against the footprint of the die it came from.

    Auditing against a pooled footprint would pass a failure that lies inside
    some other die's inspected area, which is not evidence that anyone looked
    at the place it was found.
    """
    from dataclasses import replace as _replace

    keys = failures.die_keys()
    outside, missing = [], []
    total_inside = 0
    for key in keys.unique():
        subset = _replace(failures,
                          table=failures.table[keys == key].reset_index(drop=True))
        fp = footprints.for_die(str(key))
        if fp is None:
            missing.append(str(key))
            continue
        result = audit_failures(fp, subset, dbu=dbu)
        total_inside += result["n_inside_footprint"] + result["n_within_tolerance"]
        outside.extend(f"{key}:{sid}" for sid in result["outside_sample_ids"])

    return {
        "n_failures": len(failures),
        "n_inside_footprint": total_inside,
        "n_outside_footprint": len(failures) - total_inside,
        "outside_sample_ids": outside[:10],
        "dies_without_a_footprint": missing,
        "consistent": not outside and not missing,
    }

# ----------------------------------------------------------------------
# labels/simulate.py
# ----------------------------------------------------------------------
"""Simulated failure labels, for pipeline validation only.

Every FailureSet produced here carries ``simulated=True``. Simulated labels
exist to prove the statistical pipeline recovers a driver it was given and
reports nothing when there is no driver; they are never evidence about a real
process and must not reach a report that claims measured association.
"""
def failures_from_driver(driver: np.ndarray, grid, *, n_failures: int = 120,
                         strength: float = 2.5, seed: int = 0,
                         position_sigma_um: float = 0.0,
                         lot_id: str = "SIM", wafer_id: str = "W01",
                         die_x: int = 0, die_y: int = 0) -> FailureSet:
    """Draw failure sites with probability driven by *driver*.

    ``strength`` is the logit coefficient on the z-scored driver: 0 gives a
    spatially uniform pattern (the negative-control case), larger values give
    a stronger association.
    """
    rng = np.random.default_rng(seed)
    z = (driver - driver.mean()) / (driver.std() + 1e-12)
    w = 1.0 / (1.0 + np.exp(-strength * z))
    w = w / w.sum()

    pick = rng.choice(len(grid), size=n_failures, replace=True, p=w)
    half = grid.scale_um / 2
    xs, ys = [], []
    for i in pick:
        c = grid.cells[i]
        xs.append(c.x_center + rng.uniform(-half, half))
        ys.append(c.y_center + rng.uniform(-half, half))
    xs = np.array(xs)
    ys = np.array(ys)

    if position_sigma_um > 0:
        # Registration error: the recorded position is not the true one.
        xs = xs + rng.normal(0, position_sigma_um, size=len(xs))
        ys = ys + rng.normal(0, position_sigma_um, size=len(ys))
        # A failure is on the die, so a simulated measurement of one stays on
        # it. Letting the jitter carry a near-edge failure off the die would
        # manufacture the very contradiction the footprint audit exists to
        # detect on real data, and mask it as ordinary simulation noise.
        b = grid.bbox
        xs = np.clip(xs, b.xmin, b.xmax)
        ys = np.clip(ys, b.ymin, b.ymax)

    df = pd.DataFrame({
        "sample_id": [f"S{i:04d}" for i in range(len(xs))],
        "lot_id": lot_id, "wafer_id": wafer_id, "die_x": die_x, "die_y": die_y,
        "x_um": xs, "y_um": ys,
        "failure_type": "delamination",
        "confidence": 1.0,
        "position_sigma_um": position_sigma_um,
        "coord_frame": "die_local",
    })
    return FailureSet(table=df, simulated=True,
                      source=f"simulated(strength={strength}, seed={seed})",
                      notes=["SIMULATED LABELS - not measured evidence"])


def uniform_failures(grid, *, n_failures: int = 120, seed: int = 0,
                     **kwargs) -> FailureSet:
    """Spatially uniform failures: the null case, no feature drives them."""
    return failures_from_driver(np.zeros(len(grid)), grid,
                                n_failures=n_failures, strength=0.0,
                                seed=seed, **kwargs)
