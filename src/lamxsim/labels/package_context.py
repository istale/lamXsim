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
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evidence import EvidenceClass
from ..layout.reader import BBox, LayerSpec, LayoutReader

EVIDENCE_CLASS = EvidenceClass.PACKAGE_POSITION

FEATURES = (
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


def extract(grid, die_bbox: BBox, reader: LayoutReader,
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
    from ..features.corners import classify
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
    from ..features import objects as obj

    kinds = {"bump": layers.bump, "pad": layers.pad,
             "pi_opening": layers.pi_opening}
    table, polys = {}, {}
    for kind, spec in kinds.items():
        polarity = semantics.polarity_of(kind)
        table[kind] = obj.objects_for(reader, spec, kind=kind,
                                      polarity=polarity, die_bbox=die_bbox)
        polys[kind] = (list(obj.region_for(reader, spec, polarity).each())
                       if spec is not None else [])
    # The crackstop is a structure rather than an array of objects, so it has
    # its own row shape; it is carried here so the object table can hold every
    # package object the manifest names and not a subset of them.
    table["crackstop"] = obj.objects_for(
        reader, layers.crackstop, kind="crackstop",
        polarity=semantics.polarity_of("crackstop"), die_bbox=die_bbox)

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
    return table, matches


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
    from ..features import objects as obj

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

    table, matches = object_table(reader, layers, semantics, die_bbox)

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

    # Corner-resolved, which is what can be ranked within a die and is where
    # the lever is. Each cell carries its own die corner's figures, so a
    # corner drawn narrower than the other three is locatable.
    topology = obj.corner_topology(reader, layers.crackstop, die_bbox)
    if topology:
        per_corner = topology["per_corner"]
        mx = (die_bbox.xmin + die_bbox.xmax) / 2
        my = (die_bbox.ymin + die_bbox.ymax) / 2
        narrowest = np.full(n, np.nan)
        for cell in grid.cells:
            key = ("l" if cell.x_center < mx else "r")
            key = ("l" if cell.y_center < my else "u") + key
            key = {"ll": "ll", "lr": "lr", "ul": "ul", "ur": "ur"}[key]
            narrowest[cell.cell_id] = per_corner[key]["narrowest_um"]
        out["crackstop_corner_narrowest_um"] = narrowest
        out["crackstop_corner_asymmetry"] = np.full(
            n, topology["corner_asymmetry"], dtype=float)
    return out
