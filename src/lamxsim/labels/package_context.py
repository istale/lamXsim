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
                  "bump_tangential_offset", "local_bump_pitch",
                  "bump_count_density", "under_bump_indicator"):
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
    # A zero-area box intersects nothing, so the probe is one database unit
    # across -- effectively a point, but with area for the boolean to keep.
    for i, (xi, yi) in enumerate(zip(x, y)):
        px, py = u.um_to_dbu(xi), u.um_to_dbu(yi)
        probe = db.Region(db.Box(px, py, px + 1, py + 1))
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
