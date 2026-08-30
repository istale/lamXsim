"""Per-object shape descriptors for bumps, pads, PI openings and crackstops.

Everything else in this package measures a window: how much metal, how much
boundary, how many corners. That is the right unit for routing and the wrong
one for a bump. A pad's shape is a property of the pad, and averaging it into
a 100 um window destroys it before anything can look at it -- two pads of
equal area and opposite aspect ratio produce the same window mean.

So the order here is object first, grid last:

    polygons -> objects -> shape descriptors -> matching -> relations
             -> rasterise onto the analysis grid -> literature channels

Every descriptor keeps its object id, its centroid, the definition it was
computed with, and -- where it has one -- the reason it is undefined. A square
pad has no principal axis, and reporting one taken from numerical noise in the
second moments would be worse than reporting none.

What is here is **drawn** geometry, in plan view. A GDS says what was drawn,
not what was manufactured: none of this is the post-reflow bump, the printed
opening after lithography, the assembled overlay, or any sidewall or taper
angle -- a GDS holds no Z information at all, so no vertical angle is
derivable from it by any means. Where the literature names an angle, the
manifest has to say whether it means the plan-view corner angle, which is
here, or a sidewall angle, which is not obtainable and is refused.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import klayout.db as db
import numpy as np

from ..evidence import EvidenceClass
from ..layout.reader import BBox, LayerSpec, LayoutReader

EVIDENCE_CLASS = EvidenceClass.PACKAGE_POSITION

#: Two second moments this close in relative terms leave the principal axis
#: undefined: a square, a circle and a regular octagon all have equal moments
#: about every axis, and the eigenvector is then whatever the rounding says.
ISOTROPY_TOLERANCE = 1e-3

#: Interior angle of a regular octagon, in degrees.
OCTAGON_INTERIOR_DEG = 135.0

#: Angular resolution of a corner-angle measurement, in degrees.
#:
#: Vertices sit on the database unit grid, so an angle computed from them
#: carries a snapping error: sixteen identical drawn octagons come out spread
#: over about 0.04 degrees. Ranking that spread orders the pads by arithmetic
#: rather than by geometry, and on a die of identical pads it manufactures a
#: candidate out of rounding. This is far below any drawn distinction -- the
#: departure between an octagon and a square is 45 degrees -- and comfortably
#: above the snapping noise.
ANGLE_RESOLUTION_DEG = 0.1


def _quantise(value: float, step: float) -> float:
    """Round to the resolution the measurement actually has.

    A GDS cannot express a difference smaller than its database unit, so a
    length difference below one is arithmetic, not geometry. Left in, it is
    ranked: a pad-to-bump offset of 3e-14 um made a second pad the top
    candidate on a die where every pad but one was identical.
    """
    if not math.isfinite(value) or step <= 0:
        return value
    return round(value / step) * step


@dataclass(frozen=True)
class ShapeObject:
    """One drawn polygon, described on its own terms."""
    object_id: str
    kind: str                     # bump | pad | pi_opening | crackstop
    source_layer: str
    polarity: str                 # positive | opening
    x_um: float
    y_um: float
    area_um2: float
    perimeter_um: float
    equivalent_diameter_um: float
    feret_min_um: float
    feret_max_um: float
    #: Long side over short side of the minimum-area rotated rectangle, not
    #: the caliper ratio -- see _min_area_rectangle.
    aspect_ratio: float
    circularity: float
    n_vertices: int
    n_convex_corners: int
    n_concave_corners: int
    interior_angles_deg: tuple[float, ...]
    principal_axis_rad: float                  # NaN when undefined
    orientation_undefined_reason: str
    placement_angle_rad: float                 # NaN without a die frame
    radial_distance_um: float
    definitions: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        row = {k: v for k, v in vars(self).items()
               if k not in ("interior_angles_deg", "definitions")}
        row["n_interior_angles"] = len(self.interior_angles_deg)
        row["definitions"] = ";".join(f"{k}={v}" for k, v in
                                      sorted(self.definitions.items()))
        return row


def _polygon_points(poly) -> np.ndarray:
    return np.array([[p.x, p.y] for p in poly.each_point_hull()], dtype=float)


def _centroid_and_moments(pts: np.ndarray):
    """Area centroid and second central moments, by the shoelace formulae.

    The bounding-box centre is not the centroid of anything but a rectangle,
    and an L-shaped or keyed-out pad is exactly the case where a shape
    descriptor is worth having.
    """
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area2 = cross.sum()
    if abs(area2) < 1e-12:
        return float(x.mean()), float(y.mean()), 0.0, 0.0, 0.0
    cx = float(((x + x1) * cross).sum() / (3.0 * area2))
    cy = float(((y + y1) * cross).sum() / (3.0 * area2))

    dx, dy = x - cx, y - cy
    dx1, dy1 = x1 - cx, y1 - cy
    cross_c = dx * dy1 - dx1 * dy
    area = area2 / 2.0
    mxx = float((cross_c * (dx * dx + dx * dx1 + dx1 * dx1)).sum() / (12.0 * area))
    myy = float((cross_c * (dy * dy + dy * dy1 + dy1 * dy1)).sum() / (12.0 * area))
    mxy = float((cross_c * (dx * dy1 + 2 * dx * dy + 2 * dx1 * dy1
                            + dx1 * dy)).sum() / (24.0 * area))
    return cx, cy, mxx, myy, mxy


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) > 0:
                    break
                out.pop()
            out.append(q)
        return out

    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=float).reshape(-1, 2)


def _feret(pts: np.ndarray) -> tuple[float, float]:
    """Minimum and maximum caliper width of the hull.

    The minimum width of a convex polygon is always achieved with one edge
    flush against a caliper, so evaluating the edge normals is exact rather
    than a sampled approximation -- a 1 degree sweep is out by up to 0.015 %
    on a long shape, which is small but is an error where none is needed.
    """
    hull = _convex_hull(pts)
    if len(hull) < 2:
        return 0.0, 0.0
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    keep = lengths > 0
    if not keep.any():
        return 0.0, 0.0
    normals = np.column_stack([-edges[keep, 1], edges[keep, 0]])
    normals /= lengths[keep][:, None]
    projections = hull @ normals.T
    widths = projections.max(axis=0) - projections.min(axis=0)

    diff = hull[:, None, :] - hull[None, :, :]
    diameter = float(np.hypot(diff[:, :, 0], diff[:, :, 1]).max())
    return float(widths.min()), diameter


def _min_area_rectangle(pts: np.ndarray) -> tuple[float, float]:
    """Side lengths of the smallest-area rotated rectangle enclosing the hull.

    Used for the aspect ratio in preference to the caliper pair. The maximum
    Feret diameter of a square is its diagonal, so a Feret aspect calls a
    square 1.41 and a 40x10 bar 4.12 -- both defensible, neither what a reader
    means by an aspect ratio. The minimum-area rectangle calls them 1.0 and
    4.0. Both caliper widths are still reported; only the ratio changed.
    """
    hull = _convex_hull(pts)
    if len(hull) < 3:
        return 0.0, 0.0
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    keep = lengths > 0
    if not keep.any():
        return 0.0, 0.0
    ux = edges[keep] / lengths[keep][:, None]
    best = None
    for dx, dy in ux:
        along = hull @ np.array([dx, dy])
        across = hull @ np.array([-dy, dx])
        w = float(along.max() - along.min())
        h = float(across.max() - across.min())
        if best is None or w * h < best[0] * best[1]:
            best = (w, h)
    return (max(best), min(best))


def _interior_angles(pts: np.ndarray) -> np.ndarray:
    """Interior angle at each vertex, in degrees, in plan view."""
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    a, b = prev - pts, nxt - pts
    na = np.hypot(a[:, 0], a[:, 1])
    nb = np.hypot(b[:, 0], b[:, 1])
    ok = (na > 0) & (nb > 0)
    cos = np.ones(len(pts))
    cos[ok] = np.clip(((a[ok] * b[ok]).sum(axis=1)) / (na[ok] * nb[ok]), -1, 1)
    angle = np.degrees(np.arccos(cos))
    # Reflex vertices: the cross product's sign against the ring's own
    # winding says which side the material is on. Getting this backwards
    # reports a regular octagon as eight 225-degree re-entrant corners, which
    # then fails every octagonality test for the right reason and the wrong
    # cause -- so the winding is taken from the shoelace sum rather than
    # assumed.
    x, y = pts[:, 0], pts[:, 1]
    twice_area = float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())
    turn = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    reflex = (turn > 0) if twice_area > 0 else (turn < 0)
    return np.where(reflex, 360.0 - angle, angle)


def describe(poly, *, object_id: str, kind: str, source_layer: str,
             polarity: str, dbu: float, die_bbox: BBox | None) -> ShapeObject:
    """Every plan-view descriptor of one drawn polygon."""
    pts = _polygon_points(poly) * dbu
    cx, cy, mxx, myy, mxy = _centroid_and_moments(pts)
    area = abs(poly.area()) * dbu * dbu
    perimeter = poly.perimeter() * dbu
    feret_min, feret_max = (_quantise(v, dbu) for v in _feret(pts))
    long_side, short_side = (_quantise(v, dbu)
                             for v in _min_area_rectangle(pts))
    angles = _interior_angles(pts)

    # Principal axis from the doubled-angle form, undefined when the two
    # moments are equal: a square, a circle and a regular octagon have the
    # same second moment about every axis, so any axis reported would be the
    # direction of the rounding error.
    axis, reason = float("nan"), ""
    spread = math.hypot(mxx - myy, 2 * mxy)
    scale = max(mxx + myy, 1e-12)
    if spread / scale < ISOTROPY_TOLERANCE:
        reason = (f"second moments equal to within {ISOTROPY_TOLERANCE:g}; the "
                  "shape has no long axis, so no orientation exists to report")
    else:
        axis = 0.5 * math.atan2(2 * mxy, mxx - myy) % math.pi

    placement, radial = float("nan"), float("nan")
    if die_bbox is not None:
        mx = (die_bbox.xmin + die_bbox.xmax) / 2
        my = (die_bbox.ymin + die_bbox.ymax) / 2
        placement = math.atan2(cy - my, cx - mx)
        radial = math.hypot(cx - mx, cy - my)

    convex = int((angles < 180.0 - 1e-9).sum())
    return ShapeObject(
        object_id=object_id, kind=kind, source_layer=source_layer,
        polarity=polarity, x_um=cx, y_um=cy, area_um2=area,
        perimeter_um=perimeter,
        equivalent_diameter_um=2.0 * math.sqrt(area / math.pi) if area > 0 else 0.0,
        feret_min_um=feret_min, feret_max_um=feret_max,
        aspect_ratio=(long_side / short_side) if short_side > 0 else float("nan"),
        circularity=(4 * math.pi * area / (perimeter ** 2)) if perimeter > 0
        else float("nan"),
        n_vertices=len(pts), n_convex_corners=convex,
        n_concave_corners=len(pts) - convex,
        interior_angles_deg=tuple(float(a) for a in angles),
        principal_axis_rad=axis, orientation_undefined_reason=reason,
        placement_angle_rad=placement, radial_distance_um=radial,
        definitions={
            "centroid": "area centroid (shoelace), not the bounding-box centre",
            "equivalent_diameter": "2*sqrt(area/pi)",
            "feret": "exact rotating calipers on the convex hull",
            "aspect": "long/short side of the minimum-area rotated rectangle",
            "orientation": "principal axis of the second area moments",
            "angles": "plan-view interior angles; no Z information exists in a GDS",
            "geometry": "drawn, not manufactured",
        })


def objects_for(reader: LayoutReader, spec: LayerSpec | None, *, kind: str,
                polarity: str, die_bbox: BBox | None) -> list[ShapeObject]:
    """Describe every merged polygon on one layer."""
    if spec is None:
        return []
    dbu = reader.units.dbu
    out = []
    for i, poly in enumerate(reader.region(spec).each()):
        out.append(describe(poly, object_id=f"{kind}:{i}", kind=kind,
                            source_layer=str(spec), polarity=polarity,
                            dbu=dbu, die_bbox=die_bbox))
    return out


#: How a secondary object is attached to a primary one. Declared, never
#: guessed: a pad array and a bump array of the same pitch can be matched
#: one-to-one, by containment or by nearest centroid, and the three disagree
#: exactly where the layout is interesting -- an offset pad, a missing bump, a
#: bump serving two pads.
MATCH_RULES = ("containment", "nearest", "one_to_one")


@dataclass(frozen=True)
class ObjectMatch:
    """One primary object and the secondary it was matched to."""
    primary_id: str
    secondary_id: str
    rule: str
    centroid_offset_um: float
    radial_offset_um: float
    tangential_offset_um: float
    overlap_fraction: float
    ambiguity: str = ""


def _radial_frame(x, y, die_bbox: BBox | None):
    if die_bbox is None:
        return None
    mx = (die_bbox.xmin + die_bbox.xmax) / 2
    my = (die_bbox.ymin + die_bbox.ymax) / 2
    dx, dy = x - mx, y - my
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 0 else (1.0, 0.0)


def match(primaries: list[ShapeObject], secondaries: list[ShapeObject],
          *, rule: str, die_bbox: BBox | None,
          tolerance_um: float | None = None,
          regions: "tuple[db.Region, db.Region] | None" = None,
          dbu: float = 0.001) -> list[ObjectMatch]:
    """Attach each primary to at most one secondary, and record the doubt.

    An unmatched primary produces no row rather than a row full of zeros: a
    pad with no bump over it is not a pad with a perfectly concentric bump.
    Ambiguity is recorded on the row, not resolved silently -- two candidates
    within the tolerance is a fact about the layout or about the declared
    rule, and the run should be able to say how often it happened.
    """
    if rule not in MATCH_RULES:
        raise ValueError(f"unknown object matching rule {rule!r}; "
                         f"declare one of {list(MATCH_RULES)} in the manifest")
    if not primaries or not secondaries:
        return []

    sx = np.array([s.x_um for s in secondaries])
    sy = np.array([s.y_um for s in secondaries])
    tol = tolerance_um
    out = []
    for p in primaries:
        d = np.hypot(sx - p.x_um, sy - p.y_um)
        order = np.argsort(d)
        best = int(order[0])
        ambiguity = ""

        if rule == "containment":
            inside = [i for i in range(len(secondaries))
                      if _contains(p, secondaries[i])]
            if not inside:
                continue
            if len(inside) > 1:
                ambiguity = (f"{len(inside)} secondaries fall inside this "
                             "primary; the nearest centroid was taken")
                inside.sort(key=lambda i: d[i])
            best = inside[0]
        elif rule == "nearest":
            if tol is not None and d[best] > tol:
                continue
            if len(d) > 1 and abs(d[order[1]] - d[best]) < max(1e-9, 0.01 * d[best]):
                ambiguity = ("two secondaries are equidistant to within 1%; "
                             "the match is arbitrary between them")
        else:  # one_to_one
            if tol is not None and d[best] > tol:
                continue
            back = np.hypot(np.array([q.x_um for q in primaries]) - sx[best],
                            np.array([q.y_um for q in primaries]) - sy[best])
            if primaries[int(np.argmin(back))].object_id != p.object_id:
                ambiguity = ("not a mutual nearest pair, so the one-to-one "
                             "rule does not hold here")

        s = secondaries[best]
        frame = _radial_frame(p.x_um, p.y_um, die_bbox)
        dx, dy = s.x_um - p.x_um, s.y_um - p.y_um
        radial = tangential = float("nan")
        if frame is not None:
            ux, uy = frame
            radial, tangential = dx * ux + dy * uy, -dx * uy + dy * ux

        overlap = float("nan")
        if regions is not None:
            overlap = _overlap_fraction(p, s, regions, dbu)

        # Quantised to the database unit: below it there is no geometry, only
        # the arithmetic of computing a centroid from snapped vertices.
        out.append(ObjectMatch(
            primary_id=p.object_id, secondary_id=s.object_id, rule=rule,
            centroid_offset_um=_quantise(float(math.hypot(dx, dy)), dbu),
            radial_offset_um=_quantise(float(radial), dbu),
            tangential_offset_um=_quantise(float(tangential), dbu),
            overlap_fraction=overlap, ambiguity=ambiguity))
    return out


def _contains(primary: ShapeObject, secondary: ShapeObject) -> bool:
    """Centroid containment, judged by the equivalent radius.

    A cheap test on purpose: the exact one needs the polygons, and the
    containment rule exists for a pad/bump/PI stack where the objects are
    concentric by construction. Where that is not true the manifest should
    declare `nearest` or `one_to_one` instead.
    """
    r = primary.equivalent_diameter_um / 2
    return math.hypot(secondary.x_um - primary.x_um,
                      secondary.y_um - primary.y_um) <= r


def _overlap_fraction(primary: ShapeObject, secondary: ShapeObject,
                      regions, dbu: float) -> float:
    """area(primary AND secondary) / area(primary), on the drawn shapes."""
    primary_region, secondary_region = regions
    box = db.Box(int((primary.x_um - primary.feret_max_um) / dbu),
                 int((primary.y_um - primary.feret_max_um) / dbu),
                 int((primary.x_um + primary.feret_max_um) / dbu),
                 int((primary.y_um + primary.feret_max_um) / dbu))
    clip = db.Region(box)
    a = primary_region & clip
    b = secondary_region & clip
    area_a = a.area() * dbu * dbu
    if area_a <= 0:
        return float("nan")
    return float((a & b).area() * dbu * dbu / area_a)


def corner_angle_departure(obj: ShapeObject, target_deg: float,
                           *, tolerance_deg: float = 5.0,
                           resolution_deg: float = ANGLE_RESOLUTION_DEG) -> float:
    """Mean absolute departure of the convex corners from a target angle.

    An octagonal pad recommendation is a statement about corner angles, so the
    departure has to be measured against them. Reflex corners are excluded:
    they are a different feature of the shape and averaging them in lets a
    keyed-out notch read as a rounded corner.

    ``tolerance_deg`` is not a threshold on the result. It only decides which
    corners count as *at* the target for the octagonality fraction, which is
    reported separately.
    """
    angles = np.array([a for a in obj.interior_angles_deg if a < 180.0])
    if len(angles) == 0:
        return float("nan")
    return _quantise(float(np.abs(angles - target_deg).mean()), resolution_deg)


def target_corner_fraction(obj: ShapeObject, target_deg: float,
                           *, tolerance_deg: float = 5.0) -> float:
    angles = np.array([a for a in obj.interior_angles_deg if a < 180.0])
    if len(angles) == 0:
        return float("nan")
    return float((np.abs(angles - target_deg) <= tolerance_deg).mean())


def rasterise(objects: list[ShapeObject], grid, values: dict[str, np.ndarray],
              *, prefix: str) -> dict[str, np.ndarray]:
    """Project per-object values onto the grid, by centroid.

    A cell with no object gets NaN, not zero. Zero is a value -- "this pad has
    no aspect ratio" -- and it would be ranked against the cells that do have
    pads. NaN drops out of the percentile rank, so a pad channel ranks pads
    against pads.

    Where several objects share a cell the mean is taken, and the count is
    reported beside it so a reader can see when that happened.
    """
    n = len(grid)
    out = {f"{prefix}_count": np.zeros(n)}
    if not objects:
        for name in values:
            out[f"{prefix}_{name}"] = np.full(n, np.nan)
        return out

    ox = np.array([o.x_um for o in objects])
    oy = np.array([o.y_um for o in objects])
    sums = {name: np.zeros(n) for name in values}
    counts = {name: np.zeros(n) for name in values}
    for cell in grid.cells:
        inside = ((ox >= cell.x0) & (ox < cell.x1)
                  & (oy >= cell.y0) & (oy < cell.y1))
        k = int(inside.sum())
        out[f"{prefix}_count"][cell.cell_id] = k
        if not k:
            continue
        for name, series in values.items():
            finite = inside & np.isfinite(series)
            counts[name][cell.cell_id] = finite.sum()
            sums[name][cell.cell_id] = series[finite].sum()
    for name in values:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{prefix}_{name}"] = np.where(
                counts[name] > 0, sums[name] / np.maximum(counts[name], 1),
                np.nan)
    return out


@dataclass(frozen=True)
class CrackstopStructure:
    """The crackstop as a structure, not as a distance.

    ``distance_to_crackstop`` says how far a cell is from the seal ring. The
    lever Rabie reports is about the ring itself -- how wide it is, whether
    there are two of them, and whether it is continuous. A distance cannot
    express any of that, and a ring's bounding box is the die, so a
    centre-based measure of it is numerically the distance to the die centre.
    """
    n_rails: int
    rail_width_min_um: float
    rail_width_median_um: float
    rail_width_p10_um: float
    rail_spacing_um: float          # NaN with fewer than two rails
    n_components: int
    continuity_ratio: float         # longest component / total boundary length
    n_gaps: int
    undefined_reason: str = ""


def crackstop_structure(reader: LayoutReader, spec: LayerSpec | None,
                        die_bbox: BBox | None) -> CrackstopStructure | None:
    """Measure the drawn seal-ring structure.

    Widths come from a local width probe along the ring: the polygon is
    scanned with the same opening the wide-metal measure uses, at a ladder of
    widths, and each rail's width is the largest opening it survives. That is
    a drawn width -- what the mask says, not what came out of the line.
    """
    if spec is None:
        return None
    region = reader.region(spec)
    if region.is_empty():
        return None
    u = reader.units
    polygons = list(region.each())
    n_components = len(polygons)

    # A closed ring is one polygon with one hole. Two concentric rings drawn
    # as separate shapes are two polygons; drawn as one shape with two holes
    # they are one. Counting holes covers both.
    n_rails = sum(max(p.holes(), 1) for p in polygons)

    widths = []
    for poly in polygons:
        single = db.Region()
        single.insert(poly)
        widths.append(_largest_surviving_opening(single, u))
    widths = np.array([w for w in widths if w > 0], dtype=float)
    if len(widths) == 0:
        return CrackstopStructure(
            n_rails=n_rails, rail_width_min_um=float("nan"),
            rail_width_median_um=float("nan"), rail_width_p10_um=float("nan"),
            rail_spacing_um=float("nan"), n_components=n_components,
            continuity_ratio=float("nan"), n_gaps=0,
            undefined_reason="no rail survived any opening; the layer may not "
                             "hold a seal ring")

    lengths = np.array([abs(p.perimeter()) * u.dbu for p in polygons])
    continuity = float(lengths.max() / lengths.sum()) if lengths.sum() else float("nan")

    spacing = float("nan")
    if n_components >= 2 and die_bbox is not None:
        centres = np.array([[(p.bbox().left + p.bbox().right) / 2 * u.dbu,
                             (p.bbox().bottom + p.bbox().top) / 2 * u.dbu]
                            for p in polygons])
        half = np.array([[(p.bbox().right - p.bbox().left) / 2 * u.dbu,
                          (p.bbox().top - p.bbox().bottom) / 2 * u.dbu]
                         for p in polygons])
        extent = half.max(axis=1)
        order = np.argsort(-extent)
        spacing = float(extent[order[0]] - extent[order[1]]
                        - widths.max() if len(order) > 1 else float("nan"))

    return CrackstopStructure(
        n_rails=n_rails, rail_width_min_um=float(widths.min()),
        rail_width_median_um=float(np.median(widths)),
        rail_width_p10_um=float(np.percentile(widths, 10)),
        rail_spacing_um=spacing, n_components=n_components,
        continuity_ratio=continuity,
        # A ring drawn as one closed polygon is continuous. More components
        # than rails means the ring is cut somewhere, which is a segmentation
        # the lever is about.
        n_gaps=max(n_components - 1, 0))


def _largest_surviving_opening(region, units, *, steps: int = 24) -> float:
    """The widest morphological opening the shape survives, in um.

    A rail of width w vanishes under an opening at w, so a bisection on the
    opening size recovers the drawn width without walking edges. It is the
    *narrowest* place on the rail that decides, which is the right summary for
    a structure whose job is to be continuous.
    """
    box = region.bbox()
    hi = max(box.width(), box.height()) / 2.0
    lo = 0.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        h = max(int(mid / 2), 1)
        if region.sized(-h).sized(h).is_empty():
            hi = mid
        else:
            lo = mid
        if hi - lo < 1.0:            # one dbu
            break
    return lo * units.dbu
