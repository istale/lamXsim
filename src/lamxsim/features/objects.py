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


def region_for(reader: LayoutReader, spec: LayerSpec, polarity: str):
    """The objects on a layer, whichever way round the layer draws them.

    ``positive`` means the polygon is the object. ``opening`` means the
    polygon is the film and the objects are the holes in it, so the objects
    are the holes -- extracted here rather than assumed away.

    This is the difference between measuring a 40x40 um opening and measuring
    the 200x200 um film around it. Declaring the polarity and then describing
    the drawn polygon either way gives the same wrong answer with a correct
    label on it, which is worse than no label: every PI area, diameter, aspect
    ratio, orientation and pad match would be computed on the film.
    """
    region = reader.region(spec)
    if polarity != "opening":
        return region
    holes = db.Region()
    for poly in region.each():
        for i in range(poly.holes()):
            holes.insert(db.Polygon(list(poly.each_point_hole(i))))
    if holes.is_empty():
        # A layer declared as openings whose polygons carry no holes is
        # already drawn as the openings themselves -- the common case, and the
        # one the golden and synthetic dies use. Nothing to invert.
        return region
    holes.merge()
    return holes


def objects_for(reader: LayoutReader, spec: LayerSpec | None, *, kind: str,
                polarity: str, die_bbox: BBox | None) -> list[ShapeObject]:
    """Describe every object on one layer, honouring the declared polarity."""
    if spec is None:
        return []
    dbu = reader.units.dbu
    out = []
    for i, poly in enumerate(region_for(reader, spec, polarity).each()):
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
          polygons: "tuple[list, list] | None" = None,
          dbu: float = 0.001) -> list[ObjectMatch]:
    """Attach each primary to at most one secondary, and record the doubt.

    An unmatched primary produces no row rather than a row full of zeros: a
    pad with no bump over it is not a pad with a perfectly concentric bump.

    ``one_to_one`` means it. A pair that is not mutually nearest is not a
    one-to-one match, so it is dropped rather than emitted with a note
    attached -- two pads both pointing at ``bump:0``, each row naming the
    one-to-one rule, is the rule being reported and not applied.

    ``containment`` is polygon containment. Judging it by a circle of equal
    area, as this did, mismatches every non-compact shape: a 200x10 um bar pad
    and a 40x40 um square have the same equivalent radius and contain
    completely different sets of points.
    """
    if rule not in MATCH_RULES:
        raise ValueError(f"unknown object matching rule {rule!r}; "
                         f"declare one of {list(MATCH_RULES)} in the manifest")
    if not primaries or not secondaries:
        return []
    if rule == "containment" and polygons is None:
        raise ValueError(
            "the containment rule needs the polygons themselves, and only "
            "centroids were passed. Containment judged from a centroid and an "
            "equivalent radius is a different rule wearing this one's name.")

    sx = np.array([s.x_um for s in secondaries])
    sy = np.array([s.y_um for s in secondaries])
    px = np.array([p.x_um for p in primaries])
    py = np.array([p.y_um for p in primaries])
    primary_polys, secondary_polys = polygons if polygons else (None, None)

    out = []
    for index, p in enumerate(primaries):
        d = np.hypot(sx - p.x_um, sy - p.y_um)
        order = np.argsort(d)
        best = int(order[0])
        ambiguity = ""

        if rule == "containment":
            inside = [i for i in range(len(secondaries))
                      if _point_in_polygon(sx[i], sy[i], primary_polys[index],
                                           dbu)]
            if not inside:
                continue
            if len(inside) > 1:
                ambiguity = (f"{len(inside)} secondaries fall inside this "
                             "primary; the nearest centroid was taken")
                inside.sort(key=lambda i: d[i])
            best = inside[0]
        elif rule == "nearest":
            if tolerance_um is not None and d[best] > tolerance_um:
                continue
            if len(d) > 1 and abs(d[order[1]] - d[best]) < max(1e-9, 0.01 * d[best]):
                ambiguity = ("two secondaries are equidistant to within 1%; "
                             "the match is arbitrary between them")
        else:  # one_to_one
            if tolerance_um is not None and d[best] > tolerance_um:
                continue
            back = np.hypot(px - sx[best], py - sy[best])
            if int(np.argmin(back)) != index:
                continue

        s_obj = secondaries[best]
        frame = _radial_frame(p.x_um, p.y_um, die_bbox)
        dx, dy = s_obj.x_um - p.x_um, s_obj.y_um - p.y_um
        radial = tangential = float("nan")
        if frame is not None:
            ux, uy = frame
            radial, tangential = dx * ux + dy * uy, -dx * uy + dy * ux

        overlap = float("nan")
        if primary_polys is not None and secondary_polys is not None:
            overlap = _overlap_fraction(primary_polys[index],
                                        secondary_polys[best], dbu)

        # Quantised to the database unit: below it there is no geometry, only
        # the arithmetic of computing a centroid from snapped vertices.
        out.append(ObjectMatch(
            primary_id=p.object_id, secondary_id=s_obj.object_id, rule=rule,
            centroid_offset_um=_quantise(float(math.hypot(dx, dy)), dbu),
            radial_offset_um=_quantise(float(radial), dbu),
            tangential_offset_um=_quantise(float(tangential), dbu),
            overlap_fraction=overlap, ambiguity=ambiguity))
    return out


def _point_in_polygon(x_um: float, y_um: float, poly, dbu: float) -> bool:
    """Is this point inside the drawn polygon, holes included?"""
    return poly.inside(db.Point(int(round(x_um / dbu)), int(round(y_um / dbu))))


def _overlap_fraction(primary_poly, secondary_poly, dbu: float) -> float:
    """area(this primary AND this secondary) / area(this primary).

    On the two polygons themselves. Clipping the whole layer to a box around
    the primary instead lets a neighbouring pad in a dense array into the
    denominator, so a perfectly covered pad reports less than full overlap for
    a reason that has nothing to do with it.
    """
    a = db.Region(primary_poly)
    b = db.Region(secondary_poly)
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

    Two concentric rails and one cut ring both give two polygons, and they are
    opposite things: the first is the recommended structure, the second is the
    defect. Counting components alone reported a healthy double rail as one
    gap and a continuity of 0.53. They are told apart by whether each
    component closes on itself -- a closed ring encloses the die centre and
    has a hole; an arc does not.

    Widths come from a local width probe: each rail is opened at a ladder of
    sizes and its width is the largest opening it survives *everywhere*, so it
    is the narrowest place on the rail that decides. That is the right summary
    for a structure whose job is to be continuous, and it is a drawn width --
    what the mask says, not what came out of the line.
    """
    if spec is None:
        return None
    region = reader.region(spec)
    if region.is_empty():
        return None
    u = reader.units
    polygons = list(region.each())
    n_components = len(polygons)

    # A closed ring is one polygon with at least one hole; two concentric
    # rings may be drawn as two such polygons or as one with two holes.
    closed = [p for p in polygons if p.holes() > 0]
    arcs = [p for p in polygons if p.holes() == 0]
    n_rails = sum(max(p.holes(), 1) for p in closed)

    widths = []
    for poly in polygons:
        single = db.Region()
        single.insert(poly)
        widths.append(_narrowest_width(single, u))
    widths = np.array([w for w in widths if w > 0], dtype=float)

    lengths = np.array([abs(p.perimeter()) * u.dbu for p in polygons])
    if closed:
        # Continuity is about the closed part of the ring. Arcs are what is
        # left over when the ring is cut, so they belong in the denominator
        # and not in the numerator.
        continuity = float(sum(abs(p.perimeter()) * u.dbu for p in closed)
                           / lengths.sum()) if lengths.sum() else float("nan")
        n_gaps = len(arcs)
    else:
        # No closed component at all: the ring is entirely in pieces, and the
        # number of gaps is the number of pieces.
        continuity = (float(lengths.max() / lengths.sum())
                      if lengths.sum() else float("nan"))
        n_gaps = len(arcs)

    if len(widths) == 0:
        return CrackstopStructure(
            n_rails=n_rails, rail_width_min_um=float("nan"),
            rail_width_median_um=float("nan"), rail_width_p10_um=float("nan"),
            rail_spacing_um=float("nan"), n_components=n_components,
            continuity_ratio=continuity, n_gaps=n_gaps,
            undefined_reason="no rail survived any opening; the layer may not "
                             "hold a seal ring")

    spacing = float("nan")
    if len(closed) >= 2:
        extent = np.array([max(p.bbox().width(), p.bbox().height()) / 2 * u.dbu
                           for p in closed])
        order = np.argsort(-extent)
        spacing = float(extent[order[0]] - extent[order[1]] - widths.max())

    return CrackstopStructure(
        n_rails=n_rails, rail_width_min_um=float(widths.min()),
        rail_width_median_um=float(np.median(widths)),
        rail_width_p10_um=float(np.percentile(widths, 10)),
        rail_spacing_um=spacing, n_components=n_components,
        continuity_ratio=continuity, n_gaps=n_gaps)


def corner_topology(reader: LayoutReader, spec: LayerSpec | None,
                    die_bbox: BBox | None, *, window_um: float = 100.0
                    ) -> dict:
    """The seal ring at each die corner, which is where Rabie's lever is.

    A ring measured as a whole says nothing about its corners, and the corner
    is where the package load turns. Per corner: how many rails pass through
    the window, how narrow the narrowest gets there, and how many separate
    pieces there are -- a corner that is bridged differently from the sides is
    exactly the topology the lever is about.
    """
    if spec is None or die_bbox is None:
        return {}
    region = reader.region(spec)
    if region.is_empty():
        return {}
    u = reader.units
    out = {}
    corners = {"ll": (die_bbox.xmin, die_bbox.ymin),
               "lr": (die_bbox.xmax, die_bbox.ymin),
               "ul": (die_bbox.xmin, die_bbox.ymax),
               "ur": (die_bbox.xmax, die_bbox.ymax)}
    for name, (cx, cy) in corners.items():
        box = db.Box(u.um_to_dbu(cx - window_um), u.um_to_dbu(cy - window_um),
                     u.um_to_dbu(cx + window_um), u.um_to_dbu(cy + window_um))
        local = region & db.Region(box)
        out[name] = {
            "n_pieces": local.count(),
            "narrowest_um": _narrowest_width(local, u) if not local.is_empty()
            else float("nan"),
            "metal_area_um2": local.area() * u.dbu * u.dbu,
        }
    narrowest = [v["narrowest_um"] for v in out.values()
                 if np.isfinite(v["narrowest_um"])]
    pieces = [v["n_pieces"] for v in out.values()]
    return {
        "per_corner": out,
        "corner_narrowest_um": float(min(narrowest)) if narrowest else float("nan"),
        "corner_piece_count_max": int(max(pieces)) if pieces else 0,
        # A corner drawn differently from the others is the thing worth
        # noticing; identical corners are the ordinary case.
        "corner_asymmetry": (float(max(narrowest) - min(narrowest))
                             if len(narrowest) > 1 else float("nan")),
    }


def _narrowest_width(region, units, *, steps: int = 26) -> float:
    """The narrowest local width anywhere in the shape, in um.

    Bisected on a width check rather than on a morphological opening. Two
    earlier versions of this measured the wrong thing:

    * accepting an opening if *any* part survived returns the widest place on
      the rail, which is the opposite summary for a structure whose job is to
      be continuous;
    * accepting it if 99 % of the area survived misses a pinch, because a
      10 um neck on a 1400 um ring is a fraction of a percent of its area --
      an 8 um ring pinched to 3 um still reported 8 um.

    A width check answers the question directly: the narrowest width is the
    largest threshold at which nothing is flagged. The projected metric is
    used because measured corner to corner every re-entrant corner is two
    edges a vanishing distance apart, so a Euclidian check flags every corner
    of every ring.
    """
    box = region.bbox()
    if region.is_empty():
        return 0.0
    lo, hi = 0.0, float(max(box.width(), box.height()))
    for _ in range(steps):
        mid = (lo + hi) / 2
        w = max(int(mid), 1)
        if region.width_check(w, False, db.Region.Projection).count():
            hi = mid
        else:
            lo = mid
        if hi - lo < 1.0:
            break
    return lo * units.dbu
