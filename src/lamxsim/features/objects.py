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
    """The objects on a layer, in the encoding the manifest declares.

    ``positive`` -- the polygon is the object.
    ``film_holes`` -- the polygon is the film and the objects are its holes.
    ``positive_openings`` -- the polygons are the openings, drawn directly.

    The last two used to be one value, ``opening``, resolved by looking at the
    geometry: holes if any polygon had one, the polygons themselves otherwise.
    That is a guess, and it fails silently on a layer carrying both encodings
    -- one film with holes beside a few directly drawn openings -- where every
    standalone opening is discarded because a hole was found somewhere else.
    A layer's encoding is a fact about how it was drawn, so it is declared;
    where the declaration and the geometry disagree, the run stops.
    """
    region = reader.region(spec)
    if polarity == "positive":
        return region

    holes = db.Region()
    solid = db.Region()
    for poly in region.each():
        if poly.holes():
            for i in range(poly.holes()):
                holes.insert(db.Polygon(list(poly.each_point_hole(i))))
        else:
            solid.insert(poly)

    if polarity == "film_holes":
        if holes.is_empty():
            raise ValueError(
                f"{spec} is declared film_holes and no polygon on it has a "
                "hole, so there are no openings to measure. Either the layer "
                "draws its openings directly -- declare positive_openings -- "
                "or the film is not on this layer.")
        if not solid.is_empty():
            raise ValueError(
                f"{spec} is declared film_holes and carries {solid.count()} "
                "polygon(s) with no hole beside the film. Those are either "
                "openings drawn directly, in which case the layer mixes two "
                "encodings and has to be split, or they are something else "
                "entirely. Guessing here silently discards them.")
        holes.merge()
        return holes

    if polarity == "positive_openings":
        if not holes.is_empty():
            raise ValueError(
                f"{spec} is declared positive_openings and carries a polygon "
                "with a hole. If that polygon is the film, declare "
                "film_holes; if the layer mixes both encodings it has to be "
                "split, because one reading discards the other's objects.")
        return region

    raise ValueError(
        f"unknown polarity {polarity!r} for {spec}; declare positive, "
        "film_holes or positive_openings")


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
MATCH_RULES = ("centroid_containment", "full_containment", "nearest",
               "one_to_one")

#: Rejected outright rather than mapped to one of the two. The bare word was
#: this module's own name for what is really centroid containment, and the
#: comments claimed the stricter thing -- a bump can hang most of the way out
#: of a pad and still be matched, as long as its centre is inside. Both
#: semantics are defensible and they differ exactly on the offset placements
#: worth looking at, so the manifest has to choose.
AMBIGUOUS_MATCH_RULES = {
    "containment": "centroid_containment or full_containment -- the first "
                   "asks whether the secondary's centre is inside the "
                   "primary, the second whether all of it is, and they "
                   "disagree on precisely the offset placements a "
                   "concentricity study is about"}


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

    Containment comes in two flavours and the manifest has to say which.
    ``centroid_containment`` asks whether the secondary's centre lies inside
    the primary polygon; ``full_containment`` asks whether all of it does.
    A bump hanging most of the way out of its pad passes the first and fails
    the second, which is exactly the placement a concentricity study is
    about. The bare name ``containment`` is refused rather than mapped to
    either. (Both are polygon tests: an earlier version compared the centroid
    against a circle of the primary's equivalent area, so a 200x10 um bar pad
    and a 40x40 um square -- same equivalent radius, completely different sets
    of points -- were treated alike.)
    """
    if rule in AMBIGUOUS_MATCH_RULES:
        raise ValueError(f"the matching rule {rule!r} is ambiguous; declare "
                         f"{AMBIGUOUS_MATCH_RULES[rule]}")
    if rule not in MATCH_RULES:
        raise ValueError(f"unknown object matching rule {rule!r}; "
                         f"declare one of {list(MATCH_RULES)} in the manifest")
    if not primaries or not secondaries:
        return []
    if rule.endswith("containment") and polygons is None:
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

        if rule.endswith("containment"):
            if rule == "centroid_containment":
                inside = [i for i in range(len(secondaries))
                          if _point_in_polygon(sx[i], sy[i],
                                               primary_polys[index], dbu)]
            else:
                inside = [i for i in range(len(secondaries))
                          if _fully_inside(secondary_polys[i],
                                           primary_polys[index])]
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


def _fully_inside(secondary_poly, primary_poly) -> bool:
    """Does none of the secondary lie outside the primary?"""
    return (db.Region(secondary_poly) - db.Region(primary_poly)).is_empty()


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

    from .grid import point_accumulate

    ox = np.array([o.x_um for o in objects])
    oy = np.array([o.y_um for o in objects])
    out[f"{prefix}_count"] = point_accumulate(grid, ox, oy)
    for name, series in values.items():
        # Objects whose value is undefined are excluded from both the sum and
        # its divisor, so a pad with no matched bump does not drag the mean of
        # the pads that have one towards zero.
        finite = np.isfinite(series)
        counts = point_accumulate(grid, ox[finite], oy[finite])
        sums = point_accumulate(grid, ox[finite], oy[finite], series[finite])
        out[f"{prefix}_{name}"] = np.where(counts > 0,
                                           sums / np.maximum(counts, 1.0),
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


def crackstop_width_map(reader: LayoutReader, spec: LayerSpec | None, grid,
                        *, probe_multiple: float = 2.0) -> "np.ndarray | None":
    """The seal ring's local drawn width, per grid cell, NaN off the ring.

    This replaces a quadrant broadcast that could not produce a candidate at
    all. Giving each quadrant its corner's width puts a quarter of the die on
    one value, and a quarter of the cells tied sit at the 88th percentile --
    below the 95th the atlas selects at, however narrow that corner is. The
    channel was live, scored, and structurally incapable of reporting
    anything.

    A width check returns, for every place narrower than its threshold, the
    pair of edges and the distance between them. Probed at a multiple of the
    ring's own median width, every location on the ring is reported with its
    actual width, so this is the local width and not a summary of it. Cells
    the ring does not pass through stay NaN, so the ranking compares ring to
    ring rather than ring to empty silicon -- and a pinch is located where it
    is, not attributed to a quadrant.

    The projected metric, again: measured corner to corner every re-entrant
    corner is two edges a vanishing distance apart, so a Euclidian check
    reports every corner of every ring as its narrowest place.
    """
    if spec is None:
        return None
    region = reader.region(spec)
    if region.is_empty():
        return None
    u = reader.units

    # A typical width, not the narrowest one, and not a single width check
    # over the whole ring. Two attempts failed here for different reasons and
    # both are worth keeping written down:
    #
    # * probing at twice the *narrowest* width sees only the pinch, so a ring
    #   pinched from 8um to 3um produced one cell -- and a percentile rank
    #   needs two values, so the channel reported nothing on exactly the die
    #   it was built to find;
    # * a single width check over the ring returns one edge pair per
    #   uninterrupted run, so a 1160um rail contributes one sample at its
    #   midpoint. Four samples for four sides is not a map.
    #
    # For a thin closed ring, area is width times centreline length and
    # perimeter is twice that length, so 2*area/perimeter recovers the typical
    # width and is unmoved by a short pinch. The map is then built per cell.
    area = region.area() * u.dbu * u.dbu
    perimeter = sum(abs(poly.perimeter()) for poly in region.each()) * u.dbu
    typical = (2.0 * area / perimeter) if perimeter > 0 else 0.0
    if typical <= 0:
        return None

    # One width check over the whole ring, and each violation is assigned to
    # the cells its own **corridor** covers -- the quadrilateral between the
    # two facing edges, which is the strip of material the measurement is
    # about.
    #
    # Not the corridor's bounding box, which was the first version of this. On
    # an axis-aligned rail the two are nearly the same, so a square seal ring
    # gave the right answer and every test passed. On a 45-degree ring they
    # are not: the bounding box of a diagonal corridor is a large square whose
    # interior is empty silicon, and a uniform diamond ring marked 900 of its
    # 1024 finite cells off the ring -- values on cells the crackstop does not
    # touch, under a docstring promising NaN off the ring. Chamfered and
    # diamond seal rings are ordinary, so this was wrong on real layouts and
    # right on every fixture.
    threshold = max(u.um_to_dbu(typical * probe_multiple), 2)
    out = np.full(len(grid), np.nan)
    dbu = u.dbu
    for pair in region.width_check(threshold, False, db.Region.Projection).each():
        width = pair.distance() * dbu
        corridor = db.Region(pair.polygon(0))
        box = corridor.bbox()
        x0, y0 = box.left * dbu, box.bottom * dbu
        x1, y1 = box.right * dbu, box.top * dbu
        for cell in grid.cells:
            if cell.x1 <= x0 or cell.x0 >= x1 or cell.y1 <= y0 or cell.y0 >= y1:
                continue
            cell_box = db.Region(db.Box(
                u.um_to_dbu(cell.x0), u.um_to_dbu(cell.y0),
                u.um_to_dbu(cell.x1), u.um_to_dbu(cell.y1)))
            if (corridor & cell_box).is_empty():
                continue
            current = out[cell.cell_id]
            # The narrowest place in the cell, not the mean: a rail that is
            # wide for most of a window and pinched in one spot is pinched.
            if not np.isfinite(current) or width < current:
                out[cell.cell_id] = width
    return out if np.isfinite(out).any() else None


def crackstop_gap_map(reader: LayoutReader, spec: LayerSpec | None, grid,
                      *, max_gap_um: float | None = None,
                      support: "np.ndarray | None" = None
                      ) -> "np.ndarray | None":
    """Where the seal ring is interrupted, and by how much, per grid cell.

    A break cannot be found by the width map: where the ring is absent there
    is nothing to measure, so the cell is NaN and NaN is not an extreme. The
    ring's continuity ratio and gap count say a break exists somewhere and
    cannot say where, so a cut ring produced whole-ring numbers and no
    locatable candidate.

    The gap itself is the space between two rail ends, which is a spacing
    check on the ring against itself. Each violation is assigned to the cells
    its corridor covers, exactly as the width map is, so the candidate lands
    on the break rather than near it.

    The value is the gap length in um -- larger is worse, the opposite
    direction to the width map, which is what the channel's per-input
    direction is for -- and it is **zero** where the ring is continuous, not
    NaN. That matters: a die with one break has one cell with a gap, and a
    percentile rank over a single value is undefined, so a map that was NaN
    everywhere else could never rank the break. Zero on the rest of the ring
    makes the population "every point on the ring", where a break is the
    extreme it should be. ``support`` supplies that population, normally the
    width map's own.
    """
    if spec is None:
        return None
    region = reader.region(spec)
    if region.is_empty():
        return None
    u = reader.units

    area = region.area() * u.dbu * u.dbu
    perimeter = sum(abs(poly.perimeter()) for poly in region.each()) * u.dbu
    typical = (2.0 * area / perimeter) if perimeter > 0 else 0.0
    if typical <= 0:
        return None
    # The probe has to reach past the rail width -- a 40 um break in an 8 um
    # rail is five times it -- and stop well short of the ring's own inner
    # opening, which is the die and is not a defect. Twenty rail widths, capped
    # at a quarter of the ring's shorter side.
    box = region.bbox()
    shorter = min(box.width(), box.height()) * u.dbu
    ceiling = shorter / 4.0
    limit = max_gap_um if max_gap_um is not None else min(typical * 20.0, ceiling)
    if limit <= 0 or limit >= shorter:
        return None
    threshold = max(u.um_to_dbu(limit), 2)

    out = (np.where(np.isfinite(support), 0.0, np.nan)
           if support is not None else np.full(len(grid), np.nan))
    dbu = u.dbu
    # notch_check is spacing within one polygon, which is what a break in a
    # ring is: cutting a closed ring leaves one C-shaped polygon, not two, so
    # an inter-polygon spacing check finds nothing. space_check adds the case
    # where the ring is drawn, or cut, into separate pieces.
    violations = list(region.notch_check(threshold, False,
                                         db.Region.Projection).each())
    if region.count() > 1:
        violations += list(region.space_check(threshold, False,
                                              db.Region.Projection).each())
    for pair in violations:
        gap = pair.distance() * dbu
        corridor = db.Region(pair.polygon(0))
        box = corridor.bbox()
        x0, y0 = box.left * dbu, box.bottom * dbu
        x1, y1 = box.right * dbu, box.top * dbu
        for cell in grid.cells:
            if cell.x1 <= x0 or cell.x0 >= x1 or cell.y1 <= y0 or cell.y0 >= y1:
                continue
            cell_box = db.Region(db.Box(
                u.um_to_dbu(cell.x0), u.um_to_dbu(cell.y0),
                u.um_to_dbu(cell.x1), u.um_to_dbu(cell.y1)))
            if (corridor & cell_box).is_empty():
                continue
            current = out[cell.cell_id]
            if not np.isfinite(current) or gap > current:
                out[cell.cell_id] = gap
    return out if (np.isfinite(out) & (out > 0)).any() else None


def corner_topology(reader: LayoutReader, spec: LayerSpec | None,
                    die_bbox: BBox | None, *, window_um: float | None = None
                    ) -> dict:
    """The seal ring at each of *its own* corners.

    A ring measured as a whole says nothing about its corners, and the corner
    is where the package load turns.

    The window is placed on the ring's corners and sized from the ring, not
    fixed at the die corner. A ring inset 150 um from the die edge fell
    entirely outside a hardcoded 100 um window at the die corner: all four
    corners reported zero pieces and a NaN width, the channel stayed
    "available", and the run said nothing about it. When the window still
    catches nothing the result says so rather than returning quiet NaNs.
    """
    if spec is None:
        return {}
    region = reader.region(spec)
    if region.is_empty():
        return {}
    u = reader.units
    box = region.bbox()
    ring = BBox(box.left * u.dbu, box.bottom * u.dbu,
                box.right * u.dbu, box.top * u.dbu)
    if window_um is None:
        # A tenth of the shorter side: large enough to hold the corner turn
        # and its approach, small enough not to be most of a side.
        window_um = max(min(ring.width, ring.height) / 10.0, 1.0)
    out = {}
    corners = {"ll": (ring.xmin, ring.ymin), "lr": (ring.xmax, ring.ymin),
               "ul": (ring.xmin, ring.ymax), "ur": (ring.xmax, ring.ymax)}
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
        "window_um": window_um,
        "undefined_reason": ("" if narrowest else
                             f"no crackstop geometry fell inside a "
                             f"{window_um:g}um window at any corner of the "
                             f"ring itself, so nothing corner-resolved was "
                             f"measured"),
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
