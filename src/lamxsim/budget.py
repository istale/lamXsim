"""Measure the cost of extraction on a clip, and project it to a full chip.

The runtime question cannot be answered from this repository's synthetic
dies. Measured here, the atlas costs about 79 us and 2 kB of peak memory per
polygon per scale -- flat across a sevenfold range of polygon count -- but
those constants come from Manhattan geometry with no hierarchy on one machine.
A production layout differs in polygon density, hierarchy depth, the fraction
of non-Manhattan geometry and the machine it runs on, and the projection is
linear in a constant nobody has measured for it.

So this measures the constant on the user's own layout and projects with it,
and reports what the projection is sensitive to. It answers "can this run at
all", which for a full chip is a memory question rather than a time one: at
2 kB per polygon a hundred million polygons is 200 GB, and there is no tiling
in the Python path to reduce it. Time is the easy constraint.
"""
from __future__ import annotations

import resource
import time
from dataclasses import dataclass

from .layout.reader import LayerSpec, LayoutReader


def _peak_rss_bytes() -> int:
    """Peak resident set size, in bytes.

    ru_maxrss is bytes on macOS and kilobytes on Linux, which is a difference
    of 1024 in the headline number of this whole command. It is decided by
    the platform rather than sniffed from the value, because a small process
    on Linux and a large one on macOS produce the same figure.
    """
    import sys

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


@dataclass
class Measurement:
    """What one extraction actually cost, and what it implies."""
    polygons: int
    cells: int
    scales: int
    seconds: float
    peak_rss_bytes: int
    baseline_rss_bytes: int

    @property
    def seconds_per_polygon_scale(self) -> float:
        denom = max(self.polygons * self.scales, 1)
        return self.seconds / denom

    @property
    def bytes_per_polygon(self) -> float:
        used = max(self.peak_rss_bytes - self.baseline_rss_bytes, 0)
        return used / max(self.polygons, 1)

    def project(self, polygons: int, scales: int,
                exponent: float = 1.0) -> dict:
        """Extrapolate to a layout of *polygons* at *scales* scales.

        ``exponent`` is how the *time* grows with polygon count. It is not 1.
        The windowed extractors clip the layer once per grid row and then once
        per window against that row, so the work is rows times polygons; on a
        layout both grow with die area, and measured across a sixty-fourfold
        range the cost rose 4.8x, then 5.3x, then 5.9x for each fourfold rise
        in polygons -- a local exponent climbing from 1.14 to 1.28. Projecting
        linearly from a small clip understates a full chip by more than an
        order of magnitude, which is the difference between an overnight job
        and a fortnight.

        Memory stays linear: it is the merged layers held at once.
        """
        ratio = polygons / max(self.polygons, 1)
        seconds = self.seconds * (ratio ** exponent) * (scales / max(self.scales, 1))
        peak = self.baseline_rss_bytes + self.bytes_per_polygon * polygons
        return {"polygons": polygons, "scales": scales, "exponent": exponent,
                "seconds": seconds, "hours": seconds / 3600.0,
                "peak_rss_gb": peak / 1e9}


def fit_exponent(measurements: "list[Measurement]") -> "tuple[float, str]":
    """How time grows with polygon count, from two or more clips.

    A log-log slope through the measurements. With one clip there is nothing
    to fit and the caller has to say so rather than assume 1: the assumption
    is wrong in the direction that matters, and it is wrong by a factor that
    grows with how far the projection reaches.
    """
    import numpy as np

    usable = [m for m in measurements if m.polygons > 0 and m.seconds > 0]
    if len(usable) < 2:
        return 1.0, ("only one clip was measured, so the growth of time with "
                     "polygon count could not be fitted and 1.0 was assumed. "
                     "It is not 1.0 -- the windowed extractors cost rows times "
                     "polygons, and both grow with die area -- so the time "
                     "below is a lower bound, and one that gets weaker the "
                     "further the projection reaches. Measure a second, larger "
                     "clip to fit it")
    x = np.log([m.polygons for m in usable])
    y = np.log([m.seconds / max(m.scales, 1) for m in usable])
    slope = float(np.polyfit(x, y, 1)[0])
    spread = max(m.polygons for m in usable) / min(m.polygons for m in usable)
    return slope, (f"fitted over {len(usable)} clips spanning {spread:.0f}x in "
                   "polygon count")


def count_polygons(gds_path: str, manifest) -> tuple[int, dict]:
    """Merged polygons on every layer the manifest analyses.

    Merged, because that is what the extractors see: a layer drawn as ten
    thousand abutting rectangles is one polygon to them, and counting the
    drawn shapes instead would overstate the cost by whatever the merge
    removes.
    """
    reader = LayoutReader(gds_path, top_cell=manifest.top_cell)
    per_layer = {}
    for spec in manifest.metal_layers:
        per_layer[spec.name] = reader.region(spec).count()
    for name, spec in manifest.via_layers.items():
        per_layer[spec.name] = reader.region(spec).count()
    for kind, spec in vars(manifest.package_layers).items():
        if spec is not None:
            per_layer[spec.name] = reader.region(spec).count()
    return sum(per_layer.values()), per_layer


def measure(gds_path: str, manifest) -> Measurement:
    """Run the atlas once and record what it cost."""
    from . import atlas as atlas_mod
    from .features.grid import build_multiscale

    baseline = _peak_rss_bytes()
    polygons, _ = count_polygons(gds_path, manifest)
    reader = LayoutReader(gds_path, top_cell=manifest.top_cell)
    grids = build_multiscale(reader.bbox(), manifest.scales_um)
    cells = sum(len(g) for g in grids.values())

    start = time.time()
    atlas_mod.build(gds_path, manifest)
    elapsed = time.time() - start

    return Measurement(polygons=polygons, cells=cells, scales=len(grids),
                       seconds=elapsed, peak_rss_bytes=_peak_rss_bytes(),
                       baseline_rss_bytes=baseline)
