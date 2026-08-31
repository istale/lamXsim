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

    def project(self, polygons: int, scales: int) -> dict:
        """Extrapolate to a layout of *polygons* at *scales* scales."""
        seconds = self.seconds_per_polygon_scale * polygons * scales
        peak = self.baseline_rss_bytes + self.bytes_per_polygon * polygons
        return {"polygons": polygons, "scales": scales,
                "seconds": seconds, "hours": seconds / 3600.0,
                "peak_rss_gb": peak / 1e9}


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
