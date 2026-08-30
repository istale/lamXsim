"""Package-position features (spec section 9).

These are PACKAGE_POSITION evidence, deliberately not GDS_GEOMETRY. They
exist so that the position-only baseline model can be built, because
"geometry predicts delamination" is only meaningful as a claim relative to
"die position already predicts delamination".
"""
from __future__ import annotations

import numpy as np

from ..evidence import EvidenceClass

POSITION_FEATURES = (
    "distance_to_die_edge",
    "distance_to_nearest_corner",
    "normalized_distance_from_die_center",
)
EVIDENCE_CLASS = EvidenceClass.PACKAGE_POSITION


def extract(grid, die_bbox) -> dict[str, np.ndarray]:
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
