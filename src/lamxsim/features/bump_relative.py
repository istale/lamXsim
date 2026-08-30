"""Routing orientation resolved against the package loading direction.

Rabie et al. (2018) recommend running the final metal *diagonally* under the
corner bumps. That is a directional statement about the layout relative to the
package, and no scalar distance to a bump can express it: two cells the same
distance from the same bump, one routed radially and one diagonally, are the
same in every feature the engine had until now.

These are GDS_GEOMETRY features -- they describe how the layout is drawn, and
they are the thing a designer can change -- but they cannot be computed
without a bump map. The bump-position confounders stay in the
PACKAGE_POSITION baseline, so an association found here has to beat "how far
from a bump this is" before it means anything.

Orientation is axial, so every difference is taken on doubled angles: a line
at 179 degrees and one at 1 degree differ by 2, not by 178.
"""
from __future__ import annotations

import numpy as np

from ..evidence import EvidenceClass

EVIDENCE_CLASS = EvidenceClass.GDS_GEOMETRY

FEATURES = ("routing_vs_radial_angle_rad", "routing_radial_alignment",
            "routing_diagonality")

#: Below this the window has no dominant routing direction, so an angle
#: measured from it describes rounding rather than layout.
MIN_COHERENCE = 0.15


def extract(routing_direction_rad: np.ndarray, orientation_coherence: np.ndarray,
            bump_radial_direction_rad: np.ndarray, *,
            min_coherence: float = MIN_COHERENCE) -> dict[str, np.ndarray]:
    """Resolve the routing direction against the bump radial direction.

    ``routing_radial_alignment`` is ``cos(2*delta)``: +1 when routing runs
    radially, -1 when it runs tangentially, 0 at 45 degrees.
    ``routing_diagonality`` is ``|sin(2*delta)|``, which peaks at exactly the
    45 degrees Rabie recommends and is zero at both of the other two -- so the
    lever has a feature of its own rather than being the midpoint of one.

    A window whose edges have no dominant direction gets NaN rather than an
    angle: an isotropic window and a deliberately diagonal one sit at the same
    point on the alignment axis, and only coherence separates them.
    """
    routing = np.asarray(routing_direction_rad, dtype=float)
    coherence = np.asarray(orientation_coherence, dtype=float)
    radial = np.asarray(bump_radial_direction_rad, dtype=float)

    delta = routing - radial
    acute = np.abs(np.arctan2(np.sin(2 * delta), np.cos(2 * delta))) / 2.0

    usable = (np.isfinite(routing) & np.isfinite(radial)
              & (coherence >= min_coherence))
    nan = np.full(len(routing), np.nan)

    return {
        "routing_vs_radial_angle_rad": np.where(usable, acute, nan),
        "routing_radial_alignment": np.where(usable, np.cos(2 * acute), nan),
        "routing_diagonality": np.where(usable, np.abs(np.sin(2 * acute)), nan),
    }
