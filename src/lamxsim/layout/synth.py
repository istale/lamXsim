"""Synthetic layout generation (spec sections 25 and 26).

The mandatory experiment is section 26: build patterns whose metal density matches
but whose perimeter / orientation / termination / corner content differs, then
prove the extractor separates them. Every builder here therefore takes the
target density as an explicit argument and derives the drawing parameters
from it, so "same density" is exact by construction rather than by tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import klayout.db as db


@dataclass
class SynthLayout:
    """Accumulates shapes and writes a GDS."""
    dbu: float = 0.001
    top_name: str = "TOP"
    _layout: db.Layout = field(init=False)
    _top: db.Cell = field(init=False)

    def __post_init__(self):
        self._layout = db.Layout()
        self._layout.dbu = self.dbu
        self._top = self._layout.create_cell(self.top_name)

    def _li(self, layer: int, datatype: int = 0) -> int:
        return self._layout.layer(layer, datatype)

    def _u(self, um: float) -> int:
        return int(round(um / self.dbu))

    def add_box(self, layer: int, x0: float, y0: float, x1: float, y1: float,
                datatype: int = 0) -> None:
        self._top.shapes(self._li(layer, datatype)).insert(
            db.Box(self._u(x0), self._u(y0), self._u(x1), self._u(y1))
        )

    def add_polygon(self, layer: int, points, datatype: int = 0) -> None:
        """An arbitrary polygon, for shapes a box cannot express."""
        self._top.shapes(self._li(layer, datatype)).insert(
            db.Polygon([db.Point(self._u(x), self._u(y)) for x, y in points]))

    def write(self, path: str) -> str:
        self._layout.write(str(path))
        return str(path)

    @property
    def layout(self) -> db.Layout:
        return self._layout


# ---------------------------------------------------------------------
# Pattern builders. All take (x0, y0, x1, y1) in um and a target density.
# ---------------------------------------------------------------------

def lines(sl: SynthLayout, layer: int, x0, y0, x1, y1, *,
          pitch: float, density: float, vertical: bool = False) -> None:
    """Parallel lines at *pitch* whose width is set to hit *density* exactly.

    Perimeter density scales as ~2/pitch at fixed density, so sweeping pitch
    while holding density constant is the section 26 experiment for perimeter.
    """
    width = pitch * density
    if vertical:
        n = int((x1 - x0) // pitch)
        for i in range(n):
            xa = x0 + i * pitch
            sl.add_box(layer, xa, y0, xa + width, y1)
    else:
        n = int((y1 - y0) // pitch)
        for i in range(n):
            ya = y0 + i * pitch
            sl.add_box(layer, x0, ya, x1, ya + width)


def segmented_lines(sl: SynthLayout, layer: int, x0, y0, x1, y1, *,
                    pitch: float, density: float,
                    seg_len: float, gap: float) -> None:
    """Lines chopped into segments, with width compensated to hold density.

    A continuous line of width w has the same metal area as segments of width
    w*(seg_len+gap)/seg_len at the same pitch, so density matches the
    :func:`lines` case while the line-end count goes from 2 per line to
    2 per segment.
    """
    duty = seg_len / (seg_len + gap)
    width = pitch * density / duty
    n = int((y1 - y0) // pitch)
    for i in range(n):
        ya = y0 + i * pitch
        xa = x0
        while xa + seg_len <= x1:
            sl.add_box(layer, xa, ya, xa + seg_len, ya + width)
            xa += seg_len + gap


def staircase_lines(sl: SynthLayout, layer: int, x0, y0, x1, y1, *,
                    pitch: float, density: float, step: float) -> None:
    """Lines that jog every *step* um: same density, many more corners.

    Each line is drawn as a run of overlapping boxes offset alternately in y,
    which after merging yields a staircase whose area equals the straight-line
    case (the jog moves metal rather than adding it).
    """
    width = pitch * density
    n = int((y1 - y0) // pitch)
    for i in range(n):
        ya = y0 + i * pitch
        xa = x0
        k = 0
        jog = min(width * 0.5, pitch - width) if pitch > width else 0.0
        while xa < x1:
            xb = min(xa + step, x1)
            dy = jog if (k % 2) else 0.0
            sl.add_box(layer, xa, ya + dy, xb, ya + dy + width)
            # vertical connector so the staircase stays one component
            if k and jog:
                sl.add_box(layer, xa, ya, xa + width, ya + jog + width)
            xa = xb
            k += 1


def solid(sl: SynthLayout, layer: int, x0, y0, x1, y1) -> None:
    sl.add_box(layer, x0, y0, x1, y1)


def via_array(sl: SynthLayout, layer: int, x0, y0, x1, y1, *,
              pitch: float, size: float) -> None:
    ny = int((y1 - y0) // pitch)
    nx = int((x1 - x0) // pitch)
    for j in range(ny):
        for i in range(nx):
            xa, ya = x0 + i * pitch, y0 + j * pitch
            sl.add_box(layer, xa, ya, xa + size, ya + size)


# ---------------------------------------------------------------------
# Section 26 pattern pairs: matched density, one property deliberately varied.
# ---------------------------------------------------------------------

def pair_density_vs_perimeter(tile: float = 100.0, density: float = 0.5,
                              coarse_pitch: float = 20.0,
                              fine_pitch: float = 2.0) -> SynthLayout:
    """A and B: identical metal density, ~10x different perimeter density."""
    sl = SynthLayout()
    lines(sl, 8, 0, 0, tile, tile, pitch=coarse_pitch, density=density)
    lines(sl, 8, tile * 2, 0, tile * 3, tile, pitch=fine_pitch, density=density)
    return sl


def pair_density_vs_orientation(tile: float = 100.0, density: float = 0.5,
                                pitch: float = 4.0) -> SynthLayout:
    """A and B: identical density and perimeter, orthogonal orientation."""
    sl = SynthLayout()
    lines(sl, 8, 0, 0, tile, tile, pitch=pitch, density=density, vertical=False)
    lines(sl, 8, tile * 2, 0, tile * 3, tile, pitch=pitch, density=density, vertical=True)
    return sl


def pair_density_vs_lineend(tile: float = 100.0, density: float = 0.4,
                            pitch: float = 5.0, seg_len: float = 8.0,
                            gap: float = 2.0) -> SynthLayout:
    """A: continuous lines. B: same density, many terminations."""
    sl = SynthLayout()
    lines(sl, 8, 0, 0, tile, tile, pitch=pitch, density=density)
    segmented_lines(sl, 8, tile * 2, 0, tile * 3, tile,
                    pitch=pitch, density=density, seg_len=seg_len, gap=gap)
    return sl


def pair_density_vs_corner(tile: float = 100.0, density: float = 0.4,
                           pitch: float = 5.0, step: float = 5.0) -> SynthLayout:
    """A: straight lines. B: staircase lines, similar density, many corners."""
    sl = SynthLayout()
    lines(sl, 8, 0, 0, tile, tile, pitch=pitch, density=density)
    staircase_lines(sl, 8, tile * 2, 0, tile * 3, tile,
                    pitch=pitch, density=density, step=step)
    return sl


def pair_crosslayer_alignment(tile: float = 100.0, density: float = 0.5,
                              pitch: float = 4.0) -> SynthLayout:
    """A: M8 over M7 aligned. B: same per-layer densities, M7 rotated 90deg."""
    sl = SynthLayout()
    for x_off, vertical_under in ((0.0, False), (tile * 2, True)):
        lines(sl, 8, x_off, 0, x_off + tile, tile, pitch=pitch, density=density)
        lines(sl, 7, x_off, 0, x_off + tile, tile, pitch=pitch, density=density,
              vertical=vertical_under)
    return sl


#: Region-of-interest boxes (x0, y0, x1, y1) for the A and B halves of a pair.
def pair_rois(tile: float = 100.0) -> dict[str, tuple[float, float, float, float]]:
    return {"A": (0.0, 0.0, tile, tile),
            "B": (tile * 2, 0.0, tile * 3, tile)}


# ---------------------------------------------------------------------
# Thin-slice validation die.
# ---------------------------------------------------------------------

def _smooth_field(rng, n: int, blob: int, lo: float, hi: float):
    """A blocky random field smoothed to give spatially correlated blobs."""
    import numpy as np
    coarse = rng.random((max(n // blob, 2), max(n // blob, 2)))
    # bilinear upsample to n x n
    ys = np.linspace(0, coarse.shape[0] - 1, n)
    xs = np.linspace(0, coarse.shape[1] - 1, n)
    y0 = np.clip(np.floor(ys).astype(int), 0, coarse.shape[0] - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, coarse.shape[1] - 2)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    c = coarse
    top = c[y0][:, x0] * (1 - fx) + c[y0][:, x0 + 1] * fx
    bot = c[y0 + 1][:, x0] * (1 - fx) + c[y0 + 1][:, x0 + 1] * fx
    f = top * (1 - fy) + bot * fy
    # np.ptp(f), not f.ptp(): the method was removed from ndarray in NumPy 2.0
    # and this line is reached by every synthetic die, so the failure cascades
    # through the whole suite rather than showing up as one broken feature.
    f = (f - f.min()) / max(float(np.ptp(f)), 1e-12)
    return lo + f * (hi - lo)


def validation_die(path: str, *, die_um: float = 2000.0, block_um: float = 50.0,
                   seed: int = 7,
                   density_range: tuple[float, float] = (0.30, 0.70),
                   pitch_range: tuple[float, float] = (1.5, 20.0)):
    """A die whose metal density and perimeter density vary *independently*.

    For parallel lines of pitch p and width w, metal density is w/p while
    perimeter density is ~2/p. Driving w/p from one random field and p from a
    second, uncorrelated field therefore decouples the two features spatially.
    A pipeline that has silently degenerated into a metal-density detector
    cannot recover a perimeter-driven failure pattern on this die.

    Returns (path, block_table) where block_table has the per-block ground
    truth: x, y, density, pitch, expected perimeter density.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = int(die_um // block_um)
    dens = _smooth_field(rng, n, blob=4, lo=density_range[0], hi=density_range[1])
    pitch = _smooth_field(np.random.default_rng(seed + 1000), n, blob=4,
                          lo=pitch_range[0], hi=pitch_range[1])

    sl = SynthLayout()
    rows = []
    for j in range(n):
        for i in range(n):
            x0, y0 = i * block_um, j * block_um
            d, p = float(dens[j, i]), float(pitch[j, i])
            lines(sl, 8, x0, y0, x0 + block_um, y0 + block_um, pitch=p, density=d)
            rows.append({
                "block_row": j, "block_col": i,
                "x_um": x0 + block_um / 2, "y_um": y0 + block_um / 2,
                "true_density": d, "true_pitch_um": p,
                "expected_perimeter_density": 2.0 / p,
            })
    sl.write(path)
    return path, rows


# ---------------------------------------------------------------------
# Line-end benchmark patterns. Each carries the termination count implied
# by its construction, so competing definitions can be scored rather than
# argued about.
# ---------------------------------------------------------------------

def bench_continuous_lines(sl, layer, n=8, width=1.0, pitch=3.0, length=40.0,
                           x0=0.0, y0=0.0):
    for i in range(n):
        y = y0 + i * pitch
        sl.add_box(layer, x0, y, x0 + length, y + width)
    return 2 * n                      # two tips per line


def bench_segmented_lines(sl, layer, n=6, segs=5, width=1.0, pitch=3.0,
                          seg_len=6.0, gap=2.0, x0=0.0, y0=0.0):
    for i in range(n):
        y = y0 + i * pitch
        for j in range(segs):
            x = x0 + j * (seg_len + gap)
            sl.add_box(layer, x, y, x + seg_len, y + width)
    return 2 * n * segs               # two tips per segment


def bench_solid_plate(sl, layer, size=40.0, x0=0.0, y0=0.0):
    sl.add_box(layer, x0, y0, x0 + size, y0 + size)
    return 0                          # no terminations at all


def bench_fill_squares(sl, layer, n=6, size=1.0, pitch=3.0, x0=0.0, y0=0.0):
    for j in range(n):
        for i in range(n):
            x, y = x0 + i * pitch, y0 + j * pitch
            sl.add_box(layer, x, y, x + size, y + size)
    return 0                          # dummy fill is not a terminated line


def bench_ring(sl, layer, outer=30.0, wall=2.0, x0=0.0, y0=0.0):
    sl.add_box(layer, x0, y0, x0 + outer, y0 + wall)
    sl.add_box(layer, x0, y0 + outer - wall, x0 + outer, y0 + outer)
    sl.add_box(layer, x0, y0, x0 + wall, y0 + outer)
    sl.add_box(layer, x0 + outer - wall, y0, x0 + outer, y0 + outer)
    return 0                          # a closed loop has no free end


def bench_comb(sl, layer, teeth=8, width=1.0, pitch=3.0, tooth_len=15.0,
               x0=0.0, y0=0.0):
    spine = teeth * pitch
    sl.add_box(layer, x0, y0, x0 + spine, y0 + width)
    for i in range(teeth):
        x = x0 + i * pitch
        sl.add_box(layer, x, y0, x + width, y0 + tooth_len)
    # The first tooth sits flush with the left end of the spine, so those two
    # faces merge into an L-bend rather than staying two free ends.
    return teeth + 1                  # one tip per tooth, plus the spine's right end


def bench_staircase(sl, layer, n=5, width=1.0, steps=6, step_len=6.0,
                    rise=3.0, pitch=12.0, x0=0.0, y0=0.0):
    for i in range(n):
        base = y0 + i * pitch
        for k in range(steps):
            x = x0 + k * step_len
            y = base + k * rise
            sl.add_box(layer, x, y, x + step_len, y + width)
            if k:
                sl.add_box(layer, x, y - rise, x + width, y + width)
    return 2 * n                      # jogs are not terminations


def bench_tees(sl, layer, n=4, width=1.0, arm=10.0, pitch=25.0, x0=0.0, y0=0.0):
    for i in range(n):
        cx = x0 + i * pitch + arm
        cy = y0 + arm
        sl.add_box(layer, cx - arm, cy, cx + arm, cy + width)   # crossbar
        sl.add_box(layer, cx, cy - arm, cx + width, cy + width)  # stem
    return 3 * n                      # three free ends per T


LINE_END_BENCH = {
    "continuous lines":  bench_continuous_lines,
    "segmented lines":   bench_segmented_lines,
    "solid plate":       bench_solid_plate,
    "dummy fill array":  bench_fill_squares,
    "closed ring":       bench_ring,
    "comb":              bench_comb,
    "staircase lines":   bench_staircase,
    "T junctions":       bench_tees,
}


def gradient_driver_die(path: str, *, die_um: float = 2000.0, block_um: float = 25.0,
                        seed: int = 11, wavelength_um: float = 500.0,
                        density_range: tuple[float, float] = (0.25, 0.75),
                        pitch_um: float = 4.0):
    """A die whose metal density varies sinusoidally in x.

    Built so that a feature's value and its gradient are orthogonal: for
    ``Q = sin(kx)`` the derivative is ``k cos(kx)``, and sine and cosine are
    uncorrelated over whole periods. Driving simulated failures from the
    gradient therefore tests the spec section 5 warning directly -- a pipeline that
    only ever responds to absolute values cannot recover the driver.
    """
    import numpy as np
    n = int(die_um // block_um)
    lo, hi = density_range
    sl = SynthLayout()
    for i in range(n):
        x0 = i * block_um
        phase = 2 * np.pi * (x0 + block_um / 2) / wavelength_um
        d = lo + (hi - lo) * 0.5 * (1 + np.sin(phase))
        lines(sl, 8, x0, 0.0, x0 + block_um, die_um, pitch=pitch_um, density=d)
    sl.write(path)
    return path


def crosslayer_driver_die(path: str, *, die_um: float = 2000.0,
                          block_um: float = 100.0, seed: int = 21,
                          density: float = 0.5, pitch_um: float = 4.0):
    """A two-layer die where only the *relationship* between layers varies.

    Each layer's orientation is drawn from its own independent random field at
    identical density and pitch, and the mismatch is their exclusive-or. Every
    per-layer scalar is then uninformative about the mismatch by construction:
    knowing that M8 runs vertically here says nothing about whether M7 agrees.
    Only a cross-layer feature can see the structure, which is what makes this
    a real test of spec section 7 rather than a per-layer effect in disguise.

    Returns (path, mismatch_map) where mismatch_map[j][i] is 1 where the
    layers are orthogonal and 0 where they are aligned.
    """
    import numpy as np
    n = int(die_um // block_um)
    upper = (_smooth_field(np.random.default_rng(seed), n, blob=3,
                           lo=0.0, hi=1.0) > 0.5).astype(int)
    lower = (_smooth_field(np.random.default_rng(seed + 500), n, blob=3,
                           lo=0.0, hi=1.0) > 0.5).astype(int)
    mismatch = (upper ^ lower).astype(int)

    sl = SynthLayout()
    for j in range(n):
        for i in range(n):
            x0, y0 = i * block_um, j * block_um
            lines(sl, 8, x0, y0, x0 + block_um, y0 + block_um,
                  pitch=pitch_um, density=density, vertical=bool(upper[j, i]))
            lines(sl, 7, x0, y0, x0 + block_um, y0 + block_um,
                  pitch=pitch_um, density=density, vertical=bool(lower[j, i]))
    sl.write(path)
    return path, mismatch


def bump_array(sl: SynthLayout, layer: int, die_um: float, *, pitch: float,
               size: float, margin: float = 100.0, pi_layer: int | None = None,
               pi_ratio: float = 0.7):
    """A regular C4/bump field, optionally with a PI opening inside each bump.

    Returns the bump centres in um. Bumps are the boundary condition through
    which the package loads the layout, so a synthetic die that is meant to
    exercise package-position confounding needs them present rather than
    implied by distance-to-corner alone.
    """
    import numpy as np
    centres = []
    n = int((die_um - 2 * margin) // pitch) + 1
    for j in range(n):
        for i in range(n):
            cx = margin + i * pitch
            cy = margin + j * pitch
            if cx + size / 2 > die_um or cy + size / 2 > die_um:
                continue
            h = size / 2
            sl.add_box(layer, cx - h, cy - h, cx + h, cy + h)
            if pi_layer is not None:
                hp = size * pi_ratio / 2
                sl.add_box(pi_layer, cx - hp, cy - hp, cx + hp, cy + hp)
            centres.append((cx, cy))
    return np.array(centres, dtype=float)


def crackstop_ring(sl: SynthLayout, layer: int, die_um: float, *,
                   inset: float = 20.0, width: float = 8.0):
    """A seal-ring/crackstop frame just inside the die outline."""
    a, b = inset, die_um - inset
    sl.add_box(layer, a, a, b, a + width)
    sl.add_box(layer, a, b - width, b, b)
    sl.add_box(layer, a, a, a + width, b)
    sl.add_box(layer, b - width, a, b, b)


def packaged_die(path: str, *, die_um: float = 3000.0, block_um: float = 100.0,
                 seed: int = 31, bump_pitch: float = 400.0,
                 bump_size: float = 150.0):
    """A validation die carrying metal, vias, bumps, PI openings and a crackstop.

    Layer map: 8 = M8, 7 = M7, 17 = V7, 60 = bump, 61 = PI opening,
    62 = crackstop. Structured like a real delivery so that package-context
    extraction can be exercised end to end without a production layout; on
    real data only the layer numbers change.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = int(die_um // block_um)
    dens = _smooth_field(rng, n, blob=4, lo=0.30, hi=0.70)
    pitch = _smooth_field(np.random.default_rng(seed + 7), n, blob=4,
                          lo=2.0, hi=20.0)

    sl = SynthLayout()
    for j in range(n):
        for i in range(n):
            x0, y0 = i * block_um, j * block_um
            lines(sl, 8, x0, y0, x0 + block_um, y0 + block_um,
                  pitch=float(pitch[j, i]), density=float(dens[j, i]))
            lines(sl, 7, x0, y0, x0 + block_um, y0 + block_um,
                  pitch=float(pitch[j, i]) * 1.5, density=0.4, vertical=True)
            via_array(sl, 17, x0 + 5, y0 + 5, x0 + block_um - 5,
                      y0 + block_um - 5, pitch=float(pitch[j, i]) * 2.0,
                      size=min(float(pitch[j, i]) * 0.4, 2.0))
    bumps = bump_array(sl, 60, die_um, pitch=bump_pitch, size=bump_size,
                       pi_layer=61)
    crackstop_ring(sl, 62, die_um)
    sl.write(path)
    return path, bumps


def radial_routing_die(path: str, *, die_um: float = 3000.0,
                       block_um: float = 150.0, bump_pitch: float = 500.0,
                       bump_size: float = 180.0, seed: int = 41,
                       pitch_um: float = 6.0, density: float = 0.5):
    """A die where routing direction, not density, varies with the bump frame.

    Every block carries the same metal density and pitch; only the direction
    changes, drawn from a random field that is independent of position. Blocks
    are therefore identical in every scalar feature the engine has, and differ
    only once the routing direction is resolved against the direction of the
    nearest bump -- which is what Rabie's diagonal final-metal recommendation
    is about.

    Returns (path, bumps, block_direction) where block_direction is the drawn
    orientation of each block in radians.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = int(die_um // block_um)
    choice = rng.integers(0, 2, (n, n))          # horizontal or vertical
    sl = SynthLayout()
    for j in range(n):
        for i in range(n):
            x0, y0 = i * block_um, j * block_um
            lines(sl, 8, x0, y0, x0 + block_um, y0 + block_um,
                  pitch=pitch_um, density=density, vertical=bool(choice[j, i]))
    bumps = bump_array(sl, 60, die_um, pitch=bump_pitch, size=bump_size,
                       pi_layer=61)
    crackstop_ring(sl, 62, die_um)
    sl.write(path)
    return path, bumps, np.where(choice == 1, np.pi / 2, 0.0)



def shape_variation_die(path: str, *, die_um: float = 1200.0,
                        pitch: float = 300.0, margin: float = 150.0,
                        pad_r: float = 60.0, odd_sites=((0, 0),)):
    """A die whose pad and PI-opening *shapes* vary from site to site.

    The other synthetic dies vary density, pitch and routing direction, which
    is what the window features need. None of them varies the shape of an
    individual pad or opening, so the shape channels have nothing to rank and
    correctly report nothing -- which is not the same as being tested.

    Here most sites carry an octagonal pad over a square-ish opening, and the
    sites named in ``odd_sites`` carry a square pad over an elongated opening.
    That leaves a small genuine extreme rather than a half-and-half split,
    which would put half the cells at the top value and be the top 50 %.

    Layer map: 8 = M8, 7 = M7, 60 = bump, 61 = PI opening, 64 = pad,
    62 = crackstop.
    """
    import math

    import numpy as np

    sl = SynthLayout()
    n = int((die_um - 2 * margin) // pitch) + 1
    for j in range(n):
        for i in range(n):
            cx, cy = margin + i * pitch, margin + j * pitch
            odd = (i, j) in odd_sites

            if odd:
                sl.add_box(64, cx - pad_r, cy - pad_r, cx + pad_r, cy + pad_r)
            else:
                pts = [(cx + pad_r * math.cos(k * math.pi / 4 + math.pi / 8),
                        cy + pad_r * math.sin(k * math.pi / 4 + math.pi / 8))
                       for k in range(8)]
                sl.add_polygon(64, pts)

            sl.add_box(60, cx - pad_r * 0.7, cy - pad_r * 0.7,
                       cx + pad_r * 0.7, cy + pad_r * 0.7)
            if odd:
                sl.add_box(61, cx - pad_r * 0.8, cy - pad_r * 0.2,
                           cx + pad_r * 0.8, cy + pad_r * 0.2)
            else:
                sl.add_box(61, cx - pad_r * 0.4, cy - pad_r * 0.4,
                           cx + pad_r * 0.4, cy + pad_r * 0.4)

    block = 100.0
    for j in range(int(die_um // block)):
        for i in range(int(die_um // block)):
            x0, y0 = i * block, j * block
            lines(sl, 8, x0, y0, x0 + block, y0 + block, pitch=6.0, density=0.5)
            lines(sl, 7, x0, y0, x0 + block, y0 + block, pitch=9.0,
                  density=0.4, vertical=True)
    crackstop_ring(sl, 62, die_um)
    sl.write(path)
    return path
