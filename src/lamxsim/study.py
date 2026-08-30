"""Study manifest: the human-supplied semantics that give GDS physical meaning.

GDS layer numbers are identifiers. Which of them is top metal, which is a via,
which is a bump, what counts as a routing line rather than a power strap, and
where inspection actually looked -- none of that is in the file. It has to be
declared, and the declaration has to reach the run metadata so a result can be
read against the assumptions that produced it.

Loading a manifest is therefore also a validation step: it fails on a layer
that is not in the layout, and it records every semantic that was left
unspecified as an explicit gap rather than a default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .labels.inspection import InspectionFootprint
from .labels.package_context import PackageLayers
from .layout.reader import BBox, LayerSpec, LayoutReader


def _spec(entry) -> LayerSpec | None:
    if not entry:
        return None
    return LayerSpec(entry["name"], int(entry["layer"]),
                     int(entry.get("datatype", 0)))


@dataclass
class LineRule:
    """Per-layer routing widths, from the PDK rather than from the geometry."""
    min_width_um: float
    line_max_width_um: float


@dataclass
class StudyManifest:
    metal_layers: list[LayerSpec]
    via_layers: dict[str, LayerSpec] = field(default_factory=dict)
    package_layers: PackageLayers = field(default_factory=PackageLayers)
    line_rules: dict[str, LineRule] = field(default_factory=dict)
    top_cell: str | None = None
    die_outline_um: list[float] | None = None
    footprint_spec: dict = field(default_factory=dict)
    min_coverage: float = 0.5
    fiducials: str | None = None
    allow_reflection: bool = True
    scales_um: tuple[float, ...] = (25, 50, 100, 250, 500, 1000)
    pair_selection: str = "adjacent_and_top"
    with_gradients: bool = True
    n_permutations: int = 999
    enforce_scale_gate: bool = True
    gaps: list[str] = field(default_factory=list)
    source: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "StudyManifest":
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        layout = cfg.get("layout", {})
        inspect = cfg.get("inspection", {})
        reg = cfg.get("registration", {})
        ana = cfg.get("analysis", {})

        metals = [_spec(e) for e in layout.get("metal_layers") or []]
        if not metals:
            raise ValueError(
                f"{path}: layout.metal_layers is required. Without an ordered "
                "metal stack there is no layer identity to preserve and no "
                "'top versus underlying' to define.")

        rules = {}
        for name, r in (layout.get("line_rules") or {}).items():
            rules[name] = LineRule(float(r["min_width_um"]),
                                   float(r["line_max_width_um"]))

        pkg_cfg = layout.get("package_layers") or {}
        package = PackageLayers(
            bump=_spec(pkg_cfg.get("bump")), pad=_spec(pkg_cfg.get("pad")),
            pi_opening=_spec(pkg_cfg.get("pi_opening")),
            crackstop=_spec(pkg_cfg.get("crackstop")))

        gaps = []
        for m in metals:
            if m.name not in rules:
                gaps.append(
                    f"no line_rules for {m.name}: the line-end width cutoff "
                    "will be inferred from the shortest edge in the design, "
                    "which on a layout with dummy fill is the fill edge")
        if not layout.get("die_outline_um"):
            gaps.append(
                "no die_outline_um: the bounding box of top-cell geometry is "
                "being used as the die boundary, and package-position features "
                "are measured from it")
        if not (layout.get("via_layers") or {}):
            gaps.append("no via_layers: via density is a tier-1 feature "
                        "(Vanstreels 2020, Zahedmanesh 2019) and will be absent")
        if not reg.get("fiducials"):
            gaps.append(
                "no registration fiducials: positional uncertainty must come "
                "from the failure file, and no scale can be certified from "
                "measured registration")

        return cls(
            metal_layers=metals,
            via_layers={k: _spec(v) for k, v in
                        (layout.get("via_layers") or {}).items() if v},
            package_layers=package, line_rules=rules,
            top_cell=layout.get("top_cell"),
            die_outline_um=layout.get("die_outline_um"),
            footprint_spec=inspect.get("footprint") or {},
            min_coverage=float(inspect.get("min_coverage", 0.5)),
            fiducials=reg.get("fiducials"),
            allow_reflection=bool(reg.get("allow_reflection", True)),
            scales_um=tuple(ana.get("scales_um", (25, 50, 100, 250, 500, 1000))),
            pair_selection=ana.get("pair_selection", "adjacent_and_top"),
            with_gradients=bool(ana.get("with_gradients", True)),
            n_permutations=int(ana.get("n_permutations", 999)),
            enforce_scale_gate=bool(ana.get("enforce_scale_gate", True)),
            gaps=gaps, source=str(path))

    # -- resolution against a real layout ----------------------------
    def validate_against(self, reader: LayoutReader) -> list[str]:
        """Fail on layers the manifest names but the layout does not contain."""
        present = set(reader.available_layers())
        missing = []
        named = ([(m, "metal") for m in self.metal_layers]
                 + [(v, f"via for {k}") for k, v in self.via_layers.items()]
                 + [(v, f"package/{k}") for k, v in vars(self.package_layers).items()
                    if v is not None])
        for spec, role in named:
            if spec.key not in present:
                missing.append(f"{role} layer {spec} is not in the layout")
        if missing:
            raise ValueError(
                f"{self.source}: " + "; ".join(missing)
                + f". Layers present: {sorted(present)}")
        return missing

    def die_bbox(self, reader: LayoutReader) -> BBox:
        if self.die_outline_um:
            x0, y0, x1, y1 = self.die_outline_um
            return BBox(float(x0), float(y0), float(x1), float(y1))
        return reader.bbox()

    def footprint(self, reader: LayoutReader, bbox: BBox) -> InspectionFootprint | None:
        spec = self.footprint_spec or {}
        if spec.get("gds_layer"):
            return InspectionFootprint.from_gds_layer(reader, _spec(spec["gds_layer"]))
        if spec.get("rectangles"):
            return InspectionFootprint.from_rectangles(
                spec["rectangles"], dbu=reader.units.dbu, source="manifest_rectangles")
        if spec.get("full_die"):
            return InspectionFootprint.full_die(bbox, str(spec["full_die"]),
                                                dbu=reader.units.dbu)
        return None

    def line_end_w_max_um(self) -> float | None:
        """The widest routing line across the stack, or None if unspecified."""
        if not self.line_rules:
            return None
        return max(r.line_max_width_um for r in self.line_rules.values())

    def report(self) -> dict:
        return {
            "source": self.source,
            "metal_layers": [str(m) for m in self.metal_layers],
            "via_layers": {k: str(v) for k, v in self.via_layers.items()},
            "package_layers": {k: (str(v) if v else None)
                               for k, v in vars(self.package_layers).items()},
            "line_rules": {k: vars(v) for k, v in self.line_rules.items()},
            "die_outline_um": self.die_outline_um,
            "min_coverage": self.min_coverage,
            "scales_um": list(self.scales_um),
            "pair_selection": self.pair_selection,
            "enforce_scale_gate": self.enforce_scale_gate,
            "declared_gaps": self.gaps,
        }
