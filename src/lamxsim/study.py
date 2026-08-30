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

from .labels.inspection import FootprintSet, InspectionFootprint
from .labels.package_context import PackageLayers
from .layout.reader import BBox, LayerSpec, LayoutReader


def _spec(entry) -> LayerSpec | None:
    if not entry:
        return None
    return LayerSpec(entry["name"], int(entry["layer"]),
                     int(entry.get("datatype", 0)))


#: Conditions the literature shows change the energy release rate and which no
#: GDS contains. Li et al. (2023, 2025) vary EMC thickness, underfill CTE and
#: the PI opening; Zahedmanesh & Vanstreels (2019) make the result depend on
#: material stiffness throughout. If any of these varies across the samples in
#: a study, an apparent geometry effect may be standing in for the variation.
PACKAGE_PROCESS_CONDITIONS = (
    "emc_thickness_um", "underfill_cte_ppm_k", "underfill_modulus_gpa",
    "underfill_tg_c", "reflow_profile", "thermal_cycle_condition",
    "dielectric_stack", "bump_material", "package_type",
    "inspection_method", "inspection_sensitivity_um",
)


@dataclass
class SampleConditions:
    """How each package/process condition is being handled.

    Every condition must be one of:

    * ``fixed`` -- held constant across the study, with the value recorded;
    * ``stratified`` -- varies, and the analysis is split on it;
    * ``covariate`` -- varies, and it enters the baseline model;
    * ``unknown`` -- not recorded, which is a stated limitation rather than an
      absence of one.

    Nothing here can be measured from the layout, so the software cannot
    check the declarations -- only refuse to let them go unmade.
    """
    fixed: dict = field(default_factory=dict)
    stratified: tuple = ()
    covariate: tuple = ()

    def status(self, condition: str) -> str:
        if condition in self.fixed:
            return "fixed"
        if condition in self.stratified:
            return "stratified"
        if condition in self.covariate:
            return "covariate"
        return "unknown"

    def undeclared(self) -> list[str]:
        return [c for c in PACKAGE_PROCESS_CONDITIONS
                if self.status(c) == "unknown"]

    def validate(self) -> None:
        """Reject a condition claimed in more than one way."""
        seen: dict[str, list[str]] = {}
        for role, names in (("fixed", tuple(self.fixed)),
                            ("stratified", self.stratified),
                            ("covariate", self.covariate)):
            for name in names:
                seen.setdefault(name, []).append(role)
        clashes = {k: v for k, v in seen.items() if len(v) > 1}
        if clashes:
            raise ValueError(
                f"each package/process condition takes exactly one role; "
                f"{clashes} are claimed in more than one. A condition cannot "
                "be held fixed and stratified at the same time.")

    def check_against(self, failures) -> list[str]:
        """Check the declarations against the failure table.

        The software cannot measure EMC thickness, so it cannot verify that a
        condition declared fixed really was. What it can check is that the
        declaration is not contradicted by the data in hand: a condition said
        to be fixed that varies across the file, and one said to be
        stratified or a covariate with no column to read.
        """
        notes: list[str] = []
        table = failures.table
        die_keys = failures.die_keys()

        for name, value in self.fixed.items():
            if name not in table:
                notes.append(
                    f"{name} is declared fixed at {value!r} but the failure "
                    "file carries no such column, so nothing contradicts it "
                    "and nothing confirms it either")
                continue
            observed = sorted(table[name].dropna().astype(str).unique())
            if len(observed) > 1:
                raise ValueError(
                    f"{name} is declared fixed at {value!r} but the failure "
                    f"file contains {observed}. Either the declaration is "
                    "wrong or the study spans conditions it says it does not.")
            if observed and observed[0] != str(value):
                notes.append(
                    f"{name} is declared fixed at {value!r} but the file "
                    f"reports {observed[0]!r}")

        for role, names in (("stratified", self.stratified),
                            ("covariate", self.covariate)):
            for name in names:
                if name not in table:
                    raise ValueError(
                        f"{name} is declared as a {role} but the failure file "
                        "has no such column. A condition that cannot be read "
                        "cannot be controlled for.")
                if table[name].isna().any():
                    notes.append(
                        f"{name} is missing on "
                        f"{int(table[name].isna().sum())} failure(s); those "
                        "rows cannot be assigned to a stratum or given a "
                        "covariate value")
                if role == "covariate":
                    per_die = table.groupby(die_keys)[name].nunique()
                    varying = per_die[per_die > 1]
                    if len(varying):
                        notes.append(
                            f"{name} varies within {len(varying)} die(s); it "
                            "is being used as a die-level covariate, so the "
                            "within-die variation is discarded")
        return notes

    def report(self) -> dict:
        return {"by_condition": {c: self.status(c)
                                 for c in PACKAGE_PROCESS_CONDITIONS},
                "fixed_values": dict(self.fixed),
                "undeclared": self.undeclared()}


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
    fill_layers: dict[str, LayerSpec] = field(default_factory=dict)
    wide_width_um: float = 3.0
    layout_revision: str | None = None
    #: Sample conditions that change the crack driving force and cannot be
    #: recovered from GDS. Declared as fixed, stratified, or a baseline
    #: covariate -- see :class:`SampleConditions`.
    sample_conditions: "SampleConditions" = field(
        default_factory=lambda: SampleConditions())
    top_cell: str | None = None
    die_outline_um: list[float] | None = None
    footprint_spec: dict = field(default_factory=dict)
    footprint_per_die: dict = field(default_factory=dict)
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
        if not (layout.get("fill_layers") or {}):
            gaps.append(
                "no fill_layers: dummy fill cannot be separated from functional "
                "geometry, so it contributes to every density and sets the "
                "shortest edge on the layer")
        if not layout.get("layout_revision"):
            gaps.append(
                "no layout_revision: nothing checks that every die analysed "
                "was built from the layout in this file")
        undeclared = SampleConditions(
            fixed=dict((cfg.get("sample_conditions") or {}).get("fixed") or {}),
            stratified=tuple((cfg.get("sample_conditions") or {}).get("stratified") or ()),
            covariate=tuple((cfg.get("sample_conditions") or {}).get("covariate") or ()),
        ).undeclared()
        if undeclared:
            gaps.append(
                f"{len(undeclared)} package/process condition(s) undeclared "
                f"({', '.join(undeclared[:4])}"
                f"{'...' if len(undeclared) > 4 else ''}): none can be read "
                "from GDS, and if any varies across the study an apparent "
                "geometry effect may be standing in for it")
        if not reg.get("fiducials"):
            gaps.append(
                "no registration fiducials: positional uncertainty must come "
                "from the failure file, and no scale can be certified from "
                "measured registration")

        return cls(
            metal_layers=metals,
            via_layers={k: _spec(v) for k, v in
                        (layout.get("via_layers") or {}).items() if v},
            fill_layers={k: _spec(v) for k, v in
                         (layout.get("fill_layers") or {}).items() if v},
            wide_width_um=float(layout.get("wide_width_um", 3.0)),
            layout_revision=layout.get("layout_revision"),
            sample_conditions=SampleConditions(
                fixed=dict((cfg.get("sample_conditions") or {}).get("fixed") or {}),
                stratified=tuple((cfg.get("sample_conditions") or {}).get("stratified") or ()),
                covariate=tuple((cfg.get("sample_conditions") or {}).get("covariate") or ())),
            package_layers=package, line_rules=rules,
            top_cell=layout.get("top_cell"),
            die_outline_um=layout.get("die_outline_um"),
            footprint_spec=inspect.get("footprint") or {},
            footprint_per_die=inspect.get("footprint_per_die") or {},
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
                 + [(v, f"fill for {k}") for k, v in self.fill_layers.items()]
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

    def footprint_set(self, reader: LayoutReader, bbox: BBox) -> FootprintSet:
        """Default footprint plus any per-die overrides.

        A campaign rarely inspects every die the same way, and collapsing that
        to one footprint either discards the dies inspected more thoroughly or
        credits the ones inspected less with controls nobody earned.
        """
        per_die = {}
        for key, spec in (self.footprint_per_die or {}).items():
            per_die[str(key)] = self._one_footprint(reader, bbox, spec)
        return FootprintSet(default=self.footprint(reader, bbox),
                            per_die=per_die)

    def _one_footprint(self, reader, bbox, spec) -> InspectionFootprint:
        if spec.get("gds_layer"):
            return InspectionFootprint.from_gds_layer(reader, _spec(spec["gds_layer"]))
        if spec.get("rectangles"):
            return InspectionFootprint.from_rectangles(
                spec["rectangles"], dbu=reader.units.dbu,
                source="manifest_rectangles")
        if spec.get("full_die"):
            return InspectionFootprint.full_die(bbox, str(spec["full_die"]),
                                                dbu=reader.units.dbu)
        raise ValueError(f"unrecognised footprint specification: {spec}")

    def line_end_w_max_um(self) -> float | None:
        """Single fallback cutoff, for callers that cannot take per-layer rules.

        Prefer :meth:`line_rule_map`. Collapsing the stack to one number lets a
        wide line on a finer layer be read as a terminated tip -- an M7 rule of
        1um overridden by an M8 rule of 2um.
        """
        if not self.line_rules:
            return None
        return max(r.line_max_width_um for r in self.line_rules.values())

    def line_rule_map(self) -> dict[str, tuple[float, float]]:
        """{layer name: (min_width_um, line_max_width_um)} for the extractor."""
        return {k: (v.min_width_um, v.line_max_width_um)
                for k, v in self.line_rules.items()}

    def report(self) -> dict:
        return {
            "source": self.source,
            "metal_layers": [str(m) for m in self.metal_layers],
            "via_layers": {k: str(v) for k, v in self.via_layers.items()},
            "fill_layers": {k: str(v) for k, v in self.fill_layers.items()},
            "wide_width_um": self.wide_width_um,
            "layout_revision": self.layout_revision,
            "sample_conditions": self.sample_conditions.report(),
            "package_layers": {k: (str(v) if v else None)
                               for k, v in vars(self.package_layers).items()},
            "line_rules": {k: vars(v) for k, v in self.line_rules.items()},
            "die_outline_um": self.die_outline_um,
            "min_coverage": self.min_coverage,
            "footprint_per_die": sorted(self.footprint_per_die or {}),
            "scales_um": list(self.scales_um),
            "pair_selection": self.pair_selection,
            "enforce_scale_gate": self.enforce_scale_gate,
            "declared_gaps": self.gaps,
        }
