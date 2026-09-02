"""Validation of the statistical pipeline itself, on ground truth we control.

These are the tests that decide whether any association the platform reports
can be believed. They run on a die where metal density and perimeter density
vary independently, so a pipeline that has collapsed into a density detector
fails them.
"""
import numpy as np
import pytest

from collective import workflow as pipeline
from collective.geometry import GeometryExtractor
from collective.geometry import build_grid
from collective.labels import failures_from_driver, uniform_failures
from collective.layout import LayerSpec, LayoutReader
from collective.layout import validation_die

M8 = LayerSpec("M8", 8, 0)
SCALES = (50, 100, 250)


@pytest.fixture(scope="module")
def die(tmp_path_factory):
    path = tmp_path_factory.mktemp("die") / "valdie.gds"
    validation_die(str(path), die_um=2000.0, block_um=50.0, seed=7)
    reader = LayoutReader(str(path))
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader).extract(M8, grid)
    return str(path), grid, feats


def test_die_decouples_density_from_perimeter(die):
    """The validation die is only meaningful if the two features are separable."""
    _, _, f = die
    r = np.corrcoef(f["metal_density"], f["perimeter_density"])[0, 1]
    assert abs(r) < 0.6, f"features too collinear to test discrimination (r={r:.2f})"


def test_recovers_the_true_driver(die):
    """Failures driven by perimeter must rank perimeter above metal density."""
    path, grid, f = die
    fs = failures_from_driver(f["perimeter_density"], grid, n_failures=150,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=SCALES,
                       n_permutations=0, seed=3)
    a = res.associations
    perim = a[a.feature == "perimeter_density"].roc_auc.max()
    metal = a[a.feature == "metal_density"].roc_auc.max()
    assert perim > metal + 0.10, (
        f"perimeter AUC {perim:.3f} did not clearly beat metal {metal:.3f}; "
        "the pipeline may be responding to density rather than the true driver"
    )


def test_negative_control_reports_nothing(die):
    """Spatially uniform failures must produce no significant association."""
    path, grid, _ = die
    fs = uniform_failures(grid, n_failures=150, seed=42, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=SCALES,
                       n_permutations=0, seed=5)
    q = res.associations.fdr_q_value.dropna()
    assert (q < 0.05).sum() == 0, (
        f"{(q < 0.05).sum()} spurious findings under the null: the pipeline "
        "manufactures associations and no result from it can be trusted"
    )


def test_block_permutation_rejects_spurious_position_association(die):
    """Position features are confounded by autocorrelation, not by an effect.

    The validation die has no package-position effect built in, yet a test that
    treats grid cells as independent flags position features as significant.
    The spatial null model is what removes them.
    """
    path, grid, f = die
    fs = failures_from_driver(f["perimeter_density"], grid, n_failures=150,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(50, 100),
                       n_permutations=299, seed=3)

    a = res.associations.set_index(["feature", "scale_um"])
    p = res.permutations.set_index(["feature", "scale_um"])
    pos = a[a.evidence_class == "PACKAGE_POSITION"].index
    geo = a[a.evidence_class == "GDS_GEOMETRY"].index

    naive_pos = (a.loc[pos, "p_value"] < 0.05).sum()
    perm_pos = (p.loc[pos, "p_value"] < 0.05).sum()

    assert naive_pos > 0, "test is vacuous unless the naive test is fooled"
    assert perm_pos == 0, (
        f"block permutation still calls {perm_pos} position associations "
        "significant on a die with no position effect"
    )
    # The driver itself must survive: a null model that also erases real
    # signal would be useless, not conservative. Only perimeter_density drives
    # the simulated failures here, so only it is required to survive.
    driver = p.loc[[i for i in geo if i[0] == "perimeter_density"], "p_value"]
    assert (driver < 0.05).all(), (
        "block permutation discarded the real perimeter-driven signal"
    )


def test_effective_n_is_far_below_cell_count(die):
    """Grid cells must not be reported as if they were independent samples."""
    path, grid, f = die
    fs = failures_from_driver(f["perimeter_density"], grid, n_failures=150,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(50,), n_permutations=0, seed=3)
    row = res.associations[res.associations.feature == "perimeter_density"].iloc[0]
    assert row.effective_n < row.n_cells * 0.5


def test_registration_gates_small_scales(die):
    """Scales below ~3x the positional uncertainty are marked untrustworthy."""
    path, grid, f = die
    fs = failures_from_driver(f["perimeter_density"], grid, n_failures=80,
                              strength=2.0, seed=2, position_sigma_um=40.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(25, 50, 100, 250),
                       n_permutations=0, seed=3)
    a = res.associations
    assert set(a[a.scale_um <= 100].scale_trustworthy) == {False}
    assert set(a[a.scale_um == 250].scale_trustworthy) == {True}
