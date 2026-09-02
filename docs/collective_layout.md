# Where everything went

This branch folds the forty modules of `src/lamxsim` into ten files under
`collective/`. The code is the same code: the whole test suite passes
unchanged in substance, which is the only reason to believe the fold was
faithful.

The files are ordered by the dependency graph. `foundation` depends on
nothing else here; `workflow` depends on most of it. Grouping by concern
rather than by layer would have made the graph cyclic -- and did, twice,
before it was measured: `evidence` is a leaf that everything imports and
`study` sits at the top of the graph, so putting both in one file made a
cycle that Python refuses at import time.

## The ten files

| file | lines | folded from |
|---|---:|---|
| `foundation.py` | 185 | `units.py`, `evidence.py`, `registry.py` |
| `study.py` | 507 | `study.py` |
| `layout.py` | 746 | `layout/reader.py`, `layout/synth.py` |
| `geometry.py` | 1,535 | `features/grid.py`, `features/corners.py`, `features/lineends.py`, `features/geometry.py`, `features/orientation.py`, `features/structures.py`, `features/vias.py`, `features/gradient.py`, `features/crosslayer.py`, `features/bump_relative.py` |
| `objects.py` | 984 | `features/objects.py` |
| `labels.py` | 1,253 | `labels/position.py`, `labels/package_context.py`, `labels/failure.py`, `labels/inspection.py`, `labels/simulate.py` |
| `statistics.py` | 1,109 | `stats/fdr.py`, `stats/univariate.py`, `stats/permutation.py`, `stats/power.py`, `stats/cv.py`, `stats/baseline.py`, `stats/ablation.py` |
| `calibre.py` | 1,287 | `calibre/svrf.py`, `calibre/ingest.py`, `calibre/emulate.py` |
| `exposure.py` | 1,664 | `exposure.py`, `atlas.py`, `report.py` |
| `workflow.py` | 2,182 | `registration/transform.py`, `registration/fit.py`, `registration/apply.py`, `pipeline.py`, `budget.py`, `cli.py` |

## The twelve renames

Merging modules merges their namespaces, and twelve names collided. None
was dropped: every one is referenced somewhere, so each was renamed for
what it is rather than for where it used to live.

| was | in | is now |
|---|---|---|
| `FEATURES` | `features/bump_relative.py` | `BUMP_RELATIVE_FEATURES` |
| `extract` | `features/bump_relative.py` | `bump_relative_extract` |
| `markers` | `features/corners.py` | `corner_markers` |
| `extract` | `features/crosslayer.py` | `crosslayer_extract` |
| `markers` | `features/lineends.py` | `line_end_markers` |
| `FEATURES` | `features/orientation.py` | `ORIENTATION_FEATURES` |
| `FEATURES` | `features/structures.py` | `STRUCTURE_FEATURES` |
| `FEATURES` | `features/vias.py` | `VIA_FEATURES` |
| `FEATURES` | `labels/package_context.py` | `PACKAGE_CONTEXT_FEATURES` |
| `extract` | `labels/package_context.py` | `package_context_extract` |
| `extract` | `labels/position.py` | `position_extract` |
| `write` | `report.py` | `write_reports` |

`EVIDENCE_CLASS` collided six ways in `geometry.py` and twice in
`labels.py`, but every copy held the same value, so one survives in each
file and the rest were dropped rather than renamed into identical
constants.

## Two source changes the fold required

Both are deferred imports, and both were already the pattern elsewhere in
the codebase -- the module graph has cycles that only work because some
imports sit inside functions. Hoisting them to the top of a merged file
turns a working deferred import into a real cycle, so two that had been at
module level moved into the functions that use them:

- `calibre/emulate.py` took `_file_digest` from `pipeline` at module level.
  `calibre/ingest.py` and `calibre/svrf.py` already deferred the same
  import; now `emulate` does too.
- `atlas.py` took `_covers`, `_fmt` and `_is_roi` from `pipeline` at module
  level, while `pipeline` imports `atlas` inside a function. The three are
  only used inside `build()`, so the import moved there.

## What was not done

The tests were not folded. They are 24 files organised by what they check,
which is already the right unit for them, and merging them would make a
failure harder to place rather than easier.

Nothing was rewritten for style, and no behaviour changed. A diff of this
branch against `main` should contain only moves, the twelve renames, the
two deferred imports above, and the packaging and import lines that follow
from the new layout.
