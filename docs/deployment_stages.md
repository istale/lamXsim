# Bringing lamXsim up on a real layout, one stage at a time

This is for the engineer who has been handed a GDS, a layer map and this
repository, and has to decide what to run and in what order.

Each stage ends in a **gate**: something you check before spending the next
stage's time. The gates exist because the two things most likely to waste a
day are a manifest that means something other than you thought, and a run that
cannot fit in memory — and both are cheap to find early and expensive to find
late.

Nothing here needs failure data. It needs a layout and a set of engineering
statements about what the layout's layers mean.

## What the runtime actually depends on

Two things scale differently, and mixing them up gives the wrong answer twice.

**Memory is linear in merged polygon count.** Measured on synthetic dies,
about **2.5 kB per merged polygon**, holding across a sixty-fourfold range.
The Python path holds every analysed layer merged at once and has no tiling,
so this is the whole of it: a hundred million polygons is roughly 250 GB.

**Time is not linear.** The windowed extractors clip the layer once per grid
row, then once per window against that row, so the work is rows × polygons —
and on a layout both grow with die area.

| die | windows | polygons | one layer, one scale | growth | peak RSS |
|---:|---:|---:|---:|---:|---:|
| 400 µm | 9 | 16,000 | 0.55 s | — | 0.05 GB |
| 800 µm | 49 | 64,000 | 2.67 s | 4.84× | 0.24 GB |
| 1600 µm | 225 | 256,000 | 14.03 s | 5.26× | 0.82 GB |
| 3200 µm | 961 | 1,024,000 | 82.30 s | 5.86× | 2.74 GB |

Each row is four times the polygons of the one above it. If time were linear
the growth column would read 4.00×. It reads 4.84, 5.26, 5.86 — a local
exponent climbing from 1.14 to 1.28, heading towards the 1.5 the rows ×
polygons structure implies. **Projecting linearly from a small clip
understates a full chip by more than an order of magnitude**, which is the
difference between an overnight job and a fortnight.

So the constraint depends on the machine, and the two cases give opposite
advice:

- **Ordinary hardware, tens of GB.** Memory binds first. A full chip does not
  run slowly; it does not start.
- **A large-memory machine, hundreds of GB to terabytes.** Memory stops
  binding — 3 TB is about a billion polygons — and time becomes the wall
  instead. It is not a wall a bigger machine moves: the growth is in the
  algorithm. Fewer scales and fewer analysed layers reduce it proportionally,
  which is the only lever the Python path offers.

In both cases the answer for a full chip is the Calibre deck, for different
reasons: there the moving window is a native primitive, and Python reads one
value per window instead of scanning the layer once per window.

**These constants are not your constants.** They come from Manhattan geometry
with no hierarchy on one machine. Polygon density, hierarchy depth, the
fraction of non-Manhattan geometry and the machine all move them. Stage 2
measures yours, and needs at least two clips of different size to fit the
exponent rather than assume it.

## Stage 0 — the contract, minutes

```bash
lamxsim deck chip.gds --manifest layers.yaml --outdir deck
```

This reads the layout's bounding box, top cell and digest and writes the rule
deck. It does not extract anything, so it is fast even on a full chip.

**Gate.** Every layer in the manifest is found. `die_outline_um` is declared
and is the real die, not the geometry bounding box. The printed eps margin per
metal layer is at least 2×. The gaps the manifest declares are ones you
recognise and accept.

Most of what goes wrong later is decided here. A layer number is not a
semantics: the manifest also has to say whether a PI polygon is the opening or
the film whose holes are the openings, how pads and bumps are matched, and
which corner angle a recommended pad shape is. Those are engineering
statements about the layout, not extra measurement data, which is why
requiring them keeps the run GDS-only.

## Stage 1 — a small clip, end to end, minutes

Cut a 500 µm square and run the whole thing on it:

```bash
python3 -c "
import klayout.db as db
ly = db.Layout(); ly.read('chip.gds')
u = 1 / ly.dbu
ly.clip(ly.top_cell(), db.Box(0, 0, int(500 * u), int(500 * u)))
ly.write('clip.gds')"
```

```bash
lamxsim characterize clip.gds --manifest clip_layers.yaml --outdir results/clip
```

Note `clip_layers.yaml`, not the chip's manifest. A clip is not the die, so its
`die_outline_um` has to be the clip; run it with the chip's outline and the
run stops, which is the intended behaviour rather than an obstacle. It also
means **the clip's candidates are not the die's inspection list** — they are
the extremes of the geometry you supplied, and the channels that need a
certified die frame will refuse to score rather than measure from a frame that
is not the die.

**Gate.** All the output files appear. `assumptions_and_limits.md` does not
contradict anything you know about the part. Every channel that reports
nothing gives a reason you agree with — "no pad layer" when there is no pad
layer, "the input takes too few distinct values" on a uniform clip.

This stage tests meaning, not performance.

## Stage 2 — measure your own constants, hours

Run on a full die, or on a clip that resembles the real thing rather than a
quiet corner of it:

```bash
lamxsim budget small_clip.gds large_clip.gds --manifest layers.yaml \
  --full-chip-polygons 100000000 --available-ram-gb 3000 --max-hours 24
```

**Give it at least two clips of different size.** With one it cannot fit how
time grows with polygon count, has to assume linear, and says so — but the
assumption is wrong in the direction that matters and gets worse the further
the projection reaches. Make them clips that resemble the real layout rather
than quiet corners of it, and prefer the largest pair you can afford to run:
the exponent itself drifts upward with size, so a fit from two small clips is
still optimistic. The command says how far beyond its measured range it is
extrapolating, and warns when that is more than tenfold the fitted span.

Count the full chip's polygons the same way it does — merged, over the layers
the manifest analyses.

**Gate.** Two of them. The projected peak is under half the RAM you actually
have — half, because a peak is a peak and nothing else on the machine gets to
disappear while you run. And the projected time is inside a budget you are
willing to spend, remembering that it is a lower bound.

## Stage 3 — the fork

Both projections fit: run the Python path on the full chip. Give the machine
to the job, and re-check the wall time against the projection as it runs — if
it is running long, the exponent was fitted too low and stopping early costs
less than finishing.

Either projection does not fit: go to Stage 4.

- If **memory** is what fails, no amount of patience helps: the path holds
  every analysed layer merged at once.
- If **time** is what fails, a bigger machine does not help either — the
  growth is in the algorithm, not in the hardware. The only levers the Python
  path offers are fewer scales and fewer analysed layers, and both narrow the
  study.

There is no third option in this repository today. Tiling the Python path
would flatten the growth back towards linear and is real work rather than a
flag.

## Stage 4 — validate the Calibre deck once, hours

This is the only part of the chain that has never been checked against the
tool it targets.

The deck's *text* is tested: every `DFM RDB` line contains a real path and no
unexpanded variable. Its *semantics* are checked against
`collective.calibre`'s emulator, which states in KLayout region algebra what each
rule means. That catches an error in the ingest path, the conversions or the
grid alignment. It cannot catch a difference between this repository's reading
of SVRF and Mentor's implementation of it — if `SIZE ... BY -eps` does
something other than `Region.sized`, both sides are wrong together and agree.
`PROJECTING` on the width guard and `DFM RDB ... COPY` on the marker layers
are the two constructs to check first.

```bash
lamxsim deck die.gds --manifest layers.yaml --outdir deck_real
# run deck_real/rules.svrf in Calibre
lamxsim deck die.gds --manifest layers.yaml --outdir deck_emu --emulate
```

**Gate.** Compare the two directories.

- `metal_density`, `via_density` and the corner marker counts must agree
  **exactly**. No approximation is involved in any of them, so any difference
  is a defect.
- Whole-layer perimeter must agree to 0.01 %. Per window it will differ by up
  to a few percent: the corner correction is exact for a layer and not for a
  window, because a corner just outside a window still owns band area inside
  it.
- The eps guard files must be empty. A non-empty one means the layout is
  narrower than the manifest claims and every perimeter number on that layer
  is understated — by an unknown amount, and silently, which is why the ingest
  side refuses to proceed.

Until this passes, describe results from the deck path as unvalidated.

## Stage 5 — the full chip

```bash
lamxsim characterize chip.gds --manifest layers.yaml --features-from deck_out
```

Three gates fire on their own and stop the run rather than degrading quietly:

- **Completeness.** Every layer and scale the manifest asks for must be
  present in the deck output. A missing map is not a smaller Calibre run, it
  is a Python map wearing a Calibre label.
- **The eps guard.** Required to exist and to be empty.
- **The binding.** The deck records the layout digest, top cell and bounding
  box, and the layer rules it was built against. A complete set of RDBs from
  another revision of the same design has the same layer names, the same
  scales and the same coordinates, and would otherwise be mixed with the
  orientation and package-context maps computed from the layout actually
  loaded — internally consistent, describing two different chips.

The output says which maps came from the deck and which were computed in
Python, in the CLI and in `assumptions_and_limits.md`. Orientation, gradients,
cross-layer terms, position and package context are always Python: the deck
does not produce them.

## What you have at the end, and what you do not

An exposure atlas: for every window, a deterministic geometry fact, and for
every candidate, a mechanistic engineering hypothesis with a citation and the
GDS observable it was measured from. It is a reason to look somewhere first.

It is not a failure probability, an energy release rate, a stress, or a design
rule. Nothing in it is calibrated, so candidates are ranked by percentile
**within this die** and by nothing else; the same layout on another package or
process would rank identically and mean something different. Read
`assumptions_and_limits.md` before anything else in the output directory, and
`unimplemented_gds_observables.csv` for what is not covered — including the
two things no layout can supply at all, which are the mechanically critical
bump and any sidewall or taper angle.

Turning any of it into a statistical association needs measured failure
locations in the same coordinate frame, with lot/wafer/die identity, a
registration fiducial set and an inspected footprint. `lamxsim phase0` says
how many failure sites that would take before the analysis could say anything.
