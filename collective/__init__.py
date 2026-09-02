"""lamXsim, consolidated.

The same code as the ``src/lamxsim`` package it was folded from, in ten files
grouped by what they are for rather than by one concept per module. The order
of the files is the order of the dependency graph: ``foundation`` depends on
nothing here, ``workflow`` depends on most of it.

    foundation   units, evidence classes, the feature registry
    layout       reading a GDS, and building synthetic ones
    geometry     the analysis grid and every window feature on it
    objects      per-object shape for bumps, pads, PI openings, crackstops
    labels       position, package context, failure files, footprints
    study        the manifest: what the layout's layers mean
    statistics   association, correction, resampling, power, validation
    calibre      rule-deck generation, its emulator, reading its output
    exposure     literature channels, the atlas, the reports
    workflow     registration, the pipeline, the cost model, the CLI

Twelve names were renamed to resolve collisions the merge created -- six
modules each had an ``EVIDENCE_CLASS``, four had a ``FEATURES``, and two pairs
each had an ``extract``, a ``markers`` and a ``write``. They are listed in
``docs/collective_layout.md`` beside the module each came from.
"""
