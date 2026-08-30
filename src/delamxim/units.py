"""Physical-unit helpers.

Everything the engine reasons about is in micrometres. KLayout works in
database units (dbu); conversion happens only at the layout boundary so that
no downstream module ever sees a dbu.
"""
from __future__ import annotations


class Units:
    __slots__ = ("dbu",)

    def __init__(self, dbu: float):
        if dbu <= 0:
            raise ValueError(f"dbu must be positive, got {dbu}")
        self.dbu = float(dbu)

    def um_to_dbu(self, um: float) -> int:
        return int(round(um / self.dbu))

    def dbu_to_um(self, d: float) -> float:
        return d * self.dbu

    def area_dbu2_to_um2(self, a: float) -> float:
        return a * self.dbu * self.dbu

    def length_dbu_to_um(self, l: float) -> float:
        return l * self.dbu

    def __repr__(self) -> str:
        return f"Units(dbu={self.dbu})"
