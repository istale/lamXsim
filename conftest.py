"""Put the repository root on the path.

The tests live beside the code they exercise, in ``collective/``, and import
it as ``collective.geometry`` rather than by relative import -- so a test file
reads the same whether it is run by pytest, by a debugger, or by hand.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
