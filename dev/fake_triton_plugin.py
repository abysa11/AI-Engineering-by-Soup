"""`pytest -p fake_triton_plugin` for machines without Triton: loads torch first, so it never finds
the emulator, then puts the emulator on the path for `solution`."""

import sys
from pathlib import Path

import transformers.generation.utils  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent / "fake_triton"))
