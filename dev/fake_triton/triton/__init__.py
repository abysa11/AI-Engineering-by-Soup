"""A small CPU emulator of the part of Triton that solution/ uses. NOT Triton.

Triton 3.6.0 has no macOS wheels, so this runs a kernel's Python body program by program on CPU
tensors, to test its indexing, masks and arithmetic where no CUDA GPU exists. It proves nothing
about compilation, GPU exp/log accuracy or speed: only `solution/test_kernels.py` on a T4 does.
"""

import math

from . import language

__version__ = "0.0-emulator"


class _Launch:
    def __init__(self, fn, grid):
        self.fn, self.grid = fn, grid

    def __call__(self, *args, num_warps=None, **constants):
        pointers = [language.Pointer(a) if hasattr(a, "numel") else a for a in args]
        for program in range(self.grid[0]):
            language._program_id = program
            self.fn(*pointers, **constants)


class _Jit:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return _Launch(self.fn, grid)


def jit(fn):
    return _Jit(fn)


def cdiv(a, b):
    return math.ceil(a / b)
