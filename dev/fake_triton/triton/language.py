"""The `triton.language` calls of solution/, on torch tensors. See the package docstring."""

import torch

float16, float32, int32 = torch.float16, torch.float32, torch.int32
_program_id = 0


class constexpr:
    pass


class dtype:  # torch looks this name up when it finds a `triton` package
    pass


class _Dtype:
    def __init__(self, element_ty):
        self.element_ty = element_ty


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor.detach().reshape(-1), offset
        self.dtype = _Dtype(tensor.dtype)

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)

    __iadd__ = __add__


def _fp32_only(x):
    assert x.dtype == torch.float32, "tl.exp, tl.log and tl.rsqrt are fp32 only (no fp16)"
    return x


def _read_bounds(pointer, index):
    assert index.min() >= 0 and index.max() < pointer.tensor.numel(), "out of bounds access"


def program_id(axis):
    assert axis == 0
    return _program_id


def arange(start, end):
    size = end - start
    assert size & (size - 1) == 0, "tl.arange needs a power of 2"
    return torch.arange(start, end, dtype=torch.int32)


def zeros(shape, dtype):
    return torch.zeros(shape, dtype=dtype)


def load(pointer, mask=None, other=None):
    index = torch.as_tensor(pointer.offset)
    if mask is None:
        _read_bounds(pointer, index)
        return pointer.tensor[index]
    assert other is not None
    if mask.any():
        _read_bounds(pointer, index[mask])
    out = torch.full(index.shape, float(other), dtype=pointer.tensor.dtype)
    out[mask] = pointer.tensor[index[mask]]
    return out


def store(pointer, value, mask=None):
    index = torch.as_tensor(pointer.offset)
    value = torch.as_tensor(value, dtype=pointer.tensor.dtype)
    if mask is None:
        _read_bounds(pointer, index)
        pointer.tensor[index] = value
        return
    if mask.any():
        _read_bounds(pointer, index[mask])
    pointer.tensor[index[mask]] = value[mask]


def where(condition, x, y):
    return torch.where(condition, x, y)


def sum(x, axis):
    return x.sum(dim=axis)


def max(x, axis):
    return x.max(dim=axis).values


def min(x, axis):
    return x.min(dim=axis).values


def exp(x):
    return torch.exp(_fp32_only(x))


def log(x):
    return torch.log(_fp32_only(x))


def rsqrt(x):
    return torch.rsqrt(_fp32_only(x))


def sigmoid(x):
    return torch.sigmoid(x)
