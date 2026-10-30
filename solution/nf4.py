"""Dequantization of the 4-bit weights for the backward pass, in one Triton launch.

bitsandbytes dequantizes a weight again for every backward through a 4-bit linear: over the 16
steps of our profile that was 14% of the GPU time, in two kernels (the block scales, then the
weights). Here both are one kernel. It only reads what bitsandbytes stored, so the forward pass,
the weights and their format stay as they are.
"""

import torch
import triton
import triton.language as tl
from bitsandbytes import functional as bnb_functional

BLOCK_VALUES = 64  # the quantization block of the task's NF4 weights
GROUP_BLOCKS = 256  # double quantization: blocks that share one scale
BLOCK_BYTES = BLOCK_VALUES // 2  # two 4-bit codes per byte


@triton.jit
def nf4_dequantize_kernel(packed_ptr, block_code_ptr, code_map_ptr, group_scale_ptr, table_ptr,
                          out_ptr, n_values, offset, BYTES: tl.constexpr):
    """One program per quantization block, so the block's scale is decoded once, as a scalar:
    code_map[block_code] * group_scale + offset. High nibble first."""
    block = tl.program_id(0)
    code = tl.load(block_code_ptr + block).to(tl.int32)
    scale = tl.load(code_map_ptr + code) * tl.load(group_scale_ptr + (block >> 8)) + offset

    byte_id = block * BYTES + tl.arange(0, BYTES)
    high_id = byte_id * 2
    low_id = high_id + 1
    has_high = high_id < n_values
    byte = tl.load(packed_ptr + byte_id, mask=has_high, other=0).to(tl.int32)
    high = tl.load(table_ptr + (byte >> 4))
    low = tl.load(table_ptr + (byte & 15))
    out_ty = out_ptr.dtype.element_ty
    tl.store(out_ptr + high_id, (high * scale).to(out_ty), mask=has_high)
    tl.store(out_ptr + low_id, (low * scale).to(out_ty), mask=low_id < n_values)


def _maps(state, device):
    """The two lookup tables, kept on the GPU with the state of the weight."""
    maps = getattr(state, "kernel_maps", None)
    if maps is None:
        maps = state.kernel_maps = (state.state2.code.to(device, torch.float32),
                                    bnb_functional.get_4bit_type("nf4", device=device).float())
    return maps


def _is_task_weight(packed, state):
    return (state is not None and packed.is_cuda and state.nested and state.quant_type == "nf4"
            and state.blocksize == BLOCK_VALUES and state.state2.blocksize == GROUP_BLOCKS
            and state.dtype in (torch.float16, torch.float32))


def nf4_dequantize(packed, state):
    """bitsandbytes.functional.dequantize_4bit of a double-quantized NF4 weight."""
    n_values = state.shape[0] * state.shape[1]
    code_map, table = _maps(state, packed.device)
    out = torch.empty(tuple(state.shape), dtype=state.dtype, device=packed.device)
    grid = (triton.cdiv(n_values, BLOCK_VALUES),)
    nf4_dequantize_kernel[grid](packed, state.absmax, code_map, state.state2.absmax, table, out,
                                n_values, float(state.offset), BYTES=BLOCK_BYTES, num_warps=1)
    return out


def patch_dequantize():
    """MatMul4Bit.backward looks the function up on the module at every call."""
    original = bnb_functional.dequantize_4bit

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None,
                        quant_type="fp4"):
        if absmax is None and out is None and _is_task_weight(A, quant_state):
            return nf4_dequantize(A, quant_state)
        return original(A, quant_state, absmax, out, blocksize, quant_type)

    bnb_functional.dequantize_4bit = dequantize_4bit
