"""Padding-free LoRA steps and fused Triton kernels for `soup train` on a T4.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. The patches are on classes and modules, so they
reach the model and the trainer Soup builds later, and each one leaves anything that is not this
task on its original path.
"""

from .nf4 import patch_dequantize
from .step import patch_rms_norm, patch_training_step
from .swiglu import patch_qwen2_mlp


def apply():
    patch_qwen2_mlp()  # the starter kit's example, kept running; it does not count as ours
    patch_rms_norm()
    patch_dequantize()
    patch_training_step()
