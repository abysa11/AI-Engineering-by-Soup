"""Our Triton kernels: the vocabulary head and RMSNorm, each fused into one launch.

Both are written for the shapes of the task, on a T4, and only use fp32 arithmetic inside:
`tl.exp`, `tl.log` and `tl.rsqrt` have no fp16 version.
"""

import torch
import triton
import triton.language as tl

VOCAB_BLOCK = 1024  # logits one row reads per loop iteration
NORM_BLOCK = 2048  # the smallest power of two that holds one hidden vector (1536)
WARPS = 8  # these vectors are wide: more threads per block, fewer registers per thread


@triton.jit
def vocab_terms_kernel(logits_ptr, label_ptr, loss_ptr, entropy_ptr, top_ptr, vocab, gain,
                       BLOCK: tl.constexpr):
    """One program per row of a [rows, vocab] logits tensor. Everything the head needs from one
    token, in three passes over its logits: the top class; the normalizer; then the entropy, the
    label's cross-entropy and, written over the logits, the gradient of that cross-entropy
    gain * (softmax - one_hot). A row with a negative label only gets softmax written. The logits
    may be fp16, as the head computes them; the gain keeps small probabilities out of the fp16
    rounding, as the loss scale does on the base path.

    The accumulators stay vectors, one sum per lane, until the end of a pass: fewer rounding steps
    than one running scalar, and no scalar has to be carried around the loop."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    logits_ptr += row * vocab
    label = tl.load(label_ptr + row)

    peak = tl.zeros([BLOCK], dtype=tl.float32) - float("inf")
    peak_id = tl.zeros([BLOCK], dtype=tl.int32)
    for start in range(0, vocab, BLOCK):
        ids = start + lane
        mask = ids < vocab
        x = tl.load(logits_ptr + ids, mask=mask, other=0.0).to(tl.float32)
        better = mask & (x > peak)
        peak_id = tl.where(better, ids, peak_id)
        peak = tl.where(better, x, peak)
    top_logit = tl.max(peak, axis=0)
    # The strict `>` keeps the first index of each lane, so this is torch.argmax on ties too.
    top_id = tl.min(tl.where(peak == top_logit, peak_id, vocab), axis=0)

    mass = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, vocab, BLOCK):
        ids = start + lane
        mask = ids < vocab
        x = tl.load(logits_ptr + ids, mask=mask, other=0.0).to(tl.float32)
        mass += tl.where(mask, tl.exp(x - top_logit), 0.0)
    log_total = tl.log(tl.sum(mass, axis=0))

    plogp = tl.zeros([BLOCK], dtype=tl.float32)
    label_log_p = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, vocab, BLOCK):
        ids = start + lane
        mask = ids < vocab
        x = tl.load(logits_ptr + ids, mask=mask, other=0.0).to(tl.float32)
        log_p = x - top_logit - log_total
        p = tl.where(mask, tl.exp(log_p), 0.0)
        is_label = mask & (ids == label)
        plogp += p * tl.where(mask, log_p, 0.0)
        label_log_p += tl.where(is_label, log_p, 0.0)
        grad = gain * (p - tl.where(is_label, 1.0, 0.0))
        tl.store(logits_ptr + ids, grad.to(logits_ptr.dtype.element_ty), mask=mask)

    tl.store(loss_ptr + row, -tl.sum(label_log_p, axis=0))
    tl.store(entropy_ptr + row, -tl.sum(plogp, axis=0))
    tl.store(top_ptr + row, top_id)


@triton.jit
def rms_norm_fwd_kernel(x_ptr, weight_ptr, out_ptr, width, eps, BLOCK: tl.constexpr):
    """w * x * rsqrt(mean(x^2) + eps), one program per token."""
    lane = tl.arange(0, BLOCK)
    mask = lane < width
    start = tl.program_id(0) * width
    x = tl.load(x_ptr + start + lane, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + lane, mask=mask, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / width + eps)
    tl.store(out_ptr + start + lane, (w * x * scale).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rms_norm_bwd_kernel(grad_ptr, x_ptr, weight_ptr, dx_ptr, width, eps, BLOCK: tl.constexpr):
    """J^T grad of the kernel above, with r = rsqrt(mean(x^2) + eps) and u = grad * w:
    dx_j = r * (u_j - x_j * r^2 * sum(u * x) / width)."""
    lane = tl.arange(0, BLOCK)
    mask = lane < width
    start = tl.program_id(0) * width
    grad = tl.load(grad_ptr + start + lane, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + start + lane, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + lane, mask=mask, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / width + eps)
    u = grad * w
    dx = r * (u - x * r * r * tl.sum(u * x, axis=0) / width)
    tl.store(dx_ptr + start + lane, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def vocab_terms(logits, labels, gain=1.0):
    """`logits` [rows, vocab] is overwritten with gain * (softmax - one_hot); returns the
    cross-entropy (0 where the label is negative), the entropy and the argmax of each row.
    `labels` is int32."""
    rows, vocab = logits.shape
    loss = torch.empty(rows, device=logits.device, dtype=torch.float32)
    entropy = torch.empty(rows, device=logits.device, dtype=torch.float32)
    top = torch.empty(rows, device=logits.device, dtype=torch.int32)
    vocab_terms_kernel[(rows,)](logits, labels, loss, entropy, top, vocab, gain,
                                BLOCK=VOCAB_BLOCK, num_warps=WARPS)
    return loss, entropy, top


class RmsNorm(torch.autograd.Function):
    """Qwen2RMSNorm in one launch each way instead of 8 ops. The backward pass only needs `x`."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        out = torch.empty_like(x, dtype=torch.promote_types(x.dtype, weight.dtype))
        width = x.shape[-1]
        rms_norm_fwd_kernel[(x.numel() // width,)](x, weight, out, width, eps, BLOCK=NORM_BLOCK,
                                                   num_warps=WARPS)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty_like(x)
        width = x.shape[-1]
        rms_norm_bwd_kernel[(x.numel() // width,)](grad, x, weight, dx, width, ctx.eps,
                                                   BLOCK=NORM_BLOCK, num_warps=WARPS)
        return dx, None, None


def rms_norm(x, weight, eps):
    return RmsNorm.apply(x, weight, eps)
