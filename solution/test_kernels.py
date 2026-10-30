"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-4), (torch.float16, 2e-3)])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (1, 37, 1536), (2, 1, 48)])
def test_rms_norm_matches_pytorch(dtype, tol, shape):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    from solution.kernels import rms_norm

    torch.manual_seed(0)
    norm = Qwen2RMSNorm(shape[-1], eps=1e-6).cuda().float().requires_grad_(False)
    norm.weight.data.uniform_(0.5, 1.5)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    x.data[0, 0] = 0  # a zero vector: only eps keeps the root finite
    seed = torch.randn(shape, device="cuda")

    reference = norm(x)
    reference.backward(seed)
    dx_reference, x.grad = x.grad.clone(), None

    out = rms_norm(x, norm.weight, norm.variance_epsilon)
    out.backward(seed)
    assert out.dtype == reference.dtype
    torch.testing.assert_close(out, reference, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad, dx_reference, rtol=max(tol, 1e-3), atol=max(tol, 1e-3))


def test_rms_norm_repeated_launches_are_stable():
    from solution.kernels import rms_norm

    torch.manual_seed(0)
    x, weight = torch.randn(4, 64, 1536, device="cuda"), torch.ones(1536, device="cuda")
    first = rms_norm(x, weight, 1e-6)
    for _ in range(20):
        torch.testing.assert_close(rms_norm(x, weight, 1e-6), first, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-3), (torch.float16, 5e-3)])
@pytest.mark.parametrize("rows,vocab", [(64, 151936), (3, 151936), (1, 1024), (5, 1500)])
def test_vocab_terms_match_pytorch(rows, vocab, dtype, tol):
    from solution.kernels import vocab_terms

    torch.manual_seed(0)
    logits = (torch.randn(rows, vocab, device="cuda") * 6).to(dtype)
    logits[0, 5] = 80.0  # one sure class: most exponentials underflow
    labels = torch.randint(0, vocab, (rows,), device="cuda", dtype=torch.int32)
    labels[-1] = -1  # a token outside the answer

    wide = logits.float()
    log_p = wide.log_softmax(-1)
    probe = wide.clone().requires_grad_(True)
    picked = (labels >= 0).nonzero()[:, 0]
    torch.nn.functional.cross_entropy(probe[picked], labels[picked].long(), reduction="sum").backward()
    entropy = -(log_p.exp() * log_p).sum(-1)
    token_loss = -log_p.gather(1, labels.clamp(min=0).long()[:, None])[:, 0] * (labels >= 0)

    gain = 2048.0
    written = logits.clone()
    loss, ent, top = vocab_terms(written, labels, gain)
    torch.testing.assert_close(loss, token_loss, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ent, entropy, rtol=1e-3, atol=1e-3)
    assert torch.equal(top.long(), wide.argmax(-1))
    torch.testing.assert_close(written[picked].float() / gain, probe.grad[picked], rtol=tol,
                               atol=1e-4)
    assert torch.isfinite(written).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(8960, 1536), (1536, 8960), (256, 1536), (7, 13), (1, 65)])
def test_nf4_matches_bitsandbytes(shape, dtype):
    from bitsandbytes import functional

    from solution.nf4 import nf4_dequantize

    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda") * torch.logspace(-3, 0, shape[1], device="cuda")
    packed, state = functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
    state.dtype = dtype
    expected = functional.dequantize_4bit(packed, state)
    torch.testing.assert_close(nf4_dequantize(packed.reshape(-1, 1), state), expected, rtol=0,
                               atol=0)


def test_head_terms_match_autograd():
    from solution.head import head_terms

    torch.manual_seed(0)
    vocab, width = 4096, 256
    weight = (torch.randn(vocab, width, device="cuda") * 0.02).half()
    hidden = torch.randn(4, 200, width, device="cuda", requires_grad=True)
    labels = torch.randint(0, vocab, (4, 200), device="cuda")
    mask = torch.ones_like(labels)
    labels[:, :70] = -100  # the prompt
    for row, length in enumerate((200, 150, 90, 71)):  # right padding
        labels[row, length:] = -100
        mask[row, length:] = 0
    denominator = (labels[:, 1:] != -100).sum()

    target = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
    logits = hidden @ weight.float().t()
    reference = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1),
                                                  ignore_index=-100)
    reference.backward()
    grad_reference, hidden.grad = hidden.grad.clone(), None

    terms = head_terms(hidden, weight, labels, mask, denominator)
    terms.loss.backward()
    tol = 2e-3 * grad_reference.abs().max()
    torch.testing.assert_close(terms.loss, reference, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(hidden.grad, grad_reference, rtol=1e-2, atol=tol.item())

    log_p = logits.detach().log_softmax(-1)
    entropy = (-(log_p.exp() * log_p).sum(-1) * mask).sum()
    assert terms.entropy.dtype == torch.float32
    torch.testing.assert_close(terms.entropy, entropy, rtol=2e-3, atol=2e-3)


def test_dropout_plan_draws_the_masks_of_the_padded_batch():
    """The base step calls dropout on real activations; the plan calls it on a tensor of ones. On
    CUDA that is PyTorch's fused dropout, so this checks that the values of the input do not
    change which elements are dropped."""
    from solution.step import DropoutPlan

    layers = [torch.nn.Dropout(0.05) for _ in range(4)]
    shape = (4, 300, 1536)
    activations = [torch.randn(shape, device="cuda") for _ in layers]
    torch.manual_seed(5)
    expected = [layer(x) != 0 for layer, x in zip(layers, activations)]

    plan = DropoutPlan(layers)
    torch.manual_seed(5)
    rows = torch.tensor([2, 0], device="cuda")
    plan.draw(shape, torch.float32, "cuda", [(rows, 120)])
    plan.next_group()
    for index, (layer, x) in enumerate(zip(layers, activations)):
        kept = expected[index][[2, 0], :120]
        differ = (plan.masks[index] != kept).float().mean().item()
        assert differ == 0, f"layer {index}: {differ:.6f} of the mask differs"
        replayed = layer(x[[2, 0], :120])
        torch.testing.assert_close(replayed, x[[2, 0], :120] * kept / 0.95, rtol=1e-6, atol=1e-6)
    plan.clear()
    assert layers[0](x).shape == shape  # outside a step the original dropout is back
