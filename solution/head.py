"""The output head and the loss without a [batch, seq, vocab] logits tensor.

On the base path that tensor exists several times over at the backward peak: the fp16 logits, their
fp32 copy for the loss, log_softmax and its gradient; each fp32 one is 1.16 GiB for a padded
4 x 512 batch. Here `chunk` tokens at a time go through the frozen head, one kernel launch reduces
them to the numbers the trainer needs, and the gradient on the hidden states is taken right there,
so no logits outlive their chunk.

The head works in fp16 like the base head under autocast, because the same matmul in fp32 took a
third of the GPU time. See `compact_head` in step.py for the weight.
"""

from dataclasses import dataclass

import torch

from .kernels import vocab_terms

CHUNK = 64  # tokens at a time: 19 MiB of fp16 logits for this vocabulary
GAIN = 2048.0  # lifts softmax out of the fp16 rounding; divided out of the gradient
IGNORE_INDEX = -100


@dataclass
class HeadTerms:
    loss: torch.Tensor  # sum of the cross-entropy over the labelled tokens / denominator
    row_ce: torch.Tensor  # sum of the cross-entropy of each row
    entropy: torch.Tensor  # sum of the entropy over the attended tokens
    correct: torch.Tensor  # labelled tokens whose argmax is the label


class HeadLoss(torch.autograd.Function):
    """Connects the gradient d(loss) / d(hidden) taken in `head_terms` to the decoder's graph."""

    @staticmethod
    def forward(ctx, hidden, loss, grad):
        ctx.save_for_backward(grad)
        return loss.clone()

    @staticmethod
    def backward(ctx, scale):
        (grad,) = ctx.saved_tensors
        return grad * scale, None, None


def head_terms(hidden, weight, labels, attention_mask, denominator):
    """`hidden` [rows, seq, hidden] of attended tokens and right padding, `weight` the frozen
    [vocab, hidden] head, `labels` as the collator gives them. Position t is scored against
    labels[t + 1], as in ForCausalLMLoss."""
    with torch.no_grad():
        target = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:]
        labelled = target != IGNORE_INDEX
        attended = attention_mask.bool()
        unlabelled = attended & ~labelled
        tokens = torch.cat([hidden[labelled], hidden[unlabelled]]).to(weight.dtype)
        token_labels = target[labelled].to(torch.int32)
        n_labelled = token_labels.numel()
        token_labels = torch.cat([token_labels, token_labels.new_full((len(tokens) - n_labelled,), -1)])

        grads, losses = [], []
        entropy = torch.zeros((), device=tokens.device)
        correct = torch.zeros((), dtype=torch.long, device=tokens.device)
        for start in range(0, len(tokens), CHUNK):
            stop = start + CHUNK
            piece_labels = token_labels[start:stop]
            logits = torch.mm(tokens[start:stop], weight.t())
            token_loss, token_entropy, top = vocab_terms(logits, piece_labels, GAIN)
            entropy += token_entropy.sum()
            if start < n_labelled:
                graded = min(len(piece_labels), n_labelled - start)
                grads.append(torch.mm(logits[:graded], weight).float())
                losses.append(token_loss[:graded])
                correct += (top[:graded] == piece_labels[:graded]).sum()
            del logits

        grad = torch.zeros_like(hidden, dtype=torch.float32)
        row_ce = torch.zeros(len(hidden), device=hidden.device)
        if n_labelled:
            grad[labelled] = torch.cat(grads) / (GAIN * denominator)
            per_token = torch.cat(losses)
            row_ce.index_add_(0, labelled.nonzero()[:, 0], per_token)
        loss = row_ce.sum() / denominator
    return HeadTerms(HeadLoss.apply(hidden, loss, grad), row_ce, entropy, correct)
