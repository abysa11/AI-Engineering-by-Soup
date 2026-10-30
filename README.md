# Padding-free Soup: Soup Kernel Challenge, Decentrathon 6.0

`soup train` on a Kaggle T4, same task and same adapter, without padding and without a full logits
tensor, with our own fused Triton kernels for the vocabulary head and RMSNorm.

> **Status: PENDING KAGGLE VALIDATION.** Everything below that says "measured" will come from the
> notebook. No result in this file has been measured on a T4 yet, and no GPU has run these kernels.
> See [Results](#results) and [FINAL_AUDIT.md](FINAL_AUDIT.md).

- Kaggle notebook: `PENDING KAGGLE VALIDATION` (public link, saved version with `COMMIT` set)
- Final commit: `PENDING` (link to the commit in the team repository)

## The problem

Soup fine-tunes Qwen2.5-1.5B-Instruct (NF4, LoRA r=16 on `q_proj` and `v_proj`) on 256 Kazakh
instructions: batch 4, up to 512 tokens, 64 steps, fp16, on a T4. The task is fixed. The job is to
make that same run faster or lighter.

## Why these operations

We looked at where the memory and time actually go before writing a kernel.

Tokenizing the public rows in the Qwen chat format (our reconstruction of what Soup feeds the
model, so these counts are close, not exact) gives a mean of 207 tokens, a median of 140, and 16
of 256 rows at the 512-token cap. The collator pads every row of a batch to the longest
one, so a batch costs 1,447 tokens on average for 826 real ones, and one row of 512 tokens costs a
whole 4 x 512 batch. That has two consequences:

1. **Peak memory follows the longest batch, time follows the sum of all batches.** Almost half the
   computation is padding, and the peak is set by padding too.
2. **Activations are the biggest item.** The 4-bit linears return fp32 when they get fp32, so each
   of the 28 layers keeps several 8960-wide fp32 activations for the backward pass.

The next item is the head: the fp16 logits, their fp32 copy, log_softmax and its gradient, each
`[4, 512, 151936]`, all alive at the backward peak next to the activations.

| What | Change | Helps |
|---|---|---|
| Padding (`solution/step.py`) | rows of a batch run in padding-free groups | time and memory |
| Head and loss (`solution/head.py`) | 64 tokens at a time, one fused kernel, no full logits | memory, time |
| MLP activations (`solution/swiglu.py`) | fp16 instead of fp32, fused SwiGLU (the kit's example) | memory |
| RMSNorm (`solution/kernels.py`) | one kernel each way instead of about 8 ops | time, memory |

## How it works

`solution.apply()` makes three class patches, each of which leaves anything that is not this task
on its original path: `Qwen2MLP.forward`, `Qwen2RMSNorm.forward` and `trl.SFTTrainer.training_step`
(Soup builds its trainer class from `SFTTrainer` during setup, so the patch reaches it).

**The step (`solution/step.py`).** `split_rows` sorts the rows of the batch by length and picks
the cheapest groups with a small dynamic program: a group costs a fixed overhead plus its padded
area, and may not exceed 512 positions unless it is one row. On the public rows that is 2.6 groups
per batch, 871 positions per batch instead of 1,447, and a largest group of 512 instead of 2,048.
Each group runs forward and backward, and the gradients add up before the one optimizer step of
the batch. The rows stay in their batch, nothing is mixed across batches, and Soup's own `packing`
and `padding_free` switches stay off. This is the removal of padding inside a batch that the rules
allow.

**Same dropout.** Splitting a batch would normally change the LoRA dropout masks, and with them the
adapter. `DropoutPlan` draws the masks first, with the same PyTorch call, shape, dtype and order as
the padded step, then gives each row its own mask. The random stream is the same one, so the run
stays comparable with the base run step by step, not only on average.

**The head (`solution/head.py`).** Hidden states go through the frozen fp32 head 64 tokens at a
time. `vocab_terms_kernel` then reduces each chunk in one launch, and `torch.mm` turns what it wrote
into the gradient on the hidden states. No logits outlive their chunk. A small `autograd.Function`
connects that gradient to the decoder's graph, so the GradScaler works as usual.

**Metrics.** TRL's `entropy`, `mean_token_accuracy` and `num_tokens` and Soup's `rewind.jsonl` get
the values they would compute from the logits. The step writes them itself, because `compute_loss`
does not run on this path.

## Our kernels

All in `solution/kernels.py`, all launched in every step of training, all outputs used by training.

| Kernel | One launch does | Replaces |
|---|---|---|
| `vocab_terms_kernel` | per token: max and argmax; normalizer; entropy, label cross-entropy and `softmax - one_hot` written over the logits | log_softmax, its gradient, TRL's entropy and argmax, rewind's per-row CE |
| `rms_norm_fwd_kernel` | `w * x * rsqrt(mean(x^2) + eps)` | 8 ops |
| `rms_norm_bwd_kernel` | `r * (u - x * r^2 * sum(u * x) / n)`, with `u = g * w` | the same ops' backward |

The kit's `example_swiglu_*` kernels keep running beside them. They do not count as ours.

Details that matter on a T4: `tl.exp`, `tl.log` and `tl.rsqrt` have no fp16 version, so kernels work
in fp32. `tl.arange` needs a power of two, so 1536 and 151,936 are handled with masks. Accumulators
stay one per lane until the end of a pass, which rounds less than a running scalar. The one-hot is
subtracted inside the vector store, so no position is ever written twice.

## How to run it

On Kaggle (the official way; `--repeats 2 --profile` runs five trainings):

1. **Create → New Notebook → File → Import Notebook**, pick `kaggle_notebook.ipynb` from this
   repository. It is the kit's template with one extra cell for the environment and one for a
   summary, and `run_task.py` runs base, solution, solution, base.
2. Settings: **Accelerator GPU T4 x2**, **Internet on**.
3. Set `REPO_URL` and `COMMIT` in the first code cell and press **Run All**.

On a Linux machine with a CUDA GPU:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

Without a GPU (for example on a Mac, where Triton has no wheels), `dev/` runs the kernels' Python on
CPU through a small emulator, against real transformers, peft, trl and Soup's rewind trainer. It
compares the patched step with the unpatched one from the same seed:

```bash
pip install torch transformers==5.16.1 peft==0.20.0 trl==0.29.1 accelerate==1.14.0 datasets pytest
python -m pytest dev -q
```

`dev/` is a development aid, not Triton: it says nothing about compilation or speed, and it is not
part of the submission's measured path.

## Results

`PENDING KAGGLE VALIDATION`. These cells get filled from `runs/result.json` and
`runs/checks.json` of the public notebook.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Whole `soup train`, s | PENDING | PENDING | PENDING |
| Peak GPU memory, MiB | PENDING | PENDING | PENDING |
| Eval loss (public rows) | PENDING | PENDING | PENDING |
| Own kernels' share of training time | - | PENDING | at least 5% needed |
| Checks of `harness/check_rules.py` | - | PENDING | 10 of 10 needed |

Verified so far, without a GPU, in the emulator with real transformers, peft and trl, on a tiny
random Qwen2 on CPU with autocast off: identical loss and token counts, per-parameter gradients
within 1.2e-6 relative of the base step with dropout on, and the same LoRA weights after training.
Soup's 79 regression tests pass with `apply()`. None of this is a T4 result.

## Numerical correctness and limits

- **The head runs in fp32, the base head in fp16.** Ours is more exact, so it is not bit-identical:
  loss, entropy and gradients differ from the base by that rounding. An fp16-like effect (bf16
  autocast on CPU) changed a gradient norm by about 0.15% in our local test.
- **`tl.exp` and `tl.log` are fast approximations** on GPUs. `solution/test_kernels.py` bounds their
  effect against PyTorch at 1e-3 relative on a T4. That test has not run on a T4 yet.
- **Dropout replay relies on PyTorch's fused CUDA dropout** not depending on its input's values.
  `test_dropout_plan_draws_the_masks_of_the_padded_batch` checks this on the GPU. If it fails, the
  run is still valid training, only no longer mask-identical to the base.
- **The masks cost memory**: up to about 170 MiB for a 4 x 512 batch, as one byte per element.
- **The peak is still one 512-token row.** Going lower needs recomputation for long rows.
- **Model loading has its own peak**, which no training-time change can lower.
- `GROUP_COST` is an estimate of one extra forward and backward, to be checked against the profile.
- The step only runs for the task's shape (Qwen2, single GPU, no accumulation, plain loss). Anything
  else takes the trainer's original path.

## Known issues

- No kernel in this repository has been compiled by real Triton yet: Triton 3.6.0 has no macOS
  wheels and the local source build was not available. We checked every call against the v3.6.0
  sources, but the first Kaggle run is the first compilation.
- The 5% kernel share is an estimate until measured.

## Reproducibility

Pinned in the notebook: `soup-cli==0.77.0`, `transformers==5.16.1`, `peft==0.20.0`, `trl==0.29.1`,
`bitsandbytes==0.50.2`, `accelerate==1.14.0`. No extra packages. The harness starts every run with
empty compiler caches, and `harness/`, `tests/`, `task/` are the kit's files, unchanged.

## References

We wrote the code ourselves. Ideas came from:

- Soup's `soup_cli/utils/stream_loss_chunk.py` (issue #1693): the head and loss in token chunks, with
  the gradient taken on the hidden states. Our version adds the fused kernel and the padding removal.
- Triton tutorials [02-fused-softmax](https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/02-fused-softmax.py)
  and [05-layer-norm](https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/05-layer-norm.py):
  fused softmax over a row, and per-lane accumulators in a loop.
- The kit's `solution/swiglu.py`: the fused-kernel and `autograd.Function` pattern, kept running as is.

Apache-2.0, see `LICENSE` and `NOTICE`. Task data and model licenses are listed in `NOTICE`.
# AI-Engineering-by-Soup
