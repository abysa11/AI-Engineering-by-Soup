# Submission: PENDING (team name)

## Links
- Repository, final commit: PENDING (`https://github.com/<team>/<repo>/commit/<sha>`)
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in the README): PENDING KAGGLE VALIDATION
- Demo video (up to 3 minutes): PENDING
- Presentation: PENDING

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| PENDING | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `vocab_terms_kernel` | `solution/kernels.py` | log_softmax and its gradient, TRL's entropy and argmax, rewind's per-row cross-entropy | PENDING KAGGLE VALIDATION |
| `rms_norm_fwd_kernel` | `solution/kernels.py` | `Qwen2RMSNorm.forward` | PENDING KAGGLE VALIDATION |
| `rms_norm_bwd_kernel` | `solution/kernels.py` | the backward of `Qwen2RMSNorm` | PENDING KAGGLE VALIDATION |

The kit's example kernels (`example_swiglu_*`) keep running next to these and do not count as ours.

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Soup, `soup_cli/utils/stream_loss_chunk.py` (issue #1693): head and loss in token chunks.
- Triton tutorials [02-fused-softmax](https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/02-fused-softmax.py)
  and [05-layer-norm](https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/05-layer-norm.py).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | PENDING | PENDING | speedup PENDING |
| Peak GPU memory, MiB | PENDING | PENDING | base/ours PENDING |
| Eval loss (public rows) | PENDING | PENDING | PENDING % vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits (results
  still PENDING KAGGLE VALIDATION)

## Presentation covers
Padding as the main cost of this task, padding-free groups with the same dropout masks, the fused
vocabulary head and RMSNorm kernels, results against the base (speed, memory, quality), limits and
next steps.

## Limitations
The peak is still set by one 512-token row: going lower needs recomputation for long rows. The head
runs in fp32 where the base runs fp16, so values match to rounding and not to the bit. Model loading
has its own memory peak. See the README for the full list.
