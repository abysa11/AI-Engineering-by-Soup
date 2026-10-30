# Final audit

Audit date: 2026-10-30. Written without access to a CUDA GPU, Kaggle or the team repository.

**Verdict: not ready to submit.** Compliance and local correctness pass. Every GPU-dependent item is
open, including whether the kernels compile, and the team repository is missing.

How to read the evidence column:

- **Local (real):** ran on a MacBook on CPU with the pinned transformers, peft, trl, accelerate and
  soup-cli. Python 3.14 and torch 2.15, not the task's 3.13 and 2.11.
- **Local (emulator):** kernels' Python run by `dev/fake_triton`, which is not Triton.
- **Kaggle T4:** nothing yet.

| Area | Status | Evidence | Remaining work |
|---|---|---|---|
| Repository structure | BLOCKED | The working folder was empty, not a git repository. The official starter (`MakazhanAlpamys/soup-kernel-challenge`, commit `c7c2fcd`) was copied in. Other teams' repositories were not opened. | Move this work into the platform's team repository; only that one is checked. Needs its URL. |
| Official configuration | PASS | `task/task.yaml` is byte-identical to the starter (`diff -rq`). `solution/` never reads or edits the config or trainer settings. | Check 5 confirms on T4. |
| Competition compliance | PASS | Local: `check_rules.source_kernels` finds no banned imports. No Soup switch is touched. `harness/`, `tests/`, `task/` and the template are unchanged. Row grouping stays inside each batch, which the README of the kit allows. | The jury's reading of row grouping as padding removal is the main judgement call; it is disclosed in the README. Check 2 on T4. |
| Triton kernels | NOT TESTED | Local (emulator): forward logic correct. Static: 3 own kernels recognized; likeness to the example 0.31, 0.61, 0.58 (review from 0.80). Every Triton call was read against the v3.6.0 sources. No real compilation: no macOS wheels, and the local source build was refused by the session's permission layer. | First Kaggle run is the first compilation. `python -m pytest solution -q` runs first in the notebook. |
| apply() integration | PASS | Local (real + emulator): `trl.SFTTrainer.training_step` patch reaches Soup's `RewindSFTTrainer`; the differential test proves the padding-free path ran. | Check 3 on T4. |
| Numerical correctness | PASS (local only) | Local (emulator): 17 GPU tests pass in a CPU translation, covering zero vectors, ties, one sure class, unlabelled rows, padding and dtypes. Local differential, autocast off: loss identical, worst per-parameter gradient 1.2e-6 relative, with dropout on. | Same tests on T4, where `tl.exp` and `tl.log` are approximate. Numbers here say nothing about GPU accuracy. |
| Backward correctness | PASS (local only) | Local (emulator): RMSNorm backward and head gradient against autograd. LoRA weights after training within 1e-3 relative of the base. | GradScaler and fp16 autocast have not run: CPU has neither. T4 only. |
| Soup tests | PASS (local only) | Local: 79 of 79, with and without `SOUP_APPLY_SOLUTION=1`, emulator in place of Triton, Python 3.14. | Check 10 on T4 with Python 3.13. |
| check_rules.py | NOT TESTED | Needs `runs/result.json` from a T4 run. Only its static kernel scan was run. | Run in the notebook. |
| Full training | NOT TESTED | No T4 run. Local: 4 steps of a tiny random model through the real trainer. | Run in the notebook. |
| Kernel time percentage | NOT TESTED | Unmeasured. Our estimate of about 9% is a calculation, not a result. | Read the notebook's summary cell. Below 5% means adding a kernel. |
| GPU memory | NOT TESTED | Unmeasured. Largest group area falls from 2,048 to 512 positions on the public rows; that is arithmetic on token counts, not memory. | Read the notebook's summary cell. |
| Evaluation loss | NOT TESTED | Unmeasured. | Check 7 on T4. |
| Kaggle Run All | BLOCKED | `kaggle_notebook.ipynb` is the template plus a summary and an environment cell, and `--repeats 2 --profile`. It has never run. | Needs a Kaggle account, `REPO_URL` and a person to press Run All. |
| README | PASS (results pending) | Rewritten; every result is marked `PENDING KAGGLE VALIDATION`. | Fill in real numbers and the two links. |
| SUBMISSION.md | BLOCKED | Filled as far as known. Team, links and results are `PENDING`. | Needs team details, video, presentation, final commit and Kaggle link. |

## Judge's review of the diff

Findings from reading the final code as a judge would:

1. **Kernels are not decorative.** The vocabulary kernel's outputs are the loss, the gradient on the
   hidden states and the logged metrics. RMSNorm's output is the hidden state.
2. **Row grouping is the strongest and the most exposed idea.** It changes no batch, row, order or
   trainer setting, and `tokens per step`, rewind rows and token counts stay equal by construction.
3. **The head uses `torch.mm`,** not a kernel, for its matmuls. These are cuBLAS calls; the 5% share
   must come from the kernels alone.
4. **Guards are by device and model type,** not by run mode. Nothing looks at the harness, the
   environment or whether a run is measured.
5. **Open risk, compile failure:** if a kernel fails to compile, solution runs exit non-zero and
   check 1 fails. There is deliberately no silent fallback, because a run that quietly dropped the
   kernels would be a different solution from the one described.

## What has to happen before submitting

1. Put this folder's contents in the team repository and push.
2. Run the notebook on a T4 and read `solution` test failures first.
3. Fix what the run shows, using `runs/solution_profile/profile.json`.
4. Fill README and SUBMISSION with measured numbers, set `COMMIT`, save a public version.
