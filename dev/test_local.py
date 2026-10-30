"""Local checks without a CUDA GPU: `python -m pytest dev -q` (see dev/fake_triton).

Real transformers, peft, trl, accelerate and Soup's rewind trainer, on a tiny random Qwen2 on CPU,
with solution/'s kernels run by the emulator. Each test compares the patched step with the
unpatched one from the same seed.
"""

import sys
from pathlib import Path

import pytest
import torch
import transformers.generation.utils  # noqa: F401 - before torch can find the emulator below

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "dev" / "fake_triton"), str(ROOT)]

from solution import head, kernels, nf4, step


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float16, 2e-3)])
def test_rms_norm_matches_pytorch(dtype, tol):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    torch.manual_seed(0)
    norm = Qwen2RMSNorm(48, eps=1e-6).requires_grad_(False)
    norm.weight.data.uniform_(0.5, 1.5)
    x = torch.randn(3, 7, 48, dtype=dtype, requires_grad=True)
    x.data[0, 0] = 0  # a zero vector: the eps path
    seed = torch.randn(3, 7, 48)
    reference = norm(x)
    reference.backward(seed.to(reference.dtype))
    dx_reference, x.grad = x.grad, None
    out = kernels.rms_norm(x, norm.weight, norm.variance_epsilon)
    out.backward(seed.to(out.dtype))
    assert out.dtype == reference.dtype
    torch.testing.assert_close(out, reference, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad, dx_reference, rtol=max(tol, 2e-4), atol=max(tol, 2e-4))


@pytest.mark.parametrize("vocab", [1024, 1500, 2048])
def test_vocab_terms_match_pytorch(vocab):
    torch.manual_seed(0)
    logits = torch.randn(9, vocab) * 6
    logits[1, 3] = 80.0  # a row of one sure class
    logits[2] = logits[2].max()  # a flat row: ties
    labels = torch.randint(0, vocab, (9,), dtype=torch.int32)
    labels[3] = -1
    log_p = logits.log_softmax(-1)
    probe = logits.clone().requires_grad_(True)
    picked = (labels >= 0).nonzero()[:, 0]
    torch.nn.functional.cross_entropy(probe[picked], labels[picked].long(), reduction="sum").backward()
    entropy = -(log_p.exp() * log_p).sum(-1)
    ce = -log_p.gather(1, labels.clamp(min=0).long()[:, None])[:, 0] * (labels >= 0)
    written = logits.clone()
    loss, ent, top = kernels.vocab_terms(written, labels)
    torch.testing.assert_close(loss, ce, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ent, entropy, rtol=1e-4, atol=1e-4)
    assert top.tolist() == logits.argmax(-1).tolist()
    torch.testing.assert_close(written[picked], probe.grad[picked], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(written[3], log_p[3].exp(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("shape", [(8960, 1536), (256, 1536), (7, 13), (1, 63), (1, 65)])
def test_nf4_matches_bitsandbytes(shape):
    from bitsandbytes import functional

    torch.manual_seed(0)
    weight = torch.randn(shape) * torch.logspace(-3, 0, shape[1])
    packed, state = functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
    state.dtype = torch.float16
    reference = functional.dequantize_4bit(packed, state)
    state.dtype = torch.float32
    wide = functional.dequantize_4bit(packed, state)
    assert wide.shape == tuple(shape)
    for dtype, expected in ((torch.float32, wide), (torch.float16, reference)):
        state.dtype = dtype
        torch.testing.assert_close(nf4.nf4_dequantize(packed.reshape(-1, 1), state), expected,
                                   rtol=0, atol=0)


def test_fp16_gain_keeps_the_head_close_to_fp32():
    torch.manual_seed(0)
    weight = torch.randn(300, 32) * 0.05
    hidden = torch.randn(2, 40, 32, requires_grad=True)
    labels = torch.randint(0, 300, (2, 40))
    labels[:, :10] = -100
    labels[1, 30:] = -100
    mask = torch.ones_like(labels)
    mask[1, 31:] = 0
    denominator = (labels[:, 1:] != -100).sum()
    head.head_terms(hidden, weight, labels, mask, denominator).loss.backward()
    wide, hidden.grad = hidden.grad, None
    head.head_terms(hidden, weight.half(), labels, mask, denominator).loss.backward()
    torch.testing.assert_close(hidden.grad, wide, rtol=0, atol=2e-3 * wide.abs().max().item())


def test_split_rows_never_exceeds_one_long_row():
    for lengths in ([512, 512, 512, 512], [40, 60, 80, 500], [100, 120, 130, 140], [7]):
        groups = step.split_rows(lengths)
        assert sorted(i for g in groups for i in g) == list(range(len(lengths)))
        for group in groups:
            area = len(group) * max(lengths[i] for i in group)
            assert len(group) == 1 or area <= step.MAX_GROUP_AREA
    assert sorted(map(len, step.split_rows([100, 110, 120, 125]))) == [4]
    assert sorted(map(len, step.split_rows([40, 60, 80, 400]))) == [1, 3]


def test_a_batch_really_splits():
    lengths = [12, 39, 7, 25]
    groups = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(step, "GROUP_COST", 0)
        patch.setattr(step, "MAX_GROUP_AREA", 40)
        groups = step.split_rows(lengths)
    assert len(groups) > 1


def _config(**overrides):
    from transformers import Qwen2Config

    base = {"vocab_size": 1200, "hidden_size": 48, "intermediate_size": 96, "num_hidden_layers": 3,
                "num_attention_heads": 4, "num_key_value_heads": 2, "tie_word_embeddings": True,
                "max_position_embeddings": 128, "attn_implementation": "sdpa"}
    return Qwen2Config(**{**base, **overrides})


def _rows(count=16):
    from datasets import Dataset

    generator = torch.Generator().manual_seed(0)
    lengths = torch.randint(6, 40, (count,), generator=generator).tolist()
    rows = []
    for length in lengths:
        ids = torch.randint(5, 1200, (length,), generator=generator).tolist()
        prompt = max(1, length // 3)
        rows.append({"input_ids": ids, "labels": [-100] * prompt + ids[prompt:]})
    return Dataset.from_list(rows)


def _train(tmp_path, name, patched, monkeypatch, compact=False):
    from peft import LoraConfig, TaskType, get_peft_model
    from soup_cli.trainer.rewind_hf import attach_rewind_state, make_rewind_trainer_class
    from transformers import Qwen2ForCausalLM, set_seed
    from trl import SFTConfig, SFTTrainer

    records = []

    class Sink:
        def record_batch(self, **kwargs):
            records.append(kwargs)

    set_seed(7)
    model = Qwen2ForCausalLM(_config())
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type=TaskType.CAUSAL_LM))
    start = {k: v.clone() for k, v in model.named_parameters() if v.requires_grad}
    args = SFTConfig(output_dir=str(tmp_path / name), per_device_train_batch_size=4,
                     num_train_epochs=1, learning_rate=2e-3, logging_steps=1, save_strategy="no",
                     report_to="none", use_cpu=True, bf16=False, fp16=False, seed=7,
                     optim="adamw_torch",
                     lr_scheduler_type="cosine", max_length=64)
    trainer_class = make_rewind_trainer_class(SFTTrainer)
    trainer = trainer_class(model=model, args=args, train_dataset=_rows(),
                            processing_class=_tokenizer())
    state = attach_rewind_state(trainer, Sink())
    with monkeypatch.context() as patch:
        if patched:
            patch.setattr(step, "_on_gpu", lambda tensor: True)
            if not compact:  # the base has no autocast here, so its head is fp32 on CPU
                patch.setattr(step, "compact_head", lambda causal_lm: None)
            patch.setattr(step, "GROUP_COST", 0)  # these rows are short: force real splits
            patch.setattr(step, "MAX_GROUP_AREA", 40)
            patch.setattr(SFTTrainer, "training_step", SFTTrainer.training_step)
            step.patch_training_step()
        trainer.train()
    logged = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert not state.failed
    return {"model": model, "start": start, "logged": logged, "records": records,
            "trainer": trainer}


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1},
                                                                        unk_token="<pad>")),
                                   pad_token="<pad>", eos_token="<eos>")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    folder = tmp_path_factory.mktemp("runs")
    yield (_train(folder, "base", False, monkeypatch), _train(folder, "solution", True, monkeypatch),
           _train(folder, "compact", True, monkeypatch, compact=True))
    monkeypatch.undo()


def test_step_really_ran_the_padding_free_path(runs):
    assert getattr(runs[0]["trainer"], "_padding_free_plan", None) is None
    assert getattr(runs[1]["trainer"], "_padding_free_plan", None) is not None


def test_same_loss_entropy_accuracy_tokens(runs):
    base, solution = runs[:2]
    assert len(base["logged"]) == len(solution["logged"]) == 4
    for left, right in zip(base["logged"], solution["logged"]):
        assert sorted(left) == sorted(right)
        assert left["num_tokens"] == right["num_tokens"]
        assert right["mean_token_accuracy"] == pytest.approx(left["mean_token_accuracy"], abs=1e-6)
        assert right["entropy"] == pytest.approx(left["entropy"], rel=1e-4)
        assert right["loss"] == pytest.approx(left["loss"], rel=1e-4)
        assert right["grad_norm"] == pytest.approx(left["grad_norm"], rel=1e-4)


def test_same_rewind_records(runs):
    base, solution = runs[:2]
    assert len(base["records"]) == len(solution["records"]) > 0
    for left, right in zip(base["records"], solution["records"]):
        assert left["rows"] == right["rows"] and left["row_tokens"] == right["row_tokens"]
        assert (left["step"], left["micro"]) == (right["step"], right["micro"])
        assert right["row_loss"] == pytest.approx(left["row_loss"], rel=1e-4)


def test_same_lora_weights_after_training(runs):
    base, solution = runs[:2]
    lefts = dict(base["model"].named_parameters())
    rights = dict(solution["model"].named_parameters())
    moved = 0.0
    for name, start in base["start"].items():
        torch.testing.assert_close(rights[name], lefts[name], rtol=1e-3, atol=1e-5)
        moved = max(moved, (lefts[name] - start).abs().max().item())
    assert moved > 1e-3, "training must have moved the weights for this test to mean anything"


def test_fp16_head_stays_within_rounding_of_the_base(runs):
    base, _, compact = runs
    weight = compact["model"].get_base_model().lm_head.weight
    assert weight.dtype == torch.float16
    assert weight is compact["model"].get_base_model().get_input_embeddings().weight
    for left, right in zip(base["logged"], compact["logged"]):
        assert left["num_tokens"] == right["num_tokens"]
        assert right["loss"] == pytest.approx(left["loss"], rel=2e-3)
        assert right["entropy"] == pytest.approx(left["entropy"], rel=2e-3)
        assert right["grad_norm"] == pytest.approx(left["grad_norm"], rel=2e-2)
