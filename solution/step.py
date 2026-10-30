"""One training step without padding: the rows of a batch go through the model in groups.

The collator pads every row of a batch to the longest one. On this data that is 1.75 tokens padded
per real token, and one row of 512 tokens costs a whole 4 x 512 batch of activations. So the step
takes the padding out, the way the rules allow it: the batch stays the same four rows, they just
run forward and backward in smaller groups, and the gradients add up before the one optimizer
step of the batch.

What must not change does not change:
- The loss is the same sum over the same tokens, over the same denominator.
- LoRA dropout gets the masks of the padded batch: `DropoutPlan` draws them in the order and shape
  of the base step, so the random stream is the same one and each token keeps its mask.
- TRL's metrics and Soup's rewind record get the values they would compute from the full logits.
"""

import torch

from .head import IGNORE_INDEX, head_terms
from .kernels import rms_norm

MAX_GROUP_AREA = 512  # rows x padded length of a group: the memory one longest row needs anyway
GROUP_COST = 300  # the overhead of one more forward and backward, in tokens of work (on the T4)


def split_rows(lengths):
    """Groups of row indices that cost the least, by a small dynamic program over the rows sorted
    by length: a group costs GROUP_COST plus its padded area, and stays within MAX_GROUP_AREA
    unless it is one row."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    best = [(0, ())]
    for stop in range(1, len(order) + 1):
        choices = []
        for start in range(stop):
            area = (stop - start) * lengths[order[stop - 1]]
            if stop - start == 1 or area <= MAX_GROUP_AREA:
                cost, groups = best[start]
                choices.append((cost + GROUP_COST + area, groups + (order[start:stop],)))
        best.append(min(choices, key=lambda choice: choice[0]))
    return list(best[-1][1])


class DropoutPlan:
    """Replays the LoRA dropout of the padded batch on the rows of a group.

    The base step draws one mask per LoRA layer, first layer first and q_proj before v_proj, on
    a [batch, padded length, hidden] tensor. Drawing them here the same way, with the same call on
    a tensor of the same shape and dtype, takes the same numbers from the generator."""

    def __init__(self, layers):
        self.layers = layers  # dropout modules in the order the model calls them
        self.probability = layers[0].p
        self.waiting = []  # the masks of each group still to run, freed as the step goes on
        self.masks = None
        for index, layer in enumerate(layers):
            original = layer.forward
            layer.forward = self._replay(index, original)

    def _replay(self, index, original):
        def forward(x):
            if self.masks is None:
                return original(x)
            return x * (self.masks[index].to(x.dtype) / (1.0 - self.probability))

        return forward

    def draw(self, shape, dtype, device, groups):
        """`groups` holds (rows, length) of each group, in the order they will run."""
        ones = torch.ones(shape, dtype=dtype, device=device)
        masks = [torch.nn.functional.dropout(ones, self.probability, training=True) != 0
                 for _ in self.layers]
        self.waiting = [[mask.index_select(0, rows).narrow(1, 0, length) for mask in masks]
                        for rows, length in groups]

    def next_group(self):
        self.masks = self.waiting.pop(0)

    def clear(self):
        self.masks = None


def lora_dropouts(causal_lm):
    """The dropout modules of the task's LoRA layers in call order, or None if the model is not
    the shape this step knows."""
    found = []
    for layer in causal_lm.model.layers:
        attention = layer.self_attn
        for projection in (attention.q_proj, attention.v_proj):
            dropouts = getattr(projection, "lora_dropout", None)
            if dropouts is None or len(dropouts) != 1:
                return None
            dropout = next(iter(dropouts.values()))
            if not isinstance(dropout, torch.nn.Dropout) or not 0.0 < dropout.p < 1.0:
                return None
            found.append(dropout)
    return found


def runs_here(trainer, model, inputs):
    """The Qwen2 causal LM of the task, in a plain single-GPU training step. Anything else (the
    regression tests, evaluation) stays on the trainer's own path."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    args = trainer.args
    getter = getattr(model, "get_base_model", None)
    causal_lm = getter() if callable(getter) else None
    if not isinstance(causal_lm, Qwen2ForCausalLM):
        return None
    if {"input_ids", "attention_mask", "labels"} - set(inputs) or "shift_labels" in inputs:
        return None
    plain = (not args.use_liger_kernel and args.torch_empty_cache_steps is None
             and args.gradient_accumulation_steps == 1 and args.n_gpu <= 1
             and getattr(args, "loss_type", "nll") == "nll"
             and trainer.accelerator.num_processes == 1
             and trainer.label_smoother is None and trainer.compute_loss_func is None
             and not trainer.aux_loss_enabled and trainer.num_virtual_tokens == 0)
    head = causal_lm.lm_head.weight
    tied = causal_lm.get_input_embeddings().weight is head
    return causal_lm if plain and tied and not head.requires_grad and _on_gpu(head) else None


def _on_gpu(tensor):
    return tensor.is_cuda


def compact_head(causal_lm):
    """Keep the tied embedding and head in fp16: half their memory, and the head's matmul runs at
    fp16 speed with no copy per step. The checkpoint is bf16, whose 8-bit mantissa fits in fp16, and
    the base head rounds to fp16 under autocast anyway. The embedding's output goes back to fp32."""
    weight = causal_lm.lm_head.weight
    weight.data = weight.data.half()


def padding_free_step(trainer, model, causal_lm, plan, inputs, num_items_in_batch):
    model.train()
    if hasattr(trainer.optimizer, "train") and callable(trainer.optimizer.train):
        trainer.optimizer.train()
    inputs = trainer._prepare_inputs(inputs)
    input_ids, attention_mask, labels = (inputs["input_ids"], inputs["attention_mask"],
                                         inputs["labels"])
    denominator = num_items_in_batch
    if denominator is None:
        denominator = (labels[:, 1:] != IGNORE_INDEX).sum()
    weight = causal_lm.lm_head.weight
    lengths = attention_mask.sum(dim=1)
    groups = []
    for group in split_rows(lengths.tolist()):
        rows = torch.as_tensor(group, device=input_ids.device)
        groups.append((rows, int(lengths[rows].max())))
    # fp32: the dtype of the activations the base step's dropout sees.
    plan.draw((*input_ids.shape, causal_lm.config.hidden_size), torch.float32, input_ids.device,
              groups)

    loss = torch.zeros((), device=input_ids.device)
    entropy = torch.zeros((), device=input_ids.device)
    correct = torch.zeros((), dtype=torch.long, device=input_ids.device)
    row_ce = torch.zeros(len(input_ids), device=input_ids.device)
    for rows, longest in groups:
        ids = input_ids[rows, :longest]
        mask = attention_mask[rows, :longest]
        plan.next_group()
        embeddings = torch.nn.functional.embedding(ids, weight).float()
        with trainer.compute_loss_context_manager(), trainer.accelerator.autocast():
            hidden = causal_lm.model(inputs_embeds=embeddings,
                                     attention_mask=None if mask.all() else mask,
                                     use_cache=False).last_hidden_state
        terms = head_terms(hidden, weight, labels[rows, :longest], mask, denominator)
        del hidden
        trainer.accelerator.backward(terms.loss)
        loss += terms.loss.detach()
        entropy += terms.entropy
        correct += terms.correct
        row_ce[rows] = terms.row_ce
        del terms
    plan.clear()

    row_tokens = (labels[:, 1:] != IGNORE_INDEX).sum(dim=1)
    _record(trainer, attention_mask, entropy, correct, row_tokens, row_ce)
    return loss.detach()


def _record(trainer, attention_mask, entropy, correct, row_tokens, row_ce):
    """What TRL's compute_loss and Soup's rewind recorder take from the logits."""
    attended = attention_mask.sum()
    labelled = row_tokens.sum()
    metrics = trainer._metrics["train"]
    metrics["entropy"].append((entropy / attended).item())
    trainer._total_train_tokens += attended.item()
    metrics["num_tokens"] = [trainer._total_train_tokens]
    metrics["mean_token_accuracy"].append((correct / labelled).item() if labelled > 0 else 0.0)

    state = getattr(trainer, "_soup_rewind_state", None)
    if state is None or state.failed:
        return
    try:
        means = row_ce / row_tokens.clamp(min=1)
        state.on_micro_batch(step=int(trainer.state.global_step) + 1,
                             row_loss=[float(mean) for mean in means],
                             row_tokens=[int(count) for count in row_tokens])
    except Exception as exc:  # noqa: BLE001 - as in Soup, the recorder never stops training
        state.fail(exc)


def patch_training_step():
    """A class patch on trl.SFTTrainer: Soup builds its trainer class from it during setup."""
    from trl import SFTTrainer

    original = SFTTrainer.training_step

    def training_step(self, model, inputs, num_items_in_batch=None):
        causal_lm = runs_here(self, model, inputs)
        if causal_lm is None:
            return original(self, model, inputs, num_items_in_batch)
        plan = getattr(self, "_padding_free_plan", None)
        if plan is None:
            layers = lora_dropouts(causal_lm)
            if layers is None:
                return original(self, model, inputs, num_items_in_batch)
            compact_head(causal_lm)
            plan = self._padding_free_plan = DropoutPlan(layers)
        return padding_free_step(self, model, causal_lm, plan, inputs, num_items_in_batch)

    SFTTrainer.training_step = training_step


def patch_rms_norm():
    """Qwen2RMSNorm.forward through the fused kernels. The norms' weights are frozen in this task;
    a trainable weight stays on the original forward."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        if self.weight.requires_grad or not _on_gpu(hidden_states):
            return original(self, hidden_states)
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward
