
"""Stable grading interface for the SFT and DPO implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _encode(tokenizer, text):
    """Tokenize text without automatically adding special tokens."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def _make_batch(tokenizer, prompts, responses, max_length):
    """Build a right-padded causal-LM batch with response-only labels."""
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have equal lengths")
    if not prompts:
        raise ValueError("The batch must contain at least one example")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("The tokenizer must define eos_token_id")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id

    rows = []
    label_rows = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = _encode(tokenizer, prompt)
        response_ids = _encode(tokenizer, response)

        # Always terminate the response with EOS.
        response_ids = response_ids[: max_length - 1]
        response_ids.append(eos_id)

        # Preserve the full response where possible. If needed, truncate
        # the prompt from the left to fit within max_length.
        available_prompt_length = max_length - len(response_ids)
        if available_prompt_length < 0:
            response_ids = response_ids[:max_length]
            response_ids[-1] = eos_id
            available_prompt_length = 0

        if available_prompt_length == 0:
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[-available_prompt_length:]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids.copy()

        rows.append(input_ids)
        label_rows.append(labels)

    width = max(len(row) for row in rows)
    input_ids_batch = []
    attention_batch = []
    labels_batch = []

    for input_ids, labels in zip(rows, label_rows):
        padding_length = width - len(input_ids)
        input_ids_batch.append(input_ids + [pad_id] * padding_length)
        attention_batch.append([1] * len(input_ids) + [0] * padding_length)
        labels_batch.append(labels + [-100] * padding_length)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
    }


def _model_device(model):
    """Find the device of the model's parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _response_logprobs_from_batch(model, batch):
    """Sum causal log probabilities over response tokens only."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = outputs.logits

    # A logit at position t predicts the token at position t + 1.
    shift_logits = logits[:, :-1, :].float()
    shift_labels = batch["labels"][:, 1:]
    valid = shift_labels.ne(-100)

    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_logprobs = F.log_softmax(shift_logits, dim=-1)
    selected = token_logprobs.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    return (selected * valid.to(selected.dtype)).sum(dim=-1)


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""
    return _make_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    batch = _move_batch(batch, _model_device(model))
    return _response_logprobs_from_batch(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return the mean per-example summed response negative log probability."""
    logps = compute_response_logprobs(
        model, tokenizer, prompts, responses, max_length
    )
    return -logps.mean()


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""
    if len(prompts) != len(chosen_responses) or len(prompts) != len(rejected_responses):
        raise ValueError("Prompts, chosen responses, and rejected responses must have equal lengths")

    return {
        "chosen": _make_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": _make_batch(tokenizer, prompts, rejected_responses, max_length),
    }


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""
    if beta <= 0:
        raise ValueError("beta must be positive")

    policy_chosen_logps = torch.as_tensor(policy_chosen_logps)
    device = policy_chosen_logps.device

    policy_rejected_logps = torch.as_tensor(
        policy_rejected_logps, device=device
    )
    reference_chosen_logps = torch.as_tensor(
        reference_chosen_logps, device=device
    )
    reference_rejected_logps = torch.as_tensor(
        reference_rejected_logps, device=device
    )

    policy_ratio = policy_chosen_logps - policy_rejected_logps
    reference_ratio = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_ratio - reference_ratio)

    return -F.logsigmoid(logits).mean()


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""
    batches = build_dpo_batch(
        tokenizer, prompts, chosen_responses, rejected_responses, max_length
    )

    policy_device = _model_device(policy_model)
    reference_device = _model_device(reference_model)

    policy_chosen = _response_logprobs_from_batch(
        policy_model, _move_batch(batches["chosen"], policy_device)
    )
    policy_rejected = _response_logprobs_from_batch(
        policy_model, _move_batch(batches["rejected"], policy_device)
    )

    with torch.no_grad():
        reference_chosen = _response_logprobs_from_batch(
            reference_model, _move_batch(batches["chosen"], reference_device)
        )
        reference_rejected = _response_logprobs_from_batch(
            reference_model, _move_batch(batches["rejected"], reference_device)
        )

    # Keep all four vectors on the policy device for the loss calculation.
    return compute_dpo_loss_from_logps(
        policy_chosen,
        policy_rejected,
        reference_chosen.to(policy_device),
        reference_rejected.to(policy_device),
        beta,
    )
