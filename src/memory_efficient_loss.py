"""
Utility functions for computing a KL divergence loss without materializing all logits / logprobs simultaneously
"""

from typing import Callable, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

T = TypeVar("T")


def compute_kl_divergence_loss_values(
    *,
    student_hidden_states: torch.Tensor,
    student_lm_head: nn.Module,
    teacher_hidden_states: torch.Tensor,
    teacher_lm_head: nn.Module,
    max_tokens_per_chunk: int = 256,
    checkpoint_last_chunk: bool = True,
    **checkpoint_kwargs,
) -> torch.Tensor:
    """
    Compute token-wise KL divergence loss without materializing all logits/logprobs simultaneously
    :param student_hidden_states: input hidden states for student head, [batch_size, sequence_length, student_dim]
    :param student_lm_head: a token-wise layer (e.g. nn.Linear) mapping from student_dim to logits [vocabulary_size]
    :param teacher_hidden_states: input hidden states for teacher head, [batch_size, sequence_length, teacher_dim]
    :param teacher_lm_head: a token-wise layer (e.g. nn.Linear) mapping from teacher_dim to logits [vocabulary_size]
    :note: teacher is applied to hidden states without no_grad. If required, set requires_grad=False on teacher manually
    :param max_tokens_per_chunk: materialize logits logprobs for at most this many tokens at a time
    :param checkpoint_kwargs: additional arguments passed to checkpoint (e.g. use_reentrant or determinism_check)
    :param checkpoint_last_chunk: if False, do not apply gradient checkpointing to the very last chunk of inputs
        since they are the first ones to be re-materialized anyway. Useful if loss is backpropagated immediately.
    :returns: token-wise KL loss values of shape [batch_size, sequence_length]
    """
    assert (
        student_hidden_states.requires_grad
        or teacher_hidden_states.requires_grad
        or not torch.is_grad_enabled()
    )
    assert teacher_hidden_states.shape[:-1] == student_hidden_states.shape[:-1]
    flat_student_hidden_states = student_hidden_states.flatten(0, -2)
    flat_teacher_hidden_states = teacher_hidden_states.flatten(0, -2)
    total_tokens = flat_teacher_hidden_states.shape[0]

    loss_values_by_chunk = []
    for chunk_start in range(0, total_tokens, max_tokens_per_chunk):
        is_last_chunk = chunk_start + max_tokens_per_chunk >= total_tokens
        loss_values_by_chunk.append(
            maybe_checkpoint(
                _compute_kl_div_from_flat_hidden_states,
                flat_student_hidden_states[chunk_start : chunk_start + max_tokens_per_chunk],
                student_lm_head,
                flat_teacher_hidden_states[chunk_start : chunk_start + max_tokens_per_chunk],
                teacher_lm_head,
                checkpoint_enabled=torch.is_grad_enabled() and (checkpoint_last_chunk or not is_last_chunk),
                **checkpoint_kwargs,
            )
        )
    return torch.cat(loss_values_by_chunk).reshape(*student_hidden_states.shape[:2])


def _compute_kl_div_from_flat_hidden_states(
    flat_student_hidden_states: torch.Tensor,
    student_lm_head: nn.Module,
    flat_teacher_hidden_states: torch.Tensor,
    teacher_lm_head: nn.Module,
) -> torch.Tensor:
    student_logprobs = F.log_softmax(student_lm_head(flat_student_hidden_states), dim=-1)
    teacher_logprobs = F.log_softmax(teacher_lm_head(flat_teacher_hidden_states), dim=-1)
    return F.kl_div(input=student_logprobs, target=teacher_logprobs, log_target=True, reduction="none").sum(
        -1
    )


def maybe_checkpoint(func: Callable[[...], T], *inputs, checkpoint_enabled: bool, **checkpoint_kwargs) -> T:
    """Execute function normally or with checkpointing, depending on checkpoint_enabled. Forward **checkpoint_kwargs"""
    return func(*inputs) if checkpoint_enabled else checkpoint(func, *inputs, **checkpoint_kwargs)
