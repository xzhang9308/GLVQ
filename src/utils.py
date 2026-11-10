"""Common utilities for GLVQ calibration and distributed training"""

from __future__ import annotations
import contextlib
import functools
import os
from typing import Callable, Iterator, Optional, Sequence
import torch
import torch.distributed


@functools.lru_cache()
def maybe_script(fn: callable) -> callable:
    """Apply torch.jit.script to function unless one is using TPU. TPU does not support torch.jit.script."""
    using_tpu = bool(os.environ.get("TPU_NAME"))
    should_script = int(os.environ.get("AQ_USE_JIT", not using_tpu))
    return torch.jit.script(fn) if should_script else fn


@contextlib.contextmanager
def using_tf32(enabled: bool):
    was_cudnn = torch.backends.cudnn.allow_tf32
    was_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled
    yield
    torch.backends.cudnn.allow_tf32 = was_cudnn
    torch.backends.cuda.matmul.allow_tf32 = was_matmul


def iterate_minibatches(
    *tensors: torch.Tensor,
    batch_size: int,
    allow_incomplete: bool = True,
    device: Optional[torch.device] = None,
    callback: Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]] = lambda x: x,
) -> Iterator[Sequence[torch.Tensor]]:
    """
    Samples data points *forever*, in random order, with less overhead than DataLoader;
    Adapted from https://github.com/stanis-morozov/unq/blob/master/lib/utils.py
    probably implemented over9000 times in transformers, torch, etc
    :param tensors: one or more tensors with the same 0-th dimension
    :param batch_size: sample this many points with each yield
    :param allow_incomplete: if True and if dataset size is not divisible by batch size, the last batch
        may have less than :batch_size: samples to cover the entire dataset. If False, the last batch is dropped
    :param callback: optional function to be called on each batch of tensors before it is yielded to the user
    :returns: generates a tuple of minibatches from each tensor, same length as input *tensors
        If a batch contains only one tensor, this function will yield a tensor (and not a tuple/list with one tensor)
    """
    num_samples = len(tensors[0])
    assert all((len(x) == num_samples for x in tensors))
    indices = torch.randperm(num_samples, device=tensors[0].device)
    while True:
        prev_batch = None
        for batch_start in range(0, len(indices), batch_size):
            if not allow_incomplete and batch_start + batch_size > len(indices):
                break
            batch_ix = indices[batch_start : batch_start + batch_size]
            batch = callback(tuple((tensor[batch_ix].to(device, non_blocking=True) for tensor in tensors)))
            if prev_batch is not None:
                yield prev_batch
            prev_batch = batch if isinstance(batch, (list, tuple)) and len(tensors) > 1 else batch[0]
            del batch
        yield prev_batch


@contextlib.contextmanager
def one_rank_at_a_time(local: bool = False, group_size: int = 1):
    """
    In distributed setting, let only group_size processes enter at a time
    :param local: if True, the limit is enforced within each host, i.e. distributed hosts can act concurrently
    :param group_size: if more than one is specified,
    """
    distributed = torch.distributed.is_initialized()
    rank = int(os.environ.get("LOCAL_RANK" if local else "RANK", 0)) if distributed else 0
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE" if local else "WORLD_SIZE", 0)) if distributed else 1
    if distributed:
        torch.distributed.barrier()
    for current_group_index in range(world_size // group_size):
        if current_group_index == rank // group_size:
            yield
        if distributed:
            torch.distributed.barrier()


@contextlib.contextmanager
def master_rank_first(local: bool, master_rank: int = 0):
    distributed = torch.distributed.is_initialized()
    rank = int(os.environ.get("LOCAL_RANK" if local else "RANK", 0)) if distributed else 0
    if distributed and rank != master_rank:
        torch.distributed.barrier()
    yield
    if distributed and rank == master_rank:
        torch.distributed.barrier()
