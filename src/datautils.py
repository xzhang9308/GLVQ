"""Frozen calibration/evaluation tokens and shared training data helpers."""

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed
from torch import nn
from tqdm.auto import tqdm


def set_seed(seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    """Load the frozen full WikiText2 evaluation token stream."""
    frozen_eval_file = os.environ.get("WIKITEXT2_EVAL_FILE")
    if not eval_mode or not frozen_eval_file:
        raise ValueError("Use the frozen WikiText2 evaluation tokens prepared by run.py")
    tokens = torch.load(frozen_eval_file, map_location="cpu", weights_only=True)
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("WIKITEXT2_EVAL_FILE must contain a [1, num_tokens] tensor")
    if tokens.dtype not in (torch.int32, torch.int64) or tokens.shape[1] < seqlen:
        raise ValueError("WIKITEXT2_EVAL_FILE must contain enough integer tokens for one sequence")
    print(f"Loaded frozen WikiText2 evaluation tokens from {frozen_eval_file}")
    return tokens


def get_c4(nsamples, seqlen, tokenizer, eval_mode=False):
    """Load the frozen C4 evaluation windows."""
    frozen_eval_file = os.environ.get("C4_EVAL_FILE")
    if not eval_mode or not frozen_eval_file:
        raise ValueError("Use the frozen C4 evaluation tokens prepared by run.py")
    valenc = torch.load(frozen_eval_file, map_location="cpu", weights_only=False)
    if not isinstance(valenc, torch.Tensor) or valenc.ndim != 2:
        raise ValueError("C4_EVAL_FILE must contain a 2-D token tensor")
    expected_tokens = 256 * seqlen
    if valenc.numel() != expected_tokens:
        raise ValueError(f"C4_EVAL_FILE has {valenc.numel()} tokens; expected {expected_tokens}")
    print(f"Loaded frozen C4 evaluation tokens from {frozen_eval_file}")
    return valenc


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
    model_path=None,
    use_fast_tokenizer=False,
    trust_remote_code=None,
):
    """Load saved calibration tokens or frozen WikiText2/C4 evaluation tokens.

    Model/tokenizer arguments are retained for the calibration and training
    entry-point interface; the saved tokens already use the model's tokenizer.
    """
    set_seed(seed)
    if os.path.isfile(name):
        data = torch.load(name)[:nsamples]
    elif name.lower() == "wikitext2":
        data = get_wikitext2(nsamples, seqlen, None, eval_mode=eval_mode)
    elif name.lower() == "c4":
        data = get_c4(nsamples, seqlen, None, eval_mode=eval_mode)
    else:
        raise ValueError(f"Expected a saved token tensor or frozen evaluation dataset, found {name}")
    print(f"Loaded data from {name}; {len(data)=} sequences")
    return data


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    data: torch.Tensor,
    seqlen: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> float:
    """Perplexity evaluation as per https://github.com/IST-DASLab/gptq (standard among quantization research)"""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    inps = [
        data[:, start : start + seqlen]
        for start in range(0, data.shape[1], seqlen)
        if start + seqlen < data.shape[1]
    ]  # ignore last incomplete sequence as in the GPTQ paper
    num_sequences_without_padding = len(inps)

    # pad sequences to be divisible by world_size for DDP/FSDP compatibility
    num_padding_sequences = -len(inps) % world_size
    inps.extend([inps[-1]] * num_padding_sequences)

    total_nll_and_tokens = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    total_nll, total_tokens = total_nll_and_tokens[0], total_nll_and_tokens[1]

    for sequence_index, input_ids in enumerate(
        tqdm(inps, desc="Evaluating perplexity") if rank == 0 else inps
    ):
        if sequence_index % world_size != rank:
            continue
        input_ids = input_ids.to(device)
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            lm_logits = model(input_ids).logits

        if sequence_index < num_sequences_without_padding:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_nll += loss.float() * shift_labels.numel()
            total_tokens += shift_labels.numel()

    if world_size > 1:
        torch.distributed.all_reduce(total_nll_and_tokens, op=torch.distributed.ReduceOp.SUM)
    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()
