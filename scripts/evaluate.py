"""Evaluate floating/GLVQ checkpoints with one frozen protocol; materialize for accuracy."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from src.glvq import GLVQLinear
from src.modelutils import get_model


def load_state(model, path):
    if not path:
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    status = model.load_state_dict(state, strict=False)
    if status.unexpected_keys or any(not k.endswith("codes") for k in status.missing_keys):
        raise RuntimeError(f"Incompatible state dict: {status}")


@torch.no_grad()
def materialize(model, device, dtype):
    """Replace GLVQ layers by exactly their decoded weights, preserving bias."""
    details = []
    quant_parameter_ids = set()
    for name, module in list(model.named_modules()):
        if not isinstance(module, GLVQLinear):
            continue
        q = module.quantized_weight
        quant_parameter_ids.update(id(p) for p in q.parameters())
        details.append(
            {
                "module": name,
                "weights": q.in_features * q.out_features,
                "deployable_bits": q.storage_nbits(),
                "code_bits": q.codes.numel() * (8 if q.config.codebook == "e8p" else q.config.bits),
                "generator_bits": q.generators.numel() * q.generators.element_size() * 8,
                "offset_bits": q.offsets.numel() * q.offsets.element_size() * 8,
                "scale_bits": q.log_scales.numel() * q.log_scales.element_size() * 8,
                "mu_bits": (
                    0
                    if q.companding_logits is None
                    else q.companding_logits.numel() * q.companding_logits.element_size() * 8
                ),
            }
        )
    other = {id(p): p for p in model.parameters() if id(p) not in quant_parameter_ids}
    stats = {
        "quantized_modules": details,
        "nonquantized_parameter_count": sum(p.numel() for p in other.values()),
        "nonquantized_loaded_bits": sum(p.numel() * p.element_size() * 8 for p in other.values()),
    }
    for name, module in list(model.named_modules()):
        if isinstance(module, GLVQLinear):
            q = module.quantized_weight.to(device)
            linear = nn.Linear(
                q.in_features, q.out_features, bias=module.bias is not None, device="meta", dtype=dtype
            )
            linear.weight = nn.Parameter(q().to(dtype).cpu(), requires_grad=False)
            if module.bias is not None:
                linear.bias = nn.Parameter(
                    module.bias.detach().to(device="cpu", dtype=dtype), requires_grad=False
                )
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, linear)
            del q, module
    return stats


@torch.no_grad()
def perplexity(model, tokens, seqlen, device):
    count = tokens.shape[1] // seqlen
    if count == 0:
        raise ValueError("not enough evaluation tokens")
    total_nll = 0.0
    total_tokens = 0
    for index in range(count):
        ids = tokens[:, index * seqlen : (index + 1) * seqlen].to(device)
        logits = model(ids, use_cache=False).logits
        for start in range(0, seqlen - 1, 128):
            end = min(seqlen - 1, start + 128)
            total_nll += (
                nn.functional.cross_entropy(
                    logits[:, start:end].float().reshape(-1, logits.shape[-1]),
                    ids[:, start + 1 : end + 1].reshape(-1),
                    reduction="sum",
                )
                .double()
                .item()
            )
            total_tokens += end - start
        if index % 16 == 0:
            print(f"PPL progress {index+1}/{count}", flush=True)
    return {
        "ppl": math.exp(total_nll / total_tokens),
        "nll": total_nll,
        "predicted_tokens": total_tokens,
        "sequences": count,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quantized")
    p.add_argument("--state")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--seqlen", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["ppl", "accuracy"], default="ppl")
    p.add_argument("--limit", type=int)
    p.add_argument("--task", default="hellaswag,piqa,winogrande,arc_easy,arc_challenge")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    model = get_model(args.model, args.quantized, dtype=dtype, attn_implementation="sdpa").eval()
    load_state(model, args.state)
    storage = materialize(model, device, dtype)
    # Pickled legacy blocks can retain FP32 norms even when the base is BF16.
    # Cast only after decoding, so the quantizer's saved parameters stay intact.
    model.to(device=device, dtype=dtype).eval()
    model.config.use_cache = False
    started = time.time()
    result = {
        "model": args.model,
        "quantized": args.quantized,
        "state": args.state,
        "dtype": "bfloat16",
        "sequence_length": args.seqlen,
        "storage": storage,
        "protocol": "frozen tokens; every complete nonoverlapping window; FP32 CE; decoded GLVQ weights cast BF16",
        "smoke_only": args.limit is not None,
    }
    if args.mode == "ppl":
        result["metrics"] = {}
        for name in ("wikitext2", "c4"):
            tokens = torch.load(args.data / (name + ".pt"), map_location="cpu", weights_only=True)
            digest = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
            if args.limit:
                tokens = tokens[:, : args.limit * args.seqlen]
            score = perplexity(model, tokens, args.seqlen, device)
            score["token_sha256"] = digest
            result["metrics"][name] = score
            print(f"{name} perplexity: {score['ppl']:.9f}", flush=True)
    else:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM

        use_fast = json.loads((Path(args.model) / "config.json").read_text()).get("model_type") == "qwen2"
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=use_fast)
        result["tokenizer_class"] = type(tokenizer).__name__
        wrapped = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, max_length=args.seqlen)
        result["accuracy"] = evaluator.simple_evaluate(
            model=wrapped,
            tasks=args.task.split(","),
            num_fewshot=0,
            batch_size=1,
            limit=args.limit,
            random_seed=0,
            numpy_random_seed=0,
            torch_random_seed=0,
            fewshot_random_seed=0,
            log_samples=False,
        )
    result["elapsed_seconds"] = time.time() - started
    result["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, default=str) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
