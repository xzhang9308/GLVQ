#!/usr/bin/env python3
"""Freeze the standard GPTQ-style C4 validation sample from a cached Arrow file."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import tempfile

import torch
from datasets import Dataset
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arrow_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--num-sequences", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Replace an incomplete pipeline artifact")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")

    dataset = Dataset.from_file(str(args.arrow_file))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rng = random.Random(args.seed)
    sequences = []
    selections = []
    for _ in range(args.num_sequences):
        while True:
            document_index = rng.randint(0, len(dataset) - 1)
            input_ids = tokenizer(dataset[document_index]["text"], return_tensors="pt").input_ids
            if input_ids.shape[1] >= args.sequence_length:
                break
        if input_ids.shape[1] == args.sequence_length:
            start = 0
        else:
            start = rng.randint(0, input_ids.shape[1] - args.sequence_length - 1)
        sequences.append(input_ids[:, start : start + args.sequence_length].to(torch.int64))
        selections.append({"document_index": document_index, "token_start": start})

    values = torch.hstack(sequences).contiguous()
    digest = hashlib.sha256(values.numpy().tobytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "c4_gptq_eval_tensor_v1",
        "arrow_file": str(args.arrow_file),
        "model": args.model,
        "num_sequences": args.num_sequences,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "shape": list(values.shape),
        "sha256": digest,
        "selections": selections,
    }
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as temporary:
        pending = Path(temporary) / args.output.name
        torch.save(values, pending)
        if not torch.equal(torch.load(pending, weights_only=True), values):
            raise RuntimeError("saved C4 tensor failed round-trip validation")
        pending_metadata = pending.with_suffix(pending.suffix + ".json")
        pending_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        pending.replace(args.output)
        pending_metadata.replace(args.output.with_suffix(args.output.suffix + ".json"))
    print({"shape": list(values.shape), "sha256": digest})


if __name__ == "__main__":
    main()
