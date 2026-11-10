#!/usr/bin/env python3
"""Freeze an AQLM-compatible RedPajama calibration tensor reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import tempfile

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset", default="togethercomputer/RedPajama-Data-1T-Sample")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument(
        "--arrow-directory",
        type=Path,
        default=None,
        help="Load sorted *-train-*.arrow shards from this existing datasets cache directory",
    )
    parser.add_argument(
        "--exclude-metadata",
        type=Path,
        action="append",
        default=[],
        help="Reject token spans overlapping selections in a prior freeze metadata JSON",
    )
    parser.add_argument("--num-sequences", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--preserve-special-token-ids",
        action="store_true",
        help="Keep native BOS/EOS IDs for newer tokenizers (e.g. Llama 3 / Qwen)",
    )
    return parser.parse_args()


def tensor_sha256(values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(values.contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.num_sequences <= 0 or args.sequence_length <= 0:
        raise ValueError("num-sequences and sequence-length must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    # Match AQLM's RedPajama loader exactly.
    if not args.preserve_special_token_ids:
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
    if args.arrow_directory is None:
        dataset = load_dataset(args.dataset, split="train", revision=args.dataset_revision)
    else:
        arrow_files = sorted(args.arrow_directory.glob("*-train-*.arrow"))
        if not arrow_files:
            raise FileNotFoundError(f"no cached train Arrow shards found in {args.arrow_directory}")
        dataset = concatenate_datasets([Dataset.from_file(str(path)) for path in arrow_files])
    rng = random.Random(args.seed)

    occupied_spans: dict[int, list[tuple[int, int]]] = {}
    for metadata_path in args.exclude_metadata:
        excluded = json.loads(metadata_path.read_text())
        excluded_length = int(excluded["sequence_length"])
        for selection in excluded["selections"]:
            start = int(selection["token_start"])
            occupied_spans.setdefault(int(selection["document_index"]), []).append(
                (start, start + excluded_length)
            )

    sequences: list[torch.Tensor] = []
    selections: list[dict[str, int]] = []
    rejected_overlaps = 0
    progress = tqdm(total=args.num_sequences, desc="Freezing RedPajama calibration set")
    while len(sequences) < args.num_sequences:
        while True:
            document_index = rng.randint(0, len(dataset) - 1)
            input_ids = tokenizer(dataset[document_index]["text"], return_tensors="pt").input_ids
            if input_ids.shape[1] > args.sequence_length:
                break
        start = rng.randint(0, input_ids.shape[1] - args.sequence_length - 1)
        end = start + args.sequence_length
        if any(
            start < occupied_end and occupied_start < end
            for occupied_start, occupied_end in occupied_spans.get(document_index, ())
        ):
            rejected_overlaps += 1
            continue
        sequence = input_ids[:, start : start + args.sequence_length].to(torch.int64).contiguous()
        sequences.append(sequence)
        occupied_spans.setdefault(document_index, []).append((start, end))
        selections.append(
            {
                "document_index": document_index,
                "token_start": start,
                "document_tokens": input_ids.shape[1],
            }
        )
        progress.update(1)
    progress.close()

    calibration = torch.stack(sequences)
    expected_shape = (args.num_sequences, 1, args.sequence_length)
    if tuple(calibration.shape) != expected_shape:
        raise RuntimeError(f"unexpected calibration shape {tuple(calibration.shape)} != {expected_shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "aqlm_calibration_tensor_v1",
        "model": args.model,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_class": type(tokenizer).__name__,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "dataset": args.dataset,
        "dataset_revision_requested": args.dataset_revision,
        "arrow_directory": str(args.arrow_directory) if args.arrow_directory is not None else None,
        "exclude_metadata": [str(path) for path in args.exclude_metadata],
        "rejected_overlaps": rejected_overlaps,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_split": str(dataset.split),
        "seed": args.seed,
        "num_sequences": args.num_sequences,
        "sequence_length": args.sequence_length,
        "num_tokens": args.num_sequences * args.sequence_length,
        "shape": list(calibration.shape),
        "dtype": str(calibration.dtype),
        "sha256": tensor_sha256(calibration),
        "selections": selections,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as temporary:
        pending = Path(temporary) / args.output.name
        torch.save(calibration, pending)
        reloaded = torch.load(pending, map_location="cpu", weights_only=True)
        if not torch.equal(reloaded, calibration):
            raise RuntimeError("saved calibration tensor failed round-trip validation")
        pending_metadata = pending.with_suffix(pending.suffix + ".json")
        pending_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        pending.replace(args.output)
        pending_metadata.replace(metadata_path)
    print(json.dumps({key: metadata[key] for key in ("shape", "num_tokens", "sha256")}, indent=2))
    print(f"saved tensor: {args.output}")
    print(f"saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
