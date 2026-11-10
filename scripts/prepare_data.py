"""Build model-specific calibration, disjoint E2E, and frozen evaluation tokens."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer


def frozen_tokens_complete(path, shape=None):
    """Require the saved tensor and matching metadata, not a partial write."""
    try:
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        tokens = torch.load(path, map_location="cpu", weights_only=True)
        return (
            isinstance(tokens, torch.Tensor)
            and tokens.dtype == torch.int64
            and (shape is None or tuple(tokens.shape) == shape)
            and list(tokens.shape) == metadata["shape"]
            and hashlib.sha256(tokens.numpy().tobytes()).hexdigest() == metadata["sha256"]
        )
    except (OSError, ValueError, KeyError, RuntimeError, EOFError, TypeError, pickle.UnpicklingError):
        return False


def e2e_dataset_complete(path, count, seqlen):
    try:
        dataset = load_from_disk(str(path))
        return (
            isinstance(dataset, Dataset)
            and len(dataset) == count
            and "input_ids" in dataset.column_names
            and len(dataset[0]["input_ids"]) == seqlen
            and len(dataset[-1]["input_ids"]) == seqlen
        )
    except (OSError, ValueError, KeyError, RuntimeError, EOFError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--redpajama", required=True)
    parser.add_argument("--c4", required=True)
    parser.add_argument("--wiki", required=True)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "complete.txt").unlink(missing_ok=True)
    common = ["--model", args.model, "--sequence-length", str(args.seqlen)]
    for name, count, seed in [] if args.eval_only else [("calibration", 1024, 0), ("e2e", 2048, 20260811)]:
        output = args.output / (name + ".pt")
        if not frozen_tokens_complete(output, (count, 1, args.seqlen)):
            command = [
                sys.executable,
                "scripts/freeze_redpajama_calibration.py",
                str(output),
                *common,
                "--arrow-directory",
                args.redpajama,
                "--num-sequences",
                str(count),
                "--seed",
                str(seed),
                "--local-files-only",
                "--preserve-special-token-ids",
            ]
            if name == "e2e":
                command += ["--exclude-metadata", str(args.output / "calibration.pt.json")]
            subprocess.run(command, check=True)
    dataset_dir = args.output / "e2e-hf"
    if not args.eval_only and not e2e_dataset_complete(dataset_dir, 2048, args.seqlen):
        if dataset_dir.exists():
            # Preserve an interrupted/invalid artifact, including unknown contents.
            backup = dataset_dir.with_name(f".{dataset_dir.name}.incomplete-{uuid.uuid4().hex}")
            dataset_dir.rename(backup)
            print(f"Preserved incomplete dataset at {backup}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "scripts/prepare_e2e_dataset.py",
                str(args.output / "e2e.pt"),
                str(dataset_dir),
            ],
            check=True,
        )
    c4 = args.output / "c4.pt"
    if not frozen_tokens_complete(c4, (1, 256 * args.seqlen)):
        subprocess.run(
            [
                sys.executable,
                "scripts/freeze_c4_evaluation.py",
                args.c4,
                str(c4),
                *common,
                "--overwrite",
            ],
            check=True,
        )
    wiki = args.output / "wikitext2.pt"
    if not frozen_tokens_complete(wiki):
        use_fast = json.loads((Path(args.model) / "config.json").read_text()).get("model_type") == "qwen2"
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=use_fast)
        data = Dataset.from_file(args.wiki)
        tokens = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
        metadata = {
            "model": args.model,
            "source": args.wiki,
            "shape": list(tokens.shape),
            "sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
            "protocol": f"wikitext-2-raw-v1 test; join with two newlines; use_fast={use_fast}",
            "tokenizer_class": type(tokenizer).__name__,
        }
        with tempfile.TemporaryDirectory(prefix=".wikitext2-", dir=args.output) as temporary:
            pending = Path(temporary) / wiki.name
            torch.save(tokens, pending)
            pending_metadata = pending.with_suffix(".pt.json")
            pending_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
            pending.replace(wiki)
            pending_metadata.replace(wiki.with_suffix(".pt.json"))
    (args.output / "complete.txt").write_text("complete\n")


if __name__ == "__main__":
    main()
