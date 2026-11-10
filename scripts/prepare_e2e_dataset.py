"""Convert a frozen AQLM/GLVQ calibration tensor into a tokenized HF dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import datasets
import torch


def load_input_ids(path: Path) -> torch.Tensor:
    input_ids = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(input_ids, (list, tuple)):
        if not input_ids:
            raise ValueError(f"empty token sequence collection in {path}")
        input_ids = torch.stack(
            [item.squeeze(0) if item.ndim == 2 and item.shape[0] == 1 else item for item in input_ids]
        )
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f"expected a tensor in {path}, found {type(input_ids).__name__}")
    if input_ids.ndim == 3 and input_ids.shape[1] == 1:
        input_ids = input_ids[:, 0]
    if input_ids.ndim != 2:
        raise ValueError(f"expected [num_sequences, sequence_length], found {tuple(input_ids.shape)}")
    if not input_ids.dtype.is_floating_point and input_ids.dtype != torch.bool:
        return input_ids.to(torch.int64).contiguous()
    raise TypeError(f"expected integer token ids, found {input_ids.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--num-sequences", type=int, default=None)
    parser.add_argument("--start-sequence", type=int, default=0)
    args = parser.parse_args()

    input_ids = load_input_ids(args.input)
    if not 0 <= args.start_sequence < input_ids.shape[0]:
        raise ValueError(f"invalid --start-sequence={args.start_sequence}")
    input_ids = input_ids[args.start_sequence :]
    if args.num_sequences is not None:
        if not 0 < args.num_sequences <= input_ids.shape[0]:
            raise ValueError(f"invalid --num-sequences={args.num_sequences}")
        input_ids = input_ids[: args.num_sequences]
    if args.sequence_length is not None and input_ids.shape[1] != args.sequence_length:
        raise ValueError(f"expected sequence length {args.sequence_length}, found {input_ids.shape[1]}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {args.output}")

    dataset = datasets.Dataset.from_dict({"input_ids": input_ids.numpy()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as temporary:
        pending = Path(temporary) / "dataset"
        dataset.save_to_disk(str(pending))
        reloaded = datasets.load_from_disk(str(pending))
        if len(reloaded) != input_ids.shape[0] or len(reloaded[0]["input_ids"]) != input_ids.shape[1]:
            raise RuntimeError("saved dataset failed shape validation")
        del reloaded
        pending.rename(args.output)
    print(
        f"saved source sequences [{args.start_sequence}:{args.start_sequence + input_ids.shape[0]}] "
        f"as {input_ids.shape[0]} sequences x {input_ids.shape[1]} tokens to {args.output}"
    )


if __name__ == "__main__":
    main()
