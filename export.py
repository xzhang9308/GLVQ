"""Merge fixed-code GLVQ fine-tuning weights into a layer-wise checkpoint."""

import argparse

import torch

from src.glvq import GLVQWeight
from src.modelutils import get_model, save_quantized_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--quantized_model", required=True)
    parser.add_argument("--p_finetuned_state_dict", required=True)
    parser.add_argument("--load_dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--save", required=True)
    args = parser.parse_args()
    model = get_model(args.base_model, args.quantized_model, dtype=getattr(torch, args.load_dtype))
    if not any(isinstance(module, GLVQWeight) for module in model.modules()):
        raise ValueError("Expected a GLVQ checkpoint")
    state = torch.load(args.p_finetuned_state_dict, map_location="cpu", weights_only=True)
    status = model.load_state_dict(state, strict=False)
    if status.unexpected_keys or any(not name.endswith("codes") for name in status.missing_keys):
        raise ValueError(f"Incompatible fine-tuning state: {status}")
    save_quantized_model(model, args.save)


if __name__ == "__main__":
    main()
