"""Exercise the real PTQ entry point on small, local decoder models."""

import json
import subprocess
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, MistralConfig, Qwen2Config

from run import ROOT, complete_layers
from src.glvq import GLVQLinear
from src.modelutils import get_model


@pytest.mark.parametrize("config_type", [LlamaConfig, MistralConfig, Qwen2Config])
def test_quantize_block_finetune_and_resume(tmp_path, config_type):
    torch.manual_seed(123)
    config = config_type(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    base = tmp_path / "base"
    AutoModelForCausalLM.from_config(config).save_pretrained(base)
    data = tmp_path / "calibration.pt"
    torch.save(torch.randint(0, 128, (4, 1, 8)), data)
    checkpoint = tmp_path / "checkpoint"
    recipe = json.loads((ROOT / "recipe.json").read_text())
    recipe.update(
        dtype="float32",
        nsamples=4,
        glvq_outer_steps=1,
        glvq_generator_steps=1,
        finetune_max_epochs=1,
        finetune_batch_size=1,
        offload_activations=False,
    )
    command = [
        sys.executable,
        "main.py",
        str(base),
        str(data),
        "--devices",
        "cpu",
        "--model_seqlen",
        "8",
        "--val_size",
        "1",
        "--save",
        str(checkpoint),
    ]
    for key, value in recipe.items():
        if value is True:
            command += ["--" + key]
        elif value is not False:
            command += ["--" + key, str(value)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert complete_layers(checkpoint, 1)
    model = get_model(str(base), str(checkpoint), dtype=torch.float32).eval()
    assert sum(isinstance(m, GLVQLinear) for m in model.modules()) == 7
    with torch.no_grad():
        assert torch.isfinite(model(torch.arange(8).reshape(1, -1)).logits).all()
    saved_bytes = (checkpoint / "0.pth").read_bytes()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (checkpoint / "0.pth").read_bytes() == saved_bytes
