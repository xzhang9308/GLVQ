"""Small local models exercise the shipped checkpoint export without downloads."""

import subprocess
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, MistralConfig, Qwen2Config

from run import ROOT
from scripts.evaluate import materialize
from src.glvq import GLVQConfig, GLVQLinear
from src.modelutils import get_model, save_quantized_model


@pytest.mark.parametrize("config_type", [LlamaConfig, MistralConfig, Qwen2Config])
def test_export_reload_and_decode_preserve_finetuned_weights(tmp_path, config_type):
    torch.manual_seed(42)
    config = config_type(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    base = tmp_path / "base"
    model = AutoModelForCausalLM.from_config(config).eval()
    model.save_pretrained(base)
    attention = model.model.layers[0].self_attn
    attention.q_proj = GLVQLinear.from_linear(
        attention.q_proj,
        GLVQConfig(
            codebook="e8p",
            dimension=8,
            deformation="full",
            hadamard_block_size=2048,
            hadamard_output_block_size=2048,
            companding_mu=0.25,
            companding_learnable=True,
            companding_mu_max=8,
            companding_clip=3,
            companding_soft_tail=True,
        ),
    )
    codes = attention.q_proj.quantized_weight.codes.clone()
    checkpoint = tmp_path / "quantized"
    save_quantized_model(model, str(checkpoint))
    with torch.no_grad():
        attention.q_proj.quantized_weight.generators.mul_(1.01)
    state = tmp_path / "final.pt"
    torch.save({k: v for k, v in model.state_dict().items() if not k.endswith("codes")}, state)
    exported = tmp_path / "exported"
    subprocess.run(
        [
            sys.executable,
            "export.py",
            "--base_model",
            str(base),
            "--quantized_model",
            str(checkpoint),
            "--p_finetuned_state_dict",
            str(state),
            "--load_dtype",
            "float32",
            "--save",
            str(exported),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    restored = get_model(str(base), str(exported), dtype=torch.float32).eval()
    q = restored.model.layers[0].self_attn.q_proj.quantized_weight
    assert torch.equal(q.codes, codes)
    assert torch.equal(q.generators, attention.q_proj.quantized_weight.generators)
    ids = torch.arange(8).reshape(1, -1)
    with torch.no_grad():
        expected = restored(ids).logits
        materialize(restored, torch.device("cpu"), torch.float32)
        actual = restored(ids).logits
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
