from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
from scripts.evaluate import materialize, perplexity
from run import complete_layers
from src.glvq import GLVQConfig, GLVQLinear


def test_materialization_preserves_decoded_weights_and_bias():
    layer = nn.Linear(128, 8, bias=True)
    model = nn.Sequential(GLVQLinear.from_linear(layer, GLVQConfig(dimension=8)))
    x = torch.randn(2, 128)
    before = model(x)
    stats = materialize(model, torch.device("cpu"), torch.float32)
    assert isinstance(model[0], nn.Linear)
    assert torch.allclose(model(x), before)
    assert torch.equal(model[0].bias, layer.bias)
    assert stats["quantized_modules"][0]["weights"] == 1024
    assert stats["nonquantized_parameter_count"] == 8


def test_ppl_includes_last_full_window_but_not_remainder():
    class UniformLM(nn.Module):
        def forward(self, ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros(*ids.shape, 4))

    result = perplexity(UniformLM(), torch.zeros(1, 13, dtype=torch.long), 4, "cpu")
    assert result["sequences"] == 3
    assert result["predicted_tokens"] == 9
    assert result["ppl"] == pytest.approx(4)


def test_checkpoint_completeness_uses_model_layer_count(tmp_path):
    for name in ("0.pth", "1.pth", "args.pt", "not_quantized_weights.pt"):
        (tmp_path / name).touch()
    assert complete_layers(tmp_path, 2)
    assert not complete_layers(tmp_path, 3)
