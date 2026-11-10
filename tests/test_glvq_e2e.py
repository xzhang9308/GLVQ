import torch
import torch.nn as nn
from types import SimpleNamespace

from finetune import _continuous_quantized_parameters, create_p_optimizer
from src.glvq import GLVQConfig, GLVQWeight


def test_e2e_continuous_parameter_selection_includes_glvq_but_not_codes():
    quantized = GLVQWeight(
        torch.randn(16, 128),
        GLVQConfig(
            companding_mu=0.25,
            companding_learnable=True,
            companding_mu_max=8.0,
        ),
    )
    model = nn.Sequential(quantized, nn.LayerNorm(16))

    selected = _continuous_quantized_parameters(model)

    assert quantized.generators in selected
    assert quantized.offsets in selected
    assert quantized.log_scales in selected
    assert quantized.companding_logits in selected
    assert model[1].weight not in selected
    assert quantized.codes.dtype == torch.uint8
    assert not quantized.codes.requires_grad


def test_quantized_only_optimizer_excludes_non_quantized_parameters():
    quantized = GLVQWeight(
        torch.randn(16, 128),
        GLVQConfig(companding_mu=0.25, companding_learnable=True),
    )
    layernorm = nn.LayerNorm(16)
    model = nn.Sequential(quantized, layernorm)
    args = SimpleNamespace(
        update_codebooks_and_scales=True,
        update_non_quantized_parameters=False,
        update_codes=False,
        offload_optimizer=False,
        lr=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lamb=False,
        debias=True,
        amsgrad=False,
        master_dtype=torch.float32,
    )

    optimizer = create_p_optimizer(args, model)
    optimized_parameters = {param for group in optimizer.param_groups for param in group["params"]}

    assert layernorm.weight not in optimized_parameters
    assert layernorm.bias not in optimized_parameters
    assert quantized.generators in optimized_parameters
    assert optimized_parameters == _continuous_quantized_parameters(model)


def test_extended_p_optimizer_includes_non_quantized_parameters_but_not_codes():
    quantized = GLVQWeight(
        torch.randn(16, 128),
        GLVQConfig(companding_mu=0.25, companding_learnable=True),
    )
    layernorm = nn.LayerNorm(16)
    model = nn.Sequential(quantized, layernorm)
    args = SimpleNamespace(
        update_codebooks_and_scales=True,
        update_non_quantized_parameters=True,
        update_codes=False,
        offload_optimizer=False,
        lr=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lamb=False,
        debias=True,
        amsgrad=False,
        master_dtype=torch.float32,
    )

    optimizer = create_p_optimizer(args, model)
    optimized_parameters = {param for group in optimizer.param_groups for param in group["params"]}

    assert quantized.generators in optimized_parameters
    assert quantized.offsets in optimized_parameters
    assert quantized.log_scales in optimized_parameters
    assert quantized.companding_logits in optimized_parameters
    assert layernorm.weight in optimized_parameters
    assert layernorm.bias in optimized_parameters
    assert quantized.codes.dtype == torch.uint8
    assert not quantized.codes.requires_grad
