import pytest
import torch
import torch.nn.functional as F
from src.glvq import (
    GLVQConfig,
    GLVQLinear,
    GLVQWeight,
    fit_glvq_weight,
    fit_glvq_weight_xtx,
    make_e8p_codebook,
    make_e8p_source_table,
)


def test_e8p_codebook_has_exact_size_unique_points_and_unit_rms():
    sources = make_e8p_source_table()
    codebook = make_e8p_codebook()
    assert sources.shape == (256, 8)
    assert codebook.shape == (1 << 16, 8)
    assert torch.unique(codebook, dim=0).shape[0] == 1 << 16
    assert torch.allclose(codebook.square().mean(), torch.tensor(1.0), atol=1e-06)


def test_e8p_two_byte_codes_decode_to_materialized_codebook():
    weight = torch.randn(8, 128)
    quantized = GLVQWeight(
        weight, GLVQConfig(codebook="e8p", dimension=8, bits=2, group_size=128, sigma_max=5.0)
    )
    values = torch.arange(1 << 16, dtype=torch.long)
    codes = torch.stack(((values >> 8).to(torch.uint8), (values & 255).to(torch.uint8)), dim=-1)
    assert torch.allclose(quantized._decode_e8p_codes(codes), make_e8p_codebook(), atol=1e-06)


def test_e8p_weight_uses_exactly_two_logical_bits_per_weight():
    torch.manual_seed(19)
    weight = torch.randn(16, 128)
    quantized = GLVQWeight(
        weight, GLVQConfig(codebook="e8p", dimension=8, bits=2, group_size=128, sigma_max=5.0)
    )
    assert quantized.codes.shape == (1, 256, 2)
    assert quantized.packed_codes().numel() == weight.numel() * 2 // 8
    assert quantized().shape == weight.shape
    assert torch.isfinite(quantized()).all()


def test_randomized_hadamard_rotation_is_exactly_invertible():
    torch.manual_seed(14)
    weight = torch.randn(8, 128)
    quantized = GLVQWeight(weight, GLVQConfig(hadamard_block_size=32, hadamard_seed=7, sigma_max=5.0))
    values = torch.randn(5, 128)
    restored = quantized.rotate_features(quantized.rotate_features(values), inverse=True)
    assert torch.allclose(restored, values, atol=2e-06, rtol=2e-06)


def test_rotated_xtx_preserves_quadratic_reconstruction_error():
    torch.manual_seed(15)
    weight = torch.randn(8, 128)
    quantized = GLVQWeight(weight, GLVQConfig(hadamard_block_size=128, hadamard_seed=3, sigma_max=5.0))
    delta = torch.randn_like(weight)
    inputs = torch.randn(64, 128)
    xtx = inputs.T @ inputs
    rotated_delta = quantized.rotate_features(delta)
    rotated_xtx = quantized.rotate_xtx(xtx)
    original_loss = (delta @ xtx * delta).sum()
    rotated_loss = (rotated_delta @ rotated_xtx * rotated_delta).sum()
    assert torch.allclose(rotated_loss, original_loss, atol=0.02, rtol=2e-05)


def test_mu_law_companding_is_invertible_in_weight_domain():
    torch.manual_seed(18)
    weight = torch.randn(8, 128) * 2.5
    quantized = GLVQWeight(
        weight,
        GLVQConfig(hadamard_block_size=128, hadamard_output_block_size=8, companding_mu=10.0, sigma_max=5.0),
    )
    rotated = quantized.rotate_outputs(quantized.rotate_features(weight))
    scales = quantized.scales().detach().reshape(8, 1)
    restored = quantized._decompand(quantized._compand(rotated / scales))
    restored = (restored * scales).reshape_as(weight)
    assert torch.allclose(restored, rotated, atol=2e-05, rtol=2e-05)


def test_companding_forward_stays_finite():
    torch.manual_seed(19)
    weight = torch.randn(8, 128)
    quantized = GLVQWeight(weight, GLVQConfig(companding_mu=5.0, sigma_max=5.0))
    assert torch.isfinite(quantized()).all()


def test_bounded_companding_and_learnable_group_mus():
    torch.manual_seed(20)
    weight = torch.randn(8, 256)
    quantized = GLVQWeight(
        weight,
        GLVQConfig(
            group_size=128,
            companding_mu=0.25,
            companding_clip=3.0,
            companding_learnable=True,
            companding_mu_max=1.0,
            sigma_max=5.0,
        ),
    )
    values = torch.empty(4, 2, 128).uniform_(-2.9, 2.9)
    restored = quantized._decompand(quantized._compand(values, group_dim=1), group_dim=1)
    assert torch.allclose(restored, values, atol=2e-06, rtol=2e-06)
    assert quantized.companding_logits is not None
    assert torch.all((quantized.companding_mus() > 0) & (quantized.companding_mus() < 1.0))


def test_companding_fit_optimizes_original_domain_with_joint_scale():
    torch.manual_seed(21)
    weight = torch.randn(8, 128)
    inputs = torch.randn(48, 128)
    xtx = inputs.T @ inputs
    quantized = GLVQWeight(
        weight, GLVQConfig(companding_mu=0.25, companding_clip=3.0, local_search_sweeps=1, sigma_max=5.0)
    )
    initial_delta = quantized() - weight
    initial_loss = (initial_delta @ xtx * initial_delta).sum()
    fit_glvq_weight_xtx(quantized, weight, xtx, outer_steps=2, generator_steps=2)
    final_delta = quantized() - weight
    final_loss = (final_delta @ xtx * final_delta).sum()
    assert final_loss <= initial_loss * (1 + 1e-06)
    assert quantized.log_scales.requires_grad


def test_bilateral_randomized_hadamard_rotation_is_exactly_invertible():
    torch.manual_seed(16)
    weight = torch.randn(64, 128)
    quantized = GLVQWeight(
        weight,
        GLVQConfig(
            hadamard_block_size=32,
            hadamard_seed=7,
            hadamard_output_block_size=32,
            hadamard_output_seed=11,
            sigma_max=5.0,
        ),
    )
    rotated = quantized.rotate_outputs(quantized.rotate_features(weight))
    restored = quantized.rotate_features(quantized.rotate_outputs(rotated, inverse=True), inverse=True)
    assert torch.allclose(restored, weight, atol=3e-06, rtol=3e-06)


def test_hadamard_2048_splits_llama2_intermediate_tail_on_input_axis():
    torch.manual_seed(25)
    weight = torch.randn(8, 11008)
    quantized = GLVQWeight(
        weight, GLVQConfig(group_size=128, dimension=8, hadamard_block_size=2048, sigma_max=5.0)
    )
    expected = (2048, 2048, 2048, 2048, 2048, 512, 256)
    assert quantized.hadamard_block_sizes == expected
    restored = quantized.rotate_features(quantized.rotate_features(weight), inverse=True)
    assert torch.allclose(restored, weight, atol=3e-05, rtol=3e-05)


def test_hadamard_2048_splits_llama2_intermediate_tail_on_output_axis():
    torch.manual_seed(26)
    weight = torch.randn(11008, 128)
    quantized = GLVQWeight(
        weight,
        GLVQConfig(
            group_size=128, dimension=8, hadamard_block_size=0, hadamard_output_block_size=2048, sigma_max=5.0
        ),
    )
    expected = (2048, 2048, 2048, 2048, 2048, 512, 256)
    assert quantized.hadamard_output_block_sizes == expected
    restored = quantized.rotate_outputs(quantized.rotate_outputs(weight), inverse=True)
    assert torch.allclose(restored, weight, atol=3e-05, rtol=3e-05)


def test_bilateral_rotation_preserves_hessian_quadratic_error():
    torch.manual_seed(17)
    weight = torch.randn(64, 128)
    quantized = GLVQWeight(
        weight, GLVQConfig(hadamard_block_size=128, hadamard_output_block_size=64, sigma_max=5.0)
    )
    delta = torch.randn_like(weight)
    inputs = torch.randn(64, 128)
    xtx = inputs.T @ inputs
    rotated_delta = quantized.rotate_outputs(quantized.rotate_features(delta))
    rotated_xtx = quantized.rotate_xtx(xtx)
    original_loss = (delta @ xtx * delta).sum()
    rotated_loss = (rotated_delta @ rotated_xtx * rotated_delta).sum()
    assert torch.allclose(rotated_loss, original_loss, atol=0.03, rtol=3e-05)


def test_glvq_layer_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(5)
    layer = torch.nn.Sequential(GLVQLinear.from_linear(torch.nn.Linear(128, 16)))
    inputs = torch.randn(3, 128)
    expected = layer(inputs)
    path = tmp_path / "0.pth"
    torch.save(layer, path)
    restored = torch.load(path, weights_only=False)
    assert torch.equal(restored(inputs), expected)
