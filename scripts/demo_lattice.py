"""Fit a small learnable lattice on CPU without downloads or file outputs."""

import torch

from src.glvq import GLVQConfig, GLVQWeight, fit_glvq_weight_xtx


def main():
    torch.manual_seed(42)
    reference = torch.randn(16, 128)
    inputs = torch.randn(64, 128)
    xtx = inputs.T @ inputs / inputs.shape[0]
    lattice = GLVQWeight(
        reference,
        GLVQConfig(
            group_size=128,
            init_clip=2.0,
            hadamard_block_size=2048,
            hadamard_output_block_size=2048,
            companding_mu=0.25,
            companding_learnable=True,
            companding_mu_max=8.0,
            companding_clip=3.0,
            companding_soft_tail=True,
        ),
    )
    initial = lattice.generators.detach().clone()
    loss = ((inputs @ lattice().T) - (inputs @ reference.T)).square().mean()
    loss.backward()
    assert lattice.generators.grad is not None
    assert torch.isfinite(lattice.generators.grad).all()
    assert lattice.generators.grad.abs().sum() > 0
    lattice.zero_grad(set_to_none=True)
    fit_glvq_weight_xtx(
        lattice, reference, xtx,
        outer_steps=2, generator_steps=3,
        companding_learning_rate=0.01, companding_regularization=0.0,
    )
    decoded = lattice().detach()
    assert decoded.shape == reference.shape and torch.isfinite(decoded).all()
    assert not torch.equal(initial, lattice.generators)
    assert lattice.packed_codes().numel() * 8 == reference.numel() * 2
    print(f"Decoded shape: {tuple(decoded.shape)}")
    print("Trainable generator: gradient checked and parameters updated")
    print("Code storage: 2 bits/weight (continuous metadata excluded)")
    print("Synthetic example only; no model quality or performance claim.")


if __name__ == "__main__":
    main()
