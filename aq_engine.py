from __future__ import annotations
import math
import json
from argparse import Namespace
from typing import Optional
import torch
import torch.nn as nn
from src.glvq import GLVQConfig, GLVQWeight, fit_glvq_weight_xtx


class AQEngine(nn.Module):
    """Activation statistics and fixed-recipe GLVQ fitting for one Linear layer."""

    def __init__(self, layer: nn.Linear, accumulator_dtype: torch.dtype = torch.float64):
        super().__init__()
        self.layer = layer
        self.device = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.register_buffer(
            "XTX", torch.zeros((self.columns, self.columns), dtype=accumulator_dtype, device=self.device)
        )
        self.quantized_weight: Optional[GLVQWeight] = None
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """Accumulate a minibatch of layer inputs and update the X.T @ X (aka half hessian)"""
        assert self.XTX is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        tmp = inp.shape[0]
        inp = inp.t()
        self.XTX *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(1 / self.nsamples) * inp.to(self.XTX.dtype)
        self.XTX += inp.matmul(inp.t())

    @torch.enable_grad()
    def quantize_glvq(self, *, args: Namespace, verbose: bool = True) -> GLVQWeight:
        """Fit first-stage GLVQ using the activation statistics already collected."""
        config = GLVQConfig(
            dimension=args.glvq_dimension,
            bits=args.glvq_bits,
            codebook=getattr(args, "glvq_codebook", "e8p"),
            deformation=getattr(args, "glvq_deformation", "full"),
            e8p_candidate_count=getattr(args, "glvq_e8p_candidate_count", 8),
            e8p_encode_chunk_size=getattr(args, "glvq_e8p_encode_chunk_size", 1024),
            group_size=args.glvq_group_size,
            covariance_eps=args.glvq_covariance_eps,
            init_clip=args.glvq_init_clip,
            sigma_min=args.glvq_sigma_min,
            sigma_max=args.glvq_sigma_max,
            scale_eps=getattr(args, "glvq_scale_eps", 1e-08),
            scale_min_factor=getattr(args, "glvq_scale_min_factor", 0.25),
            scale_max_factor=getattr(args, "glvq_scale_max_factor", 4.0),
            offset_max_delta=getattr(args, "glvq_offset_max_delta", 1.0),
            hessian_damp=getattr(args, "glvq_hessian_damp", 0.01),
            hessian_encoder=getattr(args, "glvq_hessian_encoder", "gptq"),
            local_search_sweeps=getattr(args, "glvq_local_search_sweeps", 0),
            hadamard_block_size=getattr(args, "glvq_hadamard_block_size", 0),
            hadamard_seed=getattr(args, "glvq_hadamard_seed", 0),
            hadamard_output_block_size=getattr(args, "glvq_hadamard_output_block_size", 0),
            hadamard_output_seed=getattr(args, "glvq_hadamard_output_seed", 1),
            hadamard_full=getattr(args, "glvq_hadamard_full", False),
            companding_mu=getattr(args, "glvq_companding_mu", 0.0),
            companding_mu_min=getattr(args, "glvq_companding_mu_min", 0.0),
            companding_clip=getattr(args, "glvq_companding_clip", 0.0),
            companding_learnable=getattr(args, "glvq_companding_learnable", False),
            companding_mu_max=getattr(args, "glvq_companding_mu_max", 1.0),
            companding_init=getattr(args, "glvq_companding_init", "constant"),
            companding_normalization=getattr(args, "glvq_companding_normalization", "rms"),
            companding_scope=getattr(args, "glvq_companding_scope", "group"),
            companding_soft_tail=getattr(args, "glvq_companding_soft_tail", False),
        )
        quantized_weight = GLVQWeight(self.layer.weight.detach().float(), config).to(self.device)
        with torch.no_grad():
            reference_weight = self.layer.weight.detach().to(dtype=torch.float32)
            xtx = self.XTX.float()
            initial_delta = quantized_weight() - reference_weight
            signal = (reference_weight @ xtx * reference_weight).sum().clamp_min(1e-12)
            initial_loss = (initial_delta @ xtx * initial_delta).sum() / signal
        history = fit_glvq_weight_xtx(
            quantized_weight,
            self.layer.weight,
            self.XTX,
            outer_steps=args.glvq_outer_steps,
            generator_steps=args.glvq_generator_steps,
            learning_rate=args.glvq_learning_rate,
            regularization=args.glvq_regularization,
            generator_learning_rate=getattr(args, "glvq_generator_learning_rate", None),
            offset_learning_rate=getattr(args, "glvq_offset_learning_rate", None),
            generator_regularization=getattr(args, "glvq_generator_regularization", None),
            offset_regularization=getattr(args, "glvq_offset_regularization", None),
            solver=getattr(args, "glvq_solver", "adam"),
            continuous_update_tolerance=getattr(args, "glvq_continuous_update_tolerance", 0.0),
            relative_tolerance=getattr(args, "glvq_relative_tolerance", 0.0001),
            patience=getattr(args, "glvq_patience", 3),
            min_learning_rate=getattr(args, "glvq_min_learning_rate", 1e-06),
            code_update_tolerance=getattr(args, "glvq_code_update_tolerance", 0.0),
            max_grad_norm=getattr(args, "glvq_max_grad_norm", 1.0),
            companding_learning_rate=getattr(args, "glvq_companding_learning_rate", None),
            companding_regularization=getattr(args, "glvq_companding_regularization", 1e-05),
        )
        if verbose:
            with torch.no_grad():
                delta = quantized_weight() - reference_weight
                final_loss = (delta @ xtx * delta).sum() / signal
                edge_fraction = (
                    torch.tensor(float("nan"), device=quantized_weight.codes.device)
                    if config.codebook == "e8p"
                    else ((quantized_weight.codes == 0) | (quantized_weight.codes == config.levels - 1))
                    .float()
                    .mean()
                )
                singular_values = torch.linalg.svdvals(quantized_weight.generators.float())
                condition_numbers = singular_values[..., 0] / singular_values[..., -1].clamp_min(1e-12)
                companding_mus = quantized_weight.companding_mus().float()
            print(
                f"GLVQ relative loss={initial_loss:.10f}->{final_loss:.10f}; outer_steps={len(history)}; codebook={config.codebook}; deformation={config.deformation}; edge_codes={edge_fraction:.6f}; generator_condition_mean={condition_numbers.mean():.6f}; generator_condition_max={condition_numbers.max():.6f}; actual bpw={quantized_weight.estimate_nbits_per_parameter():.6f}"
            )
            if config.companding_enabled:
                print(
                    f"GLVQ companding mu min={companding_mus.min():.6f}; mean={companding_mus.mean():.6f}; max={companding_mus.max():.6f}"
                )
            print(
                "GLVQ local search diagnostics="
                + json.dumps(getattr(quantized_weight, "last_search_diagnostics", {}), sort_keys=True)
            )
            print(
                "GLVQ fit diagnostics="
                + json.dumps(getattr(quantized_weight, "last_fit_diagnostics", {}), sort_keys=True)
            )
        self.quantized_weight = quantized_weight
        self.XTX = None
        return quantized_weight


def replace_parameter_(module: nn.Module, name: str, new_value: torch.Tensor):
    """A hacky way to substitute an already registered parameter with a non-parameter tensor. Breaks future use."""
    if name in module._parameters:
        module._parameters[name] = new_value
    else:
        setattr(module, name, new_value)
