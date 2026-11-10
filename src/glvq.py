"""Learnable lattice weights, Hadamard transforms and companding.

The release recipe uses d8 E8P with full generator optimization: each vector
has a 16-bit code. Companding acts in the rotated, row/group-scale-normalized
domain and is inverted during decoding. See recipe.json for release settings.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GLVQConfig:
    dimension: int = 8
    bits: int = 2
    codebook: str = "e8p"
    deformation: str = "full"
    e8p_candidate_count: int = 8
    e8p_encode_chunk_size: int = 1024
    group_size: int = 128
    covariance_eps: float = 1e-06
    init_clip: float = 1.5
    sigma_min: float = 1e-05
    sigma_max: float = 5.0
    scale_eps: float = 1e-08
    scale_min_factor: float = 0.25
    scale_max_factor: float = 4.0
    offset_max_delta: float = 1.0
    hessian_damp: float = 0.01
    hessian_encoder: str = "gptq"
    local_search_sweeps: int = 0
    hadamard_block_size: int = 0
    hadamard_seed: int = 0
    hadamard_output_block_size: int = 0
    hadamard_output_seed: int = 1
    hadamard_full: bool = False
    companding_mu: float = 0.0
    companding_mu_min: float = 0.0
    companding_clip: float = 0.0
    companding_learnable: bool = False
    companding_mu_max: float = 1.0
    companding_init: str = "constant"
    companding_normalization: str = "rms"
    companding_scope: str = "group"
    companding_soft_tail: bool = False

    def __post_init__(self) -> None:
        if self.codebook != "e8p" or self.deformation != "full" or self.dimension != 8 or (self.bits != 2):
            raise ValueError("This release supports d8 E8P/full with 2-bit codes only")
        if self.hessian_encoder != "gptq":
            raise ValueError("This release supports the GPTQ encoder only")
        if (
            self.companding_init != "constant"
            or self.companding_normalization != "rms"
            or self.companding_scope != "group"
        ):
            raise ValueError("This release supports constant-init, RMS, group-wise companding only")
        if self.hadamard_full:
            raise ValueError("This release uses block Hadamard transforms")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.e8p_candidate_count <= 0 or self.e8p_candidate_count > 512:
            raise ValueError("e8p_candidate_count must be in [1, 512]")
        if self.e8p_encode_chunk_size <= 0:
            raise ValueError("e8p_encode_chunk_size must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.group_size % self.dimension:
            raise ValueError("group_size must be divisible by dimension")
        if not 0 < self.sigma_min <= self.sigma_max:
            raise ValueError("expected 0 < sigma_min <= sigma_max")
        if not 0 < self.scale_min_factor <= 1 <= self.scale_max_factor:
            raise ValueError("expected scale_min_factor <= 1 <= scale_max_factor")
        if self.offset_max_delta < 0:
            raise ValueError("offset_max_delta must be non-negative")
        if self.hessian_damp < 0:
            raise ValueError("hessian_damp must be non-negative")
        if self.companding_mu < 0:
            raise ValueError("companding_mu must be non-negative")
        if self.companding_clip < 0:
            raise ValueError("companding_clip must be non-negative")
        if not 0 <= self.companding_mu_min < self.companding_mu_max:
            raise ValueError("expected 0 <= companding_mu_min < companding_mu_max")
        if (
            self.companding_learnable
            and self.companding_init == "constant"
            and (not self.companding_mu_min < self.companding_mu < self.companding_mu_max)
        ):
            raise ValueError("learnable companding requires mu_min < mu < mu_max")
        if self.local_search_sweeps < 0:
            raise ValueError("local_search_sweeps must be non-negative")
        if self.hadamard_block_size < 0 or (
            self.hadamard_block_size and self.hadamard_block_size & self.hadamard_block_size - 1
        ):
            raise ValueError("hadamard_block_size must be zero or a power of two")
        if self.hadamard_output_block_size < 0 or (
            self.hadamard_output_block_size
            and self.hadamard_output_block_size & self.hadamard_output_block_size - 1
        ):
            raise ValueError("hadamard_output_block_size must be zero or a power of two")

    @property
    def levels(self) -> int:
        return 1 << self.bits

    @property
    def companding_enabled(self) -> bool:
        return self.companding_mu > 0.0

    @property
    def uses_rms_scale(self) -> bool:
        return not self.companding_enabled or self.companding_normalization == "rms"


_E8P_PADDING_DOUBLED = (
    (3, 1, 1, 1, 3, 3, 3, 3),
    (1, 3, 1, 1, 3, 3, 3, 3),
    (1, 1, 3, 1, 3, 3, 3, 3),
    (1, 1, 1, 3, 3, 3, 3, 3),
    (3, 3, 3, 1, 3, 3, 1, 1),
    (3, 3, 3, 1, 3, 1, 3, 1),
    (3, 3, 3, 1, 1, 3, 3, 1),
    (3, 3, 3, 1, 3, 1, 1, 3),
    (3, 3, 3, 1, 1, 3, 1, 3),
    (3, 3, 3, 1, 1, 1, 3, 3),
    (3, 3, 1, 3, 3, 3, 1, 1),
    (3, 3, 1, 3, 3, 1, 3, 1),
    (3, 3, 1, 3, 1, 3, 3, 1),
    (3, 3, 1, 3, 3, 1, 1, 3),
    (3, 3, 1, 3, 1, 3, 1, 3),
    (3, 3, 1, 3, 1, 1, 3, 3),
    (3, 1, 3, 3, 3, 3, 1, 1),
    (3, 1, 3, 3, 3, 1, 3, 1),
    (3, 1, 3, 3, 1, 3, 3, 1),
    (3, 1, 3, 3, 3, 1, 1, 3),
    (3, 1, 3, 3, 1, 3, 1, 3),
    (1, 3, 3, 3, 1, 1, 3, 3),
    (1, 3, 3, 3, 3, 3, 1, 1),
    (1, 3, 3, 3, 3, 1, 3, 1),
    (1, 3, 3, 3, 1, 3, 3, 1),
    (1, 3, 3, 3, 3, 1, 1, 3),
    (1, 3, 3, 3, 1, 3, 1, 3),
    (1, 1, 3, 3, 1, 3, 3, 3),
    (3, 3, 1, 1, 3, 3, 3, 1),
)


def make_e8p_source_table() -> torch.Tensor:
    """Build the 256 absolute E8P representatives from the paper definition."""
    core = [values for values in product((1, 3, 5), repeat=8) if sum((v * v for v in values)) <= 40]
    if len(core) != 227:
        raise RuntimeError("internal E8P construction error")
    doubled = torch.tensor(core + list(_E8P_PADDING_DOUBLED), dtype=torch.float32)
    return doubled / 2


def _make_e8p_raw_codebook() -> torch.Tensor:
    sources = make_e8p_source_table()
    source_ids = torch.arange(256, dtype=torch.int64).repeat_interleave(256)
    suffix = torch.arange(256, dtype=torch.int64).repeat(256)
    sign_bits = suffix & 127
    shift_bits = suffix >> 7
    signs = 1 - 2 * (sign_bits[:, None] >> torch.arange(7) & 1)
    source_doubled = (sources[source_ids] * 2).to(torch.int64)
    required_parity = source_doubled.sum(dim=-1) // 2 & 1
    sign_parity = sum((sign_bits >> bit & 1 for bit in range(7))) & 1
    last_negative = sign_parity ^ required_parity
    all_signs = torch.cat((signs, (1 - 2 * last_negative)[:, None]), dim=-1).float()
    shifts = (shift_bits.float() * 2 - 1)[:, None] / 4
    return sources[source_ids] * all_signs + shifts


def make_e8p_codebook() -> torch.Tensor:
    """Materialize all 2^16 E8P points, normalized to unit coordinate RMS."""
    points = _make_e8p_raw_codebook()
    return points / points.square().mean().sqrt()


class GLVQWeight(nn.Module):
    """Group-specific learned bounded lattice representation of a 2-D weight.

    Groups partition the input-column axis.  Inside each group, weights are
    flattened into contiguous ``dimension``-vectors.  A separate generation
    matrix is shared by all vectors in that group.
    """

    def __init__(self, reference_weight: torch.Tensor, config: Optional[GLVQConfig] = None):
        super().__init__()
        if reference_weight.ndim != 2:
            raise ValueError("reference_weight must be a 2-D [out_features, in_features] tensor")
        self.config = config or GLVQConfig()
        (self.out_features, self.in_features) = reference_weight.shape
        if self.in_features % self.config.group_size:
            raise ValueError("stage one requires in_features to be divisible by group_size")
        if self.out_features * self.config.group_size % self.config.dimension:
            raise ValueError("each group must contain a whole number of lattice vectors")
        reference_weight = reference_weight.detach().float()
        input_block = self.config.hadamard_block_size
        self.hadamard_block_size = input_block
        if input_block:
            self.hadamard_block_sizes = self._decompose_hadamard_blocks(self.in_features, input_block)
            for block in self.hadamard_block_sizes:
                self._validate_hadamard_width(block)
            generator = torch.Generator(device="cpu").manual_seed(self.config.hadamard_seed)
            signs = self._random_hadamard_signs(self.hadamard_block_sizes, generator)
            self.register_buffer("hadamard_signs", signs)
            reference_weight = self.rotate_features(reference_weight)
        else:
            self.hadamard_block_sizes = ()
            self.register_buffer("hadamard_signs", torch.empty(0, dtype=torch.int8))
        output_block = min(self.config.hadamard_output_block_size, self.out_features)
        self.hadamard_output_block_size = output_block
        if output_block:
            self.hadamard_output_block_sizes = self._decompose_hadamard_blocks(
                self.out_features, output_block
            )
            for block in self.hadamard_output_block_sizes:
                self._validate_hadamard_width(block)
            generator = torch.Generator(device="cpu").manual_seed(self.config.hadamard_output_seed)
            signs = self._random_hadamard_signs(self.hadamard_output_block_sizes, generator)
            self.register_buffer("hadamard_output_signs", signs)
            reference_weight = self.rotate_outputs(reference_weight)
        else:
            self.hadamard_output_block_sizes = ()
            self.register_buffer("hadamard_output_signs", torch.empty(0, dtype=torch.int8))
        grouped = self._weight_to_groups(reference_weight)
        if self.config.uses_rms_scale:
            initial_scales = (
                grouped.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(self.config.scale_eps)
            )
        else:
            initial_scales = torch.ones(
                self.out_features,
                self.num_groups,
                1,
                device=reference_weight.device,
                dtype=reference_weight.dtype,
            )
        self.log_scales = nn.Parameter(initial_scales.log().float(), requires_grad=self.config.uses_rms_scale)
        self.register_buffer("initial_log_scales", self.log_scales.detach().clone())
        if self.config.companding_learnable:
            num_mus = self.num_groups
            initial_mus = self._initialize_companding_mus(reference_weight)
            if initial_mus.numel() != num_mus:
                raise RuntimeError("companding initializer returned an unexpected number of values")
            ratio = (initial_mus - self.config.companding_mu_min) / (
                self.config.companding_mu_max - self.config.companding_mu_min
            )
            eps = torch.finfo(ratio.dtype).eps
            ratio = ratio.clamp(eps, 1.0 - eps)
            self.companding_logits = nn.Parameter(torch.logit(ratio).float())
            self.register_buffer("initial_companding_logits", self.companding_logits.detach().clone())
        else:
            self.register_parameter("companding_logits", None)
            self.register_buffer("initial_companding_logits", torch.empty(0))
        vectors = self._weight_to_vectors(self.compand_weight(reference_weight))
        source_table = make_e8p_source_table()
        normalization = _make_e8p_raw_codebook().square().mean().sqrt()
        self.register_buffer("e8p_sources", source_table / normalization)
        self.register_buffer("e8p_shift", (0.25 / normalization).clone().detach())
        doubled_sources = torch.cat(
            (
                torch.tensor(
                    [values for values in product((1, 3, 5), repeat=8) if sum((v * v for v in values)) <= 40]
                ),
                torch.tensor(_E8P_PADDING_DOUBLED),
            )
        )
        self.register_buffer("e8p_source_parity", (doubled_sources.sum(dim=-1) // 2 & 1).long())
        initial_generators = self._initialize_generators(vectors)
        self.generators = nn.Parameter(initial_generators.float(), requires_grad=True)
        self.register_buffer("initial_generators", self.generators.detach().clone())
        initial_offsets = self._initialize_offsets(vectors)
        self.offsets = nn.Parameter(initial_offsets.float())
        self.register_buffer("initial_offsets", self.offsets.detach().clone())
        self.register_buffer("codes", self.encode_vectors(vectors).to(torch.uint8))

    @property
    def num_groups(self) -> int:
        return self.in_features // self.config.group_size

    @property
    def vectors_per_group(self) -> int:
        return self.out_features * self.config.group_size // self.config.dimension

    @property
    def logical_bits_per_weight(self) -> int:
        return self.config.bits

    @property
    def code_width(self) -> int:
        return 2

    def storage_nbits(self, generator_nbits: Optional[int] = None) -> int:
        """Return deployable storage, excluding training-only initialization state.

        Codes are counted at their logical packed width. Generators default to
        their current tensor dtype, but callers may request (for example) FP16
        generator accounting for a serialized inference format.
        """
        if generator_nbits is None:
            generator_nbits = self.generators.element_size() * 8
        if generator_nbits <= 0:
            raise ValueError("generator_nbits must be positive")
        scale_nbits = self.log_scales.element_size() * 8 if self.config.uses_rms_scale else 0
        return (
            self.codes.numel() * 8
            + self.generators.numel() * generator_nbits
            + self.offsets.numel() * self.offsets.element_size() * 8
            + self.log_scales.numel() * scale_nbits
            + self.hadamard_signs.numel()
            + self.hadamard_output_signs.numel()
            + (
                self.companding_logits.numel() * self.companding_logits.element_size() * 8
                if self.companding_logits is not None
                else 0
            )
        )

    @staticmethod
    def _normalized_fwht(values: torch.Tensor) -> torch.Tensor:
        width = values.shape[-1]
        prefix = values.shape[:-1]
        result = values
        butterfly = 1
        while butterfly < width:
            shape = (*prefix, width // (2 * butterfly), 2, butterfly)
            pairs = result.reshape(shape)
            (left, right) = pairs.unbind(dim=-2)
            result = torch.cat((left + right, left - right), dim=-1).reshape(*prefix, width)
            butterfly *= 2
        return result / math.sqrt(width)

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and value & value - 1 == 0

    @classmethod
    def _validate_hadamard_width(cls, width: int) -> None:
        if not cls._is_power_of_two(width):
            raise ValueError("Hadamard block width must be a power of two")

    @staticmethod
    def _decompose_hadamard_blocks(length: int, preferred: int) -> tuple[int, ...]:
        """Split an axis into preferred-size blocks and power-of-two tail blocks."""
        if not preferred:
            return ()
        if length <= preferred:
            return (length,)
        (full, remainder) = divmod(length, preferred)
        blocks = [preferred] * full
        while remainder:
            block = 1 << remainder.bit_length() - 1
            blocks.append(block)
            remainder -= block
        return tuple(blocks)

    @staticmethod
    def _random_hadamard_signs(block_sizes: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        chunks = [
            torch.randint(0, 2, (block,), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
            for block in block_sizes
        ]
        if len(set(block_sizes)) == 1:
            return torch.stack(chunks)
        return torch.cat(chunks)

    @staticmethod
    def _split_hadamard_signs(signs: torch.Tensor, block_sizes: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        if signs.ndim == 2:
            return tuple(signs.unbind(0))
        return tuple(torch.split(signs, block_sizes, dim=0))

    def _rotate_last_axis(
        self,
        values: torch.Tensor,
        block_sizes: tuple[int, ...],
        signs: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        if not block_sizes:
            return values
        sign_chunks = self._split_hadamard_signs(signs, block_sizes)
        transformed = []
        for chunk, block_signs in zip(torch.split(values, block_sizes, dim=-1), sign_chunks):
            block_signs = block_signs.to(device=values.device, dtype=values.dtype)
            if inverse:
                chunk = self._normalized_generalized_hadamard(chunk, transpose=True) * block_signs
            else:
                chunk = self._normalized_generalized_hadamard(chunk * block_signs)
            transformed.append(chunk)
        return torch.cat(transformed, dim=-1)

    @classmethod
    def _normalized_generalized_hadamard(
        cls, values: torch.Tensor, *, transpose: bool = False
    ) -> torch.Tensor:
        return cls._normalized_fwht(values)

    def rotate_features(self, values: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        """Apply xR or xR^T for block-diagonal R = D H."""
        return self._rotate_last_axis(values, self.hadamard_block_sizes, self.hadamard_signs, inverse=inverse)

    def rotate_outputs(self, values: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        """Apply U or U^T on the output axis using a randomized Hadamard U."""
        block = self.hadamard_output_block_size
        if not block:
            return values
        transposed = values.T
        transformed = self._rotate_last_axis(
            transposed, self.hadamard_output_block_sizes, self.hadamard_output_signs, inverse=inverse
        )
        return transformed.T

    def rotate_xtx(self, xtx: torch.Tensor) -> torch.Tensor:
        """Return R^T XTX R without materializing the orthogonal matrix R."""
        if not self.hadamard_block_size:
            return xtx
        right_rotated = self.rotate_features(xtx)
        return self.rotate_features(right_rotated.T).T

    def estimate_nbits_per_parameter(self, generator_nbits: Optional[int] = None) -> float:
        return self.storage_nbits(generator_nbits) / (self.out_features * self.in_features)

    def scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    @torch.no_grad()
    def _initialize_companding_mus(self, weight: torch.Tensor) -> torch.Tensor:
        count = self.num_groups
        return torch.full((count,), self.config.companding_mu, device=weight.device, dtype=weight.dtype)

    def companding_mus(self) -> torch.Tensor:
        if self.companding_logits is not None:
            span = self.config.companding_mu_max - self.config.companding_mu_min
            return self.config.companding_mu_min + span * torch.sigmoid(self.companding_logits)
        return torch.tensor(
            self.config.companding_mu, device=self.log_scales.device, dtype=self.log_scales.dtype
        )

    def _broadcast_mus(self, values: torch.Tensor, group_dim: Optional[int]) -> torch.Tensor:
        mus = self.companding_mus().to(device=values.device, dtype=values.dtype)
        if mus.ndim == 0 or mus.numel() == 1 or group_dim is None:
            return mus
        shape = [1] * values.ndim
        shape[group_dim] = self.num_groups
        return mus.reshape(shape)

    def _compand(self, values: torch.Tensor, *, group_dim: Optional[int] = None) -> torch.Tensor:
        mus = self._broadcast_mus(values, group_dim)
        if not self.config.companding_enabled:
            return values
        clip = float(self.config.companding_clip)
        if clip and self.config.companding_soft_tail:
            magnitude = values.abs() / clip
            core_input = magnitude.clamp_max(1)
            core = torch.log1p(mus * core_input) / torch.log1p(mus)
            slope = mus / ((1 + mus) * torch.log1p(mus))
            transformed = torch.where(magnitude <= 1, core, 1 + slope * (magnitude - 1))
            return values.sign() * transformed * clip
        normalized = (values / clip).clamp(-1, 1) if clip else values
        transformed = normalized.sign() * torch.log1p(mus * normalized.abs()) / torch.log1p(mus)
        return transformed * clip if clip else transformed

    def _decompand(self, values: torch.Tensor, *, group_dim: Optional[int] = None) -> torch.Tensor:
        mus = self._broadcast_mus(values, group_dim)
        if not self.config.companding_enabled:
            return values
        clip = float(self.config.companding_clip)
        if clip and self.config.companding_soft_tail:
            magnitude = values.abs() / clip
            core_input = magnitude.clamp_max(1)
            core = torch.expm1(core_input * torch.log1p(mus)) / mus
            slope = mus / ((1 + mus) * torch.log1p(mus))
            restored = torch.where(magnitude <= 1, core, 1 + (magnitude - 1) / slope)
            return values.sign() * restored * clip
        normalized = (values / clip).clamp(-1, 1) if clip else values
        restored = normalized.sign() * torch.expm1(normalized.abs() * torch.log1p(mus)) / mus
        return restored * clip if clip else restored

    def compand_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply the configured group-wise mu-law transform to a weight."""
        grouped = self._weight_to_groups(weight)
        if self.config.uses_rms_scale:
            grouped = grouped / self.scales().detach()
        grouped = self._compand(grouped, group_dim=1)
        if self.config.uses_rms_scale:
            grouped = grouped * self.scales().detach()
        return grouped.reshape_as(weight)

    def _weight_to_groups(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.reshape(self.out_features, self.num_groups, self.config.group_size)

    def _weight_to_vectors(self, weight: torch.Tensor) -> torch.Tensor:
        grouped = self._weight_to_groups(weight)
        if self.config.uses_rms_scale:
            grouped = grouped / self.scales()
        grouped = grouped.permute(1, 0, 2).contiguous()
        return grouped.reshape(self.num_groups, self.vectors_per_group, self.config.dimension)

    def _vectors_to_weight(self, vectors: torch.Tensor) -> torch.Tensor:
        grouped = vectors.reshape(self.num_groups, self.out_features, self.config.group_size)
        grouped = self._decompand(grouped, group_dim=0)
        grouped = grouped.permute(1, 0, 2).contiguous()
        if self.config.uses_rms_scale:
            grouped = grouped * self.scales()
        return grouped.reshape(self.out_features, self.in_features)

    @torch.no_grad()
    def _initialize_generators(self, vectors: torch.Tensor) -> torch.Tensor:
        return (
            torch.eye(8, device=vectors.device, dtype=vectors.dtype).expand(self.num_groups, -1, -1).clone()
        )

    @torch.no_grad()
    def _initialize_offsets(self, vectors: torch.Tensor) -> torch.Tensor:
        """Center E8P coordinates on the empirical vector mean."""
        vector_mean = vectors.mean(dim=1)
        mean_coordinates = torch.linalg.solve(
            self.generators.detach().float(), vector_mean.float().unsqueeze(-1)
        ).squeeze(-1)
        return mean_coordinates

    def center_codes(self) -> torch.Tensor:
        return torch.zeros_like(self.codes)

    @torch.no_grad()
    def encode_vectors(self, vectors: torch.Tensor, xtx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if xtx is not None:
            reference_weight = self._vectors_to_weight(vectors)
            return self._encode_weight_xtx_feedback(reference_weight, xtx)
        encoded = []
        identity = torch.eye(8, device=vectors.device, dtype=torch.float32)
        for group_index in range(self.num_groups):
            encoded.append(self._encode_e8p_block(vectors[group_index], group_index, identity))
        return torch.stack(encoded)

    @torch.no_grad()
    def _encode_block_babai(
        self,
        vectors: torch.Tensor,
        group_index: int,
        metric: torch.Tensor,
        *,
        apply_local_search: bool = True,
    ) -> torch.Tensor:
        """Assign one block under a positive-definite local metric."""
        return self._encode_e8p_block(vectors, group_index, metric)

    def _group_mu(self, group_index: int, values: torch.Tensor) -> torch.Tensor:
        mus = self.companding_mus().to(device=values.device, dtype=values.dtype)
        return mus if mus.ndim == 0 else mus[0] if mus.numel() == 1 else mus[group_index]

    def _transform_block(self, values: torch.Tensor, group_index: int, *, inverse: bool) -> torch.Tensor:
        if not self.config.companding_enabled:
            return values
        mu = self._group_mu(group_index, values)
        clip = float(self.config.companding_clip)
        if clip and self.config.companding_soft_tail:
            magnitude = values.abs() / clip
            slope = mu / ((1 + mu) * torch.log1p(mu))
            if inverse:
                core = torch.expm1(magnitude.clamp_max(1) * torch.log1p(mu)) / mu
                transformed = torch.where(magnitude <= 1, core, 1 + (magnitude - 1) / slope)
            else:
                core = torch.log1p(mu * magnitude.clamp_max(1)) / torch.log1p(mu)
                transformed = torch.where(magnitude <= 1, core, 1 + slope * (magnitude - 1))
            return values.sign() * transformed * clip
        normalized = (values / clip).clamp(-1, 1) if clip else values
        if inverse:
            transformed = normalized.sign() * torch.expm1(normalized.abs() * torch.log1p(mu)) / mu
        else:
            transformed = normalized.sign() * torch.log1p(mu * normalized.abs()) / torch.log1p(mu)
        return transformed * clip if clip else transformed

    @torch.no_grad()
    def _prepare_hessian_inverse_factor(self, xtx: torch.Tensor) -> torch.Tensor:
        hessian = (xtx.detach().float() + xtx.detach().float().T) * 0.5
        diagonal = torch.diagonal(hessian)
        damp = self.config.hessian_damp * diagonal.abs().mean().clamp_min(self.config.covariance_eps)
        hessian.diagonal().add_(damp)
        chol = torch.linalg.cholesky(hessian)
        hessian_inverse = torch.cholesky_inverse(chol)
        return torch.linalg.cholesky(hessian_inverse, upper=True)

    def _decode_e8p_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[-1] != 2:
            raise ValueError("E8P codes must contain exactly two bytes per vector")
        source_ids = codes[..., 0].long()
        suffix = codes[..., 1].long()
        sign_bits = suffix & 127
        bit_positions = torch.arange(7, device=codes.device)
        first_negative = (sign_bits[..., None] >> bit_positions & 1).bool()
        first_signs = torch.where(first_negative, -1.0, 1.0)
        sources = self.e8p_sources.to(device=codes.device)[source_ids]
        source_parity = self.e8p_source_parity.to(codes.device)[source_ids]
        first_parity = first_negative.long().sum(dim=-1) & 1
        last_negative = (first_parity ^ source_parity).bool()
        last_sign = torch.where(last_negative, -1.0, 1.0)[..., None]
        signs = torch.cat((first_signs, last_sign), dim=-1).to(sources.dtype)
        shift = torch.where((suffix >> 7).bool(), self.e8p_shift, -self.e8p_shift)
        return sources * signs + shift[..., None]

    @torch.no_grad()
    def _encode_e8p_block(
        self, vectors: torch.Tensor, group_index: int, metric: torch.Tensor
    ) -> torch.Tensor:
        """Top-K structured E8P assignment followed by exact metric reranking."""
        generator = self.generators.detach().float()[group_index]
        offset = self.offsets.detach().float()[group_index]
        latent = torch.linalg.solve(generator, vectors.detach().float().T).T - offset
        sources = self.e8p_sources.to(device=vectors.device, dtype=torch.float32)
        source_parity = self.e8p_source_parity.to(vectors.device)
        source_ids = torch.arange(256, device=vectors.device, dtype=torch.long)
        metric = (metric.detach().float() + metric.detach().float().T) * 0.5
        result = torch.empty(vectors.shape[0], 2, device=vectors.device, dtype=torch.uint8)
        chunk_size = self.config.e8p_encode_chunk_size
        topk = min(self.config.e8p_candidate_count, 512)
        for chunk_start in range(0, vectors.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, vectors.shape[0])
            chunk = latent[chunk_start:chunk_end]
            candidate_losses = []
            candidate_codes = []
            for shift_bit, shift in enumerate((-self.e8p_shift, self.e8p_shift)):
                adjusted = chunk - shift
                negative = adjusted < 0
                base_signs = torch.where(negative, -1.0, 1.0)
                parity = negative.long().sum(dim=-1) & 1
                mismatch = parity[:, None] != source_parity[None, :]
                flip_cost = 4 * adjusted.abs()[:, None, :] * sources[None, :, :]
                flip_index = flip_cost.argmin(dim=-1)
                signs = base_signs[:, None, :].expand(-1, 256, -1).clone()
                selected = signs.gather(-1, flip_index[..., None])
                signs.scatter_(
                    -1, flip_index[..., None], torch.where(mismatch[..., None], -selected, selected)
                )
                points = sources[None, :, :] * signs + shift
                losses = (points - chunk[:, None, :]).square().sum(dim=-1)
                sign_bits = torch.zeros(chunk.shape[0], 256, device=vectors.device, dtype=torch.long)
                for bit in range(7):
                    sign_bits |= (signs[..., bit] < 0).long() << bit
                suffix = sign_bits | shift_bit << 7
                codes = torch.stack((source_ids.expand(chunk.shape[0], -1), suffix), dim=-1)
                candidate_losses.append(losses)
                candidate_codes.append(codes)
            all_losses = torch.cat(candidate_losses, dim=-1)
            all_codes = torch.cat(candidate_codes, dim=1)
            (_, indices) = torch.topk(all_losses, k=topk, dim=-1, largest=False, sorted=False)
            shortlisted = all_codes.gather(1, indices[..., None].expand(-1, -1, 2)).to(torch.uint8)
            points = self._decode_e8p_codes(shortlisted)
            reconstructed = torch.einsum("ij,nkj->nki", generator, points + offset)
            residual = vectors[chunk_start:chunk_end].float()[:, None, :] - reconstructed
            weighted = torch.einsum("nki,ij,nkj->nk", residual, metric, residual)
            best = weighted.argmin(dim=-1)
            result[chunk_start:chunk_end] = shortlisted[
                torch.arange(shortlisted.shape[0], device=vectors.device), best
            ]
        return result

    @torch.no_grad()
    def _decode_block(self, codes: torch.Tensor, group_index: int) -> torch.Tensor:
        coordinates = self._decode_e8p_codes(codes) + self.offsets.detach().float()[group_index]
        return coordinates @ self.generators.detach().float()[group_index].T

    @torch.no_grad()
    def _encode_weight_xtx_feedback(
        self,
        reference_weight: torch.Tensor,
        xtx: torch.Tensor,
        hessian_inverse_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Block GPTQ assignment with lattice nearest points and error feedback.

        Local QR-Babai alone discards all off-diagonal Hessian blocks.  This
        routine quantizes each lattice vector conditionally and propagates its
        error into later columns using an upper Cholesky factor of ``H^-1``.
        """
        if xtx.shape != (self.in_features, self.in_features):
            raise ValueError("xtx shape does not match module input features")
        if self.config.local_search_sweeps:
            zeros = lambda: torch.zeros(
                self.config.local_search_sweeps, device=reference_weight.device, dtype=torch.float64
            )
            self._active_search_diagnostics = {
                "changed_values": zeros(),
                "values_seen": zeros(),
                "loss_before": zeros(),
                "loss_after": zeros(),
            }
        if hessian_inverse_factor is None:
            hessian_inverse_factor = self._prepare_hessian_inverse_factor(xtx)
        inverse_factor = hessian_inverse_factor.float()
        working = reference_weight.detach().float().clone()
        chunks = self.config.group_size // self.config.dimension
        structured_codes = torch.empty(
            self.num_groups,
            self.out_features,
            chunks,
            self.code_width,
            device=working.device,
            dtype=torch.float32,
        )
        scales = self.scales().detach().float()
        eye = torch.eye(self.config.dimension, device=working.device, dtype=torch.float32)
        for start in range(0, self.in_features, self.config.dimension):
            end = start + self.config.dimension
            group_index = start // self.config.group_size
            chunk_index = start % self.config.group_size // self.config.dimension
            upper_block = inverse_factor[start:end, start:end]
            inverse_upper = torch.linalg.solve_triangular(upper_block, eye, upper=True)
            conditional_metric = inverse_upper @ inverse_upper.T
            normalized_target = working[:, start:end] / scales[:, group_index]
            encoded_target = self._transform_block(normalized_target, group_index, inverse=False)
            block_codes = self._encode_block_babai(
                encoded_target,
                group_index,
                conditional_metric,
                apply_local_search=not self.config.companding_enabled,
            )
            structured_codes[group_index, :, chunk_index] = block_codes
            quantized_block = (
                self._transform_block(self._decode_block(block_codes, group_index), group_index, inverse=True)
                * scales[:, group_index]
            )
            block_error = working[:, start:end] - quantized_block
            normalized_error = torch.linalg.solve_triangular(upper_block.T, block_error.T, upper=False).T
            working[:, start:] -= normalized_error @ inverse_factor[start:end, start:]
        if self.config.local_search_sweeps:
            diagnostics = self._active_search_diagnostics
            changed = diagnostics["changed_values"]
            seen = diagnostics["values_seen"]
            before = diagnostics["loss_before"]
            after = diagnostics["loss_after"]
            self.last_search_diagnostics = {
                "changed_fraction": [
                    float(changed[index] / seen[index]) if seen[index] else 0.0
                    for index in range(self.config.local_search_sweeps)
                ],
                "metric_loss_reduction": [
                    float((before[index] - after[index]) / before[index]) if before[index] else 0.0
                    for index in range(self.config.local_search_sweeps)
                ],
            }
            del self._active_search_diagnostics
        else:
            self.last_search_diagnostics = {"changed_fraction": [], "metric_loss_reduction": []}
        return structured_codes.reshape(self.num_groups, self.vectors_per_group, self.code_width)

    @torch.no_grad()
    def propose_codes(
        self,
        reference_weight: torch.Tensor,
        xtx: Optional[torch.Tensor] = None,
        hessian_inverse_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if xtx is None:
            codes = self.encode_vectors(self._weight_to_vectors(self.compand_weight(reference_weight)))
        else:
            codes = self._encode_weight_xtx_feedback(reference_weight, xtx, hessian_inverse_factor)
        return codes.to(torch.uint8)

    @torch.no_grad()
    def refresh_codes(self, reference_weight: torch.Tensor, xtx: Optional[torch.Tensor] = None) -> None:
        self.codes.copy_(self.propose_codes(reference_weight, xtx=xtx))

    def decode_vectors(self, codes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if codes is None:
            codes = self.codes
        coordinates = self._decode_e8p_codes(codes).to(self.generators.dtype) + self.offsets[:, None, :].to(
            self.generators.dtype
        )
        return torch.einsum("gij,gvj->gvi", self.generators, coordinates)

    def forward(self) -> torch.Tensor:
        lattice_weight = self._vectors_to_weight(self.decode_vectors())
        lattice_weight = self.rotate_outputs(lattice_weight, inverse=True)
        return self.rotate_features(lattice_weight, inverse=True)

    @torch.no_grad()
    def optimize_scales_(self, reference_weight: torch.Tensor, xtx: torch.Tensor) -> None:
        """Solve the optimal positive row/group scales for fixed lattice points."""
        if self.config.companding_enabled:
            return
        decoded_groups = (
            self.decode_vectors()
            .reshape(self.num_groups, self.out_features, self.config.group_size)
            .permute(1, 0, 2)
            .float()
        )
        reference_groups = self._weight_to_groups(reference_weight.detach().float())
        hessian_blocks = (
            xtx.detach()
            .float()
            .reshape(self.num_groups, self.config.group_size, self.num_groups, self.config.group_size)
        )
        gram = torch.einsum("ogi,gihj,ohj->ogh", decoded_groups, hessian_blocks, decoded_groups)
        rhs = torch.einsum("ogi,gihj,ohj->og", decoded_groups, hessian_blocks, reference_groups)
        gram = (gram + gram.transpose(-1, -2)) * 0.5
        eye = torch.eye(self.num_groups, device=gram.device, dtype=gram.dtype)
        ridge = self.config.covariance_eps * torch.diagonal(gram, dim1=-2, dim2=-1).abs().mean(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        solved = torch.linalg.solve(gram + ridge.unsqueeze(-1) * eye, rhs.unsqueeze(-1)).squeeze(-1)
        initial_scales = self.initial_log_scales.squeeze(-1).exp()
        lower = initial_scales * self.config.scale_min_factor
        upper = initial_scales * self.config.scale_max_factor
        solved = torch.minimum(torch.maximum(solved, lower), upper).clamp_min(self.config.scale_eps)
        self.log_scales.copy_(solved.unsqueeze(-1).log())

    def packed_codes(self) -> torch.Tensor:
        return self.codes

    @torch.no_grad()
    def project_generators_(self) -> None:
        self.generators.copy_(self._clamp_singular_values(self.generators.float()).to(self.generators.dtype))

    @torch.no_grad()
    def project_parameters_(self) -> None:
        self.project_generators_()
        lower = self.initial_log_scales + torch.log(
            torch.tensor(self.config.scale_min_factor, device=self.log_scales.device)
        )
        upper = self.initial_log_scales + torch.log(
            torch.tensor(self.config.scale_max_factor, device=self.log_scales.device)
        )
        self.log_scales.clamp_(lower, upper)
        self.offsets.clamp_(
            self.initial_offsets - self.config.offset_max_delta,
            self.initial_offsets + self.config.offset_max_delta,
        )

    def _clamp_singular_values(self, generators: torch.Tensor) -> torch.Tensor:
        (u, singular_values, vh) = torch.linalg.svd(generators, full_matrices=False)
        singular_values = singular_values.clamp(self.config.sigma_min, self.config.sigma_max)
        return u * singular_values.unsqueeze(-2) @ vh


class GLVQLinear(nn.Module):
    """Reference PyTorch Linear layer backed by :class:`GLVQWeight`."""

    def __init__(self, quantized_weight: GLVQWeight, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.quantized_weight = quantized_weight
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: Optional[GLVQConfig] = None) -> "GLVQLinear":
        return cls(GLVQWeight(linear.weight.detach(), config), linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.quantized_weight().to(inputs.dtype)
        bias = self.bias.to(inputs.dtype) if self.bias is not None else None
        return F.linear(inputs, weight, bias)


def fit_glvq_weight(
    module: GLVQWeight,
    reference_weight: torch.Tensor,
    calibration_inputs: Optional[torch.Tensor] = None,
    *,
    outer_steps: int = 20,
    generator_steps: int = 10,
    learning_rate: float = 0.001,
    regularization: float = 0.001,
    generator_learning_rate: Optional[float] = None,
    offset_learning_rate: Optional[float] = None,
    generator_regularization: Optional[float] = None,
    offset_regularization: Optional[float] = None,
    solver: str = "adam",
    continuous_update_tolerance: float = 0.0,
    relative_tolerance: float = 0.0001,
    patience: int = 3,
) -> list[float]:
    """Alternate bounded coordinate assignment and generator optimization.

    When calibration inputs are supplied, the objective is layer-output MSE.
    Inputs may have any leading dimensions as long as the final dimension is
    ``in_features``.  Otherwise the function minimizes weight MSE.
    """
    if reference_weight.shape != (module.out_features, module.in_features):
        raise ValueError("reference_weight shape does not match module")
    if calibration_inputs is not None and calibration_inputs.shape[-1] != module.in_features:
        raise ValueError("the last calibration_inputs dimension must equal in_features")
    reference_weight = reference_weight.detach().to(module.generators.device, module.generators.dtype)
    if calibration_inputs is not None:
        calibration_inputs = calibration_inputs.detach().to(module.generators.device, module.generators.dtype)
        flat_inputs = calibration_inputs.reshape(-1, module.in_features)
        xtx = flat_inputs.transpose(0, 1) @ flat_inputs / max(flat_inputs.shape[0], 1)
    else:
        xtx = torch.eye(module.in_features, device=reference_weight.device, dtype=reference_weight.dtype)
    return fit_glvq_weight_xtx(
        module,
        reference_weight,
        xtx,
        outer_steps=outer_steps,
        generator_steps=generator_steps,
        learning_rate=learning_rate,
        regularization=regularization,
        generator_learning_rate=generator_learning_rate,
        offset_learning_rate=offset_learning_rate,
        generator_regularization=generator_regularization,
        offset_regularization=offset_regularization,
        solver="adam",
        continuous_update_tolerance=continuous_update_tolerance,
        relative_tolerance=relative_tolerance,
        patience=patience,
    )


def fit_glvq_weight_xtx(
    module: GLVQWeight,
    reference_weight: torch.Tensor,
    xtx: torch.Tensor,
    *,
    outer_steps: int = 20,
    generator_steps: int = 10,
    learning_rate: float = 0.001,
    regularization: float = 0.001,
    generator_learning_rate: Optional[float] = None,
    offset_learning_rate: Optional[float] = None,
    generator_regularization: Optional[float] = None,
    offset_regularization: Optional[float] = None,
    solver: str = "adam",
    continuous_update_tolerance: float = 0.0,
    relative_tolerance: float = 0.0001,
    patience: int = 3,
    min_learning_rate: float = 1e-06,
    code_update_tolerance: float = 0.0,
    max_grad_norm: float = 1.0,
    companding_learning_rate: Optional[float] = None,
    companding_regularization: float = 1e-05,
) -> list[float]:
    """Fit a lattice using the layer-input second moment collected by AQLM.

    The reconstruction term is ``tr((W_hat-W) XTX (W_hat-W)^T)``.
    This is equivalent to activation-output MSE up to a constant normalization,
    without retaining all calibration activations.
    """
    if solver != "adam":
        raise ValueError("This release supports Adam only")
    if reference_weight.shape != (module.out_features, module.in_features):
        raise ValueError("reference_weight shape does not match module")
    if xtx.shape != (module.in_features, module.in_features):
        raise ValueError("xtx shape does not match module input features")
    dtype = module.generators.dtype
    device = module.generators.device
    reference_weight = reference_weight.detach().to(device=device, dtype=dtype)
    xtx = xtx.detach().to(device=device, dtype=dtype)
    reference_weight = module.rotate_features(reference_weight)
    reference_weight = module.rotate_outputs(reference_weight)
    xtx = module.rotate_xtx(xtx)
    if outer_steps <= 0 or generator_steps <= 0:
        raise ValueError("outer_steps and generator_steps must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if continuous_update_tolerance < 0:
        raise ValueError("continuous_update_tolerance must be non-negative")
    generator_learning_rate = learning_rate if generator_learning_rate is None else generator_learning_rate
    offset_learning_rate = learning_rate if offset_learning_rate is None else offset_learning_rate
    generator_regularization = (
        regularization if generator_regularization is None else generator_regularization
    )
    offset_regularization = regularization if offset_regularization is None else offset_regularization
    if min(generator_learning_rate, offset_learning_rate) <= 0:
        raise ValueError("generator and offset learning rates must be positive")
    if min(generator_regularization, offset_regularization) < 0:
        raise ValueError("generator and offset regularization must be non-negative")
    current_generator_lr = generator_learning_rate
    current_offset_lr = offset_learning_rate
    current_companding_lr = (
        offset_learning_rate if companding_learning_rate is None else companding_learning_rate
    )
    continuous_parameters = [
        parameter for parameter in (module.generators, module.offsets) if parameter.requires_grad
    ]
    if module.log_scales.requires_grad:
        continuous_parameters.append(module.log_scales)
    if module.companding_logits is not None:
        continuous_parameters.append(module.companding_logits)

    def make_adam() -> torch.optim.Adam:
        parameter_groups = []
        if module.generators.requires_grad:
            parameter_groups.append({"params": [module.generators], "lr": current_generator_lr})
        if module.offsets.requires_grad:
            parameter_groups.append({"params": [module.offsets], "lr": current_offset_lr})
        if module.log_scales.requires_grad:
            parameter_groups.append({"params": [module.log_scales], "lr": current_offset_lr})
        if module.companding_logits is not None:
            parameter_groups.append({"params": [module.companding_logits], "lr": current_companding_lr})
        return torch.optim.Adam(parameter_groups)

    optimizer: Optional[torch.optim.Optimizer] = make_adam()
    signal = (reference_weight @ xtx * reference_weight).sum().clamp_min(torch.finfo(dtype).eps)
    hessian_inverse_factor = None
    hessian_inverse_factor = module._prepare_hessian_inverse_factor(xtx)

    def reconstruct(codes: Optional[torch.Tensor] = None) -> torch.Tensor:
        return module._vectors_to_weight(module.decode_vectors(codes))

    def reconstruction_loss(codes: Optional[torch.Tensor] = None) -> torch.Tensor:
        delta = reconstruct(codes) - reference_weight
        return (delta @ xtx * delta).sum() / signal

    def regularization_loss() -> torch.Tensor:
        generator_denominator = module.initial_generators.square().sum().clamp_min(torch.finfo(dtype).eps)
        generator_error = (
            module.generators - module.initial_generators
        ).square().sum() / generator_denominator
        offset_error = F.mse_loss(module.offsets, module.initial_offsets)
        loss = generator_regularization * generator_error + offset_regularization * offset_error
        if module.log_scales.requires_grad:
            loss = loss + offset_regularization * F.mse_loss(module.log_scales, module.initial_log_scales)
        if module.companding_logits is not None:
            loss = loss + companding_regularization * module.companding_mus().square().mean()
        return loss

    def snapshot() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            module.generators.detach().clone(),
            module.offsets.detach().clone(),
            module.log_scales.detach().clone(),
            module.codes.detach().clone(),
            (
                module.companding_logits.detach().clone()
                if module.companding_logits is not None
                else torch.empty(0, device=device)
            ),
        )

    @torch.no_grad()
    def restore(state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        module.generators.copy_(state[0])
        module.offsets.copy_(state[1])
        module.log_scales.copy_(state[2])
        module.codes.copy_(state[3])
        if module.companding_logits is not None:
            module.companding_logits.copy_(state[4])

    module.optimize_scales_(reference_weight, xtx)
    history: list[float] = []
    best_loss = float(reconstruction_loss().detach())
    best_state = snapshot()
    stale_steps = 0
    fit_diagnostics: dict[str, object] = {
        "solver": "adam",
        "continuous_loss_before": [],
        "continuous_loss_after": [],
        "continuous_accepted": [],
        "code_accepted": [],
    }
    for _ in range(outer_steps):
        continuous_state = snapshot()
        continuous_loss_before = float(reconstruction_loss().detach())
        if optimizer is None:
            optimizer = make_adam()
        for _ in range(generator_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss() + regularization_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(continuous_parameters, max_grad_norm)
            optimizer.step()
            module.project_parameters_()
        with torch.no_grad():
            module.optimize_scales_(reference_weight, xtx)
            continuous_loss_after = float(reconstruction_loss())
            continuous_accepted = math.isfinite(
                continuous_loss_after
            ) and continuous_loss_after <= continuous_loss_before * (1 + continuous_update_tolerance)
            if not continuous_accepted:
                restore(continuous_state)
                continuous_loss_after = continuous_loss_before
                optimizer = make_adam()
            fit_diagnostics["continuous_loss_before"].append(continuous_loss_before)
            fit_diagnostics["continuous_loss_after"].append(continuous_loss_after)
            fit_diagnostics["continuous_accepted"].append(continuous_accepted)
            current_loss = reconstruction_loss()
            proposed_codes = module.propose_codes(
                reference_weight, xtx=xtx, hessian_inverse_factor=hessian_inverse_factor
            )
            proposed_loss = reconstruction_loss(proposed_codes)
            code_accepted = bool(proposed_loss <= current_loss * (1 + code_update_tolerance))
            if code_accepted:
                module.codes.copy_(proposed_codes)
                module.optimize_scales_(reference_weight, xtx)
                current_loss = reconstruction_loss()
            fit_diagnostics["code_accepted"].append(code_accepted)
            current_value = float(current_loss)
            history.append(current_value)
            relative_improvement = (best_loss - current_value) / max(best_loss, 1e-12)
            if relative_improvement > relative_tolerance:
                best_loss = current_value
                best_state = snapshot()
                stale_steps = 0
            else:
                stale_steps += 1
                if current_value < best_loss:
                    best_loss = current_value
                    best_state = snapshot()
                if stale_steps >= patience:
                    restore(best_state)
                    current_generator_lr *= 0.5
                    current_offset_lr *= 0.5
                    current_companding_lr *= 0.5
                    if optimizer is not None:
                        optimizer = make_adam()
                    stale_steps = 0
                    if max(current_generator_lr, current_offset_lr) < min_learning_rate:
                        break
    restore(best_state)
    module.last_fit_diagnostics = fit_diagnostics
    return history
