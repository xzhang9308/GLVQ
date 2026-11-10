from argparse import Namespace

import torch

from aq_engine import AQEngine
from src.glvq import GLVQWeight


def test_aq_engine_reuses_xtx_for_glvq():
    torch.manual_seed(6)
    linear = torch.nn.Linear(128, 16, bias=False)
    engine = AQEngine(linear)
    engine.add_batch(torch.randn(2, 7, 128))
    args = Namespace(
        glvq_dimension=8,
        glvq_bits=2,
        glvq_group_size=128,
        glvq_covariance_eps=1e-6,
        glvq_init_clip=1.5,
        glvq_sigma_min=1e-5,
        glvq_sigma_max=5.0,
        glvq_outer_steps=2,
        glvq_generator_steps=2,
        glvq_learning_rate=2e-3,
        glvq_regularization=1e-3,
    )
    quantized = engine.quantize_glvq(args=args, verbose=False)
    assert isinstance(quantized, GLVQWeight)
    assert quantized().shape == linear.weight.shape
    assert engine.XTX is None
    assert quantized.estimate_nbits_per_parameter() > 2.0
