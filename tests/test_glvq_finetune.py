from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.finetune import _compute_mse_on_batch, _make_finetune_parameter_groups, finetune_groupwise
from src.glvq import GLVQLinear


class ToyBlock(nn.Module):

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = GLVQLinear.from_linear(linear)

    def forward(self, inputs):
        return (self.linear(inputs),)


class AttentionMaskDtypeBlock(nn.Module):

    def forward(self, inputs, attention_mask):
        assert inputs.dtype == torch.float32
        assert attention_mask.dtype == inputs.dtype
        return (inputs,)


def test_block_finetuning_aligns_floating_attention_mask_dtype():
    inputs = torch.randn(1, 8, 16, dtype=torch.float16)
    targets = inputs.clone()
    attention_mask = torch.zeros(1, 1, 8, 8, dtype=torch.float16)
    loss = _compute_mse_on_batch(
        AttentionMaskDtypeBlock(), iter([(inputs, targets)]), attention_mask=attention_mask
    )
    assert loss.item() == 0.0
    assert attention_mask.dtype == torch.float16


def test_block_finetuning_updates_continuous_parameters_but_not_codes():
    torch.manual_seed(7)
    reference = nn.Linear(128, 16, bias=False)
    block = ToyBlock(reference)
    inputs = torch.randn(8, 3, 128)
    targets = reference(inputs).detach()
    initial_loss = F.mse_loss(block(inputs)[0], targets).item()
    initial_codes = block.linear.quantized_weight.codes.clone()
    args = Namespace(
        devices=[torch.device("cpu")],
        offload_activations=False,
        finetune_lr=0.001,
        finetune_adam_beta1=0.9,
        finetune_adam_beta2=0.95,
        finetune_batch_size=2,
        local_batch_size=2,
        finetune_max_epochs=3,
        finetune_early_stop=2,
    )
    finetune_groupwise(
        layer=block,
        train_inps=[inputs],
        train_outs=[targets],
        valid_inps=[inputs],
        valid_outs=[targets],
        args=args,
        verbose=False,
    )
    assert F.mse_loss(block(inputs)[0], targets).item() <= initial_loss
    assert torch.equal(block.linear.quantized_weight.codes, initial_codes)
