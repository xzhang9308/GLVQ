import torch.nn as nn
import pytest

pytest.importorskip("transformers")
from main import _supported_forward_arg_names


class OPTLikeLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return hidden_states


class LlamaLikeLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        return hidden_states


class FlexibleLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


def test_forward_kwargs_are_filtered_for_opt():
    assert _supported_forward_arg_names(OPTLikeLayer(), ["attention_mask", "position_ids"]) == [
        "attention_mask"
    ]


def test_forward_kwargs_keep_supported_llama_arguments():
    assert _supported_forward_arg_names(LlamaLikeLayer(), ["attention_mask", "position_ids"]) == [
        "attention_mask",
        "position_ids",
    ]


def test_forward_kwargs_keep_all_for_flexible_layers():
    assert _supported_forward_arg_names(FlexibleLayer(), ["attention_mask", "position_ids"]) == [
        "attention_mask",
        "position_ids",
    ]
