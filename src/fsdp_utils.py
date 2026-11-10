"""FSDP module-class discovery."""

from typing import Tuple
from torch import nn


def infer_module_classes(model: nn.Module, class_name: str) -> Tuple[type[nn.Module], ...]:
    """find transformer block classes that should be wrapped with inner FullyShardedDataParallel (auto_wrap_policy)"""
    found_module_types = []
    for module in model.modules():
        if module.__class__.__name__ == class_name:
            found_module_types.append(type(module))
    if not found_module_types:
        raise ValueError(f"Could not find {class_name} among submodules of {model}")
    found_module_types = tuple(found_module_types)
    assert any((isinstance(module, found_module_types) for module in model.modules()))
    return found_module_types
