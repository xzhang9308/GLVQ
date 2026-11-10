import math
import os
from contextlib import contextmanager
import torch
import torch.nn as nn
import transformers
from accelerate import dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_ERROR_MSG = "Unsupported model type {} - only llama, mistral and qwen2 are supported"
LLAMA_LIKE = ("llama", "mistral", "qwen2")


@contextmanager
def suspend_nn_inits():

    def skip(*args, **kwargs):
        pass

    saved_inits = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip
    try:
        yield
    finally:
        (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_) = saved_inits


def dispatch_quantized_model(model):
    num_devices = torch.cuda.device_count()
    device_map = {"model.embed_tokens": 0, "model.norm": num_devices - 1, "lm_head": 0}
    num_layers = len(get_layers(model))
    layers_per_device = math.ceil(num_layers / num_devices)
    for layer_id in range(num_layers):
        device_id = layer_id // layers_per_device
        device_map[f"model.layers.{layer_id}"] = device_id
    model = dispatch_model(model, device_map)
    model.model.embed_tokens = model.model.embed_tokens.to("cuda:0")
    model.lm_head = model.lm_head.to("cuda:0")
    return model


def get_model(
    model_path,
    load_quantized=None,
    dtype="auto",
    device_map=None,
    attn_implementation=None,
    trust_remote_code=False,
    use_safetensors=None,
):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code).torch_dtype or "auto"
        )
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    model_kwargs = {}
    if transformers.__version__ >= "4.38.0":
        model_kwargs["attn_implementation"] = attn_implementation
    with suspend_nn_inits():
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map=None if load_quantized else device_map,
            low_cpu_mem_usage=True,
            local_files_only=True,
            use_safetensors=use_safetensors,
            **model_kwargs,
        )
        if load_quantized:
            print("Initializing model with random weights...")
            print("Loading quantized model ...")
            model = load_quantized_model(model, load_quantized)
            if device_map == "auto":
                assert (
                    model.config.model_type in LLAMA_LIKE
                ), "Dispatching is implemented only for Llama-like models."
                model = dispatch_quantized_model(model)
        else:
            print("Loading pretrained model ...")
    print("Model loaded sucсessfully ...")
    return model


def is_model_for_causal_lm(model: nn.Module):
    assert isinstance(model, transformers.PreTrainedModel)
    assert len(model.base_model_prefix) > 0 and hasattr(model, model.base_model_prefix)
    assert model.get_output_embeddings() is not None
    return True


def get_model_head_with_norm(model):
    head = torch.nn.ModuleList()
    if model.config.model_type in LLAMA_LIKE:
        if model.model.norm is not None:
            head.append(model.model.norm)
        head.append(model.lm_head)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return head


def get_lm_logits(inps_, model):
    if model.config.model_type in LLAMA_LIKE:
        hidden_states = inps_.unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return lm_logits


def get_layers(model):
    if model.config.model_type in LLAMA_LIKE:
        return model.model.layers
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def find_sublayers(module, layers=(nn.Conv2d, nn.Linear)):
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layers):
            res[name] = layer
    return res


def get_sequential_groups(model):
    if model.config.model_type in LLAMA_LIKE:
        return [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def load_quantized_model(model, load_path):
    """Load quantized model"""
    layers = get_layers(model)
    for layer_index in range(len(layers)):
        layers[layer_index] = torch.load(
            os.path.join(load_path, str(layer_index) + ".pth"),
            map_location=next(layers[layer_index].parameters()).device,
            weights_only=False,
        )
    model.load_state_dict(torch.load(os.path.join(load_path, "not_quantized_weights.pt")), strict=False)
    return model


def save_not_quantized_weights(model: nn.Module, save_dir: str):
    already_saved_weights = set()
    for layer in get_layers(model):
        for param in layer.parameters():
            already_saved_weights.add(param)
    not_quantized_weights = {
        name: param for (name, param) in model.named_parameters() if param not in already_saved_weights
    }
    torch.save(not_quantized_weights, os.path.join(save_dir, "not_quantized_weights.pt"))


def save_quantized_model(model: transformers.PreTrainedModel, save_dir: str):
    """Save dequantized model state in the same format as returned by AQLM calibration (main.py)"""
    os.makedirs(save_dir, exist_ok=True)
    for layer_index, layer in enumerate(get_layers(model)):
        layer_save_path = os.path.join(save_dir, f"{layer_index}.pth")
        torch.save(layer, layer_save_path)
    save_not_quantized_weights(model, save_dir)
