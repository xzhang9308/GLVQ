import inspect
import os
import time
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from tqdm.auto import trange
from transformers import PreTrainedModel
from aq_engine import AQEngine
from src.glvq import GLVQLinear, GLVQWeight
from src.datautils import get_loaders
from src.finetune import finetune_groupwise
from src.modelutils import (
    find_sublayers,
    get_layers,
    get_lm_logits,
    get_model,
    get_model_head_with_norm,
    get_sequential_groups,
    save_not_quantized_weights,
)
from src.utils import using_tf32

def quantize_model(model: PreTrainedModel, args: Namespace):
    """main entry point to functions for model quantization"""
    tick = time.time()
    print("Loading data ...")
    data = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.model_path,
        seqlen=args.model_seqlen,
        use_fast_tokenizer=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if args.val_size > 0:
        all_ids = torch.randperm(len(data))
        (train_ids, val_ids) = (all_ids[args.val_size :], all_ids[: args.val_size])
        train_data = [data[i] for i in train_ids]
        val_data = [data[i] for i in val_ids]
    else:
        train_data = data
        val_data = None
    results = quantize_aq(model, train_data, val_data, args)
    print(f"quantization time: {time.time() - tick:.1f}")
    return results


def _supported_forward_arg_names(layer: nn.Module, candidates: Sequence[str]) -> list[str]:
    """Return only kwargs accepted by a decoder layer's forward signature."""
    parameters = inspect.signature(layer.forward).parameters
    accepts_arbitrary_kwargs = any(
        (parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    )
    if accepts_arbitrary_kwargs:
        return list(candidates)
    return [name for name in candidates if name in parameters]


@torch.no_grad()
def get_inps(
    model: PreTrainedModel,
    data: Sequence,
    model_seqlen: int,
    devices: Sequence[torch.device],
    offload_activations: bool,
) -> Tuple[Sequence[torch.Tensor], Dict]:
    """
    mocks model launch to collect inputs to the first model layer
    :returns: a list of torch tensors with activations for each device in args.devices.
    Each tensor has shape [nsample_per_device, seq_len, hid_size]
    """
    print("catching layer inputs from data", flush=True)
    layers = get_layers(model)
    device = devices[0] if not offload_activations else torch.device("cpu")
    if isinstance(data, torch.Tensor) and data.shape[0] == 1:
        assert (
            data.ndim == 2
        ), "data must be either a single tensor with a long sequence or a list of pre-cut sequences"
        (num_sequences, num_tokens_dropped) = (data.numel() // model_seqlen, data.numel() % model_seqlen)
        data = [data[:, i * model_seqlen : (i + 1) * model_seqlen].to(device) for i in range(num_sequences)]
        print(f"Got {len(data)} sequences of {model_seqlen} tokens, dropped last {num_tokens_dropped} tokens")
        del num_sequences, num_tokens_dropped
    assert all((sequence.shape[1] == model_seqlen for sequence in data))
    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
    device = emb.weight.device
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)
    dtype = next(iter(model.parameters())).dtype
    nsamples_per_device = (len(data) - 1) // len(devices) + 1
    inps = [
        torch.zeros(
            (
                min(nsamples_per_device, len(data) - i * nsamples_per_device),
                model_seqlen,
                model.config.hidden_size,
            ),
            dtype=dtype,
            device=devices[i] if not offload_activations else "cpu",
            pin_memory=offload_activations,
        )
        for i in range(len(devices))
    ]
    forward_arg_candidates = ["attention_mask", "position_ids"]
    forward_arg_names = _supported_forward_arg_names(layers[0], forward_arg_candidates)
    cache = {"i": 0}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise CatcherExit()

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch_inps in data:
        try:
            if isinstance(batch_inps, (list, tuple)):
                (batch_inps, *_) = batch_inps
            batch_inps = batch_inps.to(device)
            model(batch_inps, attention_mask=torch.ones_like(batch_inps))
        except CatcherExit:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module
    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    torch.cuda.empty_cache()
    forward_args = {k: cache[k] for k in forward_arg_names}
    assert cache["i"] == sum(
        (len(inp_tensor) for inp_tensor in inps)
    ), "internal error: found empty rows in inps"
    return (inps, forward_args)


@torch.no_grad()
def quantize_aq(model: PreTrainedModel, data: Sequence, val_data: Optional[Sequence], args: Namespace):
    assert not torch.backends.cuda.matmul.allow_tf32
    print("\nStarting GLVQ quantization ...")
    (inps, forward_args) = get_inps(model, data, args.model_seqlen, args.devices, args.offload_activations)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
    if val_data:
        run_validation = True
        (val_inps, _) = get_inps(model, val_data, args.model_seqlen, args.devices, args.offload_activations)
        val_outs = [
            torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in val_inps
        ]
    else:
        run_validation = False
        (val_inps, val_outs) = (None, None)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    quantizers = {}
    overall_bits = 0
    number_of_quantized_params = 0
    layers = get_layers(model)
    selected_layers = set(args.quantize_layers) if args.quantize_layers is not None else None
    if selected_layers:
        invalid = sorted((index for index in selected_layers if index < 0 or index >= len(layers)))
        if invalid:
            raise ValueError(f"quantize_layers contains invalid indices: {invalid}")
        quantize_n_layers = max(selected_layers) + 1
    else:
        quantize_n_layers = (
            len(layers) if args.quantize_n_layers is None else min(args.quantize_n_layers, len(layers))
        )
    for layer_index in range(quantize_n_layers):
        print(f"\n---------------- Layer {layer_index} of {len(layers)} ----------------")
        stats_payload = {}
        start_time = time.time()
        layer_device_original = next(layers[layer_index].parameters()).device
        original_layer = None
        layer_dtype_original = next(layers[layer_index].parameters()).dtype
        print(f"layer_device_original={layer_device_original!r}")
        layer = layers[layer_index].to(args.devices[0])
        for k, v in forward_args.items():
            forward_args[k] = v.to(args.devices[0]) if isinstance(v, torch.Tensor) else v
        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(find_sublayers(layer).keys())]
        loaded_layer = False
        skip_selected_layer = selected_layers is not None and layer_index not in selected_layers
        glvq_block_train_outs = None
        independent_val_outs = None
        if args.resume:
            assert args.save is not None, "using --resume requires a --save path to resume from"
            layer_save_path = os.path.join(args.save, f"{layer_index}.pth")
            if os.path.exists(layer_save_path):
                print(f"Loading layer {layer_index} from {layer_save_path}")
                layer = torch.load(layer_save_path, map_location=args.devices[0], weights_only=False)
                loaded_layer = True
        if skip_selected_layer:
            print(f"Skipping quantization: layer {layer_index} is outside --quantize_layers")
            loaded_layer = True
        if run_validation and (not loaded_layer):
            if len(args.devices) == 1:
                assert len(val_inps) == len(val_outs) == 1
                update_outs(
                    layer, val_inps[0], val_outs[0], compute_mse=not args.skip_out_loss, **forward_args
                )
            else:
                update_outs_parallel(
                    args.devices,
                    layer,
                    val_inps,
                    val_outs,
                    compute_mse=not args.skip_out_loss,
                    **forward_args,
                )
        for sequential_index, names in enumerate(sequential):
            if loaded_layer:
                print("Skipping quantization: loaded a previously quantized layer")
                break
            store_group_outputs = not sequential_index > 0
            if len(args.devices) == 1:
                assert len(inps) == len(outs) == 1
                aq_handlers = init_aq_engines(
                    layer,
                    names,
                    inps[0],
                    outs[0],
                    store_outputs=store_group_outputs,
                    **forward_args,
                )
            else:
                aq_handlers = init_aq_engines_parallel(
                    args.devices,
                    layer,
                    names,
                    inps,
                    outs,
                    store_outputs=store_group_outputs,
                    **forward_args,
                )
            if sequential_index == 0:
                glvq_block_train_outs = outs
                print("Reusing teacher outputs as GLVQ block-finetuning targets")
            for sublayer_name in aq_handlers.keys():
                print(f"Quantizing module {sublayer_name} of layer {layer_index}")
                quantized_weight = aq_handlers[sublayer_name].quantize_glvq(args=args, verbose=True)
                with torch.no_grad():
                    assert aq_handlers[sublayer_name].layer.weight in set(layer.parameters())
                    new_linear = GLVQLinear(quantized_weight, aq_handlers[sublayer_name].layer.bias)
                    found_original = False
                    for submodule in layer.modules():
                        for child_name, child_module in submodule.named_children():
                            if child_module is aq_handlers[sublayer_name].layer:
                                setattr(submodule, child_name, new_linear)
                                found_original = True
                    assert found_original, f"could not find {sublayer_name}"
                weight_avg_bits = quantized_weight.estimate_nbits_per_parameter()
                overall_bits += int(
                    weight_avg_bits * torch.numel(aq_handlers[sublayer_name].layer.weight.data)
                )
                number_of_quantized_params += torch.numel(aq_handlers[sublayer_name].layer.weight.data)
                print("curent_avg_bits", overall_bits / number_of_quantized_params)
                quantizers["model.layers.%d.%s" % (layer_index, sublayer_name)] = ()
            del aq_handlers
            assert not loaded_layer
            should_finetune = sequential_index == len(sequential) - 1
            if args.finetune_max_epochs > 0 and should_finetune:
                print("PREPARING TO FINETUNE")
                print(layer)
                layer = layer.to(dtype=torch.float32)
                with using_tf32(enabled=True):
                    layer = finetune_groupwise(
                        layer=layer,
                        train_inps=inps,
                        train_outs=glvq_block_train_outs,
                        args=args,
                        valid_inps=val_inps,
                        valid_outs=val_outs,
                        **forward_args,
                    )
                layer = layer.to(dtype=layer_dtype_original)
                for module in layer.modules():
                    if isinstance(module, GLVQWeight):
                        module.float()
                        module.project_parameters_()
                print("FINISHED FINETUNING")
        if args.save and (not loaded_layer):
            os.makedirs(args.save, exist_ok=True)
            layer_save_path = os.path.join(args.save, f"{layer_index}.pth")
            print(f"Saving layer {layer_index}... to {layer_save_path}")
            torch.save(layer, layer_save_path)
        should_compute_mse = not (args.skip_out_loss or loaded_layer)
        if len(args.devices) == 1:
            assert len(inps) == len(outs) == 1
            out_losses = update_outs(layer, inps[0], outs[0], compute_mse=should_compute_mse, **forward_args)
        else:
            out_losses = update_outs_parallel(
                args.devices, layer, inps, outs, compute_mse=should_compute_mse, **forward_args
            )
        stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()
        if run_validation:
            if len(args.devices) == 1:
                assert len(val_inps) == len(val_outs) == 1
                out_val_losses = update_outs(
                    layer, val_inps[0], val_outs[0], compute_mse=should_compute_mse, **forward_args
                )
            else:
                out_val_losses = update_outs_parallel(
                    args.devices, layer, val_inps, val_outs, compute_mse=should_compute_mse, **forward_args
                )
            stats_payload["out_val_loss"] = torch.mean(torch.Tensor(out_val_losses)).item()
        layers[layer_index] = layer.to(layer_device_original)
        del layer
        torch.cuda.empty_cache()
        (inps, outs) = (outs, inps)
        if run_validation:
            (val_inps, val_outs) = (val_outs, val_inps)
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["Step"] = layer_index
        if not loaded_layer:
            print(stats_payload)
    print("=====================\nFinal stats:")
    glvq_modules = [module for module in model.modules() if isinstance(module, GLVQWeight)]
    storage_bits = sum((module.storage_nbits() for module in glvq_modules))
    weight_count = sum((module.out_features * module.in_features for module in glvq_modules))
    if weight_count:
        print(f"GLVQ deployable storage: {storage_bits} bits for {weight_count} weights")
        print(f"GLVQ actual average bpw (codes + FP32 generators): {storage_bits / weight_count:.6f}")
    if args.save:
        torch.save(vars(args), os.path.join(args.save, "args.pt"))
        save_not_quantized_weights(model, args.save)
    model.config.use_cache = use_cache
    print(f"quantize: torch.cuda.max_memory_allocated()={torch.cuda.max_memory_allocated():,}")
    return quantizers


@torch.no_grad()
def perplexity_eval(model: PreTrainedModel, testenc: torch.LongTensor, args: Namespace) -> float:
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")
    nsamples = testenc.numel() // args.model_seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False
    (inps, forward_args) = get_inps(model, testenc, args.model_seqlen, args.devices, args.offload_activations)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
    device = args.devices[0]
    for k, v in forward_args.items():
        forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    layers = get_layers(model)
    for i in trange(len(layers), desc="processing eval data by layer"):
        layer = layers[i].to(device)
        if len(args.devices) == 1:
            assert len(inps) == len(outs) == 1
            update_outs(layer, inps[0], outs[0], compute_mse=False, **forward_args)
        else:
            update_outs_parallel(args.devices, layer, inps, outs, compute_mse=False, **forward_args)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        (inps, outs) = (outs, inps)
    get_model_head_with_norm(model).to(device)
    testenc = testenc.to(device)
    nsamples_per_device = len(inps[0])
    assert len(set(map(len, inps[:-1]))) <= 1 and len(inps[-1]) <= len(inps[0])
    nlls = []
    for i in range(nsamples):
        inp = inps[i // nsamples_per_device][i % nsamples_per_device].to(args.devices[0], non_blocking=True)
        lm_logits = get_lm_logits(inp.to(device), model)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, i * args.model_seqlen : (i + 1) * args.model_seqlen][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * args.model_seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.model_seqlen)).item()
    print(f"\n{args.dataset_name} perplexity = {ppl:.4f}\n")
    get_model_head_with_norm(model).to(torch.device("cpu"))
    model.config.use_cache = use_cache
    return ppl


@torch.no_grad()
def init_aq_engines(
    layer: nn.Module,
    names: Sequence[str],
    inps_tensor: torch.Tensor,
    outs_tensor: torch.Tensor,
    store_outputs: bool = True,
    **forward_args: Dict[str, Any],
) -> Dict[str, AQEngine]:
    """
    Create a dictionary of AQUtil instances for each quantized layer;
    Run forward pass on each sample in inps_tensor and optionally write outputs to outs_tensor.
    Accumulate XTX to each one of aq_handlers
    :param layer: transformer layer with one or more linear layer to be quantized
    :param names: a list/tuple of string names for linear sub-layers inside :layer: that shall be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :param store_outputs: whether to retain layer outputs; XTX accumulation does not require them
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a dictionary where keys are full layer names and values are AQUtil instances ready to run .quantize
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    all_sublayers = find_sublayers(layer)
    subset = {name: all_sublayers[name] for name in names}
    assert len(subset) > 0
    aq_handlers = {}
    for sublayer_name in subset:
        aq_handlers[sublayer_name] = AQEngine(subset[sublayer_name])
    wrapped_layer_to_hander = {aq_handler.layer: aq_handler for aq_handler in aq_handlers.values()}
    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if child in wrapped_layer_to_hander:
                setattr(
                    module, child_name, _LayerWrapperThatAccumulatesXTX(child, wrapped_layer_to_hander[child])
                )
    for j in trange(len(inps_tensor), desc="calc outs before quantization", leave=False):
        layer_output = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0]
        if store_outputs:
            outs_tensor[j].copy_(layer_output.view_as(outs_tensor[j]), non_blocking=True)
    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, _LayerWrapperThatAccumulatesXTX):
                setattr(module, child_name, child.wrapped_layer)
    return aq_handlers


class _LayerWrapperThatAccumulatesXTX(nn.Module):

    def __init__(self, layer: nn.Module, aq_handler: AQEngine):
        super().__init__()
        (self.wrapped_layer, self.aq_handler) = (layer, aq_handler)

    def forward(self, input, *args, **kwargs):
        self.aq_handler.add_batch(input)
        return self.wrapped_layer(input, *args, **kwargs)


@torch.no_grad()
def init_aq_engines_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    names: Sequence[str],
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    store_outputs: bool = True,
    **forward_args,
):
    """Parallel version of init_aq_engines; works on lists of input/output tensors"""
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    layer_replicas[0] = layer
    funcs_by_device = [init_aq_engines for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], names, inps[i], outs[i], store_outputs))
        kwargs_by_device.append(
            {
                k: v.to(devices[i], non_blocking=True) if isinstance(v, torch.Tensor) else v
                for (k, v) in forward_args.items()
            }
        )
    aq_handles_by_device: Sequence[Dict[str, AQEngine]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    aq_handlers = aq_handles_by_device[0]
    for key, aq_handler in aq_handlers.items():
        replica_handlers = [device_aq_handlers[key] for device_aq_handlers in aq_handles_by_device]
        replica_nsamples = [replica_handler.nsamples for replica_handler in replica_handlers]
        total_nsamples = sum(replica_nsamples)
        aq_handler.XTX = sum(
            (
                (replica_handlers[i].XTX * (replica_nsamples[i] / total_nsamples)).to(
                    devices[0], non_blocking=True
                )
                for i in range(len(devices))
            )
        )
        aq_handler.nsamples = total_nsamples
    return aq_handlers


@torch.no_grad()
def update_outs(
    layer: nn.Module, inps_tensor: torch.Tensor, outs_tensor: torch.Tensor, compute_mse: bool, **forward_args
) -> Sequence[float]:
    """
    Update outs_tensor with new activations and optionally compute sample-wise mse loss with previous activations
    :param layer: transformer layer with one or more linear layer to be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :note: outs_tensor must contain previous activations with which to compute MSE loss
    :param compute_mse: if True, return a list of sample-wise mse losses; if False, return an empty sequence
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a list of mean squared errors for each sequence
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    out_losses = []
    for j in trange(len(inps_tensor), desc="calc outs after quantization", leave=False):
        outs_batch = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0]
        if compute_mse:
            batch_size = outs_batch.shape[0]
            outs_batch_loss = (
                (outs_batch - outs_tensor[j].to(device)).float().square().view(batch_size, -1).mean(dim=-1)
            )
            outs_batch_loss /= outs_batch.float().square().view(batch_size, -1).mean(dim=-1).clamp(min=1e-06)
            outs_batch_loss = outs_batch_loss.mean()
            out_losses.append(outs_batch_loss.item())
        outs_tensor[j].copy_(outs_batch.reshape_as(outs_tensor[j]), non_blocking=True)
    return out_losses


@torch.no_grad()
def update_outs_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    compute_mse: bool,
    **forward_args,
) -> Sequence[float]:
    """Parallel version of update_outs_and_compute_losses; works on lists of input/output tensors"""
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    funcs_by_device = [update_outs for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], inps[i], outs[i], compute_mse))
        kwargs_by_device.append(
            {
                k: v.to(devices[i], non_blocking=True) if isinstance(v, torch.Tensor) else v
                for (k, v) in forward_args.items()
            }
        )
    out_losses_by_device: Sequence[Sequence[float]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    return list(chain(*out_losses_by_device))


def main():
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "model_path", type=str, help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to frozen calibration token tensor.",
    )
    parser.add_argument("--quantizer", choices=["glvq"], default="glvq")
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        choices=["wikitext2", "c4"],
        default=None,
        help="Perplexity datasets to evaluate; defaults to WikiText2 and C4.",
    )
    parser.add_argument("--glvq_dimension", type=int, default=8)
    parser.add_argument("--glvq_bits", type=int, choices=[2], default=2)
    parser.add_argument("--glvq_codebook", choices=["e8p"], default="e8p")
    parser.add_argument("--glvq_deformation", choices=["full"], default="full")
    parser.add_argument("--glvq_e8p_candidate_count", type=int, default=8)
    parser.add_argument("--glvq_e8p_encode_chunk_size", type=int, default=1024)
    parser.add_argument("--glvq_group_size", type=int, default=128)
    parser.add_argument("--glvq_outer_steps", type=int, default=20)
    parser.add_argument("--glvq_generator_steps", type=int, default=10)
    parser.add_argument("--glvq_learning_rate", type=float, default=0.001)
    parser.add_argument("--glvq_regularization", type=float, default=0.001)
    parser.add_argument("--glvq_generator_learning_rate", type=float, default=None)
    parser.add_argument("--glvq_offset_learning_rate", type=float, default=None)
    parser.add_argument("--glvq_generator_regularization", type=float, default=None)
    parser.add_argument("--glvq_offset_regularization", type=float, default=None)
    parser.add_argument("--glvq_solver", choices=["adam"], default="adam", help="Fixed release setting")
    parser.add_argument(
        "--glvq_continuous_update_tolerance",
        type=float,
        default=0.0,
        help="Maximum relative reconstruction-loss increase accepted after a continuous solver update.",
    )
    parser.add_argument("--glvq_relative_tolerance", type=float, default=0.0001)
    parser.add_argument("--glvq_patience", type=int, default=3)
    parser.add_argument("--glvq_min_learning_rate", type=float, default=1e-06)
    parser.add_argument("--glvq_code_update_tolerance", type=float, default=0.0)
    parser.add_argument("--glvq_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--glvq_covariance_eps", type=float, default=1e-06)
    parser.add_argument("--glvq_init_clip", type=float, default=1.5)
    parser.add_argument("--glvq_sigma_min", type=float, default=1e-05)
    parser.add_argument("--glvq_sigma_max", type=float, default=5.0)
    parser.add_argument("--glvq_scale_eps", type=float, default=1e-08)
    parser.add_argument("--glvq_scale_min_factor", type=float, default=0.25)
    parser.add_argument("--glvq_scale_max_factor", type=float, default=4.0)
    parser.add_argument("--glvq_offset_max_delta", type=float, default=1.0)
    parser.add_argument("--glvq_hessian_damp", type=float, default=0.01)
    parser.add_argument(
        "--glvq_hessian_encoder", choices=["gptq"], default="gptq", help="Fixed release setting"
    )
    parser.add_argument(
        "--quantize_n_layers",
        type=int,
        default=None,
        help="Quantize only the first N decoder layers for a small validation run.",
    )
    parser.add_argument(
        "--quantize_layers",
        type=int,
        nargs="+",
        default=None,
        help="Quantize only these decoder-layer indices while propagating through intervening FP layers.",
    )
    parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        help="Skip model-level perplexity evaluation (evaluated separately by the pipeline).",
    )
    parser.add_argument(
        "--glvq_local_search_sweeps",
        type=int,
        default=0,
        help="Coordinate-descent sweeps used to refine bounded Babai lattice codes.",
    )
    parser.add_argument(
        "--glvq_hadamard_block_size",
        type=int,
        default=0,
        help="Randomized block-Hadamard width; zero disables rotation.",
    )
    parser.add_argument("--glvq_hadamard_seed", type=int, default=0)
    parser.add_argument(
        "--glvq_hadamard_output_block_size",
        type=int,
        default=0,
        help="QuIP#-style left/output randomized Hadamard width; zero disables it.",
    )
    parser.add_argument("--glvq_hadamard_output_seed", type=int, default=1)
    parser.add_argument(
        "--glvq_companding_mu",
        type=float,
        default=0.0,
        help="Optional mu-law companding strength; zero disables it.",
    )
    parser.add_argument(
        "--glvq_companding_clip",
        type=float,
        default=0.0,
        help="Normalize/clamp RMS-scaled weights to this range before companding; zero is unbounded.",
    )
    parser.add_argument(
        "--glvq_companding_learnable", action="store_true", help="Learn one bounded mu value per input group."
    )
    parser.add_argument("--glvq_companding_mu_min", type=float, default=0.0)
    parser.add_argument("--glvq_companding_mu_max", type=float, default=1.0)
    parser.add_argument(
        "--glvq_companding_init", choices=["constant"], default="constant", help="Fixed release setting"
    )
    parser.add_argument(
        "--glvq_companding_normalization", choices=["rms"], default="rms", help="Fixed release setting"
    )
    parser.add_argument("--glvq_companding_scope", choices=["group"], default="group")
    parser.add_argument("--glvq_companding_soft_tail", action="store_true")
    parser.add_argument("--glvq_companding_learning_rate", type=float, default=None)
    parser.add_argument("--glvq_companding_regularization", type=float, default=1e-05)
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--model_seqlen", type=int, default=4096, help="Model seqlen and calibration data context length."
    )
    parser.add_argument("--load", type=str, default=None, help="Path to load quantized statistics.")
    parser.add_argument("--save", type=str, default=None, help="Path to save quantized statistics.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If true, search for previously saved layers and reuse them to save time. Requires --save path.",
    )
    parser.add_argument("--devices", metavar="N", type=str, nargs="+", default=None, help="List of devices")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--no_safetensors",
        action="store_true",
        help="Load PyTorch .bin weights even when a safetensors index is present.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--skip_out_loss", action="store_true", help="Whether to skip computation of out loss."
    )
    parser.add_argument(
        "--offload_activations", action="store_true", help="Offload activations to RAM to save GPU memory."
    )
    parser.add_argument(
        "--true-sequential", action="store_true", help="Whether to run in true sequential model."
    )
    parser.add_argument(
        "--finetune_max_epochs",
        type=int,
        default=5,
        help="Run this many passes over training data when doing finetuning; No finetuning if set to 0.",
    )
    parser.add_argument(
        "--finetune_early_stop",
        type=int,
        default=3,
        help="Terminate finetuning if loss doesn't improve after this number of epochs.",
    )
    parser.add_argument("--finetune_lr", type=float, default=1e-05, help="Finetuning learning rate")
    parser.add_argument(
        "--finetune_parameter_scope", choices=["all"], default="all", help="Fixed release setting"
    )
    parser.add_argument("--finetune_generator_lr", type=float, default=None)
    parser.add_argument("--finetune_offset_lr", type=float, default=None)
    parser.add_argument("--finetune_scale_lr", type=float, default=None)
    parser.add_argument("--finetune_companding_lr", type=float, default=None)
    parser.add_argument("--finetune_non_quantized_lr", type=float, default=None)
    parser.add_argument(
        "--finetune_batch_size",
        type=int,
        default=1,
        help="(finetuning only) train on batches of this many sequences, globally across all GPUs",
    )
    parser.add_argument("--finetune_adam_beta1", type=float, default=0.9, help="Finetuning adam_beta1")
    parser.add_argument("--finetune_adam_beta2", type=float, default=0.95, help="Finetuning adam_beta2")
    parser.add_argument("--finetune_keep_best", action="store_true")
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=None,
        help="(finetuning only) Per-device and per-forward-pass batch size used to accumulate global --batch_size",
    )
    parser.add_argument("--val_size", type=int, default=0, help="Num validation sequences")
    parser.add_argument(
        "--print_frequency",
        type=int,
        default=10,
        help="Print Adam progress after each print_frequency updates",
    )
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="Skip model quantization and immediately evaluate the loaded model",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "flash_attention_2", "sdpa"],
        help="Attention implementation.",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer (some models have only fast tokenizer).",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to trust remote code.")
    torch.set_num_threads(min(16, torch.get_num_threads()))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.true_sequential = True
    if args.devices is None:
        if torch.cuda.is_available():
            args.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        else:
            args.devices = [torch.device("cpu")]
    else:
        args.devices = [torch.device(device_str) for device_str in args.devices]
    assert all((isinstance(device, torch.device) for device in args.devices))
    if args.nsamples is not None:
        assert args.val_size < args.nsamples, "Number of validation set must be smaller than train + val"
    print("\n============ Load model... ============")
    model = get_model(
        args.model_path,
        args.load,
        args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        use_safetensors=False if args.no_safetensors else None,
    ).train(False)
    if not args.load and (not args.no_quant):
        print("\n============ Quantizing model... ============")
        quantize_model(model, args)
    if args.skip_final_eval:
        print("\n============ Skipping final perplexity evaluation ============")
        return
    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    datasets = args.eval_datasets or ["wikitext2", "c4"]
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=args.model_seqlen,
            eval_mode=True,
            use_fast_tokenizer=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
        args.dataset_name = dataset
        perplexity_eval(model, testloader, args)
    print(f"eval: torch.cuda.max_memory_allocated()={torch.cuda.max_memory_allocated():,}")


if __name__ == "__main__":
    main()
