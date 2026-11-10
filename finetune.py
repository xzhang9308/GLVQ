"""
Fine-tune continuous GLVQ parameters with fixed discrete codes;
based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""

import argparse
import os
from contextlib import nullcontext
from typing import Optional
import datasets
import torch
import torch.distributed
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import transformers
from torch import nn as nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    StateDictType,
)
from tqdm.auto import tqdm
from src.configurable_adam import ConfigurableAdamW
from src.datautils import evaluate_perplexity, get_loaders
from src.glvq import GLVQWeight
from src.memory_efficient_loss import compute_kl_divergence_loss_values
from src.modelutils import get_model, is_model_for_causal_lm
from src.fsdp_utils import infer_module_classes
from src.utils import master_rank_first, one_rank_at_a_time


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument("--base_model", type=str, required=True, help="path or name of the teacher model")
    parser.add_argument("--quantized_model", type=str, required=True, help="path to quantized model")
    parser.add_argument(
        "--model_seqlen", type=int, default=4096, help="Model seqlen and calibration data context length."
    )
    parser.add_argument(
        "--master_dtype",
        type=str,
        default="float32",
        help="data type for storing master parameters and computing optimizer updates",
    )
    parser.add_argument(
        "--embed_dtype",
        type=str,
        default=None,
        help="data type for storing master input and output embeddings; defaults to master_dtype",
    )
    parser.add_argument(
        "--load_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default=None,
        help="if specified, runs automated mixed precision with this dtype",
    )
    parser.add_argument(
        "--block_type",
        type=str,
        required=True,
        help="string name of a transformer layer to wrap, e.g. LlamaDecoderLayer",
    )
    parser.add_argument(
        "--wrap_separately",
        type=str,
        nargs="*",
        default=[],
        help="module classes (by name, similar to block_type) that will be wrapped in a separate fsdp instance and do not participate in FSDP AMP (if used). Applies to the student (de)quantized model, not the teacher model.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument(
        "--limit_parallel_inits",
        type=int,
        default=1,
        help="this many ranks (per host) initialize their model in parallel. This parameter is meant to save host RAM.",
    )


def add_finetuning_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--update_codebooks_and_scales",
        action="store_true",
        help="If set, train continuous parameters of quantized representations; if not, freeze them",
    )
    parser.add_argument(
        "--update_non_quantized_parameters",
        action="store_true",
        help="If set, train the non-quantized model parameters (layernorm scales, biases, logits); if not, freeze them",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-05, help="finetuning learning rate for continuous params"
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1 for continuous params")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2 for continuous params")
    parser.add_argument(
        "--offload_optimizer", action="store_true", help="If set, adam statistics will be offloaded to RAM"
    )
    parser.add_argument(
        "--offload_teacher_params",
        action="store_true",
        help="If set, the teacher model will be offloaded to RAM and paged using FSDP's CPUOffload",
    )
    parser.add_argument(
        "--limit_all_gathers", action="store_true", help="sets limit_all_gathers in both FSDP instances"
    )
    parser.add_argument(
        "--forward_prefetch", action="store_true", help="sets forward_prefetech in both FSDP instances"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="training batch size - how many samples are processed per optimizer step, between all GPUs in total",
    )
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=None,
        help="training microbatch size - how many samples are processed per GPU per forward pass",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to apply gradient checkpointing for transformer blocks",
    )
    parser.add_argument(
        "--loss_tokens_per_chunk",
        type=int,
        default=None,
        help="If specified, compute LM logits and loss using gradient checkpointing in chunks of this size.This option slows down loss computation, but reduces memory usage. Recommended for large vocabularies",
    )
    parser.add_argument(
        "--use_fsdp_amp",
        action="store_true",
        help="Whether to use FSDP native mixed precision (excluding registered layernorms and --wrap_separately).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for calibration data and initialization. Note that the main training is not strictly deterministic.",
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save training snapshot.")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Total number of training epochs (passes over calibration data) after which the training will conclude",
    )
    parser.add_argument(
        "--max_optimizer_steps",
        type=int,
        default=None,
        help="Stop after this many optimizer updates, even if the current epoch is incomplete",
    )
    parser.add_argument(
        "--print_every_steps",
        type=int,
        default=None,
        help="print training metrics once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=None,
        help="evaluate once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument(
        "--eval_before_training",
        action="store_true",
        help="Evaluate the untouched quantized model at step 0 and use it as the keep-best baseline",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Load the requested checkpoint, evaluate --eval_datasets, and exit without updating or saving",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="If specified, clip the global student gradient norm before each optimizer update",
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=None,
        help="save state once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument("--keep_best_model", action="store_true", help="Save best model state separately")
    parser.add_argument(
        "--save_final_model",
        action="store_true",
        help="Export the final fixed-code model independently of validation selection",
    )


def add_data_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Path to the frozen tokenized dataset from scripts.prepare_data",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU data loading workers")
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["wikitext2", "c4"],
        help="Datasets to run evaluation on",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code when loading base model.",
    )


def prepare_training_dataset(args: argparse.Namespace) -> datasets.Dataset:
    """Load the frozen token dataset produced by scripts.prepare_data."""
    dataset = datasets.load_from_disk(args.dataset_name)
    if "input_ids" not in dataset.column_names or not len(dataset):
        raise ValueError("Expected a nonempty frozen token dataset")
    if len(dataset[0]["input_ids"]) != args.model_seqlen:
        raise ValueError("Training tokens do not match model_seqlen")
    return dataset


def load_teacher_model(args: argparse.Namespace, device: torch.device) -> FullyShardedDataParallel:
    """Load unquantized model with frozen parameters"""
    model = get_model(
        args.base_model,
        load_quantized=None,
        dtype=args.load_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).to(dtype=args.load_dtype if args.load_dtype != "auto" else None)
    model.train(False)
    for param in model.parameters():
        param.requires_grad = False
    model.config.use_cache = False
    transformer_block_types = infer_module_classes(model, args.block_type)
    return wrap_model_with_fsdp_(
        model,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse
        or isinstance(module, transformer_block_types),
        cpu_offload=(
            CPUOffload(offload_params=args.offload_teacher_params) if args.offload_teacher_params else None
        ),
        limit_all_gathers=args.limit_all_gathers,
        forward_prefetch=args.forward_prefetch,
        device_id=device,
    )


def load_student_model(args: argparse.Namespace, device: torch.device) -> FullyShardedDataParallel:
    """
    Load the GLVQ student for continuous-parameter fine-tuning.
    """
    student_model = get_model(
        args.base_model,
        args.quantized_model,
        dtype=args.load_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).to(args.master_dtype)
    if args.embed_dtype != args.master_dtype:
        student_model.set_output_embeddings(student_model.get_output_embeddings().to(args.embed_dtype))
        student_model.set_input_embeddings(student_model.get_input_embeddings().to(args.embed_dtype))
    student_model.config.use_cache = False
    student_model.train(True)
    if args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
        student_model.enable_input_require_grads()
    if not any((isinstance(module, GLVQWeight) for module in student_model.modules())):
        raise RuntimeError("The student checkpoint contains no GLVQ weights")
    transformer_block_types = list(infer_module_classes(student_model, args.block_type))
    layernorm_types = list(transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    extra_block_types = list()
    for extra_module_name in args.wrap_separately:
        extra_block_types.extend(infer_module_classes(student_model, extra_module_name))
    block_types_to_wrap = tuple(set(transformer_block_types + layernorm_types + extra_block_types))
    if torch.distributed.get_rank() == 0:
        print(f"Blocks to be wrapped separately: {block_types_to_wrap}\n")
    mixed_precision = None
    if args.use_fsdp_amp:
        assert args.amp_dtype is not None, "requested to use_fsdp_amp, but amp_dtype is not None"
        block_types_for_amp_to_ignore = tuple(set(layernorm_types + extra_block_types))
        if torch.distributed.get_rank() == 0:
            print(f"Blocks excluded from AMP: {block_types_for_amp_to_ignore}\n")
        mixed_precision = MixedPrecision(
            param_dtype=args.amp_dtype,
            reduce_dtype=args.amp_dtype,
            _module_classes_to_ignore=block_types_for_amp_to_ignore,
        )
    elif torch.distributed.get_rank() == 0:
        print(f"Not using FSDP native MixedPrecision; Local amp_dtype={args.amp_dtype}.")
    student_model = wrap_model_with_fsdp_(
        student_model,
        use_orig_params=True,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse or isinstance(module, block_types_to_wrap),
        cpu_offload=None,
        limit_all_gathers=args.limit_all_gathers,
        forward_prefetch=args.forward_prefetch,
        mixed_precision=mixed_precision,
        device_id=device,
    )
    return student_model


def wrap_model_with_fsdp_(
    model: transformers.PreTrainedModel, auto_wrap_policy: callable, **kwargs
) -> FullyShardedDataParallel:
    """Wrap a model *ForCausalLM components: transformer and lm_head are wrapped as FSDP instances"""
    assert isinstance(model, transformers.PreTrainedModel) and is_model_for_causal_lm(model)
    (base_model, lm_head) = (model.base_model, model.get_output_embeddings())

    def _modified_auto_wrap_policy(module, recurse, **kwargs):
        return auto_wrap_policy(module, recurse, **kwargs) or module in (base_model, lm_head)

    model = FullyShardedDataParallel(model, auto_wrap_policy=_modified_auto_wrap_policy, **kwargs)
    assert isinstance(model.module, transformers.PreTrainedModel)
    assert isinstance(model.base_model, FullyShardedDataParallel)
    assert isinstance(model.get_output_embeddings(), FullyShardedDataParallel)
    return model


def trigger_fsdp_lazy_init_(
    tokenizer: transformers.PreTrainedTokenizer,
    teacher_model: FullyShardedDataParallel,
    student_model: FullyShardedDataParallel,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
):
    """Trigger FullyShardedDataParallel lazy init in the correct order to allow both training and eval"""
    print("Initializing FSDP root")
    dummy_batch = tokenizer("I am the monument to all your sins", return_tensors="pt")
    dummy_batch = {k: v.to(device) for (k, v) in dummy_batch.items()}
    with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
        with torch.no_grad():
            teacher_model(**dummy_batch)
        (student_model(**dummy_batch).logits * 0).sum().backward()


def _continuous_quantized_parameters(model: nn.Module) -> set[nn.Parameter]:
    quantized_weight_continuous_parameters = set()
    for module in model.modules():
        if isinstance(module, GLVQWeight):
            for param in module.parameters():
                if torch.is_floating_point(param) and param.requires_grad:
                    quantized_weight_continuous_parameters.add(param)
    return quantized_weight_continuous_parameters


def create_p_optimizer(
    args: argparse.Namespace, student_model: FullyShardedDataParallel
) -> torch.optim.Optimizer:
    """Create optimizer for training only continuous parameters of a quantized model."""
    quantized_weight_continuous_parameters = _continuous_quantized_parameters(student_model)
    all_trainable_params = []
    if args.update_codebooks_and_scales:
        all_trainable_params.extend(
            (param for param in student_model.parameters() if param in quantized_weight_continuous_parameters)
        )
    if args.update_non_quantized_parameters:
        all_trainable_params.extend(
            (
                param
                for param in student_model.parameters()
                if torch.is_floating_point(param)
                and param.requires_grad
                and (param not in quantized_weight_continuous_parameters)
            )
        )
    assert (
        len(all_trainable_params) > 0
    ), "found no trainable parameters. Did you specify update_codebooks_and_scales or update_non_quantized_parameters?"
    opt_device = torch.device("cpu") if args.offload_optimizer else next(student_model.parameters()).device
    return ConfigurableAdamW(
        params=list(all_trainable_params),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        lamb=False,
        debias=None,
        amsgrad=False,
        compute_dtype=args.master_dtype,
        exp_avg_dtype=args.master_dtype,
        exp_avg_sq_dtype=args.master_dtype,
        v_hat_max_dtype=args.master_dtype,
        exp_avg_device=opt_device,
        exp_avg_sq_device=opt_device,
        v_hat_max_device=opt_device,
    )


def save_training_state(
    args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: torch.optim.Optimizer
):
    """Save model, optimizer state dict and training metadata to be loaded via load_training_state"""
    if args.save is None:
        return
    rank = torch.distributed.get_rank()
    os.makedirs(args.save, exist_ok=True)
    if rank == 0:
        print(f"Saving snapshot to {args.save}")
        torch.save(metadata, os.path.join(args.save, "metadata.pt"))
    with FullyShardedDataParallel.state_dict_type(quantized_model, StateDictType.LOCAL_STATE_DICT):
        torch.save(
            quantized_model.state_dict(), os.path.join(args.save, f"quantized_model_state_dict_rank{rank}.pt")
        )
    torch.save(optimizer.state_dict(), os.path.join(args.save, f"optimizer_state_dict_rank{rank}.pt"))


def load_training_state(
    args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: torch.optim.Optimizer
):
    """Load model, optimizer state dict and metadata saved via save_training_state; update parameters in-place"""
    rank = torch.distributed.get_rank()
    if args.save is None or not os.path.exists(args.save):
        if args.save is not None and rank == 0:
            print(f"No checkpoint found at {args.save}")
    else:
        with FullyShardedDataParallel.state_dict_type(quantized_model, StateDictType.LOCAL_STATE_DICT):
            state_dict_ptr = quantized_model.state_dict()
            loaded_state_dict = torch.load(
                os.path.join(args.save, f"quantized_model_state_dict_rank{rank}.pt"), weights_only=False
            )
            with torch.no_grad():
                for key in state_dict_ptr:
                    state_dict_ptr[key].copy_(loaded_state_dict.pop(key))
                assert len(loaded_state_dict) == 0, f"Unused keys:, {tuple(loaded_state_dict.keys())}"
            del state_dict_ptr, loaded_state_dict
        optimizer.load_state_dict(
            torch.load(
                os.path.join(args.save, f"optimizer_state_dict_rank{rank}.pt"),
                map_location="cpu",
                weights_only=False,
            )
        )
        metadata.update(torch.load(os.path.join(args.save, "metadata.pt"), weights_only=True))
        if args.eval_datasets is not None and metadata["early_stop_on"] not in args.eval_datasets:
            if rank == 0:
                print(
                    f"Stopping criterion {metadata['early_stop_on']} is not in eval_datasets; resetting best loss."
                )
            metadata["early_stop_on"] = next(iter(args.eval_datasets))
            metadata["best_eval_perplexity"] = float("inf")
            metadata["best_step"] = 0
        if rank == 0:
            print(f"Loaded training state from {args.save}: {metadata}")


def save_model(
    args: argparse.Namespace, student_model: FullyShardedDataParallel, optimizer: torch.optim.Optimizer
):
    """Save continuous GLVQ parameters while preserving codes."""
    assert any((isinstance(module, GLVQWeight) for module in student_model.modules()))
    save_p_model(args, student_model)


def save_p_model(
    args: argparse.Namespace,
    quantized_model: FullyShardedDataParallel,
    filename: str = "best_model_state_dict.pt",
):
    """Save consolidated model state dict from P-only tuning, can be exported via export.py"""
    os.makedirs(args.save, exist_ok=True)
    rank = torch.distributed.get_rank()
    with FullyShardedDataParallel.state_dict_type(
        quantized_model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=True),
    ):
        model_state_dict = quantized_model.state_dict()
    if rank == 0:
        for name, tensor in model_state_dict.items():
            model_state_dict[name] = tensor.detach().cpu()
        target = os.path.join(args.save, filename)
        torch.save(model_state_dict, target + ".tmp")
        os.replace(target + ".tmp", target)
    torch.distributed.barrier()
    if rank == 0:
        print(f"Saved model state dict to {os.path.join(args.save, filename)}")


def compute_loss_on_batch(
    batch: dict,
    teacher_model: FullyShardedDataParallel,
    student_model: FullyShardedDataParallel,
    *,
    amp_dtype: Optional[torch.dtype],
    max_tokens_per_chunk: Optional[int],
) -> torch.Tensor:
    if max_tokens_per_chunk is not None:
        with torch.no_grad():
            teacher_hidden_states = teacher_model.base_model(**batch).last_hidden_state
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
            student_hidden_states = student_model.base_model(**batch).last_hidden_state
            return compute_kl_divergence_loss_values(
                student_hidden_states=student_hidden_states,
                student_lm_head=student_model.get_output_embeddings(),
                teacher_hidden_states=teacher_hidden_states,
                teacher_lm_head=teacher_model.get_output_embeddings(),
                max_tokens_per_chunk=max_tokens_per_chunk,
                checkpoint_last_chunk=False,
                use_reentrant=False,
                determinism_check="none",
            ).mean()
    else:
        with torch.no_grad():
            teacher_logprobs = F.log_softmax(teacher_model(**batch).logits, dim=-1)
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
            student_logprobs = F.log_softmax(student_model(**batch).logits, dim=-1)
            loss = F.kl_div(
                input=student_logprobs.flatten(0, -2),
                target=teacher_logprobs.flatten(0, -2),
                log_target=True,
                reduction="batchmean",
            ).mean()
        return loss


def compute_validation_perplexities(args: argparse.Namespace, model: nn.Module, eval_datasets: dict):
    rank = torch.distributed.get_rank()
    perplexities = {}
    for dataset_name, eval_dataset in eval_datasets.items():
        if rank == 0:
            print(f"Evaluating perplexity on {dataset_name} ...")
        device = (
            model.compute_device
            if isinstance(model, FullyShardedDataParallel)
            else next(model.parameters()).device
        )
        original_dtype = args.load_dtype if args.load_dtype != "auto" else None
        amp_dtype = args.amp_dtype if args.amp_dtype is not None else original_dtype
        ppl = evaluate_perplexity(model, eval_dataset, args.model_seqlen, device=device, amp_dtype=amp_dtype)
        if rank == 0:
            print(f"{dataset_name} perplexity: {ppl:.9f}")
        perplexities[dataset_name] = ppl
    return perplexities


def main():
    assert torch.cuda.is_available() and torch.distributed.is_available()
    torch.distributed.init_process_group()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    parser = argparse.ArgumentParser(add_help=True)
    add_model_args(parser)
    add_data_args(parser)
    add_finetuning_args(parser)
    args = parser.parse_args()
    assert torch.distributed.is_initialized()
    assert args.batch_size is not None, "please specify batch size"
    assert args.batch_size % world_size == 0
    if args.microbatch_size is None:
        args.microbatch_size = args.batch_size // world_size
    assert args.batch_size % (world_size * args.microbatch_size) == 0
    grad_accumulation_steps = args.batch_size // (world_size * args.microbatch_size)
    args.master_dtype = getattr(torch, args.master_dtype)
    args.embed_dtype = getattr(torch, args.embed_dtype) if args.embed_dtype is not None else args.master_dtype
    args.load_dtype = getattr(torch, args.load_dtype) if args.load_dtype != "auto" else "auto"
    args.amp_dtype = getattr(torch, args.amp_dtype) if args.amp_dtype is not None else None
    if args.save_every_steps is not None:
        assert (
            args.save is not None
        ), f"save_every_steps={args.save_every_steps}, but --save path not specified"
    if args.keep_best_model:
        assert args.save is not None, f"--keep_best_model requires --save path"
        assert args.eval_every_steps is not None, f"--keep_best_model requires --eval_every_steps"
        assert args.eval_datasets is not None, f"--keep_best_model requires --eval_datasets"
    if args.eval_before_training:
        assert args.eval_datasets is not None, "--eval_before_training requires --eval_datasets"
    if args.max_optimizer_steps is not None:
        assert args.max_optimizer_steps > 0
    if args.max_grad_norm is not None:
        assert args.max_grad_norm > 0
    if rank == 0:
        print(args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert tokenizer.eos_token_id is not None
    tokenizer.pad_token = tokenizer.eos_token
    with master_rank_first(local=True):
        dataset = prepare_training_dataset(args)
    sampler = torch.utils.data.DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.seed
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.microbatch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=transformers.default_data_collator,
    )
    eval_datasets = {
        dataset_name: get_loaders(
            dataset_name,
            seed=args.seed,
            model_path=args.base_model,
            seqlen=args.model_seqlen,
            eval_mode=True,
            use_fast_tokenizer=args.use_fast_tokenizer,
        )
        for dataset_name in args.eval_datasets
    }
    if rank == 0:
        print("Training continuous GLVQ parameters with fixed discrete codes")
    with one_rank_at_a_time(local=True, group_size=args.limit_parallel_inits):
        teacher_model = load_teacher_model(args, device)
        student_model = load_student_model(args, device)
        if rank == 0:
            print("Wrapped model:")
            print(student_model)
            trainable_parameters = sum(
                (param.numel() for param in student_model.parameters() if param.requires_grad)
            )
            print(f"Trainable parameters on rank 0: {trainable_parameters:,}")
    optimizer = create_p_optimizer(args, student_model)
    if rank == 0:
        optimizer_parameters = sum(
            (param.numel() for group in optimizer.param_groups for param in group["params"])
        )
        print(f"Optimizer parameters on rank 0: {optimizer_parameters:,}")
    metadata = dict(
        current_epoch=0,
        microbatches_since_epoch_start=0,
        total_microbatches=0,
        total_optimizer_steps=0,
        loss_numerator=0,
        loss_denominator=0,
        aggregated_loss=float("nan"),
        grad_steps_accumulated=0,
        early_stop_on=next(iter(args.eval_datasets)) if args.eval_datasets else None,
        best_eval_perplexity=float("inf"),
        best_step=0,
    )
    load_training_state(args, metadata, student_model, optimizer)
    torch.distributed.barrier()
    trigger_fsdp_lazy_init_(tokenizer, teacher_model, student_model, device, amp_dtype=args.amp_dtype)
    if args.eval_only:
        compute_validation_perplexities(args, student_model, eval_datasets)
        torch.distributed.barrier()
        return
    if args.eval_before_training and metadata["total_optimizer_steps"] == 0:
        baseline_scores = compute_validation_perplexities(args, student_model, eval_datasets)
        for dataset_name, perplexity in baseline_scores.items():
            metadata[f"perplexity_{dataset_name}"] = perplexity
        metric_name = metadata["early_stop_on"]
        metadata["best_eval_perplexity"] = baseline_scores[metric_name]
        metadata["best_step"] = 0
        if rank == 0:
            print(f"Baseline perplexity ({metric_name}) = {baseline_scores[metric_name]:.9f}")
    stop_training = False
    for current_epoch in range(args.max_epochs):
        if current_epoch < metadata["current_epoch"]:
            continue
        sampler.set_epoch(current_epoch)
        batch_iter = (
            tqdm(train_dataloader, desc=f"Training epoch #{current_epoch}") if rank == 0 else train_dataloader
        )
        for batch_index, batch in enumerate(batch_iter):
            if batch_index < metadata["microbatches_since_epoch_start"]:
                continue
            metadata["microbatches_since_epoch_start"] += 1
            metadata["total_microbatches"] += 1
            batch = {k: v.to(device) for (k, v) in batch.items()}
            loss = compute_loss_on_batch(
                batch,
                teacher_model,
                student_model,
                amp_dtype=args.amp_dtype,
                max_tokens_per_chunk=args.loss_tokens_per_chunk,
            )
            metadata["loss_numerator"] += loss.item()
            metadata["loss_denominator"] += 1
            metadata["grad_steps_accumulated"] += 1
            if metadata["grad_steps_accumulated"] < grad_accumulation_steps:
                with nullcontext():
                    (loss / grad_accumulation_steps).backward()
            else:
                (loss / grad_accumulation_steps).backward()
                if args.max_grad_norm is not None:
                    grad_norm = student_model.clip_grad_norm_(args.max_grad_norm)
                    metadata["grad_norm"] = float(grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                metadata["grad_steps_accumulated"] = 0
                metadata["total_optimizer_steps"] += 1
                if args.print_every_steps and metadata["total_optimizer_steps"] % args.print_every_steps == 0:
                    loss_numerator_and_denominator = torch.tensor(
                        [metadata["loss_numerator"], metadata["loss_denominator"]],
                        dtype=torch.float64,
                        device=device,
                    )
                    torch.distributed.all_reduce(
                        loss_numerator_and_denominator, op=torch.distributed.ReduceOp.SUM
                    )
                    (loss_numerator, loss_denominator) = loss_numerator_and_denominator.tolist()
                    metadata["aggregated_loss"] = loss_numerator / loss_denominator
                    metadata["loss_numerator"] = metadata["loss_denominator"] = 0
                    if rank == 0:
                        print(
                            f"epoch {metadata['current_epoch']}\tbatch {batch_index}",
                            f"\t| total updates = {metadata['total_optimizer_steps']}",
                            f"\tloss = {metadata['aggregated_loss']:.9f}",
                        )
                if args.eval_every_steps and metadata["total_optimizer_steps"] % args.eval_every_steps == 0:
                    perplexity_scores = compute_validation_perplexities(args, student_model, eval_datasets)
                    for dataset_name, perplexity in perplexity_scores.items():
                        metadata[f"perplexity_{dataset_name}"] = perplexity
                    metric_name = metadata["early_stop_on"]
                    if perplexity_scores[metric_name] < metadata["best_eval_perplexity"]:
                        if rank == 0:
                            print(
                                f"New best perplexity ({metric_name}) = {perplexity_scores[metric_name]:.9f}"
                            )
                        metadata["best_eval_perplexity"] = perplexity_scores[args.eval_datasets[0]]
                        metadata["best_step"] = metadata["total_optimizer_steps"]
                        if args.keep_best_model:
                            save_model(args, student_model, optimizer)
                if args.save_every_steps and metadata["total_optimizer_steps"] % args.save_every_steps == 0:
                    save_training_state(args, metadata, student_model, optimizer)
                if (
                    args.max_optimizer_steps is not None
                    and metadata["total_optimizer_steps"] >= args.max_optimizer_steps
                ):
                    stop_training = True
                    break
        if stop_training:
            break
        metadata["microbatches_since_epoch_start"] = 0
        metadata["current_epoch"] += 1
    save_training_state(args, metadata, student_model, optimizer)
    if args.save_final_model:
        save_p_model(args, student_model, "final_model_state_dict.pt")


if __name__ == "__main__":
    main()
