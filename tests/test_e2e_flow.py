"""CPU integration of distillation, optimizer resume and both export handoffs.

This exercises real model/loss/optimizer/export code. CUDA FSDP and pinned-memory
offload require a separate GPU run and are deliberately not simulated here.
"""

import copy
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, MistralConfig, Qwen2Config

from finetune import compute_loss_on_batch, create_p_optimizer
from run import ROOT
from src.glvq import GLVQConfig, GLVQLinear
from src.modelutils import get_model, save_quantized_model


def _optimizer(model, *, extended=False):
    optimizer = create_p_optimizer(
        SimpleNamespace(
            update_codebooks_and_scales=True,
            update_non_quantized_parameters=extended,
            offload_optimizer=False,
            lr=1e-6,
            adam_beta1=0.9,
            adam_beta2=0.95,
            master_dtype=torch.float32,
        ),
        model,
    )
    # Ordinary CPU allocation: pinned CUDA transfer buffers cannot be tested on
    # a CPU-only host. This changes storage allocation, not optimizer arithmetic.
    for group in optimizer.param_groups:
        for key in ("exp_avg_device", "exp_avg_sq_device", "v_hat_max_device"):
            group[key] = None
    return optimizer


def _prepare_student(model):
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model


def _step(teacher, student, optimizer, batch):
    loss = compute_loss_on_batch(batch, teacher, student, amp_dtype=None, max_tokens_per_chunk=3)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in optimized)
    assert student.model.layers[0].self_attn.q_proj.quantized_weight.generators.grad.abs().sum() > 0
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    assert all(p.grad is None for p in teacher.parameters())
    return loss.detach()


def _export(base, checkpoint, state, output):
    subprocess.run(
        [
            sys.executable,
            "export.py",
            "--base_model",
            str(base),
            "--quantized_model",
            str(checkpoint),
            "--p_finetuned_state_dict",
            str(state),
            "--load_dtype",
            "float32",
            "--save",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return get_model(str(base), str(output), dtype=torch.float32)


def _assert_same_state(left, right):
    left_state, right_state = left.state_dict(), right.state_dict()
    assert left_state.keys() == right_state.keys()
    for name in left_state:
        assert torch.equal(left_state[name], right_state[name]), name


@pytest.mark.parametrize("config_type", [LlamaConfig, MistralConfig, Qwen2Config])
def test_distillation_resume_and_p_only_to_extended_export(tmp_path, config_type):
    torch.manual_seed(137)
    config = config_type(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=False,
    )
    teacher = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
    base = tmp_path / "base"
    teacher.save_pretrained(base)
    student = copy.deepcopy(teacher).requires_grad_(True)
    attention = student.model.layers[0].self_attn
    attention.q_proj = GLVQLinear.from_linear(
        attention.q_proj,
        GLVQConfig(
            codebook="e8p",
            dimension=8,
            deformation="full",
            hadamard_block_size=2048,
            hadamard_output_block_size=2048,
            companding_mu=0.25,
            companding_learnable=True,
            companding_mu_min=0,
            companding_mu_max=8,
            companding_clip=3,
            companding_soft_tail=True,
        ),
    )
    checkpoint = tmp_path / "ptq"
    save_quantized_model(student, str(checkpoint))
    student = _prepare_student(get_model(str(base), str(checkpoint), dtype=torch.float32))
    optimizer = _optimizer(student)
    batch = {"input_ids": torch.randint(0, config.vocab_size, (2, 8))}

    # Chunked distillation must agree with a direct full-logit KL calculation,
    # including its backward signal to the actual continuous GLVQ parameters.
    optimized = tuple(p for group in optimizer.param_groups for p in group["params"])
    chunked_loss = compute_loss_on_batch(batch, teacher, student, amp_dtype=None, max_tokens_per_chunk=3)
    chunked_loss.backward()
    chunked_grads = [param.grad.clone() for param in optimized]
    student.zero_grad()
    direct_loss = compute_loss_on_batch(batch, teacher, student, amp_dtype=None, max_tokens_per_chunk=None)
    direct_loss.backward()
    direct_grads = [param.grad.clone() for param in optimized]
    student.zero_grad()
    torch.testing.assert_close(chunked_loss, direct_loss, rtol=1e-5, atol=1e-7)
    for chunked, direct in zip(chunked_grads, direct_grads):
        torch.testing.assert_close(chunked, direct, rtol=1e-4, atol=1e-8)

    initial_state = {name: value.clone() for name, value in student.state_dict().items()}
    _step(teacher, student, optimizer, batch)

    # A real serialized optimizer/model round trip must reproduce the next
    # update, including Adam's accumulated moments rather than restarting them.
    snapshot = tmp_path / "resume.pt"
    torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict()}, snapshot)
    resumed = _prepare_student(get_model(str(base), str(checkpoint), dtype=torch.float32))
    resumed_optimizer = _optimizer(resumed)
    saved = torch.load(snapshot, weights_only=False)
    resumed.load_state_dict(saved["model"])
    resumed_optimizer.load_state_dict(saved["optimizer"])
    _assert_same_state(student, resumed)
    assert all(state["step"] == 1 for state in resumed_optimizer.state.values())
    for _ in range(2):
        expected_loss = _step(teacher, student, optimizer, batch)
        resumed_loss = _step(teacher, resumed, resumed_optimizer, batch)
        assert torch.equal(expected_loss, resumed_loss)
        _assert_same_state(student, resumed)
    assert all(state["step"] == 3 for state in resumed_optimizer.state.values())

    quantized_prefix = "model.layers.0.self_attn.q_proj.quantized_weight."
    codes_key = quantized_prefix + "codes"
    for name, before in initial_state.items():
        if not name.startswith(quantized_prefix) or name == codes_key:
            assert torch.equal(before, student.state_dict()[name]), name
    assert not torch.equal(
        initial_state[quantized_prefix + "generators"], student.state_dict()[quantized_prefix + "generators"]
    )

    p_state = tmp_path / "p-final.pt"
    torch.save(student.state_dict(), p_state)
    extended_checkpoint = tmp_path / "p-export"
    extended = _export(base, checkpoint, p_state, extended_checkpoint)
    _assert_same_state(student, extended)
    _prepare_student(extended)
    extended_optimizer = _optimizer(extended, extended=True)
    norm_before = extended.model.norm.weight.detach().clone()
    head_before = extended.lm_head.weight.detach().clone()
    _step(teacher, extended, extended_optimizer, batch)
    assert not torch.equal(norm_before, extended.model.norm.weight)
    assert not torch.equal(head_before, extended.lm_head.weight)
    assert torch.equal(initial_state[codes_key], extended.state_dict()[codes_key])

    final_state = tmp_path / "extended-final.pt"
    torch.save(extended.state_dict(), final_state)
    final = _export(base, extended_checkpoint, final_state, tmp_path / "final-export")
    _assert_same_state(extended, final)
    with torch.no_grad():
        torch.testing.assert_close(
            extended.eval()(**batch).logits,
            final.eval()(**batch).logits,
            rtol=0,
            atol=0,
        )
