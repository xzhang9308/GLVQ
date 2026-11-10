<div align="center">

# GLVQ

### Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression

**Xi Zhang · Xiaolin Wu · Jiamang Wang · Weisi Lin**<br>
**NeurIPS 2025**

[![Paper](https://img.shields.io/badge/NeurIPS_2025-Paper-8c1b13?style=flat-square)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a0183ad2088503d707669e1d222bc8c4-Abstract-Conference.html)
[![arXiv](https://img.shields.io/badge/arXiv-2510.20984-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2510.20984)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](LICENSE)

Official PyTorch reimplementation of **GLVQ**, a learnable lattice vector
quantization method for low-bit large language model compression.

[Installation](#installation) · [Quick Start](#try-the-learnable-lattice) · [Training](#run-and-resume) · [Model Loading](#load-the-final-model) · [Citation](#citation)

</div>

---

## About this release

The original GLVQ implementation used an internal company framework. This
official reimplementation redevelops the method on a publicly available
PyTorch/AQLM engineering base, without that internal dependency.

Release: **2026.11.10**. This directory is self-contained: copy its contents into
your public repository or working directory. Importing it requires no parent
development repository or company framework. Full model runs require your own
model weights and data, which are not distributed here.

## Included components

| Component | Status |
| --- | --- |
| Learnable lattice with trainable full generator matrices | Included |
| d8 E8P representation with uniform 2-bit codes | Included |
| Two-sided block Hadamard and learnable group-wise companding | Included |
| Activation-aware fitting with GPTQ error feedback | Included |
| Block reconstruction and fixed-code end-to-end fine-tuning | Included |
| Data preparation, evaluation, resume, and layer-checkpoint export | Included |
| Bit allocation / mixed-bit allocation | **Not included yet** |
| PV-Tuning | **Not included yet** |

The lattice is learned, rather than only using a fixed E8P codebook:
[`GLVQWeight`](src/glvq.py) registers a trainable full 8×8 generator for each
input-column group, together with offsets, scales, and companding parameters.
[`fit_glvq_weight_xtx`](src/glvq.py) alternates code assignment and continuous
optimization against an activation-aware reconstruction objective. Decoding
applies the learned generators and inverse companding/rotations.

During block reconstruction and both end-to-end stages, discrete codes stay
frozen. **P-only** optimizes continuous quantizer parameters; **extended E2E**
also updates non-quantized model parameters. Neither stage is PV-Tuning: this
release does not update discrete codes during E2E. Code reassignment during
initial PTQ fitting is part of the lattice quantizer.

## Release recipe

The release uses the following uniform 2-bit quantization and fine-tuning pipeline:

```text
Two-sided block-2048 Hadamard
  → learnable group-wise μ companding
  → d8 E8P / full generator fitting (group size 128, uniform 2-bit codes)
  → block reconstruction
  → P-only E2E: 5 epochs
  → extended fixed-code E2E: 10 epochs
```

[`recipe.json`](recipe.json) contains PTQ settings, including top-8 E8P candidate
search and 20×10 fitting limits with early stopping. E2E uses LR 1e-6, effective
batch 8, microbatch 1, and seed 42.

## Installation

Use Linux, Python 3.12, and NVIDIA CUDA GPUs for full quantization/training.
Four GPUs are the established full-model configuration. The launcher accepts
1, 2, 4, or 8 GPUs while preserving effective batch 8; other GPU counts are not
established full-model reproduction configurations. CPU suffices for the small
example and regression tests.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Key dependencies in [`requirements.txt`](requirements.txt) include PyTorch
2.13.0 and Transformers 4.43.4. Install the PyTorch build appropriate for your
CUDA driver. This is not a complete transitive-dependency lockfile.
Optional five-task zero-shot evaluation requires:

```bash
python -m pip install -r requirements-eval.txt
```

## Try the learnable lattice

From this directory, run a small CPU example using synthetic tensors. It fits
a lattice, checks generator gradients and parameter updates, and decodes a
weight matrix. It downloads nothing and writes no checkpoint; it is not a
benchmark.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.demo_lattice
```

See [`src/glvq.py`](src/glvq.py) and [`scripts/demo_lattice.py`](scripts/demo_lattice.py).

## Prepare a full model run

Copy [`config.example.json`](config.example.json) to `config.json` and replace
the placeholder paths. Relative paths resolve from the configuration file's
directory. Use a machine with sufficient GPU memory, host RAM, and storage.

| Field | Meaning |
| --- | --- |
| `profile` | Model profile defined in [`run.py`](run.py) |
| `model` | Local Hugging Face base model directory with weights, config, and native tokenizer |
| `output` | New output directory; reuse it to resume the same run |
| `redpajama` | RedPajama-Data-1T-Sample Arrow cache directory containing `*-train-*.arrow` |
| `c4` | C4 English validation Arrow file |
| `wiki` | WikiText-2 `wikitext-2-raw-v1` test Arrow file |
| `gpus` | Physical GPU indices; default `[0, 1, 2, 3]` |
| `accuracy` | Default `false`; enables HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge |
| `loss_tokens_per_chunk` | Optional KL memory control; default 1024 for Qwen, 256 otherwise |

Obtain models and datasets separately under their own licenses. Model folders
can be prepared with Hugging Face `snapshot_download`; Arrow inputs come from
the corresponding datasets caches. Use each model's native tokenizer. Optional
accuracy evaluation uses the standard lm_eval/Hugging Face caches.

Preparation freezes 1024 PTQ sequences including validation (seed 0), 2048 E2E
sequences (seed 20260811), WikiText-2 test text, and 256 C4 windows. E2E sampling
excludes PTQ token spans using provenance metadata and retains native special
token IDs. PPL uses complete windows, BF16 decoded weights, and FP32 cross-entropy.

## Run and resume

```bash
# Print commands without outputs, model loading, or importing PyTorch.
python run.py config.json --dry-run

# Execute; repeat the same command to resume.
python run.py config.json
```

Stages: data preparation → float PPL → PTQ/block reconstruction → PTQ PPL →
P-only → P-only PPL → merge P-only weights → extended E2E → final PPL → final
layer-checkpoint export. `accuracy=true` adds accuracy evaluation after the
float, PTQ, and extended E2E stages.

P-only runs 1280 optimizer steps and extended E2E runs 2560. Recoverable state
is saved every 64 steps. Stage handoffs and final evaluation use the
**fixed-budget final checkpoint**. `best_model_state_dict.pt` is diagnostic;
the launcher does not select it for the next stage or final reporting.

The launcher locks the output directory, verifies stage outputs, skips completed
stages, resumes PTQ by layer, and resumes E2E from saved optimizer/model state.
It checks frozen data hashes and binds each output directory to the input
paths, configuration, model config, and source. Use a new output directory after
changing these. Do not replace input model/data files in place: the launcher
does not hash every byte of the base model and corpus.

Outputs include `status.json`, per-stage logs and completion records,
`*-ppl/results.json`, and `model/` with merged final quantized layer files.
Insufficient resources cause a failure rather than silently reducing the
budget. At 1024×8192 tokens and hidden size 4096, three FP16 activation buffers
alone occupy roughly 192 GiB of host memory, before model/runtime overhead.
GPU requirements vary; evaluation materializes BF16 weights and full logits.

## Load the final model

Run from this directory with the same base model and package environment:

```python
import torch
from src.modelutils import get_model
from scripts.evaluate import materialize

model = get_model(
    "/path/to/base-model",
    "/path/to/output/model",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).eval()
materialize(model, torch.device("cuda"), torch.bfloat16)
model.to(device="cuda", dtype=torch.bfloat16).eval()
# Load the tokenizer from /path/to/base-model.
```

`model/` contains this implementation's layer checkpoints, not a directory
that can be passed directly to `AutoModelForCausalLM.from_pretrained`. Loading
requires this code and the base model. Layer files use Python pickle: load
only trusted files.

`GLVQLinear` decodes weights and uses floating-point matrix multiplication;
the example above materializes BF16 Linear layers for evaluation. There is no
dedicated low-bit acceleration kernel. **2-bit describes the packed codes**
(two bytes per eight weights). Generators, scales, and other metadata bring the
quantized Linear representation to roughly 2.25 BPW for the release settings,
excluding embeddings and the LM head. Training checkpoints also retain
initialization state and may include module caches; their file size is not a
compact deployment-size measurement. Export/merge does not implement compact
safetensors deployment.

## Tests and validation limits

```bash
python -m pip install -r requirements-dev.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Tests cover lattice encoding/decoding, learned generators and companding,
fixed-code updates, actual PTQ/block reconstruction on tiny Llama/Mistral/Qwen2
models, chunked KL gradients, optimizer resume, export/reload, data preparation
recovery, PPL evaluation, and the portable launcher. Tests create synthetic
model/data fixtures in pytest's temporary directory; no pretrained downloads
are needed. CUDA optimizer tests skip without CUDA.

Release validation on Linux/Python 3.12.3, PyTorch 2.13.0, and Transformers
4.43.4: **62 passed, 3 CUDA-only cases skipped** with GPUs hidden. The synthetic
lattice example and dependency consistency check also passed. The regression
suite ran against a standalone copy of this directory, without imports from
the parent development repository.

Full-model PTQ and both E2E stages have been exercised in the source implementation
for Mistral-7B, Llama-2-7B, and Qwen2.5-7B. Llama-3.1-8B has PTQ coverage;
its complete two-stage E2E workflow remains unvalidated. Full-scale multi-GPU
training was not rerun for this release. CPU tests do not validate CUDA/FSDP
execution or establish full-model accuracy.

## Source and license

The public engineering base is [AQLM](https://github.com/Vahe1994/AQLM), upstream
commit `e79a896`, under Apache-2.0. GLVQ-specific additions implement the
learnable lattice and supporting transforms, fitting, and training flow.
See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for licensing and attribution.

**Bit allocation and PV-Tuning remain outside this release; the core learnable
lattice is implemented and available here.**

## Citation

If you find GLVQ useful in your research, please cite our paper:

```bibtex
@inproceedings{zhang2025learning,
  title={Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression},
  author={Zhang, Xi and Wu, Xiaolin and Wang, Jiamang and Lin, Weisi},
  booktitle={Advances in Neural Information Processing Systems},
  volume={38},
  pages={110672--110698},
  year={2025}
}
```
