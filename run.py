#!/usr/bin/env python3
"""Portable, fixed-recipe GLVQ pipeline. Run --help for configuration."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROFILES = {
    "mistral7b": ("mistral", "MistralDecoderLayer", 8192, 16),
    "llama2-7b": ("llama", "LlamaDecoderLayer", 4096, 128),
    "llama31-8b": ("llama", "LlamaDecoderLayer", 8192, 16),
    "qwen25-7b": ("qwen2", "Qwen2DecoderLayer", 8192, 16),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def complete_layers(path, count):
    path = Path(path)
    return all((path / f"{i}.pth").is_file() for i in range(count)) and all(
        (path / name).is_file() for name in ("args.pt", "not_quantized_weights.pt")
    )


def exported_layers(path, count):
    path = Path(path)
    return (
        all((path / f"{i}.pth").is_file() for i in range(count))
        and (path / "not_quantized_weights.pt").is_file()
    )


def training_complete(path, steps):
    import torch

    path = Path(path)
    if not all((path / name).is_file() for name in ("metadata.pt", "final_model_state_dict.pt")):
        return False
    return (
        int(torch.load(path / "metadata.pt", map_location="cpu", weights_only=True)["total_optimizer_steps"])
        == steps
    )


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    required = {"profile", "model", "output", "redpajama", "c4", "wiki"}
    allowed = required | {"gpus", "loss_tokens_per_chunk", "accuracy"}
    if required - config.keys() or config.keys() - allowed:
        raise ValueError(f"Required keys: {sorted(required)}; allowed keys: {sorted(allowed)}")
    if config["profile"] not in PROFILES:
        raise ValueError(f"profile must be one of {list(PROFILES)}")
    for key in ("model", "output", "redpajama", "c4", "wiki"):
        p = Path(config[key]).expanduser()
        config[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    config.setdefault("gpus", [0, 1, 2, 3])
    gpus = config["gpus"]
    if (
        not isinstance(gpus, list)
        or len(gpus) not in (1, 2, 4, 8)
        or any(type(i) is not int or i < 0 for i in gpus)
        or len(set(gpus)) != len(gpus)
    ):
        raise ValueError("gpus must contain 1, 2, 4 or 8 distinct nonnegative GPU indices")
    config.setdefault("loss_tokens_per_chunk", 1024 if config["profile"] == "qwen25-7b" else 256)
    if type(config["loss_tokens_per_chunk"]) is not int or config["loss_tokens_per_chunk"] <= 0:
        raise ValueError("loss_tokens_per_chunk must be a positive integer")
    config.setdefault("accuracy", False)
    if type(config["accuracy"]) is not bool:
        raise ValueError("accuracy must be true or false")
    return config


def commands(config):
    """Build the full plan without importing torch or accessing model/data files."""
    _, block, seqlen, val = PROFILES[config["profile"]]
    output = Path(config["output"])
    data = output / "data"
    ptq = output / "ptq/checkpoint"
    p_only = output / "p-only/checkpoint"
    extended = output / "extended/checkpoint"
    exported = output / "p-only-export"
    final = output / "model"
    py = sys.executable
    plan = []

    def add(name, cmd):
        plan.append((name, list(map(str, cmd))))

    def evaluate(name, quantized=None, state=None):
        for mode in ["ppl", "accuracy"] if config["accuracy"] and name != "p-only" else ["ppl"]:
            cmd = [
                py,
                "-m",
                "scripts.evaluate",
                "--model",
                config["model"],
                "--data",
                data,
                "--seqlen",
                seqlen,
                "--output",
                output / f"{name}-{mode}/results.json",
                "--mode",
                mode,
            ]
            if quantized:
                cmd += ["--quantized", quantized]
            if state:
                cmd += ["--state", state]
            add(f"{name}-{mode}", cmd)

    def export(name, source, state, destination):
        add(
            name,
            [
                py,
                "export.py",
                "--base_model",
                config["model"],
                "--quantized_model",
                source,
                "--p_finetuned_state_dict",
                state,
                "--load_dtype",
                "bfloat16",
                "--save",
                destination,
            ],
        )

    add(
        "prepare-data",
        [
            py,
            "-m",
            "scripts.prepare_data",
            "--model",
            config["model"],
            "--output",
            data,
            "--seqlen",
            seqlen,
            "--redpajama",
            config["redpajama"],
            "--c4",
            config["c4"],
            "--wiki",
            config["wiki"],
        ],
    )
    evaluate("float")
    recipe = json.loads((ROOT / "recipe.json").read_text())
    ptq_cmd = [
        py,
        "main.py",
        config["model"],
        data / "calibration.pt",
        "--devices",
        *[f"cuda:{i}" for i in range(len(config["gpus"]))],
        "--model_seqlen",
        seqlen,
        "--val_size",
        val,
        "--save",
        ptq,
    ]
    for key, value in recipe.items():
        if value is True:
            ptq_cmd += ["--" + key]
        elif value is not False:
            ptq_cmd += ["--" + key, value]
    add("ptq", ptq_cmd)
    evaluate("ptq", ptq)
    for name, source, destination, epochs in (
        ("p-only", ptq, p_only, 5),
        ("extended", exported, extended, 10),
    ):
        if name == "extended":
            export("export-p-only", ptq, p_only / "final_model_state_dict.pt", exported)
        cmd = [
            py,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={len(config['gpus'])}",
            "finetune.py",
            "--base_model",
            config["model"],
            "--quantized_model",
            source,
            "--model_seqlen",
            seqlen,
            "--block_type",
            block,
            "--load_dtype",
            "bfloat16",
            "--amp_dtype",
            "bfloat16",
            "--dataset_name",
            data / "e2e-hf",
            "--seed",
            42,
            "--update_codebooks_and_scales",
            "--lr",
            "1e-6",
            "--adam_beta1",
            0.9,
            "--adam_beta2",
            0.95,
            "--batch_size",
            8,
            "--microbatch_size",
            1,
            "--max_epochs",
            epochs,
            "--max_optimizer_steps",
            epochs * 256,
            "--max_grad_norm",
            1.0,
            "--gradient_checkpointing",
            "--loss_tokens_per_chunk",
            config["loss_tokens_per_chunk"],
            "--print_every_steps",
            8,
            "--eval_every_steps",
            256,
            "--save_every_steps",
            64,
            "--eval_before_training",
            "--eval_datasets",
            "wikitext2",
            "--keep_best_model",
            "--save_final_model",
            "--save",
            destination,
            "--attn_implementation",
            "sdpa",
            "--offload_teacher_params",
            "--offload_optimizer",
            "--limit_all_gathers",
            "--num_workers",
            2,
        ]
        if name == "extended":
            cmd += ["--update_non_quantized_parameters"]
        if config["profile"] == "qwen25-7b":
            cmd += ["--use_fast_tokenizer"]
        add(name, cmd)
        evaluate(name, source, destination / "final_model_state_dict.pt")
    export("export-final", exported, extended / "final_model_state_dict.pt", final)
    return plan


def validate_stage(name, output, layers):
    if name == "prepare-data":
        required_exist = all(
            (output / "data" / p).is_file()
            for p in (
                "calibration.pt",
                "calibration.pt.json",
                "e2e.pt",
                "e2e.pt.json",
                "e2e-hf/state.json",
                "wikitext2.pt",
                "c4.pt",
                "complete.txt",
            )
        )
        if not required_exist:
            return False
        from scripts.prepare_data import e2e_dataset_complete, frozen_tokens_complete

        data = output / "data"
        if not all(
            frozen_tokens_complete(data / f"{name}.pt") for name in ("calibration", "e2e", "wikitext2", "c4")
        ):
            return False
        shape = json.loads((data / "e2e.pt.json").read_text())["shape"]
        return len(shape) == 3 and e2e_dataset_complete(data / "e2e-hf", 2048, shape[-1])
    if name == "ptq":
        return complete_layers(output / "ptq/checkpoint", layers)
    if name in ("p-only", "extended"):
        return training_complete(output / name / "checkpoint", 1280 if name == "p-only" else 2560)
    if name.startswith("export-"):
        return exported_layers(output / ("model" if name == "export-final" else "p-only-export"), layers)
    path = output / name / "results.json"
    if not path.is_file():
        return False
    result = json.loads(path.read_text())
    return not result.get("smoke_only", True) and bool(result.get("metrics") or result.get("accuracy"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON config; relative paths resolve beside this file")
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without loading models or writing files"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    plan = commands(config)
    if args.dry_run:
        for name, cmd in plan:
            print(f"[{name}] {shlex.join(cmd)}")
        return
    for key in ("model", "redpajama", "c4", "wiki"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"Missing {key}: {config[key]}")
    model_config = json.loads((Path(config["model"]) / "config.json").read_text())
    if model_config["model_type"] != PROFILES[config["profile"]][0]:
        raise ValueError("model_type does not match profile")
    layers = model_config["num_hidden_layers"]
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pipeline is already using this output directory") from None
        # Bind resume to inputs, settings and exact release source, not a stale success marker.
        digest = hashlib.sha256()
        for path in sorted(
            [*ROOT.glob("*.py"), *ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")]
        ):
            if output != ROOT and output in path.parents:
                continue
            if ".venv" in path.parts or "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
        digest.update((ROOT / "recipe.json").read_bytes())
        identity = {"config": config, "source_sha256": digest.hexdigest(), "model_config": model_config}
        saved = output / "run.json"
        if saved.exists() and json.loads(saved.read_text()) != identity:
            raise ValueError(
                "Inputs/settings/source changed. Use a new output directory to avoid mixing checkpoints."
            )
        if not saved.exists() and any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("Output is not empty and has no run.json; choose a fresh output directory")
        write_json(saved, identity)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(map(str, config["gpus"])),
            PYTHONPATH=str(ROOT),
            PYTHONUNBUFFERED="1",
            TOKENIZERS_PARALLELISM="false",
            WIKITEXT2_EVAL_FILE=str(output / "data/wikitext2.pt"),
            C4_EVAL_FILE=str(output / "data/c4.pt"),
        )
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        state = {"status": "running", "stages": {}}
        child = None

        def stop(signum, frame):
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
            state["status"] = "interrupted"
            write_json(output / "status.json", state)
            raise SystemExit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        for name, cmd in plan:
            stage = output / name
            marker = stage / "complete.json"
            if marker.is_file() and validate_stage(name, output, layers):
                print(f"SKIP {name}: complete", flush=True)
                state["stages"][name] = "complete"
                continue
            stage.mkdir(parents=True, exist_ok=True)
            state.update(current_stage=name, updated_at=time.time())
            state["stages"][name] = "running"
            write_json(output / "status.json", state)
            write_json(stage / "command.json", cmd)
            log = stage / f"run-{time.time_ns()}.log"
            print(f"START {name}; log: {log}", flush=True)
            try:
                with log.open("w") as stream:
                    child = subprocess.Popen(
                        cmd,
                        cwd=ROOT,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    rc = child.wait()
                    child = None
                if rc != 0 or not validate_stage(name, output, layers):
                    raise RuntimeError(f"{name} failed or produced incomplete output; see {log}")
            except Exception:
                state["status"] = "failed"
                state["stages"][name] = "failed"
                write_json(output / "status.json", state)
                raise
            write_json(marker, {"completed_at": time.time(), "command": cmd})
            state["stages"][name] = "complete"
        state.update(status="complete", updated_at=time.time())
        write_json(output / "status.json", state)
        print(f"Complete. Quantized model: {output / 'model'}", flush=True)


if __name__ == "__main__":
    main()
