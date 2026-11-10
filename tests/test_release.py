import json
from pathlib import Path
import subprocess
import sys

import pytest

from run import ROOT, commands, exported_layers, load_config, training_complete
import run


def config_file(tmp_path, **changes):
    config = dict(
        profile="qwen25-7b", model="base", output="output", redpajama="rp", c4="c4.arrow", wiki="wiki.arrow"
    )
    config.update(changes)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_portable_dry_run_writes_nothing(tmp_path):
    path = config_file(tmp_path)
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), str(path), "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[export-final]" in result.stdout
    assert "[extended-ppl]" in result.stdout
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("profile", ["mistral7b", "llama2-7b", "llama31-8b", "qwen25-7b"])
def test_fixed_recipe_and_final_checkpoint_handoff(tmp_path, profile):
    config = load_config(config_file(tmp_path, profile=profile))
    plan = dict(commands(config))
    ptq = plan["ptq"]
    assert ptq[ptq.index("--glvq_codebook") + 1] == "e8p"
    assert ptq[ptq.index("--glvq_dimension") + 1] == "8"
    for stage, steps in (("p-only", "1280"), ("extended", "2560")):
        cmd = plan[stage]
        assert cmd[cmd.index("--max_optimizer_steps") + 1] == steps
        assert "--update_codes" not in cmd and "--offload_student_params" not in cmd
    assert "--update_non_quantized_parameters" not in plan["p-only"]
    assert "--update_non_quantized_parameters" in plan["extended"]
    for stage in ("export-p-only", "export-final"):
        cmd = plan[stage]
        assert cmd[cmd.index("--p_finetuned_state_dict") + 1].endswith("final_model_state_dict.pt")
    assert ("--use_fast_tokenizer" in plan["extended"]) == (profile == "qwen25-7b")


@pytest.mark.parametrize(
    "changes",
    [
        {"gpus": [0, 0]},
        {"gpus": [0, 1, 2]},
        {"loss_tokens_per_chunk": 0},
        {"accuracy": "false"},
        {"unknown": 1},
    ],
)
def test_invalid_config_fails_early(tmp_path, changes):
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, **changes))


def test_training_completion_requires_budget_and_final_weights(tmp_path):
    import torch

    torch.save({"total_optimizer_steps": 1280}, tmp_path / "metadata.pt")
    assert not training_complete(tmp_path, 1280)
    (tmp_path / "final_model_state_dict.pt").touch()
    assert training_complete(tmp_path, 1280)
    assert not training_complete(tmp_path, 2560)


def test_export_completeness_does_not_require_ptq_args(tmp_path):
    for name in ("0.pth", "1.pth", "not_quantized_weights.pt"):
        (tmp_path / name).touch()
    assert exported_layers(tmp_path, 2)
    assert not exported_layers(tmp_path, 3)


def prepare_inputs(tmp_path):
    (tmp_path / "base").mkdir()
    (tmp_path / "base/config.json").write_text(json.dumps({"model_type": "qwen2", "num_hidden_layers": 1}))
    (tmp_path / "rp").mkdir()
    (tmp_path / "c4.arrow").touch()
    (tmp_path / "wiki.arrow").touch()
    return config_file(tmp_path)


def test_pipeline_resume_and_config_mismatch(tmp_path, monkeypatch):
    config = prepare_inputs(tmp_path)
    checkpoint = tmp_path / "output/ptq/checkpoint"
    script = (
        f"from pathlib import Path; p=Path({str(checkpoint)!r}); p.mkdir(parents=True); "
        "[(p/n).touch() for n in ('0.pth','args.pt','not_quantized_weights.pt')]"
    )
    monkeypatch.setattr(run, "commands", lambda _: [("ptq", [sys.executable, "-c", script])])
    monkeypatch.setattr(sys, "argv", ["run.py", str(config)])
    run.main()
    assert json.loads((tmp_path / "output/status.json").read_text())["status"] == "complete"
    run.main()  # Re-running the subprocess would fail at mkdir, so success verifies skipping.
    assert len(list((tmp_path / "output/ptq").glob("run-*.log"))) == 1
    changed = json.loads(config.read_text())
    changed["loss_tokens_per_chunk"] = 2048
    config.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Inputs/settings/source changed"):
        run.main()


def test_failed_stage_is_not_marked_complete(tmp_path, monkeypatch):
    config = prepare_inputs(tmp_path)
    monkeypatch.setattr(run, "commands", lambda _: [("ptq", [sys.executable, "-c", "raise SystemExit(2)"])])
    monkeypatch.setattr(sys, "argv", ["run.py", str(config)])
    with pytest.raises(RuntimeError, match="failed"):
        run.main()
    assert not (tmp_path / "output/ptq/complete.json").exists()
    assert json.loads((tmp_path / "output/status.json").read_text())["status"] == "failed"
