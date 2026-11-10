"""Exercise interrupted data writes and recovery without changing frozen samples."""

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import datasets
import pytest
import torch

from scripts import freeze_c4_evaluation, freeze_redpajama_calibration, prepare_data, prepare_e2e_dataset
from run import validate_stage


class Tokenizer:
    name_or_path = "local-test-tokenizer"
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, return_tensors):
        return SimpleNamespace(input_ids=torch.arange(1024).reshape(1, -1))


def prepare_fixture(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    data = tmp_path / "source"
    datasets.Dataset.from_dict({"text": [f"document {i}" for i in range(100)]}).save_to_disk(str(data))
    arrow = next(data.glob("*.arrow"))
    redpajama = tmp_path / "redpajama"
    redpajama.mkdir()
    shutil.copyfile(arrow, redpajama / "sample-train-00000.arrow")
    output = tmp_path / "output" / "data"
    argv = [
        "prepare_data.py",
        "--model",
        str(model),
        "--output",
        str(output),
        "--seqlen",
        "8",
        "--redpajama",
        str(redpajama),
        "--c4",
        str(arrow),
        "--wiki",
        str(arrow),
    ]
    monkeypatch.setattr(prepare_data.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    modules = {
        "freeze_redpajama_calibration.py": freeze_redpajama_calibration,
        "prepare_e2e_dataset.py": prepare_e2e_dataset,
        "freeze_c4_evaluation.py": freeze_c4_evaluation,
    }

    def run_local(command, check):
        assert check
        with monkeypatch.context() as context:
            context.setattr(sys, "argv", command[1:])
            modules[Path(command[1]).name].main()

    monkeypatch.setattr(prepare_data.subprocess, "run", run_local)
    monkeypatch.setattr(sys, "argv", argv)
    return output


def test_prepare_recovers_missing_corrupt_and_partial_artifacts(tmp_path, monkeypatch):
    output = prepare_fixture(tmp_path, monkeypatch)
    prepare_data.main()
    assert validate_stage("prepare-data", output.parent, 1)
    expected = {
        name: torch.load(output / name, weights_only=True)
        for name in ("calibration.pt", "e2e.pt", "c4.pt", "wikitext2.pt")
    }
    # Metadata alone, corrupted bytes, and a truncated dataset must not count as completion.
    (output / "calibration.pt").unlink()
    (output / "c4.pt").write_bytes(b"interrupted tensor write")
    (output / "wikitext2.pt.json").unlink()
    dataset_dir = output / "e2e-hf"
    next(dataset_dir.glob("*.arrow")).unlink()
    (dataset_dir / "keep-this.txt").write_text("preserve unknown contents")
    assert not validate_stage("prepare-data", output.parent, 1)
    prepare_data.main()
    assert validate_stage("prepare-data", output.parent, 1)
    for name, tokens in expected.items():
        assert torch.equal(torch.load(output / name, weights_only=True), tokens)
        assert prepare_data.frozen_tokens_complete(output / name)
    assert prepare_data.e2e_dataset_complete(dataset_dir, 2048, 8)
    backups = list(output.glob(".e2e-hf.incomplete-*"))
    assert len(backups) == 1
    assert (backups[0] / "keep-this.txt").read_text() == "preserve unknown contents"
    # A successful repeat performs no writes or regeneration.
    monkeypatch.setattr(prepare_data.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected rebuild"))
    prepare_data.main()


def test_e2e_save_interruption_never_publishes_incomplete_directory(tmp_path, monkeypatch):
    source = tmp_path / "tokens.pt"
    output = tmp_path / "dataset"
    torch.save(torch.arange(32).reshape(4, 1, 8), source)
    monkeypatch.setattr(sys, "argv", ["prepare_e2e_dataset.py", str(source), str(output)])
    save = datasets.Dataset.save_to_disk

    def interrupted_save(self, path):
        save(self, path)
        raise RuntimeError("simulated interrupted write")

    monkeypatch.setattr(datasets.Dataset, "save_to_disk", interrupted_save)
    with pytest.raises(RuntimeError, match="interrupted write"):
        prepare_e2e_dataset.main()
    assert not output.exists()
    monkeypatch.setattr(datasets.Dataset, "save_to_disk", save)
    prepare_e2e_dataset.main()
    assert datasets.load_from_disk(str(output))["input_ids"] == torch.arange(32).reshape(4, 8).tolist()


@pytest.mark.parametrize("name", ["calibration.pt", "c4.pt", "wikitext2.pt"])
def test_interrupted_tensor_write_preserves_previous_file(tmp_path, monkeypatch, name):
    output = prepare_fixture(tmp_path, monkeypatch)
    prepare_data.main()
    target = output / name
    previous = target.read_bytes()
    expected = torch.load(target, weights_only=True)
    target.with_suffix(".pt.json").unlink()
    save = torch.save

    def interrupted_save(values, destination):
        Path(destination).write_bytes(b"incomplete tensor")
        raise RuntimeError("simulated tensor interruption")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="tensor interruption"):
        prepare_data.main()
    assert target.read_bytes() == previous
    assert not (output / "complete.txt").exists()
    monkeypatch.setattr(torch, "save", save)
    prepare_data.main()
    assert torch.equal(torch.load(target, weights_only=True), expected)


def test_failed_prepare_removes_stale_completion_marker(tmp_path, monkeypatch):
    output = prepare_fixture(tmp_path, monkeypatch)
    output.mkdir(parents=True)
    (output / "complete.txt").write_text("complete\n")

    def interrupted_run(*args, **kwargs):
        raise RuntimeError("simulated subprocess interruption")

    monkeypatch.setattr(prepare_data.subprocess, "run", interrupted_run)
    with pytest.raises(RuntimeError, match="subprocess interruption"):
        prepare_data.main()
    assert not (output / "complete.txt").exists()
