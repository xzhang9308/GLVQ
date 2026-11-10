import pytest
import torch

pytest.importorskip("datasets")
from src.datautils import get_wikitext2


def test_frozen_wikitext_preserves_all_tokens(tmp_path, monkeypatch):
    path = tmp_path / "wiki.pt"
    tokens = torch.arange(33).reshape(1, -1)
    torch.save(tokens, path)
    monkeypatch.setenv("WIKITEXT2_EVAL_FILE", str(path))
    assert torch.equal(get_wikitext2(1, 16, None, eval_mode=True), tokens)


@pytest.mark.parametrize(
    "tokens", [torch.arange(32), torch.zeros(1, 32), torch.zeros(2, 32, dtype=torch.long)]
)
def test_frozen_wikitext_rejects_invalid_tensor(tmp_path, monkeypatch, tokens):
    path = tmp_path / "wiki.pt"
    torch.save(tokens, path)
    monkeypatch.setenv("WIKITEXT2_EVAL_FILE", str(path))
    with pytest.raises(ValueError):
        get_wikitext2(1, 16, None, eval_mode=True)
