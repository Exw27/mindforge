import json
import pytest
from fastapi.testclient import TestClient
from mindforge.server import app


def test_root_ok():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "endpoints" in data


def test_list_models_empty(tmp_path, monkeypatch):
    from mindforge import config
    monkeypatch.setattr(config, "MINDFORGE_DIR", tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    config.ensure_dirs()

    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["object"] == "list"


def test_embeddings_pad_fallback(monkeypatch):
    class DummyTokenizer:
        pad_token = None
        eos_token = None
        def __call__(self, texts, return_tensors=None, padding=False, truncation=False):
            import torch
            if isinstance(texts, list):
                return {"input_ids": torch.tensor([[1,2,3],[4,5,6]])}
            return {"input_ids": torch.tensor([[1,2,3]])}
        def add_special_tokens(self, d):
            self.pad_token = d.get("pad_token")
    class DummyModel:
        def __call__(self, *args, **kwargs):
            import torch
            class O:
                hidden_states = None
                last_hidden_state = torch.randn(2,3,5)
            return O()
    from mindforge import server
    monkeypatch.setattr(server, "_ensure_loaded", lambda model, load_opts=None: (DummyModel(), DummyTokenizer()))
    client = TestClient(app)
    r = client.post("/v1/embeddings", json={"model":"any","input":["a","b"]})
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
