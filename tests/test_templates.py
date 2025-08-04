import os
from mindforge.templates import render_chat

class Dummy:
    def __init__(self, meta=None):
        self.metadata = meta or {}


def test_render_fallback_without_jinja(monkeypatch):
    monkeypatch.setenv("MINDFORGE_LLAMA_QUIET", "1")
    m = Dummy()
    msgs = [{"role":"user","content":"Hello"}]
    out = render_chat("gpt2", m, msgs, system="sys")
    assert out.startswith("system: sys\nuser: Hello\nassistant:")


def test_gguf_chat_template_used():
    tpl = "{{# if system }}<s>[INST] <<SYS>>\n{{ system }}\n<</SYS>>\n{{/ if }}{% for m in messages %}{% if m.role=='user' %}{{ m.content }}{% endif %}{% endfor %}"
    # This template isn't valid jinja; ensure we don't crash and fallback occurs
    m = Dummy(meta={"chat_template": tpl})
    msgs = [{"role":"user","content":"Q?"}]
    out = render_chat("foo", m, msgs)
    assert "assistant:" in out


def test_named_template_file(tmp_path, monkeypatch):
    from mindforge.templates import TEMPLATES_DIR
    (TEMPLATES_DIR).mkdir(parents=True, exist_ok=True)
    name = "simple"
    (TEMPLATES_DIR / f"{name}.j2").write_text("{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}assistant:")
    m = Dummy()
    msgs = [{"role":"user","content":"Hi"}]
    out = render_chat("bar", m, msgs, params={"template": name})
    assert out.strip().endswith("assistant:")
