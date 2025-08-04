import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from jinja2 import Environment, FileSystemLoader, Template
except Exception:
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore
    Template = None  # type: ignore

from .config import MODELS_DIR

TEMPLATES_DIR = Path.home() / ".mindforge" / "templates"

_env: Optional[Environment] = None
_cache: Dict[str, Template] = {}


def _get_env() -> Optional[Environment]:
    global _env
    if _env is not None:
        return _env
    if Environment is None:
        return None
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    _env = Environment(loader=FileSystemLoader([TEMPLATES_DIR]))
    _env.filters.setdefault("tojson", lambda v: __import__("json").dumps(v))
    return _env


def _load_per_model_template(model_name: str) -> Optional[Template]:
    env = _get_env()
    if env is None:
        return None
    path = MODELS_DIR / model_name.replace("/", "_") / "template.j2"
    if path.exists():
        return env.from_string(path.read_text())
    return None


def _load_named_template(name: Optional[str]) -> Optional[Template]:
    if not name:
        return None
    env = _get_env()
    if env is None:
        return None
    try:
        return env.get_template(f"{name}.j2")
    except Exception:
        return None


def _gguf_chat_template(model: Any) -> Optional[str]:
    try:
        # llama.cpp python exposes metadata in model.metadata if available
        meta = getattr(model, "metadata", None)
        if isinstance(meta, dict):
            for k in ("chat_template", "tokenizer.chat_template", "llama.chat_template"):
                if k in meta and isinstance(meta[k], str) and meta[k].strip():
                    return meta[k]
    except Exception:
        pass
    return None


def get_renderer(model_name: str, model: Any, params: Optional[Dict[str, Any]] = None):
    key = f"{model_name}"
    if key in _cache:
        return _cache[key]
    # selection order: per-model file > PARAMS.template > gguf metadata > None
    tpl = _load_per_model_template(model_name)
    if tpl is None and params is not None:
        tpl = _load_named_template(params.get("template"))
    if tpl is None:
        txt = _gguf_chat_template(model)
        if txt:
            env = _get_env()
            if env is not None:
                try:
                    tpl = env.from_string(txt)
                except Exception:
                    tpl = None
    if tpl is not None:
        _cache[key] = tpl
        return tpl
    return None


def render_chat(model_name: str, model: Any, messages: List[Dict[str, str]], system: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None, response_format: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> str:
    tpl = get_renderer(model_name, model, params)
    if tpl is None:
        base = []
        if system:
            base.append(f"system: {system}")
        for m in messages:
            base.append(f"{m.get('role')}: {m.get('content')}")
        base.append("assistant:")
        return "\n".join(base)
    ctx = {
        "system": system,
        "messages": messages,
        "tools": tools or [],
        "response_format": response_format,
        "params": params or {},
        "model_name": model_name,
    }
    return tpl.render(**ctx)
