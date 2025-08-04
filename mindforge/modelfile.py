import json
from pathlib import Path
from .config import MODELS_DIR

def get_custom_model_path(model_name: str) -> Path:
    """Returns the path to a custom model's configuration file."""
    return MODELS_DIR / f"{model_name}.json"

def parse_modelfile(filepath: str) -> dict:
    """Parses a Modelfile and returns a dictionary of its contents."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Modelfile not found at: {filepath}")

    lines = path.read_text().splitlines()

    config: dict = {}
    in_system = False
    buf: list[str] = []
    quote = None

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not in_system:
            if stripped.upper().startswith("FROM "):
                config["from"] = stripped.split(" ", 1)[1].strip()
                continue
            if stripped.upper().startswith("TAGS "):
                rest = stripped.split(" ", 1)[1].strip()
                try:
                    config["tags"] = json.loads(rest)
                except Exception:
                    raise ValueError("TAGS must be a JSON array, e.g. TAGS [\"gguf\"]")
                continue
            if stripped.upper().startswith("PARAMS "):
                rest = stripped.split(" ", 1)[1].strip()
                parts = rest.split()
                params = {}
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        params[k] = v
                config["params"] = params
                continue
            if stripped.upper().startswith("SYSTEM "):
                rest = stripped.split(" ", 1)[1]
                if rest.startswith('"""'):
                    quote = '"""'
                    rest = rest[3:]
                elif rest.startswith("'''"):
                    quote = "'''"
                    rest = rest[3:]
                else:
                    raise ValueError("SYSTEM prompt must be enclosed in triple quotes.")
                in_system = True
                # if closing on same line
                if rest.endswith(quote):
                    content = rest[: -3]
                    config["system"] = content.strip()
                    in_system = False
                    quote = None
                else:
                    buf.append(rest)
                continue
            continue
        else:
            if stripped.endswith(quote):
                # capture up to before closing
                end = line[: len(line) - 3]
                buf.append(end)
                config["system"] = "\n".join(buf).strip()
                buf.clear()
                in_system = False
                quote = None
            else:
                buf.append(line)

    if in_system:
        raise ValueError("Unterminated SYSTEM triple-quoted block.")

    if "from" not in config:
        raise ValueError("Modelfile must contain a FROM directive.")

    return config

def save_custom_model(model_name: str, config: dict):
    """Saves a custom model's configuration to a file."""
    path = get_custom_model_path(model_name)
    path.write_text(json.dumps(config, indent=4))

def load_custom_model_config(model_name: str) -> dict | None:
    """Loads a custom model's configuration if it exists."""
    path = get_custom_model_path(model_name)
    if path.exists():
        return json.loads(path.read_text())
    return None
