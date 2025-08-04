import sys
import os
import shutil
import json
from .cli import get_parser
from .config import ensure_dirs, MODELS_DIR
from .models import download_model, load_model
from .modelfile import parse_modelfile, save_custom_model, load_custom_model_config
from llama_cpp import Llama
from . import server

def main():
    """The main entry point for the application."""
    ensure_dirs()
    parser = get_parser()
    args = parser.parse_args()

    if args.command == "run":
        run(args.model, args.prompt, html=getattr(args, 'html', False))
    elif args.command == "pull":
        pull(args.model)
    elif args.command == "list":
        list_models()
    elif args.command == "rm":
        remove_model(args.model)
    elif args.command == "serve":
        server.serve_global()
    elif args.command == "create":
        create_model(args.model, args.file)
    else:
        parser.print_help()

def run(model_name_str, prompt, html=False):
    """Runs the specified model."""
    try:
        interactive = prompt is None
        custom_config = load_custom_model_config(model_name_str)
        if custom_config:
            base_model_name = custom_config.get("from")
            repo_id, quant_name = parse_model_name(base_model_name)
        else:
            repo_id, quant_name = parse_model_name(model_name_str)

        download_model(repo_id, quant_name)
        model, tokenizer = load_model(model_name_str, quant_name)

        def _maybe_strip(s: str) -> str:
            if html:
                return s
            import re
            return re.sub(r"<[^>]+>", "", s)
        def respond_once(p: str):
            if isinstance(model, Llama):
                if custom_config and "system" in custom_config:
                    full = f"<s>[INST] <<SYS>>\n{custom_config['system']}\n<</SYS>>\n\n{p} [/INST]"
                else:
                    full = p
                out = model(full, max_tokens=128)
                return _maybe_strip(out["choices"][0]["text"])
            else:
                inputs = tokenizer(p, return_tensors="pt")
                outputs = model.generate(**inputs, max_length=50)
                return _maybe_strip(tokenizer.decode(outputs[0], skip_special_tokens=True))

        if interactive:
            print(f"Running model '{model_name_str}'. Press Ctrl+C to exit.")
            try:
                while True:
                    user = input("> ").strip()
                    if not user:
                        continue
                    print(respond_once(user))
            except KeyboardInterrupt:
                print("\nExiting...")
        else:
            print(respond_once(prompt))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def pull(model_name_str):
    """Pulls a model from the Hugging Face Hub."""
    try:
        repo_id, quant_name = parse_model_name(model_name_str)
        download_model(repo_id, quant_name)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def list_models():
    """Lists all downloaded models."""
    ensure_dirs()
    print("The following models are available:")
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            print(f"- {model_dir.name}")
    for custom_model_file in MODELS_DIR.glob("*.json"):
        print(f"- {custom_model_file.stem} (custom)")


def remove_model(model_name_str):
    """Removes a downloaded model."""
    try:
        repo_id, _ = parse_model_name(model_name_str)
        model_path = MODELS_DIR / repo_id.replace("/", "_")
        
        if not model_path.exists():
            print(f"Model '{model_name_str}' not found.")
            return

        confirm = input(f"Are you sure you want to remove '{model_name_str}'? (y/N) ")
        if confirm.lower() == 'y':
            shutil.rmtree(model_path)
            print(f"Model '{model_name_str}' removed successfully.")
        else:
            print("Removal cancelled.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def create_model(model_name: str, filepath: str):
    """Creates a custom model from a Modelfile."""
    try:
        config = parse_modelfile(filepath)
        save_custom_model(model_name, config)
        model_dir = MODELS_DIR / model_name.replace("/","_")
        model_dir.mkdir(parents=True, exist_ok=True)
        meta = {}
        if "tags" in config:
            meta["tags"] = config["tags"]
        if "params" in config:
            mp = config["params"]
            for k in ("device","dtype","quant","temperature","top_p"):
                if k in mp:
                    meta[k] = mp[k]
        (model_dir/"config.json").write_text(json.dumps(meta))
        print(f"Custom model '{model_name}' created successfully.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def parse_model_name(model_name_str):
    """Parses a model name into a repo_id and quant_name."""
    name = model_name_str.strip().replace("\\", "/")
    if name.startswith("../") or name.startswith("./") or name.startswith("/"):
        raise ValueError("Invalid model name")
    repo_id = name
    quant_name = None
    if ":" in name:
        repo_id, quant_name = name.split(":", 1)
    return repo_id, quant_name

if __name__ == "__main__":
    main()