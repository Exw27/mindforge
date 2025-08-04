import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
from .config import MODELS_DIR
from .modelfile import load_custom_model_config

def get_model_path(repo_id):
    """Returns the path to a model's directory."""
    return MODELS_DIR / repo_id.replace("/", "_")

def download_model(repo_id, quant_name=None):
    """Downloads a model and tokenizer from the Hugging Face Hub."""
    model_path = get_model_path(repo_id)
    model_path.mkdir(parents=True, exist_ok=True)
    
    if quant_name:
        gguf_files = list(model_path.glob(f"*{quant_name}*.gguf"))
        if gguf_files:
            print(f"Model '{repo_id}:{quant_name}' already exists locally.")
            return
    elif model_path.exists() and any(model_path.iterdir()):
         print(f"Model '{repo_id}' already exists locally.")
         return

    print(f"Downloading model '{repo_id}'...")
    if "gguf" in repo_id.lower() or quant_name:
        repo_files = list_repo_files(repo_id)
        gguf_files = [f for f in repo_files if f.endswith(".gguf")]

        if not gguf_files:
            raise FileNotFoundError(f"No GGUF files found in repo '{repo_id}'")

        file_to_download = None
        if quant_name:
            for f in gguf_files:
                if quant_name.lower() in f.lower():
                    file_to_download = f
                    break
            if not file_to_download:
                 raise FileNotFoundError(f"Quantization '{quant_name}' not found in repo '{repo_id}'")
        else:
            file_to_download = gguf_files[0]
            print(f"No specific GGUF file requested, downloading first available: {file_to_download}")

        hf_hub_download(repo_id=repo_id, filename=file_to_download, local_dir=model_path)
    else:
        snapshot_download(repo_id=repo_id, local_dir=model_path)

    print(f"Model '{repo_id}' downloaded successfully.")

def _load_base_model(repo_id, quant_name=None, device: str | None = None, dtype: str | None = None):
    model_path = get_model_path(repo_id)
    if not model_path.exists():
        raise FileNotFoundError(f"Model '{repo_id}' not found locally. Please download it first.")
    print(f"Loading model '{repo_id}'...")
    if quant_name:
        glob_pattern = f"*{quant_name}*.gguf"
    else:
        glob_pattern = "*.gguf"
    gguf_files = list(model_path.glob(glob_pattern))
    if gguf_files:
        model_file_path = gguf_files[0]
        if len(gguf_files) > 1 and not quant_name:
            print(f"Multiple GGUF files found, loading the first one: {model_file_path.name}")
        kwargs = dict(model_path=model_file_path.as_posix(), verbose=False, logits_all=False)
        if device and str(device).lower().startswith("cuda"):
            kwargs.update(n_gpu_layers=9999)
        level = os.getenv("LLAMA_LOG_LEVEL")
        if level is not None:
            try:
                kwargs.update(verbosity=int(level))
            except Exception:
                pass
        import contextlib, os as _os
        def _mk():
            return Llama(**kwargs)
        quiet = _os.getenv("MINDFORGE_LLAMA_QUIET", "1") != "0"
        if quiet:
            with contextlib.redirect_stderr(open(_os.devnull, "w")):
                llm = _mk()
        else:
            llm = _mk()
        return llm, None
    else:
        torch_dtype = None
        if dtype:
            try:
                import torch
                torch_dtype = getattr(torch, dtype)
            except Exception:
                torch_dtype = None
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                try:
                    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                except Exception:
                    pass
        if device and hasattr(model, 'to'):
            try:
                import torch
                if str(device).startswith('cuda') and not torch.cuda.is_available():
                    pass
                else:
                    model = model.to(device)
            except Exception:
                pass
        return model, tokenizer

def load_model(model_name, quant_name=None, device: str | None = None, dtype: str | None = None):
    """Loads a model and tokenizer from the local cache."""
    custom_config = load_custom_model_config(model_name)
    if custom_config:
        base_model_name = custom_config.get("from")
        if not base_model_name:
            raise ValueError("Invalid custom model: 'from' directive is missing.")
        from .main import parse_model_name
        repo_id, qn = parse_model_name(base_model_name)
        model, tokenizer = _load_base_model(repo_id, qn, device=device, dtype=dtype)
        if isinstance(model, Llama) and "system" in custom_config:
            try:
                model.system_message = custom_config["system"]
            except Exception:
                pass
        return model, tokenizer

    return _load_base_model(model_name, quant_name, device=device, dtype=dtype)
