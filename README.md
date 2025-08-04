# Mindforge

A simple, Ollama-like tool to run Hugging Face and GGUF language models locally. Provides an OpenAI-compatible server.

## Installation

Quick install:

```bash
curl -fsSL 'https://raw.githubusercontent.com/Exw27/mindforge/main/install.sh?ts=now' | sh -s -- --force-deps
```

From source:

```bash
sh ./install.sh
```

## Usage

### Run interactively
```bash
mindforge run <MODEL_NAME>
```
If no prompt is provided, an interactive REPL opens.

Plain text output by default. Use --html to allow markup.

### Serve OpenAI-compatible API
```bash
mindforge serve
```
Server runs at http://127.0.0.1:8000

Set LLAMA_LOG_LEVEL and MINDFORGE_LLAMA_QUIET=0 to adjust llama.cpp logging.

Endpoints:
- GET /v1/models
- POST /v1/completions
- POST /v1/chat/completions
- POST /v1/embeddings

Streaming:
```bash
curl -N localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt2","messages":[{"role":"user","content":"Say hello"}],"stream":true}'
```

Embeddings:
```bash
curl -s localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt2","input":["hello","world"]}'
```

### Custom models (Modelfile)
```
FROM <base_model_name>
TAGS ["gguf","q4"]
PARAMS device=cpu dtype=float32 quant=Q4_K_M temperature=0.7 top_p=1.0
SYSTEM """
You are a helpful AI assistant.
"""
```
Create:
```bash
mindforge create my-custom-model -f /path/to/Modelfile
```

### GGUF models
Select quant by suffix:
```bash
mindforge run unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M
```

## JSON mode and tools (experimental)
- JSON mode: set response_format {"type":"json_object"} to nudge JSON-only output
- Tools scaffolding: pass tools and tool_choice; current build injects specs into prompt

## Device/dtype/quant options
Defaults via env:
- MINDFORGE_DEVICE, MINDFORGE_DTYPE, MINDFORGE_QUANT
Per-request override:
```json
{"load_opts": {"device": "cpu|cuda|mps", "dtype": "float16", "quant": "Q4_K_M"}}
```

## Per-model defaults and tags
Create MODELS_DIR/<model>/config.json with:
```json
{
  "device": "cpu",
  "dtype": "float32",
  "quant": null,
  "temperature": 0.7,
  "top_p": 1.0,
  "tags": ["gguf","q4"]
}
```
GET /v1/models includes metadata for each model when config.json exists.

## Model management
```bash
mindforge pull <MODEL_NAME>
mindforge list
mindforge rm <MODEL_NAME>
```

Notes:
- GGUF embeddings not supported in /v1/embeddings
- Sessions/history endpoints are WIP when running via CLI; available when serving the module directly
