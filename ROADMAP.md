# Roadmap

## Done
- FastAPI server with OpenAI-compatible endpoints
- SSE streaming for chat
- GGUF via llama-cpp; quant selection with repo:quant
- Modelfile parser (FROM, TAGS, PARAMS, SYSTEM)
- JSON mode and basic tool scaffolding
- Per-model defaults via config.json and Modelfile
- Device/dtype/quant overrides and env defaults
- Installer script with venv + shim
- Default plain text output; --html to allow markup
- Quiet llama.cpp logs by default

## Next
- CI: lint/typecheck/tests
- Model registry and metadata caching
- Auth/rate limiting for server mode
- Persistent chat sessions
- Tool execution loop with safety sandbox
- GGUF embeddings or disable per-model at /v1/embeddings
- Prebuilt wheels and release install URL
