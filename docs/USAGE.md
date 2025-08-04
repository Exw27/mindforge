# Usage

## CLI

Run interactively:
```bash
mindforge run <MODEL>
```
Plain text by default; add `--html` to allow markup.

Manage models:
```bash
mindforge pull <MODEL>
mindforge list
mindforge rm <MODEL>
```

Create from Modelfile:
```bash
mindforge create my-model -f /path/to/Modelfile
```

Templates:
- Put named templates in ~/.mindforge/templates/<name>.j2 and reference with `PARAMS template=<name>` in your Modelfile
- Or place per-model template at ~/.mindforge/models/<model>/template.j2 to auto-apply
- If a GGUF exposes a chat_template, it is used by default unless overridden

## Server

Start server:
```bash
mindforge serve
```

Endpoints:
- GET /v1/models
- POST /v1/completions
- POST /v1/chat/completions
- POST /v1/embeddings

Streaming example:
```bash
curl -N localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt2","messages":[{"role":"user","content":"Say hello"}],"stream":true}'
```
