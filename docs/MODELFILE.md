# Modelfile

Example:
```
FROM org/model
TAGS ["gguf","q4"]
PARAMS device=cpu dtype=float32 quant=Q4_K_M temperature=0.7 top_p=1.0 template=chatml
SYSTEM """
You are a helpful AI assistant.
"""
```

Templates:
- Place a file at ~/.mindforge/templates/chatml.j2 to be used by `template=chatml`
- Or create ~/.mindforge/models/<model>/template.j2 for a per-model override

Create:
```bash
mindforge create my-custom -f /path/to/Modelfile
```
