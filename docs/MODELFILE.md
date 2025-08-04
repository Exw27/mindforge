# Modelfile

Example:
```
FROM org/model
TAGS ["gguf","q4"]
PARAMS device=cpu dtype=float32 quant=Q4_K_M temperature=0.7 top_p=1.0
SYSTEM """
You are a helpful AI assistant.
"""
```

Create:
```bash
mindforge create my-custom -f /path/to/Modelfile
```
