Done tasks

• Implemented global FastAPI server with OpenAI-compatible routes: /v1/models, /v1/completions, /v1/chat/completions, /v1/embeddings
• Added SSE streaming for chat; robust HF streaming via TextIteratorStreamer
• Embeddings endpoint fixed for GPT-2 by setting pad_token to eos_token
• CLI run opens interactive REPL when no prompt
• GGUF model support via llama_cpp with quant selection syntax repo:quant
• Modelfile parser supports FROM and multiline SYSTEM blocks
• JSON mode scaffolding in chat via response_format.type=json_object, with post-processing to extract first valid JSON object
• Tools scaffolding in chat (tools, tool_choice) prompt injection
• Per-model defaults and tags via config.json and Modelfile TAGS/PARAMS
• Device/dtype/quant load options with per-request overrides and env defaults

Not working or deferred

• Sessions/history endpoints (/v1/sessions/:id) not active in installed CLI server; works only when running server module directly
• Tool calling execution loop not implemented; only prompt scaffolding
• Sessions/history endpoints not wired in CLI server build
• Authentication and rate limiting not implemented
• Persistent storage for sessions not implemented
