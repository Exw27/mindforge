from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from .models import download_model, load_model, get_model_path
import torch
from .config import MODELS_DIR
from llama_cpp import Llama
import uvicorn
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class ToolSpec(BaseModel):
    type: str
    function: ToolFunction

class ResponseFormat(BaseModel):
    type: str = "text"
    json_schema: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    response_format: Optional[ResponseFormat] = None
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[Any] = None
    session_id: Optional[str] = None
    load_opts: Optional['LoadOpts'] = None

class LoadOpts(BaseModel):
    device: Optional[str] = None
    dtype: Optional[str] = None
    quant: Optional[str] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    load_opts: Optional['LoadOpts'] = None

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "user"
    metadata: Optional[Dict[str, Any]] = None

class EmbeddingsRequest(BaseModel):
    model: str
    input: List[str]
    load_opts: Optional['LoadOpts'] = None

MODEL_CACHE: Dict[str, Any] = {}
SESSIONS: Dict[str, List[ChatMessage]] = {}

def _ensure_loaded(model_name_str: str, load_opts: Optional[LoadOpts] = None):
    key = model_name_str
    model_dir = MODELS_DIR / model_name_str.replace("/", "_")
    defaults = {}
    cfg = model_dir / "config.json"
    if cfg.exists():
        try:
            defaults = json.loads(cfg.read_text())
        except Exception:
            defaults = {}
    eff_device = (getattr(load_opts, 'device', None) or defaults.get('device') or os.getenv('MINDFORGE_DEVICE'))
    if not eff_device:
        try:
            if torch.cuda.is_available():
                eff_device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                eff_device = 'mps'
            else:
                eff_device = 'cpu'
        except Exception:
            eff_device = 'cpu'
    eff_dtype = (getattr(load_opts, 'dtype', None) or defaults.get('dtype') or os.getenv('MINDFORGE_DTYPE'))
    eff_quant = (getattr(load_opts, 'quant', None) or defaults.get('quant'))
    eff_quant_env = os.getenv('MINDFORGE_QUANT')
    key += f"|{eff_device or ''}|{eff_dtype or ''}|{(eff_quant or '')}|{eff_quant_env or ''}"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    from .main import parse_model_name
    repo_id, quant = parse_model_name(model_name_str)
    quant = eff_quant or quant or os.getenv('MINDFORGE_QUANT')
    download_model(repo_id, quant)
    model, tokenizer = load_model(repo_id if quant is None else repo_id, quant, device=eff_device, dtype=eff_dtype)
    MODEL_CACHE[key] = (model, tokenizer)
    return MODEL_CACHE[key]

@app.get("/v1/models")
async def list_models():
    items = []
    for p in MODELS_DIR.iterdir():
        if p.is_dir():
            meta = None
            cfg = p / "config.json"
            if cfg.exists():
                try:
                    meta = json.loads(cfg.read_text())
                except Exception:
                    meta = None
            items.append(ModelCard(id=p.name, metadata=meta))
    return {"object": "list", "data": [m.dict() for m in items]}

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    model, tokenizer = _ensure_loaded(req.model, getattr(req, 'load_opts', None))
    if isinstance(model, Llama):
        out = model(req.prompt, max_tokens=req.max_tokens or 512, temperature=req.temperature or 0.7, top_p=req.top_p or 1.0)
        text = out["choices"][0]["text"]
    else:
        inputs = tokenizer(req.prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=(inputs["input_ids"].shape[1] + (req.max_tokens or 512)))
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "id": "cmpl-local",
        "object": "text_completion",
        "model": req.model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    messages = req.messages
    if getattr(req, 'session_id', None):
        if req.session_id not in SESSIONS:
            SESSIONS[req.session_id] = []
        SESSIONS[req.session_id].extend(messages)
        history = SESSIONS[req.session_id]
    else:
        history = messages
    sys_prefix = ""
    if getattr(req, 'response_format', None) and req.response_format.type == 'json_object':
        sys_prefix = "system: You must reply with a valid, minified JSON object only. No prose.\n"
    tools_prefix = ""
    if getattr(req, 'tools', None):
        tool_descriptions = []
        for t in req.tools:
            if t.type == 'function' and t.function and t.function.name:
                tool_descriptions.append({"name": t.function.name, "description": t.function.description or "", "parameters": t.function.parameters or {}})
        tools_prefix = f"system: Available tools (JSON): {json.dumps(tool_descriptions)}\n"
    base_prefix = sys_prefix + tools_prefix
    def render_history(msgs: List[ChatMessage]):
        try:
            from .templates import render_chat
            msg_dicts = [{"role": m.role, "content": m.content} for m in msgs]
            sys_msg = None
            if base_prefix:
                sys_msg = base_prefix.strip()
            return render_chat(req.model, model, msg_dicts, system=sys_msg, tools=[t.dict() for t in (req.tools or [])], response_format=(req.response_format.dict() if req.response_format else None), params={"temperature": req.temperature, "top_p": req.top_p})
        except Exception:
            return base_prefix + "\n".join([f"{m.role}: {m.content}" for m in msgs]) + "\nassistant:"
    model, tokenizer = _ensure_loaded(req.model, getattr(req, 'load_opts', None))

    if not req.stream:
        def run_model(p: str):
            if isinstance(model, Llama):
                out = model(p, max_tokens=req.max_tokens or 512, temperature=req.temperature or 0.7, top_p=req.top_p or 1.0)
                return out["choices"][0]["text"]
            else:
                inputs = tokenizer(p, return_tensors="pt")
                try:
                    dev = next(model.parameters()).device
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                except Exception:
                    pass
                outputs = model.generate(**inputs, max_length=(inputs["input_ids"].shape[1] + (req.max_tokens or 512)))
                return tokenizer.decode(outputs[0], skip_special_tokens=True)
        history_list = list(history)
        max_steps = 5
        tool_registry = {
            "echo": lambda text: text,
            "sum": lambda a, b: a + b,
        }
        def parse_tool_calls(txt: str):
            try:
                data = json.loads(txt)
                if isinstance(data, dict) and isinstance(data.get("tool_calls"), list):
                    return data["tool_calls"]
            except Exception:
                pass
            try:
                start_tc = txt.find('{"tool_calls"')
                if start_tc != -1:
                    end_tc = txt.rfind(']')
                    blob = txt[start_tc: end_tc+1]
                    parsed = json.loads("{" + blob if not blob.strip().startswith('{') else blob)
                    return parsed.get('tool_calls')
            except Exception:
                return None
            return None
        step = 0
        last_assistant = None
        def exec_tool_calls(tool_calls, step):
            results = []
            for i, tc in enumerate(tool_calls):
                tc_id = tc.get("id") or f"call_{step}_{i}"
                fn = None
                name = None
                args = {}
                try:
                    if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
                        name = tc["function"].get("name")
                        raw_args = tc["function"].get("arguments")
                        if isinstance(raw_args, str):
                            try:
                                args = json.loads(raw_args)
                            except Exception:
                                args = {"_raw": raw_args}
                        elif isinstance(raw_args, dict):
                            args = raw_args
                except Exception:
                    pass
                fn = tool_registry.get(name)
                result = None
                err = None
                try:
                    if callable(fn):
                        result = fn(**args)
                    else:
                        err = f"unknown tool: {name}"
                except Exception as e:
                    err = str(e)
                tool_content = json.dumps({"tool_name": name, "result": result, "error": err})
                results.append({"role": "tool", "tool_call_id": tc_id, "content": tool_content})
            return results
        last_user = history_list[-1].content if history_list else ""
        initial_tc = parse_tool_calls(last_user)
        if initial_tc:
            assistant_msg = {"role": "assistant", "content": "", "tool_calls": initial_tc}
            history_list.append(ChatMessage(role="assistant", content=""))
            for m in exec_tool_calls(initial_tc, step):
                history_list.append(ChatMessage(role="tool", content=m["content"]))
            step += 1
        while step < max_steps:
            prompt_now = render_history(history_list)
            content = run_model(prompt_now)
            tool_calls = parse_tool_calls(content)
            assistant_msg = {"role": "assistant", "content": "" if tool_calls else content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages_to_append = [assistant_msg]
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    tc_id = tc.get("id") or f"call_{step}_{i}"
                    fn = None
                    name = None
                    args = {}
                    try:
                        if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
                            name = tc["function"].get("name")
                            raw_args = tc["function"].get("arguments")
                            if isinstance(raw_args, str):
                                try:
                                    args = json.loads(raw_args)
                                except Exception:
                                    args = {"_raw": raw_args}
                            elif isinstance(raw_args, dict):
                                args = raw_args
                    except Exception:
                        pass
                    fn = tool_registry.get(name)
                    result = None
                    err = None
                    try:
                        if callable(fn):
                            result = fn(**args)
                        else:
                            err = f"unknown tool: {name}"
                    except Exception as e:
                        err = str(e)
                    tool_content = json.dumps({"tool_name": name, "result": result, "error": err})
                    messages_to_append.append({"role": "tool", "tool_call_id": tc_id, "content": tool_content})
                for m in messages_to_append:
                    if m["role"] == "tool":
                        history_list.append(ChatMessage(role="tool", content=m["content"]))
                    else:
                        history_list.append(ChatMessage(role=m["role"], content=m["content"]))
                step += 1
                last_assistant = assistant_msg
                continue
            else:
                history_list.append(ChatMessage(role="assistant", content=assistant_msg["content"]))
                last_assistant = assistant_msg
                break
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": last_assistant or {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    def sse_events():
        acc = ""
        rendered = render_history(history)
        if isinstance(model, Llama):
            for chunk in model(rendered, stream=True, max_tokens=req.max_tokens or 512, temperature=req.temperature or 0.7, top_p=req.top_p or 1.0):
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") or chunk.get("choices", [{}])[0].get("text", "")
                acc += delta
                yield f"data: {json.dumps({'id':'chatcmpl-local','object':'chat.completion.chunk','model':req.model,'choices':[{'index':0,'delta':{'content':delta}}]})}\n\n"
        else:
            from transformers import TextIteratorStreamer
            import threading
            inputs = tokenizer(rendered, return_tensors="pt")
            try:
                dev = next(model.parameters()).device
                inputs = {k: v.to(dev) for k, v in inputs.items()}
            except Exception:
                pass
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(**inputs, max_new_tokens=(req.max_tokens or 128), temperature=req.temperature or 0.7, top_p=req.top_p or 1.0, streamer=streamer)
            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()
            for text in streamer:
                yield f"data: {json.dumps({'id':'chatcmpl-local','object':'chat.completion.chunk','model':req.model,'choices':[{'index':0,'delta':{'content':text}}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_events(), media_type="text/event-stream")

@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest):
    model, tokenizer = _ensure_loaded(req.model, getattr(req, 'load_opts', None))
    if isinstance(model, Llama):
        raise HTTPException(status_code=400, detail="Embeddings not supported for GGUF in this endpoint")
    texts = req.input
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            vec = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            vec = outputs.last_hidden_state
        else:
            raise HTTPException(status_code=500, detail="Model does not return hidden states")
        hidden = vec.mean(dim=1).cpu().tolist()
    data = [
        {"object": "embedding", "index": i, "embedding": emb}
        for i, emb in enumerate(hidden)
    ]
    return {"object": "list", "data": data, "model": req.model}

@app.get("/")
async def root():
    return {"status": "ok", "endpoints": ["/v1/models", "/v1/completions", "/v1/chat/completions", "/v1/embeddings"]}


def serve_global():
    uvicorn.run(app, host="127.0.0.1", port=8000)
