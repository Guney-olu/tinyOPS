# TODO ->  Fix the api chat and add cmd run
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import random
import time
import torch
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters
from tests.tinyllm import tokenizer, prefill, device, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P,encode_role,encode_message


model = "models/llama-1b-hf-32768-fpf"  # Add model here
last_seen_toks = []

app = FastAPI()
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving
@app.get("/{filename}")
async def serve_static(filename: str):
    file_path = Path(__file__).parent / "tinychat" / filename
    return FileResponse(file_path)

@app.get("/")
async def index():
    file_path = Path(__file__).parent / "tinychat" / "index.html"
    return FileResponse(file_path)

@app.get("/v1/models")
async def models():
    return JSONResponse(content=[str(model)]) 

@app.post("/v1/internal/token-count")
async def token_count(request: Request):
    rjson = await request.json()
    return JSONResponse(content=len(tokenizer.encode(rjson.get("text", ""))))

@app.post("/v1/token/encode")
async def token_encode(request: Request):
    rjson = await request.json()
    return JSONResponse(content=tokenizer.encode(rjson.get("text", "")))

@app.post("/v1/completions")
async def completions(request: Request):
    rjson = await request.json()
    if not rjson.get("stream", False):
        raise HTTPException(status_code=400, detail="streaming required")

    def generate():
        toks = [tokenizer.bos_id] + tokenizer.encode(rjson.get("prompt", ""), allow_special=True)
        start_pos = prefill(model, toks[:-1])
        last_tok = toks[-1]
        while True:
            GlobalCounters.reset()
            tok = model(torch.tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
            start_pos += 1
            last_tok = tok
            if tok in tokenizer.stop_tokens:
                break

            res = {
                "choices": [{
                    "text": tokenizer.decode([tok]),
                }]
            }
            yield f"data: {json.dumps(res)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/v1/chat/token/encode")
async def chat_token_encode(request: Request):
    rjson = await request.json()
    if "messages" not in rjson:
        raise HTTPException(status_code=400, detail="messages required")
    
    toks = [tokenizer.bos_id]
    for message in rjson["messages"]:
        toks += encode_message(message["role"], message["content"])
    if message["role"] == "user":
        toks += encode_role("assistant")
    
    return JSONResponse(content=toks)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global last_seen_toks
    rjson = await request.json()
    if "messages" not in rjson:
        raise HTTPException(status_code=400, detail="messages required")
    if not rjson.get("stream", False):
        raise HTTPException(status_code=400, detail="streaming required")
    
    def generate():
        toks = [tokenizer.bos_id]
        for message in rjson["messages"]:
            toks += encode_message(message["role"], message["content"])
        if message["role"] != "user":
            raise HTTPException(status_code=400, detail="last message must be a user message")
        toks += encode_role("assistant")

        random_id = random.randbytes(16).hex()

        start_pos = prefill(model, toks[:-1])
        last_tok = toks[-1]
        last_seen_toks.append(last_tok)
        token_limit = 5  # Limit to 3 tokens
        token_count = 0
        generated_tokens = []

        while token_count < token_limit:
            GlobalCounters.reset()
            tok = model(torch.tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
            start_pos += 1
            last_tok = tok
            last_seen_toks.append(tok)
            token_count += 1
            generated_tokens.append(tokenizer.decode([tok]))
            if tok in tokenizer.stop_tokens:
                break

            res = {
                "id": random_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": str(model),
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": tokenizer.decode([tok]),
                    },
                    "finish_reason": None,
                }]
            }
            yield f"data: {json.dumps(res)}\n\n"

        res = {
            "id": random_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": str(model),
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }]
        }

        yield f"data: {json.dumps(res)}\n\n"
        print("Generated Tokens:", generated_tokens)

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0"
    port = 8080
    uvicorn.run(app, host=host, port=port)
    
    

