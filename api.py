"""
FastAPI RAG Chatbot API

Instructions:
To run on port 8080, use:
    uvicorn api:app --reload --port 8080
Your endpoint will be available at http://localhost:8080/chat
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from rag_utils import (
    load_texts_from_dir,
    chunk_text,
    embed_chunks,
    build_faiss_index,
    embed_query,
    top_k,
    answer_with_context,
)
from rag_server_utils import build_index

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

# Load model and build index at startup
load_dotenv()
from openrouter_provider import create_openrouter_provider
provider = create_openrouter_provider()
client = provider.get_client()
model_name = provider.get_model_name()
assets_dir = os.path.join(os.path.dirname(__file__), "assets")
chunks, index = build_index(client, assets_dir)

# Allow your frontend origin(s)
origins = [
    "http://localhost:8080",  # React/Next.js dev server
    "http://127.0.0.1:8080",  # another possible dev URL
    # Add your production domain too
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],            # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],            # Allow all headers
)

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        q = question.strip()
        if not q:
            return {"answer": "No question provided."}
        q_vec = embed_query(client, q)
        idxs, _ = top_k(index, q_vec, k=4)
        ctx = [chunks[i] for i in idxs]
        a = answer_with_context(client, q, ctx, model_name)
        return {"answer": a}
    except Exception as e:
        return JSONResponse(status_code=500, content={"answer": f"Error: {str(e)}"})
