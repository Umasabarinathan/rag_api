import os
import json
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

import faiss


load_dotenv()


def load_texts_from_dir(path: str) -> List[str]:
    texts: List[str] = []
    print(f"Scanning directory for .txt files: {path}")
    for name in os.listdir(path):
        if name.lower().endswith('.txt'):
            print(f"Found text file: {name}")
            with open(os.path.join(path, name), 'r', encoding='utf-8') as f:
                texts.append(f.read())
    print(f"Total text files loaded: {len(texts)}")
    return texts


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def embed_chunks(client: OpenAI, chunks: List[str]) -> List[List[float]]:
    """Generate mock embeddings for chunks. OpenRouter doesn't support embeddings."""
    import numpy as np
    
    # Use mock embeddings (OpenRouter doesn't support embeddings)
    print("ℹ️  Using mock embeddings (OpenRouter doesn't support embeddings)")
    mock_embeddings = []
    for i, chunk in enumerate(chunks):
        # Use hash of chunk text for deterministic but varied embeddings
        np.random.seed(hash(chunk) % 2**32)
        embedding = np.random.normal(0, 1, 1536).astype(np.float32)  # text-embedding-3-small dimension
        mock_embeddings.append(embedding.tolist())
    
    return mock_embeddings


def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatIP:
    import numpy as np
    vecs = np.array(embeddings, dtype='float32')
    # Ensure vecs is 2D and not empty
    if vecs.ndim != 2 or vecs.shape[0] == 0 or vecs.shape[1] == 0:
        raise ValueError(f"Input embeddings must be a non-empty 2D array, got shape {vecs.shape}")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def top_k(index: faiss.IndexFlatIP, query_vec: List[float], k: int = 4) -> Tuple[List[int], List[float]]:
    import numpy as np
    q = np.array([query_vec], dtype='float32')
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    return idxs[0].tolist(), scores[0].tolist()


def embed_query(client: OpenAI, query: str) -> List[float]:
    """Generate mock embedding for a single query. OpenRouter doesn't support embeddings."""
    import numpy as np
    
    # Use mock embedding (OpenRouter doesn't support embeddings)
    np.random.seed(hash(query) % 2**32)
    embedding = np.random.normal(0, 1, 1536).astype(np.float32)  # text-embedding-3-small dimension
    return embedding.tolist()


def answer_with_context(client: OpenAI, question: str, context_chunks: List[str], model_name: str = "gpt-4o-mini") -> str:
    context = "\n\n".join(context_chunks)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use ONLY the provided context to answer. If the answer is not in the context, say 'I don't know'."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        }
    ]
    
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
        
        # Extract the response text
        if resp.choices and len(resp.choices) > 0:
            return resp.choices[0].message.content.strip()
        else:
            return "No response generated"
            
    except Exception as e:
        return f"Error: {e}"


# rag_utils.py
# Utility functions for RAG (Retrieval-Augmented Generation)

# Add your RAG-related functions here
