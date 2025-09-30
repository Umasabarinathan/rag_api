"""
Simple CLI RAG: build index once per run and answer user questions.
"""

import os
from dotenv import load_dotenv
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


def build_index(client: OpenAI, assets_dir: str):
    docs = load_texts_from_dir(assets_dir)
    print(f"Loaded {len(docs)} documents from {assets_dir}")
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))
    print(f"Total chunks created: {len(chunks)}")
    vecs = embed_chunks(client, chunks)
    import numpy as np
    print(f"Embeddings shape: {np.array(vecs).shape}")
    index = build_faiss_index(vecs)
    return chunks, index


def main() -> None:
    load_dotenv()
    
    # Check for OpenRouter configuration
    try:
        from openrouter_provider import create_openrouter_provider
        provider = create_openrouter_provider()
        client = provider.get_client()
        model_name = provider.get_model_name()
        print(f"✅ Using OpenRouter model: {model_name}")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    except ImportError:
        print("Please install openai: pip install openai")
        return
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    chunks, index = build_index(client, assets_dir)

    print("RAG index ready. Ask a question (or 'exit'):")
    while True:
        q = input("Q> ").strip()
        if not q or q.lower() == "exit":
            break
        q_vec = embed_query(client, q)
        idxs, _ = top_k(index, q_vec, k=4)
        ctx = [chunks[i] for i in idxs]
        a = answer_with_context(client, q, ctx, model_name)
        print("A>", a)


if __name__ == "__main__":
    main()
