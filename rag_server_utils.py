import os
from rag_utils import (
    load_texts_from_dir,
    chunk_text,
    embed_chunks,
    build_faiss_index,
)

def build_index(client, assets_dir):
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
