import os
from typing import List
import ollama


# Set FTR_EMBEDDING_MODEL in PowerShell to benchmark another embedding model.
# Example:
#   $env:FTR_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
EMBEDDING_MODEL = os.getenv("FTR_EMBEDDING_MODEL", "qwen3-embedding:0.6b")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convert texts to vectors with the configured Ollama embedding model."""
    if not texts:
        return []

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return response.embeddings
