"""Download FTR-RAG models directly to local Colab storage.

Run:
    python download_models.py

Optional environment variables:
    FTR_MODEL_DIR=/content/models
    FTR_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
    FTR_LLM_MODEL=Qwen/Qwen3-8B
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_DIR = Path(os.getenv("FTR_MODEL_DIR", "/content/models"))
EMBEDDING_MODEL = os.getenv(
    "FTR_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
)
LLM_MODEL = os.getenv("FTR_LLM_MODEL", "Qwen/Qwen3-8B")

def download(repo_id: str, local_dir: Path) -> None:
    print(f"\nDownloading {repo_id} -> {local_dir}", flush=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
    )
    print(f"Finished: {repo_id}", flush=True)

if __name__ == "__main__":
    download(EMBEDDING_MODEL, MODEL_DIR / "Qwen3-Embedding-0.6B")
    download(LLM_MODEL, MODEL_DIR / "Qwen3-8B")
    print(f"\nAll models downloaded to {MODEL_DIR}", flush=True)
