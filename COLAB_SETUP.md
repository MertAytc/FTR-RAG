# Run FTR-RAG in Google Colab

This setup runs English retrieval and grounded generation with Qwen3 Embedding
0.6B, Qwen3 8B, BM25, ChromaDB, and the BGE reranker. Select a GPU runtime
(T4 or better) in Colab before installing the models.

## 1. Clone the benchmark branch

Run this in a Colab cell:

```python
%cd /content
!git clone --branch 100_soru_benchmark https://github.com/MertAytc/FTR-RAG.git
%cd /content/FTR-RAG
```

If the repository is already present in the runtime, skip the clone.

## 2. Install Python dependencies

```python
%pip install -r requirements-colab.txt
```

Colab supplies the CUDA-enabled PyTorch build. The Python `ollama` package is
only a client; the Ollama service must also be installed and started.

## 3. Install and start Ollama

```python
!curl -fsSL https://ollama.com/install.sh | sh
```

Start the server in the background:

```python
import os
import subprocess
import time
import urllib.request

ollama_env = os.environ.copy()
ollama_env["OLLAMA_HOST"] = "127.0.0.1:11434"
ollama_server = subprocess.Popen(
    ["ollama", "serve"], env=ollama_env,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

for _ in range(60):
    try:
        urllib.request.urlopen(os.environ["OLLAMA_HOST"], timeout=2)
        break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("Ollama service did not start")
```

Download the configured models:

```python
!ollama pull qwen3-embedding:0.6b
!ollama pull qwen3:8b
```

The Qwen3 8B Ollama tag is about 5.2 GB. Qwen3 supports a `think` setting;
grounded generation turns thinking off to return the answer directly. [Ollama
Qwen3 model](https://ollama.com/library/qwen3:8b), [Ollama thinking API](https://ollama.com/blog/thinking).

## 4. Set data and vector database paths

Set environment variables before importing retrieval modules. Keep ChromaDB on
Colab's local disk for reliable SQLite access. The database is rebuilt after a
runtime reset unless you copy it to Google Drive and restore it in a later
session.

```python
import os
from pathlib import Path

REPO = Path("/content/FTR-RAG")
os.environ["FTR_CHUNKS_DIR"] = str(REPO / "data" / "processed" / "chunks")
os.environ["FTR_VECTOR_DB_DIR"] = "/content/ftr_rag_vector_db"
os.environ["FTR_EMBEDDING_MODEL"] = "qwen3-embedding:0.6b"
os.environ["FTR_GENERATION_MODEL"] = "qwen3:8b"
```

## 5. Build the dense index once per fresh runtime

The indexer clears and rebuilds the selected Chroma collection. Run it only
when the vector database is empty or when the chunk corpus or embedding model
has changed:

```python
from src.retrieval.vector_store import count

if count() == 0:
    from src.retrieval.indexer import index_chunks
    index_chunks()
else:
    print(f"Dense index already contains {count()} chunks")
```

The chunk JSON files are read from `data/processed/chunks`; large source PDFs
are not needed for retrieval or generation.

## 6. Ask a question

```python
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.context_assembly import assemble_context
from src.generation.grounded_generation import generate_grounded_answer

question = "What exercise interventions are recommended for knee osteoarthritis?"
reranker = CrossEncoderReranker()
ranked = reranker.rerank_hybrid(question, rrf_top_k=20, top_k=5)
context = assemble_context(ranked["reranked_results"], max_sources=3)
answer = generate_grounded_answer(question, context)
print(answer.answer)
```

## Notes

- `FTR_VECTOR_DB_DIR` lets the vector database live outside the repository.
- `FTR_GENERATION_MODEL` can override `qwen3:8b` when needed.
- The reranker downloads `BAAI/bge-reranker-v2-m3` from Hugging Face on first use.
- The first dense index build embeds the chunk corpus and takes time. Do not
  call `index_chunks()` on every question; it resets the collection.
- For RAGAS evaluation, install the separate evaluation dependencies used by
  the evaluation script and provide any required API credentials through
  Colab Secrets or environment variables, never in source files.
