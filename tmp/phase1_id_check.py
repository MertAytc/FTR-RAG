import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import get_collection
bm = BM25Retriever()
dense = set(get_collection().get(include=[])["ids"])
lexical = set(bm.ids)
print("bm25", len(lexical), "dense", len(dense), "dense-only", len(dense-lexical), "bm25-only", len(lexical-dense))
print("dense-only examples", sorted(dense-lexical)[:5])
print("bm25-only examples", sorted(lexical-dense)[:5])
