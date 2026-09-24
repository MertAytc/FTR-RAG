"""FTR RAG - main entry point.

The initial prototype will be built incrementally:
PDF -> extraction -> structure-aware chunking -> embeddings/BM25
-> hybrid retrieval -> MMR -> context -> LLM -> citations -> RAGAS.
"""

if __name__ == "__main__":
    print("FTR RAG project is ready. Implementation starts with PDF ingestion.")
