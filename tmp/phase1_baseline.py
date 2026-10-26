import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.hybrid_retriever import HybridRetriever

benchmark = json.loads(Path("FTR_RAG_100_Benchmark_Questions.json").read_text(encoding="utf-8-sig"))
questions = benchmark.get("questions", benchmark) if isinstance(benchmark, dict) else benchmark
selected = questions[:3]
retriever = HybridRetriever()
rows = []
for question in selected:
    query = question["question"]
    dense = retriever._dense_retrieve(query)
    bm25 = retriever._bm25_retrieve(query)
    result = retriever.search(query, n_results=10)
    rows.append({"question_id": question.get("id"), "query": query, "dense_top5": [row["id"] for row in dense[:5]], "bm25_top5": [row["id"] for row in bm25[:5]], "rrf_top10": [row["id"] for row in result["results"]]})
    print(question.get("id"), "dense", rows[-1]["dense_top5"])
    print(question.get("id"), "bm25", rows[-1]["bm25_top5"])
    print(question.get("id"), "rrf", rows[-1]["rrf_top10"])
out = Path(__file__).resolve().parents[1] / "tests" / "baselines" / "phase1_top10_baseline.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"source": "FTR_RAG_100_Benchmark_Questions.json", "questions": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("Saved", out)
