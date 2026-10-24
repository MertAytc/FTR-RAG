"""FTR-RAG - Single-Gold Retrieval Benchmark (herhangi bir soru sayısı)
BGE vs Nano-Jev (açık kaynak, yerel) Reranker Karşılaştırması

run_retrieval_benchmark_33_bge_vs_jev.py dosyasının aynı pipeline, ayar ve
metrikleriyle çalışır; tek fark ücretli Jev API yerine yerel Nano-Jev kullanması
ve soru sayısının sabit olmaması (33, 50, ...).

Proje kökünden çalıştır:

    pip install nano-jev
    python run_retrieval_benchmark_bge_vs_nanojev.py --benchmark ftr_rag_50_benchmark.json

Hızlı deneme (ilk 5 soru):
    python run_retrieval_benchmark_bge_vs_nanojev.py --benchmark SORULAR.json --limit 5

Benchmark JSON biçimi: liste ya da {"questions": [...]}. Her soru için
    id (yoksa Q1, Q2, ...), question, gold_chunk_ids (liste ya da "id1; id2"),
    category (isteğe bağlı).
`gold_chunk_id`, `query`, `soru` gibi alternatif alan adları da kabul edilir.

Yöntemler:
    1. BM25-only
    2. Dense-only
    3. Dense + BM25 + weighted RRF
    4. RRF + BGE reranker
    5. RRF + Nano-Jev            (skor = P(directly answers) + 0.5 * P(partially relevant))
    6. RRF + Nano-Jev (direct)   (skor = P(directly answers))

BGE ve Nano-Jev aynı RRF Top-30 aday havuzunu alır. Nano-Jev modeli her soru
için bir kez çalışır, 5. ve 6. yöntem aynı olasılıklardan iki farklı sıralama üretir.
Cevap üretimi (generation) yapılmaz. Sonuçlar soru sayısını içeren yeni dosyalara
yazılır (retrieval_benchmark_<N>_bge_vs_nanojev_results.csv/.json), eskilerin üzerine yazılmaz.

Dikkat: Nano-Jev İngilizce/Wikipedia tarzı veriyle eğitilmiş küçük bir modeldir
(max 512 token). Sorgu olarak query_processor'ın İngilizce yeniden yazımı verilir.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding import EMBEDDING_MODEL
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import process_query
from src.retrieval.reranker import RERANKER_MODEL, CrossEncoderReranker

ROOT = Path(__file__).resolve().parent


# ============================================================
# RETRIEVAL SETTINGS (bge_vs_jev ile aynı)
# ============================================================
DENSE_TOP_N = 40
BM25_TOP_N = 20
RRF_K = 60
DENSE_WEIGHT = 1.10
BM25_WEIGHT = 0.85
RRF_CANDIDATE_K = 30          # BGE ve Nano-Jev aynı Top-30'u alır

RERANK_CANDIDATE_K = 30
RERANK_FINAL_K = 10
RERANK_BATCH_SIZE = 16        # BGE
RERANK_MAX_LENGTH = 512       # BGE

NANOJEV_CANDIDATE_K = 30
NANOJEV_BATCH_SIZE = 16

M_BM25 = "BM25-only"
M_DENSE = "Dense-only"
M_RRF = "Dense+BM25 RRF"
M_BGE = "RRF+BGE"
M_NJ = "RRF+NanoJev"
M_NJ_DIRECT = "RRF+NanoJev(direct)"
METHODS = [M_BM25, M_DENSE, M_RRF, M_BGE, M_NJ, M_NJ_DIRECT]


def normalize_id(value) -> str:
    return str(value).strip()


def find_benchmark() -> Path:
    """--benchmark verilmediyse proje kökünde tek bir benchmark JSON'u ara."""
    found = sorted(p for p in ROOT.glob("*.json")
                   if "benchmark" in p.name.lower() and "results" not in p.name.lower())
    if len(found) == 1:
        return found[0]
    listing = "\n  ".join(p.name for p in found) or "(hiç dosya yok)"
    raise SystemExit("Benchmark dosyası belirlenemedi. --benchmark ile açıkça ver.\n"
                     f"Proje kökünde bulunan adaylar:\n  {listing}")


# ============================================================
# NANO-JEV
# ============================================================
class NanoJevScorer:
    """Nano-Jev `relevance` kararı: her passage için
    {'irrelevant': p, 'partially relevant': p, 'directly answers': p}."""

    def __init__(self, weights: str | None = None, batch_size: int = NANOJEV_BATCH_SIZE):
        try:
            import nanojev
        except ImportError:
            raise SystemExit("nano-jev kurulu değil. Kurulum: pip install nano-jev")
        self.model = nanojev.load(weights) if weights else nanojev.load()
        self.batch_size = batch_size
        self.version = str(getattr(self.model, "version", weights or "default"))

    def probabilities(self, query: str, results: list[dict]) -> list[dict]:
        passages = [str(r.get("document", "")) for r in results]
        out: list[dict] = []
        for i in range(0, len(passages), self.batch_size):
            part = self.model.relevance(query, passages[i:i + self.batch_size])
            if isinstance(part, dict):      # tek passage dönerse
                part = [part]
            out.extend(part)
        if len(out) != len(passages):
            raise RuntimeError(f"Nano-Jev {len(passages)} passage için {len(out)} sonuç döndürdü.")
        return out


def nanojev_score(probs: dict, mode: str) -> float:
    direct = float(probs.get("directly answers", 0.0))
    partial = float(probs.get("partially relevant", 0.0))
    return direct if mode == "direct" else direct + 0.5 * partial


def rank_by_probabilities(results: list[dict], probs: list[dict], mode: str, top_k: int) -> list[dict]:
    # sorted() kararlıdır: eşit skorda RRF sırası korunur
    order = sorted(range(len(results)), key=lambda i: -nanojev_score(probs[i], mode))
    ranked = []
    for new_rank, i in enumerate(order[:top_k], 1):
        item = dict(results[i])
        item["rerank_score"] = nanojev_score(probs[i], mode)
        item["rerank_rank"] = new_rank
        item["nanojev_probs"] = probs[i]
        ranked.append(item)
    return ranked


# ============================================================
# BENCHMARK LOAD
# ============================================================
def _pick(d: dict, *keys, default=None):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return default


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).replace("|", ";").replace(",", ";").split(";")
    return [normalize_id(x) for x in items if normalize_id(x)]


def load_benchmark(path: Path, limit: int | None, expected_count: int | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = _pick(data, "questions", "benchmark", "items") if isinstance(data, dict) else data
    if not isinstance(questions, list) or not questions:
        raise ValueError("Benchmark dosyasında soru listesi bulunamadı.")
    if expected_count is not None and len(questions) != expected_count:
        raise ValueError(f"{expected_count} soru bekleniyordu, bulunan: {len(questions)}")

    normalized, seen_ids = [], set()
    for i, raw in enumerate(questions[:limit] if limit else questions, 1):
        q = dict(raw)
        q["id"] = str(_pick(q, "id", "qid", "question_id", default=f"Q{i}"))
        q["question"] = str(_pick(q, "question", "query", "soru", default="")).strip()
        q["category"] = str(_pick(q, "category", "kategori", default=""))
        gold = _as_list(_pick(q, "gold_chunk_ids", "gold_chunk_id", "gold_ids", "gold"))
        if not q["question"]:
            raise ValueError(f"{q['id']} için soru metni boş.")
        if not gold:
            raise ValueError(f"{q['id']} için en az bir gold_chunk_id gerekli.")
        if q["id"] in seen_ids:
            raise ValueError(f"Tekrarlanan soru ID'si: {q['id']}")
        seen_ids.add(q["id"])
        q["gold_chunk_ids"] = gold
        normalized.append(q)
    return normalized


# ============================================================
# METRICS (bge_vs_jev ile aynı tanımlar)
# ============================================================
def metrics(results: list[dict], gold_ids: list[str]) -> dict:
    ids = [normalize_id(x.get("id", "")) for x in results if normalize_id(x.get("id", ""))]
    gold_set = {normalize_id(x) for x in gold_ids}

    gold_ranks = [i for i, chunk_id in enumerate(ids, 1) if chunk_id in gold_set]
    first_rank = min(gold_ranks) if gold_ranks else None

    def recall(k: int) -> float:
        return 1.0 if any(c in gold_set for c in ids[:k]) else 0.0

    def ndcg(k: int) -> float:
        dcg = 0.0   # döngüyle topla: bge_vs_jev ile bit düzeyinde aynı sonuç (sum() Python 3.12'de farklı toplar)
        for r, c in enumerate(ids[:k], 1):
            if c in gold_set:
                dcg += 1.0 / math.log2(r + 1)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold_set), k) + 1))
        return dcg / idcg if idcg > 0 else 0.0

    return {
        "Hit@1": 1.0 if ids and ids[0] in gold_set else 0.0,
        "Recall@1": recall(1), "Recall@3": recall(3),
        "Recall@5": recall(5), "Recall@10": recall(10),
        "MRR": 1.0 / first_rank if first_rank else 0.0,
        "nDCG@5": ndcg(5), "nDCG@10": ndcg(10),
        "First Relevant Rank": first_rank,
        "Gold Ranks": gold_ranks,
        "Gold Count": len(gold_set),
    }


def make_row(method: str, results: list[dict], q: dict, processed: dict, elapsed_ms: float) -> dict:
    gold_ids = [normalize_id(x) for x in q["gold_chunk_ids"]]
    m = metrics(results, gold_ids)
    return {
        "Question ID": q["id"],
        "Category": q.get("category", ""),
        "Method": method,
        "Gold IDs": "; ".join(gold_ids),
        "Gold Count": m["Gold Count"],
        "Gold Ranks": "; ".join(str(x) for x in m["Gold Ranks"]),
        "First Relevant Rank": m["First Relevant Rank"],
        "Hit@1": round(m["Hit@1"], 4),
        "Recall@1": round(m["Recall@1"], 4),
        "Recall@3": round(m["Recall@3"], 4),
        "Recall@5": round(m["Recall@5"], 4),
        "Recall@10": round(m["Recall@10"], 4),
        "MRR": round(m["MRR"], 4),
        "nDCG@5": round(m["nDCG@5"], 4),
        "nDCG@10": round(m["nDCG@10"], 4),
        "Time ms": round(elapsed_ms, 1),
        "Gold Found": bool(m["First Relevant Rank"]),
        "Top 10 IDs": "; ".join(normalize_id(x.get("id", "")) for x in results[:10]),
        "Question": q["question"],
        "Rewritten Query": processed.get("rewritten_query", ""),
        "BM25 Query": processed.get("expanded_query", ""),
    }


# ============================================================
# PRINT
# ============================================================
def print_result(row: dict) -> None:
    print(
        f"  {row['Method']:<20} Hit@1={row['Hit@1']:.4f}  R@1={row['Recall@1']:.4f}  "
        f"R@3={row['Recall@3']:.4f}  R@5={row['Recall@5']:.4f}  R@10={row['Recall@10']:.4f}  "
        f"MRR={row['MRR']:.4f}  nDCG5={row['nDCG@5']:.4f}  nDCG10={row['nDCG@10']:.4f}  "
        f"GoldRank={str(row['First Relevant Rank']):<4} {row['Time ms']:.0f} ms"
    )


def sign_test_p(wins: int, losses: int) -> float:
    """İki yönlü kesin işaret testi (eşitlikler hariç)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _rank_or_inf(row: dict) -> float:
    r = row["First Relevant Rank"]
    return float(r) if r is not None else float("inf")


def print_summary(rows: list[dict], n_questions: int, pool_found: int) -> None:
    line = "=" * 150
    print("\n" + line)
    print(f"{n_questions} SORULUK FTR-RAG BENCHMARK - ORTALAMA SONUÇLAR")
    print(line)
    print(f"{'Method':<22}{'Hit@1':>8}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'R@10':>8}"
          f"{'MRR':>8}{'nDCG@5':>10}{'nDCG@10':>11}{'Avg ms':>10}")
    print("-" * 150)

    by_method = {m: [r for r in rows if r["Method"] == m] for m in METHODS}
    for method, subset in by_method.items():
        if not subset:
            continue
        print(
            f"{method:<22}"
            f"{mean(r['Hit@1'] for r in subset):>8.4f}"
            f"{mean(r['Recall@1'] for r in subset):>8.4f}"
            f"{mean(r['Recall@3'] for r in subset):>8.4f}"
            f"{mean(r['Recall@5'] for r in subset):>8.4f}"
            f"{mean(r['Recall@10'] for r in subset):>8.4f}"
            f"{mean(r['MRR'] for r in subset):>8.4f}"
            f"{mean(r['nDCG@5'] for r in subset):>10.4f}"
            f"{mean(r['nDCG@10'] for r in subset):>11.4f}"
            f"{mean(r['Time ms'] for r in subset):>10.1f}"
        )
    print("-" * 150)

    print("\nGold coverage:")
    for method, subset in by_method.items():
        if not subset:
            continue
        def top(k: int) -> int:
            return sum(r["First Relevant Rank"] is not None and r["First Relevant Rank"] <= k for r in subset)
        print(f"  {method:<22} AnyGold={sum(r['Gold Found'] for r in subset)}/{n_questions}  "
              f"Hit@1={sum(r['Hit@1'] == 1.0 for r in subset)}/{n_questions}  "
              f"Top3={top(3)}/{n_questions}  Top5={top(5)}/{n_questions}  Top10={top(10)}/{n_questions}")

    print(f"\nReranker tavanı: gold RRF Top-{RRF_CANDIDATE_K} havuzunda = {pool_found}/{n_questions} soru "
          "(reranker bu havuzun dışındakini kurtaramaz).")

    # Soru bazında eşleştirilmiş karşılaştırma (ilk gold sırası; küçük = iyi)
    bge = {r["Question ID"]: r for r in by_method[M_BGE]}
    print("\nBGE'ye karşı soru bazında karşılaştırma (ilk gold sırası):")
    for method in (M_NJ, M_NJ_DIRECT):
        other = {r["Question ID"]: r for r in by_method[method]}
        wins = sum(_rank_or_inf(other[k]) < _rank_or_inf(bge[k]) for k in bge if k in other)
        losses = sum(_rank_or_inf(other[k]) > _rank_or_inf(bge[k]) for k in bge if k in other)
        ties = sum(_rank_or_inf(other[k]) == _rank_or_inf(bge[k]) for k in bge if k in other)
        print(f"  {method:<22} daha iyi={wins}  eşit={ties}  daha kötü={losses}  "
              f"(işaret testi {_fmt_p(sign_test_p(wins, losses))})")

    print("  p < 0.05 değilse fark şansla açıklanabilir; küçük soru setinde bunu normal say.")

    cats = sorted({r["Category"] for r in rows if r["Category"]})
    if cats:
        print("\nKategori bazında MRR (Hit@1):")
        header = f"  {'Category':<28}{'n':>3}" + "".join(f"{m:>24}" for m in (M_RRF, M_BGE, M_NJ))
        print(header)
        for cat in cats:
            cells = []
            for m in (M_RRF, M_BGE, M_NJ):
                sub = [r for r in by_method[m] if r["Category"] == cat]
                cells.append(f"{mean(r['MRR'] for r in sub):.3f} ({mean(r['Hit@1'] for r in sub):.2f})" if sub else "-")
            n = len([r for r in by_method[M_RRF] if r["Category"] == cat])
            print(f"  {cat[:27]:<28}{n:>3}" + "".join(f"{c:>24}" for c in cells))


# ============================================================
# SAVE
# ============================================================
def save(rows: list[dict], total_seconds: float, extra: dict, csv_path: Path, json_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps({
            "benchmark_type": "single-gold",
            "embedding_model": EMBEDDING_MODEL,
            "bge_reranker_model": RERANKER_MODEL,
            "dense_top_n": DENSE_TOP_N, "bm25_top_n": BM25_TOP_N,
            "rrf_k": RRF_K, "dense_weight": DENSE_WEIGHT, "bm25_weight": BM25_WEIGHT,
            "rrf_candidate_k": RRF_CANDIDATE_K,
            "bge_rerank_candidate_k": RERANK_CANDIDATE_K,
            "nanojev_candidate_k": NANOJEV_CANDIDATE_K,
            "rerank_final_k": RERANK_FINAL_K,
            "generation": False,
            "total_runtime_seconds": round(total_seconds, 2),
            **extra,
            "results": rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="FTR-RAG single-gold benchmark: BGE vs Nano-Jev")
    ap.add_argument("--benchmark", type=Path, default=None,
                    help="benchmark JSON (verilmezse proje kökünde tek aday aranır)")
    ap.add_argument("--expected-count", type=int, default=None,
                    help="dosyada tam bu kadar soru olmalı (örn. 50); yanlış dosyayı yakalamak için")
    ap.add_argument("--output-prefix", default=None,
                    help="çıktı dosya öneki (varsayılan: retrieval_benchmark_<N>_bge_vs_nanojev)")
    ap.add_argument("--nanojev-weights", default=None, help="örn. v1.0 (varsayılan) veya v0.1")
    ap.add_argument("--nanojev-batch-size", type=int, default=NANOJEV_BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=None, help="sadece ilk N soruyu çalıştır (hızlı deneme)")
    args = ap.parse_args()

    benchmark_path = args.benchmark or find_benchmark()
    benchmark = load_benchmark(benchmark_path, args.limit, args.expected_count)
    n_questions = len(benchmark)
    prefix = args.output_prefix or f"retrieval_benchmark_{n_questions}_bge_vs_nanojev"
    out_csv, out_json = ROOT / f"{prefix}_results.csv", ROOT / f"{prefix}_results.json"

    print("=" * 125)
    print(f"FTR-RAG - {n_questions} SINGLE-GOLD RETRIEVAL BENCHMARK (BGE vs Nano-Jev)")
    print("=" * 125)
    nanojev = NanoJevScorer(args.nanojev_weights, args.nanojev_batch_size)
    print(f"Benchmark : {benchmark_path.name} ({n_questions} soru)")
    print(f"Embedding : {EMBEDDING_MODEL}")
    print(f"BGE       : {RERANKER_MODEL}")
    print(f"Nano-Jev  : sdmlai/nano-jev ({nanojev.version})")
    print("Pipeline  : BM25 -> Dense -> RRF -> BGE / Nano-Jev")
    print("Generation: KAPALI")
    print(f"Settings  : Dense Top-{DENSE_TOP_N}, BM25 Top-{BM25_TOP_N}, RRF k={RRF_K}, "
          f"weights={DENSE_WEIGHT}/{BM25_WEIGHT}, RRF Top-{RRF_CANDIDATE_K}, "
          f"BGE {RERANK_CANDIDATE_K}->{RERANK_FINAL_K}, Nano-Jev {NANOJEV_CANDIDATE_K}->{RERANK_FINAL_K}")

    print("\nBM25 index yükleniyor...")
    bm25 = BM25Retriever()
    corpus_ids = {normalize_id(x) for x in bm25.ids if normalize_id(x)}
    print(f"Corpus: {len(corpus_ids)} chunk")

    missing_gold = [(q["id"], g) for q in benchmark for g in q["gold_chunk_ids"] if g not in corpus_ids]
    total_gold = sum(len(q["gold_chunk_ids"]) for q in benchmark)
    if missing_gold:
        print("\n[UYARI] Index içinde bulunamayan gold chunk'lar:")
        for qid, gold_id in missing_gold:
            print(f"  Soru {qid}: {gold_id}")
        raise ValueError(f"{len(missing_gold)} gold chunk indexte bulunamadı. "
                         "Benchmark'ın gold ID'leri ile indeks aynı chunker çıktısından olmalı.")
    print(f"Gold kontrolü: {n_questions}/{n_questions} soru ve {total_gold} gold chunk index içinde bulundu.")

    hybrid = HybridRetriever(
        bm25_retriever=bm25, dense_top_n=DENSE_TOP_N, bm25_top_n=BM25_TOP_N,
        rrf_k=RRF_K, dense_weight=DENSE_WEIGHT, bm25_weight=BM25_WEIGHT,
    )
    bge_reranker = CrossEncoderReranker(batch_size=RERANK_BATCH_SIZE, max_length=RERANK_MAX_LENGTH)

    rows: list[dict] = []
    pool_found = 0
    all_start = time.perf_counter()
    extra = {"question_count": n_questions, "benchmark_file": benchmark_path.name, "nanojev_model": f"sdmlai/nano-jev ({nanojev.version})",
             "nanojev_score_modes": {"RRF+NanoJev": "direct + 0.5*partial", "RRF+NanoJev(direct)": "direct"}}

    def checkpoint() -> None:
        if rows:
            save(rows, time.perf_counter() - all_start, {**extra, "pool_gold_found": pool_found},
                 out_csv, out_json)

    for i, q in enumerate(benchmark, 1):
        print("\n" + "-" * 125)
        print(f"SORU {i}/{n_questions} | {q['id']} | {q.get('category', '')}")
        print(q["question"])
        print(f"GOLD ({len(q['gold_chunk_ids'])}): {'; '.join(q['gold_chunk_ids'])}")

        t = time.perf_counter()
        processed = process_query(q["question"])
        print(f"Query rewrite: {(time.perf_counter() - t) * 1000:.0f} ms")
        rewritten = processed["rewritten_query"]

        def run(method: str, results: list[dict], started: float) -> dict:
            row = make_row(method, results, q, processed, (time.perf_counter() - started) * 1000)
            rows.append(row)
            checkpoint()
            print_result(row)
            return row

        # 1. BM25
        t = time.perf_counter()
        bm25_results = bm25.search(processed["expanded_query"], n_results=BM25_TOP_N,
                                   apply_reference_penalty=False)
        run(M_BM25, bm25_results, t)

        # 2. Dense
        t = time.perf_counter()
        dense_results = hybrid._dense_retrieve(rewritten, top_n=DENSE_TOP_N)
        run(M_DENSE, dense_results, t)

        # 3. RRF (BGE ve Nano-Jev aynı Top-30'u kullanır)
        t = time.perf_counter()
        rrf_results = hybrid._rrf_fusion(dense_results, bm25_results,
                                         apply_quality_penalty=False)[:RRF_CANDIDATE_K]
        rrf_row = run(M_RRF, rrf_results, t)
        pool_found += bool(rrf_row["Gold Found"])

        # 4. RRF + BGE
        t = time.perf_counter()
        bge_results = bge_reranker.rerank(rewritten, rrf_results[:RERANK_CANDIDATE_K], top_k=RERANK_FINAL_K)
        bge_row = run(M_BGE, bge_results, t)

        # 5-6. RRF + Nano-Jev: model bir kez çalışır, iki sıralama üretilir
        pool = rrf_results[:NANOJEV_CANDIDATE_K]
        t = time.perf_counter()
        probs = nanojev.probabilities(rewritten, pool)
        model_ms = (time.perf_counter() - t) * 1000
        for method, mode in ((M_NJ, "expected"), (M_NJ_DIRECT, "direct")):
            t = time.perf_counter()
            ranked = rank_by_probabilities(pool, probs, mode, RERANK_FINAL_K)
            row = make_row(method, ranked, q, processed, model_ms + (time.perf_counter() - t) * 1000)
            rows.append(row)
            checkpoint()
            print_result(row)
            if method == M_NJ:
                nj_row = row

        # Tanı
        for name, r in (("BGE", bge_row), ("Nano-Jev", nj_row)):
            if r["First Relevant Rank"] is None:
                where = "havuzda var ama Top-10 dışında" if rrf_row["Gold Found"] else "RRF havuzunda yok"
                print(f"  -> {name} Top-10 içinde GOLD yok ({where})")
            else:
                print(f"  -> {name} ilk GOLD rank: {r['First Relevant Rank']}")

    total = time.perf_counter() - all_start
    print_summary(rows, n_questions, pool_found)
    save(rows, total, {**extra, "pool_gold_found": pool_found}, out_csv, out_json)

    print("\n" + "=" * 125)
    print("TEST TAMAMLANDI")
    print("=" * 125)
    print(f"Toplam süre : {total:.1f} saniye")
    print(f"CSV         : {out_csv}")
    print(f"JSON        : {out_json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest durduruldu.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nBENCHMARK HATASI: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
