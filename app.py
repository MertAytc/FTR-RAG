import json
import math
import re
import time
from pathlib import Path

import streamlit as st

from src.retrieval.query_processor import process_query
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.mmr_retriever import MMRRetriever
from src.generation.context_assembly import assemble_context
from src.generation.streaming_generation import stream_grounded_generation
from src.generation.grounded_generation import GENERATION_MODEL


st.set_page_config(page_title="FTR-RAG Evaluation Console", page_icon="🧪", layout="wide")
st.title("FTR-RAG Evaluation Console")
st.caption("Cevap → generation karşılaştırması → Dense + BM25 + RRF → MMR → metrikler ve ayrıntılı retrieval teşhisi")

if "benchmark_history" not in st.session_state:
    st.session_state.benchmark_history = []


def ms(seconds):
    return round(seconds * 1000, 2)


def get_ollama_models():
    try:
        import ollama
        data = ollama.list()
        raw_models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
        models = []
        for item in raw_models:
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
            else:
                name = getattr(item, "model", None) or getattr(item, "name", None)
            if name:
                models.append(str(name))
        return models
    except Exception:
        return []


def relevant_ids(raw):
    return {x.strip() for x in re.split(r"[,;\n]+", raw or "") if x.strip()}


def retrieval_metrics(ranked_results, gold_ids):
    """Calculate standard retrieval metrics on the Hybrid/RRF candidate ranking.

    Metrics are calculated before MMR so Recall@10 can see ranks 6-10.
    Relevance is binary: Gold chunk = 1, non-Gold = 0.
    """
    empty = {
        "recall@1": None,
        "recall@3": None,
        "recall@5": None,
        "recall@10": None,
        "mrr": None,
        "ndcg@5": None,
        "ndcg@10": None,
        "first_relevant_rank": None,
    }
    if not gold_ids:
        return empty

    ranked_ids = [
        str(x.get("id", "")).strip()
        for x in ranked_results
        if str(x.get("id", "")).strip()
    ]
    gold = {str(x).strip() for x in gold_ids if str(x).strip()}

    first_rank = next(
        (i + 1 for i, rid in enumerate(ranked_ids) if rid in gold),
        None,
    )

    def recall(k):
        return round(len(set(ranked_ids[:k]) & gold) / len(gold), 4)

    def dcg(k):
        return sum(
            (1.0 / math.log2(rank + 1))
            for rank, rid in enumerate(ranked_ids[:k], start=1)
            if rid in gold
        )

    def ndcg(k):
        ideal_relevant = min(len(gold), k)
        if ideal_relevant == 0:
            return 0.0
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_relevant + 1)
        )
        return round(dcg(k) / idcg, 4) if idcg else 0.0

    return {
        "recall@1": recall(1),
        "recall@3": recall(3),
        "recall@5": recall(5),
        "recall@10": recall(10),
        "mrr": round(1 / first_rank, 4) if first_rank else 0.0,
        "ndcg@5": ndcg(5),
        "ndcg@10": ndcg(10),
        "first_relevant_rank": first_rank,
    }


def debug_gold_value(row, *keys):
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def safe_rank(value, missing_label):
    return value if value is not None else missing_label


def fmt_score(value, digits=4, missing="-"):
    """Format optional retrieval scores without crashing on None/non-numeric values."""
    if value is None:
        return missing
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return missing


def build_gold_debug_rows(retrieval_debug):
    rows = []
    for row in retrieval_debug.get("gold", []):
        gid = str(row.get("ID", row.get("id", ""))).strip()
        if not gid:
            continue

        rrf_rank = debug_gold_value(row, "RRF rank", "RRF rank (candidate pool)", "rrf_rank")
        in_candidate = debug_gold_value(
            row, "In candidate pool", "In MMR candidates", "in_candidate_pool"
        )

        rows.append({
            "ID": gid,
            "Dense Top N": debug_gold_value(row, "Dense Top N", "dense_top_n", "in_dense_top"),
            "Dense rank (full)": debug_gold_value(row, "Dense rank (full)", "dense_rank_full"),
            "Dense distance": debug_gold_value(row, "Dense distance", "dense_distance"),
            "BM25 Top N": debug_gold_value(row, "BM25 Top N", "bm25_top_n", "in_bm25_top"),
            "BM25 rank (full)": debug_gold_value(row, "BM25 rank (full)", "bm25_rank_full"),
            "BM25 score": debug_gold_value(row, "BM25 score", "bm25_score"),
            "RRF rank": rrf_rank,
            "In candidate pool": in_candidate,
            "In MMR candidates": row.get("In MMR candidates", in_candidate),
            "BM25 raw score": row.get("BM25 raw score", row.get("bm25_raw_score")),
            "BM25 role": row.get("BM25 role", row.get("chunk_role")),
        })
    return rows


models = get_ollama_models()

# Load the fixed pilot benchmark. Gold IDs are initial candidates and should be
# confirmed by the FTR expert before they are treated as final gold labels.
BENCHMARK_PATH = Path(__file__).with_name("benchmark_questions_updated.json")
if not BENCHMARK_PATH.exists():
    BENCHMARK_PATH = Path(__file__).with_name("benchmark_questions.json")
RESULTS_LOG_PATH = Path(__file__).with_name("ftr_rag_test_log.md")


def append_test_log(record):
    """Persist a human-readable, append-only test log next to the app."""
    new_file = not RESULTS_LOG_PATH.exists() or RESULTS_LOG_PATH.stat().st_size == 0
    with RESULTS_LOG_PATH.open("a", encoding="utf-8") as f:
        if new_file:
            f.write("# FTR-RAG Test Log\n\n")
            f.write("Bu dosya RUN TEST her çalıştırıldığında otomatik olarak büyür. Gün sonunda bu dosyayı paylaşabilirsin.\n\n")
        f.write("---\n\n")
        f.write(f"## Test — {record.get('timestamp', '?')}\n\n")
        f.write(f"**Benchmark ID:** {record.get('benchmark_id', 'custom')}  \n")
        f.write(f"**Category:** {record.get('category', 'custom')}  \n")
        f.write(f"**Query:** {record.get('query', '')}\n\n")
        f.write("### Configuration\n\n")
        for k, v in record.get("configuration", {}).items():
            f.write(f"- **{k}:** {v}\n")
        f.write("\n### Answer\n\n")
        f.write(record.get("answer", "") or "(empty)")
        f.write("\n\n### Generation Model Comparison\n\n")
        for row in record.get("generation_comparison", []):
            f.write(f"#### {row.get('Model', '?')}\n\n")
            f.write(f"- TTFT: {row.get('TTFT (ms)')} ms\n")
            f.write(f"- Generation: {row.get('Generation (ms)')} ms\n")
            f.write(f"- Total: {row.get('Total (ms)')} ms\n")
            f.write(f"- Tokens: {row.get('Tokens')}\n")
            f.write(f"- Error: {row.get('Error') or '-'}\n\n")
            f.write(row.get("Answer", "") or "(empty)")
            f.write("\n\n")
        f.write("### Retrieval Metrics\n\n")
        for k in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10", "first_relevant_rank"):
            f.write(f"- **{k}:** {record.get('metrics', {}).get(k)}\n")
        f.write("\n### Query Processing\n\n")
        qd = record.get("query_debug", {})
        f.write(f"**Original query:** {qd.get('original_query', '')}\n\n")
        f.write(f"**Matched FTR terms:** {qd.get('matched_terms', [])}\n\n")
        f.write(f"**Rewritten query:** {qd.get('rewritten_query', '')}\n\n")
        f.write(f"**Expanded BM25 query:** {qd.get('expanded_query', '')}\n\n")
        debug = record.get("retrieval_debug") or {}
        if debug:
            f.write("### Retrieval Diagnosis\n\n")
            f.write(f"Dense index count: {debug.get('dense_index_count')}\n\n")
            f.write(f"BM25 index count: {debug.get('bm25_index_count')}\n\n")
            f.write("| Gold ID | Dense Top N | Dense full rank | Dense distance | BM25 Top N | BM25 full rank | BM25 score | RRF rank | In MMR candidates |\n")
            f.write("|---|:---:|---:|---:|:---:|---:|---:|---:|:---:|\n")
            for row in debug.get("gold", []):
                # Support old/new debug field names and never fail the whole
                # test log because an optional diagnostic field is absent.
                rrf_rank = debug_gold_value(row, "RRF rank (candidate pool)", "RRF rank", "rrf_rank")
                in_pool = debug_gold_value(row, "In candidate pool", "In MMR candidates", "in_candidate_pool")
                values = [
                    row.get("ID"), row.get("Dense Top N"), row.get("Dense rank (full)"),
                    row.get("Dense distance"), row.get("BM25 Top N"), row.get("BM25 rank (full)"),
                    row.get("BM25 score"), rrf_rank, in_pool,
                ]
                values = ["" if v is None else str(v).replace("|", "\\|") for v in values]
                f.write("| " + " | ".join(values) + " |\n")
            f.write("\n")
        f.write("### Hybrid/RRF Candidate Ranking\n\n")
        f.write("| Rank | ID | Gold | RRF | Dense rank | BM25 rank | Dense score | BM25 score | Page |\n")
        f.write("|---:|---|:---:|---:|---:|---:|---:|---:|---:|\n")
        for row in record.get("hybrid_ranking", []):
            f.write("| {Rank} | {ID} | {Gold} | {RRF} | {Dense rank} | {BM25 rank} | {Dense score} | {BM25 score} | {Page} |\n".format(**{k: ("" if v is None else str(v).replace("|", "\\|")) for k,v in row.items()}))
        f.write("\n### Final MMR Results\n\n")
        for row in record.get("mmr_results", []):
            f.write(f"#### #{row.get('Rank')} — {row.get('ID')}\n\n")
            f.write(f"MMR={row.get('MMR')}, Relevance={row.get('Relevance')}, RRF={row.get('RRF')}, Dense rank={row.get('Dense rank')}, BM25 rank={row.get('BM25 rank')}, Dense={row.get('Dense')}, BM25={row.get('BM25')}\n\n")
            f.write(f"Document/page/section: {row.get('Source')}\n\n")
            f.write(row.get("Text", "") or "")
            f.write("\n\n")
        f.write("### Timing Summary\n\n")
        for k, v in record.get("timings", {}).items():
            f.write(f"- **{k}:** {v} ms\n")
        f.write("\n### Context Sent to LLM\n\n```text\n")
        f.write(record.get("context", "") or "")
        f.write("\n```\n\n")
try:
    BENCHMARK = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
except Exception:
    BENCHMARK = []

with st.sidebar:
    st.header("Test Ayarları")
    candidate_k = st.number_input(
        "MMR candidate_k", 10, 100, 40,
        help="RRF sıralamasından MMR'a aktarılacak aday sayısı."
    )
    final_k = st.number_input("MMR final_k", 1, 30, 6)
    lambda_mult = st.slider(
        "MMR lambda", 0.0, 1.0, 0.80, 0.05,
        help="MMR'ın relevance/diversity dengesini belirler."
    )
    rrf_weight = st.slider(
        "MMR relevance: RRF weight", 0.0, 1.0, 0.85, 0.05,
        help="MMR relevance hesabındaki RRF ağırlığı."
    )
    semantic_weight = round(1.0 - rrf_weight, 2)
    dense_top_n = st.number_input("Dense top N", 1, 100, 40)
    bm25_top_n = st.number_input("BM25 top N", 1, 100, 40)
    rrf_k = st.number_input("RRF k", 1, 200, 60)
    max_chunks_per_document = st.number_input(
        "MMR max chunk / document", 1, 10, 2,
        help="Belge çeşitliliği için MMR'a uygulanan başlangıç sınırı."
    )
    language = st.selectbox("Yanıt dili", ["Turkish", "English"])
    num_predict = st.number_input("Max output tokens", 32, 2000, 300, 16)

    st.divider()
    st.subheader("Generation Model Test")
    if models:
        default_index = models.index(GENERATION_MODEL) if GENERATION_MODEL in models else 0
        generation_model = st.selectbox(
            "Ana generation modeli",
            models,
            index=default_index,
            key="generation_model_select",
        )
        compare_models = st.multiselect(
            "Aynı context ile karşılaştırılacak modeller",
            models,
            default=[generation_model],
            key="generation_model_compare",
        )
    else:
        generation_model = st.text_input("Generation model", GENERATION_MODEL)
        compare_models = [generation_model]
        st.caption("Ollama modelleri listelenemedi; model adını elle girebilirsin.")

st.subheader("Pilot Benchmark — 10 Soru")
if BENCHMARK:
    labels = [f"{x['id']} — {x['category']}" for x in BENCHMARK]
    selected_label = st.selectbox("Hazır benchmark sorusu", ["Serbest soru"] + labels)
    selected = None if selected_label == "Serbest soru" else BENCHMARK[labels.index(selected_label)]
else:
    selected = None

default_query = selected["question"] if selected else ""
default_gold = ", ".join(selected["gold_chunk_ids"]) if selected else ""

query = st.text_area(
    "FTR klinik sorusu",
    value=default_query,
    placeholder="Örn. Diz osteoartriti olan yetişkinlerde kas güçlendirme egzersizleri öneriliyor mu?",
    height=100,
)

# Keep gold IDs visible but editable so expert corrections can be made.
# Include the benchmark Gold value in the widget key so Streamlit cannot
# silently keep an old Gold-ID value after the benchmark file is updated.
gold_widget_key = f"gold_ids_{selected['id'] if selected else 'custom'}_{default_gold}"
gold_raw = st.text_area(
    "Gold relevant chunk ID(leri)",
    value=default_gold,
    key=gold_widget_key,
    placeholder="Örn: 9789240071100-eng_chunk_0021, 9789240071100-eng_chunk_0025",
    height=70,
)
if selected:
    st.caption("Bu 10 soruluk pilot setteki gold ID'ler başlangıç adaylarıdır; FTR uzmanı onayından sonra final gold olarak kullanılmalıdır.")

run = st.button("RUN TEST", type="primary", width="stretch")

if run:
    if not query.strip():
        st.warning("Lütfen bir soru gir.")
        st.stop()

    timings = {}
    gold_ids = relevant_ids(gold_raw)

    # -----------------------------
    # 1) RETRIEVAL / CONTEXT first, because answer needs context.
    # -----------------------------
    t0 = time.perf_counter()
    try:
        processed = process_query(query.strip())
        timings["Query Processing"] = ms(time.perf_counter() - t0)
    except Exception as exc:
        st.error(f"Query Processing failed: {exc}")
        st.stop()

    try:
        retriever = MMRRetriever(
            candidate_k=int(candidate_k),
            final_k=int(final_k),
            lambda_mult=float(lambda_mult),
            rrf_weight=float(rrf_weight),
            semantic_weight=float(semantic_weight),
            max_chunks_per_document=int(max_chunks_per_document),
            hybrid_retriever=HybridRetriever(
                dense_top_n=int(dense_top_n),
                bm25_top_n=int(bm25_top_n),
                rrf_k=int(rrf_k),
            ),
        )
        t0 = time.perf_counter()
        retrieval = retriever.search(query.strip(), debug_gold_ids=gold_ids)
        timings["Dense + BM25 + RRF + MMR"] = ms(time.perf_counter() - t0)
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        st.stop()

    results = retrieval.get("results", [])
    hybrid_results = retrieval.get("hybrid_results", [])
    metrics = retrieval_metrics(hybrid_results, gold_ids)
    retrieved_gold = [
        str(item.get("id", "")).strip()
        for item in hybrid_results
        if str(item.get("id", "")).strip() in {str(x).strip() for x in gold_ids}
    ]

    t0 = time.perf_counter()
    try:
        context_result = assemble_context(results, max_sources=int(final_k), exclude_reference_only=True)
        timings["Context Assembly"] = ms(time.perf_counter() - t0)
    except Exception as exc:
        st.error(f"Context Assembly failed: {exc}")
        st.stop()

    # -----------------------------
    # 2) ANSWER first: primary generation model streams immediately.
    # -----------------------------
    st.subheader("Cevap")
    answer_box = st.empty()
    answer_status = st.empty()
    a1, a2, a3, a4 = st.columns(4)
    ttft_box, gen_box, total_box, tok_box = a1.empty(), a2.empty(), a3.empty(), a4.empty()

    answer = ""
    primary_stats = {}
    generation_started = time.perf_counter()
    try:
        for event in stream_grounded_generation(
            query.strip(),
            context_result.get("context_text", ""),
            language=language,
            num_predict=int(num_predict),
            model=generation_model,
        ):
            if event["event"] in ("first_token", "token"):
                answer += event.get("token", "")
                answer_box.markdown(answer)
                tok_box.metric("Tokens", event.get("token_count", 0))
                if event["event"] == "first_token":
                    ttft_box.metric("TTFT", f"{event['ttft_ms']:.0f} ms")
                    answer_status.info("İlk token geldi — cevap akıyor...")
            elif event["event"] == "done":
                primary_stats = event
                answer = event.get("answer", answer)
                answer_box.markdown(answer)
                ttft_box.metric("TTFT", f"{event['ttft_ms']:.0f} ms" if event.get("ttft_ms") is not None else "-")
                gen_box.metric("Generation", f"{event['generation_ms']:.0f} ms")
                total_box.metric("Total", f"{event['total_ms']:.0f} ms")
                tok_box.metric("Tokens", event.get("token_count", 0))
                answer_status.success(f"{generation_model} tamamlandı.")
            elif event["event"] == "error":
                primary_stats = event
                answer_status.error(
                    f"{generation_model} generation error: {event.get('error')}"
                )
    except Exception as exc:
        primary_stats = {
            "error": str(exc),
            "total_ms": ms(time.perf_counter() - generation_started),
            "token_count": 0,
        }
        answer_status.error(f"{generation_model} generation failed: {exc}")

    timings["LLM TTFT"] = primary_stats.get("ttft_ms")
    timings["LLM Generation"] = primary_stats.get("generation_ms")
    timings["LLM Total"] = primary_stats.get("total_ms", ms(time.perf_counter() - generation_started))

    # -----------------------------
    # 3) Generation model comparison, below the primary answer.
    # -----------------------------
    st.divider()
    st.subheader("Generation Model Karşılaştırması")
    st.caption("Aynı soru ve aynı retrieved context kullanılır. Böylece generation modelinin etkisini retrieval etkisinden ayırabiliriz.")

    comparison_rows = []
    if generation_model not in compare_models:
        compare_models = [generation_model] + compare_models
    for model in compare_models:
        if model == generation_model:
            comparison_rows.append({
                "Model": model,
                "TTFT (ms)": primary_stats.get("ttft_ms"),
                "Generation (ms)": primary_stats.get("generation_ms"),
                "Total (ms)": primary_stats.get("total_ms"),
                "Tokens": primary_stats.get("token_count"),
                "Error": primary_stats.get("error"),
                "Answer": answer,
            })
            continue
        with st.expander(f"{model} — aynı context ile test", expanded=False):
            box = st.empty()
            status = st.empty()
            other_answer = ""
            stats = {}
            try:
                for event in stream_grounded_generation(
                    query.strip(), context_result.get("context_text", ""),
                    language=language, num_predict=int(num_predict), model=model,
                ):
                    if event["event"] in ("first_token", "token"):
                        other_answer += event.get("token", "")
                        box.markdown(other_answer)
                    elif event["event"] == "done":
                        stats = event
                        other_answer = event.get("answer", other_answer)
                        box.markdown(other_answer)
                        status.success("Tamamlandı.")
                    elif event["event"] == "error":
                        stats = event
                        status.error(event.get("error", "Generation error"))
            except Exception as exc:
                stats = {
                    "error": str(exc),
                    "total_ms": None,
                    "token_count": 0,
                }
                status.error(f"{model} generation failed: {exc}")
            comparison_rows.append({
                "Model": model,
                "TTFT (ms)": stats.get("ttft_ms"),
                "Generation (ms)": stats.get("generation_ms"),
                "Total (ms)": stats.get("total_ms"),
                "Tokens": stats.get("token_count"),
                "Error": stats.get("error"),
                "Answer": other_answer,
            })

    st.dataframe(
        [{k: v for k, v in row.items() if k != "Answer"} for row in comparison_rows],
        width="stretch",
        hide_index=True,
    )
    for row in comparison_rows:
        with st.expander(f"{row['Model']} — cevap"):
            st.write(row["Answer"])

    # -----------------------------
    # 4) Retrieval debug and metrics.
    # -----------------------------
    st.divider()
    st.subheader("Retrieval ve Değerlendirme")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall@1", "-" if metrics["recall@1"] is None else metrics["recall@1"])
    m2.metric("Recall@3", "-" if metrics["recall@3"] is None else metrics["recall@3"])
    m3.metric("Recall@5", "-" if metrics["recall@5"] is None else metrics["recall@5"])
    m4.metric("Recall@10", "-" if metrics["recall@10"] is None else metrics["recall@10"])
    n1, n2, n3 = st.columns(3)
    n1.metric("MRR", "-" if metrics["mrr"] is None else metrics["mrr"])
    n2.metric("nDCG@5", "-" if metrics["ndcg@5"] is None else metrics["ndcg@5"])
    n3.metric("nDCG@10", "-" if metrics["ndcg@10"] is None else metrics["ndcg@10"])

    if gold_ids:
        st.info(f"Gold chunk sayısı: {len(gold_ids)} — metrikler Hybrid/RRF aday sıralaması üzerinden hesaplandı.")
        if retrieved_gold:
            st.success(f"Hybrid/RRF içinde bulunan Gold ID: {', '.join(retrieved_gold)}")
        else:
            st.warning("Gold ID'lerin hiçbiri Hybrid/RRF aday sıralamasında bulunamadı.")
    else:
        st.warning("Gold chunk ID girilmediği için retrieval metrikleri hesaplanmadı.")

    # Diagnostic retrieval panel. It only explains ranking; it does not alter it.
    retrieval_debug = retrieval.get("debug") or {}
    mmr_debug = retrieval.get("mmr_debug") or {}

    if gold_ids and retrieval_debug:
        st.divider()
        st.subheader("Retrieval Teşhis Ekranı")
        st.caption(
            "Gold chunk'ın Dense → BM25 → RRF → MMR zincirindeki konumunu gösterir. "
            "Bu panel retrieval sıralamasını değiştirmez."
        )

        gold_rows = build_gold_debug_rows(retrieval_debug)
        if gold_rows:
            st.markdown("**Gold chunk izleme**")
            st.dataframe(gold_rows, width="stretch", hide_index=True)

            for row in gold_rows:
                gid = row["ID"]
                d_top = "EVET" if row["Dense Top N"] else "HAYIR"
                b_top = "EVET" if row["BM25 Top N"] else "HAYIR"
                d_rank = safe_rank(row["Dense rank (full)"], "indekste yok")
                b_rank = safe_rank(row["BM25 rank (full)"], "indekste yok")
                r_rank = safe_rank(row["RRF rank"], "RRF havuzunda yok")
                pool = "EVET" if row["In candidate pool"] else "HAYIR"
                st.info(
                    f"{gid} → Dense Top-N: {d_top} (tam sıra: {d_rank}) | "
                    f"BM25 Top-N: {b_top} (tam sıra: {b_rank}) | "
                    f"RRF: {r_rank} | Candidate pool: {pool}"
                )

        d1, d2 = st.columns(2)
        with d1:
            dense_top = retrieval_debug.get("dense_top", [])
            with st.expander(f"Dense Top-{len(dense_top)}", expanded=False):
                rows = []
                for rank, item in enumerate(dense_top, 1):
                    md = item.get("metadata") or {}
                    rows.append({
                        "Rank": rank, "ID": item.get("id"),
                        "Gold": str(item.get("id", "")).strip() in gold_ids,
                        "Distance": item.get("score"),
                        "Document": md.get("document_id"),
                        "Page": md.get("page"),
                        "Section": md.get("section"),
                        "Role": item.get("chunk_role", md.get("chunk_role")),
                    })
                st.dataframe(rows, width="stretch", hide_index=True)

        with d2:
            bm25_top = retrieval_debug.get("bm25_top", [])
            with st.expander(f"BM25 Top-{len(bm25_top)}", expanded=False):
                rows = []
                for rank, item in enumerate(bm25_top, 1):
                    md = item.get("metadata") or {}
                    rows.append({
                        "Rank": rank, "ID": item.get("id"),
                        "Gold": str(item.get("id", "")).strip() in gold_ids,
                        "Score": item.get("score"),
                        "Raw score": item.get("raw_score"),
                        "Document": md.get("document_id"),
                        "Page": md.get("page"),
                        "Section": md.get("section"),
                        "Role": item.get("chunk_role", md.get("chunk_role")),
                    })
                st.dataframe(rows, width="stretch", hide_index=True)

        with st.expander("RRF Candidate Ranking — teşhis görünümü", expanded=True):
            rows = []
            for rank, item in enumerate(retrieval_debug.get("rrf_top", []), 1):
                md = item.get("metadata") or {}
                rows.append({
                    "Rank": rank, "ID": item.get("id"),
                    "Gold": str(item.get("id", "")).strip() in gold_ids,
                    "RRF": item.get("rrf_score"),
                    "Dense rank": item.get("dense_rank"),
                    "BM25 rank": item.get("bm25_rank"),
                    "Dense score": item.get("dense_score"),
                    "BM25 score": item.get("bm25_score"),
                    "Document": md.get("document_id"),
                    "Page": md.get("page"),
                    "Role": item.get("chunk_role", md.get("chunk_role")),
                })
            st.dataframe(rows, width="stretch", hide_index=True)

        if mmr_debug:
            st.markdown("**MMR davranışı**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Candidate k", mmr_debug.get("candidate_k", "-"))
            c2.metric("Final k", mmr_debug.get("final_k", "-"))
            c3.metric("Lambda", mmr_debug.get("lambda_mult", "-"))
            c4.metric("Max/document", mmr_debug.get("max_chunks_per_document", "-"))

            st.caption(
                "MMR yalnızca relevance ile redundancy arasındaki dengeyi kurar. "
                "Selection Trace, bu seçimin nasıl oluştuğunu gösteren teşhis verisidir."
            )

            trace = mmr_debug.get("selection_trace", [])
            if trace:
                with st.expander("MMR Selection Trace", expanded=True):
                    rows = [{
                        "Selection": r.get("selection_rank"),
                        "ID": r.get("id"),
                        "Document": r.get("document_id"),
                        "RRF rank": r.get("rrf_rank"),
                        "RRF": r.get("rrf_score"),
                        "Relevance": r.get("relevance"),
                        "Semantic similarity": r.get("semantic_similarity"),
                        "Redundancy": r.get("redundancy"),
                        "MMR": r.get("mmr_score"),
                    } for r in trace]
                    st.dataframe(rows, width="stretch", hide_index=True)

            doc_counts = mmr_debug.get("selected_document_counts", {})
            if doc_counts:
                with st.expander("MMR Document Diversity", expanded=False):
                    st.dataframe(
                        [{"Document": d, "Selected chunks": n}
                         for d, n in sorted(doc_counts.items())],
                        width="stretch", hide_index=True
                    )

        with st.expander("Otomatik teşhis", expanded=True):
            for row in gold_rows:
                gid = row["ID"]
                d_rank = row["Dense rank (full)"]
                b_rank = row["BM25 rank (full)"]
                r_rank = row["RRF rank"]
                in_pool = bool(row["In candidate pool"])

                if d_rank is None and b_rank is None:
                    st.error(f"{gid}: Gold chunk Dense ve BM25 indekslerinde bulunamadı.")
                elif not row["Dense Top N"] and not row["BM25 Top N"]:
                    st.warning(f"{gid}: Gold her iki retriever'ın Top-N dışında.")
                elif r_rank is None:
                    st.warning(f"{gid}: Gold RRF sonucunda görünmüyor.")
                elif not in_pool:
                    st.warning(f"{gid}: Gold candidate_k sınırının dışında; MMR'ye ulaşmadı.")
                else:
                    st.success(f"{gid}: Gold MMR candidate pool içinde; MMR seçimi incelenebilir.")

    with st.expander("Query Debug", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original query**")
            st.code(processed.get("original_query", query))
            st.markdown("**Matched FTR terms**")
            st.json(processed.get("matched_terms", []))
        with c2:
            st.markdown("**Rewritten query**")
            st.code(processed.get("rewritten_query", ""))
            st.markdown("**Expanded BM25 query**")
            st.code(processed.get("expanded_query", ""))

    with st.expander(f"Final MMR Results — {len(results)}", expanded=False):
        for rank, item in enumerate(results, 1):
            metadata = item.get("metadata") or {}
            with st.container(border=True):
                st.markdown(f"### #{rank} `{item.get('id', '?')}`")
                cols = st.columns(7)
                cols[0].metric("MMR", fmt_score(item.get("mmr_score")))
                cols[1].metric("Relevance", fmt_score(item.get("relevance")))
                cols[2].metric("RRF", fmt_score(item.get("rrf_score")))
                cols[3].metric("Dense rank", str(item.get("dense_rank")))
                cols[4].metric("BM25 rank", str(item.get("bm25_rank")))
                cols[5].metric("Dense", fmt_score(item.get("dense_score")))
                cols[6].metric("BM25", fmt_score(item.get("bm25_score")))
                st.caption(
                    f"{metadata.get('document_id', '?')} | page={metadata.get('page', '?')} | "
                    f"section={metadata.get('section', '?')} | role={item.get('chunk_role', metadata.get('chunk_role', '?'))}"
                )
                st.write(item.get("document", ""))

    if mmr_debug.get("semantic_query"):
        with st.expander("MMR Configuration / Semantic Query", expanded=False):
            st.code(mmr_debug.get("semantic_query", ""), language="text")
            st.json({
                "candidate_k": mmr_debug.get("candidate_k"),
                "final_k": mmr_debug.get("final_k"),
                "lambda_mult": mmr_debug.get("lambda_mult"),
                "rrf_weight": mmr_debug.get("rrf_weight"),
                "semantic_weight": mmr_debug.get("semantic_weight"),
                "max_chunks_per_document": mmr_debug.get("max_chunks_per_document"),
            })

    with st.expander(f"Hybrid/RRF Candidate Ranking — {len(hybrid_results)}", expanded=False):
        rows = []
        for rank, item in enumerate(hybrid_results, 1):
            rows.append({
                "Rank": rank,
                "ID": item.get("id"),
                "Gold": item.get("id") in gold_ids if gold_ids else None,
                "RRF": item.get("rrf_score"),
                "Dense rank": item.get("dense_rank"),
                "BM25 rank": item.get("bm25_rank"),
                "Dense score": item.get("dense_score"),
                "BM25 score": item.get("bm25_score"),
                "Page": (item.get("metadata") or {}).get("page"),
            })
        st.dataframe(rows, width="stretch", hide_index=True)

    with st.expander("Context sent to LLM", expanded=False):
        st.code(context_result.get("context_text", ""), language="text")

    with st.expander("Timing Summary", expanded=False):
        timing_rows = [
            {"Stage": "Query Processing", "Duration (ms)": timings.get("Query Processing")},
            {"Stage": "Dense + BM25 + RRF + MMR", "Duration (ms)": timings.get("Dense + BM25 + RRF + MMR")},
            {"Stage": "Context Assembly", "Duration (ms)": timings.get("Context Assembly")},
            {"Stage": "LLM TTFT", "Duration (ms)": timings.get("LLM TTFT")},
            {"Stage": "LLM Generation", "Duration (ms)": timings.get("LLM Generation")},
            {"Stage": "LLM Total", "Duration (ms)": timings.get("LLM Total")},
        ]
        st.dataframe(timing_rows, width="stretch", hide_index=True)

    # -----------------------------
    # 5) Persist the complete test result to an append-only file.
    #    This survives Streamlit reruns and can be shared at the end.
    # -----------------------------
    hybrid_log = []
    for rank, item in enumerate(hybrid_results, 1):
        hybrid_log.append({
            "Rank": rank,
            "ID": item.get("id"),
            "Gold": item.get("id") in gold_ids if gold_ids else None,
            "RRF": item.get("rrf_score"),
            "Dense rank": item.get("dense_rank"),
            "BM25 rank": item.get("bm25_rank"),
            "Dense score": item.get("dense_score"),
            "BM25 score": item.get("bm25_score"),
            "Page": (item.get("metadata") or {}).get("page"),
        })

    mmr_log = []
    for rank, item in enumerate(results, 1):
        metadata = item.get("metadata") or {}
        mmr_log.append({
            "Rank": rank,
            "ID": item.get("id"),
            "MMR": item.get("mmr_score", 0),
            "Relevance": item.get("relevance", 0),
            "RRF": item.get("rrf_score", 0),
            "Dense rank": item.get("dense_rank"),
            "BM25 rank": item.get("bm25_rank"),
            "Dense": item.get("dense_score", 0),
            "BM25": item.get("bm25_score", 0),
            "Source": f"{metadata.get('document_id', '?')} | page={metadata.get('page', '?')} | section={metadata.get('section', '?')}",
            "Text": item.get("document", ""),
        })

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_id": selected.get("id") if selected else "custom",
        "category": selected.get("category") if selected else "custom",
        "query": query.strip(),
        "gold_ids": sorted(gold_ids),
        "configuration": {
            "generation_model": generation_model,
            "compare_models": compare_models,
            "embedding_model": "qwen3-embedding:0.6b",
            "candidate_k": int(candidate_k),
            "final_k": int(final_k),
            "lambda_mult": float(lambda_mult),
            "rrf_weight": float(rrf_weight),
            "semantic_weight": float(semantic_weight),
            "dense_top_n": int(dense_top_n),
            "bm25_top_n": int(bm25_top_n),
            "rrf_k": int(rrf_k),
            "max_chunks_per_document": int(max_chunks_per_document),
            "language": language,
            "num_predict": int(num_predict),
        },
        "answer": answer,
        "generation_comparison": comparison_rows,
        "metrics": metrics,
        "query_debug": {
            "original_query": processed.get("original_query", query),
            "matched_terms": processed.get("matched_terms", []),
            "rewritten_query": processed.get("rewritten_query", ""),
            "expanded_query": processed.get("expanded_query", ""),
        },
        "retrieval_debug": retrieval_debug,
        "mmr_debug": mmr_debug,
        "hybrid_ranking": hybrid_log,
        "mmr_results": mmr_log,
        "timings": timings,
        "context": context_result.get("context_text", ""),
    }
    try:
        append_test_log(record)
        st.success(f"Test sonucu kaydedildi: {RESULTS_LOG_PATH.name}")
    except Exception as exc:
        st.warning(f"Test sonucu dosyaya yazılamadı: {exc}")

    # Session summary table for quick review.
    st.session_state.benchmark_history.append({
        "Benchmark ID": selected.get("id") if selected else "custom",
        "Category": selected.get("category") if selected else "custom",
        "Query": query.strip(),
        "Generation Model": generation_model,
        "Embedding Model": "qwen3-embedding:0.6b",
        "Retrieval": "Dense + BM25 + RRF + MMR",
        "Recall@1": metrics["recall@1"],
        "Recall@3": metrics["recall@3"],
        "Recall@5": metrics["recall@5"],
        "Recall@10": metrics["recall@10"],
        "MRR": metrics["mrr"],
        "nDCG@5": metrics["ndcg@5"],
        "nDCG@10": metrics["ndcg@10"],
        "First Relevant Rank": metrics["first_relevant_rank"],
        "TTFT (ms)": timings.get("LLM TTFT"),
        "LLM Total (ms)": timings.get("LLM Total"),
    })

    st.divider()
    st.subheader("Benchmark Tablosu — Gün Sonu")
    st.dataframe(st.session_state.benchmark_history, width="stretch", hide_index=True)
    st.caption("Bu tablo Streamlit oturumu boyunca birikir. Sayfayı kapatmadan tüm testleri çalıştırıp gün sonunda CSV olarak dışa aktarabilirsin.")

    csv = __import__("pandas").DataFrame(st.session_state.benchmark_history).to_csv(index=False).encode("utf-8-sig")
    st.download_button("Benchmark CSV indir", csv, "ftr_rag_benchmark_results.csv", "text/csv")
    if RESULTS_LOG_PATH.exists():
        st.download_button(
            "Tüm test logunu indir",
            RESULTS_LOG_PATH.read_bytes(),
            "ftr_rag_test_log.md",
            "text/markdown",
        )
