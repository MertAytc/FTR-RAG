from __future__ import annotations

import os
from pathlib import Path
from typing import List

import ollama
import torch


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "FTR_EMBEDDING_MODEL",
    "qwen3-embedding:0.6b",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# LOCAL HUGGING FACE MODELS
# ============================================================

HF_MODEL_PATHS = {
    "bge-m3": PROJECT_ROOT / "models" / "bge-m3",
    "BAAI/bge-m3": PROJECT_ROOT / "models" / "bge-m3",

    "multilingual-e5-large": (
        PROJECT_ROOT / "models" / "multilingual-e5-large"
    ),
    "intfloat/multilingual-e5-large": (
        PROJECT_ROOT / "models" / "multilingual-e5-large"
    ),
}


# ============================================================
# GPU / DEVICE
# ============================================================

EMBEDDING_DEVICE = os.getenv(
    "FTR_EMBEDDING_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv("FTR_EMBEDDING_BATCH_SIZE", "32")
)


# ============================================================
# MODEL CACHE
# ============================================================

_ST_MODEL = None
_ST_MODEL_KEY = None


def _is_ollama_model(model_name: str) -> bool:
    return (
        model_name.startswith("qwen3-embedding:")
        or model_name.startswith("ollama:")
    )


def _resolve_hf_model_path(model_name: str) -> Path:
    if model_name in HF_MODEL_PATHS:
        path = HF_MODEL_PATHS[model_name]
    else:
        candidate = Path(model_name)

        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate

        path = candidate

    if not path.exists():
        raise FileNotFoundError(
            f"Embedding modeli bulunamadı: {path}\n"
            "Önce modeli local olarak models/ klasörüne indirin."
        )

    return path


def _get_sentence_transformer(model_name: str):
    global _ST_MODEL, _ST_MODEL_KEY

    if _ST_MODEL is not None and _ST_MODEL_KEY == model_name:
        return _ST_MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "BGE-M3 veya multilingual-e5-large kullanmak için "
            "sentence-transformers kurulmalı:\n"
            "pip install sentence-transformers"
        ) from exc

    model_path = _resolve_hf_model_path(model_name)

    device = EMBEDDING_DEVICE

    if device == "cuda" and not torch.cuda.is_available():
        print(
            "UYARI: CUDA seçildi fakat PyTorch CUDA kullanamıyor. "
            "CPU'ya geçiliyor."
        )
        device = "cpu"

    print(f"Local embedding modeli yükleniyor: {model_path}")
    print(f"Embedding device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA kullanılabilir: {torch.cuda.is_available()} | "
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        )

        torch.set_float32_matmul_precision("high")

    _ST_MODEL = SentenceTransformer(
        str(model_path),
        device=device,
        local_files_only=True,
    )

    _ST_MODEL_KEY = model_name

    print(
        f"Local embedding modeli hazır. Device: {device}"
    )

    return _ST_MODEL


# Qwen3-Embedding uses asymmetric retrieval prompts: queries receive a task
# instruction while indexed passages are embedded without that instruction.
# This exact format follows the model's official retrieval example.
QWEN3_EMBEDDING_QUERY_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def _qwen3_query_instruct(query: str) -> str:
    return f"Instruct: {QWEN3_EMBEDDING_QUERY_TASK}\n Query:{query}"


def embed_texts(
    texts: List[str],
    input_type: str = "document",
) -> List[List[float]]:

    if not texts:
        return []

    if input_type not in {"query", "document"}:
        raise ValueError(
            "input_type 'query' veya 'document' olmalıdır."
        )

    # --------------------------------------------------------
    # QWEN → OLLAMA
    # --------------------------------------------------------

    if _is_ollama_model(EMBEDDING_MODEL):
        model_name = EMBEDDING_MODEL.removeprefix("ollama:")

        ollama_texts = texts
        if input_type == "query" and model_name.startswith("qwen3-embedding"):
            ollama_texts = [_qwen3_query_instruct(text) for text in texts]

        response = ollama.embed(
            model=model_name,
            input=ollama_texts,
        )

        return response.embeddings

    # --------------------------------------------------------
    # BGE-M3 / E5 → LOCAL GPU
    # --------------------------------------------------------

    model = _get_sentence_transformer(
        EMBEDDING_MODEL
    )

    # E5 için query / passage prefix
    if "e5" in EMBEDDING_MODEL.lower():
        prefix = (
            "query: "
            if input_type == "query"
            else "passage: "
        )

        texts = [
            prefix + text
            for text in texts
        ]

    actual_device = (
        EMBEDDING_DEVICE
        if EMBEDDING_DEVICE != "cuda"
        or torch.cuda.is_available()
        else "cpu"
    )

    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=actual_device,
    )

    return embeddings.tolist()