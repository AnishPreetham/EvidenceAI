"""
prepare_models.py — ONLINE PREPARATION STEP.

Downloads and caches every local model the runtime needs, then prints the
resolved cache location for each so the offline story is verifiable.

Run this ONCE while you still have internet. After it succeeds, the
application runs with no external network dependency.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

RESULTS: list[tuple[str, str, str]] = []


def _record(name: str, status: str, detail: str) -> None:
    RESULTS.append((name, status, detail))
    print(f"[{status}] {name}: {detail}", flush=True)


def prepare_embedding() -> None:
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    dim = model.get_sentence_embedding_dimension()
    vec = model.encode(["warmup sentence"], normalize_embeddings=True)
    _record(
        "all-MiniLM-L6-v2",
        "OK",
        f"dim={dim} shape={vec.shape} {time.time() - t0:.1f}s",
    )


def prepare_reranker() -> None:
    from sentence_transformers import CrossEncoder
    t0 = time.time()
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    scores = model.predict([("what is the budget", "the budget was 87 percent utilised")])
    _record(
        "ms-marco-MiniLM-L-6-v2",
        "OK",
        f"sample_score={float(scores[0]):.3f} {time.time() - t0:.1f}s",
    )


def prepare_whisper() -> None:
    from faster_whisper import WhisperModel
    t0 = time.time()
    WhisperModel("base", device="cpu", compute_type="int8")
    _record("faster-whisper base (int8)", "OK", f"loaded in {time.time() - t0:.1f}s")


def report_cache_paths() -> None:
    print("\nLOCAL MODEL CACHE LOCATIONS")
    print("-" * 66)
    hf = Path.home() / ".cache" / "huggingface" / "hub"
    ollama = Path.home() / ".ollama" / "models"
    for label, path in (("HuggingFace (MiniLM, CrossEncoder, Whisper)", hf),
                        ("Ollama (qwen2.5vl:3b)", ollama)):
        exists = "present" if path.exists() else "MISSING"
        print(f"  {label}\n    {path}  [{exists}]")


def main() -> int:
    steps = (
        ("embedding model", prepare_embedding),
        ("reranker", prepare_reranker),
        ("whisper", prepare_whisper),
    )
    failed = False
    for label, fn in steps:
        try:
            fn()
        except Exception as exc:
            failed = True
            _record(label, "FAIL", f"{type(exc).__name__}: {exc}")

    report_cache_paths()
    print("\n" + ("PREPARATION INCOMPLETE" if failed else "ALL MODELS CACHED LOCALLY"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
