"""
verify_offline.py -- prove the runtime needs no internet.

Two things are checked, and they are different claims:

1. NO OUTBOUND SOCKETS. Every socket connection to a non-loopback address is
   blocked at the Python level for the duration of the run. If any component
   silently reaches for the HuggingFace hub or any other host, it raises here
   instead of quietly succeeding while you still have Wi-Fi on.

2. THE PIPELINE STILL WORKS. Models load, ingestion runs, retrieval,
   reranking, relationships, generation and citations all complete.

Loopback (127.0.0.1) stays allowed, because Ollama runs there. That is a
local process, not the internet, and this script says so rather than
pretending zero sockets are opened.

Run:  .venv\\Scripts\\python scripts\\verify_offline.py

For a belt-and-braces check, disable your network adapter and run it again:
both this guard AND the real network will then be down.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force the HF stack into offline mode before anything imports it.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BLOCKED: list[str] = []
LOOPBACK_PREFIXES = ("127.", "::1", "localhost")


class OutboundNetworkBlocked(RuntimeError):
    pass


def _install_network_guard() -> None:
    """Allow loopback, refuse everything else."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _is_loopback(address) -> bool:
        if not isinstance(address, tuple) or not address:
            return False
        host = str(address[0])
        return any(host.startswith(prefix) for prefix in LOOPBACK_PREFIXES)

    def guarded_connect(self, address):
        if not _is_loopback(address):
            BLOCKED.append(str(address))
            raise OutboundNetworkBlocked(f"outbound connection blocked: {address}")
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if not _is_loopback(address):
            BLOCKED.append(str(address))
            raise OutboundNetworkBlocked(f"outbound connection blocked: {address}")
        return real_connect_ex(self, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex


CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        CHECKS.append((name, True, detail))
        print(f"[ OK ] {name}: {detail}", flush=True)
    except Exception as exc:
        CHECKS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)


def main() -> int:
    _install_network_guard()
    print("Evidence AI -- OFFLINE RUNTIME VERIFICATION")
    print("Outbound (non-loopback) sockets are blocked for this entire run.")
    print("=" * 74)

    from config import settings

    def embeddings_load():
        from embeddings import embed_query, embedding_dimension
        dim = embedding_dimension()
        vector = embed_query("offline check")
        return f"MiniLM loaded from cache, dim={dim}, vector={vector.shape}"

    def reranker_load():
        from reranker import get_model
        scores = get_model().predict([("budget", "budget utilisation was 87 percent")])
        return f"CrossEncoder loaded from cache, score={float(scores[0]):.3f}"

    def whisper_load():
        from audio import _get_model
        _get_model()
        return f"faster-whisper '{settings.whisper_model_size}' loaded from cache"

    def ollama_local():
        from llm import ollama_available
        ok, detail = ollama_available()
        if not ok:
            raise RuntimeError(detail)
        return f"{detail} (via loopback {settings.ollama_host})"

    def index_loads():
        from vector_store import VectorStore
        stats = VectorStore().stats()
        if stats["items"] == 0:
            raise RuntimeError("index is empty -- ingest data before verifying")
        return (f"{stats['sources']} sources, {stats['items']} items, "
                f"{stats['vectors']} vectors, {stats['relationships']} relationships")

    def full_query():
        from rag_pipeline import get_pipeline
        from schemas import QueryRequest
        response = get_pipeline().query(QueryRequest(
            query="What evidence do we have about Project Alpha's 2024 development?",
            top_k=10, rerank_top_k=5,
        ))
        if response.abstained:
            raise RuntimeError(f"unexpected abstention: {response.answer}")
        if not response.citations:
            raise RuntimeError("no citations produced")
        modalities = sorted({r.item.modality.value for r in response.retrieved_items})
        return (f"answered with {len(response.citations)} citations across "
                f"modalities {modalities}")

    def abstention_works():
        from rag_pipeline import get_pipeline
        from schemas import QueryRequest
        response = get_pipeline().query(QueryRequest(
            query="What was Project Alpha's budget in 2040?", top_k=10, rerank_top_k=5,
        ))
        if not response.abstained:
            raise RuntimeError("should have abstained")
        return "abstained correctly on an unsupported question"

    def ocr_local():
        import shutil
        if not (Path(settings.tesseract_cmd).exists() or shutil.which("tesseract")):
            raise RuntimeError("Tesseract not found")
        return "Tesseract binary present (local process, no network)"

    check("embedding model loads offline", embeddings_load)
    check("reranker loads offline", reranker_load)
    check("whisper loads offline", whisper_load)
    check("tesseract available", ocr_local)
    check("ollama reachable on loopback", ollama_local)
    check("persistent index loads", index_loads)
    check("end-to-end query", full_query)
    check("abstention", abstention_works)

    print("=" * 74)
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"{passed}/{len(CHECKS)} checks passed")

    if BLOCKED:
        print("\nOUTBOUND CONNECTION ATTEMPTS (these break the offline claim):")
        for address in sorted(set(BLOCKED)):
            print(f"  - {address}")
    else:
        print("\nNo outbound network connections were attempted.")
        print("The only sockets used were loopback to Ollama.")

    ok = passed == len(CHECKS) and not BLOCKED
    print("\nOFFLINE RUNTIME: " + ("VERIFIED" if ok else "NOT VERIFIED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
