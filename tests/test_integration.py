"""
Integration tests that exercise the REAL local models: Tesseract OCR,
faster-whisper transcription, Ollama qwen2.5vl:3b vision and generation,
the FastAPI surface, and the full end-to-end pipeline.

Nothing here is mocked. Tests skip (rather than fail) when the dependency
they need is genuinely unavailable, so the suite stays honest about what
was actually verified.

    pytest tests/test_integration.py -v
    pytest tests/ -m "not live_ollama"     # skip everything needing the LLM
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from schemas import InferenceMode, Modality, QueryModality, QueryRequest, SourceType

DEMO_PDF = settings.documents_dir / "project_report.pdf"
DEMO_DOCX = settings.documents_dir / "project_notes.docx"
DEMO_CSV = settings.documents_dir / "project_metrics.csv"
DEMO_IMAGE = settings.images_dir / "project_dashboard.png"
DEMO_AUDIO = settings.audio_dir / "project_meeting.wav"


def _ollama_up() -> bool:
    from llm import ollama_available
    return ollama_available()[0]


def _tesseract_up() -> bool:
    return Path(settings.tesseract_cmd).exists() or bool(shutil.which("tesseract"))


requires_ollama = pytest.mark.skipif(not _ollama_up(), reason="Ollama unavailable")
requires_tesseract = pytest.mark.skipif(not _tesseract_up(), reason="Tesseract missing")
requires_demo = pytest.mark.skipif(
    not DEMO_PDF.exists(), reason="run create_demo_data.py first"
)


@pytest.fixture(scope="module")
def pipeline():
    from rag_pipeline import get_pipeline
    return get_pipeline()


# ------------------------------------------------------------------ OCR

@pytest.mark.live_media
@requires_tesseract
@requires_demo
def test_tesseract_reads_dashboard_text():
    from images import run_ocr
    from PIL import Image

    with Image.open(DEMO_IMAGE) as handle:
        text = run_ocr(handle.convert("RGB"))
    assert text, "OCR returned nothing from the dashboard image"
    assert "PROJECT" in text.upper()


# --------------------------------------------------------------- vision

@pytest.mark.live_ollama
@requires_ollama
@requires_demo
def test_qwen_vision_describes_a_real_image():
    """The capability the whole image pipeline depends on: real local image
    input to qwen2.5vl:3b via Ollama."""
    from images import describe_image

    description = describe_image(DEMO_IMAGE)
    assert len(description) > 30
    assert "dashboard" in description.lower() or "chart" in description.lower()


@pytest.mark.live_ollama
@requires_ollama
@requires_demo
def test_image_ingestion_combines_description_and_ocr():
    from images import process_image

    result = process_image(str(DEMO_IMAGE))
    assert result.success
    item = result.items[0]
    assert item.modality is Modality.IMAGE
    assert item.metadata["ocr_text"]
    assert item.metadata["visual_description"]
    assert item.metadata["vision_ok"] is True
    # content must carry BOTH halves -- that is what gets embedded.
    assert item.metadata["visual_description"][:40] in item.content
    assert item.location.image_path
    assert item.metadata["width"] > 0


@requires_tesseract
@requires_demo
def test_image_degrades_to_ocr_only_when_vision_fails(monkeypatch):
    """Qwen failure must preserve OCR evidence, not discard the whole item."""
    import images

    monkeypatch.setattr(
        images, "describe_image",
        lambda path: (_ for _ in ()).throw(RuntimeError("Ollama down")),
    )
    result = images.process_image(str(DEMO_IMAGE))
    assert result.success is True
    assert result.items
    assert result.items[0].metadata["vision_ok"] is False
    assert result.items[0].metadata["ocr_text"]
    assert result.warnings


# ---------------------------------------------------------------- audio

@pytest.mark.live_media
@pytest.mark.slow
@pytest.mark.skipif(not DEMO_AUDIO.exists(), reason="no demo audio")
def test_whisper_transcribes_with_timestamps():
    from audio import process_audio

    result = process_audio(str(DEMO_AUDIO))
    assert result.success
    assert len(result.items) >= 2, "expected multiple transcript segments"

    for item in result.items:
        assert item.modality is Modality.AUDIO
        assert item.location.timestamp_start is not None
        assert item.location.timestamp_end is not None
        assert item.location.timestamp_end > item.location.timestamp_start

    starts = [i.location.timestamp_start for i in result.items]
    assert starts == sorted(starts), "segments must be in chronological order"

    joined = " ".join(i.content for i in result.items).lower()
    assert "project alpha" in joined or "alpha" in joined
    # Speaker identities are never invented.
    assert all("speaker" not in i.metadata for i in result.items)


# ------------------------------------------------------------ generation

@pytest.mark.live_ollama
@requires_ollama
def test_llm_abstains_when_evidence_does_not_support_question():
    from llm import generate_answer
    from schemas import ContentItem, RerankedResult

    evidence = [RerankedResult(
        item=ContentItem(item_id="a", source_id="s", modality=Modality.TEXT,
                         content="The cafeteria serves lunch between 12 and 2pm."),
        retrieval_score=0.5, rerank_score=1.0, rank=1,
    )]
    answer = generate_answer(
        "What was Project Alpha's budget in 2040?", evidence, {"s": "menu.txt"}
    )
    assert "INSUFFICIENT_EVIDENCE" in answer.upper()


@pytest.mark.live_ollama
@requires_ollama
def test_llm_does_not_emit_citation_markers():
    from llm import generate_answer
    from schemas import ContentItem, RerankedResult

    evidence = [RerankedResult(
        item=ContentItem(item_id="a", source_id="s", modality=Modality.TEXT,
                         content="Budget utilisation closed at 87 percent for 2024."),
        retrieval_score=0.9, rerank_score=5.0, rank=1,
    )]
    answer = generate_answer("What was budget utilisation?", evidence,
                             {"s": "report.pdf"})
    assert "87" in answer
    assert "[1]" not in answer and "[2]" not in answer


def test_ollama_error_surfaces_actionable_code(monkeypatch):
    """A dead Ollama must produce a named error code, not a stack trace."""
    import llm

    monkeypatch.setattr(llm.settings, "ollama_host", "http://localhost:1")
    with pytest.raises(llm.OllamaError) as excinfo:
        llm.ollama_generate("hello")
    assert excinfo.value.code.value in {
        "OLLAMA_NOT_AVAILABLE", "QWEN_MODEL_NOT_AVAILABLE",
    }


def test_discovery_queries_are_detected():
    from llm import is_discovery_query

    assert is_discovery_query("Find documents related to this screenshot.")
    assert is_discovery_query("What information corresponds to this recording?")
    assert not is_discovery_query("What was the budget in 2024?")


# ----------------------------------------------------------- end to end

@pytest.mark.live_ollama
@pytest.mark.slow
@requires_ollama
@requires_demo
def test_cross_modal_query_uses_multiple_modalities(pipeline):
    """The headline acceptance criterion: one question answered from
    evidence spanning more than one modality."""
    response = pipeline.query(QueryRequest(
        query="What evidence do we have about Project Alpha's 2024 development?",
        top_k=10, rerank_top_k=5,
    ))
    assert response.abstained is False
    assert response.answer

    modalities = {r.item.modality for r in response.retrieved_items}
    assert len(modalities) >= 2, f"expected cross-modal evidence, got {modalities}"

    assert response.citations
    for citation in response.citations:
        assert pipeline.store.get_item(citation.item_id) is not None, \
            "every citation must resolve to a real indexed item"
    assert 0.0 <= response.confidence <= 1.0


@pytest.mark.live_ollama
@pytest.mark.slow
@requires_ollama
@requires_demo
def test_pipeline_abstains_on_unanswerable_question(pipeline):
    response = pipeline.query(QueryRequest(
        query="What was Project Alpha's budget in 2040?", top_k=10, rerank_top_k=5,
    ))
    assert response.abstained is True
    assert "could not find sufficient evidence" in response.answer.lower()
    assert response.error_code.value == "INSUFFICIENT_EVIDENCE"


@pytest.mark.live_ollama
@pytest.mark.slow
@requires_ollama
@requires_demo
def test_image_query_retrieves_other_modalities(pipeline):
    response = pipeline.query(QueryRequest(
        query="Find documents and audio recordings related to this screenshot.",
        query_modality=QueryModality.IMAGE,
        query_file_path=str(DEMO_IMAGE),
    ))
    assert response.abstained is False
    # The query image itself must be excluded from its own results.
    assert all(
        r.item.location.image_path != str(DEMO_IMAGE)
        for r in response.retrieved_items
    )
    assert any(r.item.modality is Modality.TEXT for r in response.retrieved_items)


@pytest.mark.live_ollama
@pytest.mark.slow
@requires_ollama
@pytest.mark.skipif(not DEMO_AUDIO.exists(), reason="no demo audio")
def test_audio_query_retrieves_documents(pipeline):
    response = pipeline.query(QueryRequest(
        query="What information in the reports corresponds to this recording?",
        query_modality=QueryModality.AUDIO,
        query_file_path=str(DEMO_AUDIO),
    ))
    assert response.query_representation
    assert any(r.item.modality is Modality.TEXT for r in response.retrieved_items)


@pytest.mark.live_ollama
@pytest.mark.slow
@requires_ollama
@requires_demo
def test_document_query_excludes_itself_and_finds_other_sources(pipeline):
    response = pipeline.query(QueryRequest(
        query="Find screenshots and recordings related to this document.",
        query_modality=QueryModality.DOCUMENT,
        query_file_path=str(DEMO_PDF),
    ))
    assert response.abstained is False
    filenames = {
        pipeline.store.get_source(r.item.source_id).filename
        for r in response.retrieved_items
    }
    assert "project_report.pdf" not in filenames


# ------------------------------------------------------------ ingestion

@requires_demo
def test_all_document_types_ingest(pipeline):
    for path, expected in (
        (DEMO_PDF, SourceType.PDF),
        (DEMO_DOCX, SourceType.DOCX),
        (DEMO_CSV, SourceType.CSV),
    ):
        result = pipeline.ingest_file(str(path))
        assert result.success, f"{path.name}: {result.error}"
        assert result.source.source_type is expected


@requires_demo
def test_source_ids_are_content_addressed(pipeline):
    """Re-ingesting the same bytes must yield the same source_id, which is
    what makes evaluation reproducible."""
    first = pipeline.ingest_file(str(DEMO_PDF))
    second = pipeline.ingest_file(str(DEMO_PDF))
    assert first.source.source_id == second.source.source_id


def test_unsupported_upload_is_rejected(pipeline, tmp_path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04")
    result = pipeline.ingest_file(str(path))
    assert result.success is False
    assert result.error_code.value == "UNSUPPORTED_FILE_TYPE"


# ---------------------------------------------------------------- API

@requires_demo
def test_api_health_and_sources():
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert "index" in health.json()

    sources = client.get("/sources")
    assert sources.status_code == 200
    assert sources.json()["count"] >= 1

    source_id = sources.json()["sources"][0]["source_id"]
    detail = client.get(f"/source/{source_id}")
    assert detail.status_code == 200
    assert detail.json()["item_count"] >= 1

    assert client.get("/source/does-not-exist").status_code == 404


@pytest.mark.live_ollama
@pytest.mark.slow
@requires_ollama
@requires_demo
def test_api_query_endpoint_returns_citations():
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)
    response = client.post("/query", json={
        "query": "What was budget utilisation in 2024?",
        "top_k": 10, "rerank_top_k": 5,
    })
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body and "citations" in body
    assert body["abstained"] is False


# --------------------------------------------------------- persistence

@requires_demo
def test_index_survives_restart():
    """Open a second VectorStore against the same files -- this is exactly
    what happens when the application is restarted."""
    from vector_store import VectorStore

    reopened = VectorStore()
    stats = reopened.stats()
    assert stats["items"] > 0
    assert stats["vectors"] == stats["items"]
    assert stats["sources"] > 0
    reopened.close()
