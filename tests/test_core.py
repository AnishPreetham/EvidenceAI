"""
Unit tests for schemas, chunking, document/CSV extraction, embeddings,
vector store persistence, retrieval, reranking, relationships and citations.

These run without Ollama. Tests that need the live LLM are marked
`live_ollama` and live in test_integration.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citations import build_citations, render_citation, split_sentences
from documents import chunk_text, process_document
from embeddings import embed_query, embed_texts, embedding_dimension
from relationships import compute_related_to
from schemas import (
    ContentItem,
    IngestionResult,
    Modality,
    RerankedResult,
    Relationship,
    RelationshipType,
    Source,
    SourceLocation,
    SourceType,
    content_hash,
    derive_id,
)
from vector_store import VectorStore


# ---------------------------------------------------------------- schemas

def test_deterministic_ids_are_stable():
    assert derive_id("a", 1) == derive_id("a", 1)
    assert derive_id("a", 1) != derive_id("a", 2)


def test_content_hash_is_content_addressed():
    assert content_hash(b"hello") == content_hash(b"hello")
    assert content_hash(b"hello") != content_hash(b"world")


def test_location_renders_page_and_timestamp():
    assert SourceLocation(page_number=14).human() == "Page 14"
    assert SourceLocation(timestamp_start=252.0, timestamp_end=274.0).human() == "04:12-04:34"
    assert SourceLocation().human() == ""


def test_relationship_confidence_bounds_are_enforced():
    with pytest.raises(Exception):
        Relationship(
            relationship_id="r", source_item_id="a", target_item_id="b",
            relationship_type=RelationshipType.RELATED_TO, confidence=1.4,
        )


def test_content_item_rejects_unknown_fields():
    with pytest.raises(Exception):
        ContentItem(
            item_id="i", source_id="s", modality=Modality.TEXT,
            content="x", bogus_field=1,
        )


# --------------------------------------------------------------- chunking

def test_chunking_respects_size_and_overlaps():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)


def test_short_text_is_one_chunk():
    assert chunk_text("short text") == ["short text"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ") == []


# ------------------------------------------------------ document ingestion

def test_txt_ingestion_produces_text_items(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("Project Alpha reached 99.4 percent accuracy in 2024. " * 20)
    result = process_document(str(path))
    assert result.success
    assert result.items
    assert all(i.modality is Modality.TEXT for i in result.items)
    assert result.source.source_type is SourceType.TXT


def test_csv_ingestion_preserves_row_index(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("quarter,records\nQ1,240000\nQ2,310000\n")
    result = process_document(str(path))
    assert result.success
    assert len(result.items) == 2
    assert result.items[0].metadata["row_index"] == 0
    assert "quarter" in result.items[0].content


def test_pdf_ingestion_preserves_page_numbers(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for n in range(3):
        page = doc.new_page()
        page.insert_text((60, 80), f"Page {n + 1} discusses budget utilisation "
                                   f"and accuracy results for quarter {n + 1}.")
    doc.save(str(path))
    doc.close()

    result = process_document(str(path))
    assert result.success
    pages = {i.location.page_number for i in result.items}
    assert pages == {1, 2, 3}


def test_docx_uses_sections_never_page_numbers(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "notes.docx"
    document = docx.Document()
    document.add_heading("Budget", level=1)
    document.add_paragraph("Budget utilisation closed at 87 percent for 2024.")
    document.save(str(path))

    result = process_document(str(path))
    assert result.success
    assert all(i.location.page_number is None for i in result.items)
    assert any(i.location.section for i in result.items)


def test_unsupported_extension_fails_cleanly(tmp_path):
    path = tmp_path / "thing.xyz"
    path.write_text("data")
    result = process_document(str(path))
    assert result.success is False
    assert result.error_code is not None
    assert result.items == []


def test_missing_file_fails_cleanly():
    result = process_document("does/not/exist.pdf")
    assert result.success is False
    assert isinstance(result, IngestionResult)


# ------------------------------------------------------------- embeddings

def test_embedding_dimension_is_384():
    assert embedding_dimension() == 384


def test_embeddings_are_normalized_and_batched():
    vectors = embed_texts(["alpha project", "budget report", "audio segment"])
    assert vectors.shape == (3, 384)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_semantically_close_text_scores_higher():
    a = embed_query("classification accuracy reached 99.4 percent")
    close = embed_query("the model achieved 99.4% accuracy")
    far = embed_query("a recipe for chocolate cake")
    assert float(a @ close) > float(a @ far)


def test_embedding_cache_returns_identical_vectors():
    first = embed_query("repeatable text")
    second = embed_query("repeatable text")
    assert np.array_equal(first, second)


# ------------------------------------------------ vector store persistence

def _item(n: int, source_id: str = "src1") -> ContentItem:
    return ContentItem(
        item_id=f"item{n}", source_id=source_id, modality=Modality.TEXT,
        content=f"Evidence number {n} about project alpha budget and accuracy.",
        location=SourceLocation(page_number=n),
    )


def _source(source_id: str = "src1") -> Source:
    return Source(
        source_id=source_id, filename="report.pdf",
        source_type=SourceType.PDF, file_path="/tmp/report.pdf",
    )


def test_store_insert_search_and_reload(tmp_path):
    index_path, db_path = tmp_path / "f.index", tmp_path / "s.db"
    store = VectorStore(index_path, db_path)
    store.add_source(_source())
    items = [_item(i) for i in range(1, 6)]
    stored = store.add_items(items, embed_texts([i.content for i in items]))
    assert len(stored) == 5
    assert all(i.vector_id is not None for i in stored)

    hits = store.search(embed_query("project alpha budget"), top_k=3)
    assert len(hits) == 3
    store.close()

    # Reopen from disk -- this is the restart-persistence guarantee.
    reopened = VectorStore(index_path, db_path)
    assert reopened.stats()["items"] == 5
    assert reopened.stats()["vectors"] == 5
    assert reopened.get_item("item1") is not None
    assert reopened.get_source("src1").filename == "report.pdf"
    reopened.close()


def test_reingesting_identical_items_is_idempotent(tmp_path):
    store = VectorStore(tmp_path / "f.index", tmp_path / "s.db")
    store.add_source(_source())
    items = [_item(i) for i in range(1, 4)]
    vectors = embed_texts([i.content for i in items])

    assert len(store.add_items(items, vectors)) == 3
    assert len(store.add_items([_item(i) for i in range(1, 4)], vectors)) == 0
    assert store.stats()["vectors"] == 3
    store.close()


def test_relationships_round_trip(tmp_path):
    store = VectorStore(tmp_path / "f.index", tmp_path / "s.db")
    store.add_source(_source())
    items = [_item(1), _item(2)]
    store.add_items(items, embed_texts([i.content for i in items]))
    store.add_relationships([Relationship(
        relationship_id="r1", source_item_id="item1", target_item_id="item2",
        relationship_type=RelationshipType.SAME_PAGE, confidence=1.0,
    )])
    found = store.get_relationships_for_items(["item1"])
    assert len(found) == 1
    assert found[0].relationship_type is RelationshipType.SAME_PAGE
    store.close()


# ------------------------------------------------- retrieval and reranking

def test_retriever_ranks_semantically(tmp_path):
    from retriever import Retriever

    store = VectorStore(tmp_path / "f.index", tmp_path / "s.db")
    store.add_source(_source())
    items = [
        ContentItem(item_id="a", source_id="src1", modality=Modality.TEXT,
                    content="Budget utilisation closed at 87 percent for 2024."),
        ContentItem(item_id="b", source_id="src1", modality=Modality.TEXT,
                    content="A recipe for baking sourdough bread at home."),
    ]
    store.add_items(items, embed_texts([i.content for i in items]))

    results = Retriever(store).retrieve("how much of the budget was used?", top_k=2)
    assert results[0].item.item_id == "a"
    assert results[0].rank == 1
    store.close()


def test_retriever_excludes_named_sources(tmp_path):
    from retriever import Retriever

    store = VectorStore(tmp_path / "f.index", tmp_path / "s.db")
    store.add_source(_source("src1"))
    store.add_source(_source("src2"))
    items = [_item(1, "src1"), _item(2, "src2")]
    store.add_items(items, embed_texts([i.content for i in items]))

    results = Retriever(store).retrieve(
        "project alpha", top_k=5, exclude_source_ids={"src1"}
    )
    assert all(r.item.source_id != "src1" for r in results)
    store.close()


def test_reranker_orders_and_preserves_retrieval_score():
    from reranker import Reranker, score_to_confidence
    from schemas import RetrievalResult

    candidates = [
        RetrievalResult(item=ContentItem(
            item_id="x", source_id="s", modality=Modality.TEXT,
            content="Sourdough bread baking instructions."), score=0.9, rank=1),
        RetrievalResult(item=ContentItem(
            item_id="y", source_id="s", modality=Modality.TEXT,
            content="Budget utilisation closed at 87 percent."), score=0.8, rank=2),
    ]
    ranked = Reranker().rerank("what was budget utilisation?", candidates, top_k=2)
    assert ranked[0].item.item_id == "y"
    assert ranked[0].retrieval_score == 0.8  # carried through untouched
    assert ranked[0].rank == 1
    assert 0.0 <= score_to_confidence(ranked[0].rerank_score) <= 1.0


# ----------------------------------------------------------- relationships

def _reranked(item_id: str, source_id: str, content: str, modality=Modality.TEXT):
    return RerankedResult(
        item=ContentItem(item_id=item_id, source_id=source_id,
                         modality=modality, content=content),
        retrieval_score=0.8, rerank_score=2.0, rank=1,
    )


def test_related_to_respects_threshold_and_stays_in_bounds():
    evidence = [
        _reranked("a", "s1", "Classification accuracy reached 99.4 percent in 2024."),
        _reranked("b", "s2", "The model achieved 99.4% accuracy during 2024.",
                  Modality.IMAGE),
        _reranked("c", "s3", "Instructions for repairing a bicycle tyre."),
    ]
    high = compute_related_to(evidence, threshold=0.5)
    assert any({r.source_item_id, r.target_item_id} == {"a", "b"} for r in high)
    assert all(0.0 <= r.confidence <= 1.0 for r in high)
    assert compute_related_to(evidence, threshold=0.99) == []


def test_related_to_skips_same_source_pairs():
    evidence = [
        _reranked("a", "same", "Accuracy reached 99.4 percent."),
        _reranked("b", "same", "Accuracy reached 99.4 percent."),
    ]
    assert compute_related_to(evidence, threshold=0.1) == []


def test_related_to_needs_two_items():
    assert compute_related_to([_reranked("a", "s", "text")]) == []


# --------------------------------------------------------------- citations

def test_citations_map_to_real_items_only():
    evidence = [
        _reranked("a", "s1", "Budget utilisation closed at 87 percent for 2024."),
        _reranked("b", "s2", "Classification accuracy reached 99.4 percent."),
    ]
    citations, sentence_map = build_citations(
        "Budget utilisation was 87 percent. Accuracy reached 99.4 percent.",
        evidence, {"s1": "report.pdf", "s2": "notes.docx"},
    )
    assert citations
    item_ids = {c.item_id for c in citations}
    assert item_ids <= {"a", "b"}
    assert all(c.citation_id > 0 for c in citations)
    assert sentence_map


def test_citation_rendering_by_modality():
    from schemas import Citation

    pdf = Citation(citation_id=1, item_id="i", source_id="s",
                   filename="report.pdf", modality=Modality.TEXT,
                   location=SourceLocation(page_number=14))
    audio = Citation(citation_id=2, item_id="i", source_id="s",
                     filename="meeting.wav", modality=Modality.AUDIO,
                     location=SourceLocation(timestamp_start=252.0, timestamp_end=274.0))
    image = Citation(citation_id=3, item_id="i", source_id="s",
                     filename="shot.png", modality=Modality.IMAGE,
                     location=SourceLocation(image_path="/tmp/shot.png"))

    assert render_citation(pdf) == "[1] report.pdf - Page 14"
    assert render_citation(audio) == "[2] meeting.wav - 04:12-04:34"
    assert render_citation(image) == "[3] shot.png"


def test_empty_evidence_yields_no_citations():
    citations, mapping = build_citations("Some answer.", [], {})
    assert citations == []
    assert mapping == {}


def test_sentence_splitting_drops_fragments():
    sentences = split_sentences("This is a full sentence about budgets. Ok. "
                                "Another complete sentence about accuracy.")
    assert len(sentences) == 2
