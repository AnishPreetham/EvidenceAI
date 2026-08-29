"""Run the six demonstration scenarios end to end against the live system."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citations import render_citation
from config import settings
from rag_pipeline import get_pipeline
from schemas import InferenceMode, QueryModality, QueryRequest

PIPELINE = get_pipeline()


def show(title: str, request: QueryRequest) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    response = PIPELINE.query(request)

    print(f"ABSTAINED : {response.abstained}")
    if response.confidence is not None:
        print(f"CONFIDENCE: {response.confidence:.3f}")
    print(f"ANSWER    : {response.answer}\n")

    modalities = sorted({r.item.modality.value for r in response.retrieved_items})
    print(f"MODALITIES IN EVIDENCE: {modalities}")

    if response.citations:
        print("CITATIONS:")
        for citation in response.citations:
            print(f"  {render_citation(citation)}")

    cross = [r for r in response.relationships if r.metadata.get("cross_modal")]
    if cross:
        print(f"CROSS-MODAL LINKS: {len(cross)}")
        for rel in cross[:3]:
            print(f"  {rel.relationship_type.value} conf={rel.confidence:.3f}")

    total = response.latency_ms.get("total_ms")
    if total:
        print(f"LATENCY: {total / 1000:.1f}s")


def main() -> None:
    show("TEST 1 - Cross-modal text query", QueryRequest(
        query="What evidence do we have about Project Alpha's 2024 development?",
        top_k=10, rerank_top_k=5,
    ))

    show("TEST 2 - IMAGE query -> documents and audio", QueryRequest(
        query="Find documents and audio recordings related to this screenshot.",
        query_modality=QueryModality.IMAGE,
        query_file_path=str(settings.images_dir / "project_dashboard.png"),
    ))

    show("TEST 3 - AUDIO query -> documents and images", QueryRequest(
        query="What information in the reports corresponds to this recording?",
        query_modality=QueryModality.AUDIO,
        query_file_path=str(settings.audio_dir / "project_meeting.wav"),
    ))

    show("TEST 4 - DOCUMENT query, grounded answer", QueryRequest(
        query="What were the main findings in this document?",
        query_modality=QueryModality.DOCUMENT,
        query_file_path=str(settings.documents_dir / "project_report.pdf"),
    ))

    show("TEST 5 - DOCUMENT query -> screenshots and recordings", QueryRequest(
        query="Find screenshots and recordings related to this document.",
        query_modality=QueryModality.DOCUMENT,
        query_file_path=str(settings.documents_dir / "project_report.pdf"),
    ))

    show("TEST 6 - Unanswerable question (expect ABSTENTION)", QueryRequest(
        query="What was Project Alpha's budget in 2040?",
    ))

    show("TEST 7 - Direct image inspection (LOCAL_VISION)", QueryRequest(
        query="What does the dashboard show about quarterly records processed?",
        inference_mode=InferenceMode.LOCAL_VISION,
    ))


if __name__ == "__main__":
    main()
