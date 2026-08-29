"""
Tests for the ingest-once / ask-many contract and audio robustness.

Covers the two behaviours that were reported as problems:

  1. A normal TEXT query must search the existing index and must NOT require
     a file upload -- `query_file_path` stays None and retrieval still works.
  2. Audio ingestion must handle real speech, and must fail gracefully and
     honestly on non-speech input rather than inventing a transcript.
"""
from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from schemas import (
    ContentItem,
    IngestionResult,
    Modality,
    QueryModality,
    QueryRequest,
)

DEMO_AUDIO = settings.audio_dir / "project_meeting.wav"
DEMO_IMAGE = settings.images_dir / "project_dashboard.png"


def _ollama_up() -> bool:
    from llm import ollama_available
    return ollama_available()[0]


requires_ollama = pytest.mark.skipif(not _ollama_up(), reason="Ollama unavailable")


@pytest.fixture(scope="module")
def pipeline():
    from rag_pipeline import get_pipeline
    return get_pipeline()


# --------------------------------------------------------------- fixtures

def _write_tone_wav(path: Path, seconds: int = 6, sample_rate: int = 16000) -> Path:
    """Melodic tones -- audio with no speech in it at all."""
    notes = [261.63, 329.63, 392.0, 523.25]
    frames = []
    for index in range(sample_rate * seconds):
        t = index / sample_rate
        freq = notes[int(t / (seconds / len(notes))) % len(notes)]
        value = 0.4 * math.sin(2 * math.pi * freq * t)
        frames.append(struct.pack("<h", int(value * 20000)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(frames))
    return path


def _write_silence_wav(path: Path, seconds: int = 4, sample_rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * sample_rate * seconds)
    return path


# ------------------------------------------------- TEXT query needs no file

@pytest.mark.live_ollama
@requires_ollama
@pytest.mark.skipif(not DEMO_IMAGE.exists(), reason="run create_demo_data.py first")
def test_text_query_searches_index_without_any_upload(pipeline):
    """The core ingest-once / ask-many guarantee."""
    request = QueryRequest(
        query="What was budget utilisation in 2024?",
        top_k=10,
        rerank_top_k=5,
    )
    assert request.query_modality is QueryModality.TEXT
    assert request.query_file_path is None, "a text query must carry no file"

    response = pipeline.query(request)
    assert response.retrieved_items, "text query must retrieve from the index"
    assert response.abstained is False
    assert response.citations
    for citation in response.citations:
        assert pipeline.store.get_item(citation.item_id) is not None


def test_text_query_representation_is_the_question_itself(pipeline):
    """No file means the embedded representation is just the question -- no
    hidden re-ingestion step."""
    request = QueryRequest(query="budget utilisation 2024")
    assert pipeline.build_query_representation(request) == "budget utilisation 2024"


def test_text_query_never_excludes_any_source(pipeline):
    """Self-exclusion applies only to file-based queries; a text question must
    be able to reach every indexed source."""
    request = QueryRequest(query="anything at all")
    assert pipeline._self_source_ids(request) == set()


@pytest.mark.skipif(not DEMO_IMAGE.exists(), reason="run create_demo_data.py first")
def test_file_query_excludes_only_its_own_source(pipeline):
    """An image used as a query is excluded from its own results, and that
    exclusion is derived from content -- not from the upload path."""
    request = QueryRequest(
        query="Find related material.",
        query_modality=QueryModality.IMAGE,
        query_file_path=str(DEMO_IMAGE),
    )
    excluded = pipeline._self_source_ids(request)
    assert len(excluded) == 1


def test_empty_text_query_is_rejected(pipeline):
    response = pipeline.query(QueryRequest(query="   "))
    assert response.abstained is True
    assert response.error_code.value == "INVALID_QUERY"


# ------------------------------------------------------- audio robustness

@pytest.mark.live_media
@pytest.mark.slow
@pytest.mark.skipif(not DEMO_AUDIO.exists(), reason="no demo speech audio")
def test_real_speech_produces_audio_items_with_timestamps():
    from audio import process_audio

    result = process_audio(str(DEMO_AUDIO))
    assert isinstance(result, IngestionResult)
    assert result.success, result.error
    assert result.items

    for item in result.items:
        assert isinstance(item, ContentItem)
        assert item.modality is Modality.AUDIO
        assert item.content.strip(), "transcript must not be empty"
        assert item.location.timestamp_start is not None
        assert item.location.timestamp_end is not None
        assert item.location.timestamp_end > item.location.timestamp_start
        assert item.vector_id is None, "vector_id is assigned at index time"

    starts = [i.location.timestamp_start for i in result.items]
    assert starts == sorted(starts)


@pytest.mark.live_media
@pytest.mark.slow
def test_music_fails_gracefully_without_inventing_a_transcript(tmp_path):
    """A song must not be indexed as though it were speech."""
    from audio import process_audio

    result = process_audio(str(_write_tone_wav(tmp_path / "music.wav")))
    assert result.success is False
    assert result.items == [], "no fabricated transcript may be produced"
    assert result.error_code.value == "TRANSCRIPTION_FAILED"
    # The message must say which problem it was, not just "failed".
    assert "no speech" in result.error.lower()


@pytest.mark.live_media
@pytest.mark.slow
def test_silence_fails_gracefully(tmp_path):
    from audio import process_audio

    result = process_audio(str(_write_silence_wav(tmp_path / "silence.wav")))
    assert result.success is False
    assert result.items == []


def test_low_confidence_segments_are_discarded(monkeypatch, tmp_path):
    """Whisper emitting confident-looking text for non-speech must be filtered
    on no_speech_prob / avg_logprob rather than indexed."""
    import audio

    class FakeSegment:
        def __init__(self, start, end, text, no_speech_prob, avg_logprob):
            self.start, self.end, self.text = start, end, text
            self.no_speech_prob, self.avg_logprob = no_speech_prob, avg_logprob

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            return ([
                FakeSegment(0.0, 5.0, "Real speech about the annual budget report.",
                            0.05, -0.3),
                FakeSegment(5.0, 9.0, "Thank you. Thank you. Thank you.", 0.95, -0.4),
                FakeSegment(9.0, 12.0, "gibberish from music", 0.10, -2.5),
            ], None)

    monkeypatch.setattr(audio, "_get_model", lambda: FakeModel())
    segments = audio.transcribe(tmp_path / "irrelevant.wav")

    assert len(segments) == 1
    assert "annual budget report" in segments[0]["text"]


def test_unsupported_audio_extension_rejected(tmp_path):
    from audio import process_audio

    bad = tmp_path / "clip.aiff"
    bad.write_bytes(b"not audio")
    result = process_audio(str(bad))
    assert result.success is False
    assert result.error_code.value == "UNSUPPORTED_FILE_TYPE"


# ------------------------------------------------------ image regression

@pytest.mark.live_ollama
@requires_ollama
@pytest.mark.skipif(not DEMO_IMAGE.exists(), reason="run create_demo_data.py first")
def test_image_contract_unchanged():
    """Guards the image pipeline against regression from the audio/UI work."""
    from images import process_image

    result = process_image(str(DEMO_IMAGE))
    assert result.success
    item = result.items[0]
    assert item.modality is Modality.IMAGE
    assert item.content.strip()
    assert item.location.image_path and Path(item.location.image_path).exists()
    assert item.metadata["ocr_text"]
    assert item.metadata["visual_description"]
    assert item.vector_id is None
    # content must be the semantic description, not merely the OCR dump
    assert item.metadata["visual_description"][:40] in item.content
