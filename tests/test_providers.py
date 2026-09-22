"""
Tests for the offline/online inference provider layer.

Every online test here is MOCKED. Nothing in this file makes a real cloud
API call, and nothing depends on a real API key being present -- the key is
explicitly blanked or faked per test so results are identical on a machine
with credentials and one without.
"""
from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from providers import (
    AutoProvider,
    Backend,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    _redact,
    get_provider,
    resolve_backend,
    supports_vision,
)
from schemas import ContentItem, Modality, QueryRequest, RerankedResult

FAKE_KEY = "sk-test-FAKEKEY-0000000000000000000000000000"


@pytest.fixture
def evidence():
    return [RerankedResult(
        item=ContentItem(
            item_id="a", source_id="s", modality=Modality.TEXT,
            content="Budget utilisation closed at 87 percent for 2024.",
        ),
        retrieval_score=0.9, rerank_score=5.0, rank=1,
    )]


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", FAKE_KEY)
    return FAKE_KEY


@pytest.fixture
def without_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")


def _fake_response(payload: dict):
    class FakeResponse:
        def read(self):
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    return FakeResponse()


# ---------------------------------------------------- provider selection

def test_backend_resolution():
    assert resolve_backend("online") is Backend.ONLINE
    assert resolve_backend("OFFLINE") is Backend.OFFLINE
    assert resolve_backend(Backend.ONLINE) is Backend.ONLINE
    assert resolve_backend("auto") is Backend.AUTO
    # An unknown value must fall back safely to AUTO rather than raise.
    assert resolve_backend("nonsense") is Backend.AUTO


def test_provider_selection_returns_right_class():
    assert isinstance(get_provider(Backend.OFFLINE), OllamaProvider)
    assert isinstance(get_provider(Backend.ONLINE), OpenAIProvider)


def test_default_backend_comes_from_config(monkeypatch):
    monkeypatch.setattr(settings, "inference_backend", "online")
    assert resolve_backend(None) is Backend.ONLINE
    monkeypatch.setattr(settings, "inference_backend", "offline")
    assert resolve_backend(None) is Backend.OFFLINE


# --------------------------------------------------- online configuration

def test_online_config_loads_from_settings(with_key):
    provider = OpenAIProvider()
    assert provider.model_name == settings.online_model
    assert settings.online_base_url.startswith("http")
    assert settings.online_api_key_env == "OPENAI_API_KEY"


def test_missing_key_gives_friendly_readiness(without_key):
    readiness = OpenAIProvider().readiness()
    assert readiness.ready is False
    assert "isn't configured" in readiness.message
    # The variable to set must be named so the user can act on it.
    assert "OPENAI_API_KEY" in readiness.technical


def test_missing_key_generate_raises_friendly_error(without_key, evidence):
    with pytest.raises(ProviderError) as excinfo:
        OpenAIProvider().generate("q", evidence, {"s": "report.pdf"})
    assert "isn't configured" in excinfo.value.user_message


def test_present_key_reports_ready(monkeypatch, with_key):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _fake_response({"models": []}),
    )
    assert OpenAIProvider().readiness().ready is True


# ------------------------------------------- online works without Ollama

def test_online_provider_instantiates_without_ollama(monkeypatch, with_key):
    """Constructing and checking the online provider must not import or
    contact Ollama in any way."""
    import llm

    def explode(*_args, **_kwargs):
        raise AssertionError("online mode must not touch Ollama")

    monkeypatch.setattr(llm, "ollama_available", explode)
    monkeypatch.setattr(llm, "ollama_generate", explode)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _fake_response({"models": []}),
    )

    provider = OpenAIProvider()
    assert provider.readiness().ready is True


def test_online_health_does_not_probe_ollama(monkeypatch, with_key):
    """pipeline.health('online') must not call ollama_available()."""
    import llm
    from rag_pipeline import get_pipeline

    monkeypatch.setattr(
        llm, "ollama_available",
        lambda: (_ for _ in ()).throw(AssertionError("Ollama probed in online mode")),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _fake_response({"models": []}),
    )
    health = get_pipeline().health(Backend.ONLINE)
    assert health["backend"] == "online"
    assert health["inference"]["ready"] is True
    assert health["ollama"]["available"] is False
    assert "not required" in health["ollama"]["detail"].lower()


def test_offline_health_does_probe_ollama(monkeypatch):
    """The offline path must still use the real Ollama check."""
    import llm
    from rag_pipeline import get_pipeline

    called: list[bool] = []

    def fake_available():
        called.append(True)
        return True, "Ollama ready with test-model"

    monkeypatch.setattr(llm, "ollama_available", fake_available)
    health = get_pipeline().health(Backend.OFFLINE)
    assert called, "offline health must check Ollama"
    assert health["backend"] == "offline"


# ------------------------------------------------- online generation path

def test_online_generation_returns_answer(monkeypatch, with_key, evidence):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode())
        return _fake_response(
            {"choices": [{"message": {"content": "Budget utilisation was 87 percent."}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    answer = OpenAIProvider().generate(
        "What was budget utilisation?", evidence, {"s": "report.pdf"}
    )

    assert answer == "Budget utilisation was 87 percent."
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["model"] == settings.online_model
    # The retrieved evidence must actually reach the cloud model -- this is
    # RAG, not a bare chatbot call.
    prompt = captured["body"]["messages"][0]["content"]
    assert "87 percent" in prompt
    assert "report.pdf" in prompt


def test_online_uses_the_same_grounding_rules_as_offline(
    monkeypatch, with_key, evidence
):
    """Both backends must share llm.build_prompt so abstention and citation
    rules cannot drift between modes."""
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode())
        return _fake_response({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    OpenAIProvider().generate("q", evidence, {"s": "report.pdf"})

    from llm import build_prompt
    expected = build_prompt("q", evidence, {"s": "report.pdf"})
    assert captured["body"]["messages"][0]["content"] == expected
    assert "INSUFFICIENT_EVIDENCE" in expected


def test_online_sends_image_when_inspecting(monkeypatch, with_key, evidence, tmp_path):
    from PIL import Image

    image_path = tmp_path / "shot.png"
    Image.new("RGB", (32, 32), "white").save(image_path)

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode())
        return _fake_response({"choices": [{"message": {"content": "described"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    OpenAIProvider().generate("q", evidence, {"s": "x"}, inspect_image=image_path)

    content = captured["body"]["messages"][0]["content"]
    assert isinstance(content, list)
    kinds = {part["type"] for part in content}
    assert kinds == {"text", "image_url"}
    image_part = next(p for p in content if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


# --------------------------------------------------- online error mapping

def _http_error(code: int, body: bytes = b"{}"):
    return urllib.error.HTTPError(
        "https://api.openai.com/v1/chat/completions", code, "err", {},
        __import__("io").BytesIO(body),
    )


@pytest.mark.parametrize("code,expected", [
    (401, "could not authenticate"),
    (403, "could not authenticate"),
    (429, "rate-limited"),
    (404, "not available"),
    (500, "returned an error"),
])
def test_http_errors_map_to_friendly_messages(
    monkeypatch, with_key, evidence, code, expected
):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(code)),
    )
    with pytest.raises(ProviderError) as excinfo:
        OpenAIProvider().generate("q", evidence, {"s": "x"})
    assert expected in excinfo.value.user_message.lower()
    # The raw exception must never be the headline.
    assert "HTTPError" not in excinfo.value.user_message


def test_connectivity_failure_is_friendly(monkeypatch, with_key, evidence):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("no route")),
    )
    with pytest.raises(ProviderError) as excinfo:
        OpenAIProvider().generate("q", evidence, {"s": "x"})
    assert "currently unavailable" in excinfo.value.user_message.lower()
    assert "internet connection" in excinfo.value.user_message.lower()


def test_malformed_response_is_handled(monkeypatch, with_key, evidence):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _fake_response({"unexpected": True}),
    )
    with pytest.raises(ProviderError) as excinfo:
        OpenAIProvider().generate("q", evidence, {"s": "x"})
    assert "unexpected response" in excinfo.value.user_message.lower()


# ------------------------------------------------------------- security

def test_api_key_never_appears_in_error_messages(monkeypatch, with_key, evidence):
    """The key must not leak through the HTTP error body, the user message,
    or the technical detail."""
    leaky_body = json.dumps({"error": {"message": f"bad key {FAKE_KEY}"}}).encode()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(401, leaky_body)),
    )
    with pytest.raises(ProviderError) as excinfo:
        OpenAIProvider().generate("q", evidence, {"s": "x"})

    assert FAKE_KEY not in excinfo.value.user_message
    assert FAKE_KEY not in excinfo.value.technical
    assert FAKE_KEY not in str(excinfo.value)
    assert "REDACTED" in excinfo.value.technical


def test_redact_helper_masks_the_key(with_key):
    assert FAKE_KEY not in _redact(f"Authorization: Bearer {FAKE_KEY}")


def test_readiness_never_contains_the_key(with_key):
    readiness = OpenAIProvider().readiness()
    assert FAKE_KEY not in readiness.message
    assert FAKE_KEY not in readiness.technical


def test_env_example_has_no_real_key():
    text = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
    assert "OPENAI_API_KEY=" in text
    for line in text.splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            value = line.split("=", 1)[1].strip()
            assert value == "", f".env.example must hold no real key, found {value!r}"


def test_no_secrets_in_tracked_source():
    """Guards against a real key being pasted into tracked code.

    Matches an OpenAI-shaped token specifically -- a bare 'sk-' substring
    also occurs inside ordinary words such as 'ask-many'.
    """
    import re

    # sk- or sk-proj- followed by a long opaque run of key characters.
    pattern = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")
    root = Path(__file__).resolve().parent.parent

    suspicious = []
    for path in list(root.glob("*.py")) + list((root / "tests").glob("*.py")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for match in pattern.findall(line):
                if "FAKEKEY" in match or match.startswith("sk-test"):
                    continue  # the deliberate fixture value
                suspicious.append(f"{path.name}:{number}")
    assert not suspicious, f"possible hard-coded key at {suspicious}"


# ------------------------------------------------------- vision capability

def test_vision_support_per_backend(monkeypatch):
    assert supports_vision(Backend.OFFLINE) is True
    monkeypatch.setattr(settings, "online_model_supports_vision", False)
    assert supports_vision(Backend.ONLINE) is False
    monkeypatch.setattr(settings, "online_model_supports_vision", True)
    assert supports_vision(Backend.ONLINE) is True


# ------------------------------------------- pipeline contract under online

def test_online_query_preserves_response_contract(monkeypatch, with_key):
    """A full pipeline query on the online backend must return a valid
    QueryResponse with real citations drawn from the local index."""
    from rag_pipeline import get_pipeline
    from schemas import QueryResponse

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _fake_response(
            {"choices": [{"message": {"content":
                "Budget utilisation in 2024 was 87 percent."}}]}
        ),
    )
    pipeline = get_pipeline()
    if pipeline.store.stats()["items"] == 0:
        pytest.skip("index is empty; run scripts/ingest_demo.py")

    response = pipeline.query(
        QueryRequest(query="What was budget utilisation in 2024?"),
        backend=Backend.ONLINE,
    )
    assert isinstance(response, QueryResponse)
    assert response.abstained is False
    assert response.retrieved_items
    assert response.citations
    for citation in response.citations:
        assert pipeline.store.get_item(citation.item_id) is not None


def test_online_abstention_gate_fires_before_the_cloud_is_called(
    monkeypatch, with_key
):
    """Path 1: when reranking scores fall below ABSTENTION_THRESHOLD the
    pipeline abstains without ever calling the cloud model -- so an
    unanswerable question costs nothing and cannot be improvised."""
    from rag_pipeline import get_pipeline

    def explode(*_a, **_k):
        raise AssertionError("cloud model called despite insufficient evidence")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    pipeline = get_pipeline()
    if pipeline.store.stats()["items"] == 0:
        pytest.skip("index is empty; run scripts/ingest_demo.py")

    response = pipeline.query(
        QueryRequest(
            query="What is the airspeed velocity of an unladen swallow in Patagonia?"
        ),
        backend=Backend.ONLINE,
    )
    assert response.abstained is True
    assert response.error_code.value == "INSUFFICIENT_EVIDENCE"
    assert response.citations == []


def test_online_model_signalled_abstention_is_preserved(monkeypatch, with_key):
    """Path 2: evidence clears the gate but the model judges it insufficient.
    The online backend must honour that exactly as offline does, replacing
    the raw marker with the standard abstention message."""
    from rag_pipeline import ABSTENTION_MESSAGE, get_pipeline

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _fake_response(
            {"choices": [{"message": {"content": "INSUFFICIENT_EVIDENCE"}}]}
        ),
    )
    pipeline = get_pipeline()
    if pipeline.store.stats()["items"] == 0:
        pytest.skip("index is empty; run scripts/ingest_demo.py")

    response = pipeline.query(
        QueryRequest(query="What was Project Alpha's budget in 2040?"),
        backend=Backend.ONLINE,
    )
    assert response.abstained is True
    assert response.answer == ABSTENTION_MESSAGE
    assert response.error_code.value == "INSUFFICIENT_EVIDENCE"
    assert response.citations == []


def test_online_provider_error_surfaces_technical_detail_separately(
    monkeypatch, with_key
):
    """The friendly message goes in QueryResponse.answer; the raw detail is
    reachable via last_technical_detail() and never in the headline."""
    from rag_pipeline import get_pipeline, last_technical_detail

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(401)),
    )
    pipeline = get_pipeline()
    if pipeline.store.stats()["items"] == 0:
        pytest.skip("index is empty; run scripts/ingest_demo.py")

    response = pipeline.query(
        QueryRequest(query="What was budget utilisation in 2024?"),
        backend=Backend.ONLINE,
    )
    assert "could not authenticate" in response.answer.lower()
    assert response.abstained is True
    detail = last_technical_detail()
    assert "401" in detail
    assert FAKE_KEY not in detail


# ---------------------------------------------------- AUTO provider

def test_auto_provider_instantiates():
    provider = get_provider(Backend.AUTO)
    assert isinstance(provider, AutoProvider)
    assert provider.backend is Backend.AUTO


def test_auto_uses_online_when_key_present(monkeypatch, with_key, evidence):
    """AUTO routes to online when the key is configured."""
    AutoProvider._online_ready = None  # clear cache

    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["called"] = True
        return _fake_response({"choices": [{"message": {"content": "answer"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("llm.ollama_available", lambda: (_ for _ in ()).throw(
        AssertionError("offline must not be used when online is ready")
    ))

    result = AutoProvider().generate("q", evidence, {"s": "f.pdf"})
    assert result == "answer"
    assert captured.get("called")


def test_auto_falls_back_to_offline_when_no_key(monkeypatch, without_key, evidence):
    """AUTO falls back to offline when no API key is configured."""
    AutoProvider._online_ready = None  # clear cache

    def fake_ollama_available():
        return True, "Ollama ready"

    def fake_ollama_generate(prompt, image_paths=None):
        return "offline answer"

    monkeypatch.setattr("llm.ollama_available", fake_ollama_available)
    monkeypatch.setattr("llm.ollama_generate", fake_ollama_generate)

    result = AutoProvider().generate("q", evidence, {"s": "f.pdf"})
    assert result == "offline answer"


def test_auto_falls_back_on_connectivity_error(monkeypatch, with_key, evidence):
    """AUTO retries via offline when online fails with a transient error."""
    AutoProvider._online_ready = None  # clear cache

    call_count = {"online": 0, "offline": 0}

    def fake_urlopen(request, timeout=None):
        call_count["online"] += 1
        raise urllib.error.URLError("no route to host")

    def fake_ollama_available():
        return True, "Ollama ready"

    def fake_ollama_generate(prompt, image_paths=None):
        call_count["offline"] += 1
        return "offline fallback answer"

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("llm.ollama_available", fake_ollama_available)
    monkeypatch.setattr("llm.ollama_generate", fake_ollama_generate)

    result = AutoProvider().generate("q", evidence, {"s": "f.pdf"})
    assert result == "offline fallback answer"
    assert call_count["online"] == 1
    assert call_count["offline"] == 1


def test_auto_falls_back_to_offline_on_auth_error(monkeypatch, with_key, evidence):
    """AUTO falls back to offline when the online API key is rejected."""
    AutoProvider._online_ready = None  # clear cache

    # Probe returns 401 → online_r.ready=False → generate routes to offline.
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(401)),
    )
    monkeypatch.setattr("llm.ollama_available", lambda: (True, "Ollama ready"))
    monkeypatch.setattr(
        "llm.ollama_generate",
        lambda prompt, image_paths=None: "offline fallback",
    )

    result = AutoProvider().generate("q", evidence, {"s": "f.pdf"})
    assert result == "offline fallback"


def test_auto_readiness_uses_online_when_key_present(monkeypatch, with_key):
    AutoProvider._online_ready = None
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _fake_response({"models": []}),
    )
    r = AutoProvider().readiness()
    assert r.ready is True
    assert "online" in r.technical.lower()


def test_auto_readiness_falls_back_to_offline(monkeypatch, without_key):
    AutoProvider._online_ready = None

    monkeypatch.setattr("llm.ollama_available", lambda: (True, "Ollama ready"))
    r = AutoProvider().readiness()
    assert r.ready is True
    assert "offline" in r.technical.lower()


def test_auto_readiness_reports_both_unavailable(monkeypatch, without_key):
    AutoProvider._online_ready = None

    monkeypatch.setattr(
        "llm.ollama_available",
        lambda: (False, "OLLAMA_NOT_AVAILABLE: connection refused"),
    )
    r = AutoProvider().readiness()
    assert r.ready is False
    assert "no ai provider" in r.message.lower()


def test_auto_health_includes_per_provider_detail(monkeypatch, with_key):
    """The health dict for AUTO mode must include online and offline sub-status."""
    AutoProvider._online_ready = None

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _fake_response({"choices": [{"message": {"content": "ok"}}]}),
    )
    monkeypatch.setattr("llm.ollama_available", lambda: (False, "not running"))

    from rag_pipeline import get_pipeline
    h = get_pipeline().health(backend=Backend.AUTO)

    assert h["backend"] == "auto"
    assert "online" in h["inference"]
    assert "offline" in h["inference"]
    assert h["inference"]["active_provider"] in ("online", "offline", None)
    assert isinstance(h["inference"]["fallback"], bool)
