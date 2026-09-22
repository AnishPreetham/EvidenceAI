"""
providers.py -- the inference provider layer.

Evidence AI supports two mandatory inference backends. Everything upstream
of generation (ingestion, embeddings, FAISS, retrieval, reranking,
relationships, abstention, citations) is shared and provider-independent;
only the final "evidence -> prose" step differs.

    RAG pipeline
         |
    InferenceProvider
       /            \
    OllamaProvider   OpenAIProvider
    (offline, local) (online, cloud)

Design notes:

* The `Backend` axis lives here, NOT in schemas.py. `InferenceMode` in the
  locked contract already means something else (whether the original image
  is handed to the vision model), so overloading it would conflate two
  independent choices. schemas.py is unchanged.

* Both providers reuse `llm.build_prompt`, so the grounding rules,
  discovery-vs-factual prompt selection, evidence ordering and context
  budget are identical online and offline. There is exactly one set of
  prompt rules in this codebase.

* Online mode never touches Ollama. `readiness()` for ONLINE checks
  credentials only, so the application is fully usable on a machine where
  Ollama is not installed.

* No API key is ever logged, echoed into an exception message, or returned
  in a readiness detail. See `_redact`.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from config import settings
from schemas import ErrorCode, Modality, RerankedResult

logger = logging.getLogger(__name__)


class Backend(str, Enum):
    """Which inference provider serves generation."""
    OFFLINE = "offline"
    ONLINE = "online"
    AUTO = "auto"


@dataclass(frozen=True)
class Readiness:
    """Whether a backend can serve a request right now.

    `message` is written for a non-technical user. `technical` carries the
    detail that belongs behind a "Technical details" expander -- never in
    the headline.
    """
    ready: bool
    message: str
    technical: str = ""
    code: ErrorCode | None = None
    # UI hint: offline setup problems are actionable in different ways.
    can_autostart: bool = False


class ProviderError(RuntimeError):
    """Generation failed. Carries a user-safe message plus technical detail."""

    def __init__(
        self,
        user_message: str,
        technical: str,
        code: ErrorCode,
        is_transient: bool = False,
    ):
        super().__init__(user_message)
        self.user_message = user_message
        self.technical = technical
        self.code = code
        # True for connectivity/service failures where a fallback makes sense.
        # False for configuration problems (bad key, wrong model, quota) where
        # switching providers does not help.
        self.is_transient = is_transient


def _redact(text: str) -> str:
    """Strip anything resembling a credential out of text bound for a user,
    a log, or an exception. Defence in depth: the key should never reach
    these paths in the first place."""
    key = settings.openai_api_key
    if key and len(key) > 8:
        text = text.replace(key, "***REDACTED***")
    return text


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class InferenceProvider(ABC):
    """Turns retrieved evidence into a grounded answer."""

    backend: Backend
    label: str

    @abstractmethod
    def readiness(self) -> Readiness:
        """Can this provider serve a request? Must not raise."""

    @abstractmethod
    def generate(
        self,
        query: str,
        evidence: list[RerankedResult],
        filenames: dict[str, str],
        inspect_image: Path | None = None,
    ) -> str:
        """Produce the answer text. Raises ProviderError on failure."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier shown in the UI."""


# ---------------------------------------------------------------------------
# Offline -- local Ollama
# ---------------------------------------------------------------------------

class OllamaProvider(InferenceProvider):
    """Wraps the existing local Ollama integration in llm.py.

    This adds no new inference logic: it delegates to the functions that
    already power offline mode, so offline behaviour is unchanged.
    """

    backend = Backend.OFFLINE
    label = "Offline AI"

    @property
    def model_name(self) -> str:
        return settings.ollama_model

    def readiness(self) -> Readiness:
        from llm import ollama_available

        ok, detail = ollama_available()
        if ok:
            return Readiness(True, f"Offline AI ready — {settings.ollama_model}", detail)

        if detail.startswith(ErrorCode.QWEN_MODEL_NOT_AVAILABLE.value):
            return Readiness(
                False,
                f"The local model '{settings.ollama_model}' isn't installed yet.",
                detail,
                ErrorCode.QWEN_MODEL_NOT_AVAILABLE,
            )
        return Readiness(
            False,
            "Offline AI isn't ready yet.",
            detail,
            ErrorCode.OLLAMA_NOT_AVAILABLE,
            can_autostart=True,
        )

    def generate(
        self,
        query: str,
        evidence: list[RerankedResult],
        filenames: dict[str, str],
        inspect_image: Path | None = None,
    ) -> str:
        from llm import OllamaError, generate_answer

        try:
            return generate_answer(query, evidence, filenames,
                                   inspect_image=inspect_image)
        except OllamaError as exc:
            raise ProviderError(
                "Offline AI could not generate an answer.",
                _redact(str(exc)),
                exc.code,
            ) from exc


def start_ollama() -> tuple[bool, str]:
    """Best-effort attempt to start a locally installed Ollama service.

    Returns (started, detail). Never raises -- a failure here is a UI state,
    not a crash.
    """
    import shutil
    import subprocess
    import time

    from llm import ollama_available

    if ollama_available()[0]:
        return True, "Ollama was already running."

    binary = shutil.which("ollama")
    if not binary:
        return False, "Ollama is not installed (no 'ollama' on PATH)."

    try:
        subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception as exc:
        return False, f"Could not launch Ollama: {exc}"

    for _ in range(10):
        time.sleep(1.5)
        if ollama_available()[0]:
            return True, "Ollama started."
    return False, "Ollama was launched but did not become ready in time."


def ollama_installed() -> bool:
    """Whether the Ollama binary exists, independent of whether it runs."""
    import shutil
    return bool(shutil.which("ollama"))


# ---------------------------------------------------------------------------
# Online -- OpenAI-compatible cloud API
# ---------------------------------------------------------------------------

class OpenAIProvider(InferenceProvider):
    """Cloud generation over the OpenAI-compatible chat-completions API.

    Uses the standard library rather than the `openai` SDK: the payload is a
    single JSON POST, and avoiding the SDK keeps the offline dependency set
    untouched and adds no new package to install.

    `online_base_url` makes this usable with any OpenAI-compatible endpoint
    (Azure OpenAI, OpenRouter, a self-hosted gateway) without code changes.
    """

    backend = Backend.ONLINE
    label = "Online AI"

    @property
    def model_name(self) -> str:
        return settings.online_model

    def readiness(self) -> Readiness:
        """Probes GET /models to confirm auth actually works.
        "Ready" means the endpoint accepted the key — not merely that a key
        is configured."""
        if not settings.openai_api_key.strip():
            return Readiness(
                False,
                "Online AI isn't configured yet.",
                f"No API key found. Set {settings.online_api_key_env} in your "
                f".env file (or your host's secret settings) and reload.",
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            )
        try:
            req = urllib.request.Request(
                f"{settings.online_base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                resp.read()
            return Readiness(
                True,
                f"Online AI ready — {settings.online_model}",
                f"Endpoint {settings.online_base_url}; key loaded from "
                f"{settings.online_api_key_env}.",
            )
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return Readiness(
                    False,
                    "Online AI could not authenticate — API key rejected.",
                    f"HTTP {exc.code} on GET /models. Check "
                    f"{settings.online_api_key_env} in your .env file.",
                    ErrorCode.OLLAMA_NOT_AVAILABLE,
                )
            return Readiness(
                False,
                "Online AI is temporarily unavailable.",
                f"HTTP {exc.code} on GET /models.",
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            )
        except urllib.error.URLError as exc:
            return Readiness(
                False,
                "Online AI is unreachable — check your internet connection.",
                _redact(f"URLError on GET /models: {exc.reason}"),
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            )
        except Exception as exc:
            return Readiness(
                False,
                "Online AI is unavailable.",
                _redact(f"{type(exc).__name__}: {exc}"),
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            )

    def _build_messages(
        self,
        prompt: str,
        inspect_image: Path | None,
    ) -> list[dict]:
        """Chat payload. When an image is supplied it is inlined as a base64
        data URI, which is how the chat-completions API accepts local images
        without uploading them anywhere first."""
        if inspect_image is None:
            return [{"role": "user", "content": prompt}]

        path = Path(inspect_image)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                },
            ],
        }]

    def generate(
        self,
        query: str,
        evidence: list[RerankedResult],
        filenames: dict[str, str],
        inspect_image: Path | None = None,
    ) -> str:
        # Same prompt builder as offline: identical grounding rules,
        # identical evidence formatting, identical abstention instruction.
        from llm import build_prompt

        if not settings.openai_api_key.strip():
            raise ProviderError(
                "Online AI isn't configured yet.",
                f"Set {settings.online_api_key_env} in your .env file and reload.",
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            )

        prompt = build_prompt(query, evidence, filenames)
        payload = {
            "model": settings.online_model,
            "messages": self._build_messages(prompt, inspect_image),
            "temperature": settings.ollama_temperature,
        }

        request = urllib.request.Request(
            f"{settings.online_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.openai_api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=settings.online_timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = _redact(exc.read().decode("utf-8", errors="replace")[:400])
            if exc.code in (401, 403):
                raise ProviderError(
                    "Online AI could not authenticate.",
                    f"HTTP {exc.code}. The API key was rejected. Check "
                    f"{settings.online_api_key_env} in your .env file. {body}",
                    ErrorCode.OLLAMA_NOT_AVAILABLE,
                ) from exc
            if exc.code == 429:
                raise ProviderError(
                    "Online AI is rate-limited or out of quota.",
                    f"HTTP 429. {body}",
                    ErrorCode.OLLAMA_NOT_AVAILABLE,
                ) from exc
            if exc.code == 404:
                raise ProviderError(
                    f"The online model '{settings.online_model}' is not available.",
                    f"HTTP 404. {body}",
                    ErrorCode.QWEN_MODEL_NOT_AVAILABLE,
                ) from exc
            raise ProviderError(
                "Online AI returned an error.",
                f"HTTP {exc.code}. {body}",
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                "Online AI is currently unavailable. Please check your "
                "internet connection or the API service.",
                _redact(f"{type(exc).__name__}: {exc.reason}"),
                ErrorCode.OLLAMA_NOT_AVAILABLE,
                is_transient=True,
            ) from exc
        except Exception as exc:
            raise ProviderError(
                "Online AI is currently unavailable.",
                _redact(f"{type(exc).__name__}: {exc}"),
                ErrorCode.OLLAMA_NOT_AVAILABLE,
                is_transient=True,
            ) from exc

        try:
            answer = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise ProviderError(
                "Online AI returned an unexpected response.",
                _redact(f"Malformed payload: {json.dumps(data)[:300]}"),
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            ) from exc

        if not answer:
            raise ProviderError(
                "Online AI returned an empty answer.",
                "The API responded successfully but the message was empty.",
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            )
        return answer


# ---------------------------------------------------------------------------
# AUTO -- selects Online when available, falls back to Offline
# ---------------------------------------------------------------------------

class AutoProvider(InferenceProvider):
    """Automatically routes to Online when it can serve a request; falls back
    to Offline otherwise. Neither the UI nor the user needs to choose.

    Online readiness is cached for _CACHE_TTL seconds so every request does
    not pay a probe round-trip. The cache is invalidated immediately whenever
    a real generation call fails with a transient error, so the next request
    re-checks rather than trusting a stale "ready" result.
    """

    backend = Backend.AUTO
    label = "Auto"

    _CACHE_TTL: float = 30.0
    # Class-level cache shared across all instances (singleton in _PROVIDERS).
    _online_ready: "Readiness | None" = None
    _online_ready_at: float = 0.0

    def __init__(self) -> None:
        self._online = OpenAIProvider()
        self._offline = OllamaProvider()

    @property
    def model_name(self) -> str:
        r = self._cached_online_readiness()
        return self._online.model_name if r.ready else self._offline.model_name

    # ---------------------------------------------------------------- cache

    def _cached_online_readiness(self) -> Readiness:
        import time
        now = time.monotonic()
        if (
            AutoProvider._online_ready is None
            or now - AutoProvider._online_ready_at >= AutoProvider._CACHE_TTL
        ):
            AutoProvider._online_ready = self._online.readiness()
            AutoProvider._online_ready_at = now
        return AutoProvider._online_ready  # type: ignore[return-value]

    def _invalidate_online_cache(self) -> None:
        AutoProvider._online_ready = None
        AutoProvider._online_ready_at = 0.0

    # ------------------------------------------------------------ readiness

    def readiness(self) -> Readiness:
        online_r = self._cached_online_readiness()
        offline_r = self._offline.readiness()

        if online_r.ready:
            return Readiness(
                ready=True,
                message=online_r.message,
                technical=f"AUTO → online. {online_r.technical}",
            )
        if offline_r.ready:
            return Readiness(
                ready=True,
                message=offline_r.message,
                technical=(
                    f"AUTO → offline (online unavailable: {online_r.technical}). "
                    f"{offline_r.technical}"
                ),
            )
        return Readiness(
            ready=False,
            message=(
                "No AI provider is currently available.\n\n"
                f"Online: {online_r.message}\n"
                f"Offline: {offline_r.message}"
            ),
            technical=(
                f"Online: {online_r.technical or online_r.message}. "
                f"Offline: {offline_r.technical or offline_r.message}"
            ),
            code=ErrorCode.OLLAMA_NOT_AVAILABLE,
        )

    # ------------------------------------------------------------- generate

    def generate(
        self,
        query: str,
        evidence: list[RerankedResult],
        filenames: dict[str, str],
        inspect_image: Path | None = None,
    ) -> str:
        online_r = self._cached_online_readiness()

        if online_r.ready:
            try:
                return self._online.generate(
                    query, evidence, filenames, inspect_image
                )
            except ProviderError as exc:
                # Any online failure in AUTO mode: invalidate cache, try offline.
                self._invalidate_online_cache()
                offline_r = self._offline.readiness()
                if offline_r.ready:
                    logger.info(
                        "AUTO: online failed (%s), falling back to offline",
                        exc.technical[:120],
                    )
                    return self._offline.generate(
                        query, evidence, filenames, inspect_image
                    )
                # Offline also not ready: propagate the online error.
                raise

        offline_r = self._offline.readiness()
        if offline_r.ready:
            return self._offline.generate(
                query, evidence, filenames, inspect_image
            )

        # Both unavailable.
        online_r2 = self._cached_online_readiness()
        raise ProviderError(
            (
                "No AI provider is currently available.\n\n"
                f"Online provider: {online_r2.message}\n\n"
                f"Offline AI: {offline_r.message}\n\n"
                "Configure the server-side Online provider (OPENAI_API_KEY in "
                ".env) or install and start Ollama."
            ),
            (
                f"Online: {online_r2.technical or online_r2.message}. "
                f"Offline: {offline_r.technical or offline_r.message}"
            ),
            ErrorCode.OLLAMA_NOT_AVAILABLE,
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

_PROVIDERS: dict[Backend, InferenceProvider] = {}


def resolve_backend(backend: Backend | str | None) -> Backend:
    """Normalise a backend selection, falling back to configuration."""
    if backend is None:
        backend = settings.inference_backend
    if isinstance(backend, Backend):
        return backend
    try:
        return Backend(str(backend).strip().lower())
    except ValueError:
        logger.warning("Unknown backend %r; using auto.", backend)
        return Backend.AUTO


def get_provider(backend: Backend | str | None = None) -> InferenceProvider:
    """Return the provider for a backend. Instances are cached and stateless."""
    resolved = resolve_backend(backend)
    if resolved not in _PROVIDERS:
        if resolved is Backend.ONLINE:
            _PROVIDERS[resolved] = OpenAIProvider()
        elif resolved is Backend.OFFLINE:
            _PROVIDERS[resolved] = OllamaProvider()
        else:
            _PROVIDERS[resolved] = AutoProvider()
    return _PROVIDERS[resolved]


def supports_vision(backend: Backend | str | None = None) -> bool:
    """Whether direct image inspection is available on this backend.

    Offline uses Qwen2.5-VL, which is a vision model. Online depends on the
    configured model, so this reflects the configured capability flag rather
    than guessing from the model name.
    """
    if resolve_backend(backend) is Backend.OFFLINE:
        return True
    return settings.online_model_supports_vision


def image_evidence_present(evidence: list[RerankedResult]) -> bool:
    """Whether any retrieved evidence is an image."""
    return any(r.item.modality is Modality.IMAGE for r in evidence)
