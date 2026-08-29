"""
app.py -- Streamlit interface for Evidence AI.

Calls rag_pipeline directly (the same object the API uses), so the UI and
the API cannot drift apart.

Run:  .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from config import settings
from schemas import InferenceMode, Modality, QueryModality, QueryRequest, clock

st.set_page_config(page_title="Evidence AI", page_icon="🔍", layout="wide")

MODALITY_ICON = {
    Modality.TEXT: "📄",
    Modality.IMAGE: "🖼️",
    Modality.AUDIO: "🎧",
}


@st.cache_resource(show_spinner="Loading local models and index...")
def load_pipeline():
    from rag_pipeline import get_pipeline
    return get_pipeline()


def render_health(pipeline) -> None:
    health = pipeline.health()
    index = health["index"]

    with st.sidebar:
        st.subheader("Runtime")
        ollama_ok = health["ollama"]["available"]
        st.write(("🟢" if ollama_ok else "🔴") + f" Ollama — {settings.ollama_model}")
        if not ollama_ok:
            st.caption(health["ollama"]["detail"])
        st.write(("🟢" if health["tesseract"]["available"] else "🔴") + " Tesseract OCR")
        st.write(("🟢" if health["libreoffice"]["available"] else "⚪")
                 + " LibreOffice (.doc)")

        st.subheader("Index")
        col_a, col_b = st.columns(2)
        col_a.metric("Sources", index["sources"])
        col_b.metric("Items", index["items"])
        col_a.metric("Vectors", index["vectors"])
        col_b.metric("Links", index["relationships"])
        if index["by_modality"]:
            st.caption(" · ".join(f"{k}: {v}" for k, v in index["by_modality"].items()))

        st.subheader("Models (all local)")
        for label, value in health["models"].items():
            st.caption(f"{label}: `{value}`")


def render_evidence(pipeline, response) -> None:
    st.subheader("Evidence")
    for result in response.retrieved_items:
        item = result.item
        source = pipeline.store.get_source(item.source_id)
        filename = source.filename if source else "unknown"
        icon = MODALITY_ICON.get(item.modality, "📄")
        where = item.location.human()
        label = f"{icon} #{result.rank} · {filename}" + (f" · {where}" if where else "")

        with st.expander(label):
            cols = st.columns(3)
            cols[0].metric("Retrieval", f"{result.retrieval_score:.3f}")
            cols[1].metric("Rerank", f"{result.rerank_score:.3f}")
            cols[2].metric("Modality", item.modality.value)

            if item.modality is Modality.IMAGE and item.location.image_path:
                image_path = Path(item.location.image_path)
                if image_path.exists():
                    st.image(str(image_path), width=420)
                description = item.metadata.get("visual_description")
                ocr_text = item.metadata.get("ocr_text")
                if description:
                    st.markdown("**Visual description (Qwen2.5-VL)**")
                    st.write(description)
                if ocr_text:
                    st.markdown("**OCR text (Tesseract)**")
                    st.code(ocr_text[:1200])
                if item.metadata.get("vision_ok") is False:
                    st.warning("Vision model unavailable — OCR-only representation.")
            elif item.modality is Modality.AUDIO:
                start = item.location.timestamp_start
                end = item.location.timestamp_end
                if start is not None and end is not None:
                    st.caption(f"Transcript segment {clock(start)}–{clock(end)}")
                if source and Path(source.file_path).exists():
                    st.audio(source.file_path)
                st.write(item.content)
            else:
                st.write(item.content)

            st.caption(f"item_id `{item.item_id}` · source_id `{item.source_id}`")


def render_relationships(response) -> None:
    if not response.relationships:
        return
    st.subheader("Relationships")
    for rel in response.relationships[:12]:
        confidence = f" · {rel.confidence:.2f}" if rel.confidence is not None else ""
        cross = " · cross-modal" if rel.metadata.get("cross_modal") else ""
        st.caption(
            f"`{rel.relationship_type.value}`{confidence}{cross} — "
            f"{rel.source_item_id[:8]} ↔ {rel.target_item_id[:8]}"
        )


def render_response(pipeline, response) -> None:
    if response.abstained:
        st.warning(f"**Abstained.** {response.answer}")
        if response.error_code:
            st.caption(f"Reason: `{response.error_code.value}`")
    else:
        st.success("Grounded answer")
        st.markdown(response.answer)

    if response.confidence is not None:
        st.caption(
            f"Confidence {response.confidence:.2f} — a normalised rerank score, "
            "not a calibrated probability."
        )

    if response.citations:
        from citations import render_citation

        st.subheader("Citations")
        for citation in response.citations:
            with st.container(border=True):
                st.markdown(f"**{render_citation(citation)}**")
                st.caption(f"{citation.modality.value} · source `{citation.source_id}`")
                if citation.excerpt:
                    st.caption(citation.excerpt)
                if citation.modality is Modality.IMAGE and citation.location.image_path:
                    path = Path(citation.location.image_path)
                    if path.exists():
                        st.image(str(path), width=320)

    render_relationships(response)
    render_evidence(pipeline, response)

    if response.latency_ms:
        with st.expander("Timing"):
            for stage, value in response.latency_ms.items():
                st.caption(f"{stage}: {value:.0f} ms")


def main() -> None:
    st.title("🔍 Evidence AI")
    st.caption(
        "Offline-runtime multimodal RAG — documents, images and audio. "
        "All inference is local."
    )

    pipeline = load_pipeline()
    render_health(pipeline)

    tab_query, tab_ingest, tab_sources = st.tabs(
        ["Ask", "Ingest", "Sources"]
    )

    # ---------------------------------------------------------------- ingest
    with tab_ingest:
        st.subheader("Add files to the index")
        uploads = st.file_uploader(
            "PDF, DOC, DOCX, TXT, CSV, PNG, JPG, WEBP, WAV, MP3, M4A",
            type=["pdf", "doc", "docx", "txt", "csv", "png", "jpg", "jpeg",
                  "webp", "wav", "mp3", "m4a"],
            accept_multiple_files=True,
        )
        if uploads and st.button("Ingest files", type="primary"):
            for upload in uploads:
                with st.spinner(f"Processing {upload.name}..."):
                    path = pipeline.store_upload(upload.name, upload.getvalue())
                    result = pipeline.ingest_file(str(path))
                if result.success:
                    st.success(
                        f"{result.source.filename} — {len(result.items)} items, "
                        f"{len(result.relationships)} relationships"
                    )
                else:
                    st.error(
                        f"{result.source.filename} — "
                        f"{result.error_code.value if result.error_code else 'FAILED'}: "
                        f"{result.error}"
                    )
                for warning in result.warnings:
                    st.warning(warning)
            st.cache_resource.clear()

    # ----------------------------------------------------------------- query
    with tab_query:
        mode = st.radio(
            "Query type",
            ["Text", "Image", "Audio", "Document"],
            horizontal=True,
        )
        modality = QueryModality[mode.upper()]

        query_file = None
        if modality is not QueryModality.TEXT:
            extensions = {
                QueryModality.IMAGE: ["png", "jpg", "jpeg", "webp"],
                QueryModality.AUDIO: ["wav", "mp3", "m4a"],
                QueryModality.DOCUMENT: ["pdf", "docx", "txt", "csv"],
            }[modality]
            query_file = st.file_uploader(
                f"{mode} to query with", type=extensions, key=f"q_{mode}"
            )

        default_question = (
            "What evidence do we have about Project Alpha's 2024 development?"
            if modality is QueryModality.TEXT
            else f"Find evidence related to this {mode.lower()}."
        )
        question = st.text_area("Question", value=default_question, height=80)

        col_a, col_b, col_c = st.columns(3)
        top_k = col_a.slider("Retrieve (top_k)", 1, 50, settings.top_k)
        rerank_top_k = col_b.slider("Rerank to", 1, 20, settings.rerank_top_k)
        use_vision = col_c.checkbox(
            "Let Qwen inspect the top image",
            value=False,
            help="Passes one original image to the vision model. Slower "
                 "(~1030 extra context tokens).",
        )

        if st.button("Search", type="primary"):
            temp_path = None
            if modality is not QueryModality.TEXT:
                if query_file is None:
                    st.error(f"Upload {mode.lower()} to run this query type.")
                    st.stop()
                suffix = Path(query_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(query_file.getvalue())
                    temp_path = handle.name

            request = QueryRequest(
                query=question or default_question,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                inference_mode=(InferenceMode.LOCAL_VISION if use_vision
                                else InferenceMode.LOCAL),
                query_modality=modality,
                query_file_path=temp_path,
            )
            with st.spinner("Retrieving, reranking, generating..."):
                response = pipeline.query(request)
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

            if response.query_representation and modality is not QueryModality.TEXT:
                with st.expander("Query representation used for retrieval"):
                    st.code(response.query_representation[:2000])

            render_response(pipeline, response)

    # --------------------------------------------------------------- sources
    with tab_sources:
        sources = pipeline.store.list_sources()
        if not sources:
            st.info("Nothing indexed yet. Use the Ingest tab.")
        for source in sources:
            with st.expander(f"{source.filename}  ·  {source.source_type.value}"):
                st.caption(
                    f"source_id `{source.source_id}` · "
                    f"{(source.file_size or 0) / 1024:.0f} KB · {source.file_path}"
                )
                items = pipeline.store.get_items_by_source(source.source_id)
                st.write(f"{len(items)} indexed items")
                for item in items[:20]:
                    where = item.location.human()
                    prefix = f"{where} — " if where else ""
                    st.caption(f"{prefix}{item.content[:180]}")


if __name__ == "__main__":
    main()
