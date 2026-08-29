"""
verify_doc.py -- close out the one unverified acceptance item: legacy .doc.

This is the only format whose pipeline could not be executed during the
build, because it depends on LibreOffice headless conversion and
LibreOffice was not installed.

What it does:
  1. Locate soffice.exe and report the version.
  2. Build a REAL .doc file by asking LibreOffice to convert the demo
     .docx into Word 97-2003 format. (Nothing else on the machine can
     produce a genuine legacy .doc, and a renamed .docx would not
     actually exercise the conversion path.)
  3. Run that .doc through documents.process_document -- the same code
     path the API and UI use.
  4. Index it and query it end to end.

Run:  .venv\\Scripts\\python scripts\\verify_doc.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

CHECKS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, ok, detail))
    print(f"[{'  OK  ' if ok else ' FAIL '}] {name}: {detail}", flush=True)


def find_soffice() -> str | None:
    candidates = [
        settings.soffice_cmd,
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return shutil.which("soffice")


def make_real_doc(soffice: str, source_docx: Path, out_dir: Path) -> Path:
    """Convert .docx -> genuine Word 97-2003 .doc using LibreOffice."""
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [soffice, "--headless", "--convert-to", "doc:MS Word 97",
         "--outdir", str(out_dir), str(source_docx)],
        capture_output=True, text=True, timeout=300,
    )
    produced = out_dir / (source_docx.stem + ".doc")
    if not produced.exists():
        raise RuntimeError(
            f"LibreOffice produced no .doc. stdout={proc.stdout[:200]} "
            f"stderr={proc.stderr[:200]}"
        )
    return produced


def main() -> int:
    print("Evidence AI -- LEGACY .DOC VERIFICATION")
    print("=" * 70)

    soffice = find_soffice()
    if not soffice:
        record("LibreOffice present", False,
               "soffice.exe not found. Install LibreOffice, then re-run.")
        print("\n.DOC SUPPORT: NOT VERIFIED")
        return 1

    version = subprocess.run(
        [soffice, "--version"], capture_output=True, text=True, timeout=120
    ).stdout.strip() or "version not reported"
    record("LibreOffice present", True, f"{soffice} ({version})")

    source_docx = settings.documents_dir / "project_notes.docx"
    if not source_docx.exists():
        record("source .docx available", False,
               f"{source_docx} missing -- run create_demo_data.py first")
        return 1
    record("source .docx available", True, source_docx.name)

    # 1. Produce a genuine legacy .doc
    try:
        doc_path = make_real_doc(soffice, source_docx, settings.documents_dir)
        header = doc_path.read_bytes()[:8]
        # A real Word 97-2003 file is an OLE2 compound document.
        is_ole2 = header.startswith(b"\xd0\xcf\x11\xe0")
        record("genuine .doc created", True,
               f"{doc_path.name}, {doc_path.stat().st_size:,} bytes, "
               f"OLE2={is_ole2}")
        if not is_ole2:
            record("format check", False, "not an OLE2 compound document")
            return 1
    except Exception as exc:
        record("genuine .doc created", False, str(exc))
        return 1

    # 2. Run it through the real ingestion path
    from documents import process_document

    result = process_document(str(doc_path))
    if not result.success:
        record("process_document(.doc)", False,
               f"{result.error_code} {result.error}")
        return 1
    record("process_document(.doc)", True,
           f"{len(result.items)} items, type={result.source.source_type.value}")

    sections = {i.location.section for i in result.items if i.location.section}
    record("sections preserved", bool(sections),
           ", ".join(sorted(s for s in sections if s)[:3]) or "none")
    record("conversion warning surfaced", bool(result.warnings),
           result.warnings[0] if result.warnings else "none")

    # 3. Index and query it end to end
    from rag_pipeline import get_pipeline
    from schemas import QueryRequest

    pipeline = get_pipeline()
    indexed = pipeline.ingest_file(str(doc_path))
    record("indexed into FAISS + SQLite", indexed.success,
           f"{len(indexed.items)} items"
           + (f" -- {indexed.warnings[0]}" if indexed.warnings else ""))

    response = pipeline.query(QueryRequest(
        query="What did the Q4 review meeting confirm about records processed?",
        top_k=10, rerank_top_k=5,
    ))
    record("query returns an answer", not response.abstained,
           response.answer[:90].replace("\n", " "))

    doc_cited = [c for c in response.citations if c.filename.endswith(".doc")]
    record("evidence retrievable from the .doc",
           bool(doc_cited) or any(
               r.item.source_id == indexed.source.source_id
               for r in response.retrieved_items),
           f"{len(doc_cited)} citation(s) from the .doc file")

    print("=" * 70)
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"{passed}/{len(CHECKS)} checks passed")
    ok = all(ok for _, ok, _ in CHECKS)
    print("\n.DOC SUPPORT: " + ("VERIFIED" if ok else "NOT VERIFIED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
