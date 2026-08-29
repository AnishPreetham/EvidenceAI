"""Verify every runtime dependency actually imports. Prints a machine-readable report."""
import importlib, platform, sys

MODULES = [
    "numpy", "torch", "faiss", "sentence_transformers", "faster_whisper",
    "fitz", "docx", "pandas", "PIL", "cv2", "pytesseract",
    "fastapi", "uvicorn", "streamlit", "pydantic", "pydantic_settings",
    "dotenv", "requests", "pytest",
]

def main() -> int:
    print(f"python  {platform.python_version()}  ({sys.executable})")
    print("-" * 62)
    failed = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"OK    {name:22s} {getattr(mod, '__version__', 'n/a')}")
        except Exception as exc:
            failed.append(name)
            print(f"FAIL  {name:22s} {type(exc).__name__}: {exc}")
    print("-" * 62)
    print("FAILED:", ", ".join(failed) if failed else "NONE")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
