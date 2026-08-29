"""Ingest the demo corpus and report exactly what was indexed."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import settings
from rag_pipeline import get_pipeline

pipeline = get_pipeline()
t0 = time.time()
results = pipeline.ingest_directory(str(settings.data_dir))

print(f"{'FILE':28s} {'OK':5s} {'ITEMS':>6s} {'RELS':>5s}  NOTE")
print("-" * 78)
for r in results:
    note = r.error or (r.warnings[0] if r.warnings else "")
    print(f"{r.source.filename:28s} {str(r.success):5s} {len(r.items):>6d} "
          f"{len(r.relationships):>5d}  {note[:28]}")

print("-" * 78)
print(f"ingested in {time.time() - t0:.1f}s")
print("INDEX:", pipeline.store.stats())
