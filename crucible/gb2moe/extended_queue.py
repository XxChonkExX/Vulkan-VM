"""Extended VLM-OCR queue: high-value non-SRD scans + mixed files.

Reads grab_manifest.json, excludes Pathfinder decks/maps/artwork, runs
vlm_ocr.py per book (resumable). Text-first per page inside vlm_ocr means
mixed files only pay GPU on their image pages.
"""
import json
import os
import subprocess
import sys
import time

PY = sys.executable
OCR = r"C:\llmwork\refusal_arm\vlm_ocr.py"
OUT = r"F:\llmtrainingpipeline\ocr_extended"
MAN = r"C:\llmwork\refusal_arm\scored\grab_manifest.json"

EXCLUDE = ("deck", "map", "artwork", "harrow", "chronicles",
           "character sheet", "screen")

man = json.load(open(MAN, encoding="utf-8"))
books = []
for cls in ("scanned", "mixed"):
    for r in man[cls]:
        base = os.path.basename(r["path"]).lower()
        if any(k in base for k in EXCLUDE):
            continue
        books.append((cls, r["path"], r["pages"]))
books.sort(key=lambda x: -x[2])
print(f"[queue] {len(books)} books, "
      f"{sum(b[2] for b in books)} pages", flush=True)

os.makedirs(OUT, exist_ok=True)
for cls, path, pages in books:
    name = os.path.splitext(os.path.basename(path))[0]
    name = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)[:60]
    outdir = os.path.join(OUT, name.strip())
    if os.path.exists(os.path.join(outdir, "ocr_meta.json")):
        print(f"[skip] {name}", flush=True)
        continue
    mode = "auto" if cls == "mixed" else "scan"
    t0 = time.time()
    print(f"[book] {name} ({pages}p, {cls})", flush=True)
    subprocess.run([PY, OCR, path, "-o", outdir, "--mode", mode])
    print(f"[book] {name}: {time.time()-t0:.0f}s", flush=True)

print("EXTENDED_QUEUE_DONE", flush=True)
