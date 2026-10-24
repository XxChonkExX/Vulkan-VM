"""PDF triage: classify each PDF as digital (text layer), scanned, or mixed.

Samples pages, measures extracted-text density per page.
  digital : mean >= 500 chars/page  -> direct fitz extract, no OCR needed
  scanned : mean <  100 chars/page  -> VLM-OCR candidate
  mixed   : in between              -> per-page decision
Usage: python pdf_triage.py <root> [<root> ...]
"""
import json
import os
import sys

import fitz

for root in sys.argv[1:]:
    if not os.path.isdir(root):
        print(f"MISSING {root}")
        continue
    print(f"=== {root} ===")
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                d = fitz.open(p)
                n = d.page_count
                if n == 0:
                    d.close()
                    continue
                sample = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
                tot = 0
                for i in sample:
                    tot += len(d[i].get_text().strip())
                mean = tot / len(sample)
                d.close()
                cls = ("digital" if mean >= 500 else
                       "scanned" if mean < 100 else "mixed")
                out.append((cls, fn, n, round(mean)))
            except Exception as e:
                out.append(("ERROR", fn, 0, 0))
    for cls in ("digital", "mixed", "scanned", "ERROR"):
        rows = [r for r in out if r[0] == cls]
        if not rows:
            continue
        pages = sum(r[2] for r in rows)
        print(f"  {cls}: {len(rows)} pdfs, {pages} pages")
        if cls in ("scanned", "mixed", "ERROR"):
            for r in rows[:40]:
                print(f"    {r[1]}  ({r[2]}p, {r[3]} ch/pg)")
    json.dump(out, open(os.path.join(
        r"C:\llmwork\refusal_arm\scored",
        "triage_" + os.path.basename(root.rstrip("\\")) + ".json"),
        "w"), indent=1)
