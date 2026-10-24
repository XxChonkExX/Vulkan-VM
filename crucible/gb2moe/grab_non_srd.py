"""Select non-SRD / lore / setting PDFs and route them:
  digital -> direct fitz text extract (free, exact)
  scanned -> VLM-OCR queue

Skips SRD-covered core rulebooks (SRD-gettable) and already-ingested
corpora (pf1e-prd, srd51). Manifest mode prints the plan; --run executes.
"""
import argparse
import json
import os
import re

import fitz

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="store_true")
ap.add_argument("--manifest", default=r"C:\llmwork\refusal_arm\scored\grab_manifest.json")
ap.add_argument("--roots", nargs="+", default=[
    r"F:\llmtrainingpipeline\dnd training",
    r"F:\llmtrainingpipeline\3.5 D&D Books",
    r"F:\llmtrainingpipeline\Forgotten Realms Lore",
    r"F:\llmtrainingpipeline\Bundle-FRRegions1",
    r"F:\llmtrainingpipeline\Pathfinder RPG",
])
a = ap.parse_args()

# SRD-gettable core (skip) — these are covered by OGL SRD / already ingested
SKIP = re.compile(
    r"(player'?s handbook(?!\s*ii)|dungeon master'?s guide(?!\s*ii)|"
    r"monster manual(?!\s*(ii|iii|iv))|core rulebook|srd|"
    r"bestiary(?!\s*\d)|pathfinder rpg core|gamemastery guide)",
    re.I)


def classify(p):
    d = fitz.open(p)
    n = d.page_count
    if n == 0:
        d.close()
        return None, 0, 0
    sample = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    tot = sum(len(d[i].get_text().strip()) for i in sample)
    d.close()
    mean = tot / len(sample)
    return ("digital" if mean >= 500 else
            "scanned" if mean < 100 else "mixed"), n, round(mean)


plan = {"digital": [], "scanned": [], "mixed": [], "skipped": []}
for root in a.roots:
    if not os.path.isdir(root):
        continue
    for dp, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue
            p = os.path.join(dp, fn)
            if SKIP.search(fn):
                plan["skipped"].append(p)
                continue
            cls, n, mean = classify(p)
            if cls is None:
                continue
            plan[cls].append({"path": p, "pages": n, "chpg": mean})

json.dump(plan, open(a.manifest, "w"), indent=1)
for k in ("digital", "mixed", "scanned", "skipped"):
    pages = sum(r["pages"] for r in plan[k])
    print(f"{k}: {len(plan[k])} pdfs, {pages} pages")
if not a.run:
    print("\n--scanned (VLM-OCR candidates)--")
    for r in sorted(plan["scanned"], key=lambda x: -x["pages"])[:30]:
        print(f"  {r['pages']:>4}p  {os.path.basename(r['path'])[:70]}")
    print("\nmanifest ->", a.manifest)
