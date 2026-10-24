"""VLM-OCR: PDF pages -> Gemma-4 vision transcription -> markdown text.

For grainy scans and table-heavy pages where tesseract fails. Renders
pages at 300 DPI via PyMuPDF (no poppler), routes table-heavy pages to
the table prompt, transcribes via Ollama gemma4 vision.

Usage:
  python vlm_ocr.py book.pdf -o outdir -p 1-12
  python vlm_ocr.py book.pdf              # all pages
"""
import argparse
import base64
import json
import os
import sys
import time

import fitz
import requests

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "gemma4:e4b"
DPI = 300
MAX_EDGE = 1536

PROMPTS = {
    "general": (
        "You are a precise OCR engine. Extract ALL text from this page.\n"
        "Rules:\n"
        "1. Preserve the original layout, paragraph breaks, and formatting.\n"
        "2. For tables, output Markdown table format.\n"
        "3. Keep the original language - do NOT translate.\n"
        "4. If text is unclear, mark it as [unclear: best_guess].\n"
        "5. Output ONLY the extracted text, no commentary."),
    "table": (
        "You are a precise OCR engine specialized in tables and structured "
        "game data.\nExtract ALL content from this page.\n"
        "Rules:\n"
        "1. Output tables in Markdown format with proper alignment.\n"
        "2. Preserve ALL numbers, stat values, dice notation, and units "
        "exactly.\n"
        "3. For merged cells, repeat the content in each spanned position.\n"
        "4. Keep row/column order exactly as shown.\n"
        "5. Output ONLY the extracted content, no commentary."),
    "scan": (
        "You are a precise OCR engine for old scanned game books.\n"
        "The scan may be grainy, yellowed, or skewed.\n"
        "Rules:\n"
        "1. Transcribe ALL text, ignoring scan artifacts and background "
        "noise.\n"
        "2. Preserve headings, paragraphs, lists, and table structures "
        "(tables as Markdown).\n"
        "3. Stat blocks: keep every number and line exactly.\n"
        "4. If a word is unclear, mark [unclear: best_guess].\n"
        "5. Output ONLY the extracted text, no commentary."),
}


def render_page(page, dpi):
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


def has_tables(page):
    try:
        return len(page.find_tables().tables) > 0
    except Exception:
        return False


def transcribe(png_bytes, mode):
    prompt = PROMPTS[mode]
    b64 = base64.b64encode(png_bytes).decode()
    body = json.dumps({"model": MODEL, "stream": False, "messages": [
        {"role": "user", "content": prompt, "images": [b64]}],
        "options": {"temperature": 0.1, "num_predict": 2000,
                    "num_ctx": 8192}})
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                OLLAMA, data=body.encode(),
                headers={"Content-Type": "application/json"})
            j = json.load(urllib.request.urlopen(req, timeout=900))
            return j["message"]["content"]
        except Exception as e:
            last = e
            print(f"[retry {attempt+1}/3] {e}", flush=True)
            time.sleep(15 * (attempt + 1))
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("-p", "--pages", default="all")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--mode", choices=["auto", "general", "table", "scan"],
                    default="auto")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    doc = fitz.open(a.pdf)
    n = doc.page_count
    if a.pages == "all":
        pages = list(range(1, n + 1))
    else:
        pages = []
        for part in a.pages.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                pages += list(range(int(lo), int(hi) + 1))
            else:
                pages.append(int(part))
    outdir = a.outdir or (os.path.splitext(a.pdf)[0] + "_vlmocr")
    os.makedirs(outdir, exist_ok=True)
    md_path = os.path.join(outdir, "transcript.md")
    done_pages = set()
    if os.path.exists(md_path) and a.pages != "all":
        pass  # explicit page lists always re-run
    elif os.path.exists(md_path):
        import re as _re
        for m in _re.finditer(r"<!-- page (\d+)", open(md_path,
                              encoding="utf-8").read()):
            done_pages.add(int(m.group(1)))
    todo = [p for p in pages if p not in done_pages]
    print(f"[ocr] {a.pdf}: {len(todo)} pages to do "
          f"({len(done_pages)} already done), dpi {a.dpi}, "
          f"mode {a.mode}", flush=True)

    full = []
    t0 = time.time()
    for pn in todo:
        if pn < 1 or pn > n:
            continue
        page = doc[pn - 1]
        txt = page.get_text().strip()
        if len(txt) >= 400:
            with open(md_path, "a", encoding="utf-8") as f:
                f.write(f"<!-- page {pn} (text-layer) -->\n\n{txt}\n\n---\n\n")
            print(f"[p{pn}] text-layer {len(txt)} chars", flush=True)
            continue
        mode = a.mode
        if mode == "auto":
            mode = "table" if has_tables(page) else "general"
        png = render_page(page, a.dpi)
        tp = time.time()
        try:
            text = transcribe(png, mode)
        except Exception as e:
            text = f"[OCR-FAILED: {type(e).__name__}]"
            print(f"[p{pn}] FAILED {e}", flush=True)
        dt = time.time() - tp
        with open(md_path, "a", encoding="utf-8") as f:
            f.write(f"<!-- page {pn} ({mode}) -->\n\n{text}\n\n---\n\n")
        print(f"[p{pn}] {mode} {len(text)} chars {dt:.0f}s "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    json.dump({"pdf": a.pdf, "pages": pages, "mode": a.mode,
               "model": MODEL}, open(os.path.join(outdir, "ocr_meta.json"),
                                     "w"), indent=1)
    print(f"[done] {md_path}", flush=True)


import urllib.request

if __name__ == "__main__":
    main()
