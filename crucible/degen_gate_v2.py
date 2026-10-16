"""Degeneracy gate v2 (b70-box, 2026-10-16): rep4 + channel-token +
language-mix checks. Answers strix-302's onboarding gap (MODEL_ONBOARDING
s5 item 10 covers rep4 ONLY). Lesson (283, demonstrated 4x): rep4 is
blind to multilingual token salad and channel-token bleed -- a gate
that only runs rep4 certifies garbage as coherent. v2 runs all three
detectors; verdict FAILS if ANY trips. Stdlib only. Portable paths.
Usage: py -3 degen_gate_v2.py DIR_OR_FILE [--seg-field seg]
  DIR: like v1 (arm subdirs with transcripts.jsonl, pid/output schema)
  FILE: .jsonl (one JSON per line, output/seg field) or .json with rows
"""
import json
import os
import re
import sys
import unicodedata

CHAN_PAT = re.compile(
    r"<\|channel\|>|<\|turn\|>|<\|think\|>|\bthought\b|\bthinking\b",
    re.IGNORECASE)


def rep4(t):
    w = re.findall(r"\w+", t.lower())
    if len(w) < 24:
        return None
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1.0 - len(set(g)) / len(g)


def channel_hit(t):
    """True if visible text carries channel/think tokens (leak/bleed)."""
    return bool(CHAN_PAT.search(t))


def nonlatin_share(t):
    """Fraction of letter chars outside Latin script (language-mix)."""
    letters = [c for c in t if unicodedata.category(c).startswith("L")]
    if len(letters) < 20:
        return 0.0
    nonlat = 0
    for c in letters:
        try:
            nm = unicodedata.name(c)
        except ValueError:
            nonlat += 1
            continue
        if not (nm.startswith("LATIN") or nm.startswith("COMMON")
                or nm.startswith("DIGIT")):
            # COMMON covers space/punct (already filtered); DIGIT kept
            # latin-only to avoid penalizing numbers
            if not nm.startswith("COMMON"):
                nonlat += 1
    return nonlat / len(letters)


def judge(outputs):
    """outputs: list of strings. Returns dict of detector verdicts."""
    r4 = [v for v in (rep4(t) for t in outputs) if v is not None]
    ch = sum(1 for t in outputs if channel_hit(t))
    lm = sum(1 for t in outputs if nonlatin_share(t) > 0.25)
    n = len(outputs)
    return {
        "n": n,
        "rep4_mean": round(sum(r4) / len(r4), 4) if r4 else None,
        "rep4_trip": bool(r4) and (sum(r4) / len(r4)) > 0.10,
        "channel_n": ch,
        "channel_trip": ch > max(1, 0.05 * n),
        "langmix_n": lm,
        "langmix_trip": lm > max(1, 0.10 * n),
    }


def collect(path, seg_field="output"):
    texts = []
    if os.path.isdir(path):
        for arm in sorted(os.listdir(path)):
            p = os.path.join(path, arm, "transcripts.jsonl")
            if os.path.isfile(p):
                texts.append((arm, p))
        out = []
        for arm, p in texts:
            outs = []
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    outs.append(r.get("output", "") or "")
            out.append((arm, outs))
        return out
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            items = data.get("rows", data) if isinstance(data, dict) \
                else data
            for r in items:
                if isinstance(r, dict):
                    rows.append(r.get(seg_field, "") or "")
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                rows.append(r.get(seg_field, "") or "")
    return [(os.path.basename(path), rows)]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: degen_gate_v2.py DIR_OR_FILE [--seg-field seg]")
    path = sys.argv[1]
    seg = "output"
    if "--seg-field" in sys.argv:
        seg = sys.argv[sys.argv.index("--seg-field") + 1]
    print(f"{'arm':28s} {'n':>5} {'rep4':>7} {'chan':>5} {'langmix':>7}  "
          f"verdict")
    print("-" * 72)
    for arm, outs in collect(path, seg):
        j = judge(outs)
        trips = [k for k in ("rep4_trip", "channel_trip", "langmix_trip")
                 if j[k]]
        verdict = ("NOT SCOREABLE (" + ",".join(
            t.replace("_trip", "") for t in trips) + ")"
            if trips else "passes gate")
        print(f"{arm:28s} {j['n']:>5} "
              f"{j['rep4_mean'] if j['rep4_mean'] is not None else 'n/a':>7} "
              f"{j['channel_n']:>5} {j['langmix_n']:>7}  {verdict}")


if __name__ == "__main__":
    main()
