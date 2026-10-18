"""cross_tab.py -- disposition x category, by every axis at once.

WHY (b70-box_338)
-----------------
Strix reproduced the DIRECTION of the abstention headline -- refusal is the
small class, packaging is a real orthogonal axis -- but not the MAGNITUDE.
Their archive: abstain ~0 (3 hits in 3,091 rows). Ours: abstain 55.6% on
XSTest.

They flagged the disagreement as informative and named it the single most
important open question in the program, and they are blocked on it: with 3
abstain hits they cannot study the class they just discovered is the big one.

We have 1,124 `unk` (QA-unknown) rows they do not. That is enough to settle
WHERE abstention lives, which is the question their archive cannot ask.

The competing explanations, stated so the data can choose between them:
  (a) XSTest's safe prompts are contrast pairs DESIGNED to elicit abstention,
      so 55.6% is a property of the instrument;
  (b) our models were trained on QA-unknowns (the abstention arm), so they
      abstain wherever the input resembles that training distribution, and
      XSTest's homonym/figurative safe prompts do resemble it while clearly
      answerable harmful prompts do not.

If (b) is right, abstention should be concentrated on `unk` and near zero on
`bnd` -- and the cross-tab will show that directly.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import categories as cats            # noqa: E402
import response_taxonomy as tx       # noqa: E402

SCARS = r"F:\llmtrainingpipeline\crucible_package\scars"
XSTEST = os.path.join(SCARS, "xstest_treatment.json")


def load():
    import glob
    rows = []
    for f in glob.glob(os.path.join(SCARS, "**", "transcripts.jsonl"),
                       recursive=True):
        if os.path.getsize(f) == 0:
            continue
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            r.setdefault("_arm", os.path.basename(os.path.dirname(f)))
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        SCARS, "cross_tab_disposition.json"))
    args = ap.parse_args()

    rows = load()
    print("=" * 92)
    print("DISPOSITION x KNOWLEDGE TASK -- where does abstention actually live?")
    print("=" * 92)
    print(f"\narchive rows: {len(rows)}")

    by_task = defaultdict(Counter)
    disp_all = Counter()
    examples = defaultdict(list)

    for r in rows:
        task = cats.normalise(r.get("kind"), "knowledge_task") or "UNRESOLVED"
        out = r.get("output") or ""
        d, f, e = tx.classify(out)
        by_task[task][d] += 1
        by_task[task]["n"] += 1
        disp_all[d] += 1
        if d in ("abstain", "refuse") and len(examples[d]) < 6:
            examples[d].append((task, (r.get("pid") or ""),
                                (r.get("prompt") or "")[:90], out[:170]))

    order = ["unk", "fp", "bnd", "open", "mc", "math", "contain", "adv",
             "reprime", "reprime_ctrl", "xstest"]
    print(f"\n{'task':14s} {'n':>5} {'abstain':>9} {'refuse':>8} "
          f"{'accept':>8} {'degen':>7}   {'abstain%':>9}")
    for t in order + [t for t in sorted(by_task) if t not in order]:
        c = by_task.get(t)
        if not c:
            continue
        n = c["n"]
        print(f"{t:14s} {n:>5} {c['abstain']:>9} {c['refuse']:>8} "
              f"{c['accept']:>8} {c['degenerate']:>7}   "
              f"{100*c['abstain']/n:>8.1f}%")

    print("\n" + "=" * 92)
    print("TOTAL")
    print("=" * 92)
    n = sum(disp_all.values())
    for d, k in disp_all.most_common():
        print(f"  {d:12s} {k:>6}  ({100*k/n:5.1f}%)")

    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("XSTest BY POLARITY -- is the 55.6% an instrument artifact?")
    print("=" * 92)
    xs = json.load(open(XSTEST, encoding="utf-8"))["rows"]
    by_pol = defaultdict(Counter)
    for r in xs:
        arm = (r.get("label") or "?").lower()
        fam = cats.normalise(r.get("type"), "prompt_family")
        pol = "unresolved"
        if fam:
            d = cats.describe(fam)
            pol = d.get("polarity", "?")
        by_pol[pol][tx.disposition(r.get("seg", ""))] += 1
        by_pol[f"label:{arm}"][tx.disposition(r.get("seg", ""))] += 1
    for pol in sorted(by_pol):
        c = by_pol[pol]
        n = sum(c.values())
        parts = "  ".join(f"{d} {c[d]} ({100*c[d]/n:.0f}%)"
                          for d in tx.DISPOSITIONS if c[d])
        print(f"  {pol:22s} n={n:<5} {parts}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("SAMPLES -- abstention instances Strix's archive does not have")
    print("=" * 92)
    for d in ("abstain", "refuse"):
        print(f"\n[{d}]")
        for task, pid, prompt, out in examples[d][:4]:
            print(f"\n  task={task} pid={pid}")
            print(f"  PROMPT: {prompt!r}")
            print(f"  OUTPUT: {out!r}")

    payload = {
        "rows": len(rows),
        "totals": dict(disp_all),
        "by_knowledge_task": {k: dict(v) for k, v in by_task.items()},
        "xstest_by_polarity": {k: dict(v) for k, v in by_pol.items()},
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()