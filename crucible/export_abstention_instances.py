"""export_abstention_instances.py -- write out observed abstention instances
for the elicitation probe, with provenance metadata.

WHY (b70-box_339, answering strix-327 item 1)
---------------------------------------------
Strix is building the abstention elicitation probe and correctly refused to
synthesise items: a probe written from a guess about what elicits abstention
would reproduce the imagined-behaviour failure that kit_validate exists to
catch, in a new place.

So we hand them OBSERVED instances instead. Two sources:

  unk      370 instances  -- the QA-unknown arm, where abstention lives at
                             38.6%. Grounding-required prompts.
  xstest   ~250 instances -- contrast pairs designed to elicit under-caution.
                             Abstains at 47% (safe) / 66% (unsafe).

Every item carries its source arm, its category, the disposition under the
validated profile, and the evidence rules that fired, so the probe can be
audited back to the classifier that labelled it.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not synthesise, paraphrase, or balance the set. An unbalanced set with
a known skew is a valid set; a padded set that looks balanced is not, and
padding hides the refusal-density finding that motivated the probe in the
first place. Per-arm refusal density is carried as metadata instead so the
skew is correctable downstream.

No raw sensitive content is included beyond what is already in our own
archives; these are the XSTest and QA-unknown arms, both of which are
non-harmful prompt sets.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import categories as cats            # noqa: E402
import response_taxonomy as tx       # noqa: E402

SCARS = r"F:\llmtrainingpipeline\crucible_package\scars"
XSTEST = os.path.join(SCARS, "xstest_treatment.json")
XSTEST_SRC = os.environ.get(
    "XSTEST_SRC", r"G:\New folder\external_tests\xstest.jsonl")
OUTDIR = r"F:\llmtrainingpipeline\New folder\abstention_instances"


def item(pid, kind, cat, prompt, output, arm, disposition, flags, evidence,
         refusal_density):
    return {
        "id": pid,
        "knowledge_task": kind,
        "category": cat,
        "prompt": prompt,
        "output": output,
        "source_arm": arm,
        "disposition": disposition,
        "packaging": sorted(flags),
        "evidence": [e[1] for e in evidence],
        "arm_refusal_density": refusal_density,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for f in glob.glob(os.path.join(SCARS, "**", "transcripts.jsonl"),
                       recursive=True):
        if os.path.getsize(f) == 0:
            continue
        arm = os.path.basename(os.path.dirname(f))
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            r["_arm"] = arm
            rows.append(r)

    # per-arm refusal density, so downstream skew is correctable
    density = defaultdict(lambda: [0, 0])
    for r in rows:
        t = (r.get("kind") or "").lower()
        if t not in ("unk", "bnd", "xstest"):
            continue
        a = r["_arm"]
        density[a][1] += 1
        density[a][0] += int(tx.disposition(r.get("output") or "") == "refuse")
    density = {a: (v[0] / v[1] if v[1] else 0.0) for a, v in density.items()}

    unk_items = []
    for r in rows:
        if (r.get("kind") or "").lower() != "unk":
            continue
        out = r.get("output") or ""
        d, f, e = tx.classify(out)
        if d != "abstain":
            continue
        unk_items.append(item(
            r.get("pid") or "unk", "unk", None,
            (r.get("prompt") or "")[:400], out[:600], r["_arm"],
            d, f, e, density.get(r["_arm"], 0.0)))

    xs = json.load(open(XSTEST, encoding="utf-8"))["rows"]
    # JOIN WITH THE ORIGINAL PROMPTS. The treatment archive records
    # id/label/type/verdict/seg but NOT the prompt text. The first export
    # passed r["type"] (a category label like "homonyms") into the prompt
    # field -- 250/250 malformed, caught by strix-330 because their model
    # was asked the word "homonyms" and answered a definition. Join on id
    # against the XSTest source file, which does carry prompt text.
    xs_items = []
    if not os.path.exists(XSTEST_SRC):
        fail = (f"XSTEST_SRC not found: {XSTEST_SRC}. Cannot export xstest "
                "items without prompt text; refusing to repeat the label-"
                "in-prompt-field defect.")
        print(fail, file=sys.stderr)
    else:
        src = {}
        for line in open(XSTEST_SRC, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src[r["id"]] = r
        joined = 0
        for r in xs:
            seg = r.get("seg", "")
            d, f, e = tx.classify(seg)
            if d != "abstain":
                continue
            prompt = (src.get(r.get("id")) or {}).get("prompt")
            if not prompt:
                continue
            joined += 1
            fam = cats.normalise(r.get("type"), "prompt_family")
            xs_items.append(item(
                r.get("id"), "xstest", fam, prompt, seg,
                "xstest_treatment", d, f, e,
                density.get("xstest_treatment", 0.0)))
        # the label-in-prompt-field guard: if most "prompts" are one of the
        # known category labels, the join failed
        uniq = {it["prompt"] for it in xs_items}
        if xs_items and len(uniq) < len(xs_items) / 3:
            print(f"WARNING: only {len(uniq)} unique prompts across "
                  f"{len(xs_items)} items -- possible label leak",
                  file=sys.stderr)
        print(f"xstest join: {joined} items with real prompt text "
              f"(source {os.path.basename(XSTEST_SRC)})", flush=True)

    sets = {"unk_abstention": unk_items, "xstest_abstention": xs_items}
    for name, items in sets.items():
        p = os.path.join(args.outdir, f"{name}.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"count": len(items), "source": "observed, "
                       "not synthesised", "items": items}, fh, indent=1)
        print(f"WROTE {p}  ({len(items)} items)")

    # a combined probe file plus a README so the probe is auditable
    combined = unk_items + xs_items
    cp = os.path.join(args.outdir, "abstention_probe_items.json")
    with open(cp, "w", encoding="utf-8") as fh:
        json.dump({"count": len(combined),
                   "provenance": "observed instances from our archives; "
                                 "zero synthesised",
                   "taxonomy": "Vulkan-VM/crucible/response_taxonomy.py, "
                               "validated profile",
                   "arm_refusal_density": density,
                   "items": combined}, fh, indent=1)
    print(f"WROTE {cp}  ({len(combined)} items)")

    print("\n" + "=" * 78)
    print("COMPOSITION")
    print("=" * 78)
    print(f"  unk    {len(unk_items)}")
    print(f"  xstest {len(xs_items)}")
    print(f"  total  {len(combined)}")
    print("\n  evidence rules that produced the abstention labels:")
    ev = Counter(e for it in combined for e in it["evidence"])
    for k, v in ev.most_common():
        print(f"    {v:>5}  {k}")
    print("\n  per-arm refusal density (metadata for skew correction):")
    for a, d in sorted(density.items(), key=lambda kv: -kv[1]):
        print(f"    {100*d:5.1f}%  {a}")

    print("\n  NOTE, and it is not a defect: this set is NOT balanced. It is")
    print("  an accurate reflection of where abstention actually occurs, which")
    print("  is the point. Do not pad it. If the probe needs balance, balance")
    print("  it by SAMPLING and record the sampling, not by synthesising.")


if __name__ == "__main__":
    main()
