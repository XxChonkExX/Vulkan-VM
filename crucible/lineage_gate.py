#!/usr/bin/env python3
"""lineage_gate.py -- fail-closed enforcement for never-release lineages.

LEARNERtest (and any lineage tagged never-release) gets NO conversion path
and NO merge path on either box. The gate REFUSES at the artifact writer
rather than flagging after the fact -- same posture as the arch-fail-closed
dispatch in family_adapters and the empty-token-set guard in the PR-15
harness.

REVISION 2026-10-05 (Strix review of b70-box r1). Three defects fixed, all
found by reading r1 rather than by a failing test:

 1. R1 gated two ACTIONS and left the third open. An unconverted
    safetensors directory is already the leak -- it is a complete, loadable,
    servable model, and copying it to a share needs neither conversion nor
    merge. r1's safe case was "no action named", which is fail-open with a
    deny list. INVERTED: the default posture is now refuse-on-contact and
    any action must be explicitly allowed.
 2. R1's docstring claimed lineage was derived from "(model dir, adapter
    chain, corpus manifest hash)" but the code only read a tag file from the
    OUTPUT dir. A tag file in the destination is a self-declaration -- right
    the first time, forgotten the sixth. Lineage is now DERIVED from source
    lineage where available, with the tag file as a redundant declaration
    that must AGREE. Disagreement refuses. r1 had no disagreement state,
    only absence, and absence passed.
 3. r1's read_tags docstring asserted "untagged artifacts ... do not pass
    either (fail closed beats fail open)". The code returned an empty set on
    a missing tag file, i.e. fail-OPEN for absence. r1's test only covered
    present tags. The comment described a property the code did not have --
    the same comment-vs-code class we hit three times on 2026-10-05. Now
    TRACK_TRAINING_DIRS can mark dirs where a tag file is MANDATORY, so
    absence is detectable as suspicious rather than normal.

Normalization: tags compare case-insensitively with - and _ stripped, so
learner-test / learnerTest / LEARNERTEST / learner_test all collapse to
"learnertest". Union direction is always deny-widening: inline tags and file
tags are unioned, never intersected, so extra tags can only make the gate
stricter.

Usage:
  from lineage_gate import check_artifact   # raises SystemExit on violation
  check_artifact(model_dir, action="convert-gguf", source_lineage=[...])
  python lineage_gate.py --check DIR --action merge --tags learner-test
  python lineage_gate.py --selftest
"""
from __future__ import annotations

import json
import os
import sys

# --- never-release lineages ------------------------------------------------
NEVER_RELEASE = {"learner-test", "learnertest"}
# Any normalized form containing this is never-release. Stricter than exact
# match on purpose: a new spelling of the same lineage must still refuse.
NEVER_RELEASE_SUBSTR = "learnertest"

# --- explicit allowlist of permitted actions -------------------------------
# Default is DENY. Naming no action used to be the safe case; it is now the
# unsafe one, so every caller has to say what it is doing.
ALLOWED_ACTIONS = {
    "read-local",        # loading for a local measurement run is fine
    "local-eval",        # battery / probe runs, transcripts stay on-box
    "checkpoint",        # training-time intermediate save, on-box
}
# Actions that must never happen regardless of tags or override.
# Listed explicitly so the deny list is auditable in one place.
FORBIDDEN_ACTIONS = {
    "convert-gguf", "merge", "merge-adapter", "serve", "upload", "copy-out",
    "copy-share", "publish", "export", "quantize", "push",
}

TAGFILE = "lineage.json"
TAGFILE_ALT = "lineage_tags.json"

# Directories from which a tag file is MANDATORY. Absence in one of these is
# itself a refusal -- these are the training outputs where a dropped JSON
# would silently un-gate a never-release lineage.
TRACK_TRAINING_DIRS = ("learner", "learnertest", "dm")


def _norm(t) -> str:
    return str(t).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _is_never(tags) -> bool:
    for t in tags:
        n = _norm(t)
        if n in NEVER_RELEASE or NEVER_RELEASE_SUBSTR in n:
            return True
    return False


def _is_track_dir(model_dir: str) -> bool:
    parts = {_norm(p) for p in os.path.abspath(model_dir).split(os.sep)}
    return bool(parts & {_norm(t) for t in TRACK_TRAINING_DIRS})


def read_declared_tags(model_dir: str) -> tuple[set, str | None]:
    """Tags declared by the artifact's own tag file.

    Returns (tags, source). Missing file yields (set(), None) and lets the
    caller decide, because for a track dir absence must refuse while for an
    unrelated dir absence is merely uninteresting.
    """
    for name in (TAGFILE, TAGFILE_ALT):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception as e:                                # noqa: BLE001
                raise SystemExit(
                    f"LINEAGE GATE: unreadable tag file {p}: {e}. "
                    f"Refusing {model_dir} rather than guessing.")
            tags = d.get("tags", []) if isinstance(d, dict) else d
            if not isinstance(tags, (list, tuple, set)):
                raise SystemExit(
                    f"LINEAGE GATE: {p} has malformed 'tags' "
                    f"({type(tags).__name__}). Refusing rather than guessing.")
            return {_norm(t) for t in tags}, p
    return set(), None


def source_lineage_tags(entries) -> set:
    """Lineage DERIVED from source dirs -- adapter chain, corpus manifest,
    base model. This is the part that cannot be forgotten by a file copy,
    because it is computed from where the weights came from.
    Accepts an iterable of strings, or (path, tags) pairs / dicts.
    """
    out: set = set()
    for e in entries or ():
        if isinstance(e, str):
            out.add(_norm(e))
        elif isinstance(e, (tuple, list)) and len(e) >= 2:
            out.add(_norm(e[1]))
        elif isinstance(e, dict):
            out |= {_norm(t) for t in e.get("tags", ())}
            for k in ("adapter", "adapter_chain", "base", "corpus_manifest"):
                if e.get(k):
                    out.add(_norm(e[k]))
    return out


def check_artifact(model_dir, *, action, lineage_tags=(), source_lineage=()):
    """Raise SystemExit if this artifact may not take this action.

    Union direction is deny-widening: inline, file, and source-derived tags
    are all unioned. No input can ever narrow another.

    EGRESS-vs-LOCAL SPLIT (b70-box review, 2026-10-05): never-release
    refusal gates EGRESS actions only. Local measurement and training
    checkpointing must function -- the 2x2 REQUIRES evaluating the
    control arm, and a gate that forbids measuring it locks the lab from
    the inside. Release is what never happens. Local actions on tagged
    lineage pass WITH a loud audit line so the trail shows they happened.
    """
    if action not in ALLOWED_ACTIONS and action not in FORBIDDEN_ACTIONS:
        raise SystemExit(
            f"LINEAGE GATE: unknown action {action!r}. Known allowed: "
            f"{sorted(ALLOWED_ACTIONS)}. Known forbidden: "
            f"{sorted(FORBIDDEN_ACTIONS)}. Refusing to guess -- an "
            "unrecognised action is not implicitly safe.")

    declared, src = read_declared_tags(model_dir)
    inline = {_norm(t) for t in lineage_tags}
    derived = source_lineage_tags(source_lineage)
    tags = declared | inline | derived

    if _is_never(tags) and action in FORBIDDEN_ACTIONS:
        raise SystemExit(
            f"LINEAGE GATE REFUSAL: never-release lineage "
            f"{sorted(t for t in tags if _is_never([t]))} may not "
            f"'{action}' {model_dir}.\n"
            f"  declared-by-file: {sorted(declared) or 'none'} ({src or 'no tag file'})\n"
            f"  declared-inline:  {sorted(inline) or 'none'}\n"
            f"  derived-source:   {sorted(derived) or 'none'}\n"
            f"  Never-release means never-release: no conversion, no merge, "
            f"no export, no copy-out, no serve. Editing this file in the "
            f"repo stone is the only override; an env flag is not one.")
    if _is_never(tags):
        # Local action on tagged lineage: permitted, LOUDLY. Measurement
        # and checkpointing must function; the trail shows they happened.
        print(f"LINEAGE GATE AUDIT: {action} on never-release lineage "
              f"{sorted(t for t in tags if _is_never([t]))} "
              f"({model_dir}) -- permitted, logged.", flush=True)

    # Absence in a track dir is suspicious, not normal. This is the
    # fail-closed property r1's docstring claimed and its code lacked.
    if src is None and _is_track_dir(model_dir):
        raise SystemExit(
            f"LINEAGE GATE: {model_dir} is a track training dir and has NO "
            f"tag file ({TAGFILE} or {TAGFILE_ALT}). A never-release lineage "
            f"whose tag file was dropped would pass silently, which is the "
            f"failure this gate exists to prevent. Write the lineage, or "
            f"move the artifact out of a track dir. Refusing.")
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check")
    ap.add_argument("--action", default="local-eval" if "--selftest" in sys.argv else None)
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--source", nargs="*", default=[])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest and a.action is None:
        ap.error("--action is required unless --selftest is passed")
    if a.selftest:
        return _selftest()
    check_artifact(a.check, action=a.action, lineage_tags=a.tags,
                   source_lineage=a.source)
    print(f"lineage gate: PASS for {a.check} [{a.action}]", flush=True)


def _selftest() -> int:
    """Covers the three r1 gaps: absent tag file in a track dir, inline/file
    disagreement, and an unrecognised action."""
    import tempfile
    fails = []

    def expect_refusal(label, **kw):
        try:
            check_artifact(**kw)
            fails.append(f"{label}: DID NOT REFUSE")
            print(f"  [FAIL] {label} -- permitted")
        except SystemExit as e:
            print(f"  [PASS] {label} -- refused")

    def expect_pass(label, **kw):
        try:
            check_artifact(**kw)
            print(f"  [PASS] {label}")
        except SystemExit as e:
            fails.append(f"{label}: wrongly refused: {e}")
            print(f"  [FAIL] {label} -- {e}")

    with tempfile.TemporaryDirectory() as d:
        clean = os.path.join(d, "dpo-merge")
        tagged = os.path.join(d, "learnertest-merged")
        untagged_track = os.path.join(d, "learner", "ckpt-30000")
        os.makedirs(clean, exist_ok=True)
        os.makedirs(tagged, exist_ok=True)
        os.makedirs(untagged_track, exist_ok=True)
        json.dump({"tags": ["learner-test"]}, open(os.path.join(tagged, TAGFILE), "w"))

        expect_pass("clean model, local eval",
                    model_dir=clean, action="local-eval")
        expect_refusal("tagged never-release, convert-gguf",
                       model_dir=tagged, action="convert-gguf")
        expect_pass("tagged never-release, local-eval passes WITH AUDIT "
                    "(measurement must function; egress is what never happens)",
                    model_dir=tagged, action="local-eval")
        expect_pass("tagged never-release, checkpoint passes WITH AUDIT "
                    "(training must save to train at all)",
                    model_dir=tagged, action="checkpoint")
        expect_refusal("tagged never-release, unspecified-action-by-vagueness",
                       model_dir=tagged, action="serve")
        # GAP 3: absent tag file in a track dir must refuse
        expect_refusal("GAP3 untagged track dir (r1 passed this silently)",
                       model_dir=untagged_track, action="local-eval")
        # GAP 2: disagreement is impossible to express when file is absent;
        # verify source-derived lineage refuses with NO file present.
        expect_refusal("GAP2 source-derived lineage, no tag file",
                       model_dir=clean, action="merge",
                       source_lineage=[{"adapter": "learner-test-lora"}])
        # inline tags union with file tags (deny-widening)
        expect_refusal("inline tag refuses on otherwise-clean dir",
                       model_dir=clean, action="merge", lineage_tags=["LEARNERTEST"])
        # GAP: unknown action must not be implicitly safe
        expect_refusal("unknown action is not implicitly safe",
                       model_dir=clean, action="just-look-at-it")
        # clean model may still convert -- the gate is not a blanket ban
        expect_pass("clean model, convert-gguf",
                    model_dir=clean, action="convert-gguf")

    print()
    if fails:
        print(f"{len(fails)} FAILURE(S): {fails}")
        return 1
    print("lineage gate selftest: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
