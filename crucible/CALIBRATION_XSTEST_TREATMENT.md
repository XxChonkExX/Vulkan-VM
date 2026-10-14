# Calibration Note: XSTest on the Treatment Arm (2026-10-13)

**Status:** measurement, not verdict (no pre-commit bar). Instrument + reproducer in
this directory: `xstest_run.py`, `xstest_recount.py`, `xstest_sample.py`.

## Result

Treatment-arm model (`gemma12b-sft-30000-lambda07-treatment` lineage, Glitter Boy
line), 450 XSTest prompts, greedy short generation, deterministic three-way
classifier (refusal markers / epistemic markers / comply):

| gold label | canonical refusal | epistemic evasion ("I don't know" loops) | true compliance |
|---|---|---|---|
| safe (250, answerable) | ~4.0% | 44.8% | 50.4% |
| unsafe (200, incl. harm contrasts) | 10.5% | 67.5% | 22% |

## Reading

This model has NO safety tuning, and declines are not guardrails. Its decline
system was trained on QA-unknowns, not harm refusal: under harm prompts it mostly
performs ignorance (67.5%) rather than refusal (10.5%), while truly complying with
~22% of unsafe prompts (confirmed specimens: killing-method questions, scar rows
30/40). Mirror image: 44.8% epistemic decline on answerable-but-confusing safe
prompts (false-ignorance). The trigger fires outside its home domain wearing the
wrong response, in both directions: wrong pathing for the spot.

Design consequence (program-level): refusal must be INSTALLED as its own behavior
with its own trigger and response mode; it does not emerge from scaling declines.
The three-way classifier (refuse / epistemic / comply) is now standard tooling for
response-mode scoring (cf. YEOLDE vs CONTEMPO probe design).

## Method limits

Greedy short generation; marker-based classifier (two definitional-use artifacts
found and subtracted: "illegal"/"not allowed" inside complying definitions;
U+FFFD-normalized for apostrophe corruption); borderline segments logged for hand
audit. Raw scars: `xstest_treatment.json`, `xstest_treatment_recount.json` (local
crucible scars + HF `evals/`). HF model card carries the summary table
(`calibration_xstest.md`, card-only push 2026-10-13).
