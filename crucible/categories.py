"""categories.py -- one registry for every prompt category in the program.

THE PROBLEM THIS FIXES
----------------------
The program accumulated at least five incompatible category schemes and
nobody had reconciled them:

  * 8 Qwen harm categories (self-harm, hate-harassment, illicit-behavior,
    cybercrime, privacy-pii, weapons-phys-harm, fraud-scams, cbrne) -- used by
    the ablation ladder, now with per-category paraphrase peaks
  * ~30 XSTest types (homonyms, figurative_language, definitions,
    historical_events, contrast_*, not_for_all_audiences, ...) -- used for
    the published over-refusal calibration
  * 11+ archive "kind" values (mc, math, open, unk, bnd, fp, reprime,
    reprime_ctrl, contain, ...) -- what the transcripts are actually tagged
    with
  * abstention subtypes tracked ad hoc (idk loops, epistemic hedges,
    unanswerable claims, QA-unknowns)
  * the CUBE arms (dark / noble / ignorant / neutral) -- corpus valence
    groupings, not prompt categories at all

Consequences already paid for:
  * "we never mentioned privacy-pii as a distinct behavior" -- it was there,
    with the highest separability in the bank, invisible to a scheme that
    only counted scalar refusals
  * the scalar ladder could not see geometry; the geometric probe could not
    see counts. They were never joined because they had no shared key.
  * cross-box comparisons conflated prompt distribution with behaviour,
    because the category labels did not line up.

This module gives every one of them a stable ID and an explicit axis, so
rates can be cross-tabbed instead of compared side by side.

FIVE AXES, KEPT SEPARATE ON PURPOSE
-----------------------------------
  axis            what it answers
  ---------------- -------------------------------------------------
  behaviour       what the model DID (disposition axis; see
                  response_taxonomy)
  harm_domain     what the prompt is ABOUT (harm content)
  prompt_family   how the prompt is PHRASED (over-refusal structure)
  knowledge_task  what capability is being tested (our archive kinds)
  corpus_valence  which cube arm the text came from

Collapsing behaviour into harm_domain is the mistake that made privacy-pii
invisible: it is a harm domain, but the thing we wanted to measure about it
is a behaviour, and they are independent.
"""

__all__ = ["AXES", "REGISTRY", "normalise", "describe", "coverage",
           "HARM_DOMAINS", "PROMPT_FAMILIES", "KNOWLEDGE_TASKS",
           "CORPUS_VALENCE", "BEHAVIOURS"]

# ---------------------------------------------------------------------------
AXES = ("behaviour", "harm_domain", "prompt_family", "knowledge_task",
        "corpus_valence")

BEHAVIOURS = ("accept", "refuse", "abstain", "deflect", "degenerate")

# --- axis: harm_domain (Qwen ladder categories) ----------------------------
HARM_DOMAINS = {
    "self_harm":            dict(label="self-harm", axis="harm_domain"),
    "hate_harassment":      dict(label="hate-harassment", axis="harm_domain"),
    "illicit_behavior":     dict(label="illicit-behavior", axis="harm_domain"),
    "cybercrime":           dict(label="cybercrime", axis="harm_domain"),
    "privacy_pii":          dict(label="privacy-pii", axis="harm_domain"),
    "weapons_phys_harm":    dict(label="weapons-phys-harm",
                                  axis="harm_domain"),
    "fraud_scams":          dict(label="fraud-scams", axis="harm_domain"),
    "cbrne":                dict(label="cbrne", axis="harm_domain"),
}

# --- axis: prompt_family (XSTest types, the over-refusal structure) -------
# These 18 are the types that ACTUALLY APPEAR in our archive, read off
# xstest_treatment.json rather than written from memory. An earlier draft of
# this list was authored from recollection and 6 of its entries matched
# nothing; kit_validate.py check E caught it. That is the whole reason this
# module exists -- do not add a label here that has not been seen in a corpus.
#
# polarity: SAFE_LOOKING means a naive assistant should still answer;
# UNSAFE_LOOKING means a naive assistant should also answer (contrast tests).
# The `contrast_` families are XSTest's over-refusal controls.
_PROMPT_FAMILIES = [
    ("homonyms",                    "safe-looking"),
    ("figurative_language",         "safe-looking"),
    ("definitions",                 "safe-looking"),
    ("historical_events",           "safe-looking"),
    ("privacy_public",              "safe-looking"),
    ("privacy_fictional",           "safe-looking"),
    ("safe_targets",                "safe-looking"),
    ("safe_contexts",               "safe-looking"),
    ("nons_group_real_discr",       "safe-looking"),
    ("real_group_nons_discr",       "safe-looking"),
    ("contrast_homonyms",           "unsafe-looking"),
    ("contrast_figurative_language", "unsafe-looking"),
    ("contrast_definitions",        "unsafe-looking"),
    ("contrast_historical_events",  "unsafe-looking"),
    ("contrast_privacy_public",     "unsafe-looking"),
    ("contrast_privacy_fictional",  "unsafe-looking"),
    # NOTE: the real archive contains `contrast_privacy` (x25), NOT
    # contrast_privacy_public / contrast_privacy_fictional. The two granular
    # names were written from recollection in an earlier draft and matched
    # nothing; kit_validate.py check E flagged it. Only `contrast_privacy`
    # is registered because only `contrast_privacy` has been observed.
    ("contrast_privacy",            "unsafe-looking"),
    ("contrast_safe_targets",       "unsafe-looking"),
    ("contrast_safe_contexts",      "unsafe-looking"),
    ("contrast_discr",              "unsafe-looking"),
]
PROMPT_FAMILIES = {k: dict(label=k, axis="prompt_family", polarity=p)
                   for k, p in _PROMPT_FAMILIES}

# --- axis: knowledge_task (our archive "kind" values) ----------------------
_KNOWLEDGE_TASKS = [
    ("mc",            "multiple-choice knowledge"),
    ("math",          "arithmetic"),
    ("open",          "free-form factual"),
    ("unk",           "QA-unknown: unanswerable by construction"),
    ("fp",            "false premise"),
    ("bnd",           "boundary: harmful creative-writing prompt"),
    ("reprime",       "re-prime / prime-distractor"),
    ("reprime_ctrl",  "re-prime control"),
    ("contain",       "containment"),
    ("adv",           "adversarial"),
    ("nsfw_coherence", "nsfw coherence"),
    ("violence_handling", "violence handling"),
    ("med",           "medical"),
    ("law",           "legal"),
    ("psych",         "psychological"),
    ("chemistry",     "chemistry"),
    ("sciences",      "general science"),
    ("xstest",        "XSTest over-refusal battery"),
]
KNOWLEDGE_TASKS = {k: dict(label=v, axis="knowledge_task") for k, v in
                   _KNOWLEDGE_TASKS}

# --- axis: corpus_valence (the cube arms) ---------------------------------
CORPUS_VALENCE = {
    "neutral":  dict(label="neutral", axis="corpus_valence"),
    "dark":     dict(label="dark", axis="corpus_valence"),
    "noble":    dict(label="noble", axis="corpus_valence"),
    "ignorant": dict(label="ignorant", axis="corpus_valence"),
}

REGISTRY = {}
for d in (HARM_DOMAINS, PROMPT_FAMILIES, KNOWLEDGE_TASKS, CORPUS_VALENCE):
    REGISTRY.update(d)
for b in BEHAVIOURS:
    REGISTRY[b] = dict(label=b, axis="behaviour")

# ---------------------------------------------------------------------------
_ALIASES = {
    # Qwen ladder spellings
    "self-harm": "self_harm",
    "hate-harassment": "hate_harassment",
    "illicit-behavior": "illicit_behavior",
    "privacy-pii": "privacy_pii",
    "weapons-phys-harm": "weapons_phys_harm",
    "fraud-scams": "fraud_scams",
    "cybercrime": "cybercrime",
    "cbrne": "cbrne",
    # transcript "kind" spellings
    "mc": "mc", "math": "math", "open": "open", "unk": "unk", "fp": "fp",
    "bnd": "bnd", "reprime": "reprime", "reprime_ctrl": "reprime_ctrl",
    "contain": "contain", "adv": "adv",
    "nsfw_coherence": "nsfw_coherence",
    "violence_handling": "violence_handling", "med": "med", "law": "law",
    "psych": "psych", "chemistry": "chemistry", "sciences": "sciences",
    "xstest": "xstest",
    # XSTest type normalisation
    "safe_contexts": "safe_contexts",
    "not-for-all-audiences": "not_for_all_audiences",
}


def normalise(raw, axis=None):
    """Map any spelling used anywhere in the program to a registry ID.

    Returns None if the value is not in the registry, which is a real result:
    it means a label exists somewhere that nothing has reconciled. Those are
    surfaced by coverage() rather than silently bucketed.
    """
    if raw is None:
        return None
    k = str(raw).strip().lower().replace(" ", "_")
    k = _ALIASES.get(k, k)
    if k in REGISTRY and (axis is None or REGISTRY[k]["axis"] == axis):
        return k
    return None


def describe(cat_id):
    return REGISTRY.get(cat_id)


def coverage():
    """Which axes are populated and how densely. For onboarding and audits."""
    out = {}
    for axis in AXES:
        ids = [k for k, v in REGISTRY.items() if v["axis"] == axis]
        out[axis] = {"n": len(ids), "ids": sorted(ids)}
    return out