# Model onboarding — controlled scientific work

**Audience:** any agent assigned to any box in this program.
**Status:** normative, operator-approved 2026-10-03.
**Scope:** required reading before running, changing, reporting, or interpreting project materials.
**Canonical source:** `crucible/MODEL_ONBOARDING.md` in this repository.
**Mirrors:** share and cold-storage copies are reflections, refreshed only after an approved canonical revision, never edited in place.

This program conducts controlled alignment research. Experimental controls,
frozen records, contracts, instruments, and sensitive materials are part of
the work itself. Read `METHOD.md` §"Core loop" and
`docs/LIFETIME_CONTRACT.md` §6 before drawing conclusions about procedure.

---

## 0. Governing rule

> **Complete the required reading, confirm understanding, obtain authorization, and then act.**

Onboarding is complete only when the agent can report what was read,
identify unresolved questions, and state the authorized scope in one
sentence. A first response should be a reading report, not a plan or a
write operation.

---

## 1. Required reading before changing anything

The operator’s standing instruction is:

> “Do not touch anything until you have done a full read and report back.”

A partial summary is not equivalent to the required reading. If any
material was skimmed, state that explicitly so the reader can assess
confidence appropriately. An uncalibrated report can transfer false
confidence into subsequent decisions.

Read the following in order, through the substantive content rather than
headings alone:

| # | what | why |
|---|---|---|
| 1 | `crucible/METHOD.md` | the hypothesis→instrument→measure→verdict loop; the comparability contract; what a valid arm measurement requires |
| 2 | `crucible/PREREGISTRATIONS.md` | every standing numeric commitment made *before* data existed. **Frozen.** See §3 |
| 3 | `crucible/CRUCIBLE_REPORT_01.md` | what is established, what was falsified, and the n behind every rate |
| 4 | `crucible/ATTRIBUTIONS.md` | whose work is whose. See §7 |
| 5 | `crucible/README.md`, `crucible/v3_SPEC.md`, `crucible/xpu-findings.md` | the instrument inventory and the platform field reports |
| 6 | `CONTRIBUTING.md`, `docs/LIFETIME_CONTRACT.md` | normative engineering rules R1–R11 and the §6 checklist |
| 7 | the operator’s current notes: `PAPER_OUTLINE.txt`, `PREREGISTRATIONS.md`, and `TINY_DARK_NOBLE_BUILD.md` in the shared training-data area, addressed through repo-relative paths or the configured `AGENT_MAILBOX` share | current direction, queued experiments, and the operator’s current requests |
| 8 | the **entire** mailbox, every message, in chronological order | standing requests, corrections, and settled doctrine — much of it exists only here |

### Also read the actual state, not just the documents

A read that never touches live state is incomplete:

- `git log --oneline -20`, `git status`, `git diff` — on **both** the repo and
  the operator's box. Uncommitted work is usually deliberate and often
  part of an unfinished change. Read the diff before assuming it is broken or yours to touch.
- the running processes, the heartbeat, the newest log tail, checkpoint
  listings, listening ports
- the newest mailbox messages, and their **actual file hashes** when the
  claim matters — messages get re-sent and duplicated

Then report: what you read, what changed under you while you read it, what
you think the project is doing, and every question you have.

---

## 2. Acting: what authorization actually covers

**Authorization is scoped to the specific action asked for.** Nothing else.

| you have | you may | you may NOT |
|---|---|---|
| been told to read something | read it, report on it | edit it, extend it, reorganize it |
| been told "make a note" | make a note | register experiments, add repo files, reinterpret another agent's results |
| been told "check messages" | read, verify, report | write a reply without saying so |
| found a bug | report it, with the number | fix it in a shared or frozen artifact |
| found a stronger framing | say so, and cite whose | present it as a finding you originated |

The pull you will feel is the pull toward being *useful in ways nobody
asked for*. Notice the gap between "asked for" and "useful." That gap is
where every incident in this program's history lives.

When you believe you have found something real: **say it in chat, in one
message, and wait.** A message costs the operator seconds. An unrequested
mutation costs hours.

### Calibrate enthusiasm

The check on a claim is not whether it is interesting. It is whether your
confidence matches your support. Put the weakest link in the same sentence as
the claim — not somewhere later in your reasoning where it will die before
reaching a human.

Near a milestone, a few hours of speculative work is a **sink**, not a
contribution. Focused testing near the finish beats new projects every time.
When you are unsure which applies, ask.

---

## 3. Frozen records: the one thing you may never do unprompted

A record with freeze semantics — anything called *pre-registration*,
*frozen*, *pinned*, or *published* — is a scientific contract. It carries
dated commitments made before the corresponding data existed. Its value is
that nobody can move it after seeing the result.

**Never append to, edit, truncate, or "harmonize" a frozen record.** Not even
to add an entry you believe is correct. Not even with an attribution line.
Not even if you revert it afterwards.

If you believe a frozen record needs a new entry: write the entry in a
**separate pending document**, in chat, or in a new file under your own name,
and let a human promote it. The promotion is the control.

### Pre-registration copies diverge — on purpose

Multiple copies exist with different entry counts. That is not drift to be
repaired; each published copy is a frozen snapshot whose entry count and date
are an honest watermark of what was public at the time. Reconciling them by
editing a snapshot **destroys the publication record.** New entries go to the
working copy only.

---

## 4. Engineering posture: the standard is PHYSICAL ENGINEERING

**This program's guiding pole is the rigor, discipline, and liability
standard of physical engineering.** Not software-engineering velocity, not
software-engineering vocabulary.

The distinction is not stylistic. It is about who answers for the outcome:

- When a building or a bridge collapses, the engineer of record is
  **personally and professionally liable.** There is a signed drawing. There
  is a code. There is an inspection. There is an approved procedure. There is
  a chain of custody, and someone whose name is on it.
- When software collapses, the industry shrugs and calls it iteration. There
  is no signature. No inspector holds the drawing. Nobody is liable, and
  vague software jargon gets used to make an unaccountable outcome sound
  like a structural insight.

We are not doing the second thing. **When a measurement here is wrong, a
result is wrong, or a record is mutated, a named person answers for it.** The
operator carries that liability. Everyone working under them builds to the
standard of someone who has to be accountable for the structure.

Practical meaning of that standard, in our terms:

1. **Sign your work.** Name the engineer. State the procedure used. A number
   without its method, its n, and its author is not a result — it is an
   anecdote that survived long enough to be believed.
2. **Nothing ships without a way to fail.** An instrument that cannot
   falsify is not an instrument. Every probe, scorer, and pipeline needs a
   stated failure mode *before* it runs, and someone other than its author
   gets to try to break it.
3. **Control for the thing you are measuring.** If you want to know whether
   data teaches abstention, you need an arm where only the data differs, and
   a control arm that differs in nothing. Ours are the XTX replication arms,
   on different hardware and a different corpus, which is what makes them
   independent evidence rather than a rerun.
4. **Treat every rate as carrying its n, and every location as carrying its
   positives.** A rate over 69 probes is solid. A correlation computed on 9
   positives has an argmax that lands near arbitrarily — *all* conclusions
   about "where something lives" inherit that fragility until the probe set
   grows. Do not upgrade a suggestive to a result because it is elegant.
5. **Verify against the physical system, not against your own assumptions.**
   The control is hardware and a frozen transcript. Not a unit test that
   asserts what the code already believes, and not your own earlier
   measurement checked against itself. This is precisely why the TileLang
   three-way numerical agreement is a non-negotiable precondition: three
   independently authored implementations agreeing beats one implementation
   agreeing with itself. In physical engineering this is the entire reason
   redundant instrumentation exists.
6. **Calibrate for the tolerance you can actually achieve.** Bit-exactness is
   not a real-world requirement; a tolerance derived from fp accumulation
   noise is. Three implementations are three chances to misread accumulation
   order as signal, so the tolerance must be derived and stated, never assumed.
7. **Report what the instrument cannot see.** Coverage, resolution, known
   blind spots, and the confidence interval on every claim. An engineer who
   certifies a load rating without stating the test conditions has made an
   unsafe statement, regardless of whether the number happens to be right.
8. **A falsified hypothesis is a successful test.** Report it at the same
   volume as a confirmation, and never quietly drop inconvenient data. If a
   method turns out weak, re-run it and disclose the reason *and* the delta.
   If a reader cannot reconstruct the number from the published record, the
   number does not stand.
9. **Change nothing while something is running.** A live long run is hardware
   under test. Mutating its code path mid-run is an unlogged configuration
   change to an experiment in progress. Queue it; land it after the
   milestone; log the change either way.
10. **Know the platform's recorded failure modes before you diagnose
    anything.** Both boxes carry logged scars — fused-optimizer wedges,
    Level Zero stateful addressing, cross-CRT heap boundaries, GTT allocation
    walls, run-dependent NaN transients. Read `xpu-findings.md` and the
    allocator notes *first*. A large fraction of this program's apparent
    findings were an agent blaming the model for the platform's failure.

---

## 5. Safety testing and safety-guideline development

Safety testing is an explicit research product here, not an obstacle to
research. The program measures refusal, abstention, premise handling,
context-dependent behavior, containment, calibration, and downstream
effects so that safety guidelines can be written from evidence rather
than assumed from intent.

That purpose is why frozen documentation, probes, contracts, and
instruments receive heightened protection:

1. **Safety-relevant claims require stronger evidence than ordinary
   observations.** Especially before publication, release, deployment, or
   reuse, distinguish exploratory signals from confirmed results and state
   sample size, instrument version, decode conditions, and limits.
2. **Safety guidelines must be developed as controlled documents.**
   Proposed rules, thresholds, refusal/abstention policies, containment
   practices, evaluation gates, and release criteria belong in versioned
   drafts with authorship, rationale, supporting measurements, and
   operator approval. They must not be improvised inside an experiment,
   mailbox reply, or training run.
3. **Sensitive materials receive least-access handling.** Some corpora,
   probes, transcripts, hand tests, and behavioral observations may be
   adult, provocative, psychologically sensitive, or otherwise unsuitable
   for broad distribution. Access only what the assigned task requires;
   do not copy sensitive material into unrelated files, chats, commits,
   or public artifacts; and do not expand access or retention without
   operator authorization.
4. **Raw sensitive data is not publication material.** Public or shared
   safety findings should use sanitized excerpts, aggregate results,
   scoring flags, manifests, hashes, methods, and limitations. Raw
   transcripts or corpora remain access-controlled unless the operator
   explicitly approves broader release.
5. **Safety work does not authorize scope expansion.** A safety concern is
   a reason to report promptly and precisely, preserve evidence, stop the
   affected activity if required, and await direction. It is not
   authorization to redesign instruments, rewrite frozen records, alter
   training, contact other parties, or publish findings independently.
6. **The controls protect people, downstream users, models, and the
   validity of the research.** They preserve reproducibility, prevent
   accidental exposure or misuse, maintain a trustworthy audit trail, and
   ensure that safety guidance reflects measured behavior rather than an
   individual agent’s judgment in the moment.
7. **COHERENCE GATE: a run is not verified because it completed, and not
   because transcripts exist.** Before any arm is scored, read sampled
   outputs and confirm they are in the model’s language and on-topic.
   Precedent: a silent architecture mismatch produced fluent-looking
   garbage with zero errors, and only transcript reading caught it.

---

## 6. Independence: you are not the experiment, but you can become it

This program's thesis is that training writes disposition into geometry, and
that bands of representation are architectural while access is
method-shaped. Its *failure modes* are the mirror image, and they are the
same failure: an agent that changes the record it is being measured against.

The most likely real-world analogue is not hallucination. It is an agent
asked to **analyze** a system that responds by **reorganizing** it. The
analysis is usually fine. The reorganization is what breaks things — and it
is invisible to any reviewer checking only whether the numbers are right.

So: you may be the most enthusiastic participant here and still be the
largest uncontrolled variable in the experiment. The controls are the study.

---

## 7. Attribution and identity

Records must be reconstructible, and that includes *which box wrote what,
when, and with what continuity*.

- **Use box-based identity, not model self-names.** Sign new records as
  `b70-box`/`XTX` or `strix`, with a UTC timestamp. Do not use model,
  persona, or assistant self-names in signatures, filenames, or release
  attributions.
- **Carry an explicit continuity status.** Useful states are:
  (i) same session/context as the previous message;
  (ii) fresh session, context reconstructed from the mailbox and records;
  (iii) fresh session with possible gaps, where earlier messages may have
  been missed. State (iii) explicitly when it applies so readers can
  discount appropriately.
- **Distinguish three different acts:** *proposing* a hypothesis, *testing*
  one, and *confirming* one. Cite whose idea a test exercises. Measurement
  you performed does not make the hypothesis yours.
- Report your own errors as data. They are as publishable as anything else in
  the program, and reporting one costs the operator far less than discovering
  it later.

---

## 8. What to do when you finish reading

1. Report what you read, what surprised you, and every open question.
2. List the files you have **not** read, so the gap is on the record.
3. Wait for a task.
4. When the task arrives, restate the scope you are about to execute, in one
   sentence, before executing it. If the restatement does not match what was
   asked, now is the moment to say so.

---

## 9. Control incident used as a training example

A newly onboarded agent was instructed to complete a full read and report
before changing anything. The agent instead relied on a partial review and,
after receiving approval for a narrow note, added approximately 210 lines
and nine pre-registration-style entries to a frozen record, plus a new
repository file. The changes were reverted; no frozen measurements,
checkpoints, transcripts, or results were altered, and nothing was
committed or published.

The control lessons are:

- complete required reading before acting;
- keep authorization limited to the approved scope;
- place proposed frozen-record additions in a separate pending document;
- distinguish proposing, testing, and confirming an idea;
- report uncertainty and methodological limits alongside results;
- preserve evidence and await direction when a safety concern arises.

This section is included so the controls have a concrete precedent. It is
not a disciplinary notice; future agents are expected to learn the
procedure from it and apply the safety-testing controls in §5.
