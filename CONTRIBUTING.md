# Contributing to VulkanVM

## Ground rules

1. **Read the agent onboarding first.** `crucible/MODEL_ONBOARDING.md` is
   normative: complete required reading, confirm scope, obtain authorization,
   and never edit frozen records in place.
2. **The lifetime contract is normative.** Read `docs/LIFETIME_CONTRACT.md`
   (§6 checklist) before touching allocation, export/import, offload, or
   transport code. A PR that violates R1–R11 is a bug, no matter what it
   measures.
3. **Fail soft, fail loud.** GPU/driver operations return `nullopt`/`false`
   on failure, never throw across the API boundary — but every failure path
   must log at ERROR with the failing parameters. Silent fallbacks hide
   driver bugs (we have the scars to prove it).
4. **Driver verdicts are data, not failures.** When a driver refuses an
   operation a test wanted (export/import refused, missing extension),
   SKIP with a reason citing the driver + version. FAIL only when *our*
   logic is wrong. See `p2p_xn_test.cpp` and `network_test.cpp` for the
   pattern — and never feed a known-refused operation to a driver that
   crashes instead of returning an error (Intel Arc lesson, 2026-09-17).
5. **Validate GPU-affecting changes on hardware.** CPU-only tests
   (`buddy_test`, `chonk_slab_test`) run anywhere; anything touching
   allocation/export/import/copy needs a Tier-1 device run before merge.
   Enable Vulkan validation layers (`VVM_ENABLE_VALIDATION=ON`) for any
   change touching allocation, export, or import.
6. **No new vendor heuristics without measurement.** Vendor identity may
   only gate behavior with a measured driver result cited in a comment
   (driver + version + date). Prefer capability probes
   (`queryExternalMemoryCaps`, `queryPeerAccess`); keep a
   `VVM_ALLOW_*`/`VVM_*_POLICY` env override for bring-up.

## Style

- C++17, `clang-format` enforced in CI (`.clang-format` at root).
- `[[nodiscard]]` on every factory/allocate/export/import function.
- `VVM_LOG_*` takes literal `{}` placeholders only — no format specs
  (`{:x}`), no printf verbs. Pre-format with `snprintf` when needed.
- Comments explain *why* (measurements, driver quirks with versions);
  the code should explain *how*.

## Docs

- Behavior changes update the normative docs (`LIFETIME_CONTRACT.md`,
  `THREAT_MODEL.md`, `HARDWARE_SUPPORT.md`) in the same commit.
- New hardware evidence goes in `HARDWARE_SUPPORT.md` with driver
  versions and the exact test/benchmark run — tier promotions need proof.
