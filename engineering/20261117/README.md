# LipNet engineering mission status — 17 November 2026

This mission is **not complete**. The [62-finding register](FINDINGS.md) retains every finding 01–48 and X01–X14 with a disposition, correction and limitation. Its [JSON version](findings.json) additionally records files and verification references. No scientific or frontend acceptance is inferred from a passing software suite.

The resumed baseline was clean `main` at `bf0f9af3`, containing three newer commits than the saved implementation session. Work is preserved on `engineering-resume-20261117`. A recoverable tracked-source backup is at `C:\Users\Dell\.codex\lipnet_safety\resume_20261117`; the original source/artifact baseline remains at `C:\Users\Dell\.codex\lipnet_safety\20261117_124132`. `test_tmp.npy` was already absent at the original implementation baseline; it was not deleted or recreated.

Concurrent frontend/personal-adapter edits were detected and preserved. Backend overlap was paused until the user confirmed ownership, then reconciled: collection remains authenticated staging and the global unvalidated personal service remains unavailable for serving. Frontend component/state files belong to Antigravity and were not independently edited by this Codex implementation. The changed-file inventory distinguishes those preserved concurrent edits.

## Verification

The verification record and logs report actual commands and outcomes, including intermediate failures. The final broad backend run passed **716 tests with 3 skips in 163.61 seconds**. A preceding targeted run passed **92 tests** after additional privacy/resource corrections. Intermediate failures and their corrections remain in the verification record.

The frontend lockfile completed an actual clean `npm ci --ignore-scripts`, TypeScript/production build, **6 adaptive**, **41 live** and **26 mathematical-rendering** Node tests. The audit reported **zero vulnerabilities** at that execution. Further frontend edits are concurrent and require testing at their final coordinated revision. No real-browser multi-tab/offline acceptance has been claimed.

The recorded [artifact comparison](artifact-comparison.json) found **10,120 unchanged baseline artifacts**, with none missing. The protected alphabet hash is `7c00d5117b7fae9a76c05980ca70edd680e79be4c8a152ea2595c98ded90c4c3`; the eight-word head hash is `bc07a9eca0979e556af573a387733396a891fe41faa10715c9f4d323e9c4cc74`.

An actual selected eight-word model request completed on one genuine historical **training** ROI with that head hash. The smoke took **17.73 seconds including loading**, on an i3-1115G4 with 2 physical/4 logical cores and 8,311,832,576 bytes installed RAM. It was one execution check, not an accuracy/latency benchmark, fresh participant result or final-demonstration hardware guarantee. The first smoke attempt completed inference but failed optional psutil metadata collection; its failure log is preserved and the successful retry omits the unavailable memory measurement.

## Dataset and model evidence

The [dataset-quality artifact](dataset-quality.json) reports current structural checks: 32,891 GRID records with resolvable paths, missing W, and no new full-corpus decode. Each legacy LRW split has 500 rows and 485 missing paths. All 45 local LRW MP4 files have the same byte hash, so the real-data loader rejects them as development fixtures.

The [model evidence artifact](model-evidence.json) binds available historical reports to checkpoint hashes and labels scope. Eight-word historical cached-feature replay reports raw accuracy 182/192 (94.79%), accepted accuracy 123/123, coverage 123/192 (64.06%), and phrase-negative false acceptance 5/60 (8.33%) under the replayed consensus policy. Phrase negatives are not independent isolated OOV evidence. Historical held-out identities F10/F11/M08 were previously exposed; they are not new participants. Twenty-word results are preserved as research evidence with their own scope and serving disabled by default. Unbound alphabet historical summaries do not become authenticated reliability evidence.

## Deliverables and blockers

- [Security/privacy verification](SECURITY_PRIVACY.md)
- [API/frontend migration contract](FRONTEND_CONTRACT.md), [endpoint inventory](ENDPOINTS.md), [OpenAPI](openapi.json)
- [Deployment/reproducibility](DEPLOYMENT.md)
- [External validation/license/consent checklist](VALIDATION_PROTOCOL.md)
- Verification records, changed-file inventory and final audit alongside these documents

Docker is absent, so no container build/runtime pass is claimed. The existing encoder environment runs reference inference but has dependency conflicts and has not passed a clean reproducible installation. No new consenting participants, genuine independent W/LRW/OOV/webcam evidence or institutional license/similarity authority are available. Antigravity browser integration remains coordinated work. Remaining in-scope engineering gaps are listed explicitly in the register and privacy report; they have not been silently classified as external blockers or completed.

Historical reports and datasets are preserved. This status document corrects their interpretation without overwriting original evidence.
