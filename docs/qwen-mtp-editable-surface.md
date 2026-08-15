# Qwen 3.8 native-MTP track — the definitive editable-vs-trusted table

> **NOT LIVE — 3.6 → 3.8 cutover in progress.** The editable/trusted LINE below
> is a design decision and survives the model change unchanged; the ARTIFACTS on
> the trusted side are re-provisioned during the 3.8 bring-up. As of 2026-08-14
> the **target checkpoint** is our own mlx-0.32.0 conversion of the official
> bf16 base, `EigenLabs/Qwen3.8-27B-4bit` (geometry verified identical to 3.6),
> published at `eda45ab47f465d08d6558f0353a2346e2eb9d5b3` with a generated
> 10-record byte manifest; the **MTP head** `EigenLabs/Qwen3.8-27B-MTP-bf16` is
> published at `26a328e070875b0314d652a039b6b59902690f03` with a 4-record one
> (both repositories are private); goldens and calibration remain
> `QWEN38-PENDING-RELEASE`.

**Track:** `qwen3.8-27b-mtp-v1` · **Status:** operator-ratified 2026-08-14 ·
**Machine-readable authority:** `editablePaths` in `benchmark.json`
(byte-identical mirror `benchmark.qwen-mtp.json`). This document is the prose
authority for *why* each side of the line is where it is; where the two ever
disagree, the manifest wins and this file is the bug.

## The rule that decides every row

> **Anything that only PROPOSES tokens is EDITABLE. Anything that VERIFIES,
> MEASURES or LEDGERS is TRUSTED.**

That is not a compromise between openness and integrity — it is the reason
openness is *safe* here. Every token this track emits is decided by the
organizer-pinned target and then re-checked, after the clock has stopped, by
the trusted parent against a hidden serial trajectory the candidate never sees.
A drafting policy, a draft depth, even a *different MTP head* can change how
fast the right answer arrives; none of them can change what the answer is. So
the whole proposal side can be handed to the competitor without weakening a
single guarantee, and the verification side cannot be handed over at all.

The competitive surface is therefore a **union**:

1. **MLX runner and kernel optimisation** — what this challenge family has
   always been, and still the primary axis: the Qwen 3.6 text tower, the
   offline transform, and the vendored MLX/Metal kernel families the forward
   pass dispatches.
2. **The full speculative apparatus** — drafting code, draft depth and
   per-round schedule, and the MTP head weights themselves.

## EDITABLE

| Path | Role | Why it is safe |
| --- | --- | --- |
| `mtp-head.manifest.json` | Head declaration: `source` (`pinned` \| `remote` \| `in_branch`), `source_url`/`path`, `sha256`, `bytes`, `max_bytes` | The head only proposes. The runner fetches and digest-verifies pre-sandbox, fails closed on mismatch or oversize (2 GiB), and applies the result to the **candidate leg only** — the serial denominator always runs the §9d-pinned head. |
| `mtp-head/` | In-branch head weights (small-delta fallback) | Same argument. Exempt from the source byte budget (`editableSurfaceByteBudget.exemptPaths`) and bounded instead by the manifest's declared digest + byte count. |
| `Sources/MLXFastModel` | The Qwen 3.6 runtime tower **and the whole MTP apparatus**: `Qwen36MTPBlockSession` (drafting, verify-block assembly, accept walk, KV snapshot/rollback/repair), `Qwen36MTPHeadAttachment`, `Qwen36MTPTarget`, `Qwen36MTPLimits`, plus `Qwen35*` attention/cache/MLP/RoPE/ops | The session proposes tokens and shapes the round; the target decides them. Depth and schedule live here (see the depth note below), so "draft 1", "draft 8", "draft 0 this round" and any adaptive policy are all ordinary source edits. |
| `Sources/MLXFastTransform` | Offline checkpoint transform | Long-standing editable path. Its output is hashed into the ranked audit (`qwen-mtp-weights.sha256`) and the transformed tree is re-validated by the trusted harness's `TransformVerification` before any gate runs. |
| `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`, `Qwen35MTP.swift`, `Qwen35MoE.swift` | Vendored Qwen model definitions, **including the vendored MTP head module** | Forward-pass code. Same proposal/decision split. |
| `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/*` (14 files: attention utils, KV caches, compiled decode, RoPE, switch layers, …) | Vendored inference-path support | Kernel/runtime optimisation surface. |
| `Vendor/mlx-swift/Source/Cmlx/mlx-generated/*` (28 files) and `Vendor/mlx-swift/.../backend/metal/kernels/**` (43 entries incl. `steel/attn`, `steel/gemm`) | The MLX kernel families the forward pass dispatches | The primary axis of the challenge. Bit-exactness of the emitted stream is enforced downstream, so a faster kernel that changes a *logit value* is fine and a kernel that changes a *decision* is caught. |

**89 entries total.** The count is asserted mechanically by
`QwenMTPTrackNamingTests`; README quotes it too.

### The depth/schedule freedom, stated precisely

The trusted driver no longer dictates a draft depth.

* The parent offers a per-round ceiling; the **candidate chooses the actual
  draft count**, 0…8, per round, adaptively if it likes.
* `8` is `MLXFastConstants.qwenMTPMaxDraftDepth`, a **trusted** constant. It is
  the only bound, and it exists for one reason: it bounds the verify width a
  round may ask the target for.
* **Empty-draft rounds are legal.** Under serial-anchored scoring a
  non-drafting candidate *is* serial and scores 1.0, so this games nothing.
* The candidate **declares nothing** about depth to the harness. Effective
  per-round draft counts are read out of the ledger and summarised in the
  sealed report (`effective_mean_draft_len`, `effective_max_draft_len`,
  `effective_draft_lengths`, `non_drafting_round_count`) — which is also the
  fix for the requested-vs-effective provenance gap (k-matrix finding 5): the
  report used to carry only the *requested* depth, so a run that drafted
  something else looked identical to one that did not.
* Every ledger-closure and bit-exactness rule is derived from the **actual**
  draft count of each round, never from a declared or requested one.

## TRUSTED CORE — never editable

| Path / artifact | Role | Why it cannot move |
| --- | --- | --- |
| `fixtures/reference_qwen3_8_27b_4bit.sha256`, `fixtures/qwen3_6_27b_config.json`, `fixtures/qwen3_6_27b_tensor_inventory.json` | Target weights + transform contract | The target is the *decider*. If the target can move, nothing downstream means anything. |
| `fixtures/qwen3_8_27b_mtp_head.sha256` | The **pinned** head's byte manifest | Pins the baseline leg's head and the `source: "pinned"` default. A submission brings its own head through the manifest, not by re-pinning this. |
| `fixtures/qwen3_8_27b_mtp_track.json` | Track contract fixture: enablement flags, timed prompt pool + digests, scoring semantics, calibration | The scoring contract itself. |
| Tokenizer + `Sources/MLXFastCore/QwenChatTemplate.swift` | Prompt framing | Changes the question, not the answer speed. |
| Hidden goldens, GPQA fixtures, `correctness_prompts/**` | Organizer material | Out of bounds for this repo entirely — findings get reported, never fixed in place. |
| `Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift` | **The verify/accept/ledger machinery**: the round journal, the post-window reference replay, `requireStructurallySound`, exactness against the serial trajectory | This is the audit. Byte-verified against the trusted ref before every build (`verify-trusted-source-scope.sh`). |
| `Sources/MLXFastTrustedHarness/QwenRuntimeMTP.swift` | `QwenMTPRowAccounting` (the L3 row equations), report shape, contract-violation kinds | Same. |
| Rest of `Sources/MLXFastTrustedHarness` | Benchmark support, editable-surface byte budget, transform verification, vendored-Metal fingerprint | Same. |
| `Sources/MLXFastCLI` | The trusted CLI: verbs, the evidence payload, `score.json` emission | Same. |
| `Sources/MLXFastCore` | Constants (incl. `qwenMTPMaxDraftDepth`, the baseline constants, the GPQA floor), `Score`, `Golden`, `Safetensors` | Same. |
| `Package.swift`, `Package.resolved` | Which sources feed which product | Frozen; byte-verified. A submission that could repoint a target could replace the trusted binary. |
| `Sources/MLXFastHarness` (the participant **worker support** target) | Worker protocol layer: request validation, `mtp_decode_*` dispatch, **and the `mtp_reference_prefill` / `mtp_reference_rows` handlers the parent's audit replay calls** | Mixed-role, resolved to trusted — see the note below. |
| `.github/workflows/**`, `.github/scripts/**` | Gates, overlay, surface enforcement, R2 fetch, static review | The pipeline cannot be part of the surface it judges. |
| `benchmark.json`, `benchmark.qwen-mtp.json` | The manifest naming the surface | Read from the BASE commit by every gate, precisely so a submission cannot widen itself. |
| Trusted halves of `benchmark.sh`, `benchmark-qwen-mtp.sh`, `setup.sh`, `setup-qwen-mtp.sh`, `tools/**` | Build/measure orchestration, fan control, metallib build | Timing and telemetry. |
| Timing/telemetry: parent wall clock, `block_request_seconds`, macmon capture, thermal gate, stall guardrail | Measurement | The measured party never asserts its own measurement. |
| `deploy/qwen36-mtp/measure-qwen-mtp-job.sh` (operator mirror, installed box-side, §9d-signed) | The paired timing wrapper and the sealed `results.json` | Not in this repo, not in any submission's reach. |

### Mixed-role files, and how each was resolved

**`Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift` → TRUSTED.** It is the
sandboxed worker's *protocol codec*, not its drafting algorithm: it validates a
round request, hands the depth to `Qwen36MTPBlockSession.generateRound` and
serialises the result. All drafting, verification-block assembly, acceptance
and rollback logic it calls lives in `Sources/MLXFastModel`, which is fully
editable — so depth/schedule/policy freedom is *complete* without it, and
opening it buys a competitor nothing but the ability to break the wire format.
It also hosts `mtp_reference_prefill` / `mtp_reference_rows`, which the trusted
parent's post-window audit replay calls; those are verification by the rule
above. (The same directory carries a compiled-but-unreached copy of the driver
sources; the *trusted* copy the CLI runs is `Sources/MLXFastTrustedHarness`,
byte-verified before every build.)

**`Sources/MLXFastModel/Qwen36MTPReferenceSession.swift` → EDITABLE, with a
bounded and disclosed exposure.** This file serves the audit replay from inside
an editable directory. It is pre-existing (that directory has been
`editablePaths[0]` since the track was authored) and it is *not* a scoring
exposure:

* the replay runs **after the clock stops**, so tampering buys no time;
* it is only reached for **rejected-tail rows** — rows past the first rejection
  in a round, which are never emitted tokens;
* every **emitted** token is checked against the hidden serial golden, which
  the candidate cannot read or influence;
* what a tampered reference session *could* do is make a candidate's own
  rejected-tail rows agree with themselves, i.e. convert a would-be
  `rejectedTailDiverged` failure into a pass.

So the residual is a fidelity-audit weakening with no score gain.
**Flagged for the operator** as the one recommended follow-up: move this file
into a trusted-scope target that the worker links but cannot edit. It is not
done here because it needs a new SwiftPM target inside the byte-pinned trusted
scope, which is a bigger change than the surface recut and deserves its own
decision.

## What changed on 2026-08-14, in one paragraph

Head weights, drafting code and depth/schedule joined the kernel/runtime
surface, so the manifest gained `mtp-head.manifest.json` and `mtp-head/` (87 →
89 entries) and a byte-budget exemption for the weights path; the pinned-head
booleans (`uses_pinned_mtp_head`, `uses_native_mtp_head`, `mtp_head_attached`)
stopped being pass-conditions and became recorded provenance alongside a new
`head_provenance` (sha256 + source) in every sealed report; the trusted driver
gave up its depth dictate to `qwenMTPMaxDraftDepth = 8` with empty-draft rounds
legal; and scoring was re-anchored at serial = 1.0 — the median of eight raw
serial-relative speedups, floored at 0.90, ceilinged at 3.0, with the pool's
no-op references retained as informational diagnostics that no longer divide
anything.
