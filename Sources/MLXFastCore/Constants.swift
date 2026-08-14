public enum MLXFastConstants {
    // Qwen native-MTP track identity. The artifact is named Qwen3.8; its
    // immutable internal architecture name is `qwen3_5_text` (the raw
    // checkpoint declares `qwen3_5`), which is why the runtime sources are
    // Qwen35*-prefixed. That name did not move across the 3.6 -> 3.8 change,
    // and neither did the tower geometry.
    //
    // REPINNED 2026-08-14 onto the ADOPTED 3.8 backbone. The cutover commit
    // (5bfae9c) deliberately left these five at their 3.6 values while the
    // ranked workflow's MLXFAST_QWEN_MTP_CHECKPOINT_REPO / _REVISION read
    // QWEN38-PENDING-RELEASE, on the reasoning that a compiled-in PLACEHOLDER
    // would break a build that has to keep passing. That reasoning was about
    // placeholders, and it is spent: there is now a real artifact to name, so
    // the split between what the workflow pins and what the binary compiles
    // against is closed rather than widened.
    //
    // WHAT THAT COSTS, stated instead of discovered. `Golden.swift` validates
    // the provenance block of every golden against repository+revision, and the
    // CHECKED-IN public correctness goldens in correctness_prompts/ still carry
    // the 3.6 provenance, because they are 3.6 captures awaiting regeneration
    // under the 3.8 tokenizer (their workflow pins read QWEN38-PENDING-RELEASE
    // for the same reason). So those goldens are now REJECTED against these
    // constants. That is the fail-closed direction -- a 3.6 golden must not
    // validate against a 3.8 target -- but it is a real consequence: the local
    // public drift gate cannot pass until the goldens are regenerated, and it
    // should not.
    //
    // The other two consumers moved with these in the same commit: setup.sh
    // builds its Hugging Face download URL from the same pair, and the
    // BenchmarkScriptTests pin that asserts the two agree. Tests/Fixtures/
    // Qwen3627B4bit/ and fixtures/qwen3_6_27b_* keep their 3.6 names and 3.6
    // content deliberately -- they describe the checkpoint they were captured
    // from, and the geometry they record is now known to describe both towers.
    //
    // Provenance of the pin: EigenLabs/Qwen3.8-27B-4bit is OUR OWN MLX 4-bit
    // affine / group_size 64 conversion, produced under a pinned mlx 0.32.0
    // toolchain, of the official bf16 base Qwen/Qwen3.8-27B @
    // 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
    //
    // IDENTITY SWAP, second decision of 2026-08-14. These constants briefly
    // named a third-party personal-account conversion of the same bf16 base.
    // That adoption was TERMINATED by the operator's validation kill-switch:
    // the deterministic reconversion cross-check it had reserved was performed,
    // and 994 of 1,847 tensors differ numerically from our own mlx-0.32.0
    // conversion. A reference whose bytes we cannot reproduce is not a
    // reference. The termination is recorded once, in the contract fixture's
    // target.upstream_source_note; this comment names only the replacement.
    //
    // THE REVISION IS A REAL SHA AGAIN, as of the backbone publish on
    // 2026-08-14. Between the identity swap and the upload it carried the
    // backbone-publish pending marker, deliberately: a marker fails the
    // pin-completeness gate by NAME, while a stale-but-real sha would have been
    // the dangerous option because it would RESOLVE, against somebody else's
    // commit. The upload landed, so that marker is retired here and at every
    // other site that spelled it, in one commit -- setup.sh's download URL, the
    // cache path below, the ranked and provisioning workflows' three spellings
    // of the snapshot directory, and the manifest fixture's Revision header.
    // Nothing about the fail-closed posture changed: the goldens are still 3.6
    // captures and Golden.swift still rejects them against these constants.
    public static let referenceModelRepository = "EigenLabs/Qwen3.8-27B-4bit"
    public static let referenceModelRevision = "eda45ab47f465d08d6558f0353a2346e2eb9d5b3"
    public static let referenceModelName = "Qwen3.8-27B-4bit"
    public static let defaultReferencePath = "reference_weights/Qwen3.8-27B-4bit"
    public static let defaultReferenceCachePath = ".cache/huggingface/hub/models--EigenLabs--Qwen3.8-27B-4bit/snapshots/eda45ab47f465d08d6558f0353a2346e2eb9d5b3"
    public static let defaultWeightsPath = "weights"
    public static let defaultGoldenPath = "correctness_golden.json"
    public static let defaultPublicCorrectnessPromptPath = "correctness_prompts/public_longcopy_gate_english_512.txt"
    public static let defaultPublicCorrectnessGoldenPath = "correctness_prompts/public_longcopy_gate_english_512_256.json"
    public static let defaultPublicLocalSubmitGoldenPath = "correctness_prompts/public_longcopy_gate_english_512_1024.json"
    public static let defaultScorePath = "score.json"
    public static let defaultLocalIterateScorePath = "score.local-iterate.json"

    // Frozen text-tower geometry of the pinned EigenLabs/Qwen3.8-27B-4bit
    // target, mirrored from Sources/MLXFastModel/Qwen35Config.swift.
    // VERIFIED 2026-08-14 against the 3.8 config.json: every number in
    // this block was read off the 3.6 config and is byte-identical in the 3.8
    // one -- 64 layers on the same 4-layer hybrid repeat, vocab 248320, hidden
    // 5120, 24 attention heads / 4 KV heads, head_dim 256. The values were
    // CARRIED across the cutover rather than guessed and the carry is now
    // confirmed, so the QWEN38-VERIFY-AT-RELEASE hedge that stood here is
    // discharged. The backbone identity swap later that day did NOT reopen it:
    // geometry is a property of the official bf16 base and is inherited by
    // every conversion of it, and the cross-check that terminated the earlier
    // adoption was numerical, not structural.
    // They still move as a set if they ever move: this block,
    // MLXFAST_EXPECTED_NUM_LAYERS in the ranked workflow, and the contract
    // fixture's target.* geometry.
    // (MLXFastCore is trusted and cannot import the editable model target).
    // The tower is 64 layers on a 4-layer repeat: every 4th layer (index % 4
    // == 3) is full attention, the other three are gated-delta linear
    // attention. `attentionHeads` is the full-attention query-head count.
    public static let vocabSize = 248_320
    public static let hiddenSize = 5_120
    public static let intermediateSize = 17_408
    public static let numHiddenLayers = 64
    public static let attentionHeads = 24
    /// `full_attention_interval`: the tower repeats every 4 layers and the
    /// LAST layer of each group (index % 4 == 3) is full attention, so 16 of
    /// the 64 layers carry a KV cache and the other 48 carry gated-delta
    /// recurrent state. Trusted code that has to reason about the hybrid cache
    /// stack -- the runtime worker's pinned-config gate and its cache-position
    /// check -- reads the schedule from here instead of repeating the literal.
    public static let fullAttentionInterval = 4
    public static let correctnessPromptTokens = 512
    // Keep the public gate long enough to catch broad decode regressions while
    // leaving budget for the hidden GPQA behavior checks in the official job.
    public static let correctnessSteps = 64
    public static let correctnessTopLogits = 8
    public static let correctnessLogitTieTolerance = 1e-6
    public static let correctnessMaxAnchorContextTokens = 1_024
    public static let correctnessMaxFreeRunSteps = 256
    public static let correctnessMaxBehaviorPromptTokens = 2_048
    public static let correctnessMaxBehaviorSteps = 128
    public static let correctnessGPQACaseCount = 9
    // Cross-machine greedy decode can drift on hidden GPQA even with pinned
    // Swift/MLX. Semantic GPQA behavior captures a short continuation for the
    // private judge; exact token enforcement stays on the long copy gate and
    // non-semantic behavior fixtures.
    // 128 (was 64; before that 10, DeepSeek-era): the 10->64 history and its
    // calibration runs predate the GPQA prompt-encoding (BOS) fix and
    // measured degenerate no-BOS completions, so they no longer bind. With
    // BOS the reference answers letter-first and then explains; 128 lets the
    // explanation finish for the judge instead of cutting mid-sentence.
    // Generation happens in the untimed gates phase (never the frozen timed
    // window), so the cost is ~10-15s of job wall-clock, not score.
    public static let correctnessGPQAMaxNewTokens = 128
    // Semantic judging uses short hidden GPQA answers as a baseline-calibrated
    // gate for optimizations that preserve the exact prefix but damage answer
    // sense. 9 (was 5): raised to the full fixture together with the GPQA
    // prompt-encoding (BOS) fix -- selection takes the first N budget-valid
    // cases in file order, and the old window of 5 contained only two of the
    // five cases the correctly-prompted reference answers right. Per-case
    // cost is one short untimed generation plus one judge call; the 4 extra
    // cases add roughly a minute to the job.
    // The captured answer is a prefix of the behavior-gate generation, so
    // semanticGPQAMaxNewTokens is only effective up to
    // correctnessGPQAMaxNewTokens (and must stay <= correctnessMaxBehaviorSteps).
    public static let semanticGPQACaseCount = 9
    public static let semanticGPQAMaxNewTokens = 128
    // 7 of 9, set from measurement rather than prediction. The gate compares
    // the candidate against the pinned reference model's own recorded answers
    // (accepted_responses in the hidden fixture), so it is a regression check:
    // an unmodified candidate reproduces the reference on every case by
    // construction, independent of whether those answers are factually right.
    // That is the point of the design -- the reference model is at chance on
    // these questions, so a correctness-based gate could only ever sit on the
    // noise floor (see the 2026-07-27 measurements: 1-4 of 9 correct depending
    // on option order, with a single case carrying the entire margin).
    // Calibration, 2026-07-27, offline against the real gate script:
    //   self-match (unmodified candidate), 27 runs / 243 judgements: 9/9 every
    //     run, zero variance, including the one case whose reference output is
    //     degenerate.
    //   three answers changed to a different option, 8 runs: exactly 6/9 every
    //     run, failing only the changed cases.
    //   answer content preserved but label flipped or tail truncated, 8 runs:
    //     9/9 -- cosmetic near-tie drift is tolerated.
    // So judge nondeterminism costs nothing, each damaged answer costs exactly
    // one case, and a floor of 7 absorbs two independent damaged answers. It
    // also clears the >= 6 needed to reject a submission that hardcodes one
    // fixed letter: the reference selects a spread of letters, so a constant
    // answer matches at most 5 of 9.
    // Earlier floors of 1 were calibrated against pre-BOS-fix runs whose
    // reference never answered at all, so they justified nothing.
    // Keep in sync with the workflow env MLXFAST_SEMANTIC_GPQA_MIN_PASS and
    // run-semantic-gpqa-gate.sh. Regenerating the fixture's accepted_responses
    // (new prompts, token budget, or reference checkpoint) invalidates this
    // calibration -- re-run it.
    public static let semanticGPQAMinPassCount = 7
    public static let benchmarkPrefillPromptTokens = 512
    // Stable public identifier for the private timed-evaluation prompt. The
    // prompt bytes remain operator-provisioned; changing this identifier is a
    // ranking-contract change and forces build-hash-keyed timed oracles to be
    // regenerated.
    public static let benchmarkEvaluationTargetID = "lowsim-prose-qwen38-v1"
    // Offline prompt-lookup susceptibility gate. The analyzer simulates a
    // longest recurrent suffix predictor over these orders. At <= 3% accepted
    // draft tokens, even an idealized zero-overhead predictor is capped near
    // 1.03x before verification and lookup overhead.
    public static let benchmarkNGramSelfSimilarityOrders = [1, 2, 3]
    public static let benchmarkMaxPromptLookupHitRate = 0.03
    // Scored decode is parent-measured wall time for decode setup plus this
    // many checked token steps. Charging setup prevents submitted model code
    // from precomputing future decode tokens in an unscored seed-prefill phase.
    public static let benchmarkDecodeSteps = 128
    // Local iterate charges the same 512-token seed prefill as the official
    // decode window, so it must use the same denominator to produce a
    // comparable decode seconds-per-token estimate.
    public static let localIterateBenchmarkDecodeSteps = benchmarkDecodeSteps
    // Local submit uses a longer public fixture so the Yukon pre-submit hook
    // exercises one continuous decode trajectory for about ten minutes instead
    // of repeating the short local-iterate correctness window.
    public static let localSubmitBenchmarkDecodeSteps = 1023
    public static let localSubmitBenchmarkRepeats = 1
    // Seed measured decode with the full prompt. A short instruction-prefix
    // seed can free-run differently across Apple Silicon/MLX versions even
    // when teacher-forced correctness agrees, which makes the timed oracle
    // fragile for reasons unrelated to kernel performance.
    public static let benchmarkDecodeSeedTokens = 512
    // Official paired timing runs LAST at workflow level, after correctness,
    // GPQA, and the hidden-material scrub. The on-box wrapper launches the
    // baseline and candidate in fresh worker processes, and each timed prefill
    // starts without an in-process warmup. The calibration below used that
    // same shape, so keep zero warmup and one measured run.
    public static let benchmarkPrefillWarmupRuns = 0
    public static let benchmarkPrefillTimedRuns = 1
    // Acceptance bands (see AcceptanceBand + docs/thermal-variance-investigation.md).
    // Prefill and decode are noisy single measurements, gated against the same-VM
    // paired baseline B (which cancels host-speed differences). Each run's value must
    // land within [B*(1-down), B*(1+up)]; > +up = slowdown/regression (fail),
    // < -down = improvement too large for one submission / lucky reading (fail).
    //
    // Prefill: +/-5% symmetric -- prefill is not a real optimization axis, so it is a
    // health gate (regression and lucky-fast both fail past 5%).
    //
    // Decode: +2% regression / -5% gain -- tight on regressions (decode is the primary
    // scored axis), and a single submission's decode gain is capped at 5%; larger wins
    // must be CHUNKED across submissions (bounds lucky-measurement inflation and forces
    // incremental, verifiable progress). Decode is the axis the score rewards, but the
    // per-submission step is capped, not the cumulative total across submissions.
    public static let prefillBandUpTolerance = 0.05
    public static let prefillBandDownTolerance = 0.05
    public static let decodeBandUpTolerance = 0.02
    public static let decodeBandDownTolerance = 0.05
    // Poolside v2 cached calibration as of 2026-07-23: the mean of the
    // `baseline_decode_seconds_per_token` / `baseline_prefill_seconds_per_token`
    // fields published by four consecutive successful ranked runs on the
    // self-hosted M5 Max (m5-bench): 30011903540, 30015338806, 30022640438,
    // and 30027994180. Each field is measure-job's same-session timing of the
    // pinned Poolside baseline commit 15852ee52858def42ddd4f32bca7e59d275e020e.
    // Decode ranged 0.01382311946875-0.0139106712265625 (CV 0.26%);
    // prefill ranged 0.00036540633203125-0.000371515869140625 (CV 0.65%).
    //
    // These constants are NOT the ranked scoring denominator. The ranked runner
    // uses a self-hosted M5 Max (m5-bench), where measure-job times the
    // candidate and the PINNED on-box reference tree back to back in the same
    // session behind the same 40C thermal gate; the paired ratio against that
    // live same-session baseline is what overlay-paired-timing.sh folds into
    // the final score. These constants keep two roles: local-mode score
    // estimates (--local-iterate / --local-submit) and the gates-only pass's
    // placeholder timing fields, which the paired-timing overlay replaces. See
    // the paired-baseline section of docs/benchmark-window-freeze.md.
    public static let officialBaselinePrefillSecondsPerToken = 0.00036751938916015625
    public static let officialBaselineDecodeSecondsPerToken = 0.01385621216015625
    public static let scorePrefillWeight = 0.25
    public static let scoreDecodeWeight = 0.75
    public static let scorePrefillSpeedupFloor = 0.95
    public static let scoreDecodeSpeedupFloor = 0.95
    // The Poolside Laguna XS 2.1 NVFP4 text tower is ~21.6 GB; 25 GiB keeps ample
    // headroom for shard alignment/padding without approving a second full
    // copy of the model.
    public static let defaultMaxTransformedWeightsBytes = 25 * 1024 * 1024 * 1024
    public static let defaultMaxSubmissionSourceBytes = 256 * 1024 * 1024
    // Diagnostic (non-ranking) real-valued score fields are published rounded to
    // this many significant figures. Submitted model code controls its own
    // latency/memory, so every full-precision analog field it can influence
    // (RAM, bandwidth, wall/preflight/correctness/TTFT seconds, hit rate) is a
    // covert channel for exfiltrating the hidden prompt/golden it sees. Coarsening
    // these -- which carry no ranking weight -- collapses each from ~30 bits to a
    // few. The ranking fields (decode/prefill seconds-per-token and speedups) are
    // left precise here on purpose; bounding that residual channel is a publishing/
    // rate-limit decision on the scoring backend, not a repo-side change.
    public static let publicDiagnosticSignificantFigures = 2

    // DFlash block-decode track (laguna-xs-2.1-dflash-v1) protocol limits.
    // The organizer-provisioned DFlash drafter proposes a masked block in ONE
    // forward, so its ceiling is the checkpoint's own `block_size` (16) rather
    // than the retired MTP drafter's autoregressive 4. Measured peak speedup on
    // the m5-bench M5 Max sits near K=8; the ceiling stays at the checkpoint
    // value so the parent-randomized K schedule (contract layer L6) can sample
    // the whole legal range. Keep these explicit rather than derived so
    // widening either bound requires a conscious review.
    public static let experimentalDFlashMaxBlockSize = 16
    // Compatibility default for callers that do not select an explicit oracle
    // length. Mirrors the retired MTP default.
    public static let experimentalDFlashMaxTotalTokens = 128
    // Trusted-parent configured runs may exercise longer tail boundaries when
    // the selected oracle carries enough tokens. 1,536 keeps the diagnostics
    // able to wrap Laguna's 512-position sliding-window cache three times,
    // which is what makes the contract's wrap-seam leg (layer L4) reachable.
    public static let experimentalDFlashMaxConfiguredTotalTokens = 1_536

    // MARK: - Qwen 3.8 native-MTP track (qwen3.8-27b-mtp-v1)

    /// Tensors the MTP head carries.
    ///
    /// It lives HERE, in the no-MLX core, because both sides need it and they
    /// cannot share a type: the worker enforces it at load
    /// (`Qwen36MTPHeadAttachment`), and the trusted CLI reports it in the
    /// evidence payload without linking any model code.
    ///
    /// MEASURED 2026-08-14 ON THE 3.8 HEAD; the QWEN38-VERIFY-AT-RELEASE hedge
    /// that stood here is DISCHARGED. The head extracted from the official bf16
    /// base `Qwen/Qwen3.8-27B` @ `1d4bf0f2` carries **15 bf16 tensors**
    /// (849,398,784 tensor bytes in a single 849,400,347-byte
    /// `model.safetensors`).
    ///
    /// WHY IT WAS 31 AND WHY THAT IS NOT A CONTRADICTION. The old count came
    /// from the 3.6 head's own `model.safetensors.index.json`, and that head was
    /// a 4-bit group-64 MLX conversion: its 8 matrices each appear as a
    /// weight/scales/biases TRIPLE (24 entries) alongside 7 norms, which is 31.
    /// The 3.8 head is bf16 and unquantized, so the same 8 matrices are 8
    /// entries: 8 + 7 = 15. The head did not change shape; the count was
    /// counting a quantization layout, not an architecture.
    ///
    /// CONSEQUENCE, stated rather than discovered. `setup-qwen-mtp.sh` still
    /// defaults to the 3.6 head for the local path, and that head has 31
    /// tensors, so the local load now refuses on THIS constant. That is the
    /// same intended refusal the mismatched local pair already carried (a 3.6
    /// head cannot merge onto a 3.8 backbone), arriving one step earlier and
    /// with a clearer message.
    public static let qwenMTPHeadTensorCount = 15

    /// TRUSTED BOUND on the drafts a single round may actually propose.
    ///
    /// OPERATOR-RATIFIED 2026-08-14. The track no longer pins a draft depth:
    /// depth and per-round schedule are part of the competitive surface, and a
    /// candidate picks its own draft count per round — 0 through this value,
    /// adaptively if it likes. This constant is the ONE bound, and it exists
    /// for exactly one reason: it bounds the verify width a round may ask the
    /// target for, which bounds the memory and the row ledger a single round
    /// can demand. It is deliberately NOT a statement about which depth is
    /// good; the measured envelope (1.74x @ 32 tokens, 1.34x @ 128, ~1.0x @
    /// 512, depth 3 exact-but-slower) is now a competitor's problem, not a
    /// pinned parameter.
    ///
    /// It lives in the no-MLX core because the TRUSTED parent enforces it
    /// (`QwenRuntimeMTPDriver.requireStructurallySound`, over the draft count a
    /// round ACTUALLY proposed) while the worker mirrors it as a request bound
    /// through `Qwen36MTPLimits.maxDepth`. The parent's check is the one that
    /// binds: the worker's copy sits in editable model code.
    public static let qwenMTPMaxDraftDepth = 8

    /// Wire/request spelling of the same bound, kept because the worker
    /// protocol and `Qwen36MTPLimits` were written against this name. It is an
    /// alias, not a second knob — a submission that raised one and not the
    /// other would still be bounded by `qwenMTPMaxDraftDepth` at the parent.
    public static let qwenMTPMaxDepth = qwenMTPMaxDraftDepth

    /// The depth of the TRUE SERIAL CONTROL: 0, meaning MTP OFF.
    ///
    /// Operator design, and it is not a naming choice. Depth 1 is NOT serial: it
    /// is a one-deep speculative decoder that still drafts, still verifies and
    /// still accepts — measured on box 3 at 512 tokens it ran 302 rounds with a
    /// 0.699 accept rate, i.e. an ALREADY-ACCELERATED baseline. Dividing by it
    /// measures depth-2 against one-deep speculation, not against serial decode,
    /// and that is how a 0.875x "speedup" came out of a method the authors
    /// measured at 1.34x @ 128 against true serial.
    ///
    /// Depth 0 therefore drafts nothing and consults the head not at all: one
    /// token per target forward. Depth 1 remains available as a labelled
    /// speculative-depth-1 diagnostic and is never the denominator.
    public static let qwenMTPSerialControlDepth = 0

    /// SCORED. The serial denominator of this track's paired ratio: mean
    /// depth-0 seconds/token of the pinned baseline at the ranked 512-token
    /// window.
    ///
    /// Provenance: gated calibration sessions on m5-max-128gb-3, 2026-08-13,
    /// authored via `measure-qwen-mtp-job.sh --calibration-bootstrap` and
    /// installed as the on-box band in
    /// `/opt/bench-runner/state/qwen3.8-27b-mtp-v1/baseline-calibration.json`.
    /// Reproduced by every ranked run since: the pooled serial means of the
    /// four go-live calibration dispatches
    /// (31712368539, 31715555814, 31718615518, 31721547429) band against this
    /// value at ratios 1.001674 / 1.001795 / 1.000355 / 1.001264 — every one
    /// inside `[0.95, 1.05]`, and all four within 0.18% of the pinned mean,
    /// which is the evidence that this constant still describes the box.
    ///
    /// It is TRACK-SCOPED on purpose. The unprefixed
    /// `officialBaselineDecodeSecondsPerToken` above is the Poolside/Laguna
    /// serial calibration that the live DFlash track compiles against; this
    /// value is 2.74x larger and overwriting the shared constant with it would
    /// silently retune another live track's acceptance-band reference.
    ///
    /// The depth-0 leg does prompt-INDEPENDENT work (512 plain forwards), which
    /// is why one pooled denominator serves all 8 timed prompts: the measured
    /// pool spread lives entirely in the acceptance rate, which only the
    /// numerator sees.
    /// QWEN38-VERIFY-AT-RELEASE. This is a QWEN 3.6 MEASUREMENT taken on box 3
    /// in gated calibration sessions. It does not describe Qwen 3.8 27B and
    /// must be re-derived on the 3.8 baseline before any 3.8 score is
    /// published; the ranked track is held closed by
    /// MLXFAST_QWEN_MTP_CALIBRATION_READY="0" until it is.
    public static let qwenMTPOfficialBaselineDecodeSecondsPerToken = 0.037994794617407023

    /// UNSCORED — informational / historical tracking only.
    ///
    /// This track's score has NO PREFILL COMPONENT. `mtp_decode_speedup` is a
    /// decode-only ratio-of-means; the seed prefill is charged *inside* the
    /// decode measurement window, identically on both legs of every pair; and
    /// `measure-qwen-mtp-job.sh` seals `prefill_component: "none"` in the
    /// `results.json` it signs. Nothing reads this constant to compute a score,
    /// and a future reader who assumes it participates will mis-tune the track.
    ///
    /// Provenance: RUNBOOK section 3.4, measured in the same gated sessions and
    /// the same thermal/fan regime as the decode figure above, over the same
    /// 512-token prefill window the ranked workflow pins
    /// (`benchmarkPrefillPromptTokens`). 3 observations, spread 0.17%.
    ///
    /// It is deliberately NOT wired into the local-mode estimate. The Qwen-MTP
    /// local path (`benchmark-qwen-mtp.sh`, `mtp-timed`) consumes no pinned
    /// baseline constant at all: it reports a same-session paired ratio, timing
    /// both legs in the run. There is therefore no seam to redirect, and
    /// introducing one would replace a self-normalising measurement with a
    /// hardware-absolute one — changing what the local number MEANS rather than
    /// improving it.
    /// QWEN38-VERIFY-AT-RELEASE. This is a QWEN 3.6 MEASUREMENT taken on box 3
    /// in gated calibration sessions. It does not describe Qwen 3.8 27B and
    /// must be re-derived on the 3.8 baseline before any 3.8 score is
    /// published; the ranked track is held closed by
    /// MLXFAST_QWEN_MTP_CALIBRATION_READY="0" until it is.
    public static let qwenMTPOfficialBaselinePrefillSecondsPerToken = 0.00115714

    /// Semantic GPQA min-pass for THIS track, derived per NEW-MODEL-BRINGUP 7.4
    /// as `min(observed) - 1` over the four baseline-equivalent ranked
    /// calibration dispatches of 2026-08-13 (31712368539, 31715555814,
    /// 31718615518, 31721547429). Every one of them judged 9/9, so
    /// `min(observed) = 9` and the floor is 8 — one case of error budget for
    /// judge nondeterminism, which is the whole point of the -1.
    ///
    /// TRACK-SCOPED deliberately. The unprefixed `semanticGPQAMinPassCount = 7`
    /// above is the LIVE DFlash track's calibrated floor, tied to the DFlash
    /// workflow env by `reusedGateCalibrationInTheDFlashWorkflowMatchesConstants`;
    /// raising the shared constant would retune a live track that has not been
    /// re-derived. Mirrored into MLXFAST_SEMANTIC_GPQA_MIN_PASS in the Qwen
    /// workflow ONLY, and pinned to it by
    /// `theQwenWorkflowMirrorsTheTrackScopedGPQAFloor`. The shared
    /// `run-semantic-gpqa-gate.sh` default stays 7: it serves both tracks, and
    /// the value that binds is the workflow env, not the script default.
    ///
    /// KNOWN LIMITATION — this floor cannot reject a constant-"A" answerer, and
    /// raising it to 9 would not fix that. Every `answer_key` in the hidden
    /// fixture is "A" while each prompt offers four options, and the reference
    /// model is measurably position-biased toward A (it tracks the correct
    /// option only ~2/9 when option order is rotated). Since
    /// `accepted_responses` was filled from the reference model's own captures,
    /// the gate measures FIDELITY TO THE REFERENCE, not question-answering
    /// accuracy: 8 of the 9 reference captures are "A", so a constant-"A"
    /// answerer scores exactly 8 — precisely this floor. Going to 9 would only
    /// delete the judge-nondeterminism budget while still admitting it. The
    /// real fix is shuffling option order, which is organizer material and is
    /// tracked operator-side, deliberately not done in this repo.
    /// (Do NOT restate the older "the reference selects a spread of letters"
    /// rationale attached to the shared constant above — it is false for this
    /// fixture and this floor does not rest on it.)
    /// QWEN38-VERIFY-AT-RELEASE: 8 is min(observed) - 1 over four QWEN 3.6
    /// ranked calibration dispatches. NEW-MODEL-BRINGUP 7.4 requires that
    /// derivation to be re-run against the 3.8 model -- and the regenerated
    /// GPQA fixture -- before this floor means anything.
    public static let qwenMTPSemanticGPQAMinPassCount = 8

    // Criterion E residual bucket: emitted tokens that match NEITHER the
    // reference K=1 argmax NOR the reference argmax in the candidate-declared
    // block frame are counted here and must additionally sit inside the
    // reference top-2. Measured honest divergence on M5-C was 14 events across
    // 2,304 emitted positions (<= 0.61%), every one of them explained by the
    // declared-frame admission above, so this bucket exists only for residual
    // candidate-vs-reference kernel drift and is deliberately small: a large
    // cap would be spendable by a cheating submission.
    public static let experimentalDFlashResidualDivergenceBudgetPerThousand = 5

    /// Logit distance below the REFERENCE's own top-1 within which the
    /// reference's ordering is not a fact about the model. A row is admitted as
    /// a near tie when the reference prices the EMITTED token inside this
    /// envelope of its own top-1 (contract Amendment 16); the same number
    /// bounded the top-1/top-2 gap under the Amendment 14 form it replaces.
    ///
    /// Derived, and RE-derived, because the first basis was contaminated.
    /// Amendment 6 measured a maximum candidate-vs-reference top-2 logit delta
    /// of 3.375 and 2 x 3.375 = 6.75 was the envelope -- but that statistic was
    /// taken against a PRE-GENERATED golden, so it folded golden
    /// pre-generation drift into what was supposed to be build-to-build drift.
    /// Under the live post-run replay of the candidate's own chain
    /// (Amendment 15's split) the same statistic drops. Measured on M5-C at the
    /// frozen 128-token window, live replay, both fixtures, several schedule
    /// seeds:
    ///
    ///     K=1 (serial control)   0        (bit-identical to the reference walk)
    ///     K=4                    1.75 .. 2.4375
    ///
    /// So the envelope is 2 x 2.4375 = **4.875**: reordering top-1 and top-2
    /// needs `gap < |e1 - e2|`, which is bounded by twice the per-logit drift.
    ///
    /// Scope, stated rather than hidden. The same sweep measured 1.5 .. 2.75 at
    /// K=3 (the ranked width) and 2.625 .. 3.25 at the off-ranked K=8
    /// diagnostic, so against the ranked-width worst case 4.875 is 1.77x rather
    /// than the full 2x. It is NOT raised to 5.5 to buy that back: widening an
    /// admission gate is the move this contract exists to prevent, and every
    /// near-tie row observed to date has a reference distance under 1 logit --
    /// far below either candidate value. If a future honest run is rejected at a
    /// row whose distance falls between 4.875 and 5.5, THAT is the evidence that
    /// would justify the wider number.
    ///
    /// Rows inside it are admitted WITHOUT spending the residual budget, because a
    /// coin-flip the reference cannot break is not evidence about the candidate.
    /// Measured density: 2-3 such rows per 128 positions on varied prose, 0 per
    /// 128 on the degenerate self-continuation fixtures -- which is precisely why
    /// the old single-slot budget survived every test until real text was run
    /// (Amendment 10).
    ///
    /// Note this cannot be gamed: the logits belong to the reference, so a
    /// submission can neither manufacture near-tie rows nor predict which
    /// positions are near-ties without doing the reference's work.
    public static let experimentalDFlashNearTieLogitEnvelope = 4.875

    /// Cap on near-tie admissions, as a rate per thousand scored tokens. 40 gives
    /// 6 slots at the frozen 128-token window against a measured need of 3, i.e.
    /// 2x headroom. It is a backstop, not the primary control -- the envelope test
    /// above is what does the work -- but it bounds the blast radius if a prompt
    /// turns out to be far flatter than anything measured.
    public static let experimentalDFlashNearTieAdmissionBudgetPerThousand = 40

    // MARK: - L2 work-binding tolerance

    /// Absolute arm of the L2 work-binding tolerance: the largest admissible
    /// `|candidate - reference|` on a per-row top-2 logit VALUE.
    ///
    /// UNCHANGED at 4.875 by a calibration that set out to tighten it per block
    /// width and measured that it must not be tightened at all. The measurement
    /// is recorded here because the number now means something different from
    /// what Amendments 6/15/16/19 thought it meant.
    ///
    /// WHAT WAS MEASURED (M5, 2026-07-30, frozen 128-token window, live post-run
    /// replay of the candidate's own chain, Laguna XS 2.1 NVFP4 target + the
    /// pinned BF16 drafter). Logits are BF16, so one ULP at these magnitudes is
    /// 0.125.
    ///
    /// 1. HONEST SAME-BUILD DRIFT, PER VERIFY WIDTH. 50 runs, both fixtures,
    ///    widths 1-8, up to 8 schedule seeds, 12,800 comparisons, every gap
    ///    attributed to the width of the round that produced it (a run's schedule
    ///    mixes widths, so a per-run aggregate attributes to no single width --
    ///    which is why every earlier "per width" number in this contract was
    ///    actually a per-mixture number):
    ///
    ///        width  comparisons  runs     max     p99    p50    mean
    ///          1        512        2    0.0000  0.0000  0.000  0.0000
    ///          2       3182       47    3.1250  1.7500  0.250  0.3168
    ///          3       2666       44    3.3750  1.7500  0.250  0.3212
    ///          4       1964       36    2.4375  1.7500  0.250  0.3442
    ///          5       1194       28    3.3750  2.1250  0.250  0.3944
    ///          6       1438       28    4.5000  1.8750  0.250  0.3563
    ///          7        718       16    2.5000  1.5000  0.250  0.3251
    ///          8       1126       16    3.2500  2.2500  0.250  0.3635
    ///
    ///    The width dependence this table was built to find IS NOT THERE beyond
    ///    the first step. Width 1 is exactly 0 -- at width 1 the candidate
    ///    reproduces the reference's own width-1 walk bit for bit -- but from
    ///    width 2 on the DISTRIBUTION is flat in width (p50 0.25 and p99
    ///    1.75-2.25 at every width) and only the extreme tail wanders: 3.125 at
    ///    width 2, 2.4375 at width 4, 4.5 at width 6, 2.5 at width 7. Pooled over
    ///    widths >= 2 the tail is heavy and smooth -- p99 1.875, p99.9 2.875,
    ///    p99.99 3.375, max 4.5 -- i.e. the maximum is a function of how many
    ///    comparisons were drawn, not of the width.
    ///
    ///    The same width also drifts differently depending on the SCHEDULE it sat
    ///    in: width 3 maxes at 2.75 inside a `--block-size 3` run and at 3.375
    ///    inside a `--block-size 6` run. Ring-cache frame state carries across
    ///    rounds, so "the width of this round" is not even the whole input.
    ///
    /// 2. CROSS-BUILD DRIFT, measured for the first time. Amendment 6 flagged this
    ///    term as unmeasured and additive and it stayed that way for thirteen
    ///    amendments, because both the candidate and the reference worker are
    ///    spawned from the same binary. Measured by giving the reference worker a
    ///    different binary from the candidate's:
    ///
    ///        pairing                          width 1   width 2   width 3
    ///        same build (control)              0.0000    2.2500    2.7500
    ///        `-Ounchecked` candidate           0.0000       --     2.7500
    ///        one reassociated MoE reduction    4.6250    4.2500    4.0000
    ///
    ///    `-Ounchecked` changes Swift codegen and the binary hash but NOT one bit
    ///    of arithmetic: the numerics live in MLX's C++/Metal, so it reproduces
    ///    the same-build control exactly. That is the control proving the
    ///    two-binary plumbing itself perturbs nothing.
    ///
    ///    One semantics-preserving reassociation -- rotating the top-K axis of the
    ///    MoE expert reduction, the same 8 (expert, weight) pairs summed in a
    ///    different order -- produces up to **4.6250** at width 1, where
    ///    same-build drift is exactly 0. That is the cross-build term isolated,
    ///    and it is 95% of this constant. At the ranked widths it is 4.25 (width
    ///    2) and 4.00 (width 3).
    ///
    /// WHAT THAT MEANS FOR THE CONSTANT. Cross-build drift dominates the frame
    /// term and nearly exhausts the arm on its own, from a ONE-SITE change that a
    /// real submission would consider trivial. So:
    ///   * The arm must NOT be tightened. Per-width calibration would have put
    ///     the ranked width near 3.5-4.1 (Amendment 19's estimate); honest
    ///     cross-build traffic at the ranked widths already reaches 4.0-4.25, so
    ///     that would false-reject honest submissions.
    ///   * The arm is NOT 1.5x anything any more. Against the honest cross-build
    ///     maximum it is 1.05x (4.875 / 4.6250) -- two ULPs. The 1.5x discipline
    ///     Amendments 6 and 15 applied was computed against a same-build
    ///     statistic that is not what a submission produces.
    ///   * Widening it is an operator decision, not a calibration one, and is
    ///     deliberately NOT taken here: widening an anti-cheat gate is the move
    ///     this contract exists to prevent, and the evidence that would justify it
    ///     is an honest submission actually being rejected.
    ///
    /// The consequence for the gate's discriminating power is recorded rather than
    /// hidden: the cheats of Amendment 19 died at 5.125 and 5.0, and honest
    /// cross-build drift reaches 4.625. Four ULPs separate honest from fabricated,
    /// so the absolute arm no longer distinguishes them by any margin worth
    /// calling a margin. See contract Amendment 20.
    public static let experimentalDFlashWorkBindingLogitToleranceAbsolute = 4.875

    /// Relative arm: the largest admissible `|candidate - reference|` divided by
    /// the larger of the two magnitudes. BOTH arms must hold (Amendment 18).
    ///
    /// Unchanged at 0.25, and deliberately not recalibrated per width. Amendment
    /// 19 proved by additive instrumentation that every fatal comparison in both
    /// surviving cheats breached the ABSOLUTE arm only -- zero comparisons
    /// breached both -- so the absolute arm is the load-bearing half and the
    /// relative arm's job is to bind at small magnitudes where the absolute arm is
    /// meaningless. Measured honest relative maxima are 0.1362 same-build at the
    /// ranked widths, 0.1915 same-build at width 8, and 0.1953 cross-build at
    /// width 1, all inside 0.25.
    ///
    /// A per-width relative calibration is not attempted because the published
    /// calibration output carries `|delta|` but not the magnitude it was divided
    /// by, so the per-width relative distribution cannot be reconstructed from a
    /// calibration run. Tightening it would need that field added first.
    public static let experimentalDFlashWorkBindingLogitToleranceRelative = 0.25
    // Seed length used ONLY to warm block-decode kernel shapes before the
    // protocol hello. Deliberately far below Laguna's 512-position sliding
    // window: a warmup seeded AT the window size plus a widest-block verify
    // pushes the rejected rows past the rotating cache's wrap seam, after which
    // rollback cannot trim them and the round fails with `untrimmableCache`.
    // That seam is a genuine hazard for the scored window and is the subject of
    // the contract's wrap-seam leg (layer L4); a shape warmup must not be what
    // discovers it, and must not fail startup because of it.
    /// Warmup seed length for DFlash block decode: one full sliding window plus
    /// one widest block, so the untimed warm compiles the SATURATED ring shapes
    /// the scored 512-token seed reaches on its first round. Anything shorter
    /// leaves those to compile inside the timed window.
    /// 512 is Laguna's sliding window (`LagunaConstants.slidingWindow`, which
    /// lives in MLXFastModel and so cannot be referenced from this module).
    public static let experimentalDFlashWarmupSeedTokens =
        512 + experimentalDFlashMaxBlockSize
}
