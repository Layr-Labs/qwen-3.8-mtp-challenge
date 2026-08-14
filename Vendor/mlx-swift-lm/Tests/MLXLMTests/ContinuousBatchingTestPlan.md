# Continuous Batching — Test Plan

Covers all changes introduced in the `omlx-continuous-batching` PR.
Each section notes the corresponding omlx test file where the case
originates, so coverage gaps are traceable to the upstream reference.

---

## Existing coverage

`ContinuousBatchingTests.swift` already covers the low-level primitives.
These must remain green and are not duplicated below.

| Test | What it verifies |
|------|-----------------|
| `testBatchKVCacheMergeExtendFilterAndExtract` | BatchKVCache merge / extend / filter / per-row extract |
| `testBatchRotatingKVCacheKeepsSlidingWindowRows` | Sliding-window eviction and per-row extract |
| `testArraysCachePreservesBatchMetadataThroughFilterAndExtend` | ArraysCache mask generation after filter+extend |
| `testArraysCacheAdvancesLengthsForChunkedPrefill` | ArraysCache length tracking for chunked prefill |
| `testSequenceStateMachineMatchesMultiTokenStopsAndTransitions` | Multi-token stop sequence + state transition |
| `testSequenceStateMachineMatchesOverlappingStopSequence` | Overlapping prefix in stop sequence |
| `testRowSamplerTopKOneAlwaysSelectsBestToken` | `makeRowSampler` with `topK=1` |
| `testBatchGeneratorAdmitsQueuedRowsAndReportsFinishReasons` | Full BatchGenerator lifecycle, stop vs length finish |
| `testBatchGeneratorCancelRemovesQueuedRequest` | Cancel before prefill |
| `testBatchGeneratorCancelRemovesActiveRequest` | Cancel during decode |

---

## New tests required

### 1. `RequestStatus` and `SamplingParams` — `Request.swift`
*omlx ref: `test_request.py :: TestRequestStatus`, `TestSamplingParams`*

**`testRequestStatusFinishedPredicateAndFinishReasons`**
- `.waiting`, `.running`, `.preempted` → `isFinished == false`.
- `.finishedStopped` → `isFinished == true`, `finishReason == "stop"`.
- `.finishedLengthCapped` → `isFinished == true`, `finishReason == "length"`.
- `.finishedAborted` → `isFinished == true`, `finishReason == "abort"`.
- Ordering: `waiting < finishedStopped`.

**`testSamplingParamsDefaults`**
- `SamplingParams()` has `maxTokens == 256`, `temperature == 0.7`, `topP == 0.9`,
  `topK == 0`, `minP == 0.0`, `repetitionPenalty == 1.0`, `presencePenalty == 0.0`,
  `frequencyPenalty == 0.0`, `stop == []`, `stopTokenIds == []`.

**`testRequestAppendOutputToken`**
- `appendOutputToken` increments `outputTokenIds` and `numComputedTokens`.

**`testRequestSetFinished`**
- `setFinished(.finishedStopped)` sets `status` and `finishReason == "stop"`.
- `setFinished(.finishedAborted, reason: "user_cancelled")` uses the override reason.

---

### 2. `RequestOutputCollector` — `OutputCollector.swift`
*omlx ref: indirectly exercised throughout `test_engine_core.py`*

**`testCollectorGetNowaitReturnsPutOutput`**
- `put(output)` then `getNowait()` returns `output`; second `getNowait()` returns nil.

**`testCollectorAggregationMergesTokenLists`**
- `put(a)` then `put(b)` with `aggregate: true`; `getNowait()` returns merged
  `newTokenIds == a.newTokenIds + b.newTokenIds`.

**`testCollectorGetBlocksUntilPut`**
- Spawn a Task calling `await get()`; verify it suspends; `put(output)` resumes
  it with the correct value.

**`testCollectorNoDuplicateDeliveryAfterContinuationResume`**
*Regression — bug fixed in commit 4*
- `await get()` is waiting; `put(output)` resumes it; immediately call
  `getNowait()` → must return nil (output delivered directly, not buffered).

**`testCollectorClearDiscardsPendingOutput`**
- `put(output)`, `clear()`, `getNowait()` returns nil.

---

### 3. Sampling — `Sampling.swift`
*omlx ref: `test_request.py :: TestSamplingParams.test_greedy_sampling`*

**`testApplyTopPFiltersLowProbTokens`**
- One dominant token; `applyTopP(..., topP: 0.9)` masks all but the top token.

**`testApplyMinPFiltersTokensBelowThreshold`**
- `applyMinP(..., minP: 0.5)` keeps only tokens with prob ≥ 0.5 × max.

**`testTemperatureZeroIsGreedy`**
- `makeRowSampler(temperature: 0)` always returns `argMax`.

---

### 4. Repetition penalty — `Scheduler.swift`

**`testRepetitionSamplerReducesPositiveLogit`**
- History `[tok]`; positive logit; `repetitionPenalty: 2.0` → logit halved.

**`testRepetitionSamplerAmplifiesNegativeLogit`**
- History `[tok]`; negative logit; `repetitionPenalty: 2.0` → logit doubled (more negative).

**`testPresencePenaltyAppliesFlatDecrease`**
- History `[tok]`; `presencePenalty: 1.0` → logit decreases by exactly 1.0.

**`testFrequencyPenaltyScalesWithCount`**
- History `[tok, tok, tok]`; `frequencyPenalty: 0.5` → logit decreases by 1.5.

**`testRepetitionSamplerPassesThroughWithNoHistory`**
- Empty history → output logits equal input logits.

---

### 5. Block hash — `PrefixCache.swift`
*omlx ref: `test_prefix_cache.py :: compute_block_hash` usage*

**`testComputeBlockHashIsDeterministic`**
- Same `(parentHash, tokenIds, modelName)` → same hash.

**`testComputeBlockHashDiffersOnDifferentTokens`**
- `[1, 2]` vs `[1, 3]` → different hashes.

**`testComputeBlockHashDiffersOnDifferentParent`**
- Same tokens, different parent → different hash.

**`testComputeBlockHashMatchesPythonRepr`**
- Token repr must use `"(1, 2, 3)"` format (comma-space, parenthesised) to
  stay cross-compatible with omlx SSD files.

---

### 6. `PrefixCache` — `PrefixCache.swift`
*omlx ref: `test_prefix_cache.py :: TestBlockAwarePrefixCache`*

**`testPrefixCacheMissForShortPrompt`**
- Prompt shorter than `blockSize` → `(nil, tokens)`.

**`testPrefixCacheRoundTrip`**
- `storePrefix` then `fetchPrefix` returns caches covering the stored prefix.

**`testPrefixCacheHitSavesTokens`**
- After a hit, `tokensSaved` increases by cached-token count.

**`testPrefixCacheEvictsLRUBlockWhenFull`**
- `maxBlocks: 2`; store A, B; access A; store C → B evicted (LRU).

**`testPrefixCacheReleaseDecrementsRefCount`**
- `fetchPrefix` increments refCount; `releaseRequest` decrements it; block
  becomes evictable.

**`testPrefixCacheSharedPrefixHitsBothRequests`**
- Two requests with identical prefix; second gets a cache hit after first completes.

---

### 7. `Scheduler` — end-to-end with `IncrementingLanguageModel`

**`testSchedulerSingleRequestStopsOnEOS`**
- `eosTokenIds: [5]`; run to completion; `finishReason == "stop"`;
  EOS token absent from `outputTokenIds`.

**`testSchedulerLengthFinishIncludesFinalToken`**
*Regression — bug fixed in commit 4*
- `maxTokens: 3`, no EOS; `outputTokenIds.count == 3`.
  Without the fix the count was 2.

**`testSchedulerStopStringHaltsGeneration`**
- `samplingParams.stop` containing a token sequence; generation stops exactly there.

**`testSchedulerConcurrentRequestsBatchCorrectly`**
- Two requests added; after one step `getNumRunning() == 2`.

**`testSchedulerAbortRemovesRequest`**
*omlx ref: `test_engine_core.py :: test_abort_request_no_ghost_in_scheduler`*
- Abort one of two active requests; verify it never appears in subsequent
  outputs and scheduler internal state (uid maps) is fully cleaned.

**`testSchedulerRepetitionPenaltyReducesRepetition`**
- High `repetitionPenalty`; verify the repeated token is suppressed.

---

### 8. `EngineCore` lifecycle — `EngineCore.swift`
*omlx ref: `test_engine_core.py :: TestEngineCoreStartStop`*

**`testEngineCoreStartSetsRunning`**
- After `start()`, `isRunning == true` and internal task is non-nil.

**`testEngineCoreStopClearsRunning`**
- After `start()` + `stop()`, `isRunning == false`.

**`testEngineCoreDoubleStartIsNoop`**
*omlx ref: `test_double_start_noop`*
- Calling `start()` twice; the engine loop Task is not replaced.

**`testEngineCoreStatsInitialValues`**
*omlx ref: `test_engine_core.py :: TestEngineCoreGetStats`*
- After start, `getStats()` has `running: true`, `stepsExecuted: 0`,
  `active_requests: 0`, `num_waiting: 0`, `num_running: 0`.

---

### 9. `EngineCore` request management — `EngineCore.swift`
*omlx ref: `test_engine_core.py :: TestEngineCoreAddRequest`, `TestEngineCoreAbortRequest`*

**`testEngineCoreAddRequestReturnsId`**
- `addRequest` returns the request's `requestId`.

**`testEngineCoreAddRequestCreatesCollector`**
- After `addRequest`, `getStats()["active_requests"] == 1`.

**`testEngineCoreAbortRequestReturnsFalseForUnknownId`**
- `abortRequest("unknown")` returns `false`.

**`testEngineCoreAbortRequestSignalsConsumerWithAbortOutput`**
*omlx ref: `test_abort_request_signals_consumer`*
- `abortRequest` puts an output with `finished: true`, `finishReason: "abort"`,
  `error != nil` into the collector.

**`testEngineCoreAbortRequestWakesBlockedStreamOutputs`**
*omlx ref: `test_abort_request_wakes_blocked_stream_outputs` — critical regression*
- Start a long-running `streamOutputs`; while it is suspended in `await collector.get()`,
  call `abortRequest` from another Task. The stream must complete (not hang)
  and yield an output with `error != nil`.

**`testEngineCoreAbortRequestNoGhostInScheduler`**
*omlx ref: `test_abort_request_no_ghost_in_scheduler` — regression*
- After `abortRequest` completes and the next scheduler step runs, the request
  must not appear in `scheduler.requests`, uid maps, or `genBatch`.

**`testEngineCoreAbortAllRequestsSendsErrorToAll`**
*omlx ref: `test_abort_all_requests`*
- Add two requests; `abortAllRequests()` returns 2 and each collector receives
  an output with `finishReason: "error"`.

**`testEngineCoreAbortAllRequestsEmptyReturnsZero`**
- No active requests → `abortAllRequests()` returns 0.

**`testEngineCoreEngineKeepsRunningAfterAbortAll`**
*omlx ref: `test_abort_all_requests_engine_keeps_running`*
- After `abortAllRequests()`, engine is still running and a new request can
  be added successfully.

---

### 10. `EngineCore` generation — `EngineCore.swift`
*omlx ref: `test_engine_core.py :: TestEngineCoreErrorPropagation`,
`TestEngineCoreGenerateCancellation`*

**`testEngineCoreGenerateReturnsCompleteOutput`**
- `generate(prompt:maxTokens:)` returns `finished: true` with non-empty `outputText`.

**`testEngineCoreStreamOutputsDeliversAllTokens`**
- Stream yields intermediate tokens followed by a final output with `finished: true`.

**`testEngineCoreNoDuplicateTokensInStream`**
*Regression — bug fixed in commit 4*
- Concatenated `newTokenIds` across all stream outputs equals `outputTokenIds`
  in the final output (no duplicates, no gaps).

**`testEngineCoreStreamOutputsRaisesOnErrorOutput`**
*omlx ref: `test_stream_outputs_raises_on_error`*
- Manually put an error `RequestOutput` (with `error != nil`) into the collector;
  `streamOutputs` must propagate via `EngineError.generationFailed`.

**`testEngineCoreGenerateThrowsOnErrorOutput`**
*omlx ref: `test_generate_raises_on_error`*
- Inject an error output into the collector for an in-flight `generate()` call;
  verify it throws `EngineError.generationFailed`.

**`testEngineCoreGenerateCancellationAbortsRequest`**
*omlx ref: `test_generate_cancel_aborts_request`*
- Swift Task wrapping `generate(...)` is cancelled; the request must be removed
  from the active set (collector cleaned up, `active_requests` returns to zero).

**`testEngineCoreGenerateCancelOneDoesNotAffectOther`**
*omlx ref: `test_generate_cancel_multiple_requests`*
- Two concurrent `generate` Tasks; cancel one; the other must still complete
  successfully.

**`testEngineCoreGenerateForwardsTopKAndMinP`**
*Regression — bug fixed in commit 4*
- With `topK: 1` and `temperature > 0`, generation selects the best token
  deterministically (verifies params are forwarded, not silently dropped).

**`testEngineCoreCleanupReleasesCollectorAfterFinish`**
- After a completed stream, `getStats()["active_requests"]` returns to zero.

---

### 11. Thread safety (serial queue) — `EngineCore.swift`
*omlx ref: `test_engine_core.py :: TestGlobalMLXExecutor` —
omlx tests that their `ThreadPoolExecutor(max_workers=1)` serializes all
`scheduler.step()` calls; our `engineQueue` serial queue is the Swift equivalent.*

**`testEngineQueueSerializesSteps`**
- Replace `scheduler.step` with a tracked closure that records concurrent
  execution. Run the engine for 100 ms. Assert `maxConcurrent == 1`
  (the serial queue never executes two steps simultaneously).

**`testConcurrentAddRequestsNoCrash`**
- Add 20 requests simultaneously from separate Tasks; engine processes all
  without crashing, hanging, or reporting a lower `active_requests` count.
  (Swift TSAN will catch dict races if this regresses.)

**`testAddRequestDictSetupOnEngineQueue`**
- Verify that `getStats()["active_requests"]` reflects the new request
  immediately after `await addRequest` returns (the `await` guarantees the
  engine-queue dispatch has completed and the dict entry is visible).

---

### 12. `BatchedEngine` — `BatchedEngine.swift`

**`testBatchedEngineGenerateReturnsString`**
- `generate(prompt:)` returns a non-empty string and does not throw.

**`testBatchedEngineStreamGenerateYieldsChunks`**
- `streamGenerate(prompt:maxTokens:)` yields at least one non-empty chunk.

**`testBatchedEngineChatAppliesTemplate`**
- `chat(messages:)` passes a prompt containing the expected role markers
  from the tokenizer's chat template.

---

### 13. `SSDCacheManager` — `SSDCacheManager.swift`
*omlx ref: `test_paged_ssd_cache.py`*

**`testSSDCacheManagerRoundTrip`**
- `saveBlock` + wait for background write + `loadBlock` → key/value arrays match.

**`testSSDCacheManagerHasBlockAfterSave`**
- `hasBlock(hash:) == true` after save completes.

**`testSSDCacheManagerLRUEviction`**
- `maxSizeBytes` smaller than two blocks; save A then B → `evictions == 1`.

**`testSSDCacheManagerScanExistingOnInit`**
- Write a valid `.safetensors` file at the expected path; new `SSDCacheManager`
  on the same directory → `hasBlock` returns true without saving again.

**`testSSDCacheManagerFileLayout`**
- After saving, file is at `cacheDir/<firstHexChar>/<64hex>.safetensors`.

---

### 14. `GenerationBatch` tensor shape — `GenerationBatch.swift`

**`testGenerationBatchDecodeInputShape`**
- Single-row batch: `step()` exercises `currentTokens.reshaped(1, 1)` and
  returns a valid integer token without shape errors.

**`testGenerationBatchLogitExtractionPerRow`**
- Multi-row batch: sampled tokens match `argMax` of `logits[0..., -1, 0...]`
  per row (verifies the logit-extraction dimension is correct).

---

## Integration tests (require model download)

Located in `IntegrationTesting/`; gate on network/GPU availability.

**`testPrefixCacheHitReducesPrefillTime`**
- Same prompt twice; second run faster and `getStats()["hits"] > 0`.

**`testSSDCachePersistsAcrossRestart`**
- Generate; restart engine with same `SSDCacheConfig`; same prompt again →
  `ssd_hits > 0`.

**`testConcurrentRequestsThroughput`**
- 8 concurrent requests; batched tokens/sec ≥ 1.5× sequential at batch ≥ 4.

**`testStopStringHaltsAtCorrectBoundary`**
- Output does not contain the stop string and halts at the correct token index.

---

## Regression index

| Bug | Commit | Test(s) |
|-----|--------|---------|
| OutputCollector duplicate delivery | 4 | `testCollectorNoDuplicateDeliveryAfterContinuationResume`, `testEngineCoreNoDuplicateTokensInStream` |
| Length-finish drops final token | 4 | `testSchedulerLengthFinishIncludesFinalToken` |
| topK/minP silently dropped | 4 | `testEngineCoreGenerateForwardsTopKAndMinP` |
| EngineCore dict data race | 5 | `testConcurrentAddRequestsNoCrash`, `testEngineQueueSerializesSteps` |
| Abort doesn't wake blocked consumer | 5 | `testEngineCoreAbortRequestWakesBlockedStreamOutputs` |
| Ghost request after abort | 5 | `testEngineCoreAbortRequestNoGhostInScheduler` |
| Semaphore blocks engine queue | 5 | `testEngineCoreGenerateCancellationAbortsRequest` |
