# Timed Decode Evaluation Target

The ranked decode measurement uses a private prompt that is independent of the
public correctness fixture, the hidden teacher-forced fixture, and the hidden
GPQA cases. Its stable contract identifier is
`MLXFastConstants.benchmarkEvaluationTargetID`. The workflow downloads the
prompt only after correctness has completed and its hidden material has been
scrubbed, verifies the prompt against the operator-managed SHA-256 and byte
count, and passes it explicitly to the trusted measurement wrapper.

The wrapper generates a checked benchmark oracle from that prompt separately
for the pinned baseline binary and the candidate binary. The oracle supplies:

- the 512-token prefill prompt and its next token;
- the 512-token decode seed and its next token;
- the 128 expected tokens checked during the timed decode loop.

Oracle cache identity must include the binary hash, evaluation target ID, and
prompt SHA-256. A prompt rotation must therefore miss the old cache even when a
binary did not otherwise change.

## Target definition

`lowsim-prose-v1` is an organizer-authored prose target selected by
measurement, not by content class. It was re-tokenized and re-validated on
`m5-bench` for the Laguna XS 2.1 tokenizer (vocab 100352) as part of the
completed model re-pin. The public contract is its shape and its
gate: original text whose first 512 target-tokenizer tokens form the seed, and
whose 129-token greedy continuation (the seed next-token plus the 128 checked
decode tokens) passes the self-similarity metric below on the ranked M5
hardware. The specific text, its subject matter, and its genre are private,
exactly like the hidden correctness prompts; publishing the class would only
help submissions specialize against it.

This remains a representative text-generation workload: a normal 512-token
prefill followed by a 129-token greedy continuation. Authoring guidance for
future rotations (which content shapes keep a greedy continuation diverse
instead of collapsing into repetition) lives in the private operator RUNBOOK.
Expectations are not a substitute for measurement: the actual greedy
continuation generated on the ranked M5 must pass the metric below before a
target is activated, and candidates that fail are discarded, not tuned around.

## Self-similarity metric

For every continuation token and each order in `{1, 2, 3}`, take the immediately
preceding `k` tokens. Look for earlier occurrences of that suffix whose
following token was already present in the request context. The analyzer
reports:

- recurrence rate for each order;
- hit rate when the most recent matching occurrence supplies the draft token;
- an optimistic hit rate when any matching occurrence had the actual follower;
- a practical aggregate that chooses the longest recurrent suffix and drafts
  the most recent occurrence's follower.

The aggregate never reads future continuation tokens. It approximates a
longest-suffix prompt-lookup candidate generator while the optimistic fields
show how much ambiguity the deterministic tie-break leaves on the table.

Score a generated base golden or assembled benchmark oracle without loading a
model:

```bash
.build/release/mlxfast-swift analyze-ngram-similarity \
  --golden /path/to/generated-timed-golden.json \
  --orders 1,2,3 \
  --max-hit-rate 0.03
```

For a base golden, the command scores the first 129 expected tokens. For an
assembled benchmark oracle, it scores
`expected_decode_seed_token + expected_decode_tokens[0..<128]` against the
512-token decode seed.

The activation threshold is a longest-match, most-recent-follower hit rate of
at most `0.03`. Three percent caps an idealized zero-overhead single-token
reuse benefit near 1.03x; lookup and target-verification overhead should reduce
the realized gain toward 1.0x. Treat the per-order and optimistic rates as
diagnostics and reject a target with a conspicuous repeated run even if the
aggregate narrowly passes.

For comparison, the checked-in longcopy fixture currently scores 59 hits in
129 positions (`0.45736`) under the same aggregate metric. The new threshold is
therefore more than an order of magnitude lower than the repetitive workload it
replaces.

This figure is **tokenizer-dependent** — it is a property of how the fixture's
text tokenizes, not of the text alone, so it moves when the target tokenizer
changes. `0.45736` is the value measured on this branch under the Qwen 3.6
tokenizer (`analyze-ngram-similarity --orders 1,2,3` over
`correctness_prompts/public_longcopy_gate_english_512_{256,1024}.json`, which
agree). The previously documented `37/129` (`0.2868`) was the serial-era
tokenizer's value for the same fixture; re-measure rather than carrying either
number forward to a new target.

## Correctness separation

Changing the timed prompt must not change:

- `correctness_prompts/public_longcopy_gate_english_512.txt` or either checked-in
  public golden;
- the hidden teacher-forced correctness object or its SHA/byte pins;
- the hidden GPQA reference object, exact-token checks, semantic judge, or TTFT
  gate;
- `correctnessSteps`, behavior budgets, or QA thresholds.

Only the timed prompt object, self-generated timed oracle, prompt-target
metadata, and baseline timing calibration move when this target rotates.
