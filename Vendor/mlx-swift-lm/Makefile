.PHONY: test test-cb build-server serve loc

# Count lines of code in the Libraries folder.
loc:
	tokei Libraries/

# Show lines added/removed in Libraries/ and Tests/ since this branch diverged from main.
loc-pr:
	@base=$$(git merge-base main HEAD); \
	for dir in Libraries Tests; do \
		git diff --numstat $$base HEAD -- $$dir/ \
		| awk -v d=$$dir 'BEGIN{a=0;r=0} {a+=$$1; r+=$$2} END{printf "+%d / -%d lines in %s/\n", a, r, d}'; \
	done

# Run the full test suite (requires Metal; use xcodebuild so shaders compile).
test:
	xcodebuild test \
		-scheme mlx-swift-lm-Package \
		-destination 'platform=macOS' \
		2>&1 | xcpretty || xcodebuild test \
		-scheme mlx-swift-lm-Package \
		-destination 'platform=macOS'

# Run only the continuous-batching tests.
test-cb:
	xcodebuild test \
		-scheme mlx-swift-lm-Package \
		-destination 'platform=macOS' \
		-only-testing:MLXLMTests/CBRequestStatusTests \
		-only-testing:MLXLMTests/CBOutputCollectorTests \
		-only-testing:MLXLMTests/CBSamplingTests \
		-only-testing:MLXLMTests/CBRepetitionPenaltyTests \
		-only-testing:MLXLMTests/CBBlockHashTests \
		-only-testing:MLXLMTests/CBPrefixCacheTests \
		-only-testing:MLXLMTests/CBSchedulerTests \
		-only-testing:MLXLMTests/CBEngineCoreLifecycleTests \
		-only-testing:MLXLMTests/CBEngineCoreRequestTests \
		-only-testing:MLXLMTests/CBEngineCoreGenerationTests \
		-only-testing:MLXLMTests/CBEngineCoreThreadSafetyTests \
		-only-testing:MLXLMTests/CBBatchedEngineTests \
		-only-testing:MLXLMTests/CBGenerationBatchShapeTests \
		2>&1 | grep -E "Test Case|Test Suite|SUCCEEDED|FAILED|error:"

# Build the inference server (release mode for benchmarking).
build-server:
	swift build -c release --product mlx-server

# Run the server. Specify the model with MODEL=.
# Usage: make serve MODEL=/path/to/model [PORT=8080] [HOST=127.0.0.1]
serve: build-server
	.build/release/mlx-server \
		--model $(MODEL) \
		--port $(or $(PORT),8080) \
		--host $(or $(HOST),127.0.0.1)
