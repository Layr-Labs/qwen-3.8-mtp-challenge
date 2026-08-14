# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 7.4 / 16 cores; no darkbloom process |
| Date | 2026-07-02T10:43:32Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Correctness (v2 contiguous, greedy)

prompt tokens: target=79 short=74 long=1600 eos=[199999, 200002, 200012]
[b1-sanity] finish=stop tokens=200 loop=false
[b1-sanity] text: <|channel|>analysis<|message|>We need to explain why the sky is blue in two sentences. So we need to produce a concise explanation. The sky is blue because Rayleigh scattering of sunlight by air molecules scatters shorter wavelengths (blue) more strongly than longer wavelengths (red). So we can say: The sky appears blue because sunlight is scattered by the Earth's atmosphere, and shorter blue wavelengths are scattered more efficiently than longer red wavelengths, making the sky look blue. That's one sentence. But we need two sentences. So we can split: "Sunlight contains all colors, but when it passes through the atmosphere, the shorter blue wavelengths are scattered in all directions by air molecules. This scattering makes the sky appear blue to our eyes." That is two sentences. That should be fine.<|end|><|start|>assistant<|channel|>final<|message|>Sunlight contains all colors, but as it passes through the atmosphere the shorter blue wavelengths are scattered in every direction by air molecules. This Rayleigh scattering makes the sky appear blue to our eyes.
[invariance] solo tokens=64 finish=length
[invariance] solo-repeat divergence=nil
[invariance] solo text: <|channel|>analysis<|message|>We need to explain why the sky is blue in two sentences. So we need to produce a concise explanation. The sky is blue because Rayleigh scattering of sunlight by air molecules scatters shorter wavelengths (blue) more strongly than longer wavelengths (red). So we can say: The sky appears blue because sunlight
[invariance] burst target tokens=64 divergence=nil
[invariance] mid-join: neighbors had (8, 5) tokens at join; target tokens=64 divergence=nil
[invariance] neighbor-invariance (burst vs mid-join at B=3): divergence=nil
[chunked-prefill] chunked=64 tok unchunked=64 tok divergence=nil
[chunked-prefill] chunked text: <|channel|>analysis<|message|>The user asks: "Summarize the following passage in one sentence:" and then repeats the same sentence many times. The passage is basically: "The quick brown fox jumps over the lazy dog while the seasoned cartographer annotates ancient maps with meticulous margin notes about tides, trade winds, and the slow
[compiled] solo divergence=nil compiledSteps=64 fallbacks=[:] warmup: b1=0.04s b2=0.03s b4=0.04s
[compiled] text: <|channel|>analysis<|message|>We need to explain why the sky is blue in two sentences. So we need to produce a concise explanation. The sky is blue because Rayleigh scattering of sunlight by air molecules scatters shorter wavelengths (blue) more strongly than longer wavelengths (red). So we can say: The sky appears blue because sunlight
[compiled] burst-vs-solo divergence=nil compiledSteps=99 fallbacks=[:]

- **b1-sanity**: PASS — finish=stop completion=200 repetitionLoop=false
- **invariance**: PASS — solo=64 tok; solo-repeat deterministic; neighbor-invariant at B=3 (burst == mid-join); burst identical to solo; mid-join identical to solo
- **chunked-prefill**: PASS — token-exact over 64 tokens (prompt=1600)
- **compiled-parity**: PASS — compiledSteps=64; token-exact vs eager over 64 tokens
- **compiled-invariance**: PASS — compiled burst identical to compiled solo (compiledSteps=99)
