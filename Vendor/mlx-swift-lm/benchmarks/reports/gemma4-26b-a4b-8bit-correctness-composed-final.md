# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-8bit/snapshots/d87327f1c28d03b74ef795156059e59b8290fb3e |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 7.3 / 16 cores; no darkbloom process |
| Date | 2026-07-02T10:58:03Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Correctness (v2 contiguous, greedy)

prompt tokens: target=24 short=19 long=1543 eos=[1, 50, 106]
[b1-sanity] finish=stop tokens=52 loop=false
[b1-sanity] text: The sky appears blue because sunlight reaches Earth's atmosphere and is scattered in all directions by the gases and particles in the air. Blue light travels in shorter, smaller waves and is scattered more strongly than other colors, making it more visible to our eyes.
[invariance] solo tokens=52 finish=stop
[invariance] solo-repeat divergence=nil
[invariance] solo text: The sky appears blue because sunlight reaches Earth's atmosphere and is scattered in all directions by the gases and particles in the air. Blue light travels in shorter, smaller waves and is scattered more strongly than other colors, making it more visible to our eyes.
[invariance] burst target tokens=52 divergence=nil
[invariance] mid-join: neighbors had (9, 5) tokens at join; target tokens=52 divergence=nil
[invariance] neighbor-invariance (burst vs mid-join at B=3): divergence=nil
[chunked-prefill] chunked=28 tok unchunked=28 tok divergence=nil
[chunked-prefill] chunked text: A quick brown fox jumps over a lazy dog while a seasoned cartographer meticulously annotates ancient maps with details regarding maritime and geological phenomena.
[compiled] solo divergence=nil compiledSteps=52 fallbacks=[:] warmup: b1=0.04s b2=0.03s b4=0.05s
[compiled] text: The sky appears blue because sunlight reaches Earth's atmosphere and is scattered in all directions by the gases and particles in the air. Blue light travels in shorter, smaller waves and is scattered more strongly than other colors, making it more visible to our eyes.
[compiled] burst-vs-solo divergence=nil compiledSteps=52 fallbacks=[:]

- **b1-sanity**: PASS — finish=stop completion=52 repetitionLoop=false
- **invariance**: PASS — solo=52 tok; solo-repeat deterministic; neighbor-invariant at B=3 (burst == mid-join); burst identical to solo; mid-join identical to solo
- **chunked-prefill**: PASS — token-exact over 28 tokens (prompt=1543)
- **compiled-parity**: PASS — compiledSteps=52; token-exact vs eager over 52 tokens
- **compiled-invariance**: PASS — compiled burst identical to compiled solo (compiledSteps=52)
