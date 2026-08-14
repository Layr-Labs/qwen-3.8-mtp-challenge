# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-qat-4bit/snapshots/0e3cbab38ce568cf6e23543010d08d03b731910c |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 11.0 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T19:12:55Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Correctness (v2 contiguous, greedy)

prompt tokens: target=24 short=19 long=1543 eos=[1, 50, 106]
[b1-sanity] finish=stop tokens=46 loop=false
[b1-sanity] text: The sky appears blue because sunlight is scattered by the gases and particles in Earth's atmosphere. Blue light travels in shorter, smaller waves and is scattered more strongly than other colors, making the sky look blue from the ground.
[invariance] solo tokens=46 finish=stop
[invariance] solo-repeat divergence=nil
[invariance] solo text: The sky appears blue because sunlight is scattered by the gases and particles in Earth's atmosphere. Blue light travels in shorter, smaller waves and is scattered more strongly than other colors, making the sky look blue from the ground.
[invariance] burst target tokens=46 divergence=nil
[invariance] mid-join: neighbors had (8, 5) tokens at join; target tokens=46 divergence=nil
[invariance] neighbor-invariance (burst vs mid-join at B=3): divergence=nil
[chunked-prefill] chunked=32 tok unchunked=32 tok divergence=nil
[chunked-prefill] chunked text: A quick brown fox jumps over a lazy dog while a seasoned cartographer meticulously annotates ancient maps with details regarding tides, trade winds, and continental drift.
[compiled] solo divergence=nil compiledSteps=46 fallbacks=[:] warmup: b1=0.05s b2=0.03s b4=0.05s
[compiled] text: The sky appears blue because sunlight is scattered by the gases and particles in Earth's atmosphere. Blue light travels in shorter, smaller waves and is scattered more strongly than other colors, making the sky look blue from the ground.
[compiled] burst-vs-solo divergence=nil compiledSteps=46 fallbacks=[:]

- **b1-sanity**: PASS — finish=stop completion=46 repetitionLoop=false
- **invariance**: PASS — solo=46 tok; solo-repeat deterministic; neighbor-invariant at B=3 (burst == mid-join); burst identical to solo; mid-join identical to solo
- **chunked-prefill**: PASS — token-exact over 32 tokens (prompt=1543)
- **compiled-parity**: PASS — compiledSteps=46; token-exact vs eager over 46 tokens
- **compiled-invariance**: PASS — compiled burst identical to compiled solo (compiledSteps=46)
