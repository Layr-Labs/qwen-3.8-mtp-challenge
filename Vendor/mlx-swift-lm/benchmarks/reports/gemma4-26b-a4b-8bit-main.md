# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-8bit/snapshots/d87327f1c28d03b74ef795156059e59b8290fb3e |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | false |
| Date | 2026-07-02T05:52:00Z |

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
[invariance] mid-join: neighbors had (8, 5) tokens at join; target tokens=52 divergence=nil
[invariance] neighbor-invariance (burst vs mid-join at B=3): divergence=nil
[chunked-prefill] chunked=28 tok unchunked=28 tok divergence=nil
[chunked-prefill] chunked text: A quick brown fox jumps over a lazy dog while a seasoned cartographer meticulously annotates ancient maps with details regarding maritime and geological phenomena.

- **b1-sanity**: PASS — finish=stop completion=52 repetitionLoop=false
- **invariance**: PASS — solo=52 tok; solo-repeat deterministic; neighbor-invariant at B=3 (burst == mid-join); burst identical to solo; mid-join identical to solo
- **chunked-prefill**: PASS — token-exact over 28 tokens (prompt=1543)

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| legacy | 1 | 500 | 43.6 | 38.1 | 449 | 18.1 |
| legacy | 2 | 100/1500 | 34.5 | 36.4 | 3342 | 24.9 |
| legacy | 4 | 100/500/1500/500 | 25.6 | 44.7 | 6487 | 33.7 |
| v2 | 1 | 500 | 44.0 | 37.5 | 523 | 16.8 |
| v2 | 2 | 100/1500 | 28.8 | 42.3 | 1442 | 23.2 |
| v2 | 4 | 100/500/1500/500 | 18.6 | 54.3 | 1981 | 35.5 |
| v2-paged | - | - | skipped: backendIneligible(reason: "layer 5 (headDim 512, GQA 8) needs 32832 B threadgroup memory at NSG=1 (> 32768); the paged part kernel would trap at dispatch on this hardware") | | | |

Per-request detail:
  legacy B=1:
    req 0: prompt=500 tokens=128 ttft=449ms decodeTPS=43.6 finish=length
  legacy B=2:
    req 0: prompt=100 tokens=128 ttft=3342ms decodeTPS=34.5 finish=length
    req 1: prompt=1500 tokens=128 ttft=3342ms decodeTPS=34.5 finish=length
  legacy B=4:
    req 0: prompt=100 tokens=128 ttft=6487ms decodeTPS=25.6 finish=length
    req 1: prompt=500 tokens=128 ttft=6487ms decodeTPS=25.6 finish=length
    req 2: prompt=1500 tokens=128 ttft=6487ms decodeTPS=25.6 finish=length
    req 3: prompt=500 tokens=128 ttft=6487ms decodeTPS=25.6 finish=length
  v2 B=1:
    req 0: prompt=500 tokens=128 ttft=523ms decodeTPS=44.0 finish=length
  v2 B=2:
    req 0: prompt=100 tokens=128 ttft=736ms decodeTPS=25.0 finish=length
    req 1: prompt=1500 tokens=128 ttft=2148ms decodeTPS=32.5 finish=length
  v2 B=4:
    req 0: prompt=100 tokens=128 ttft=1981ms decodeTPS=17.9 finish=length
    req 1: prompt=500 tokens=128 ttft=1981ms decodeTPS=17.9 finish=length
    req 2: prompt=1500 tokens=128 ttft=3349ms decodeTPS=20.9 finish=length
    req 3: prompt=500 tokens=128 ttft=1981ms decodeTPS=17.9 finish=length
