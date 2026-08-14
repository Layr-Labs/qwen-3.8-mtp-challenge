# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-8bit/snapshots/1382fb7268fa62c07b83ffa174ff6fa455fb0686 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Date | 2026-07-02T07:40:19Z |

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

- **b1-sanity**: PASS — finish=stop completion=52 repetitionLoop=false
- **invariance**: PASS — solo=52 tok; solo-repeat deterministic; neighbor-invariant at B=3 (burst == mid-join); burst identical to solo; mid-join identical to solo
- **chunked-prefill**: PASS — token-exact over 28 tokens (prompt=1543)

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| v2 | 1 | 500 | 67.2 | 50.6 | 638 | 14.6 |
| v2 | 2 | 100/1500 | 40.1 | 45.0 | 2479 | 21.0 |
| v2 | 4 | 100/500/1500/500 | 21.5 | 56.3 | 2675 | 40.5 |
| v2-paged | 1 | 500 | 66.8 | 51.2 | 597 | 14.8 |
| v2-paged | 2 | 100/1500 | 31.9 | 43.3 | 1524 | 19.8 |
| v2-paged | 4 | 100/500/1500/500 | 22.0 | 59.3 | 2327 | 39.2 |

Per-request detail:
  v2 B=1:
    req 0: prompt=500 tokens=128 ttft=638ms decodeTPS=67.2 finish=length
  v2 B=2:
    req 0: prompt=100 tokens=128 ttft=2125ms decodeTPS=35.7 finish=length
    req 1: prompt=1500 tokens=128 ttft=2833ms decodeTPS=44.4 finish=length
  v2 B=4:
    req 0: prompt=100 tokens=128 ttft=2675ms decodeTPS=20.2 finish=length
    req 1: prompt=500 tokens=128 ttft=2675ms decodeTPS=20.2 finish=length
    req 2: prompt=1500 tokens=128 ttft=4084ms decodeTPS=25.3 finish=length
    req 3: prompt=500 tokens=128 ttft=2675ms decodeTPS=20.2 finish=length
  v2-paged B=1:
    req 0: prompt=500 tokens=128 ttft=597ms decodeTPS=66.8 finish=length
  v2-paged B=2:
    req 0: prompt=100 tokens=128 ttft=303ms decodeTPS=23.8 finish=length
    req 1: prompt=1500 tokens=128 ttft=2745ms decodeTPS=40.0 finish=length
  v2-paged B=4:
    req 0: prompt=100 tokens=128 ttft=2327ms decodeTPS=20.7 finish=length
    req 1: prompt=500 tokens=128 ttft=2327ms decodeTPS=20.7 finish=length
    req 2: prompt=1500 tokens=128 ttft=3737ms decodeTPS=25.9 finish=length
    req 3: prompt=500 tokens=128 ttft=2327ms decodeTPS=20.7 finish=length
