# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | false |
| Host at start | load avg (1m) 11.9 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T07:08:40Z |

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
[invariance] mid-join: neighbors had (9, 5) tokens at join; target tokens=64 divergence=nil
[invariance] neighbor-invariance (burst vs mid-join at B=3): divergence=nil
[chunked-prefill] chunked=64 tok unchunked=64 tok divergence=nil
[chunked-prefill] chunked text: <|channel|>analysis<|message|>The user asks: "Summarize the following passage in one sentence:" and then repeats the same sentence many times. The passage is basically: "The quick brown fox jumps over the lazy dog while the seasoned cartographer annotates ancient maps with meticulous margin notes about tides, trade winds, and the slow

- **b1-sanity**: PASS — finish=stop completion=200 repetitionLoop=false
- **invariance**: PASS — solo=64 tok; solo-repeat deterministic; neighbor-invariant at B=3 (burst == mid-join); burst identical to solo; mid-join identical to solo
- **chunked-prefill**: PASS — token-exact over 64 tokens (prompt=1600)

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| legacy | 1 | 500 | 77.2 | 60.9 | 457 | 12.9 |
| legacy | 2 | 100/1500 | 58.8 | 53.6 | 2615 | 17.0 |
| legacy | 4 | 100/500/1500/500 | 41.1 | 61.2 | 5272 | 24.3 |
| v2 | 1 | 500 | 96.1 | 68.1 | 558 | 10.4 |
| v2 | 2 | 100/1500 | 53.8 | 70.0 | 1006 | 13.7 |
| v2 | 4 | 100/500/1500/500 | 33.2 | 84.6 | 1842 | 21.3 |
| v2-paged | 1 | 500 | 0.4 | 0.4 | 4599 | 2340.7 |
| v2-paged | 2 | 100/1500 | 0.5 | 0.9 | 11892 | 1890.9 |
| v2-paged | 4 | 100/500/1500/500 | 0.0 | 0.0 | 6920 | 0.0 |

Per-request detail:
  legacy B=1:
    req 0: prompt=500 tokens=128 ttft=457ms decodeTPS=77.2 finish=length
  legacy B=2:
    req 0: prompt=100 tokens=128 ttft=2615ms decodeTPS=58.8 finish=length
    req 1: prompt=1500 tokens=128 ttft=2615ms decodeTPS=58.8 finish=length
  legacy B=4:
    req 0: prompt=100 tokens=128 ttft=5272ms decodeTPS=41.1 finish=length
    req 1: prompt=500 tokens=128 ttft=5272ms decodeTPS=41.1 finish=length
    req 2: prompt=1500 tokens=128 ttft=5272ms decodeTPS=41.1 finish=length
    req 3: prompt=500 tokens=128 ttft=5272ms decodeTPS=41.1 finish=length
  v2 B=1:
    req 0: prompt=500 tokens=128 ttft=558ms decodeTPS=96.1 finish=length
  v2 B=2:
    req 0: prompt=100 tokens=128 ttft=147ms decodeTPS=36.6 finish=length
    req 1: prompt=1500 tokens=128 ttft=1865ms decodeTPS=70.9 finish=length
  v2 B=4:
    req 0: prompt=100 tokens=128 ttft=1842ms decodeTPS=30.4 finish=length
    req 1: prompt=500 tokens=128 ttft=1842ms decodeTPS=30.4 finish=length
    req 2: prompt=1500 tokens=128 ttft=2994ms decodeTPS=41.5 finish=length
    req 3: prompt=500 tokens=128 ttft=1842ms decodeTPS=30.4 finish=length
  v2-paged B=1:
    req 0: prompt=500 tokens=51 ttft=4599ms decodeTPS=0.4 finish=error("request exceeded 120s deadline")
  v2-paged B=2:
    req 0: prompt=100 tokens=57 ttft=5637ms decodeTPS=0.5 finish=error("request exceeded 120s deadline")
    req 1: prompt=1500 tokens=55 ttft=18148ms decodeTPS=0.5 finish=error("request exceeded 120s deadline")
  v2-paged B=4:
    req 0: prompt=0 tokens=1 ttft=13841ms decodeTPS=0.0 finish=error("engine step exceeded 30s watchdog")
    req 1: prompt=0 tokens=0 ttft=0ms decodeTPS=0.0 finish=error("engine step exceeded 30s watchdog")
    req 2: prompt=0 tokens=0 ttft=0ms decodeTPS=0.0 finish=error("engine step exceeded 30s watchdog")
    req 3: prompt=0 tokens=1 ttft=13841ms decodeTPS=0.0 finish=error("engine step exceeded 30s watchdog")
