# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 8.8 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T19:22:07Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| v2-paged | 1 | 500 | 93.0 | 69.5 | 477 | 10.7 |
| v2-paged | 4 | 100/500/1500/500 | 39.7 | 106.7 | 1464 | 19.3 |

Per-request detail:
  v2-paged B=1:
    req 0: prompt=500 tokens=128 ttft=477ms decodeTPS=93.0 finish=length
  v2-paged B=4:
    req 0: prompt=100 tokens=128 ttft=1464ms decodeTPS=38.5 finish=length
    req 1: prompt=500 tokens=128 ttft=467ms decodeTPS=29.6 finish=length
    req 2: prompt=1500 tokens=128 ttft=2366ms decodeTPS=52.2 finish=length
    req 3: prompt=500 tokens=128 ttft=1464ms decodeTPS=38.5 finish=length
