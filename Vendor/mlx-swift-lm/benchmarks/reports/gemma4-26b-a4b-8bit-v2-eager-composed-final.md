# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-8bit/snapshots/d87327f1c28d03b74ef795156059e59b8290fb3e |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 9.5 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T10:51:07Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| v2 | 1 | 500 | 78.8 | 62.7 | 431 | 12.7 |
| v2 | 2 | 100/1500 | 47.1 | 66.5 | 1033 | 17.8 |
| v2 | 4 | 100/500/1500/500 | 32.4 | 90.1 | 1401 | 25.9 |

Per-request detail:
  v2 B=1:
    req 0: prompt=500 tokens=128 ttft=431ms decodeTPS=78.8 finish=length
  v2 B=2:
    req 0: prompt=100 tokens=128 ttft=565ms decodeTPS=40.2 finish=length
    req 1: prompt=1500 tokens=128 ttft=1501ms decodeTPS=54.1 finish=length
  v2 B=4:
    req 0: prompt=100 tokens=128 ttft=1401ms decodeTPS=30.6 finish=length
    req 1: prompt=500 tokens=128 ttft=1401ms decodeTPS=30.6 finish=length
    req 2: prompt=1500 tokens=128 ttft=2322ms decodeTPS=37.8 finish=length
    req 3: prompt=500 tokens=128 ttft=1401ms decodeTPS=30.6 finish=length
