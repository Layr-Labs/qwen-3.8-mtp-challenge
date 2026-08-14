# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-8bit/snapshots/d87327f1c28d03b74ef795156059e59b8290fb3e |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 6.5 / 16 cores; no darkbloom process |
| Date | 2026-07-02T10:55:43Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| v2-paged | 1 | 500 | 74.9 | 58.9 | 477 | 13.4 |
| v2-paged | 2 | 100/1500 | 49.6 | 68.0 | 1081 | 16.8 |
| v2-paged | 4 | 100/500/1500/500 | 32.8 | 89.0 | 1487 | 24.9 |

Per-request detail:
  v2-paged B=1:
    req 0: prompt=500 tokens=128 ttft=477ms decodeTPS=74.9 finish=length
  v2-paged B=2:
    req 0: prompt=100 tokens=128 ttft=624ms decodeTPS=42.1 finish=length
    req 1: prompt=1500 tokens=128 ttft=1538ms decodeTPS=57.0 finish=length
  v2-paged B=4:
    req 0: prompt=100 tokens=128 ttft=1487ms decodeTPS=30.8 finish=length
    req 1: prompt=500 tokens=128 ttft=1487ms decodeTPS=30.8 finish=length
    req 2: prompt=1500 tokens=128 ttft=2498ms decodeTPS=39.0 finish=length
    req 3: prompt=500 tokens=128 ttft=1487ms decodeTPS=30.8 finish=length
