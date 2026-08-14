# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-8bit/snapshots/d87327f1c28d03b74ef795156059e59b8290fb3e |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | false |
| Host at start | load avg (1m) 9.1 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T10:48:50Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| legacy | 1 | 500 | 62.0 | 52.4 | 393 | 16.0 |
| legacy | 2 | 100/1500 | 48.5 | 53.6 | 2159 | 20.6 |
| legacy | 4 | 100/500/1500/500 | 34.7 | 64.9 | 4235 | 28.8 |

Per-request detail:
  legacy B=1:
    req 0: prompt=500 tokens=128 ttft=393ms decodeTPS=62.0 finish=length
  legacy B=2:
    req 0: prompt=100 tokens=128 ttft=2159ms decodeTPS=48.5 finish=length
    req 1: prompt=1500 tokens=128 ttft=2159ms decodeTPS=48.5 finish=length
  legacy B=4:
    req 0: prompt=100 tokens=128 ttft=4235ms decodeTPS=34.7 finish=length
    req 1: prompt=500 tokens=128 ttft=4235ms decodeTPS=34.7 finish=length
    req 2: prompt=1500 tokens=128 ttft=4235ms decodeTPS=34.7 finish=length
    req 3: prompt=500 tokens=128 ttft=4235ms decodeTPS=34.7 finish=length
