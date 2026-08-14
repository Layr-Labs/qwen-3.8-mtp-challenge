# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-8bit/snapshots/d87327f1c28d03b74ef795156059e59b8290fb3e |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 7.7 / 16 cores; no darkbloom process |
| Date | 2026-07-02T10:46:29Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| legacy-compiled | 1 | 500 | 77.1 | 62.2 | 411 | 13.0 |
| legacy-compiled | 2 | 100/1500 | 61.1 | 59.4 | 2234 | 16.3 |
| legacy-compiled | 4 | 100/500/1500/500 | 41.2 | 69.8 | 4251 | 23.9 |

Per-request detail:
  legacy-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=411ms decodeTPS=77.1 finish=length
  legacy-compiled B=2:
    req 0: prompt=100 tokens=128 ttft=2234ms decodeTPS=61.1 finish=length
    req 1: prompt=1500 tokens=128 ttft=2234ms decodeTPS=61.1 finish=length
  legacy-compiled B=4:
    req 0: prompt=100 tokens=128 ttft=4251ms decodeTPS=41.2 finish=length
    req 1: prompt=500 tokens=128 ttft=4251ms decodeTPS=41.2 finish=length
    req 2: prompt=1500 tokens=128 ttft=4251ms decodeTPS=41.2 finish=length
    req 3: prompt=500 tokens=128 ttft=4251ms decodeTPS=41.2 finish=length
