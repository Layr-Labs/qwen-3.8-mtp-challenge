# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-8bit/snapshots/d87327f1c28d03b74ef795156059e59b8290fb3e |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Date | 2026-07-02T05:52:51Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| legacy-compiled | 1 | 500 | 51.2 | 43.1 | 488 | 14.2 |
| legacy-compiled | 2 | 100/1500 | 39.2 | 34.9 | 4090 | 17.3 |
| legacy-compiled | 4 | 100/500/1500/500 | 28.3 | 45.3 | 6813 | 28.5 |

Per-request detail:
  legacy-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=488ms decodeTPS=51.2 finish=length
  legacy-compiled B=2:
    req 0: prompt=100 tokens=128 ttft=4090ms decodeTPS=39.2 finish=length
    req 1: prompt=1500 tokens=128 ttft=4090ms decodeTPS=39.2 finish=length
  legacy-compiled B=4:
    req 0: prompt=100 tokens=128 ttft=6813ms decodeTPS=28.3 finish=length
    req 1: prompt=500 tokens=128 ttft=6813ms decodeTPS=28.3 finish=length
    req 2: prompt=1500 tokens=128 ttft=6813ms decodeTPS=28.3 finish=length
    req 3: prompt=500 tokens=128 ttft=6813ms decodeTPS=28.3 finish=length
