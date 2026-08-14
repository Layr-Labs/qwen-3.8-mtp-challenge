# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-8bit/snapshots/d87327f1c28d03b74ef795156059e59b8290fb3e |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 7.5 / 16 cores; no darkbloom process |
| Date | 2026-07-02T10:53:25Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
    [v2-compiled B=1] warmup 0.57s (b1=0.28s)
    [v2-compiled B=1] compiledSteps=10 rebinds=3 scratchResets=0 fallbacks=[:]
| v2-compiled | 1 | 500 | 77.4 | 61.6 | 435 | 12.9 |
| v2-compiled | 2 | 100/1500 | 48.4 | 68.0 | 1023 | 17.3 |
| v2-compiled | 4 | 100/500/1500/500 | 31.6 | 87.5 | 1425 | 26.0 |

Per-request detail:
    [v2-compiled B=1] warmup 0.11s (b1=0.03s)
    [v2-compiled B=1] compiledSteps=130 rebinds=3 scratchResets=0 fallbacks=[:]
  v2-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=435ms decodeTPS=77.4 finish=length
    [v2-compiled B=2] warmup 0.25s (b1=0.03s b2=0.14s)
    [v2-compiled B=2] compiledSteps=132 rebinds=8 scratchResets=0 fallbacks=[:]
  v2-compiled B=2:
    req 0: prompt=100 tokens=128 ttft=567ms decodeTPS=41.2 finish=length
    req 1: prompt=1500 tokens=128 ttft=1480ms decodeTPS=55.6 finish=length
    [v2-compiled B=4] warmup 0.19s (b1=0.03s b2=0.03s b4=0.05s)
    [v2-compiled B=4] compiledSteps=132 rebinds=14 scratchResets=0 fallbacks=[:]
  v2-compiled B=4:
    req 0: prompt=100 tokens=128 ttft=1425ms decodeTPS=29.6 finish=length
    req 1: prompt=500 tokens=128 ttft=1425ms decodeTPS=29.6 finish=length
    req 2: prompt=1500 tokens=128 ttft=2463ms decodeTPS=37.5 finish=length
    req 3: prompt=500 tokens=128 ttft=1425ms decodeTPS=29.6 finish=length
