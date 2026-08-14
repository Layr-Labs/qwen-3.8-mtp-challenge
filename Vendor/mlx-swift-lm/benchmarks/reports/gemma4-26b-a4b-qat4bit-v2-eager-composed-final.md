# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-qat-4bit/snapshots/0e3cbab38ce568cf6e23543010d08d03b731910c |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 6.4 / 16 cores; no darkbloom process |
| Date | 2026-07-02T19:15:20Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| v2 | 1 | 500 | 98.3 | 73.7 | 444 | 10.1 |
| v2 | 2 | 100/1500 | 55.1 | 76.5 | 831 | 14.0 |
| v2 | 4 | 100/500/1500/500 | 39.1 | 102.4 | 1407 | 20.7 |

Per-request detail:
  v2 B=1:
    req 0: prompt=500 tokens=128 ttft=444ms decodeTPS=98.3 finish=length
  v2 B=2:
    req 0: prompt=100 tokens=128 ttft=144ms decodeTPS=40.8 finish=length
    req 1: prompt=1500 tokens=128 ttft=1518ms decodeTPS=69.4 finish=length
  v2 B=4:
    req 0: prompt=100 tokens=128 ttft=1407ms decodeTPS=36.2 finish=length
    req 1: prompt=500 tokens=128 ttft=1407ms decodeTPS=36.2 finish=length
    req 2: prompt=1500 tokens=128 ttft=2333ms decodeTPS=47.6 finish=length
    req 3: prompt=500 tokens=128 ttft=1407ms decodeTPS=36.2 finish=length
