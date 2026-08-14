# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-qat-4bit/snapshots/0e3cbab38ce568cf6e23543010d08d03b731910c |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 8.6 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T17:15:47Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| legacy-compiled | 1 | 500 | 107.3 | 81.5 | 386 | 9.3 |
| legacy-compiled | 2 | 100/1500 | 83.0 | 71.9 | 2032 | 12.0 |
| legacy-compiled | 4 | 100/500/1500/500 | 53.8 | 82.4 | 3853 | 18.5 |

Per-request detail:
  legacy-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=386ms decodeTPS=107.3 finish=length
  legacy-compiled B=2:
    req 0: prompt=100 tokens=128 ttft=2032ms decodeTPS=83.0 finish=length
    req 1: prompt=1500 tokens=128 ttft=2032ms decodeTPS=83.0 finish=length
  legacy-compiled B=4:
    req 0: prompt=100 tokens=128 ttft=3853ms decodeTPS=53.8 finish=length
    req 1: prompt=500 tokens=128 ttft=3853ms decodeTPS=53.8 finish=length
    req 2: prompt=1500 tokens=128 ttft=3853ms decodeTPS=53.8 finish=length
    req 3: prompt=500 tokens=128 ttft=3853ms decodeTPS=53.8 finish=length
