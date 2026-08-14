# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-qat-4bit/snapshots/0e3cbab38ce568cf6e23543010d08d03b731910c |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 7.6 / 16 cores; no darkbloom process |
| Date | 2026-07-02T19:17:05Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| v2-paged | 1 | 500 | 93.8 | 70.1 | 473 | 10.6 |
| v2-paged | 2 | 100/1500 | 59.7 | 80.4 | 846 | 12.8 |
| v2-paged | 4 | 100/500/1500/500 | 41.2 | 105.4 | 1443 | 19.6 |

Per-request detail:
  v2-paged B=1:
    req 0: prompt=500 tokens=128 ttft=473ms decodeTPS=93.8 finish=length
  v2-paged B=2:
    req 0: prompt=100 tokens=128 ttft=186ms decodeTPS=43.6 finish=length
    req 1: prompt=1500 tokens=128 ttft=1506ms decodeTPS=75.7 finish=length
  v2-paged B=4:
    req 0: prompt=100 tokens=128 ttft=1443ms decodeTPS=38.1 finish=length
    req 1: prompt=500 tokens=128 ttft=1443ms decodeTPS=38.1 finish=length
    req 2: prompt=1500 tokens=128 ttft=2343ms decodeTPS=50.5 finish=length
    req 3: prompt=500 tokens=128 ttft=1443ms decodeTPS=38.1 finish=length
