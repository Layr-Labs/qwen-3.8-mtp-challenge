# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 7.8 / 16 cores; no darkbloom process |
| Date | 2026-07-02T10:41:11Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| v2-paged | 1 | 500 | 96.5 | 71.6 | 472 | 10.4 |
| v2-paged | 2 | 100/1500 | 60.9 | 82.4 | 827 | 12.9 |
| v2-paged | 4 | 100/500/1500/500 | 42.8 | 109.1 | 1429 | 18.8 |

Per-request detail:
  v2-paged B=1:
    req 0: prompt=500 tokens=128 ttft=472ms decodeTPS=96.5 finish=length
  v2-paged B=2:
    req 0: prompt=100 tokens=128 ttft=173ms decodeTPS=43.8 finish=length
    req 1: prompt=1500 tokens=128 ttft=1481ms decodeTPS=78.0 finish=length
  v2-paged B=4:
    req 0: prompt=100 tokens=128 ttft=1429ms decodeTPS=39.2 finish=length
    req 1: prompt=500 tokens=128 ttft=1429ms decodeTPS=39.2 finish=length
    req 2: prompt=1500 tokens=128 ttft=2322ms decodeTPS=53.5 finish=length
    req 3: prompt=500 tokens=128 ttft=1429ms decodeTPS=39.2 finish=length
