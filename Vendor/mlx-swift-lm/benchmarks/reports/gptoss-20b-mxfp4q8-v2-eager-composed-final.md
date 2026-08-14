# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 8.6 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T10:36:46Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| v2 | 1 | 500 | 104.9 | 78.4 | 423 | 9.5 |
| v2 | 2 | 100/1500 | 64.0 | 83.8 | 973 | 13.0 |
| v2 | 4 | 100/500/1500/500 | 41.4 | 107.9 | 1369 | 19.3 |

Per-request detail:
  v2 B=1:
    req 0: prompt=500 tokens=128 ttft=423ms decodeTPS=104.9 finish=length
  v2 B=2:
    req 0: prompt=100 tokens=128 ttft=545ms decodeTPS=51.1 finish=length
    req 1: prompt=1500 tokens=128 ttft=1402ms decodeTPS=76.8 finish=length
  v2 B=4:
    req 0: prompt=100 tokens=128 ttft=1369ms decodeTPS=37.9 finish=length
    req 1: prompt=500 tokens=128 ttft=1369ms decodeTPS=37.9 finish=length
    req 2: prompt=1500 tokens=128 ttft=2304ms decodeTPS=52.0 finish=length
    req 3: prompt=500 tokens=128 ttft=1369ms decodeTPS=37.9 finish=length
