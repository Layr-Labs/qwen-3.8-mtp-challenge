# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 8.8 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T19:18:49Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| v2 | 1 | 500 | 101.3 | 76.1 | 428 | 9.7 |
| v2 | 4 | 100/500/1500/500 | 38.4 | 106.6 | 1398 | 19.8 |

Per-request detail:
  v2 B=1:
    req 0: prompt=500 tokens=128 ttft=428ms decodeTPS=101.3 finish=length
  v2 B=4:
    req 0: prompt=100 tokens=128 ttft=128ms decodeTPS=27.4 finish=length
    req 1: prompt=500 tokens=128 ttft=1398ms decodeTPS=37.6 finish=length
    req 2: prompt=1500 tokens=128 ttft=2315ms decodeTPS=51.1 finish=length
    req 3: prompt=500 tokens=128 ttft=1398ms decodeTPS=37.6 finish=length
