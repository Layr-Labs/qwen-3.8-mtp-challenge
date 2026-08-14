# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | false |
| Host at start | load avg (1m) 9.9 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T10:34:34Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| legacy | 1 | 500 | 76.8 | 62.6 | 390 | 13.0 |
| legacy | 2 | 100/1500 | 59.8 | 60.6 | 2099 | 16.7 |
| legacy | 4 | 100/500/1500/500 | 43.2 | 71.1 | 4259 | 23.1 |

Per-request detail:
  legacy B=1:
    req 0: prompt=500 tokens=128 ttft=390ms decodeTPS=76.8 finish=length
  legacy B=2:
    req 0: prompt=100 tokens=128 ttft=2099ms decodeTPS=59.8 finish=length
    req 1: prompt=1500 tokens=128 ttft=2099ms decodeTPS=59.8 finish=length
  legacy B=4:
    req 0: prompt=100 tokens=128 ttft=4259ms decodeTPS=43.2 finish=length
    req 1: prompt=500 tokens=128 ttft=4259ms decodeTPS=43.2 finish=length
    req 2: prompt=1500 tokens=128 ttft=4259ms decodeTPS=43.2 finish=length
    req 3: prompt=500 tokens=128 ttft=4259ms decodeTPS=43.2 finish=length
