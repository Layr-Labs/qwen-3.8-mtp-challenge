# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 14.3 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T10:32:18Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| legacy-compiled | 1 | 500 | 101.2 | 77.4 | 399 | 9.8 |
| legacy-compiled | 2 | 100/1500 | 77.0 | 68.1 | 2112 | 13.0 |
| legacy-compiled | 4 | 100/500/1500/500 | 51.5 | 76.0 | 4270 | 19.3 |

Per-request detail:
  legacy-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=399ms decodeTPS=101.2 finish=length
  legacy-compiled B=2:
    req 0: prompt=100 tokens=128 ttft=2112ms decodeTPS=77.0 finish=length
    req 1: prompt=1500 tokens=128 ttft=2112ms decodeTPS=77.0 finish=length
  legacy-compiled B=4:
    req 0: prompt=100 tokens=128 ttft=4270ms decodeTPS=51.5 finish=length
    req 1: prompt=500 tokens=128 ttft=4270ms decodeTPS=51.5 finish=length
    req 2: prompt=1500 tokens=128 ttft=4270ms decodeTPS=51.5 finish=length
    req 3: prompt=500 tokens=128 ttft=4270ms decodeTPS=51.5 finish=length
