# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 18.8 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T07:08:57Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| legacy-compiled | 1 | 500 | 103.3 | 78.9 | 393 | 9.7 |
| legacy-compiled | 2 | 100/1500 | 78.1 | 69.2 | 2072 | 12.8 |
| legacy-compiled | 4 | 100/500/1500/500 | 44.2 | 61.1 | 5500 | 22.1 |

Per-request detail:
  legacy-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=393ms decodeTPS=103.3 finish=length
  legacy-compiled B=2:
    req 0: prompt=100 tokens=128 ttft=2072ms decodeTPS=78.1 finish=length
    req 1: prompt=1500 tokens=128 ttft=2072ms decodeTPS=78.1 finish=length
  legacy-compiled B=4:
    req 0: prompt=100 tokens=128 ttft=5500ms decodeTPS=44.2 finish=length
    req 1: prompt=500 tokens=128 ttft=5500ms decodeTPS=44.2 finish=length
    req 2: prompt=1500 tokens=128 ttft=5500ms decodeTPS=44.2 finish=length
    req 3: prompt=500 tokens=128 ttft=5500ms decodeTPS=44.2 finish=length
