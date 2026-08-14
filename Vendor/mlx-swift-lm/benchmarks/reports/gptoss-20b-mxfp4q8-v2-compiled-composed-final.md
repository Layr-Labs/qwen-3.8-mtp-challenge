# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 10.2 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T10:38:58Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
    [v2-compiled B=1] warmup 0.21s (b1=0.04s)
    [v2-compiled B=1] compiledSteps=10 rebinds=3 scratchResets=0 fallbacks=[:]
| v2-compiled | 1 | 500 | 102.2 | 76.7 | 426 | 9.8 |
| v2-compiled | 2 | 100/1500 | 62.0 | 81.9 | 981 | 13.5 |
| v2-compiled | 4 | 100/500/1500/500 | 39.4 | 102.5 | 1460 | 20.5 |

Per-request detail:
    [v2-compiled B=1] warmup 0.09s (b1=0.02s)
    [v2-compiled B=1] compiledSteps=130 rebinds=3 scratchResets=0 fallbacks=[:]
  v2-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=426ms decodeTPS=102.2 finish=length
    [v2-compiled B=2] warmup 0.12s (b1=0.02s b2=0.03s)
    [v2-compiled B=2] compiledSteps=132 rebinds=8 scratchResets=0 fallbacks=[:]
  v2-compiled B=2:
    req 0: prompt=100 tokens=128 ttft=548ms decodeTPS=49.7 finish=length
    req 1: prompt=1500 tokens=128 ttft=1415ms decodeTPS=74.2 finish=length
    [v2-compiled B=4] warmup 0.15s (b1=0.02s b2=0.03s b4=0.04s)
    [v2-compiled B=4] compiledSteps=132 rebinds=15 scratchResets=0 fallbacks=[:]
  v2-compiled B=4:
    req 0: prompt=100 tokens=128 ttft=1460ms decodeTPS=36.2 finish=length
    req 1: prompt=500 tokens=128 ttft=1460ms decodeTPS=36.2 finish=length
    req 2: prompt=1500 tokens=128 ttft=2402ms decodeTPS=49.0 finish=length
    req 3: prompt=500 tokens=128 ttft=1460ms decodeTPS=36.2 finish=length
