# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-qat-4bit/snapshots/0e3cbab38ce568cf6e23543010d08d03b731910c |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 6.7 / 16 cores; no darkbloom process |
| Date | 2026-07-02T19:27:41Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
    [v2-compiled B=1] warmup 0.16s (b1=0.05s)
    [v2-compiled B=1] compiledSteps=10 rebinds=3 scratchResets=0 fallbacks=[:]
| v2-compiled | 1 | 500 | 103.9 | 77.9 | 421 | 9.6 |
| v2-compiled | 2 | 100/1500 | 62.3 | 81.5 | 990 | 13.1 |
| v2-compiled | 4 | 100/500/1500/500 | 39.1 | 103.4 | 1366 | 20.6 |

Per-request detail:
    [v2-compiled B=1] warmup 0.09s (b1=0.02s)
    [v2-compiled B=1] compiledSteps=130 rebinds=3 scratchResets=0 fallbacks=[:]
  v2-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=421ms decodeTPS=103.9 finish=length
    [v2-compiled B=2] warmup 0.18s (b1=0.03s b2=0.09s)
    [v2-compiled B=2] compiledSteps=132 rebinds=8 scratchResets=0 fallbacks=[:]
  v2-compiled B=2:
    req 0: prompt=100 tokens=128 ttft=554ms decodeTPS=50.6 finish=length
    req 1: prompt=1500 tokens=128 ttft=1426ms decodeTPS=74.0 finish=length
    [v2-compiled B=4] warmup 0.16s (b1=0.03s b2=0.03s b4=0.04s)
    [v2-compiled B=4] compiledSteps=132 rebinds=14 scratchResets=0 fallbacks=[:]
  v2-compiled B=4:
    req 0: prompt=100 tokens=128 ttft=1366ms decodeTPS=36.3 finish=length
    req 1: prompt=500 tokens=128 ttft=1366ms decodeTPS=36.3 finish=length
    req 2: prompt=1500 tokens=128 ttft=2278ms decodeTPS=47.5 finish=length
    req 3: prompt=500 tokens=128 ttft=1366ms decodeTPS=36.3 finish=length
