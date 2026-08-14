# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-qat-4bit/snapshots/0e3cbab38ce568cf6e23543010d08d03b731910c |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Host at start | load avg (1m) 7.4 / 16 cores; no darkbloom process |
| Date | 2026-07-02T19:29:15Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
    [v2-compiled B=1] warmup 0.15s (b1=0.04s)
    [v2-compiled B=1] compiledSteps=10 rebinds=3 scratchResets=0 fallbacks=[:]
| v2-compiled | 1 | 500 | 104.8 | 78.5 | 419 | 9.5 |

Per-request detail:
    [v2-compiled B=1] warmup 0.09s (b1=0.02s)
    [v2-compiled B=1] compiledSteps=130 rebinds=3 scratchResets=0 fallbacks=[:]
  v2-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=419ms decodeTPS=104.8 finish=length
