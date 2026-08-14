# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | false |
| Host at start | load avg (1m) 17.9 / 16 cores; no darkbloom process — **HOST CONTENDED, numbers suspect** |
| Date | 2026-07-02T07:09:03Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Decode-step profile (B=1, maxTokens 128)

### legacy (legacy) B=1 — decodeTPS 78.6

| phase | n | total ms | mean ms | p50 ms | p95 ms | max ms |
|---|---|---|---|---|---|---|
| leg.schedstep.wall | 130 | 2000.1 | 15.385 | 12.657 | 13.190 | 374.595 |
| leg.step.wall | 129 | 1621.1 | 12.567 | 12.534 | 13.045 | 14.972 |
| leg.asyncEval.submit | 129 | 961.1 | 7.451 | 7.423 | 7.747 | 9.328 |
| leg.forward.build | 129 | 657.8 | 5.100 | 5.129 | 5.368 | 5.630 |
| leg.detok.next | 127 | 11.5 | 0.091 | 0.089 | 0.129 | 0.166 |
| leg.readback.wait | 129 | 1.7 | 0.013 | 0.012 | 0.019 | 0.024 |
| leg.detok.append | 127 | 0.3 | 0.002 | 0.002 | 0.005 | 0.006 |

### v2 (contiguous) B=1 — decodeTPS 102.5

| phase | n | total ms | mean ms | p50 ms | p95 ms | max ms |
|---|---|---|---|---|---|---|
| v2.step.wall | 130 | 1724.4 | 13.265 | 9.740 | 10.112 | 366.325 |
| v2.launch.total | 128 | 1331.2 | 10.400 | 9.539 | 9.950 | 116.148 |
| v2.asyncEval.submit | 128 | 1174.8 | 9.178 | 8.333 | 8.741 | 114.493 |
| v2.forward.build | 128 | 154.1 | 1.204 | 1.180 | 1.376 | 1.618 |
| v2.detok.push | 128 | 10.6 | 0.083 | 0.084 | 0.125 | 0.142 |
| v2.readback.wait | 129 | 8.1 | 0.063 | 0.037 | 0.054 | 3.320 |
| v2.stream.emit | 128 | 2.2 | 0.017 | 0.017 | 0.028 | 0.032 |
| v2.boundary | 128 | 1.0 | 0.008 | 0.007 | 0.015 | 0.021 |
| v2.sampler.build | 128 | 0.3 | 0.003 | 0.003 | 0.004 | 0.005 |

