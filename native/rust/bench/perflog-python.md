# perflog of the Python layer's pass

`perflog.py --cores rust` on all workloads, twice in a row on the same machine, once the overlap queue had finished
(load 2 to 3 on 10 cores), under `guard.sh 16`: before = `release/2.1` at 351a26f4, after = dd566742
(`perf(python): the Python layer's deferred perflog proposals`). Both load the same compiled core (source digest
19bc0c97...), so only the Python sources differ: the before tree is 351a26f4's `src/` with that extension
copied in, put first on `PYTHONPATH`. Every driver run returned the call's own result. Full tables:
`perflog-python-before.json`, `perflog-python-after.json` (`perflog.py --from-json`).

## Reading it

- **Audits** spend 3.1 to 3.9 times fewer instructions preparing the cover (`game pre`: 2,428M to 783M capped,
  4,251M to 1,089M uncapped M=-1), because plain-int rows skip the per-label `Integral` check. The whole call is 1.3
  to 1.7 times cheaper (4,029M to 2,385M; 320 to 211 ms).
- **Default starts** are left to the core, so `game pre` drops to almost nothing (BA 50k: 1.6 ms to 0.04 ms, 16.8M
  to 0.1M instructions). `detect` also loses the start rows' read (BA 50k partition: 150.0M to 139.7M).
- **Partitions' `game post`** (the stored rows, built with the cyclic collector held) is 1.4 to 3.6 times cheaper:
  DBLP top5000 goes from 26.5 to 7.4 ms (278M to 115M). DBLP partition total: 71.0 to 46.6 ms.
- **Covers' `native post`** (`VertexCover` under `gc_paused`) is up to 4.8 times cheaper on DBLP (M=3: 48.3 to
  10.1 ms; M=-1: 67.8 to 16.6 ms) and 1.6 times on YouTube M=-1 (771.6M to 498.1M instructions). The call's Python
  objects and peak footprint fall with it (DBLP M=3: 29.1 to 19.8 MB tracemalloc peak).
- The core is unchanged. Detect time on the long uncapped covers moves by up to 7% each way (YouTube M=-1 2,147 to
  2,297 ms) while its instruction count stays the same (21,549M to 21,533M): that is machine noise.

## Time (ms per call, median; before -> after)
| workload | total | game pre | detect | native post | game post |
|---|---:|---:|---:|---:|---:|
| zachary partition x100 | 0.034 -> 0.031 | 0.006 -> 0.005 | 0.013 -> 0.012 | 0.003 -> 0.003 | 0.004 -> 0.004 |
| zachary cover M=3 x100 | 0.045 -> 0.044 | 0.007 -> 0.005 | 0.023 -> 0.023 | 0.003 -> 0.005 | 0.004 -> 0.004 |
| BA 50k partition | 25.6 -> 19.8 | 1.629 -> 0.043 | 15.6 -> 14.4 | 1.490 -> 1.473 | 6.549 -> 3.892 |
| BA 50k cover M=2 | 118.2 -> 111.7 | 3.484 -> 0.051 | 102.1 -> 102.1 | 5.071 -> 4.709 | 6.220 -> 4.853 |
| BA 50k cover M=5 | 199.1 -> 192.4 | 3.471 -> 0.059 | 175.8 -> 176.8 | 9.777 -> 8.394 | 8.963 -> 6.697 |
| BA 50k cover M=-1 | 1698.1 -> 1610.0 | 3.500 -> 0.051 | 1601.8 -> 1545.0 | 39.6 -> 37.2 | 34.8 -> 27.8 |
| SBM 20k weighted partition | 11.7 -> 9.495 | 1.425 -> 0.475 | 6.882 -> 6.658 | 0.611 -> 0.621 | 2.560 -> 1.613 |
| SBM 20k node-weighted partition | 17.2 -> 13.8 | 1.056 -> 0.097 | 12.8 -> 11.5 | 0.664 -> 0.618 | 2.620 -> 1.524 |
| SBM 20k random start K=40 | 14.5 -> 12.5 | 1.096 -> 0.292 | 10.7 -> 10.4 | 0.513 -> 0.497 | 2.025 -> 1.228 |
| DBLP top5000 partition | 71.0 -> 46.6 | 3.460 -> 0.043 | 38.6 -> 36.5 | 2.679 -> 2.635 | 26.5 -> 7.434 |
| DBLP top5000 cover M=-1 | 611.3 -> 587.4 | 6.659 -> 0.058 | 521.1 -> 557.7 | 67.8 -> 16.6 | 15.7 -> 13.2 |
| Amazon top5000 partition | 8.204 -> 6.700 | 0.627 -> 0.040 | 4.889 -> 4.696 | 0.541 -> 0.548 | 2.199 -> 1.367 |
| Amazon top5000 cover M=-1 | 49.2 -> 43.8 | 1.209 -> 0.048 | 40.3 -> 38.0 | 3.497 -> 3.051 | 3.271 -> 2.533 |
| YouTube top5000 partition | 29.7 -> 25.3 | 2.275 -> 0.047 | 21.0 -> 20.8 | 1.287 -> 1.278 | 5.048 -> 3.139 |
| YouTube top5000 cover M=-1 | 2229.9 -> 2352.9 | 3.606 -> 0.053 | 2147.4 -> 2296.9 | 52.1 -> 33.5 | 29.3 -> 23.0 |
| DBLP top5000 cover M=3 | 241.5 -> 191.0 | 6.906 -> 0.052 | 174.5 -> 171.1 | 48.3 -> 10.1 | 11.8 -> 8.949 |
| audit DBLP M=3 cover, capped | 320.2 -> 211.0 | 152.1 -> 48.4 | 160.6 -> 159.0 | - | 3.491 -> 3.496 |
| audit DBLP M=3 cover, uncapped | 324.7 -> 212.3 | 154.2 -> 46.7 | 161.4 -> 161.0 | - | 3.439 -> 3.469 |
| audit DBLP M=-1 cover, uncapped | 1076.4 -> 889.2 | 265.9 -> 70.1 | 808.6 -> 814.8 | - | 3.494 -> 3.566 |

## Instructions (millions per call, median; before -> after)
| workload | total | game pre | detect | native post | game post |
|---|---:|---:|---:|---:|---:|
| BA 50k partition | 292.1 -> 225.4 | 16.8 -> 0.1 | 150.0 -> 139.7 | 25.3 -> 25.3 | 99.9 -> 60.5 |
| BA 50k cover M=2 | 1,049.2 -> 957.1 | 45.6 -> 0.1 | 852.6 -> 832.9 | 73.8 -> 70.4 | 77.5 -> 53.6 |
| BA 50k cover M=5 | 1,432.0 -> 1,335.4 | 45.4 -> 0.1 | 1,155.4 -> 1,136.8 | 129.4 -> 123.3 | 102.8 -> 75.0 |
| BA 50k cover M=-1 | 15,434.8 -> 15,186.8 | 46.3 -> 0.1 | 14,382.6 -> 14,364.9 | 556.9 -> 491.2 | 447.7 -> 336.5 |
| SBM 20k weighted partition | 132.6 -> 104.5 | 11.1 -> 1.8 | 71.5 -> 67.6 | 10.1 -> 10.1 | 39.2 -> 24.2 |
| SBM 20k node-weighted partition | 179.5 -> 150.1 | 9.8 -> 0.5 | 119.7 -> 115.4 | 10.1 -> 10.1 | 39.1 -> 23.8 |
| SBM 20k random start K=40 | 140.0 -> 120.6 | 10.1 -> 3.5 | 85.9 -> 85.9 | 9.7 -> 9.7 | 34.1 -> 21.3 |
| DBLP top5000 partition | 632.7 -> 415.1 | 36.5 -> 0.1 | 272.6 -> 253.0 | 47.3 -> 47.3 | 278.3 -> 114.5 |
| DBLP top5000 cover M=-1 | 4,546.3 -> 4,054.1 | 89.3 -> 0.1 | 3,703.0 -> 3,669.2 | 549.0 -> 232.9 | 208.8 -> 151.7 |
| Amazon top5000 partition | 89.0 -> 67.3 | 5.7 -> 0.1 | 42.8 -> 39.1 | 8.4 -> 8.4 | 31.9 -> 19.5 |
| Amazon top5000 cover M=-1 | 439.2 -> 398.4 | 14.9 -> 0.1 | 324.5 -> 316.2 | 53.9 -> 48.8 | 45.9 -> 33.1 |
| YouTube top5000 partition | 267.5 -> 207.5 | 21.6 -> 0.1 | 152.1 -> 144.1 | 19.7 -> 19.7 | 73.9 -> 43.5 |
| YouTube top5000 cover M=-1 | 22,837.9 -> 22,378.2 | 44.0 -> 0.1 | 21,549.1 -> 21,533.3 | 771.6 -> 498.1 | 473.5 -> 358.5 |
| DBLP top5000 cover M=3 | 2,005.6 -> 1,617.4 | 91.0 -> 0.1 | 1,410.2 -> 1,375.7 | 352.3 -> 133.5 | 154.2 -> 107.9 |
| audit DBLP M=3 cover, capped | 4,029.1 -> 2,385.3 | 2,427.5 -> 782.8 | 1,516.6 -> 1,517.2 | - | 59.1 -> 60.7 |
| audit DBLP M=3 cover, uncapped | 4,040.4 -> 2,399.0 | 2,396.0 -> 753.4 | 1,559.5 -> 1,560.8 | - | 58.6 -> 59.0 |
| audit DBLP M=-1 cover, uncapped | 12,738.4 -> 9,574.1 | 4,251.3 -> 1,089.2 | 8,399.9 -> 8,399.7 | - | 59.7 -> 58.9 |

## Memory (MB, median of the memory reps; before -> after)
| workload | call +maxrss | call peak footprint | Python objects peak / kept |
|---|---:|---:|---:|
| zachary partition x100 | 1.5 -> 1.8 | 1.2 -> 1.6 | 1.7 / 1.4 -> 1.7 / 1.4 |
| zachary cover M=3 x100 | 1.5 -> 1.9 | 1.2 -> 1.6 | 1.7 / 1.4 -> 1.7 / 1.4 |
| BA 50k partition | 0.0 -> 0.0 | 6.1 -> 4.8 | 10.0 / 6.8 -> 8.5 / 6.8 |
| BA 50k cover M=2 | 1.5 -> 0.0 | 16.1 -> 10.8 | 15.8 / 9.4 -> 10.8 / 9.4 |
| BA 50k cover M=5 | 6.5 -> 1.8 | 21.2 -> 16.5 | 19.9 / 12.4 -> 14.9 / 12.4 |
| BA 50k cover M=-1 | 81.3 -> 75.1 | 96.0 -> 89.9 | 63.8 / 41.6 -> 58.8 / 41.6 |
| SBM 20k weighted partition | 0.0 -> 0.0 | 3.8 -> 2.0 | 4.5 / 3.5 -> 3.9 / 3.5 |
| SBM 20k node-weighted partition | 0.0 -> 0.0 | 3.0 -> 2.1 | 4.5 / 3.5 -> 3.9 / 3.5 |
| SBM 20k random start K=40 | 0.0 -> 0.0 | 2.6 -> 2.1 | 4.0 / 3.6 -> 4.1 / 3.6 |
| DBLP top5000 partition | 0.0 -> 0.0 | 17.6 -> 12.7 | 15.2 / 11.4 -> 12.3 / 11.3 |
| DBLP top5000 cover M=-1 | 11.7 -> 4.3 | 50.4 -> 43.0 | 38.5 / 23.0 -> 29.1 / 23.0 |
| Amazon top5000 partition | 0.0 -> 0.0 | 2.2 -> 1.0 | 4.0 / 3.1 -> 3.5 / 3.1 |
| Amazon top5000 cover M=-1 | 7.1 -> 3.7 | 9.8 -> 6.4 | 8.0 / 5.9 -> 6.4 / 5.9 |
| YouTube top5000 partition | 0.0 -> 0.0 | 4.3 -> 4.2 | 6.8 / 5.0 -> 5.6 / 5.0 |
| YouTube top5000 cover M=-1 | 82.8 -> 86.8 | 106.6 -> 110.5 | 67.7 / 44.8 -> 63.7 / 44.8 |
| DBLP top5000 cover M=3 | 0.7 -> 0.0 | 39.4 -> 26.3 | 29.1 / 17.0 -> 19.8 / 17.0 |
| audit DBLP M=3 cover, capped | 4.7 -> 1.4 | 25.6 -> 25.1 | 16.5 / 15.8 -> 16.5 / 15.8 |
| audit DBLP M=3 cover, uncapped | 1.7 -> 4.3 | 24.6 -> 25.8 | 16.0 / 15.3 -> 16.0 / 15.3 |
| audit DBLP M=-1 cover, uncapped | 20.8 -> 18.1 | 31.8 -> 29.1 | 16.5 / 15.8 -> 18.0 / 15.8 |

