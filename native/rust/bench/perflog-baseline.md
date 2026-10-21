# perflog baseline

macOS-27.0.1-arm64-arm-64bit, arm64, Python 3.12.6; load average at start 3.1; reps 5 (time), 3 (memory); 2026-10-20 22:29

## Time (ms per call: median of the reps; spread = (max - min) / median of the whole call)

Columns: total = Game call; game/native pre and post = the Python layers; detect = _core.detect; start read = detect - detect(start=None); capsule = vcount/ecount/capsule/3 probes from Python; core = the driver's whole core call on the same input (check and adjacency are its first steps, timed apart); binding = detect(start=None) - core - capsule (generator, buffers, result lists). Audit rows: game pre = _prepare in Python (normalize_membership, buffers), detect = _audit_prepare + _audit_intervals, core = driver prepare + intervals, of which the core adjacency column holds the prepare and the run column the intervals; binding = in-process intervals - driver intervals (the result lists); game post = after the intervals.

| workload | core | total | spread | game pre | native pre | detect | start read | capsule | binding | core | core check | core adjacency | run (core - check - adj) | native post | game post | Ginstr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zachary partition x100 | c | 0.050 | 2% | 0.004 | 0.008 | 0.031 | 0.001 | 0.001 | 0.023 | 0.006 | 0.000 | 0.000 | 0.006 | 0.003 | 0.004 | 0.00 |
| zachary partition x100 | rust | 0.029 | 4% | 0.005 | 0.008 | 0.009 | 0.001 | 0.001 | 0.000 | 0.007 | 0.000 | 0.000 | 0.006 | 0.003 | 0.004 | 0.00 |
| zachary cover M=3 x100 | c | 0.067 | 7% | 0.006 | 0.008 | 0.043 | 0.001 | 0.001 | 0.027 | 0.014 | 0.000 | 0.000 | 0.014 | 0.004 | 0.005 | 0.00 |
| zachary cover M=3 x100 | rust | 0.045 | 9% | 0.006 | 0.008 | 0.022 | 0.002 | 0.001 | 0.004 | 0.015 | 0.000 | 0.000 | 0.015 | 0.004 | 0.004 | 0.00 |
| BA 50k partition | c | 26.4 | 22% | 1.636 | 0.019 | 16.5 | -0.503 | 0.002 | -0.131 | 17.1 | 0.287 | 0.835 | 16.0 | 1.520 | 6.700 | 0.29 |
| BA 50k partition | rust | 27.5 | 29% | 1.733 | 0.019 | 17.4 | 0.780 | 0.002 | -1.557 | 18.2 | 0.230 | 0.896 | 17.0 | 1.553 | 6.851 | 0.29 |
| BA 50k cover M=2 | c | 142.2 | 16% | 3.606 | 0.020 | 126.8 | 8.219 | 0.002 | -22.217 | 140.8 | 0.353 | 0.882 | 139.5 | 5.304 | 6.421 | 1.08 |
| BA 50k cover M=2 | rust | 119.7 | 6% | 3.524 | 0.018 | 104.0 | -15.092 | 0.002 | 13.5 | 105.6 | 0.299 | 0.895 | 104.4 | 5.325 | 6.577 | 1.11 |
| BA 50k cover M=5 | c | 210.3 | 10% | 3.398 | 0.019 | 189.3 | -0.934 | 0.002 | 3.540 | 186.7 | 0.345 | 0.842 | 185.6 | 9.153 | 8.489 | 1.47 |
| BA 50k cover M=5 | rust | 236.5 | 20% | 3.464 | 0.020 | 214.3 | 4.598 | 0.002 | 41.7 | 168.0 | 0.290 | 0.905 | 166.8 | 9.380 | 9.275 | 1.51 |
| BA 50k cover M=-1 | c | 4494.1 | 7% | 3.710 | 0.019 | 4402.2 | -22.632 | 0.002 | 338.2 | 4086.6 | 0.354 | 0.890 | 4085.4 | 47.0 | 40.6 | 28.32 |
| BA 50k cover M=-1 | rust | 3225.8 | 6% | 3.596 | 0.019 | 3131.1 | 19.1 | 0.002 | 48.6 | 3063.3 | 0.309 | 0.923 | 3062.1 | 47.4 | 38.3 | 16.54 |
| SBM 20k weighted partition | c | 12.4 | 63% | 1.426 | 0.083 | 7.606 | 0.017 | 0.002 | 0.607 | 6.980 | 0.163 | 0.439 | 6.378 | 0.641 | 2.680 | 0.13 |
| SBM 20k weighted partition | rust | 12.1 | 68% | 1.452 | 0.090 | 7.306 | 0.221 | 0.002 | 0.015 | 7.068 | 0.128 | 0.486 | 6.454 | 0.640 | 2.757 | 0.13 |
| SBM 20k node-weighted partition | c | 17.9 | 42% | 1.046 | 0.050 | 13.4 | 0.430 | 0.002 | -0.746 | 13.7 | 0.167 | 0.375 | 13.2 | 0.637 | 2.653 | 0.18 |
| SBM 20k node-weighted partition | rust | 17.2 | 61% | 1.020 | 0.051 | 12.6 | -0.027 | 0.002 | 0.273 | 12.4 | 0.136 | 0.405 | 11.9 | 0.661 | 2.763 | 0.17 |
| SBM 20k random start K=40 | c | 15.7 | 101% | 1.117 | 0.016 | 11.8 | - | 0.002 | -0.157 | 11.9 | 0.168 | 0.370 | 11.4 | 0.528 | 2.180 | 0.14 |
| SBM 20k random start K=40 | rust | 15.7 | 110% | 1.142 | 0.017 | 11.7 | - | 0.002 | 0.339 | 11.4 | 0.127 | 0.440 | 10.8 | 0.543 | 2.286 | 0.14 |
| DBLP top5000 partition | c | 106.5 | 17% | 3.728 | 0.025 | 64.7 | -6.878 | 0.002 | 16.7 | 54.9 | 0.646 | 1.803 | 52.4 | 2.824 | 34.6 | 0.63 |
| DBLP top5000 partition | rust | 93.9 | 10% | 3.674 | 0.024 | 55.7 | -1.493 | 0.002 | -3.310 | 60.5 | 0.506 | 1.969 | 58.1 | 2.846 | 31.8 | 0.62 |
| DBLP top5000 cover M=-1 | c | 1109.0 | 12% | 7.031 | 0.022 | 1011.8 | -5.880 | 0.002 | -22.730 | 1040.4 | 0.774 | 1.866 | 1037.8 | 81.2 | 17.7 | 5.45 |
| DBLP top5000 cover M=-1 | rust | 775.9 | 11% | 6.999 | 0.024 | 688.2 | -29.928 | 0.002 | 1.409 | 716.7 | 0.685 | 1.953 | 714.1 | 73.9 | 17.2 | 5.26 |
| Amazon top5000 partition | c | 9.152 | 87% | 0.592 | 0.020 | 5.493 | -0.116 | 0.002 | 0.563 | 5.045 | 0.097 | 0.261 | 4.687 | 0.575 | 2.408 | 0.09 |
| Amazon top5000 partition | rust | 8.361 | 95% | 0.609 | 0.018 | 4.896 | 0.198 | 0.002 | -0.107 | 4.804 | 0.075 | 0.272 | 4.457 | 0.546 | 2.301 | 0.09 |
| Amazon top5000 cover M=-1 | c | 72.7 | 20% | 1.288 | 0.019 | 59.4 | -1.866 | 0.002 | 0.283 | 61.0 | 0.115 | 0.266 | 60.6 | 3.941 | 3.508 | 0.50 |
| Amazon top5000 cover M=-1 | rust | 56.6 | 51% | 1.162 | 0.019 | 48.2 | -0.637 | 0.002 | 5.387 | 43.4 | 0.102 | 0.282 | 43.0 | 4.512 | 3.608 | 0.47 |
| YouTube top5000 partition | c | 38.4 | 33% | 2.374 | 0.019 | 27.3 | 1.664 | 0.002 | 1.650 | 24.0 | 0.411 | 1.091 | 22.5 | 1.378 | 5.517 | 0.26 |
| YouTube top5000 partition | rust | 33.8 | 20% | 2.369 | 0.023 | 22.4 | -0.888 | 0.002 | -1.129 | 24.5 | 0.311 | 1.159 | 23.0 | 1.332 | 5.385 | 0.26 |
| YouTube top5000 cover M=-1 | c | 5194.2 | 7% | 3.960 | 0.022 | 5104.9 | 42.8 | 0.002 | 281.9 | 4780.2 | 0.496 | 1.120 | 4778.6 | 55.1 | 31.9 | 35.92 |
| YouTube top5000 cover M=-1 | rust | 4025.1 | 36% | 3.964 | 0.023 | 3935.0 | -18.333 | 0.002 | 408.3 | 3545.0 | 0.417 | 1.164 | 3543.4 | 58.9 | 31.3 | 23.76 |
| DBLP top5000 cover M=3 | c | 500.0 | 23% | 7.316 | 0.022 | 397.7 | 29.2 | 0.002 | -19.630 | 388.1 | 0.784 | 1.888 | 385.4 | 62.7 | 13.6 | 1.81 |
| DBLP top5000 cover M=3 | rust | 307.9 | 16% | 7.577 | 0.026 | 232.0 | -11.563 | 0.002 | 12.7 | 230.9 | 0.682 | 1.972 | 228.2 | 58.2 | 12.4 | 2.32 |
| audit DBLP M=3 cover, capped | c | 426.8 | 19% | 179.4 | - | 245.6 | - | - | 36.9 | 208.0 | - | 7.986 | 200.0 | - | 3.733 | 4.02 |
| audit DBLP M=3 cover, capped | rust | 347.1 | 7% | 170.6 | - | 174.4 | - | - | 17.6 | 155.9 | - | 5.195 | 150.7 | - | 3.795 | 3.95 |
| audit DBLP M=3 cover, uncapped | c | 384.8 | 6% | 175.1 | - | 201.3 | - | - | 12.9 | 187.6 | - | 7.915 | 179.7 | - | 3.699 | 3.93 |
| audit DBLP M=3 cover, uncapped | rust | 472.3 | 16% | 224.3 | - | 210.7 | - | - | 24.5 | 185.3 | - | 5.338 | 179.9 | - | 3.714 | 3.96 |
| audit DBLP M=-1 cover, uncapped | c | 1695.6 | 14% | 303.1 | - | 1398.3 | - | - | 79.4 | 1318.3 | - | 11.8 | 1306.4 | - | 4.090 | 13.92 |
| audit DBLP M=-1 cover, uncapped | rust | 1189.7 | 3% | 290.0 | - | 892.7 | - | - | 41.1 | 850.5 | - | 7.072 | 843.4 | - | 3.720 | 12.51 |

## Instructions (millions per call, medians; single-call workloads only, so x100 rows are blank)

The same layers counted in retired instructions (proc_pid_rusage), which a loaded machine does not blur. core = the driver's whole call (for the audit, intervals only); binding = detect(start=None) - core (audit: in-process intervals - driver intervals).

| workload | core | total | game pre | native pre | detect | start read | binding | core | native post | game post |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BA 50k partition | c | 285.7 | 16.8 | 0.1 | 144.1 | 2.9 | 14.6 | 126.6 | 25.3 | 99.4 |
| BA 50k partition | rust | 288.0 | 17.3 | 0.1 | 145.2 | 6.2 | 12.5 | 126.5 | 25.3 | 98.7 |
| BA 50k cover M=2 | c | 1,080.6 | 46.0 | 0.1 | 883.3 | -12.9 | 85.0 | 811.2 | 73.5 | 77.6 |
| BA 50k cover M=2 | rust | 1,110.6 | 45.5 | 0.1 | 914.1 | 12.8 | 102.6 | 798.7 | 73.5 | 77.4 |
| BA 50k cover M=5 | c | 1,465.5 | 45.0 | 0.1 | 1,190.5 | -19.9 | 115.3 | 1,095.0 | 129.2 | 103.1 |
| BA 50k cover M=5 | rust | 1,511.5 | 45.0 | 0.1 | 1,234.6 | 5.4 | 155.7 | 1,073.5 | 129.1 | 102.7 |
| BA 50k cover M=-1 | c | 28,322.7 | 46.4 | 0.1 | 27,273.3 | -65.1 | 492.4 | 26,846.0 | 557.1 | 446.0 |
| BA 50k cover M=-1 | rust | 16,542.3 | 45.7 | 0.1 | 15,491.7 | -39.5 | 590.3 | 14,940.9 | 557.2 | 447.3 |
| SBM 20k weighted partition | c | 128.7 | 10.7 | 0.8 | 67.6 | 1.2 | 6.0 | 60.4 | 10.1 | 39.5 |
| SBM 20k weighted partition | rust | 128.4 | 10.7 | 0.8 | 67.2 | 2.2 | 4.8 | 60.2 | 10.1 | 39.7 |
| SBM 20k node-weighted partition | c | 183.0 | 9.3 | 0.4 | 123.8 | 1.2 | 8.5 | 114.1 | 10.1 | 39.2 |
| SBM 20k node-weighted partition | rust | 174.5 | 9.3 | 0.4 | 115.1 | 2.1 | 5.7 | 107.3 | 10.1 | 39.6 |
| SBM 20k random start K=40 | c | 141.0 | 10.1 | 0.1 | 87.1 | - | - | 80.4 | 9.7 | 33.9 |
| SBM 20k random start K=40 | rust | 136.6 | 10.1 | 0.1 | 82.2 | - | - | 76.6 | 9.7 | 34.4 |
| DBLP top5000 partition | c | 630.4 | 36.1 | 0.1 | 266.3 | 6.6 | 27.4 | 232.3 | 47.3 | 281.9 |
| DBLP top5000 partition | rust | 621.8 | 36.2 | 0.1 | 261.6 | 12.1 | 23.6 | 225.9 | 47.3 | 279.2 |
| DBLP top5000 cover M=-1 | c | 5,454.0 | 89.2 | 0.1 | 4,607.5 | -35.9 | 237.1 | 4,406.3 | 549.2 | 209.2 |
| DBLP top5000 cover M=-1 | rust | 5,260.8 | 89.7 | 0.1 | 4,422.8 | 13.1 | 319.5 | 4,090.3 | 549.1 | 208.9 |
| Amazon top5000 partition | c | 87.5 | 5.6 | 0.1 | 41.1 | 1.2 | 4.2 | 35.7 | 8.4 | 32.2 |
| Amazon top5000 partition | rust | 87.1 | 5.7 | 0.1 | 40.8 | 2.3 | 3.9 | 34.6 | 8.4 | 32.0 |
| Amazon top5000 cover M=-1 | c | 501.2 | 14.6 | 0.1 | 386.4 | -7.0 | 44.1 | 349.4 | 53.8 | 46.0 |
| Amazon top5000 cover M=-1 | rust | 470.7 | 14.6 | 0.1 | 355.6 | 1.3 | 65.7 | 288.5 | 54.1 | 45.9 |
| YouTube top5000 partition | c | 260.7 | 21.5 | 0.1 | 144.8 | 4.1 | 9.0 | 131.7 | 19.7 | 74.4 |
| YouTube top5000 partition | rust | 260.0 | 21.4 | 0.1 | 144.1 | 6.1 | 5.9 | 132.2 | 19.7 | 74.6 |
| YouTube top5000 cover M=-1 | c | 35,924.3 | 43.8 | 0.1 | 34,646.2 | -57.3 | 522.5 | 34,181.0 | 771.5 | 473.7 |
| YouTube top5000 cover M=-1 | rust | 23,756.5 | 44.7 | 0.1 | 22,470.1 | -33.6 | 486.3 | 22,017.4 | 772.6 | 474.1 |
| DBLP top5000 cover M=3 | c | 1,806.0 | 91.0 | 0.1 | 1,211.3 | -22.0 | 158.8 | 1,074.6 | 352.5 | 153.4 |
| DBLP top5000 cover M=3 | rust | 2,322.7 | 91.3 | 0.1 | 1,725.0 | 19.1 | 213.4 | 1,492.5 | 352.7 | 156.2 |
| audit DBLP M=3 cover, capped | c | 4,017.2 | 2,428.7 | - | 1,502.5 | - | 170.5 | 1,250.7 | - | 60.0 |
| audit DBLP M=3 cover, capped | rust | 3,946.6 | 2,436.3 | - | 1,424.2 | - | 147.8 | 1,201.1 | - | 61.2 |
| audit DBLP M=3 cover, uncapped | c | 3,928.8 | 2,400.1 | - | 1,443.5 | - | 153.6 | 1,209.3 | - | 58.6 |
| audit DBLP M=3 cover, uncapped | rust | 3,960.8 | 2,406.1 | - | 1,469.4 | - | 131.9 | 1,261.8 | - | 59.1 |
| audit DBLP M=-1 cover, uncapped | c | 13,922.0 | 4,250.6 | - | 9,580.8 | - | 508.2 | 8,964.1 | - | 59.9 |
| audit DBLP M=-1 cover, uncapped | rust | 12,506.1 | 4,251.0 | - | 8,166.0 | - | 446.6 | 7,621.4 | - | 60.0 |

## Memory (MB; one fresh child per measurement, median of the memory reps)

child peak = ru_maxrss of the child (wait4); before call = its high-water mark when the call starts; call +maxrss = growth of the high-water mark across the call; call peak footprint = the peak physical footprint during the call minus the footprint before it (proc_pid_rusage interval maximum); Python objects = tracemalloc peak during the call / what is still allocated after it; core heap = the driver's live peak above its inputs, bytes and count of allocations (one call).

| workload | core | child peak | before call | call +maxrss | call peak footprint | Python objects peak / kept | core heap peak | core bytes allocated | core allocations | adjacency heap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zachary partition x100 | c | 87.1 | 86.5 | 0.5 | 0.5 | 1.4 / 1.2 | 0.0 | 0.0 | 18 | 0.0 |
| zachary partition x100 | rust | 87.2 | 86.2 | 0.8 | 0.6 | 1.6 / 1.4 | 0.0 | 0.0 | 20 | 0.0 |
| zachary cover M=3 x100 | c | 86.8 | 86.3 | 0.5 | 0.5 | 1.4 / 1.2 | 0.0 | 0.0 | 27 | 0.0 |
| zachary cover M=3 x100 | rust | 87.2 | 86.4 | 0.8 | 0.6 | 1.6 / 1.4 | 0.0 | 0.0 | 35 | 0.0 |
| BA 50k partition | c | 119.7 | 119.7 | 0.0 | 6.6 | 9.8 / 6.5 | 5.0 | 5.2 | 15 | 1.5 |
| BA 50k partition | rust | 120.5 | 120.5 | 0.0 | 9.1 | 10.0 / 6.8 | 5.5 | 5.7 | 17 | 1.5 |
| BA 50k cover M=2 | c | 120.5 | 120.0 | 0.8 | 15.6 | 15.5 / 9.2 | 8.5 | 10.0 | 41 | 1.5 |
| BA 50k cover M=2 | rust | 121.7 | 119.1 | 2.5 | 17.3 | 15.8 / 9.5 | 8.7 | 10.7 | 55 | 1.5 |
| BA 50k cover M=5 | c | 126.9 | 120.0 | 6.1 | 20.9 | 19.7 / 12.2 | 11.3 | 14.9 | 45 | 1.5 |
| BA 50k cover M=5 | rust | 131.1 | 118.5 | 10.9 | 25.7 | 20.0 / 12.5 | 11.4 | 16.1 | 62 | 1.5 |
| BA 50k cover M=-1 | c | 199.2 | 119.7 | 79.9 | 94.8 | 63.5 / 41.4 | 43.0 | 63.4 | 62 | 1.5 |
| BA 50k cover M=-1 | rust | 211.0 | 119.2 | 91.8 | 103.1 | 63.8 / 41.6 | 43.5 | 65.4 | 70 | 1.5 |
| SBM 20k weighted partition | c | 104.1 | 104.1 | 0.0 | 3.0 | 4.3 / 3.3 | 3.6 | 3.6 | 16 | 2.2 |
| SBM 20k weighted partition | rust | 103.1 | 103.1 | 0.0 | 2.7 | 4.5 / 3.5 | 3.8 | 3.9 | 18 | 2.2 |
| SBM 20k node-weighted partition | c | 104.0 | 104.0 | 0.0 | 1.8 | 4.3 / 3.3 | 2.2 | 2.3 | 15 | 0.8 |
| SBM 20k node-weighted partition | rust | 104.5 | 104.5 | 0.0 | 2.7 | 4.5 / 3.5 | 2.6 | 2.7 | 18 | 1.0 |
| SBM 20k random start K=40 | c | 103.0 | 103.0 | 0.0 | 2.0 | 3.7 / 3.3 | 2.2 | 2.2 | 15 | 0.8 |
| SBM 20k random start K=40 | rust | 103.4 | 103.4 | 0.0 | 2.4 | 3.9 / 3.6 | 2.4 | 2.4 | 17 | 0.8 |
| DBLP top5000 partition | c | 167.0 | 167.0 | 0.0 | 16.3 | 14.9 / 11.1 | 9.5 | 9.9 | 15 | 3.3 |
| DBLP top5000 partition | rust | 167.7 | 167.7 | 0.0 | 17.2 | 15.1 / 11.4 | 10.6 | 11.0 | 17 | 3.3 |
| DBLP top5000 cover M=-1 | c | 182.8 | 167.3 | 15.5 | 53.4 | 38.2 / 22.8 | 25.5 | 32.2 | 53 | 3.3 |
| DBLP top5000 cover M=-1 | rust | 188.9 | 167.3 | 22.1 | 59.9 | 38.5 / 23.1 | 26.8 | 35.6 | 69 | 3.3 |
| Amazon top5000 partition | c | 95.5 | 95.5 | 0.0 | 1.6 | 3.7 / 2.9 | 1.6 | 1.7 | 15 | 0.5 |
| Amazon top5000 partition | rust | 95.0 | 95.0 | 0.0 | 1.5 | 3.9 / 3.1 | 1.8 | 1.9 | 17 | 0.5 |
| Amazon top5000 cover M=-1 | c | 101.9 | 95.6 | 6.3 | 9.2 | 7.8 / 5.7 | 5.3 | 7.4 | 41 | 0.5 |
| Amazon top5000 cover M=-1 | rust | 103.6 | 95.7 | 8.5 | 11.3 | 8.0 / 5.9 | 5.6 | 8.1 | 60 | 0.5 |
| YouTube top5000 partition | c | 131.8 | 131.8 | 0.0 | 5.4 | 6.5 / 4.8 | 5.0 | 5.2 | 18 | 2.0 |
| YouTube top5000 partition | rust | 132.4 | 132.4 | 0.0 | 4.1 | 6.7 / 5.0 | 5.5 | 5.6 | 20 | 2.0 |
| YouTube top5000 cover M=-1 | c | 220.8 | 133.3 | 84.2 | 108.1 | 67.4 / 44.6 | 56.6 | 87.5 | 65 | 2.0 |
| YouTube top5000 cover M=-1 | rust | 238.4 | 133.7 | 104.7 | 128.5 | 67.7 / 44.8 | 56.9 | 88.6 | 73 | 2.0 |
| DBLP top5000 cover M=3 | c | 169.2 | 167.8 | 1.8 | 39.6 | 28.9 / 16.8 | 19.3 | 23.1 | 44 | 3.3 |
| DBLP top5000 cover M=3 | rust | 172.1 | 167.5 | 5.9 | 43.7 | 29.2 / 17.0 | 19.7 | 24.3 | 60 | 3.3 |
| audit DBLP M=3 cover, capped | c | 169.4 | 167.1 | 2.1 | 25.3 | 16.4 / 15.7 | 11.1 | 13.7 | 48 | - |
| audit DBLP M=3 cover, capped | rust | 174.3 | 167.2 | 8.0 | 28.4 | 16.5 / 15.8 | 16.7 | 17.3 | 93,474 | - |
| audit DBLP M=3 cover, uncapped | c | 168.1 | 167.4 | 0.9 | 25.6 | 15.9 / 15.2 | 11.6 | 14.2 | 46 | - |
| audit DBLP M=3 cover, uncapped | rust | 173.6 | 168.0 | 4.5 | 27.8 | 16.0 / 15.3 | 16.9 | 17.5 | 88,706 | - |
| audit DBLP M=-1 cover, uncapped | c | 186.8 | 167.6 | 19.7 | 30.9 | 16.4 / 15.7 | 14.5 | 17.4 | 46 | - |
| audit DBLP M=-1 cover, uncapped | rust | 189.2 | 167.3 | 21.9 | 32.8 | 16.5 / 15.8 | 20.0 | 20.9 | 93,473 | - |

## Reading it

Built from `release/2.1` (the C extension of `native/` and `maturin develop --release` of `rust/`), arm64 venv,
under `guard.sh 16`, with other agents' jobs on the machine (load 3 to 5 on 10 cores): wall times of the
millisecond workloads spread by up to 100%, so compare those by instructions. Every driver run returned the
call's own result (count, quality, hash of the rows), so the core columns are the same game as the call.
`native/rust/bench/perflog-baseline.json` holds every measurement (`perflog.py --from-json` renders it);
`native/rust/bench/profiles/*.top.txt` are the macOS `sample` summaries of five workloads, both cores.

Hotspots, Rust first:

1. **The cover's best-set sort (both cores).** On uncapped covers the visit sorts every candidate label
   (`Scored`, `sort_unstable_by` / `qsort` with `hig_scored_compare`): YouTube top5000 M=-1 spends about 6,800
   of 14,700 Rust samples and 7,150 of 13,200 C samples in the sort, more than the rest of the run. The audit's
   `resort` / `audit_sort_cuts` is its counterpart (DBLP M=-1 audit: 2,760 Rust, 5,800 C samples).
2. **Python result objects and the cyclic GC.** A partition's `game_post` (`[[label] for label in
   membership]`, the occupied set) is 99M (BA 50k) to 279M (DBLP) instructions, as much as the core
   (126M, 226M); 93k one-element lists run 24.7 ms with the GC on and 2.4 ms with it off. GC frames
   (`gc_collect_main`, `list_traverse`, `deduce_unreachable`, `visit_*`) are 20% of the DBLP partition's
   samples. A cover's `native_post` (igraph's `VertexCover`, which copies every cluster) is 549M (DBLP) and
   772M (YouTube) instructions; its Python objects peak at 67 MB for YouTube M=-1, above the core's own
   heap (57 MB).
3. **Rust binding: start rows in, cover lists out.** detect(start=None) minus the core is 20 to 80M
   instructions more in Rust than in C on every cover (BA 50k M=5: 155.7M against 115.3M; DBLP M=-1: 319.5M
   against 237.1M); reading the Game's `[[v] ...]` start costs Rust 13 to 19M more (`fast()` clones every item
   into a `Vec<Bound>`, one `Vec` per row), and `cover_lists` grows `Vec<Vec<Bound>>` per cluster and appends to
   unsized lists where C counts, preallocates and fills.
4. **Rust peak memory on large covers.** The call's high-water mark grows 104.7 MB (Rust) against 84.2 MB (C) on
   YouTube M=-1, 91.8 against 79.9 on BA 50k M=-1, 22.1 against 15.5 on DBLP M=-1, while the cores' own heaps
   (56.9 / 56.6, 43.5 / 43.0, 26.8 / 25.5 MB) and the Python objects (tracemalloc) are the same: the gap is the
   binding's transient vectors (start rows, cluster vectors alongside their lists).
5. **Audit preparation in Python, and the Rust audit's per-vertex vectors.** `_audit._prepare`
   (`normalize_membership`: an `isinstance(label, Integral)` ABC check per label) is 2.4G of 3.9G instructions
   of the capped DBLP M=3 audit (170 to 224 ms against 150 to 180 ms of core). The Rust audit returns
   `Vec<Vec<(f64, f64)>>`: 93,474 allocations against C's 48, and a core heap peak of 16.7 MB against 11.1 MB.
6. **Default starts built in Python.** For a default start the Game builds `list(range(n))` or
   `[[v] for v in range(n)]` (n lists, more GC) and `any(game.is_loop())` (m bools; 160 samples in the DBLP
   partition profile), and the binding reads the rows back, where `start=None` gives the core the same
   singletons (checked bit for bit by the harness).
