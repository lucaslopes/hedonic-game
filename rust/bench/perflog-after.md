# perflog after the Rust performance pass

The same harness and workloads as `perflog-baseline.md`, measured on the
commit that ends the pass (`perf(rust): drop the cover's label-record
prefetch stage`). The C core did not change, so its rows show how much the
machine moved between the two runs. Its times differ by up to 49% (DBLP
cover M=3, 388 to 199 ms core) while its instruction counts stay within
0.5%. Read the time columns against the C rows, and trust the instruction
and memory columns first.

## Reading it

Rust before and after, core call (`core`) and core heap:

| workload | core ms | core Minstr | core heap MB | core allocations |
|---|---:|---:|---:|---:|
| BA 50k cover M=-1 | 3063 -> 1778 | 14,941 -> 14,086 | 43.5 -> 36.1 | 70 -> 56 |
| YouTube top5000 cover M=-1 | 3545 -> 2132 | 22,017 -> 21,195 | 56.9 -> 50.1 | 73 -> 60 |
| DBLP top5000 cover M=-1 | 717 -> 566 | 4,090 -> 3,496 | 26.8 -> 22.2 | 69 -> 55 |
| DBLP top5000 cover M=3 | 231 -> 172 | 1,493 -> 1,247 | 19.7 -> 16.5 | 60 -> 50 |
| Amazon top5000 cover M=-1 | 43.4 -> 34.5 | 289 -> 282 | 5.6 -> 4.5 | 60 -> 48 |
| BA 50k cover M=2 / M=5 | 106 -> 98 / 168 -> 173 | 799 -> 763 / 1,074 -> 1,046 | 8.7 -> 7.3 / 11.4 -> 9.4 | 55 -> 45 / 62 -> 49 |
| partitions (DBLP, BA) | within noise | equal | 10.6 -> 9.1, 5.5 -> 4.6 | equal |
| audit DBLP M=3, capped | 156 -> 140 | 1,201 -> 1,138 | 16.7 -> 11.6 | 93,474 -> 58 |
| audit DBLP M=-1, uncapped | 851 -> 798 | 7,621 -> 7,450 | 20.0 -> 14.9 | 93,473 -> 57 |

- The uncapped covers gain the most. That is the integer-key sort of the
  long best-set selection.
- Heap peaks fall to C's level or below. Three changes do it: the run state
  is released before the result is built, the audit's intervals sit in one
  buffer, and the slack's open list is sized exactly.
- detect's call growth falls because the result lists are built at their
  final size and the start rows are read in place. YouTube M=-1 goes from
  104.7 to 87.4 MB. C's is 80.6 MB; the single maxrss reading is noisy at
  this size.
- Rust still retires more instructions than C on the capped DBLP cover
  (M=3: 1,247M against 1,069M), although its time is lower.


macOS-27.0.1-arm64-arm-64bit, arm64, Python 3.12.6; load average at start 2.8; reps 5 (time), 3 (memory); 2026-10-21 00:06

## Time (ms per call: median of the reps; spread = (max - min) / median of the whole call)

Columns: total = Game call; game/native pre and post = the Python layers; detect = _core.detect; start read = detect - detect(start=None); capsule = vcount/ecount/capsule/3 probes from Python; core = the driver's whole core call on the same input (check and adjacency are its first steps, timed apart); binding = detect(start=None) - core - capsule (generator, buffers, result lists). Audit rows: game pre = _prepare in Python (normalize_membership, buffers), detect = _audit_prepare + _audit_intervals, core = driver prepare + intervals, of which the core adjacency column holds the prepare and the run column the intervals; binding = in-process intervals - driver intervals (the result lists); game post = after the intervals.

| workload | core | total | spread | game pre | native pre | detect | start read | capsule | binding | core | core check | core adjacency | run (core - check - adj) | native post | game post | Ginstr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zachary partition x100 | c | 0.052 | 1% | 0.005 | 0.008 | 0.032 | 0.001 | 0.001 | 0.024 | 0.006 | 0.000 | 0.000 | 0.006 | 0.003 | 0.004 | 0.00 |
| zachary partition x100 | rust | 0.029 | 5% | 0.005 | 0.008 | 0.009 | 0.001 | 0.001 | 0.001 | 0.006 | 0.000 | 0.000 | 0.005 | 0.003 | 0.004 | 0.00 |
| zachary cover M=3 x100 | c | 0.062 | 6% | 0.006 | 0.007 | 0.041 | 0.001 | 0.001 | 0.024 | 0.015 | 0.000 | 0.000 | 0.015 | 0.003 | 0.004 | 0.00 |
| zachary cover M=3 x100 | rust | 0.041 | 7% | 0.006 | 0.008 | 0.019 | 0.001 | 0.001 | 0.003 | 0.014 | 0.000 | 0.000 | 0.014 | 0.004 | 0.004 | 0.00 |
| BA 50k partition | c | 28.2 | 29% | 1.709 | 0.019 | 16.4 | 0.277 | 0.002 | 0.567 | 15.6 | 0.296 | 0.853 | 14.4 | 1.521 | 6.927 | 0.29 |
| BA 50k partition | rust | 25.4 | 27% | 1.683 | 0.018 | 15.7 | 0.539 | 0.002 | 0.355 | 14.8 | 0.228 | 0.885 | 13.7 | 1.521 | 6.472 | 0.29 |
| BA 50k cover M=2 | c | 122.6 | 6% | 3.555 | 0.018 | 105.8 | -6.239 | 0.002 | 9.297 | 102.7 | 0.344 | 0.865 | 101.5 | 5.117 | 6.332 | 1.08 |
| BA 50k cover M=2 | rust | 124.5 | 20% | 3.450 | 0.018 | 109.4 | 2.669 | 0.002 | 8.527 | 98.2 | 0.294 | 0.874 | 97.0 | 5.264 | 6.641 | 1.04 |
| BA 50k cover M=5 | c | 207.7 | 7% | 3.547 | 0.020 | 185.6 | 3.576 | 0.002 | 6.646 | 175.4 | 0.342 | 0.880 | 174.1 | 9.482 | 8.729 | 1.47 |
| BA 50k cover M=5 | rust | 195.2 | 8% | 3.467 | 0.019 | 173.2 | -1.460 | 0.002 | 1.292 | 173.4 | 0.296 | 0.886 | 172.2 | 9.316 | 8.727 | 1.43 |
| BA 50k cover M=-1 | c | 4364.6 | 5% | 3.559 | 0.022 | 4279.6 | 25.5 | 0.002 | 342.4 | 3911.7 | 0.343 | 0.872 | 3910.5 | 44.6 | 36.5 | 28.34 |
| BA 50k cover M=-1 | rust | 2141.9 | 5% | 3.673 | 0.030 | 2040.6 | -18.994 | 0.002 | 282.1 | 1777.5 | 0.302 | 0.893 | 1776.3 | 48.4 | 41.7 | 15.51 |
| SBM 20k weighted partition | c | 12.2 | 49% | 1.368 | 0.073 | 7.257 | 0.139 | 0.002 | 0.190 | 6.926 | 0.167 | 0.434 | 6.325 | 0.627 | 2.619 | 0.13 |
| SBM 20k weighted partition | rust | 12.1 | 71% | 1.443 | 0.084 | 7.435 | 0.488 | 0.002 | 0.706 | 6.239 | 0.125 | 0.448 | 5.666 | 0.616 | 2.620 | 0.13 |
| SBM 20k node-weighted partition | c | 17.3 | 36% | 0.975 | 0.049 | 13.1 | 0.231 | 0.002 | 0.220 | 12.7 | 0.162 | 0.352 | 12.2 | 0.638 | 2.530 | 0.18 |
| SBM 20k node-weighted partition | rust | 17.1 | 49% | 1.056 | 0.057 | 12.7 | 0.109 | 0.002 | 0.634 | 12.0 | 0.133 | 0.405 | 11.4 | 0.651 | 2.643 | 0.17 |
| SBM 20k random start K=40 | c | 15.4 | 98% | 1.108 | 0.016 | 11.5 | - | 0.002 | 0.540 | 10.9 | 0.162 | 0.357 | 10.4 | 0.521 | 2.097 | 0.14 |
| SBM 20k random start K=40 | rust | 14.7 | 114% | 1.118 | 0.018 | 11.0 | - | 0.002 | 0.486 | 10.5 | 0.125 | 0.390 | 9.990 | 0.519 | 2.086 | 0.14 |
| DBLP top5000 partition | c | 79.1 | 13% | 3.519 | 0.019 | 44.6 | 2.064 | 0.002 | -4.759 | 47.3 | 0.628 | 1.830 | 44.8 | 2.726 | 27.7 | 0.63 |
| DBLP top5000 partition | rust | 81.6 | 29% | 3.550 | 0.020 | 44.7 | 2.249 | 0.002 | -1.634 | 44.1 | 0.507 | 1.881 | 41.7 | 2.792 | 30.1 | 0.62 |
| DBLP top5000 cover M=-1 | c | 943.2 | 7% | 6.821 | 0.021 | 856.5 | -30.212 | 0.002 | -13.260 | 900.0 | 0.752 | 1.871 | 897.4 | 68.0 | 16.7 | 5.45 |
| DBLP top5000 cover M=-1 | rust | 624.9 | 4% | 6.840 | 0.020 | 533.7 | -8.523 | 0.002 | -24.196 | 566.4 | 0.639 | 1.927 | 563.9 | 68.3 | 16.2 | 4.53 |
| Amazon top5000 partition | c | 8.990 | 71% | 0.593 | 0.018 | 5.275 | -0.075 | 0.002 | 0.519 | 4.829 | 0.094 | 0.254 | 4.481 | 0.552 | 2.259 | 0.09 |
| Amazon top5000 partition | rust | 8.471 | 89% | 0.597 | 0.019 | 5.060 | 0.185 | 0.002 | 0.357 | 4.515 | 0.073 | 0.265 | 4.178 | 0.559 | 2.234 | 0.09 |
| Amazon top5000 cover M=-1 | c | 58.6 | 11% | 1.127 | 0.016 | 50.6 | -0.716 | 0.002 | 5.819 | 45.5 | 0.111 | 0.250 | 45.1 | 3.517 | 3.328 | 0.50 |
| Amazon top5000 cover M=-1 | rust | 47.4 | 15% | 1.156 | 0.018 | 39.2 | -0.740 | 0.002 | 5.370 | 34.5 | 0.094 | 0.265 | 34.2 | 3.654 | 3.391 | 0.43 |
| YouTube top5000 partition | c | 31.1 | 21% | 2.296 | 0.018 | 22.3 | 0.453 | 0.002 | -0.922 | 22.7 | 0.410 | 1.108 | 21.2 | 1.302 | 5.102 | 0.26 |
| YouTube top5000 partition | rust | 29.1 | 24% | 2.225 | 0.016 | 20.7 | 0.271 | 0.001 | -0.711 | 21.1 | 0.300 | 1.177 | 19.6 | 1.299 | 4.945 | 0.26 |
| YouTube top5000 cover M=-1 | c | 4718.9 | 1% | 3.692 | 0.020 | 4631.9 | 59.5 | 0.002 | 198.7 | 4373.7 | 0.481 | 1.109 | 4372.1 | 52.7 | 29.5 | 35.93 |
| YouTube top5000 cover M=-1 | rust | 2108.3 | 12% | 3.580 | 0.018 | 2024.7 | -0.967 | 0.002 | -106.345 | 2132.0 | 0.414 | 1.134 | 2130.5 | 51.8 | 29.2 | 22.84 |
| DBLP top5000 cover M=3 | c | 258.4 | 16% | 7.045 | 0.021 | 189.8 | -2.354 | 0.002 | -7.057 | 199.2 | 0.753 | 1.786 | 196.6 | 49.3 | 11.9 | 1.80 |
| DBLP top5000 cover M=3 | rust | 241.2 | 10% | 6.840 | 0.020 | 174.0 | -3.612 | 0.002 | 6.000 | 171.6 | 0.632 | 1.863 | 169.1 | 49.0 | 11.7 | 1.99 |
| audit DBLP M=3 cover, capped | c | 352.8 | 2% | 153.1 | - | 194.2 | - | - | 12.3 | 181.2 | - | 7.567 | 173.6 | - | 3.581 | 4.02 |
| audit DBLP M=3 cover, capped | rust | 311.0 | 2% | 153.5 | - | 151.4 | - | - | 11.3 | 139.6 | - | 5.045 | 134.5 | - | 3.552 | 3.87 |
| audit DBLP M=3 cover, uncapped | c | 339.6 | 2% | 151.1 | - | 184.0 | - | - | 10.2 | 173.4 | - | 7.793 | 165.6 | - | 3.485 | 3.93 |
| audit DBLP M=3 cover, uncapped | rust | 308.5 | 3% | 150.3 | - | 152.9 | - | - | 8.100 | 144.0 | - | 4.881 | 139.1 | - | 3.487 | 3.90 |
| audit DBLP M=-1 cover, uncapped | c | 1438.9 | 2% | 270.8 | - | 1163.3 | - | - | 59.2 | 1103.7 | - | 11.8 | 1091.9 | - | 3.565 | 13.91 |
| audit DBLP M=-1 cover, uncapped | rust | 1123.6 | 10% | 281.8 | - | 839.6 | - | - | 41.3 | 797.5 | - | 7.072 | 790.4 | - | 3.578 | 12.22 |

## Instructions (millions per call, medians; single-call workloads only, so x100 rows are blank)

The same layers counted in retired instructions (proc_pid_rusage), which a loaded machine does not blur. core = the driver's whole call (for the audit, intervals only); binding = detect(start=None) - core (audit: in-process intervals - driver intervals).

| workload | core | total | game pre | native pre | detect | start read | binding | core | native post | game post |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BA 50k partition | c | 285.2 | 17.3 | 0.1 | 144.2 | 3.2 | 14.4 | 126.5 | 25.3 | 98.4 |
| BA 50k partition | rust | 287.3 | 17.1 | 0.1 | 145.3 | 6.2 | 12.5 | 126.5 | 25.3 | 99.6 |
| BA 50k cover M=2 | c | 1,078.7 | 45.9 | 0.1 | 883.3 | -12.2 | 86.5 | 809.0 | 73.3 | 77.1 |
| BA 50k cover M=2 | rust | 1,041.8 | 44.8 | 0.1 | 842.6 | -3.8 | 83.3 | 763.1 | 74.0 | 77.9 |
| BA 50k cover M=5 | c | 1,466.3 | 44.9 | 0.1 | 1,189.9 | -20.2 | 115.8 | 1,094.3 | 129.8 | 103.5 |
| BA 50k cover M=5 | rust | 1,427.1 | 45.1 | 0.1 | 1,146.5 | -11.7 | 112.3 | 1,046.0 | 130.0 | 104.4 |
| BA 50k cover M=-1 | c | 28,337.2 | 46.4 | 0.1 | 27,280.7 | -65.2 | 514.1 | 26,831.8 | 557.3 | 447.0 |
| BA 50k cover M=-1 | rust | 15,506.6 | 46.3 | 0.1 | 14,443.2 | -57.4 | 414.9 | 14,085.7 | 558.8 | 449.6 |
| SBM 20k weighted partition | c | 129.4 | 10.7 | 0.8 | 67.8 | 1.4 | 6.0 | 60.4 | 10.1 | 39.9 |
| SBM 20k weighted partition | rust | 128.6 | 11.1 | 0.8 | 67.3 | 2.4 | 4.9 | 60.0 | 10.1 | 39.2 |
| SBM 20k node-weighted partition | c | 183.2 | 9.3 | 0.4 | 124.0 | 1.3 | 8.6 | 114.1 | 10.1 | 39.3 |
| SBM 20k node-weighted partition | rust | 174.3 | 9.8 | 0.4 | 115.1 | 2.2 | 6.1 | 106.8 | 10.1 | 38.9 |
| SBM 20k random start K=40 | c | 141.9 | 10.1 | 0.1 | 87.2 | - | - | 80.4 | 9.7 | 34.4 |
| SBM 20k random start K=40 | rust | 136.3 | 10.1 | 0.1 | 82.4 | - | - | 76.6 | 9.7 | 34.0 |
| DBLP top5000 partition | c | 627.8 | 35.9 | 0.1 | 266.2 | 6.8 | 26.1 | 233.3 | 47.3 | 280.1 |
| DBLP top5000 partition | rust | 623.7 | 36.5 | 0.1 | 262.4 | 12.9 | 24.1 | 225.4 | 47.3 | 281.0 |
| DBLP top5000 cover M=-1 | c | 5,451.5 | 89.1 | 0.1 | 4,605.4 | -39.7 | 235.4 | 4,409.7 | 549.1 | 210.4 |
| DBLP top5000 cover M=-1 | rust | 4,534.0 | 89.2 | 0.1 | 3,685.5 | -22.3 | 212.3 | 3,495.5 | 549.9 | 212.4 |
| Amazon top5000 partition | c | 87.2 | 5.7 | 0.1 | 41.2 | 1.4 | 4.2 | 35.7 | 8.4 | 31.7 |
| Amazon top5000 partition | rust | 86.9 | 5.8 | 0.1 | 40.7 | 2.2 | 3.8 | 34.7 | 8.4 | 32.0 |
| Amazon top5000 cover M=-1 | c | 499.1 | 14.6 | 0.1 | 384.9 | -8.2 | 44.5 | 348.6 | 53.7 | 45.7 |
| Amazon top5000 cover M=-1 | rust | 434.8 | 14.6 | 0.1 | 320.5 | -4.9 | 43.1 | 282.4 | 53.8 | 45.7 |
| YouTube top5000 partition | c | 260.0 | 21.6 | 0.1 | 144.8 | 4.2 | 9.2 | 131.5 | 19.7 | 73.9 |
| YouTube top5000 partition | rust | 259.6 | 21.6 | 0.1 | 144.7 | 6.8 | 5.7 | 132.2 | 19.7 | 73.4 |
| YouTube top5000 cover M=-1 | c | 35,931.3 | 44.1 | 0.1 | 34,643.4 | -48.7 | 521.2 | 34,170.9 | 771.6 | 473.2 |
| YouTube top5000 cover M=-1 | rust | 22,838.7 | 44.0 | 0.1 | 21,558.5 | -50.4 | 413.8 | 21,195.1 | 771.6 | 473.7 |
| DBLP top5000 cover M=3 | c | 1,804.9 | 90.9 | 0.1 | 1,208.0 | -24.9 | 163.5 | 1,069.4 | 352.4 | 155.0 |
| DBLP top5000 cover M=3 | rust | 1,987.2 | 90.4 | 0.1 | 1,391.9 | -9.3 | 154.7 | 1,246.5 | 352.2 | 153.6 |
| audit DBLP M=3 cover, capped | c | 4,024.0 | 2,435.1 | - | 1,500.7 | - | 170.8 | 1,249.2 | - | 61.3 |
| audit DBLP M=3 cover, capped | rust | 3,872.2 | 2,430.4 | - | 1,354.8 | - | 142.7 | 1,137.9 | - | 61.2 |
| audit DBLP M=3 cover, uncapped | c | 3,927.5 | 2,398.2 | - | 1,443.5 | - | 153.9 | 1,208.9 | - | 59.1 |
| audit DBLP M=3 cover, uncapped | rust | 3,896.6 | 2,397.1 | - | 1,414.9 | - | 126.9 | 1,213.7 | - | 58.9 |
| audit DBLP M=-1 cover, uncapped | c | 13,913.2 | 4,250.6 | - | 9,575.9 | - | 507.0 | 8,960.0 | - | 59.5 |
| audit DBLP M=-1 cover, uncapped | rust | 12,222.0 | 4,251.2 | - | 7,882.5 | - | 335.2 | 7,449.7 | - | 59.6 |

## Memory (MB; one fresh child per measurement, median of the memory reps)

child peak = ru_maxrss of the child (wait4); before call = its high-water mark when the call starts; call +maxrss = growth of the high-water mark across the call; call peak footprint = the peak physical footprint during the call minus the footprint before it (proc_pid_rusage interval maximum); Python objects = tracemalloc peak during the call / what is still allocated after it; core heap = the driver's live peak above its inputs, bytes and count of allocations (one call).

| workload | core | child peak | before call | call +maxrss | call peak footprint | Python objects peak / kept | core heap peak | core bytes allocated | core allocations | adjacency heap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zachary partition x100 | c | 72.6 | 71.1 | 1.4 | 1.4 | 1.4 / 1.2 | 0.0 | 0.0 | 18 | 0.0 |
| zachary partition x100 | rust | 72.8 | 71.0 | 1.8 | 1.6 | 1.6 / 1.4 | 0.0 | 0.0 | 20 | 0.0 |
| zachary cover M=3 x100 | c | 72.7 | 71.2 | 1.4 | 1.4 | 1.4 / 1.2 | 0.0 | 0.0 | 27 | 0.0 |
| zachary cover M=3 x100 | rust | 73.0 | 71.0 | 1.9 | 1.6 | 1.6 / 1.4 | 0.0 | 0.0 | 31 | 0.0 |
| BA 50k partition | c | 107.0 | 107.0 | 0.0 | 6.5 | 9.8 / 6.5 | 5.0 | 5.2 | 15 | 1.5 |
| BA 50k partition | rust | 107.6 | 107.6 | 0.0 | 6.9 | 10.0 / 6.8 | 4.6 | 5.5 | 17 | 1.5 |
| BA 50k cover M=2 | c | 109.2 | 107.0 | 2.1 | 16.9 | 15.5 / 9.2 | 8.5 | 10.0 | 41 | 1.5 |
| BA 50k cover M=2 | rust | 108.8 | 107.3 | 1.6 | 16.3 | 15.8 / 9.4 | 7.3 | 10.4 | 45 | 1.5 |
| BA 50k cover M=5 | c | 114.0 | 107.2 | 6.7 | 21.5 | 19.7 / 12.2 | 11.3 | 14.9 | 45 | 1.5 |
| BA 50k cover M=5 | rust | 114.0 | 107.9 | 6.7 | 21.4 | 19.9 / 12.4 | 9.4 | 15.3 | 49 | 1.5 |
| BA 50k cover M=-1 | c | 188.0 | 106.8 | 80.5 | 95.4 | 63.5 / 41.4 | 43.0 | 63.4 | 62 | 1.5 |
| BA 50k cover M=-1 | rust | 190.0 | 107.5 | 81.6 | 96.4 | 63.8 / 41.6 | 36.1 | 64.0 | 56 | 1.5 |
| SBM 20k weighted partition | c | 91.0 | 91.0 | 0.0 | 2.7 | 4.3 / 3.3 | 3.6 | 3.6 | 16 | 2.2 |
| SBM 20k weighted partition | rust | 91.5 | 91.5 | 0.0 | 2.4 | 4.5 / 3.5 | 3.4 | 3.8 | 18 | 2.2 |
| SBM 20k node-weighted partition | c | 90.8 | 90.8 | 0.0 | 2.0 | 4.3 / 3.3 | 2.2 | 2.3 | 15 | 0.8 |
| SBM 20k node-weighted partition | rust | 90.8 | 90.8 | 0.0 | 2.2 | 4.5 / 3.5 | 2.2 | 2.6 | 18 | 1.0 |
| SBM 20k random start K=40 | c | 91.7 | 91.7 | 0.0 | 2.2 | 3.7 / 3.3 | 2.2 | 2.2 | 15 | 0.8 |
| SBM 20k random start K=40 | rust | 91.2 | 91.2 | 0.0 | 2.6 | 3.9 / 3.6 | 2.1 | 2.3 | 17 | 0.8 |
| DBLP top5000 partition | c | 155.6 | 155.6 | 0.0 | 20.0 | 14.9 / 11.1 | 9.5 | 9.9 | 15 | 3.3 |
| DBLP top5000 partition | rust | 156.6 | 156.6 | 0.0 | 17.8 | 15.1 / 11.4 | 9.1 | 10.6 | 17 | 3.3 |
| DBLP top5000 cover M=-1 | c | 170.8 | 154.6 | 16.4 | 55.2 | 38.2 / 22.8 | 25.5 | 32.2 | 53 | 3.3 |
| DBLP top5000 cover M=-1 | rust | 169.3 | 155.6 | 13.7 | 52.4 | 38.5 / 23.0 | 22.2 | 33.9 | 55 | 3.3 |
| Amazon top5000 partition | c | 84.0 | 84.0 | 0.0 | 1.3 | 3.7 / 2.9 | 1.6 | 1.7 | 15 | 0.5 |
| Amazon top5000 partition | rust | 84.0 | 84.0 | 0.0 | 2.5 | 3.9 / 3.1 | 1.5 | 1.8 | 17 | 0.5 |
| Amazon top5000 cover M=-1 | c | 90.7 | 83.6 | 7.1 | 9.9 | 7.8 / 5.7 | 5.3 | 7.4 | 41 | 0.5 |
| Amazon top5000 cover M=-1 | rust | 90.2 | 83.9 | 5.5 | 9.3 | 8.0 / 5.9 | 4.5 | 7.7 | 48 | 0.5 |
| YouTube top5000 partition | c | 122.2 | 122.2 | 0.0 | 6.8 | 6.5 / 4.8 | 5.0 | 5.2 | 18 | 2.0 |
| YouTube top5000 partition | rust | 122.5 | 122.5 | 0.0 | 5.2 | 6.7 / 5.0 | 4.8 | 5.5 | 20 | 2.0 |
| YouTube top5000 cover M=-1 | c | 202.1 | 122.1 | 80.6 | 104.5 | 67.4 / 44.6 | 56.6 | 87.5 | 65 | 2.0 |
| YouTube top5000 cover M=-1 | rust | 210.2 | 122.7 | 87.4 | 112.2 | 67.7 / 44.8 | 50.1 | 87.8 | 60 | 2.0 |
| DBLP top5000 cover M=3 | c | 158.4 | 157.0 | 1.2 | 40.1 | 28.9 / 16.8 | 19.3 | 23.1 | 44 | 3.3 |
| DBLP top5000 cover M=3 | rust | 156.5 | 154.5 | 1.6 | 40.4 | 29.1 / 17.0 | 16.5 | 23.9 | 50 | 3.3 |
| audit DBLP M=3 cover, capped | c | 159.0 | 155.7 | 3.3 | 25.4 | 16.4 / 15.7 | 11.1 | 13.7 | 48 | - |
| audit DBLP M=3 cover, capped | rust | 156.9 | 154.9 | 2.0 | 25.0 | 16.5 / 15.8 | 11.6 | 14.1 | 58 | - |
| audit DBLP M=3 cover, uncapped | c | 157.3 | 154.2 | 3.1 | 25.2 | 15.9 / 15.2 | 11.6 | 14.2 | 46 | - |
| audit DBLP M=3 cover, uncapped | rust | 156.2 | 155.8 | 0.0 | 24.3 | 16.0 / 15.3 | 12.1 | 14.7 | 55 | - |
| audit DBLP M=-1 cover, uncapped | c | 174.8 | 155.7 | 19.0 | 30.2 | 16.4 / 15.7 | 14.5 | 17.4 | 46 | - |
| audit DBLP M=-1 cover, uncapped | rust | 176.2 | 155.5 | 21.2 | 32.3 | 16.5 / 15.8 | 14.9 | 17.8 | 57 | - |
