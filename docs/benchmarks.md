# Benchmarks

The harness runs every planner across 10 scenarios (2 fixed demo maps + 8 seeded
random maps) and records standardized metrics.

```bash
python -m navstack.benchmark   # -> benchmarks/results.csv + benchmarks/summary.png
python -m navstack.analytics   # -> media/benchmark_dashboard.png (radar + panels)
```

## Metrics

| Metric | Meaning | Better |
|--------|---------|--------|
| success | reached the goal | higher |
| length_m | total path length | lower |
| smoothness_deg | mean heading change per waypoint | lower |
| min_clear_m | minimum obstacle clearance | higher |
| time_s | wall-clock planning time | lower |

A run only counts as a success if the path actually ends within tolerance of the
goal — so reactive planners that stall in local minima are reported honestly.

## Results

Mean ± std over successful runs across 10 scenarios (times are machine-dependent):

| Planner | Success | Length (m) | Smoothness (°/wp) | Clearance (m) | Time (s) |
|---------|:-------:|:----------:|:-----------------:|:-------------:|:--------:|
| A\*      | 10/10 | 65.07 ± 2.36 | 9.71 ± 3.78  | 1.01 ± 0.04 | 0.047 |
| Dijkstra | 10/10 | 65.07 ± 2.36 | 4.29 ± 1.36  | 1.00 ± 0.00 | 0.160 |
| RRT      | 10/10 | 78.78 ± 7.00 | 30.45 ± 4.95 | 1.50 ± 1.34 | 0.028 |
| RRT\*    | 10/10 | 65.87 ± 1.67 | 16.29 ± 5.54 | 1.18 ± 0.16 | 5.071 |
| PRM      | 10/10 | 66.58 ± 1.88 | 18.80 ± 6.59 | 1.46 ± 0.38 | 0.162 |
| PSO      | 10/10 | 62.72 ± 1.73 | 1.23 ± 0.97  | 1.01 ± 0.04 | 2.707 |
| APF      |  7/10 | 68.98 ± 2.52 | 4.97 ± 6.19  | 2.39 ± 0.21 | 0.059 |
