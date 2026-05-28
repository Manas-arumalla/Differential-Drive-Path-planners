# benchmark.py
"""
Reproducible benchmark harness for the path planners.

Runs every planner across a set of seeded scenarios and records standardized
metrics, then writes a CSV and a summary plot and prints a leaderboard.

    python benchmark.py            # run all planners on all scenarios
    python benchmark.py --no-plot  # skip the summary figure

Metrics per run:
    success        : whether a path was returned
    length_m       : total Euclidean path length (meters)
    smoothness_deg : mean absolute heading change between segments (deg/waypoint)
    min_clear_m    : minimum obstacle clearance along the path (meters)
    time_s         : wall-clock planning time (seconds)
    waypoints      : number of waypoints
"""
import argparse
import csv
import os
import random
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from navstack.environment import Environment, create_demo_environment
from navstack.planners import (
    AStarPlanner,
    DijkstraPlanner,
    RRTPlanner,
    RRTStarPlanner,
    PRMPlanner,
    PSOPlanner,
    APFPlanner,
)

ROBOT_RADIUS = 0.5
GOAL_TOL = 2.0  # a run only counts as success if the path ends within this of the goal
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")

# Planner factories use the same parameters as the comparison demo for comparability.
PLANNERS = {
    "A*": lambda env: AStarPlanner(env, robot_radius=ROBOT_RADIUS),
    "Dijkstra": lambda env: DijkstraPlanner(env, robot_radius=ROBOT_RADIUS),
    "RRT": lambda env: RRTPlanner(env, robot_radius=ROBOT_RADIUS, step_size=2.0, max_iterations=5000),
    "RRT*": lambda env: RRTStarPlanner(env, robot_radius=ROBOT_RADIUS, step_size=2.0, max_iterations=3000),
    "PRM": lambda env: PRMPlanner(env, robot_radius=ROBOT_RADIUS, num_samples=300, k_neighbors=10),
    "PSO": lambda env: PSOPlanner(env, robot_radius=ROBOT_RADIUS, num_particles=50, max_iterations=80),
    "APF": lambda env: APFPlanner(env, robot_radius=ROBOT_RADIUS, attractive_gain=5.0, repulsive_gain=150.0),
}


def _random_env(seed, n_circles=8):
    """A seeded sparse random map on the same 50x50 world as the demo."""
    rng = random.Random(seed)
    env = Environment(width=50.0, height=50.0, resolution=0.5)
    for _ in range(n_circles):
        env.add_circular_obstacle(rng.uniform(10, 40), rng.uniform(10, 40), rng.uniform(2, 4))
    return env


def make_scenarios(n_random=8):
    """List of (name, env, start, goal): two fixed demo maps + N seeded random maps."""
    scenarios = [
        ("demo_diag", create_demo_environment(), (3.0, 3.0), (45.0, 45.0)),
        ("demo_anti", create_demo_environment(), (5.0, 45.0), (45.0, 5.0)),
    ]
    corners = [((3.0, 3.0), (47.0, 47.0)), ((3.0, 47.0), (47.0, 3.0))]
    for seed in range(1, n_random + 1):
        start, goal = corners[seed % 2]
        scenarios.append((f"random_s{seed}", _random_env(seed), start, goal))
    return scenarios


def path_length(path):
    return sum(np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


def smoothness_deg(path):
    """Mean absolute heading change per waypoint (lower is smoother)."""
    if len(path) < 3:
        return 0.0
    angles = []
    for i in range(1, len(path) - 1):
        a = np.array(path[i]) - np.array(path[i - 1])
        b = np.array(path[i + 1]) - np.array(path[i])
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        cosang = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosang)))
    return float(np.mean(angles)) if angles else 0.0


def min_clearance(env, path):
    """Minimum distance from the path to the nearest obstacle, via a distance transform."""
    clearance = distance_transform_edt(~env.grid) * env.resolution
    vals = []
    for x, y in path:
        gx, gy = env.world_to_grid(x, y)
        gx = min(max(gx, 0), env.grid_width - 1)
        gy = min(max(gy, 0), env.grid_height - 1)
        vals.append(clearance[gy, gx])
    return float(min(vals)) if vals else 0.0


def run():
    scenarios = make_scenarios()
    rows = []
    for sc_name, env, start, goal in scenarios:
        for p_name, factory in PLANNERS.items():
            random.seed(0)
            np.random.seed(0)
            planner = factory(env)
            t0 = time.time()
            path = planner.plan(start, goal)
            dt = time.time() - t0
            reached = bool(path) and len(path) >= 2 and \
                np.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) <= GOAL_TOL
            if reached:
                rows.append({
                    "scenario": sc_name, "planner": p_name, "success": 1,
                    "length_m": round(path_length(path), 2),
                    "smoothness_deg": round(smoothness_deg(path), 2),
                    "min_clear_m": round(min_clearance(env, path), 2),
                    "time_s": round(dt, 4), "waypoints": len(path),
                })
            else:
                rows.append({
                    "scenario": sc_name, "planner": p_name, "success": 0,
                    "length_m": "", "smoothness_deg": "", "min_clear_m": "",
                    "time_s": round(dt, 4), "waypoints": 0,
                })
    return scenarios, rows


def write_csv(rows, path):
    fields = ["scenario", "planner", "success", "length_m", "smoothness_deg",
              "min_clear_m", "time_s", "waypoints"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_leaderboard(rows):
    """Print mean +/- std (over successful runs) so variance across maps is visible."""
    names = list(PLANNERS.keys())
    n_scen = len({r["scenario"] for r in rows})
    print("\n" + "=" * 86)
    print(f"BENCHMARK LEADERBOARD  ({n_scen} scenarios; mean +/- std over successful runs)")
    print("=" * 86)
    print(f"{'Planner':<10}{'Success':<9}{'Length (m)':<18}{'Smooth (deg)':<16}{'Clear (m)':<14}{'Time (s)':<16}")
    print("-" * 86)

    def ms(rows_ok, col, prec=2):
        vals = [r[col] for r in rows_ok]
        return f"{np.mean(vals):.{prec}f}+/-{np.std(vals):.{prec}f}"

    for n in names:
        runs = [r for r in rows if r["planner"] == n]
        ok = [r for r in runs if r["success"] == 1]
        rate = f"{len(ok)}/{len(runs)}"
        if ok:
            print(f"{n:<10}{rate:<9}"
                  f"{ms(ok, 'length_m'):<18}"
                  f"{ms(ok, 'smoothness_deg'):<16}"
                  f"{ms(ok, 'min_clear_m'):<14}"
                  f"{ms(ok, 'time_s', 3):<16}")
        else:
            print(f"{n:<10}{rate:<9}{'-':<18}{'-':<16}{'-':<14}{'-':<16}")
    print("=" * 86)


def plot_summary(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(PLANNERS.keys())
    avg_len, avg_t = [], []
    for n in names:
        ok = [r for r in rows if r["planner"] == n and r["success"] == 1]
        avg_len.append(np.mean([r["length_m"] for r in ok]) if ok else 0)
        avg_t.append(np.mean([r["time_s"] for r in ok]) if ok else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(names, avg_len, color="#00D4FF")
    ax1.set_title("Average path length (m)")
    ax1.set_ylabel("meters")
    ax2.bar(names, avg_t, color="#FF6B6B")
    ax2.set_title("Average planning time (s)")
    ax2.set_ylabel("seconds")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved summary plot to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true", help="skip the summary figure")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    scenarios, rows = run()

    csv_path = os.path.join(OUT_DIR, "results.csv")
    write_csv(rows, csv_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print_leaderboard(rows)
    if not args.no_plot:
        plot_summary(rows, os.path.join(OUT_DIR, "summary.png"))


if __name__ == "__main__":
    main()
