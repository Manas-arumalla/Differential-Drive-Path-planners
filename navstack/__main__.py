"""Unified command-line interface for navstack.

    navstack version
    navstack plan --algo A* --start 3,3 --goal 45,45 [--save out.png]
    navstack benchmark [--no-plot]
    navstack dashboard
    navstack gui

(Equivalent to `python -m navstack <command>`.)
"""
import argparse
import sys
import time

import numpy as np

from navstack import __version__
from navstack.environment import create_demo_environment
from navstack.planners import (
    AStarPlanner, DijkstraPlanner, RRTPlanner, RRTStarPlanner,
    PRMPlanner, PSOPlanner, APFPlanner,
)

_PLANNERS = {
    "A*": lambda env, r: AStarPlanner(env, robot_radius=r),
    "Dijkstra": lambda env, r: DijkstraPlanner(env, robot_radius=r),
    "RRT": lambda env, r: RRTPlanner(env, robot_radius=r, step_size=2.0, max_iterations=5000),
    "RRT*": lambda env, r: RRTStarPlanner(env, robot_radius=r, step_size=2.0, max_iterations=3000),
    "PRM": lambda env, r: PRMPlanner(env, robot_radius=r, num_samples=300, k_neighbors=10),
    "PSO": lambda env, r: PSOPlanner(env, robot_radius=r, num_particles=50, max_iterations=80),
    "APF": lambda env, r: APFPlanner(env, robot_radius=r, attractive_gain=5.0, repulsive_gain=150.0),
}


def _xy(text):
    x, y = text.split(",")
    return float(x), float(y)


def _cmd_plan(args):
    if args.algo not in _PLANNERS:
        print(f"Unknown planner '{args.algo}'. Choices: {', '.join(_PLANNERS)}")
        return 1
    env = create_demo_environment()
    planner = _PLANNERS[args.algo](env, args.radius)
    t0 = time.time()
    path = planner.plan(args.start, args.goal)
    dt = time.time() - t0
    if not path:
        print(f"{args.algo}: no path found.")
        return 1
    length = sum(np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                 for i in range(len(path) - 1))
    print(f"{args.algo}: length={length:.2f} m, waypoints={len(path)}, time={dt:.4f} s")
    if args.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from navstack.visualize import plot_environment, plot_start_goal, plot_path
        fig, ax = plt.subplots(figsize=(9, 9))
        plot_environment(ax, env, f"{args.algo}")
        plot_path(ax, path, algorithm=args.algo, label=f"{length:.1f} m")
        plot_start_goal(ax, args.start, args.goal)
        ax.legend(loc="upper right")
        fig.savefig(args.save, dpi=150, facecolor="#1A202C", bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    return 0


def _cmd_benchmark(args):
    from navstack import benchmark
    sys.argv = ["navstack-benchmark"] + (["--no-plot"] if args.no_plot else [])
    benchmark.main()
    return 0


def _cmd_dashboard(args):
    from navstack.analytics import make_dashboard
    make_dashboard()
    return 0


def _cmd_gui(args):
    from navstack.gui.control_center import main as gui_main
    gui_main()
    return 0


def _cmd_version(args):
    print(f"navstack {__version__}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="navstack", description="Mobile-robot navigation & control platform")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print the version").set_defaults(func=_cmd_version)

    pp = sub.add_parser("plan", help="run a planner on the demo map")
    pp.add_argument("--algo", default="A*", help="planner name (A*, Dijkstra, RRT, RRT*, PRM, PSO, APF)")
    pp.add_argument("--start", type=_xy, default=(3.0, 3.0), help="x,y")
    pp.add_argument("--goal", type=_xy, default=(45.0, 45.0), help="x,y")
    pp.add_argument("--radius", type=float, default=0.5)
    pp.add_argument("--save", help="save a figure to this path")
    pp.set_defaults(func=_cmd_plan)

    pb = sub.add_parser("benchmark", help="run the benchmark harness")
    pb.add_argument("--no-plot", action="store_true")
    pb.set_defaults(func=_cmd_benchmark)

    sub.add_parser("dashboard", help="render the analytics dashboard").set_defaults(func=_cmd_dashboard)
    sub.add_parser("gui", help="launch the PySide6 control center").set_defaults(func=_cmd_gui)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
