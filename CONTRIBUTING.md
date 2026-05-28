# Contributing / Development Guide

## Setup

```bash
pip install -e .                      # core planners (numpy/scipy/matplotlib)
pip install -e ".[dev]"               # + pytest
pip install -e ".[sim,control,rl]"    # + MuJoCo sim, optimal control, RL
```

## Tests

```bash
pytest -q
```

Tests that depend on heavy optional packages (mujoco, cvxpy, control,
stable-baselines3, PySide6) skip automatically via `pytest.importorskip`, so the
suite stays green on a minimal install.

## Project layout

```
navstack/
├── planners/      # A*, Dijkstra, RRT, RRT*, PRM, PSO, APF, semantic A*
├── controllers/   # Pure Pursuit, Stanley, DWA, MPC, Velocity Obstacles/RVO, swarm
├── sim/           # MuJoCo simulator + procedural MJCF generator (6 drive types)
├── balancing/     # self-balancing Segway: control, dynamics, ROA, Gym env, RL
├── perception/    # LiDAR + vision
├── gui/           # PySide6 control center
├── environment.py # occupancy grid + collision checking
├── robot.py       # differential-drive model
├── visualize.py   # dark-theme plotting helpers
├── benchmark.py   # benchmark harness
└── analytics.py   # benchmark dashboard
examples/          # runnable demos
Trail/             # kinodynamic Hybrid A* + DWA demo with moving obstacles
dashboard/         # Streamlit explorer
tests/             # pytest suite
```

## Conventions

- Coordinates are world meters `(x, y)`; grid indices are `(gx, gy)` — always convert
  through `Environment`, never hardcode the resolution.
- Planners share one interface: `Planner(env, robot_radius=..., **kwargs)` and
  `plan(start, goal) -> [(x, y), ...] | None`.
- Trajectory-tracking controllers share `compute_control(pose, path, idx) -> (v, omega, idx)`.
- Headings are normalized to `[-pi, pi]`.
- Run example scripts from the repo root (or after `pip install -e .`). For headless
  GUI imports set `QT_QPA_PLATFORM=offscreen`.

## Architecture notes

- **MuJoCo sim:** adding or changing a drive type means updating both the generator
  function in `navstack/sim/model_generator.py` and the control branch in
  `navstack/sim/mujoco_sim.py`. Bundled models resolve via
  `navstack.balancing.model_path()`.
- **Reusing visuals:** prefer the helpers in `navstack/visualize.py`
  (`ALGORITHM_COLORS`, `plot_path`, `compare_algorithms`) over new matplotlib styling.
- **Velocity Obstacles:** `RVOPlanner` works in velocity space; use plain VO for
  non-cooperative obstacles and reciprocal (RVO) mode for cooperative multi-robot.
