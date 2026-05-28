# navstack

A mobile-robot navigation and control platform that brings classical planning,
physics simulation, modern control, and reinforcement learning under one roof.

## Modules

| Module | Contents |
|--------|----------|
| `navstack.planners` | A\*, Dijkstra, RRT, RRT\*, PRM, PSO, APF |
| `navstack.sim` | MuJoCo simulator + MJCF generator for 6 drive types |
| `navstack.controllers` | Pure Pursuit, Stanley, DWA + Velocity Obstacles (RVO) |
| `navstack.balancing` | Self-balancing Segway: LQR / MPC / SMC / pole-placement / RL |
| `navstack.perception` | LiDAR raycasting + color-blob vision |
| `navstack.gui` | PySide6 control center |

## Install

```bash
pip install -e .                    # core planners
pip install -e ".[sim,control,rl]"  # + MuJoCo sim, optimal control, RL
pip install -e ".[docs]"            # to build these docs
```

## Quick start

```bash
python examples/compare_all.py        # compare all 7 planners
python -m navstack.benchmark          # reproducible benchmark
python examples/rvo_demo.py           # dynamic obstacle avoidance
python -m navstack.gui.control_center # MuJoCo simulator GUI
```

See the sidebar for per-module documentation.
