# Controllers

`navstack.controllers` provides trajectory-tracking controllers and dynamic
obstacle avoidance.

## Path following

```python
from navstack.controllers.navigation import get_controller

ctrl = get_controller("PurePursuit")          # or Stanley / Proportional / DWA
v, omega, idx = ctrl.compute_control((x, y, theta), path, current_idx)
```

A `GAControllerTuner` auto-tunes controller gains with a genetic algorithm.

## Dynamic obstacle avoidance (Velocity Obstacles / RVO)

`RVOPlanner` selects a collision-free velocity amid moving obstacles using
truncated velocity obstacles.

```python
from navstack.controllers.velocity_obstacles import RVOPlanner

planner = RVOPlanner(robot_radius=0.3, max_speed=1.5, time_horizon=3.0)
# obstacles: (x, y, vx, vy, radius)
v_xy = planner.compute_velocity(pos, vel, goal, obstacles)
v, omega = RVOPlanner.to_unicycle(v_xy, theta)
```

!!! note "VO vs RVO"
    Use plain VO (`reciprocal=False`, the default) for non-cooperative moving
    obstacles, so the robot takes full responsibility for avoidance. Use RVO
    (`reciprocal=True`) only when every agent runs the same planner, e.g. in a
    multi-robot swarm.

See `examples/rvo_demo.py` for a rendered example, and `navstack.controllers.swarm.simulate_circle_swap` (with `examples/swarm_demo.py`) for multi-robot coordination.

## MPC path following

`MPCPathFollower` (`navstack.controllers.mpc_follower`) tracks a reference path
by linearizing the unicycle tracking-error dynamics around the path and solving
a convex QP each step (LTV-MPC).

```python
from navstack.controllers.mpc_follower import MPCPathFollower

mpc = MPCPathFollower(horizon=12, dt=0.1, v_ref=1.2)
v, omega, idx = mpc.compute_control((x, y, theta), path, current_idx)
```

Requires the `[control]` extra (CVXPY). See `examples/mpc_demo.py`.
