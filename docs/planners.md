# Planners

All global planners share one interface:

```python
from navstack.environment import create_demo_environment
from navstack.planners import AStarPlanner

env = create_demo_environment()
planner = AStarPlanner(env, robot_radius=0.5)
path = planner.plan((3, 3), (45, 45))   # -> [(x, y), ...] or None
```

| Algorithm | Type | Optimal | Notes |
|-----------|------|---------|-------|
| A\*       | Grid | yes | Euclidean heuristic, 8-connected |
| Dijkstra  | Grid | yes | Uniform-cost, no heuristic |
| RRT       | Sampling | no | Fast, goal-biased |
| RRT\*     | Sampling | asymptotic | Rewiring for optimality |
| PRM       | Sampling | local | Reusable roadmap |
| PSO       | Optimization | local | Smooth spline paths |
| APF       | Reactive | no | Potential fields; can trap in local minima |

The `Environment` (`navstack.environment`) is a 2D occupancy grid with
world↔grid conversion and robot-radius-inflated collision checks
(`is_valid`, `is_path_valid`).

## Semantic / terrain-aware A\*

`SemanticAStarPlanner` minimizes a *terrain-weighted* path cost using a
`SemanticCostmap` of per-region traversal-cost multipliers, so it prefers cheap
terrain (road) and routes around expensive terrain (grass/mud) even when that is
geometrically longer.

```python
from navstack.planners import SemanticCostmap, SemanticAStarPlanner

cm = SemanticCostmap(env)
cm.add_rect_cost(10, 10, 10, 10, cost=8.0)   # expensive "mud" block
path = SemanticAStarPlanner(env, cm, robot_radius=0.4).plan((3, 3), (27, 27))
```

See `examples/semantic_demo.py`.

## Kinodynamic Hybrid A\*

`Trail/hybrid_astar.py` adds a kinematic-bicycle Hybrid A\* whose motion
primitives produce feasible, drivable paths. `Trail/main.py` wires it into a
hierarchical Hybrid A\* + DWA pipeline with moving obstacles.
