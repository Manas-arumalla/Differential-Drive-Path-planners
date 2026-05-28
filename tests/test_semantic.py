"""Tests for semantic / terrain-aware A* planning."""
import numpy as np

from navstack.environment import Environment
from navstack.planners.astar_planner import AStarPlanner
from navstack.planners.semantic_astar import SemanticAStarPlanner, SemanticCostmap

START, GOAL = (3.0, 3.0), (27.0, 27.0)


def _cells_in_block(path, x0, y0, x1, y1):
    return sum(1 for x, y in path if x0 <= x <= x1 and y0 <= y <= y1)


def test_semantic_routes_around_expensive_terrain():
    env = Environment(30.0, 30.0, 0.3)
    cm = SemanticCostmap(env)
    cm.add_rect_cost(10, 10, 10, 10, 10.0)  # expensive block over the direct diagonal

    sem = SemanticAStarPlanner(env, cm, robot_radius=0.3).plan(START, GOAL)
    geo = AStarPlanner(env, robot_radius=0.3).plan(START, GOAL)
    assert sem is not None and geo is not None

    sem_mud = _cells_in_block(sem, 10, 10, 20, 20)
    geo_mud = _cells_in_block(geo, 10, 10, 20, 20)
    assert geo_mud > 0, "geometric A* should cut through the block"
    assert sem_mud == 0, "semantic A* should avoid the expensive block entirely"


def test_uniform_costmap_matches_geometric_endpoints():
    env = Environment(30.0, 30.0, 0.3)
    cm = SemanticCostmap(env)  # all ones -> behaves like plain A*
    sem = SemanticAStarPlanner(env, cm, robot_radius=0.3).plan(START, GOAL)
    assert sem is not None
    assert np.hypot(sem[0][0] - START[0], sem[0][1] - START[1]) < 1.0
    assert np.hypot(sem[-1][0] - GOAL[0], sem[-1][1] - GOAL[1]) < 1.0
