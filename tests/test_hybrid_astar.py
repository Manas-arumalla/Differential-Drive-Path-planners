"""Tests for the kinodynamic Hybrid A* planner used by the Trail demo."""
import math
import os
import sys

import numpy as np
import pytest

# The Trail demo is a standalone script package; add it to the path.
TRAIL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Trail")
if TRAIL_DIR not in sys.path:
    sys.path.insert(0, TRAIL_DIR)

from hybrid_astar import HybridAStar  # noqa: E402

RES = 0.5
STEP = 0.18


@pytest.fixture
def grid():
    # Same obstacle layout as Trail/main.py's create_test_map (80m x 60m at 0.5 res).
    g = np.zeros((120, 160), dtype=bool)
    g[6:9, 6:42] = True
    g[22:24, 12:72] = True
    g[36:39, 12:62] = True
    g[11:19, 46:49] = True
    g[41:56, 31:34] = True
    return g


def _planner(grid):
    return HybridAStar(grid, resolution=RES, vehicle_length=0.7, step_size=STEP,
                       theta_res_deg=10, radius=0.18)


def test_finds_short_segment(grid):
    path = _planner(grid).plan((2.0, 2.0, 0.0), (5.0, 5.0), max_iter=60000, goal_tolerance=0.6)
    assert path is not None and len(path) >= 2
    assert path[0] == (2.0, 2.0, 0.0)
    assert math.hypot(path[-1][0] - 5.0, path[-1][1] - 5.0) <= 0.6


def test_edges_are_kinematically_feasible(grid):
    """Each motion primitive advances by exactly one step length (bicycle integration)."""
    path = _planner(grid).plan((2.0, 2.0, 0.0), (9.0, 9.0), max_iter=60000, goal_tolerance=0.6)
    assert path is not None
    for i in range(len(path) - 1):
        d = math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        assert d <= STEP + 1e-6


def test_rejects_goal_inside_obstacle(grid):
    # (24, 18) lies inside the g[36:39, 12:62] wall.
    path = _planner(grid).plan((20.0, 15.0, 0.0), (24.0, 18.0), max_iter=20000, goal_tolerance=0.4)
    assert path is None


def test_rejects_start_in_collision(grid):
    # Start placed inside the g[6:9, 6:42] wall (cell ~ x=10m, y=3.5m).
    path = _planner(grid).plan((10.0, 3.5, 0.0), (20.0, 20.0), max_iter=20000)
    assert path is None
