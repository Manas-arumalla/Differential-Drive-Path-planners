"""Smoke tests for the 7 path planners.

Every planner must return a path that starts at the start, ends at the goal, and
(for the deterministic grid planners) is collision-free for the planning radius.
Sampling/optimization planners are seeded and checked for goal reachability.
"""
import random

import numpy as np
import pytest

from navstack.environment import create_demo_environment
from navstack.planners import (
    AStarPlanner,
    DijkstraPlanner,
    RRTPlanner,
    RRTStarPlanner,
    PRMPlanner,
    PSOPlanner,
    APFPlanner,
)

START = (3.0, 3.0)
GOAL = (45.0, 45.0)
ROBOT_RADIUS = 0.5


@pytest.fixture(autouse=True)
def _seed():
    """Make the sampling/optimization planners reproducible."""
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def env():
    return create_demo_environment()


def _endpoints_ok(path, start, goal, tol=1.5):
    assert path is not None and len(path) >= 2
    assert np.hypot(path[0][0] - start[0], path[0][1] - start[1]) <= tol
    assert np.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) <= tol


def _is_collision_free(env, path, robot_radius):
    return all(
        env.is_path_valid(path[i], path[i + 1], robot_radius)
        for i in range(len(path) - 1)
    )


# --- deterministic grid planners: strict collision-free guarantee -------------
@pytest.mark.parametrize("Planner", [AStarPlanner, DijkstraPlanner])
def test_grid_planner_returns_collision_free_path(env, Planner):
    planner = Planner(env, robot_radius=ROBOT_RADIUS)
    path = planner.plan(START, GOAL)
    _endpoints_ok(path, START, GOAL)
    assert _is_collision_free(env, path, ROBOT_RADIUS)


# --- sampling / optimization planners: seeded goal-reachability ---------------
def test_rrt_reaches_goal(env):
    path = RRTPlanner(env, robot_radius=ROBOT_RADIUS, step_size=2.0, max_iterations=5000).plan(START, GOAL)
    _endpoints_ok(path, START, GOAL)


def test_rrt_star_reaches_goal(env):
    path = RRTStarPlanner(env, robot_radius=ROBOT_RADIUS, step_size=2.0, max_iterations=3000).plan(START, GOAL)
    _endpoints_ok(path, START, GOAL)


def test_prm_reaches_goal(env):
    path = PRMPlanner(env, robot_radius=ROBOT_RADIUS, num_samples=300, k_neighbors=10).plan(START, GOAL)
    _endpoints_ok(path, START, GOAL)


def test_pso_reaches_goal(env):
    path = PSOPlanner(env, robot_radius=ROBOT_RADIUS, num_particles=50, max_iterations=80).plan(START, GOAL)
    _endpoints_ok(path, START, GOAL)


def test_apf_reaches_goal(env):
    path = APFPlanner(env, robot_radius=ROBOT_RADIUS, attractive_gain=5.0, repulsive_gain=150.0).plan(START, GOAL)
    _endpoints_ok(path, START, GOAL)


# --- invalid queries should fail gracefully (return None, not crash) ----------
def test_astar_rejects_goal_inside_obstacle(env):
    # (15, 35) is the center of a circular obstacle in the demo environment.
    assert AStarPlanner(env, robot_radius=ROBOT_RADIUS).plan(START, (15.0, 35.0)) is None
