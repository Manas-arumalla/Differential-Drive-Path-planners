"""Tests for the Velocity-Obstacles dynamic obstacle avoidance planner."""
import numpy as np

from navstack.controllers.velocity_obstacles import RVOPlanner


def _simulate(obstacles_init, reciprocal=False, steps=300, dt=0.1):
    planner = RVOPlanner(robot_radius=0.3, max_speed=1.5, time_horizon=3.0,
                         reciprocal=reciprocal, safety_margin=0.2)
    pos = np.array([0.0, 0.0]); vel = np.array([0.0, 0.0]); goal = np.array([10.0, 0.0])
    obs = [list(o) for o in obstacles_init]
    min_clear = np.inf
    reached = None
    for k in range(steps):
        v = planner.compute_velocity(pos, vel, goal, [tuple(o) for o in obs])
        vel = v
        pos = pos + vel * dt
        for o in obs:
            o[0] += o[2] * dt; o[1] += o[3] * dt
            min_clear = min(min_clear, np.hypot(pos[0] - o[0], pos[1] - o[1]) - (0.3 + o[4]))
        if np.hypot(*(goal - pos)) < 0.4 and reached is None:
            reached = k
    return min_clear, reached


def test_reaches_goal_without_obstacles():
    _, reached = _simulate([])
    assert reached is not None


def test_avoids_crossing_obstacle():
    min_clear, reached = _simulate([(5.0, 5.0, 0.0, -1.0, 0.4)])
    assert min_clear > 0.0
    assert reached is not None


def test_avoids_head_on_obstacle():
    min_clear, _ = _simulate([(8.0, 0.0, -1.0, 0.0, 0.4)])
    assert min_clear > 0.0


def test_avoids_two_obstacles():
    min_clear, _ = _simulate([(5.0, 4.0, 0.0, -1.0, 0.4), (6.0, -4.0, 0.0, 1.0, 0.4)])
    assert min_clear > 0.0


def test_preferred_velocity_points_at_goal():
    p = RVOPlanner(max_speed=1.5)
    v = p.preferred_velocity([0, 0], [10, 0])
    assert v[0] > 0 and abs(v[1]) < 1e-9


def test_to_unicycle_zero_when_stopped():
    v, w = RVOPlanner.to_unicycle([0.0, 0.0], 0.0)
    assert v == 0.0 and w == 0.0
