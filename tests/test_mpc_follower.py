"""Tests for the LTV-MPC path-following controller."""
import numpy as np
import pytest


def _follow(path, steps=400, dt=0.1):
    pytest.importorskip("cvxpy")
    from navstack.controllers.mpc_follower import MPCPathFollower
    from navstack.robot import DifferentialDriveRobot

    mpc = MPCPathFollower(horizon=12, dt=dt, v_ref=1.2, v_max=1.6, omega_max=2.5)
    robot = DifferentialDriveRobot(x=path[0][0], y=path[0][1], theta=0.0,
                                   max_speed=1.6, max_omega=2.5)
    pts = np.array(path, float)
    cte = []
    for _ in range(steps):
        v, w, _ = mpc.compute_control((robot.x, robot.y, robot.theta), path)
        robot.step(v, w, dt=dt)
        cte.append(float(np.min(np.hypot(pts[:, 0] - robot.x, pts[:, 1] - robot.y))))
        if np.hypot(robot.x - path[-1][0], robot.y - path[-1][1]) < 0.4:
            break
    final = float(np.hypot(robot.x - path[-1][0], robot.y - path[-1][1]))
    return final, float(np.mean(cte))


def test_tracks_scurve():
    xs = np.linspace(0, 10, 50)
    path = [(float(x), float(2 * np.sin(0.5 * x))) for x in xs]
    final, mean_cte = _follow(path)
    assert final < 0.5
    assert mean_cte < 0.3


def test_tracks_straight_line():
    path = [(float(x), 0.0) for x in np.linspace(0, 8, 20)]
    final, mean_cte = _follow(path)
    assert final < 0.5
    assert mean_cte < 0.2


def test_available_via_controller_factory():
    pytest.importorskip("cvxpy")
    from navstack.controllers.navigation import get_controller
    ctrl = get_controller("MPC", horizon=10, v_ref=1.0)
    path = [(float(x), 0.0) for x in np.linspace(0, 5, 15)]
    v, w, idx = ctrl.compute_control((0.0, 0.0, 0.0), path, 0)
    assert np.isfinite(v) and np.isfinite(w)
