"""Headless closed-loop physics path following for a differential-drive base.

Drives a MuJoCo differential-drive robot along a waypoint path using a robust
"turn-then-go" controller with calibrated wheel torques, stepping real physics
(no rendering). An optional per-step callback lets callers capture frames.
"""
from __future__ import annotations

import numpy as np


def _yaw(quat):  # mujoco quat order (w, x, y, z)
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def drive_path(model, data, path, *, lookahead=0.5, goal_tol=0.3, max_steps=8000,
               turn_thresh=0.35, drive_torque=1.3, turn_torque=1.2, steer_gain=2.0,
               on_step=None):
    """Drive the robot (free joint + 2 wheel motors) along ``path`` under physics.

    Args:
        model, data: a loaded MuJoCo differential-drive model and its data.
        path: iterable of (x, y) waypoints in world coordinates.
        on_step(model, data, step): optional callback invoked each physics step
            (use it to render frames).
    Returns:
        dict: {'reached': bool, 'trajectory': (T,3) array of (x,y,theta), 'steps': int}.
    """
    import mujoco

    path = np.asarray(path, float)
    goal = path[-1]
    idx = 0
    traj = []
    reached = False
    steps = max_steps
    for step in range(max_steps):
        x, y = float(data.qpos[0]), float(data.qpos[1])
        th = _yaw(data.qpos[3:7])
        if not np.isfinite(x) or not np.isfinite(y):
            break
        traj.append((x, y, th))

        while idx < len(path) - 1 and np.hypot(path[idx][0] - x, path[idx][1] - y) < lookahead:
            idx += 1
        tgt = path[idx]
        herr = _wrap(np.arctan2(tgt[1] - y, tgt[0] - x) - th)
        if abs(herr) > turn_thresh:
            base, diff = 0.0, turn_torque * np.sign(herr)   # turn (nearly) in place
        else:
            base, diff = drive_torque, steer_gain * herr     # drive forward + steer
        data.ctrl[0] = np.clip(base - diff, -5, 5)
        data.ctrl[1] = np.clip(base + diff, -5, 5)
        mujoco.mj_step(model, data)

        if on_step is not None:
            on_step(model, data, step)

        if np.hypot(x - goal[0], y - goal[1]) < goal_tol:
            reached, steps = True, step + 1
            break

    return {"reached": reached, "trajectory": np.array(traj), "steps": steps}
