"""Model-Predictive path-following controller for a differential-drive base.

Linearizes the unicycle tracking-error dynamics around a moving reference drawn
from the path and solves a small convex QP each step (CVXPY) for the optimal
velocity/turn-rate corrections. This is the standard LTV-MPC tracker:

    e_x+ = e_x + dt ( w_r e_y - dv )
    e_y+ = e_y + dt (-w_r e_x + v_r e_theta )
    e_th+ = e_theta - dt dw

with control deviations (dv, dw) = (v - v_r, w - w_r) about the reference
(v_r, w_r). Provides the same interface as the other controllers in
``navstack.controllers.navigation``.
"""
from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:  # pragma: no cover
    _HAS_CVXPY = False


def _normalize(a):
    return np.arctan2(np.sin(a), np.cos(a))


class MPCPathFollower:
    def __init__(self, horizon=12, dt=0.1, v_ref=1.0,
                 q_pos=12.0, q_theta=2.0, r_v=1.0, r_omega=1.0,
                 v_max=1.5, omega_max=2.0, goal_tol=0.4):
        if not _HAS_CVXPY:
            raise ImportError("MPCPathFollower requires cvxpy (pip install -e '.[control]').")
        self.N = int(horizon)
        self.dt = float(dt)
        self.v_ref = float(v_ref)
        self.Q = np.diag([q_pos, q_pos, q_theta])
        self.R = np.diag([r_v, r_omega])
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.goal_tol = float(goal_tol)

    # --------------------------------------------------------------- reference
    def _reference(self, pose, path):
        """Build N+1 reference poses spaced v_ref*dt along the path from the
        nearest point, plus per-stage (v_r, w_r)."""
        pts = np.asarray(path, float)[:, :2]
        d = np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1])
        i0 = int(np.argmin(d))

        # cumulative arc length from the nearest point forward
        seg = np.hypot(np.diff(pts[i0:, 0]), np.diff(pts[i0:, 1]))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        fwd = pts[i0:]
        step = self.v_ref * self.dt

        ref_xy = []
        for k in range(self.N + 1):
            target_s = k * step
            if target_s >= s[-1]:
                ref_xy.append(fwd[-1])
            else:
                j = int(np.searchsorted(s, target_s, side="right"))
                j = min(max(j, 1), len(s) - 1)
                t = (target_s - s[j - 1]) / max(s[j] - s[j - 1], 1e-9)
                ref_xy.append(fwd[j - 1] + t * (fwd[j] - fwd[j - 1]))
        ref_xy = np.array(ref_xy)

        ref_th = np.zeros(self.N + 1)
        for k in range(self.N):
            dxy = ref_xy[k + 1] - ref_xy[k]
            ref_th[k] = np.arctan2(dxy[1], dxy[0]) if np.hypot(*dxy) > 1e-6 else (ref_th[k - 1] if k else pose[2])
        ref_th[-1] = ref_th[-2]

        v_r = np.full(self.N, self.v_ref)
        w_r = np.array([_normalize(ref_th[k + 1] - ref_th[k]) / self.dt for k in range(self.N)])
        # if the reference has run out (at goal), command a stop
        if s[-1] < step:
            v_r[:] = 0.0
        return ref_xy, ref_th, v_r, w_r, i0

    # ----------------------------------------------------------------- control
    def compute_control(self, pose, path, current_idx=0):
        if path is None or len(path) < 2:
            return 0.0, 0.0, current_idx
        x, y, th = pose
        ref_xy, ref_th, v_r, w_r, i0 = self._reference(pose, path)

        # initial tracking error in the robot frame
        dx, dy = ref_xy[0, 0] - x, ref_xy[0, 1] - y
        e0 = np.array([np.cos(th) * dx + np.sin(th) * dy,
                       -np.sin(th) * dx + np.cos(th) * dy,
                       _normalize(ref_th[0] - th)])

        e = cp.Variable((3, self.N + 1))
        u = cp.Variable((2, self.N))  # [dv, dw]
        cost = 0
        cons = [e[:, 0] == e0]
        dt = self.dt
        for k in range(self.N):
            A = np.array([[1, w_r[k] * dt, 0],
                          [-w_r[k] * dt, 1, v_r[k] * dt],
                          [0, 0, 1]])
            B = np.array([[-dt, 0], [0, 0], [0, -dt]])
            cons += [e[:, k + 1] == A @ e[:, k] + B @ u[:, k]]
            cons += [v_r[k] + u[0, k] <= self.v_max, v_r[k] + u[0, k] >= -self.v_max,
                     w_r[k] + u[1, k] <= self.omega_max, w_r[k] + u[1, k] >= -self.omega_max]
            cost += cp.quad_form(e[:, k], self.Q) + cp.quad_form(u[:, k], self.R)
        cost += cp.quad_form(e[:, self.N], self.Q)

        prob = cp.Problem(cp.Minimize(cost), cons)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
        except Exception:
            prob.solve(solver=cp.ECOS)

        if u.value is None:
            return 0.0, 0.0, i0
        v = float(np.clip(v_r[0] + u.value[0, 0], -self.v_max, self.v_max))
        w = float(np.clip(w_r[0] + u.value[1, 0], -self.omega_max, self.omega_max))
        return v, w, i0
