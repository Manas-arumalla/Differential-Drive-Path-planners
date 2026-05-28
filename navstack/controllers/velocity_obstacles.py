"""Dynamic obstacle avoidance via truncated Velocity Obstacles (VO / RVO).

A reactive local planner for moving obstacles. Given the robot's state, a goal,
and a set of moving obstacles (each with a position, velocity, and radius), it
selects a collision-free velocity as close as possible to the velocity that
heads straight at the goal.

- VO (Velocity Obstacles): the robot takes full responsibility for avoidance.
- RVO (Reciprocal VO): both agents are assumed to cooperate, so the robot only
  needs to take half the avoiding action — set ``reciprocal=True``. This is the
  right mode for multi-robot / swarm scenarios where every agent runs the same
  planner.

The planner works in 2D velocity space (holonomic). Use :meth:`to_unicycle` to
convert the chosen velocity vector into differential-drive ``(v, omega)``.
"""
from __future__ import annotations

import numpy as np

Vec = np.ndarray


class RVOPlanner:
    def __init__(self, robot_radius: float = 0.3, max_speed: float = 1.5,
                 time_horizon: float = 2.0, n_speed: int = 6, n_angle: int = 36,
                 reciprocal: bool = False, safety_margin: float = 0.2):
        """
        Args:
            robot_radius: radius of the robot (m).
            max_speed: maximum holonomic speed (m/s).
            time_horizon: only obstacles colliding within this window matter (s).
            n_speed, n_angle: velocity-sample grid resolution.
            reciprocal: RVO mode (shared avoidance) for *cooperative* agents that
                run the same planner. For non-cooperative moving obstacles leave
                this False (plain VO) so the robot takes full responsibility.
        """
        self.r = float(robot_radius)
        self.v_max = float(max_speed)
        self.tau = float(time_horizon)
        self.n_speed = int(n_speed)
        self.n_angle = int(n_angle)
        self.reciprocal = bool(reciprocal)
        self.safety_margin = float(safety_margin)

    # ------------------------------------------------------------------ helpers
    def preferred_velocity(self, pos: Vec, goal: Vec, speed: float | None = None) -> Vec:
        """Velocity pointing straight at the goal at the given (or max) speed."""
        pos, goal = np.asarray(pos, float), np.asarray(goal, float)
        d = goal - pos
        n = np.linalg.norm(d)
        if n < 1e-9:
            return np.zeros(2)
        speed = self.v_max if speed is None else min(speed, self.v_max)
        # ease off within one second of the goal so we don't overshoot
        return (d / n) * min(speed, n)

    def _time_to_collision(self, rel_pos: Vec, rel_vel: Vec, combined_r: float) -> float:
        """Earliest time the moving disc breaches ``combined_r``; inf if it never does.

        rel_pos: obstacle position relative to robot.
        rel_vel: robot velocity relative to obstacle (closing velocity = -rel_vel).
        """
        # Distance(t)^2 = |rel_pos - rel_vel*t|^2  (obstacle closes on robot at -rel_vel)
        a = float(rel_vel @ rel_vel)
        if a < 1e-12:
            # no relative motion: collision only if already overlapping
            return 0.0 if float(rel_pos @ rel_pos) <= combined_r ** 2 else np.inf
        b = -2.0 * float(rel_pos @ rel_vel)
        c = float(rel_pos @ rel_pos) - combined_r ** 2
        if c <= 0.0:
            return 0.0  # already in collision
        disc = b * b - 4 * a * c
        if disc <= 0.0:
            return np.inf  # closest approach never breaches the radius
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        # earliest non-negative entry time
        if t1 >= 0:
            return t1
        if t2 >= 0:
            return 0.0
        return np.inf

    def _candidate_velocities(self, v_pref: Vec) -> list[Vec]:
        cands = [np.zeros(2), v_pref.copy()]
        pref_angle = np.arctan2(v_pref[1], v_pref[0]) if np.linalg.norm(v_pref) > 1e-9 else 0.0
        for s in np.linspace(self.v_max / self.n_speed, self.v_max, self.n_speed):
            for da in np.linspace(-np.pi, np.pi, self.n_angle, endpoint=False):
                ang = pref_angle + da
                cands.append(np.array([s * np.cos(ang), s * np.sin(ang)]))
        return cands

    # -------------------------------------------------------------------- solve
    def compute_velocity(self, pos, vel, goal, obstacles) -> Vec:
        """Pick the safest velocity closest to heading at the goal.

        Args:
            pos: robot position (x, y).
            vel: robot current velocity (vx, vy).
            goal: goal position (x, y).
            obstacles: iterable of (x, y, vx, vy, radius).
        Returns:
            chosen velocity (vx, vy).
        """
        pos = np.asarray(pos, float)
        vel = np.asarray(vel, float)
        goal = np.asarray(goal, float)
        obs = [ (np.array([o[0], o[1]], float), np.array([o[2], o[3]], float), float(o[4]))
                for o in obstacles ]

        v_pref = self.preferred_velocity(pos, goal)

        # Among velocities that stay collision-free for the whole horizon, take the
        # one closest to the goal-ward preference. If none are safe (obstacle is
        # already too close), fall back to whichever buys the most time.
        best_safe_v, best_safe_cost = None, np.inf
        fallback_v, fallback_ttc = np.zeros(2), -np.inf
        for v_cand in self._candidate_velocities(v_pref):
            if np.linalg.norm(v_cand) > self.v_max + 1e-9:
                continue
            min_ttc = np.inf
            for o_pos, o_vel, o_r in obs:
                rel_pos = o_pos - pos
                if self.reciprocal:
                    rel_vel = 2.0 * v_cand - vel - o_vel
                else:
                    rel_vel = v_cand - o_vel
                min_ttc = min(min_ttc, self._time_to_collision(
                    rel_pos, rel_vel, self.r + o_r + self.safety_margin))

            if min_ttc >= self.tau:
                cost = float(np.linalg.norm(v_cand - v_pref))
                if cost < best_safe_cost:
                    best_safe_cost, best_safe_v = cost, v_cand
            elif min_ttc > fallback_ttc:
                fallback_ttc, fallback_v = min_ttc, v_cand

        return best_safe_v if best_safe_v is not None else fallback_v

    @staticmethod
    def to_unicycle(v_vec, theta, k_omega: float = 4.0, max_omega: float = 3.0):
        """Map a desired holonomic velocity to differential-drive (v, omega)."""
        v_vec = np.asarray(v_vec, float)
        speed = float(np.linalg.norm(v_vec))
        if speed < 1e-6:
            return 0.0, 0.0
        desired_heading = np.arctan2(v_vec[1], v_vec[0])
        err = np.arctan2(np.sin(desired_heading - theta), np.cos(desired_heading - theta))
        omega = float(np.clip(k_omega * err, -max_omega, max_omega))
        # slow down when we must turn hard
        v = speed * max(0.0, np.cos(err))
        return v, omega
