# hybrid_astar.py
"""
Kinodynamic Hybrid A* for a kinematic-bicycle vehicle.

Expands nodes with motion primitives (a fan of fixed steering angles, each
integrated forward over a constant arc length) so every edge is kinematically
feasible for the KinematicBicycle model used by the simulator. The continuous
(x, y, theta) state is discretized into (cell_x, cell_y, heading_bin) only for
the visited/closed bookkeeping, which is what makes this "hybrid" rather than a
pure grid search.

Interface matches Trail/main.py:
    planner = HybridAStar(occ, resolution=, vehicle_length=, step_size=,
                          theta_res_deg=, radius=)
    path = planner.plan(start_pose, goal_xy, max_iter=, goal_tolerance=, verbose=)
where start_pose is (x, y, theta), goal_xy is (x, y), and the return value is a
list of (x, y, theta) tuples in world meters, or None if no path was found.
"""
import heapq
import math


class HybridAStar:
    def __init__(self, occ_grid, resolution=0.5, vehicle_length=0.7,
                 step_size=0.18, theta_res_deg=10, radius=0.18,
                 max_steer=0.6, n_steer=7, reverse=False):
        """
        Args:
            occ_grid: boolean occupancy array indexed [row=y_cell, col=x_cell].
            resolution: meters per cell.
            vehicle_length: wheelbase L (must match the executing vehicle).
            step_size: arc length of each motion primitive (m).
            theta_res_deg: heading discretization for the closed set (deg).
            radius: robot radius for collision inflation (m).
            max_steer: steering limit (rad), should match vehicle.max_steer.
            n_steer: number of steering samples in [-max_steer, max_steer].
            reverse: if True, also expand reverse motions.
        """
        self.occ = occ_grid
        self.res = float(resolution)
        self.L = float(vehicle_length)
        self.step_size = float(step_size)
        self.theta_res = math.radians(theta_res_deg)
        self.radius = float(radius)
        self.max_steer = float(max_steer)
        self.n_steer = int(n_steer)
        self.reverse = bool(reverse)
        self.h, self.w = occ_grid.shape

        # Precompute the steering fan once.
        if self.n_steer <= 1:
            self._steers = [0.0]
        else:
            self._steers = [
                -self.max_steer + 2.0 * self.max_steer * i / (self.n_steer - 1)
                for i in range(self.n_steer)
            ]

    # ------------------------------------------------------------------ helpers
    def _state_key(self, x, y, theta):
        gx = int(round(x / self.res))
        gy = int(round(y / self.res))
        gt = int(round(((theta + math.pi) % (2.0 * math.pi)) / self.theta_res))
        return (gx, gy, gt)

    def _in_bounds(self, x, y):
        gx = int(round(x / self.res))
        gy = int(round(y / self.res))
        return 0 <= gx < self.w and 0 <= gy < self.h

    def _collision(self, x, y):
        """True if the inflated robot footprint at (x, y) hits an obstacle or leaves the map."""
        gx = int(round(x / self.res))
        gy = int(round(y / self.res))
        if gx < 0 or gy < 0 or gx >= self.w or gy >= self.h:
            return True
        rad_cells = int(math.ceil(self.radius / self.res))
        y0 = max(0, gy - rad_cells)
        y1 = min(self.h, gy + rad_cells + 1)
        x0 = max(0, gx - rad_cells)
        x1 = min(self.w, gx + rad_cells + 1)
        return bool(self.occ[y0:y1, x0:x1].any())

    def _step(self, x, y, theta, delta, direction):
        """Integrate the bicycle model over one primitive; None if it sweeps through an obstacle."""
        n_sub = 5
        ds = direction * self.step_size / n_sub
        for _ in range(n_sub):
            x += ds * math.cos(theta)
            y += ds * math.sin(theta)
            theta += (ds / self.L) * math.tan(delta)
            theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
            if self._collision(x, y):
                return None
        return x, y, theta

    # -------------------------------------------------------------------- search
    def plan(self, start_pose, goal_xy, max_iter=60000, goal_tolerance=0.6, verbose=False):
        sx, sy, stheta = start_pose[0], start_pose[1], start_pose[2]
        gx, gy = goal_xy[0], goal_xy[1]

        if self._collision(sx, sy):
            if verbose:
                print("HybridA*: start pose is in collision.")
            return None

        start_key = self._state_key(sx, sy, stheta)
        counter = 0
        open_heap = [(math.hypot(gx - sx, gy - sy), 0.0, counter, (sx, sy, stheta))]
        came_from = {start_key: None}
        gscore = {start_key: 0.0}
        node_state = {start_key: (sx, sy, stheta)}

        directions = (1.0, -1.0) if self.reverse else (1.0,)

        it = 0
        while open_heap and it < max_iter:
            it += 1
            _, g, _, (x, y, theta) = heapq.heappop(open_heap)
            cur_key = self._state_key(x, y, theta)

            # Skip stale heap entries superseded by a cheaper path to the same cell.
            if g > gscore.get(cur_key, float("inf")):
                continue

            if math.hypot(gx - x, gy - y) <= goal_tolerance:
                path = []
                k = cur_key
                while k is not None:
                    path.append(node_state[k])
                    k = came_from[k]
                path.reverse()
                if verbose:
                    print(f"HybridA*: goal reached in {it} iters, {len(path)} states.")
                return path

            for direction in directions:
                for delta in self._steers:
                    res = self._step(x, y, theta, delta, direction)
                    if res is None:
                        continue
                    nx, ny, ntheta = res
                    if not self._in_bounds(nx, ny):
                        continue
                    nkey = self._state_key(nx, ny, ntheta)
                    # Penalize steering effort and reversing for smoother, forward-biased paths.
                    move_cost = self.step_size + 0.5 * abs(delta)
                    if direction < 0:
                        move_cost += self.step_size  # reverse penalty
                    tentative = g + move_cost
                    if tentative < gscore.get(nkey, float("inf")):
                        gscore[nkey] = tentative
                        came_from[nkey] = cur_key
                        node_state[nkey] = (nx, ny, ntheta)
                        counter += 1
                        h = math.hypot(gx - nx, gy - ny)
                        heapq.heappush(open_heap, (tentative + h, tentative, counter, (nx, ny, ntheta)))

        if verbose:
            print(f"HybridA*: no path after {it} iterations.")
        return None
