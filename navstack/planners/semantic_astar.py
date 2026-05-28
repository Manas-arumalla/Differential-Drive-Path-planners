"""Semantic / terrain-aware A* path planning.

Beyond binary free/occupied, real robots prefer some terrain over others — drive
on the road, avoid the grass, never enter the water. A :class:`SemanticCostmap`
assigns a traversal-cost multiplier to regions, and :class:`SemanticAStarPlanner`
minimizes the *terrain-weighted* path cost, so it will take a longer geometric
route to stay on cheap terrain.
"""
import heapq
from typing import List, Optional, Tuple

import numpy as np


class SemanticCostmap:
    """Per-cell traversal-cost multipliers aligned to an Environment's grid."""

    def __init__(self, env):
        self.env = env
        self.cost = np.ones((env.grid_height, env.grid_width), dtype=float)

    def add_rect_cost(self, x, y, w, h, cost):
        gx0, gy0 = self.env.world_to_grid(x, y)
        gx1, gy1 = self.env.world_to_grid(x + w, y + h)
        gx0, gx1 = sorted((max(0, gx0), min(self.env.grid_width, gx1)))
        gy0, gy1 = sorted((max(0, gy0), min(self.env.grid_height, gy1)))
        self.cost[gy0:gy1, gx0:gx1] = cost

    def add_circle_cost(self, cx, cy, r, cost):
        for gy in range(self.env.grid_height):
            for gx in range(self.env.grid_width):
                wx, wy = self.env.grid_to_world(gx, gy)
                if np.hypot(wx - cx, wy - cy) <= r:
                    self.cost[gy, gx] = cost

    def at(self, gx, gy):
        return self.cost[gy, gx]


class SemanticAStarPlanner:
    def __init__(self, env, costmap: SemanticCostmap, robot_radius: float = 0.3):
        self.env = env
        self.costmap = costmap
        self.robot_radius = robot_radius
        self._rad_cells = int(np.ceil(robot_radius / env.resolution))

    def _collision(self, gx, gy):
        e = self.env
        if not (0 <= gx < e.grid_width and 0 <= gy < e.grid_height):
            return True
        r = self._rad_cells
        sub = e.grid[max(0, gy - r):min(e.grid_height, gy + r + 1),
                     max(0, gx - r):min(e.grid_width, gx + r + 1)]
        return bool(sub.any())

    def _neighbors(self, gx, gy):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                if self._collision(nx, ny):
                    continue
                move = 1.4142 if (dx and dy) else 1.0
                yield nx, ny, move

    def plan(self, start: Tuple[float, float],
             goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        e = self.env
        s = e.world_to_grid(*start)
        g = e.world_to_grid(*goal)
        if self._collision(*s) or self._collision(*g):
            return None

        def h(a):
            return np.hypot(a[0] - g[0], a[1] - g[1])

        open_heap = [(h(s), 0.0, s)]
        came = {}
        gscore = {s: 0.0}
        closed = set()
        while open_heap:
            _, gc, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == g:
                path = [goal]
                node = cur
                while node in came:
                    node = came[node]
                    path.append(e.grid_to_world(node[0], node[1]))
                path.reverse()
                path[0] = start
                return path
            for nx, ny, move in self._neighbors(*cur):
                nb = (nx, ny)
                if nb in closed:
                    continue
                # terrain-weighted edge cost: geometric distance x average terrain cost
                edge = move * 0.5 * (self.costmap.at(*cur) + self.costmap.at(nx, ny))
                tentative = gscore[cur] + edge
                if tentative < gscore.get(nb, float("inf")):
                    came[nb] = cur
                    gscore[nb] = tentative
                    heapq.heappush(open_heap, (tentative + h(nb), tentative, nb))
        return None
