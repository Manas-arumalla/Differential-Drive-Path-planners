# environment.py
"""
Environment module for mobile robot path planning.
Handles map creation, obstacles, and collision detection.
"""
import numpy as np
from typing import List, Tuple, Optional

class Environment:
    """2D environment with obstacles for path planning."""
    
    def __init__(self, width: float = 50.0, height: float = 50.0, resolution: float = 0.5):
        """
        Initialize environment.
        
        Args:
            width: World width in meters
            height: World height in meters
            resolution: Grid resolution (meters per cell)
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # Occupancy grid (True = obstacle)
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Circular obstacles list: [(x, y, radius), ...]
        self.obstacles: List[Tuple[float, float, float]] = []
    
    def add_rectangular_obstacle(self, x: float, y: float, w: float, h: float):
        """Add rectangular obstacle at position (x,y) with size (w,h)."""
        gx_start = max(0, int(x / self.resolution))
        gy_start = max(0, int(y / self.resolution))
        gx_end = min(self.grid_width, int((x + w) / self.resolution))
        gy_end = min(self.grid_height, int((y + h) / self.resolution))
        self.grid[gy_start:gy_end, gx_start:gx_end] = True
    
    def add_circular_obstacle(self, cx: float, cy: float, radius: float):
        """Add circular obstacle at center (cx, cy) with given radius."""
        self.obstacles.append((cx, cy, radius))
        # Also mark in grid
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                wx, wy = self.grid_to_world(gx, gy)
                if np.hypot(wx - cx, wy - cy) <= radius:
                    self.grid[gy, gx] = True
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int(round(x / self.resolution))
        gy = int(round(y / self.resolution))
        return gx, gy
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        return gx * self.resolution, gy * self.resolution
    
    def is_valid(self, x: float, y: float, robot_radius: float = 0.3) -> bool:
        """Check if position is valid (in bounds and collision-free)."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        gx, gy = self.world_to_grid(x, y)
        if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
            return False
        
        # Check grid cells within robot radius
        rad_cells = int(np.ceil(robot_radius / self.resolution))
        for dy in range(-rad_cells, rad_cells + 1):
            for dx in range(-rad_cells, rad_cells + 1):
                check_gx = gx + dx
                check_gy = gy + dy
                if 0 <= check_gx < self.grid_width and 0 <= check_gy < self.grid_height:
                    if self.grid[check_gy, check_gx]:
                        return False
        return True
    
    def is_path_valid(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                      robot_radius: float = 0.3, step: float = 0.2) -> bool:
        """Check if straight line path between two points is collision-free."""
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 1e-6:
            return self.is_valid(p1[0], p1[1], robot_radius)
        
        steps = max(2, int(dist / step))
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if not self.is_valid(x, y, robot_radius):
                return False
        return True
    
    def get_neighbors_grid(self, gx: int, gy: int) -> List[Tuple[int, int, float]]:
        """Get 8-connected neighbors with movement costs."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    if not self.grid[ny, nx]:
                        cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                        neighbors.append((nx, ny, cost))
        return neighbors


def create_demo_environment() -> Environment:
    """Create a demo environment with various obstacles."""
    env = Environment(width=50.0, height=50.0, resolution=0.5)
    
    # Add rectangular obstacles (walls)
    env.add_rectangular_obstacle(10, 10, 2, 20)   # Vertical wall
    env.add_rectangular_obstacle(20, 5, 15, 2)    # Horizontal wall
    env.add_rectangular_obstacle(25, 25, 2, 15)   # Another vertical wall
    env.add_rectangular_obstacle(35, 15, 2, 20)   # Right wall
    
    # Add circular obstacles
    env.add_circular_obstacle(15, 35, 3)
    env.add_circular_obstacle(30, 10, 2.5)
    env.add_circular_obstacle(40, 40, 4)
    env.add_circular_obstacle(8, 42, 2)
    
    return env


if __name__ == "__main__":
    # Test environment
    import matplotlib.pyplot as plt
    
    env = create_demo_environment()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(env.grid, origin='lower', 
              extent=[0, env.width, 0, env.height],
              cmap='gray_r', alpha=0.7)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Demo Environment')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Test collision checking
    start = (2, 2)
    goal = (45, 45)
    ax.plot(start[0], start[1], 'go', markersize=15, label='Start')
    ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
