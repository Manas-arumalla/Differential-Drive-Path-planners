# planners/astar_planner.py
"""
A* Path Planning Algorithm

Grid-based optimal path planning using A* search.
Guarantees the shortest path on a discrete grid.
"""
import numpy as np
import heapq
from typing import List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AStarPlanner:
    """
    A* path planner on occupancy grid.
    
    Uses 8-connected grid search with Euclidean heuristic.
    """
    
    def __init__(self, environment, robot_radius: float = 0.3):
        """
        Initialize A* planner.
        
        Args:
            environment: Environment object with grid
            robot_radius: Robot radius for collision checking
        """
        self.env = environment
        self.robot_radius = robot_radius
        self.path = []
        self.explored_nodes = []  # For visualization
        
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.hypot(a[0] - b[0], a[1] - b[1])
    
    def get_neighbors(self, gx: int, gy: int) -> List[Tuple[int, int, float]]:
        """Get valid 8-connected neighbors."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                
                # Check bounds
                if not (0 <= nx < self.env.grid_width and 0 <= ny < self.env.grid_height):
                    continue
                    
                # Check if cell is free (considering robot radius)
                if self.env.grid[ny, nx]:
                    continue
                
                # Check inflation for robot radius
                rad_cells = int(np.ceil(self.robot_radius / self.env.resolution))
                collision = False
                for iy in range(max(0, ny-rad_cells), min(self.env.grid_height, ny+rad_cells+1)):
                    for ix in range(max(0, nx-rad_cells), min(self.env.grid_width, nx+rad_cells+1)):
                        if self.env.grid[iy, ix]:
                            collision = True
                            break
                    if collision:
                        break
                
                if not collision:
                    cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                    neighbors.append((nx, ny, cost))
        
        return neighbors
    
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using A*.
        
        Args:
            start: Start position (x, y) in world coordinates
            goal: Goal position (x, y) in world coordinates
            
        Returns:
            List of waypoints [(x, y), ...] or None if no path found
        """
        # Convert to grid coordinates
        start_grid = self.env.world_to_grid(start[0], start[1])
        goal_grid = self.env.world_to_grid(goal[0], goal[1])
        
        # Validate start and goal
        if not self.env.is_valid(start[0], start[1], self.robot_radius):
            print("A* Error: Start position is invalid!")
            return None
        if not self.env.is_valid(goal[0], goal[1], self.robot_radius):
            print("A* Error: Goal position is invalid!")
            return None
        
        # Priority queue: (f_cost, g_cost, (gx, gy))
        open_heap = []
        heapq.heappush(open_heap, (self.heuristic(start_grid, goal_grid), 0, start_grid))
        
        # Tracking dictionaries
        came_from = {}
        g_score = {start_grid: 0}
        closed_set = set()
        
        self.explored_nodes = []
        
        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            self.explored_nodes.append(current)
            
            # Check if goal reached
            if current == goal_grid:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    wx, wy = self.env.grid_to_world(node[0], node[1])
                    path.append((wx, wy))
                    node = came_from[node]
                wx, wy = self.env.grid_to_world(start_grid[0], start_grid[1])
                path.append((wx, wy))
                path.reverse()
                
                # Add exact goal position
                path[-1] = goal
                
                self.path = path
                print(f"A* found path with {len(path)} waypoints")
                return path
            
            # Explore neighbors
            for nx, ny, cost in self.get_neighbors(current[0], current[1]):
                neighbor = (nx, ny)
                
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + cost
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor))
        
        print("A* Error: No path found!")
        return None
    
    def get_explored_nodes(self) -> List[Tuple[float, float]]:
        """Get list of explored nodes in world coordinates (for visualization)."""
        return [(self.env.grid_to_world(n[0], n[1])) for n in self.explored_nodes]


# ============== Demo ==============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment import create_demo_environment
    from visualize import plot_environment, plot_start_goal, plot_path
    
    print("=" * 50)
    print("A* Path Planning Demo")
    print("=" * 50)
    
    # Create environment
    env = create_demo_environment()
    
    # Define start and goal
    start = (3, 3)
    goal = (45, 45)
    
    # Create planner and find path
    planner = AStarPlanner(env, robot_radius=0.5)
    
    import time
    t_start = time.time()
    path = planner.plan(start, goal)
    t_end = time.time()
    
    print(f"Planning time: {t_end - t_start:.4f} seconds")
    
    if path:
        # Calculate path length
        path_length = sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) 
                         for i in range(len(path)-1))
        print(f"Path length: {path_length:.2f} meters")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_environment(ax, env, "A* Path Planning")
    
    # Plot explored nodes
    explored = planner.get_explored_nodes()
    if explored:
        exp_arr = np.array(explored)
        ax.scatter(exp_arr[:, 0], exp_arr[:, 1], c='lightblue', s=5, alpha=0.3, label='Explored')
    
    # Plot path
    if path:
        plot_path(ax, path, algorithm='A*', linewidth=3, label='A* Path')
    
    plot_start_goal(ax, start, goal)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('astar_result.png', dpi=150)
    print("Saved result to astar_result.png")
    plt.show()
