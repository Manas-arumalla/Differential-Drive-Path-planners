# planners/dijkstra_planner.py
"""
Dijkstra's Path Planning Algorithm

Classic uniform-cost search algorithm that finds the shortest path
by exploring all nodes in order of increasing distance from start.
"""
import numpy as np
import heapq
from typing import List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DijkstraPlanner:
    """
    Dijkstra's algorithm path planner.
    
    Unlike A*, Dijkstra explores uniformly in all directions without
    a heuristic bias toward the goal. Guarantees shortest path.
    """
    
    def __init__(self, environment, robot_radius: float = 0.3):
        """
        Initialize Dijkstra planner.
        
        Args:
            environment: Environment object with grid
            robot_radius: Robot radius for collision checking
        """
        self.env = environment
        self.robot_radius = robot_radius
        self.path = []
        self.explored_nodes = []
        
    def get_neighbors(self, gx: int, gy: int) -> List[Tuple[int, int, float]]:
        """Get valid 8-connected neighbors with costs."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                
                if not (0 <= nx < self.env.grid_width and 0 <= ny < self.env.grid_height):
                    continue
                    
                if self.env.grid[ny, nx]:
                    continue
                
                # Check robot radius clearance
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
        Plan path using Dijkstra's algorithm.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of waypoints or None
        """
        start_grid = self.env.world_to_grid(start[0], start[1])
        goal_grid = self.env.world_to_grid(goal[0], goal[1])
        
        if not self.env.is_valid(start[0], start[1], self.robot_radius):
            print("Dijkstra Error: Start is invalid!")
            return None
        if not self.env.is_valid(goal[0], goal[1], self.robot_radius):
            print("Dijkstra Error: Goal is invalid!")
            return None
        
        # Priority queue: (cost, (gx, gy))
        open_heap = []
        heapq.heappush(open_heap, (0, start_grid))
        
        came_from = {}
        cost_so_far = {start_grid: 0}
        closed_set = set()
        
        self.explored_nodes = []
        
        while open_heap:
            current_cost, current = heapq.heappop(open_heap)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            self.explored_nodes.append(current)
            
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
                path[-1] = goal
                
                self.path = path
                print(f"Dijkstra found path with {len(path)} waypoints, explored {len(self.explored_nodes)} nodes")
                return path
            
            for nx, ny, move_cost in self.get_neighbors(current[0], current[1]):
                neighbor = (nx, ny)
                
                if neighbor in closed_set:
                    continue
                
                new_cost = cost_so_far[current] + move_cost
                
                if new_cost < cost_so_far.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(open_heap, (new_cost, neighbor))
        
        print("Dijkstra Error: No path found!")
        return None
    
    def get_explored_nodes(self) -> List[Tuple[float, float]]:
        """Get explored nodes for visualization."""
        return [(self.env.grid_to_world(n[0], n[1])) for n in self.explored_nodes]


# ============== Demo ==============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment import create_demo_environment
    from visualize import plot_environment, plot_start_goal, plot_path, plot_explored_nodes
    
    print("=" * 50)
    print("Dijkstra Path Planning Demo")
    print("=" * 50)
    
    env = create_demo_environment()
    start = (3, 3)
    goal = (45, 45)
    
    planner = DijkstraPlanner(env, robot_radius=0.5)
    
    import time
    t_start = time.time()
    path = planner.plan(start, goal)
    t_end = time.time()
    
    print(f"Planning time: {t_end - t_start:.4f} seconds")
    
    if path:
        path_length = sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) 
                         for i in range(len(path)-1))
        print(f"Path length: {path_length:.2f} meters")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_environment(ax, env, "Dijkstra Path Planning")
    
    explored = planner.get_explored_nodes()
    plot_explored_nodes(ax, explored, algorithm='Dijkstra')
    
    if path:
        plot_path(ax, path, algorithm='Dijkstra', linewidth=3, label='Dijkstra Path')
    
    plot_start_goal(ax, start, goal)
    ax.legend(loc='upper right', facecolor='#2D3748', edgecolor='#4A5568', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig('dijkstra_result.png', dpi=150, facecolor='#1A202C')
    print("Saved result to dijkstra_result.png")
    plt.show()
