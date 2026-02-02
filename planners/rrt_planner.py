# planners/rrt_planner.py
"""
RRT (Rapidly-exploring Random Tree) Path Planning Algorithm

Sampling-based path planning that builds a tree by randomly sampling
the configuration space and connecting to the nearest node.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RRTPlanner:
    """
    Basic RRT path planner.
    
    Builds a random tree from start toward goal.
    """
    
    def __init__(self, environment, robot_radius: float = 0.3,
                 step_size: float = 1.0, max_iterations: int = 5000,
                 goal_sample_rate: float = 0.1):
        """
        Initialize RRT planner.
        
        Args:
            environment: Environment object
            robot_radius: Robot radius for collision checking
            step_size: Maximum step size for tree expansion
            max_iterations: Maximum planning iterations
            goal_sample_rate: Probability of sampling the goal
        """
        self.env = environment
        self.robot_radius = robot_radius
        self.step_size = step_size
        self.max_iter = max_iterations
        self.goal_rate = goal_sample_rate
        
        # Tree storage
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.parent: Dict[int, Optional[int]] = {}
        self.path = []
        
    def random_sample(self, goal: Tuple[float, float]) -> Tuple[float, float]:
        """Sample random point, with bias toward goal."""
        if np.random.random() < self.goal_rate:
            return goal
        
        x = np.random.uniform(0, self.env.width)
        y = np.random.uniform(0, self.env.height)
        return (x, y)
    
    def nearest_node(self, point: Tuple[float, float]) -> int:
        """Find nearest node in tree to given point."""
        min_dist = float('inf')
        nearest_id = 0
        
        for node_id, node_pos in self.nodes.items():
            dist = np.hypot(point[0] - node_pos[0], point[1] - node_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_id = node_id
        
        return nearest_id
    
    def steer(self, from_pos: Tuple[float, float], 
              to_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Steer from one position toward another, limited by step_size."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        dist = np.hypot(dx, dy)
        
        if dist <= self.step_size:
            return to_pos
        else:
            ratio = self.step_size / dist
            new_x = from_pos[0] + dx * ratio
            new_y = from_pos[1] + dy * ratio
            return (new_x, new_y)
    
    def is_collision_free(self, from_pos: Tuple[float, float], 
                          to_pos: Tuple[float, float]) -> bool:
        """Check if path between two points is collision free."""
        return self.env.is_path_valid(from_pos, to_pos, self.robot_radius)
    
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float],
             goal_threshold: float = 1.0) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using RRT.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            goal_threshold: Distance threshold to consider goal reached
            
        Returns:
            List of waypoints or None if no path found
        """
        # Initialize tree with start node
        self.nodes = {0: start}
        self.parent = {0: None}
        
        # Validate start and goal
        if not self.env.is_valid(start[0], start[1], self.robot_radius):
            print("RRT Error: Start position is invalid!")
            return None
        if not self.env.is_valid(goal[0], goal[1], self.robot_radius):
            print("RRT Error: Goal position is invalid!")
            return None
        
        node_id = 0
        
        for iteration in range(self.max_iter):
            # Sample random point
            random_point = self.random_sample(goal)
            
            # Find nearest node
            nearest_id = self.nearest_node(random_point)
            nearest_pos = self.nodes[nearest_id]
            
            # Steer toward random point
            new_pos = self.steer(nearest_pos, random_point)
            
            # Check collision
            if not self.is_collision_free(nearest_pos, new_pos):
                continue
            
            if not self.env.is_valid(new_pos[0], new_pos[1], self.robot_radius):
                continue
            
            # Add node to tree
            node_id += 1
            self.nodes[node_id] = new_pos
            self.parent[node_id] = nearest_id
            
            # Check if goal reached
            dist_to_goal = np.hypot(new_pos[0] - goal[0], new_pos[1] - goal[1])
            
            if dist_to_goal <= goal_threshold:
                # Try to connect directly to goal
                if self.is_collision_free(new_pos, goal):
                    node_id += 1
                    self.nodes[node_id] = goal
                    self.parent[node_id] = node_id - 1
                    
                    # Reconstruct path
                    path = self._reconstruct_path(node_id)
                    self.path = path
                    print(f"RRT found path with {len(path)} waypoints in {iteration+1} iterations")
                    return path
        
        print(f"RRT Warning: Max iterations ({self.max_iter}) reached!")
        return None
    
    def _reconstruct_path(self, goal_id: int) -> List[Tuple[float, float]]:
        """Reconstruct path from goal to start."""
        path = []
        current = goal_id
        
        while current is not None:
            path.append(self.nodes[current])
            current = self.parent[current]
        
        path.reverse()
        return path
    
    def get_tree(self) -> Tuple[Dict, Dict]:
        """Get tree nodes and parent relationships for visualization."""
        return self.nodes, self.parent


# ============== Demo ==============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment import create_demo_environment
    from visualize import plot_environment, plot_start_goal, plot_path, plot_tree
    
    print("=" * 50)
    print("RRT Path Planning Demo")
    print("=" * 50)
    
    # Create environment
    env = create_demo_environment()
    
    # Define start and goal
    start = (3, 3)
    goal = (45, 45)
    
    # Create planner and find path
    planner = RRTPlanner(env, robot_radius=0.5, step_size=2.0, 
                         max_iterations=3000, goal_sample_rate=0.15)
    
    import time
    t_start = time.time()
    path = planner.plan(start, goal)
    t_end = time.time()
    
    print(f"Planning time: {t_end - t_start:.4f} seconds")
    
    if path:
        path_length = sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) 
                         for i in range(len(path)-1))
        print(f"Path length: {path_length:.2f} meters")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_environment(ax, env, "RRT Path Planning")
    
    # Plot tree
    nodes, parent = planner.get_tree()
    plot_tree(ax, nodes, parent, color='lightgreen', alpha=0.5)
    
    # Plot path
    if path:
        plot_path(ax, path, algorithm='RRT', linewidth=3, label='RRT Path')
    
    plot_start_goal(ax, start, goal)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('rrt_result.png', dpi=150)
    print("Saved result to rrt_result.png")
    plt.show()
