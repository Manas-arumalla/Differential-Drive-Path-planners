# planners/rrt_star_planner.py
"""
RRT* (Optimal RRT) Path Planning Algorithm

An asymptotically optimal variant of RRT that rewires the tree
to find shorter paths as more samples are added.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RRTStarPlanner:
    """
    RRT* (Optimal RRT) path planner.
    
    Extends RRT with:
    - Neighborhood search for better parent selection
    - Tree rewiring to reduce path cost
    """
    
    def __init__(self, environment, robot_radius: float = 0.3,
                 step_size: float = 1.5, max_iterations: int = 3000,
                 goal_sample_rate: float = 0.1, search_radius: float = 3.0):
        """
        Initialize RRT* planner.
        
        Args:
            environment: Environment object
            robot_radius: Robot radius for collision checking
            step_size: Maximum step size for tree expansion
            max_iterations: Maximum planning iterations
            goal_sample_rate: Probability of sampling the goal
            search_radius: Radius for neighborhood search and rewiring
        """
        self.env = environment
        self.robot_radius = robot_radius
        self.step_size = step_size
        self.max_iter = max_iterations
        self.goal_rate = goal_sample_rate
        self.search_radius = search_radius
        
        # Tree storage
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.parent: Dict[int, Optional[int]] = {}
        self.cost: Dict[int, float] = {}  # Cost from start to node
        self.path = []
        
    def random_sample(self, goal: Tuple[float, float]) -> Tuple[float, float]:
        """Sample random point with bias toward goal."""
        if np.random.random() < self.goal_rate:
            return goal
        
        x = np.random.uniform(0, self.env.width)
        y = np.random.uniform(0, self.env.height)
        return (x, y)
    
    def nearest_node(self, point: Tuple[float, float]) -> int:
        """Find nearest node in tree."""
        min_dist = float('inf')
        nearest_id = 0
        
        for node_id, node_pos in self.nodes.items():
            dist = np.hypot(point[0] - node_pos[0], point[1] - node_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_id = node_id
        
        return nearest_id
    
    def near_nodes(self, point: Tuple[float, float]) -> List[int]:
        """Find all nodes within search radius."""
        near = []
        for node_id, node_pos in self.nodes.items():
            dist = np.hypot(point[0] - node_pos[0], point[1] - node_pos[1])
            if dist <= self.search_radius:
                near.append(node_id)
        return near
    
    def steer(self, from_pos: Tuple[float, float], 
              to_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Steer toward target, limited by step_size."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        dist = np.hypot(dx, dy)
        
        if dist <= self.step_size:
            return to_pos
        else:
            ratio = self.step_size / dist
            return (from_pos[0] + dx * ratio, from_pos[1] + dy * ratio)
    
    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return np.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def is_collision_free(self, from_pos: Tuple[float, float], 
                          to_pos: Tuple[float, float]) -> bool:
        """Check if path is collision free."""
        return self.env.is_path_valid(from_pos, to_pos, self.robot_radius)
    
    def choose_parent(self, new_pos: Tuple[float, float], 
                     near_ids: List[int], nearest_id: int) -> Tuple[int, float]:
        """
        Choose best parent from near nodes (minimizes cost).
        """
        best_parent = nearest_id
        best_cost = self.cost[nearest_id] + self.distance(self.nodes[nearest_id], new_pos)
        
        for node_id in near_ids:
            node_pos = self.nodes[node_id]
            potential_cost = self.cost[node_id] + self.distance(node_pos, new_pos)
            
            if potential_cost < best_cost:
                if self.is_collision_free(node_pos, new_pos):
                    best_parent = node_id
                    best_cost = potential_cost
        
        return best_parent, best_cost
    
    def rewire(self, new_id: int, near_ids: List[int]):
        """
        Rewire tree to reduce costs through new node.
        """
        new_pos = self.nodes[new_id]
        
        for node_id in near_ids:
            if node_id == self.parent[new_id]:
                continue
            
            node_pos = self.nodes[node_id]
            potential_cost = self.cost[new_id] + self.distance(new_pos, node_pos)
            
            if potential_cost < self.cost[node_id]:
                if self.is_collision_free(new_pos, node_pos):
                    # Rewire - change parent to new node
                    self.parent[node_id] = new_id
                    self.cost[node_id] = potential_cost
    
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float],
             goal_threshold: float = 1.5) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using RRT*.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            goal_threshold: Distance threshold for goal
            
        Returns:
            List of waypoints or None
        """
        # Initialize tree
        self.nodes = {0: start}
        self.parent = {0: None}
        self.cost = {0: 0.0}
        
        # Validate
        if not self.env.is_valid(start[0], start[1], self.robot_radius):
            print("RRT* Error: Start is invalid!")
            return None
        if not self.env.is_valid(goal[0], goal[1], self.robot_radius):
            print("RRT* Error: Goal is invalid!")
            return None
        
        node_id = 0
        goal_node_id = None
        best_goal_cost = float('inf')
        
        for iteration in range(self.max_iter):
            # Sample
            random_point = self.random_sample(goal)
            
            # Find nearest
            nearest_id = self.nearest_node(random_point)
            nearest_pos = self.nodes[nearest_id]
            
            # Steer
            new_pos = self.steer(nearest_pos, random_point)
            
            # Check collision
            if not self.is_collision_free(nearest_pos, new_pos):
                continue
            if not self.env.is_valid(new_pos[0], new_pos[1], self.robot_radius):
                continue
            
            # Find near nodes
            near_ids = self.near_nodes(new_pos)
            
            # Choose best parent
            best_parent, new_cost = self.choose_parent(new_pos, near_ids, nearest_id)
            
            # Add node
            node_id += 1
            self.nodes[node_id] = new_pos
            self.parent[node_id] = best_parent
            self.cost[node_id] = new_cost
            
            # Rewire tree
            self.rewire(node_id, near_ids)
            
            # Check goal
            dist_to_goal = self.distance(new_pos, goal)
            if dist_to_goal <= goal_threshold:
                if self.is_collision_free(new_pos, goal):
                    goal_cost = new_cost + dist_to_goal
                    if goal_cost < best_goal_cost:
                        goal_node_id = node_id
                        best_goal_cost = goal_cost
        
        # Connect best node to goal
        if goal_node_id is not None:
            node_id += 1
            self.nodes[node_id] = goal
            self.parent[node_id] = goal_node_id
            self.cost[node_id] = best_goal_cost
            
            path = self._reconstruct_path(node_id)
            self.path = path
            print(f"RRT* found path with {len(path)} waypoints, cost: {best_goal_cost:.2f}")
            return path
        
        print("RRT* Warning: No path found!")
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
        """Get tree for visualization."""
        return self.nodes, self.parent


# ============== Demo ==============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment import create_demo_environment
    from visualize import plot_environment, plot_start_goal, plot_path, plot_tree
    
    print("=" * 50)
    print("RRT* Path Planning Demo")
    print("=" * 50)
    
    # Create environment
    env = create_demo_environment()
    
    start = (3, 3)
    goal = (45, 45)
    
    # Create planner
    planner = RRTStarPlanner(env, robot_radius=0.5, step_size=2.0,
                             max_iterations=2000, goal_sample_rate=0.15,
                             search_radius=4.0)
    
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
    plot_environment(ax, env, "RRT* Path Planning (Optimal)")
    
    # Plot tree
    nodes, parent = planner.get_tree()
    plot_tree(ax, nodes, parent, color='lightblue', alpha=0.4)
    
    if path:
        plot_path(ax, path, algorithm='RRT*', linewidth=3, label='RRT* Path')
    
    plot_start_goal(ax, start, goal)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('rrt_star_result.png', dpi=150)
    print("Saved result to rrt_star_result.png")
    plt.show()
