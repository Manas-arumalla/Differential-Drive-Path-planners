# planners/prm_planner.py
"""
PRM (Probabilistic Roadmap) Path Planning Algorithm

A two-phase sampling-based planner:
1. Learning phase: Build a roadmap by sampling random configurations
2. Query phase: Connect start/goal to roadmap and find path
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import heapq
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PRMPlanner:
    """
    Probabilistic Roadmap path planner.
    
    Builds a graph of collision-free configurations connected
    by local paths, then searches this graph for a path.
    """
    
    def __init__(self, environment, robot_radius: float = 0.3,
                 num_samples: int = 300, k_neighbors: int = 10):
        """
        Initialize PRM planner.
        
        Args:
            environment: Environment object
            robot_radius: Robot radius for collision checking
            num_samples: Number of random samples for roadmap
            k_neighbors: Number of nearest neighbors to connect
        """
        self.env = environment
        self.robot_radius = robot_radius
        self.n_samples = num_samples
        self.k = k_neighbors
        
        # Roadmap storage
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.edges: Dict[int, List[Tuple[int, float]]] = {}  # node_id -> [(neighbor_id, cost), ...]
        self.roadmap_built = False
        
        self.path = []
        
    def _sample_free(self) -> Tuple[float, float]:
        """Sample a random collision-free configuration."""
        for _ in range(100):  # Max attempts
            x = np.random.uniform(1, self.env.width - 1)
            y = np.random.uniform(1, self.env.height - 1)
            if self.env.is_valid(x, y, self.robot_radius):
                return (x, y)
        return None
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return np.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def _k_nearest(self, point: Tuple[float, float], k: int) -> List[int]:
        """Find k nearest nodes to a point."""
        distances = []
        for node_id, node_pos in self.nodes.items():
            dist = self._distance(point, node_pos)
            distances.append((dist, node_id))
        
        distances.sort(key=lambda x: x[0])
        return [node_id for _, node_id in distances[:k]]
    
    def _can_connect(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if two points can be connected with a collision-free path."""
        return self.env.is_path_valid(p1, p2, self.robot_radius)
    
    def build_roadmap(self):
        """Build the PRM roadmap (learning phase)."""
        print(f"PRM: Building roadmap with {self.n_samples} samples...")
        
        # Sample nodes
        node_id = 0
        attempts = 0
        while len(self.nodes) < self.n_samples and attempts < self.n_samples * 3:
            sample = self._sample_free()
            if sample:
                self.nodes[node_id] = sample
                self.edges[node_id] = []
                node_id += 1
            attempts += 1
        
        print(f"  Sampled {len(self.nodes)} valid nodes")
        
        # Connect nodes to k-nearest neighbors
        edge_count = 0
        for nid, pos in self.nodes.items():
            neighbors = self._k_nearest(pos, self.k + 1)  # +1 because it includes self
            
            for neighbor_id in neighbors:
                if neighbor_id == nid:
                    continue
                
                neighbor_pos = self.nodes[neighbor_id]
                
                # Check if edge already exists
                existing = [e[0] for e in self.edges[nid]]
                if neighbor_id in existing:
                    continue
                
                # Check if can connect
                if self._can_connect(pos, neighbor_pos):
                    dist = self._distance(pos, neighbor_pos)
                    self.edges[nid].append((neighbor_id, dist))
                    self.edges[neighbor_id].append((nid, dist))
                    edge_count += 1
        
        print(f"  Created {edge_count} edges")
        self.roadmap_built = True
    
    def _add_node(self, point: Tuple[float, float]) -> Optional[int]:
        """Add a temporary node to the roadmap and connect it."""
        node_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        self.nodes[node_id] = point
        self.edges[node_id] = []
        
        # Connect to nearest neighbors
        neighbors = self._k_nearest(point, self.k * 2)
        
        connected = False
        for neighbor_id in neighbors:
            if neighbor_id == node_id:
                continue
            neighbor_pos = self.nodes[neighbor_id]
            
            if self._can_connect(point, neighbor_pos):
                dist = self._distance(point, neighbor_pos)
                self.edges[node_id].append((neighbor_id, dist))
                self.edges[neighbor_id].append((node_id, dist))
                connected = True
        
        if connected:
            return node_id
        else:
            # Remove if couldn't connect
            del self.nodes[node_id]
            del self.edges[node_id]
            return None
    
    def _search_path(self, start_id: int, goal_id: int) -> Optional[List[int]]:
        """A* search on the roadmap."""
        goal_pos = self.nodes[goal_id]
        
        open_heap = []
        heapq.heappush(open_heap, (0, start_id))
        
        came_from = {}
        g_score = {start_id: 0}
        closed = set()
        
        while open_heap:
            _, current = heapq.heappop(open_heap)
            
            if current in closed:
                continue
            closed.add(current)
            
            if current == goal_id:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            for neighbor_id, cost in self.edges.get(current, []):
                if neighbor_id in closed:
                    continue
                
                tentative_g = g_score[current] + cost
                
                if tentative_g < g_score.get(neighbor_id, float('inf')):
                    came_from[neighbor_id] = current
                    g_score[neighbor_id] = tentative_g
                    h = self._distance(self.nodes[neighbor_id], goal_pos)
                    heapq.heappush(open_heap, (tentative_g + h, neighbor_id))
        
        return None
    
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path using PRM.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of waypoints or None
        """
        if not self.env.is_valid(start[0], start[1], self.robot_radius):
            print("PRM Error: Start is invalid!")
            return None
        if not self.env.is_valid(goal[0], goal[1], self.robot_radius):
            print("PRM Error: Goal is invalid!")
            return None
        
        # Build roadmap if not done
        if not self.roadmap_built:
            self.build_roadmap()
        
        # Add start and goal to roadmap
        start_id = self._add_node(start)
        goal_id = self._add_node(goal)
        
        if start_id is None:
            print("PRM Error: Could not connect start to roadmap!")
            return None
        if goal_id is None:
            print("PRM Error: Could not connect goal to roadmap!")
            return None
        
        # Search for path
        node_path = self._search_path(start_id, goal_id)
        
        if node_path is None:
            print("PRM Error: No path found in roadmap!")
            return None
        
        # Convert to waypoints
        path = [self.nodes[nid] for nid in node_path]
        
        # Ensure exact start and goal
        path[0] = start
        path[-1] = goal
        
        self.path = path
        print(f"PRM found path with {len(path)} waypoints")
        return path
    
    def get_roadmap(self) -> Tuple[Dict, Dict]:
        """Get roadmap for visualization."""
        return self.nodes, self.edges


# ============== Demo ==============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment import create_demo_environment
    from visualize import plot_environment, plot_start_goal, plot_path
    
    print("=" * 50)
    print("PRM Path Planning Demo")
    print("=" * 50)
    
    env = create_demo_environment()
    start = (3, 3)
    goal = (45, 45)
    
    planner = PRMPlanner(env, robot_radius=0.5, num_samples=250, k_neighbors=8)
    
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
    plot_environment(ax, env, "PRM Path Planning")
    
    # Plot roadmap
    nodes, edges = planner.get_roadmap()
    node_positions = np.array(list(nodes.values()))
    ax.scatter(node_positions[:, 0], node_positions[:, 1], 
              c='#FF69B4', s=15, alpha=0.5, zorder=2, label='Roadmap nodes')
    
    # Plot edges
    for nid, edge_list in edges.items():
        pos = nodes[nid]
        for neighbor_id, _ in edge_list:
            neighbor_pos = nodes[neighbor_id]
            ax.plot([pos[0], neighbor_pos[0]], [pos[1], neighbor_pos[1]],
                   'pink', alpha=0.2, linewidth=0.5, zorder=1)
    
    if path:
        plot_path(ax, path, algorithm='PRM', linewidth=3, label='PRM Path')
    
    plot_start_goal(ax, start, goal)
    ax.legend(loc='upper right', facecolor='#2D3748', edgecolor='#4A5568', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig('prm_result.png', dpi=150, facecolor='#1A202C')
    print("Saved result to prm_result.png")
    plt.show()
