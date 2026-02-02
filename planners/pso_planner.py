# planners/pso_planner.py
"""
PSO (Particle Swarm Optimization) Path Planning Algorithm

Optimization-based path planning that uses a swarm of particles
to search for an optimal path represented by control points.
"""
import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import CubicSpline
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PSOPlanner:
    """
    Particle Swarm Optimization path planner.
    
    Represents paths as spline control points and optimizes
    using PSO to minimize path length while avoiding obstacles.
    """
    
    def __init__(self, environment, robot_radius: float = 0.3,
                 num_particles: int = 50, max_iterations: int = 150,
                 num_control_points: int = 4, path_resolution: int = 50):
        """
        Initialize PSO planner.
        
        Args:
            environment: Environment object
            robot_radius: Robot radius for collision checking
            num_particles: Number of particles in swarm
            max_iterations: Maximum optimization iterations
            num_control_points: Number of intermediate control points
            path_resolution: Number of points to render the path
        """
        self.env = environment
        self.robot_radius = robot_radius
        self.n_particles = num_particles
        self.max_iter = max_iterations
        self.n_control = num_control_points
        self.resolution = path_resolution
        
        # PSO parameters
        self.w = 0.7      # Inertia weight
        self.c1 = 1.5     # Cognitive parameter
        self.c2 = 1.5     # Social parameter
        self.w_damp = 0.99  # Inertia damping
        
        self.path = []
        self.best_costs = []  # For visualization of convergence
        
    def _generate_path_from_control_points(self, start: Tuple[float, float],
                                           goal: Tuple[float, float],
                                           control_points: np.ndarray) -> List[Tuple[float, float]]:
        """
        Generate smooth path using cubic spline interpolation.
        
        Args:
            start: Start position
            goal: Goal position
            control_points: Array of shape (n_control, 2) - intermediate points
            
        Returns:
            List of (x, y) waypoints
        """
        # Build full point sequence: start -> control points -> goal
        all_points = np.vstack([
            [start],
            control_points.reshape(-1, 2),
            [goal]
        ])
        
        n_points = len(all_points)
        t = np.linspace(0, 1, n_points)
        t_fine = np.linspace(0, 1, self.resolution)
        
        # Fit cubic splines
        cs_x = CubicSpline(t, all_points[:, 0])
        cs_y = CubicSpline(t, all_points[:, 1])
        
        # Generate path
        path_x = cs_x(t_fine)
        path_y = cs_y(t_fine)
        
        return [(float(x), float(y)) for x, y in zip(path_x, path_y)]
    
    def _evaluate_cost(self, path: List[Tuple[float, float]]) -> float:
        """
        Evaluate cost of a path (length + collision penalty).
        """
        # Path length
        length = 0
        for i in range(len(path) - 1):
            length += np.hypot(path[i+1][0] - path[i][0], 
                              path[i+1][1] - path[i][1])
        
        # Collision penalty
        collision_penalty = 0
        for x, y in path:
            if not self.env.is_valid(x, y, self.robot_radius):
                collision_penalty += 100  # Heavy penalty for collisions
        
        # Smoothness penalty (penalize sharp turns)
        smoothness_penalty = 0
        if len(path) >= 3:
            for i in range(1, len(path) - 1):
                v1 = np.array([path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]])
                v2 = np.array([path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]])
                
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 > 0.01 and norm2 > 0.01:
                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    smoothness_penalty += angle * 0.5
        
        return length + collision_penalty + smoothness_penalty
    
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using PSO.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of waypoints or None
        """
        # Validate
        if not self.env.is_valid(start[0], start[1], self.robot_radius):
            print("PSO Error: Start is invalid!")
            return None
        if not self.env.is_valid(goal[0], goal[1], self.robot_radius):
            print("PSO Error: Goal is invalid!")
            return None
        
        print(f"PSO: Optimizing with {self.n_particles} particles, {self.max_iter} iterations...")
        
        # Dimension: n_control_points * 2 (x, y for each)
        dim = self.n_control * 2
        
        # Initialize particles - spread between start and goal
        particles = np.zeros((self.n_particles, dim))
        for i in range(self.n_particles):
            for j in range(self.n_control):
                # Interpolate between start and goal with random offset
                t = (j + 1) / (self.n_control + 1)
                base_x = start[0] + t * (goal[0] - start[0])
                base_y = start[1] + t * (goal[1] - start[1])
                
                # Add random perturbation
                particles[i, 2*j] = base_x + np.random.uniform(-5, 5)
                particles[i, 2*j + 1] = base_y + np.random.uniform(-5, 5)
                
                # Clamp to environment bounds
                particles[i, 2*j] = np.clip(particles[i, 2*j], 1, self.env.width - 1)
                particles[i, 2*j + 1] = np.clip(particles[i, 2*j + 1], 1, self.env.height - 1)
        
        # Initialize velocities
        velocities = np.random.uniform(-1, 1, (self.n_particles, dim))
        
        # Evaluate initial costs
        costs = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            control_pts = particles[i].reshape(-1, 2)
            path = self._generate_path_from_control_points(start, goal, control_pts)
            costs[i] = self._evaluate_cost(path)
        
        # Personal best
        pbest_positions = particles.copy()
        pbest_costs = costs.copy()
        
        # Global best
        gbest_idx = np.argmin(costs)
        gbest_position = particles[gbest_idx].copy()
        gbest_cost = costs[gbest_idx]
        
        self.best_costs = [gbest_cost]
        w = self.w
        
        # Main PSO loop
        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                # Update velocity
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)
                
                cognitive = self.c1 * r1 * (pbest_positions[i] - particles[i])
                social = self.c2 * r2 * (gbest_position - particles[i])
                
                velocities[i] = w * velocities[i] + cognitive + social
                
                # Update position
                particles[i] += velocities[i]
                
                # Clamp to bounds
                particles[i][0::2] = np.clip(particles[i][0::2], 1, self.env.width - 1)
                particles[i][1::2] = np.clip(particles[i][1::2], 1, self.env.height - 1)
                
                # Evaluate cost
                control_pts = particles[i].reshape(-1, 2)
                path = self._generate_path_from_control_points(start, goal, control_pts)
                cost = self._evaluate_cost(path)
                costs[i] = cost
                
                # Update personal best
                if cost < pbest_costs[i]:
                    pbest_positions[i] = particles[i].copy()
                    pbest_costs[i] = cost
                    
                    # Update global best
                    if cost < gbest_cost:
                        gbest_position = particles[i].copy()
                        gbest_cost = cost
            
            # Dampen inertia
            w *= self.w_damp
            
            self.best_costs.append(gbest_cost)
            
            if (iteration + 1) % 30 == 0:
                print(f"  Iteration {iteration + 1}: Best cost = {gbest_cost:.2f}")
        
        # Generate final path
        best_control_pts = gbest_position.reshape(-1, 2)
        best_path = self._generate_path_from_control_points(start, goal, best_control_pts)
        
        # Check if path is valid
        collision_free = all(self.env.is_valid(p[0], p[1], self.robot_radius) for p in best_path)
        
        if collision_free:
            self.path = best_path
            print(f"PSO found collision-free path with {len(best_path)} points")
            return best_path
        else:
            print("PSO Warning: Best path has collisions. Returning anyway for visualization.")
            self.path = best_path
            return best_path


# ============== Demo ==============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment import create_demo_environment
    from visualize import plot_environment, plot_start_goal, plot_path
    
    print("=" * 50)
    print("PSO Path Planning Demo")
    print("=" * 50)
    
    # Create environment
    env = create_demo_environment()
    
    start = (3, 3)
    goal = (45, 45)
    
    # Create planner
    planner = PSOPlanner(env, robot_radius=0.5, num_particles=60,
                        max_iterations=100, num_control_points=5,
                        path_resolution=80)
    
    import time
    t_start = time.time()
    path = planner.plan(start, goal)
    t_end = time.time()
    
    print(f"Planning time: {t_end - t_start:.4f} seconds")
    
    if path:
        path_length = sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) 
                         for i in range(len(path)-1))
        print(f"Path length: {path_length:.2f} meters")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Path plot
    plot_environment(ax1, env, "PSO Path Planning")
    if path:
        plot_path(ax1, path, algorithm='PSO', linewidth=3, label='PSO Path')
    plot_start_goal(ax1, start, goal)
    ax1.legend(loc='upper right')
    
    # Right: Convergence plot
    ax2.plot(planner.best_costs, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Cost')
    ax2.set_title('PSO Convergence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pso_result.png', dpi=150)
    print("Saved result to pso_result.png")
    plt.show()
