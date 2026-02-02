# planners/apf_planner.py
"""
APF (Artificial Potential Field) Path Planning Algorithm

Reactive path planning using attractive potential toward goal
and repulsive potential from obstacles.
"""
import numpy as np
from typing import List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class APFPlanner:
    """
    Artificial Potential Field path planner.
    
    Uses gradient descent on a potential field where:
    - Goal creates attractive potential
    - Obstacles create repulsive potential
    """
    
    def __init__(self, environment, robot_radius: float = 0.3,
                 attractive_gain: float = 5.0, repulsive_gain: float = 100.0,
                 influence_distance: float = 5.0, step_size: float = 0.3,
                 max_iterations: int = 2000):
        """
        Initialize APF planner.
        
        Args:
            environment: Environment object
            robot_radius: Robot radius
            attractive_gain: Gain for attractive potential (goal)
            repulsive_gain: Gain for repulsive potential (obstacles)
            influence_distance: Distance within which obstacles affect robot
            step_size: Step size for gradient descent
            max_iterations: Maximum iterations
        """
        self.env = environment
        self.robot_radius = robot_radius
        self.k_att = attractive_gain
        self.k_rep = repulsive_gain
        self.d0 = influence_distance
        self.step_size = step_size
        self.max_iter = max_iterations
        
        self.path = []
        
    def _attractive_force(self, pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Calculate attractive force toward goal.
        Uses conic potential for large distances, parabolic near goal.
        """
        diff = goal - pos
        dist = np.linalg.norm(diff)
        
        if dist < 0.01:
            return np.array([0.0, 0.0])
        
        # Parabolic near goal (within 5m), conic otherwise
        threshold = 5.0
        if dist <= threshold:
            # Parabolic: F = k_att * (goal - pos)
            force = self.k_att * diff
        else:
            # Conic: constant magnitude, direction toward goal
            force = self.k_att * threshold * diff / dist
        
        return force
    
    def _repulsive_force(self, pos: np.ndarray) -> np.ndarray:
        """
        Calculate repulsive force from nearby obstacles.
        """
        force = np.array([0.0, 0.0])
        
        # Sample obstacle cells
        gx, gy = self.env.world_to_grid(pos[0], pos[1])
        search_radius = int(self.d0 / self.env.resolution) + 1
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                check_gx = gx + dx
                check_gy = gy + dy
                
                if not (0 <= check_gx < self.env.grid_width and 
                        0 <= check_gy < self.env.grid_height):
                    continue
                
                if not self.env.grid[check_gy, check_gx]:
                    continue
                
                # Obstacle cell found
                obs_x, obs_y = self.env.grid_to_world(check_gx, check_gy)
                diff = pos - np.array([obs_x, obs_y])
                dist = np.linalg.norm(diff)
                
                if dist < 0.1:
                    dist = 0.1  # Avoid division by zero
                
                if dist <= self.d0:
                    # Repulsive force magnitude
                    magnitude = self.k_rep * (1.0/dist - 1.0/self.d0) * (1.0 / dist**2)
                    direction = diff / dist
                    force += magnitude * direction
        
        return force
    
    def _total_force(self, pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Calculate total force (attractive + repulsive)."""
        f_att = self._attractive_force(pos, goal)
        f_rep = self._repulsive_force(pos)
        return f_att + f_rep
    
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float],
             goal_threshold: float = 0.5) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using APF.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            goal_threshold: Distance to consider goal reached
            
        Returns:
            List of waypoints or None
        """
        # Validate
        if not self.env.is_valid(start[0], start[1], self.robot_radius):
            print("APF Error: Start is invalid!")
            return None
        if not self.env.is_valid(goal[0], goal[1], self.robot_radius):
            print("APF Error: Goal is invalid!")
            return None
        
        pos = np.array(start, dtype=float)
        goal_arr = np.array(goal, dtype=float)
        
        path = [(float(pos[0]), float(pos[1]))]
        
        # Oscillation detection
        history = []
        oscillation_window = 20
        
        for iteration in range(self.max_iter):
            # Check goal
            dist_to_goal = np.linalg.norm(pos - goal_arr)
            if dist_to_goal < goal_threshold:
                path.append(goal)
                self.path = path
                print(f"APF reached goal in {iteration} iterations")
                return path
            
            # Calculate total force
            force = self._total_force(pos, goal_arr)
            force_mag = np.linalg.norm(force)
            
            if force_mag < 1e-6:
                print("APF Warning: Stuck in local minimum!")
                # Try random perturbation
                pos += np.random.uniform(-0.5, 0.5, 2)
                continue
            
            # Normalize and step
            direction = force / force_mag
            new_pos = pos + self.step_size * direction
            
            # Check if new position is valid
            if self.env.is_valid(new_pos[0], new_pos[1], self.robot_radius):
                pos = new_pos
                path.append((float(pos[0]), float(pos[1])))
            else:
                # Try smaller step or tangent direction
                for scale in [0.5, 0.3, 0.1]:
                    test_pos = pos + scale * self.step_size * direction
                    if self.env.is_valid(test_pos[0], test_pos[1], self.robot_radius):
                        pos = test_pos
                        path.append((float(pos[0]), float(pos[1])))
                        break
                else:
                    # Try perpendicular directions
                    perp1 = np.array([-direction[1], direction[0]])
                    perp2 = np.array([direction[1], -direction[0]])
                    
                    for perp in [perp1, perp2]:
                        test_pos = pos + self.step_size * perp
                        if self.env.is_valid(test_pos[0], test_pos[1], self.robot_radius):
                            pos = test_pos
                            path.append((float(pos[0]), float(pos[1])))
                            break
            
            # Oscillation detection
            history.append((pos[0], pos[1]))
            if len(history) > oscillation_window:
                history.pop(0)
                
                if len(history) >= oscillation_window:
                    variance = np.var(history, axis=0).sum()
                    if variance < 0.1:
                        print("APF Warning: Oscillation detected, adding random perturbation")
                        pos += np.random.uniform(-1, 1, 2)
                        pos[0] = np.clip(pos[0], 1, self.env.width - 1)
                        pos[1] = np.clip(pos[1], 1, self.env.height - 1)
                        history = []
        
        print(f"APF Warning: Max iterations reached. Distance to goal: {dist_to_goal:.2f}")
        self.path = path
        return path if len(path) > 1 else None
    
    def simplify_path(self, path: List[Tuple[float, float]], 
                     tolerance: float = 0.3) -> List[Tuple[float, float]]:
        """
        Simplify path by removing redundant points.
        Uses Douglas-Peucker-like simplification.
        """
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        last_added = 0
        
        for i in range(1, len(path) - 1):
            # Check if we can skip this point
            if self.env.is_path_valid(simplified[-1], path[i + 1], self.robot_radius):
                continue
            else:
                simplified.append(path[i])
        
        simplified.append(path[-1])
        return simplified


# ============== Demo ==============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment import create_demo_environment
    from visualize import plot_environment, plot_start_goal, plot_path
    
    print("=" * 50)
    print("APF Path Planning Demo")
    print("=" * 50)
    
    # Create environment
    env = create_demo_environment()
    
    start = (3, 3)
    goal = (45, 45)
    
    # Create planner
    planner = APFPlanner(env, robot_radius=0.5, attractive_gain=5.0,
                        repulsive_gain=150.0, influence_distance=4.0,
                        step_size=0.4, max_iterations=3000)
    
    import time
    t_start = time.time()
    path = planner.plan(start, goal)
    t_end = time.time()
    
    print(f"Planning time: {t_end - t_start:.4f} seconds")
    
    if path:
        # Raw path length
        raw_length = sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) 
                        for i in range(len(path)-1))
        print(f"Raw path: {len(path)} points, length: {raw_length:.2f} meters")
        
        # Simplify
        simplified = planner.simplify_path(path)
        simp_length = sum(np.hypot(simplified[i+1][0]-simplified[i][0], 
                                   simplified[i+1][1]-simplified[i][1]) 
                         for i in range(len(simplified)-1))
        print(f"Simplified path: {len(simplified)} points, length: {simp_length:.2f} meters")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_environment(ax, env, "APF Path Planning")
    
    if path:
        # Plot raw path (faded)
        plot_path(ax, path, algorithm='APF', linewidth=1, label=f'APF Raw ({len(path)} pts)', alpha=0.3)
        # Plot simplified path
        simplified = planner.simplify_path(path)
        plot_path(ax, simplified, algorithm='APF', linewidth=3, label=f'APF Simplified ({len(simplified)} pts)')
    
    plot_start_goal(ax, start, goal)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('apf_result.png', dpi=150)
    print("Saved result to apf_result.png")
    plt.show()
