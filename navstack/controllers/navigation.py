"""
Navigation Controllers for Mobile Robot Path Following
Includes: Pure Pursuit, Stanley, Proportional, DWA, and GA Auto-Tuner
"""
import numpy as np
from abc import ABC, abstractmethod


class BaseNavController(ABC):
    """Base class for navigation controllers."""
    
    def __init__(self, name="BaseController"):
        self.name = name
    
    @abstractmethod
    def compute_control(self, pose, path, current_idx):
        """
        Compute control commands for path following.
        
        Args:
            pose: (x, y, theta) current robot pose
            path: List of (x, y) waypoints
            current_idx: Current waypoint index
            
        Returns:
            v: Linear velocity command
            omega: Angular velocity command
            new_idx: Updated waypoint index
        """
        pass
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def distance_to_point(self, pose, point):
        """Calculate distance from pose to point."""
        return np.sqrt((point[0] - pose[0])**2 + (point[1] - pose[1])**2)


class PurePursuit(BaseNavController):
    """
    Pure Pursuit Controller - Geometric path following.
    Tracks a lookahead point on the path.
    """
    
    def __init__(self, lookahead=0.5, v_max=1.0, min_lookahead=0.2, max_lookahead=1.5):
        super().__init__("PurePursuit")
        self.lookahead = lookahead
        self.v_max = v_max
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.wheelbase = 0.3  # For ackermann/bicycle
    
    def find_lookahead_point(self, pose, path, current_idx):
        """Find the lookahead point on the path."""
        x, y, theta = pose
        
        # Search from current index
        for i in range(current_idx, len(path)):
            dist = self.distance_to_point(pose, path[i])
            if dist >= self.lookahead:
                return path[i], i
        
        # If no point found, return last point
        return path[-1], len(path) - 1
    
    def compute_control(self, pose, path, current_idx):
        x, y, theta = pose
        
        # Check if close to final goal
        final_goal = path[-1]
        dist_to_goal = self.distance_to_point(pose, final_goal)
        if dist_to_goal < 0.1:  # Reached goal
            return 0.0, 0.0, len(path) - 1
        
        # Clamp current_idx to valid range
        current_idx = min(current_idx, len(path) - 1)
        
        # Find lookahead point (or use final goal if at last waypoint)
        if current_idx >= len(path) - 1:
            target = path[-1]
            new_idx = len(path) - 1
        else:
            target, new_idx = self.find_lookahead_point(pose, path, current_idx)
        
        # Calculate steering
        dx = target[0] - x
        dy = target[1] - y
        target_angle = np.arctan2(dy, dx)
        alpha = self.normalize_angle(target_angle - theta)
        
        # Pure pursuit curvature
        ld = self.distance_to_point(pose, target)
        if ld < 0.05:
            return 0.0, 0.0, new_idx
        
        curvature = 2 * np.sin(alpha) / ld
        
        if ld < 0.05:
            return 0.0, 0.0, new_idx
        
        curvature = 2 * np.sin(alpha) / ld
        
        if ld < 0.05:
            return 0.0, 0.0, new_idx
        
        curvature = 2 * np.sin(alpha) / ld
        
        # Smooth velocity control (Cosine scaling)
        # Allows robot to move while turning, prevents 2WD stall
        v_scale = max(0.4, np.cos(alpha))
        if abs(alpha) > np.deg2rad(60):
            v_scale = 0.2  # Slow down for sharp turns
            
        v = self.v_max * v_scale
        
        # Angular velocity
        omega = v * curvature
        
        # Boost turning if heading error is large (helps 4WD/Skid-steer)
        if abs(alpha) > np.deg2rad(30):
            # Add extra turning component independent of velocity
            # This ensures omni/mecanum/4wd turn even if v is small
            omega += np.sign(alpha) * 0.5
        
        return v, omega, new_idx


class StanleyController(BaseNavController):
    """
    Stanley Controller - Front axle error based.
    Uses cross-track error and heading error.
    """
    
    def __init__(self, k_gain=2.5, k_soft=1.0, v_max=1.0):
        super().__init__("Stanley")
        self.k = k_gain  # Cross-track gain
        self.k_soft = k_soft  # Softening constant
        self.v_max = v_max
    
    def find_closest_point(self, pose, path, current_idx):
        """Find closest point on path to the front axle."""
        min_dist = float('inf')
        closest_idx = current_idx
        
        # Search nearby points
        search_range = min(len(path), current_idx + 20)
        for i in range(current_idx, search_range):
            dist = self.distance_to_point(pose, path[i])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx, min_dist
    
    def compute_control(self, pose, path, current_idx):
        x, y, theta = pose
        
        if current_idx >= len(path) - 1:
            return 0.0, 0.0, current_idx
        
        # Find closest path point
        closest_idx, cross_track_error = self.find_closest_point(pose, path, current_idx)
        
        # Calculate path heading at closest point
        if closest_idx < len(path) - 1:
            dx = path[closest_idx + 1][0] - path[closest_idx][0]
            dy = path[closest_idx + 1][1] - path[closest_idx][1]
            path_heading = np.arctan2(dy, dx)
        else:
            path_heading = theta
        
        # Heading error
        heading_error = self.normalize_angle(path_heading - theta)
        
        # Cross-track error sign (left/right of path)
        path_vec = np.array([np.cos(path_heading), np.sin(path_heading)])
        to_robot = np.array([x - path[closest_idx][0], y - path[closest_idx][1]])
        cross_product = path_vec[0] * to_robot[1] - path_vec[1] * to_robot[0]
        signed_cte = cross_track_error * np.sign(cross_product)
        
        # Linear velocity
        v = self.v_max * (1 - 0.3 * abs(heading_error))
        v = max(0.1, min(v, self.v_max))
        
        # Stanley steering law
        # delta = heading_error + arctan(k * cte / (k_soft + v))
        steering = heading_error + np.arctan(self.k * signed_cte / (self.k_soft + v))
        
        # Convert to angular velocity
        omega = v * np.tan(steering) / 0.3  # Approximate wheelbase
        
        # Update waypoint index
        new_idx = closest_idx
        if closest_idx > current_idx:
            new_idx = closest_idx
        
        return v, omega, new_idx


class ProportionalNav(BaseNavController):
    """
    Proportional Navigation Controller.
    Simple but effective for point-to-point navigation.
    """
    
    def __init__(self, k_rho=1.0, k_alpha=3.0, k_beta=-0.5, v_max=1.0):
        super().__init__("Proportional")
        self.k_rho = k_rho      # Distance gain
        self.k_alpha = k_alpha  # Heading-to-goal gain
        self.k_beta = k_beta    # Final orientation gain
        self.v_max = v_max
        self.goal_threshold = 0.3
    
    def compute_control(self, pose, path, current_idx):
        x, y, theta = pose
        
        if current_idx >= len(path):
            return 0.0, 0.0, current_idx
        
        # Target point
        target = path[min(current_idx, len(path) - 1)]
        
        # Distance and angle to target
        dx = target[0] - x
        dy = target[1] - y
        rho = np.sqrt(dx**2 + dy**2)
        
        # Check if reached waypoint
        if rho < self.goal_threshold and current_idx < len(path) - 1:
            return self.compute_control(pose, path, current_idx + 1)
        
        if rho < 0.1:  # Very close to final goal
            return 0.0, 0.0, current_idx
        
        # Angle to target
        target_angle = np.arctan2(dy, dx)
        alpha = self.normalize_angle(target_angle - theta)
        
        # Control law
        v = self.k_rho * rho
        v = min(v, self.v_max)
        
        # Reduce speed when not aligned
        if abs(alpha) > np.pi / 4:
            v *= 0.3
        
        omega = self.k_alpha * alpha
        
        return v, omega, current_idx


class DWAController(BaseNavController):
    """
    Dynamic Window Approach Controller.
    Samples velocity space and evaluates trajectories.
    """
    
    def __init__(self, v_max=1.0, w_max=2.0, predict_time=2.0,
                 goal_weight=1.0, speed_weight=0.5, obstacle_weight=1.0,
                 v_samples=11, w_samples=21):
        super().__init__("DWA")
        self.v_max = v_max
        self.w_max = w_max
        self.predict_time = predict_time
        self.goal_weight = goal_weight
        self.speed_weight = speed_weight
        self.obstacle_weight = obstacle_weight
        self.v_samples = v_samples
        self.w_samples = w_samples
        self.obstacles = []
        self.robot_radius = 0.2
        self.dt = 0.1
        self.goal_threshold = 0.5
    
    def set_obstacles(self, obstacles):
        """Set obstacle list for collision checking."""
        self.obstacles = obstacles
    
    def predict_trajectory(self, pose, v, w):
        """Predict trajectory for given control inputs."""
        x, y, theta = pose
        trajectory = [(x, y, theta)]
        
        t = 0
        while t < self.predict_time:
            theta += w * self.dt
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            trajectory.append((x, y, theta))
            t += self.dt
        
        return trajectory
    
    def calc_goal_cost(self, trajectory, goal):
        """Cost based on distance to goal at end of trajectory."""
        end_x, end_y, _ = trajectory[-1]
        dist = np.sqrt((goal[0] - end_x)**2 + (goal[1] - end_y)**2)
        return dist
    
    def calc_speed_cost(self, v):
        """Reward higher speeds."""
        return self.v_max - v
    
    def calc_obstacle_cost(self, trajectory):
        """Cost based on proximity to obstacles."""
        min_dist = float('inf')
        
        for x, y, _ in trajectory:
            for obs in self.obstacles:
                if isinstance(obs, dict):
                    ox, oy = obs['pos'][0], obs['pos'][1]
                    if obs.get('type') == 'cylinder':
                        obs_r = obs['size'][0]
                    else:
                        obs_r = max(obs['size'][0], obs['size'][1]) / 2
                else:
                    ox, oy, obs_r = obs[0], obs[1], obs[2] if len(obs) > 2 else 0.5
                
                dist = np.sqrt((x - ox)**2 + (y - oy)**2) - obs_r - self.robot_radius
                if dist < 0:
                    return float('inf')  # Collision
                min_dist = min(min_dist, dist)
        
        if min_dist == float('inf'):
            return 0
        return 1.0 / (min_dist + 0.1)
    
    def compute_control(self, pose, path, current_idx):
        x, y, theta = pose
        
        if current_idx >= len(path):
            return 0.0, 0.0, current_idx
        
        goal = path[min(current_idx, len(path) - 1)]
        
        # Check if goal reached
        dist_to_goal = self.distance_to_point(pose, goal)
        if dist_to_goal < self.goal_threshold and current_idx < len(path) - 1:
            current_idx += 1
            goal = path[min(current_idx, len(path) - 1)]
        
        if dist_to_goal < 0.1:
            return 0.0, 0.0, current_idx
        
        # Sample velocity space
        best_v, best_w = 0, 0
        min_cost = float('inf')
        
        for v in np.linspace(0, self.v_max, self.v_samples):
            for w in np.linspace(-self.w_max, self.w_max, self.w_samples):
                # Predict trajectory
                traj = self.predict_trajectory(pose, v, w)
                
                # Calculate costs
                goal_cost = self.goal_weight * self.calc_goal_cost(traj, goal)
                speed_cost = self.speed_weight * self.calc_speed_cost(v)
                obs_cost = self.obstacle_weight * self.calc_obstacle_cost(traj)
                
                total_cost = goal_cost + speed_cost + obs_cost
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_v, best_w = v, w
        
        return best_v, best_w, current_idx


class GAControllerTuner:
    """
    Genetic Algorithm for auto-tuning controller gains.
    Optimizes controller parameters to minimize path following error.
    """
    
    def __init__(self, controller_type, population_size=20, generations=30,
                 mutation_rate=0.2, crossover_rate=0.7):
        self.controller_type = controller_type
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Parameter ranges for each controller type
        self.param_ranges = {
            'PurePursuit': {
                'lookahead': (0.2, 2.0),
                'v_max': (0.5, 2.0)
            },
            'Stanley': {
                'k_gain': (0.5, 10.0),
                'k_soft': (0.1, 5.0),
                'v_max': (0.5, 2.0)
            },
            'Proportional': {
                'k_rho': (0.5, 3.0),
                'k_alpha': (1.0, 10.0),
                'k_beta': (-2.0, 0.0),
                'v_max': (0.5, 2.0)
            },
            'DWA': {
                'v_max': (0.5, 2.0),
                'w_max': (1.0, 4.0),
                'predict_time': (1.0, 4.0),
                'goal_weight': (0.5, 3.0),
                'speed_weight': (0.1, 1.5),
                'obstacle_weight': (0.5, 3.0)
            }
        }
        
        self.test_path = None
        self.test_obstacles = []
    
    def set_test_scenario(self, path, obstacles=None):
        """Set the path and obstacles for fitness evaluation."""
        self.test_path = path
        self.test_obstacles = obstacles or []
    
    def create_individual(self):
        """Create a random individual (set of gains)."""
        params = self.param_ranges.get(self.controller_type, {})
        individual = {}
        for param, (low, high) in params.items():
            individual[param] = np.random.uniform(low, high)
        return individual
    
    def create_controller(self, individual):
        """Create a controller with given parameters."""
        if self.controller_type == 'PurePursuit':
            return PurePursuit(**individual)
        elif self.controller_type == 'Stanley':
            return StanleyController(**individual)
        elif self.controller_type == 'Proportional':
            return ProportionalNav(**individual)
        elif self.controller_type == 'DWA':
            ctrl = DWAController(**individual)
            ctrl.set_obstacles(self.test_obstacles)
            return ctrl
        return None
    
    def evaluate_fitness(self, individual):
        """
        Evaluate fitness of an individual by simulating path following.
        Lower is better (minimizing error).
        """
        if self.test_path is None or len(self.test_path) < 2:
            return float('inf')
        
        controller = self.create_controller(individual)
        if controller is None:
            return float('inf')
        
        # Simple kinematic simulation
        x, y, theta = self.test_path[0][0], self.test_path[0][1], 0
        idx = 1
        total_error = 0
        steps = 0
        max_steps = 500
        dt = 0.1
        
        while idx < len(self.test_path) and steps < max_steps:
            v, omega, idx = controller.compute_control((x, y, theta), self.test_path, idx)
            
            # Update pose
            theta += omega * dt
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            
            # Calculate error to current target
            target = self.test_path[min(idx, len(self.test_path) - 1)]
            error = np.sqrt((target[0] - x)**2 + (target[1] - y)**2)
            total_error += error
            steps += 1
            
            if v < 0.01 and idx >= len(self.test_path) - 1:
                break
        
        # Penalize if didn't reach goal
        final_target = self.test_path[-1]
        final_error = np.sqrt((final_target[0] - x)**2 + (final_target[1] - y)**2)
        
        fitness = total_error / max(1, steps) + 5 * final_error
        return fitness
    
    def crossover(self, parent1, parent2):
        """Single-point crossover."""
        child = {}
        keys = list(parent1.keys())
        crossover_point = np.random.randint(0, len(keys))
        
        for i, key in enumerate(keys):
            if i < crossover_point:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        return child
    
    def mutate(self, individual):
        """Gaussian mutation."""
        params = self.param_ranges.get(self.controller_type, {})
        mutated = individual.copy()
        
        for param, value in mutated.items():
            if np.random.random() < self.mutation_rate:
                low, high = params[param]
                sigma = (high - low) * 0.1
                mutated[param] = np.clip(value + np.random.normal(0, sigma), low, high)
        
        return mutated
    
    def optimize(self, verbose=True):
        """
        Run genetic algorithm optimization.
        
        Returns:
            dict: Best parameters found
        """
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        best_individual = None
        best_fitness = float('inf')
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(ind) for ind in population]
            
            # Track best
            gen_best_idx = np.argmin(fitness_scores)
            if fitness_scores[gen_best_idx] < best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            if verbose and gen % 5 == 0:
                print(f"Gen {gen}: Best Fitness = {best_fitness:.4f}")
            
            # Selection (tournament)
            new_population = []
            for _ in range(self.population_size):
                # Tournament selection
                i1, i2 = np.random.randint(0, len(population), 2)
                if fitness_scores[i1] < fitness_scores[i2]:
                    parent1 = population[i1]
                else:
                    parent1 = population[i2]
                
                i1, i2 = np.random.randint(0, len(population), 2)
                if fitness_scores[i1] < fitness_scores[i2]:
                    parent2 = population[i1]
                else:
                    parent2 = population[i2]
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                child = self.mutate(child)
                new_population.append(child)
            
            # Elitism: keep best
            new_population[0] = best_individual.copy()
            population = new_population
        
        if verbose:
            print(f"\nOptimization Complete!")
            print(f"Best Fitness: {best_fitness:.4f}")
            print(f"Best Parameters: {best_individual}")
        
        return best_individual


# Factory function to get controller by name
def get_controller(name, **kwargs):
    """
    Factory function to create controller by name.
    
    Args:
        name: Controller name ('PurePursuit', 'Stanley', 'Proportional', 'DWA', 'MPC')
        **kwargs: Controller-specific parameters

    Returns:
        A controller exposing compute_control(pose, path, idx) -> (v, omega, idx)
    """
    if name == "MPC":
        # lazy import so the cvxpy dependency is only required when MPC is used
        from navstack.controllers.mpc_follower import MPCPathFollower
        return MPCPathFollower(**kwargs)

    controllers = {
        'PurePursuit': PurePursuit,
        'Stanley': StanleyController,
        'Proportional': ProportionalNav,
        'DWA': DWAController
    }

    controller_class = controllers.get(name)
    if controller_class is None:
        raise ValueError(f"Unknown controller: {name}")

    return controller_class(**kwargs)


if __name__ == "__main__":
    # Test controllers
    test_path = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (5, 3)]
    
    print("Testing Navigation Controllers...")
    
    for name in ['PurePursuit', 'Stanley', 'Proportional', 'DWA']:
        ctrl = get_controller(name)
        v, w, _ = ctrl.compute_control((0, 0, 0), test_path, 1)
        print(f"{name}: v={v:.2f}, omega={w:.2f}")
    
    # Test GA tuner
    print("\nTesting GA Tuner (quick test with 5 generations)...")
    tuner = GAControllerTuner('PurePursuit', population_size=10, generations=5)
    tuner.set_test_scenario(test_path)
    best_params = tuner.optimize(verbose=True)
