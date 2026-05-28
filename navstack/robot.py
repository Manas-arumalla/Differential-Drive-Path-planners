# robot.py
"""
Simple differential drive robot model.
"""
import numpy as np
from typing import Tuple

class DifferentialDriveRobot:
    """Simple differential drive mobile robot."""
    
    def __init__(self, x: float = 0, y: float = 0, theta: float = 0,
                 radius: float = 0.3, max_speed: float = 1.0, max_omega: float = 1.5):
        """
        Initialize robot.
        
        Args:
            x, y: Initial position (meters)
            theta: Initial heading (radians)
            radius: Robot radius for collision checking
            max_speed: Maximum linear velocity (m/s)
            max_omega: Maximum angular velocity (rad/s)
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.radius = radius
        self.max_speed = max_speed
        self.max_omega = max_omega
        
        # Current velocities
        self.v = 0.0
        self.omega = 0.0
        
        # History for visualization
        self.trajectory: list = []
        self.trajectory.append((x, y))
    
    def set_pose(self, x: float, y: float, theta: float):
        """Set robot pose."""
        self.x = x
        self.y = y
        self.theta = theta
        self.trajectory = [(x, y)]
    
    def get_pose(self) -> Tuple[float, float, float]:
        """Get current pose (x, y, theta)."""
        return self.x, self.y, self.theta
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position (x, y)."""
        return self.x, self.y
    
    def step(self, v: float, omega: float, dt: float = 0.1):
        """
        Update robot state with velocity commands.
        
        Args:
            v: Linear velocity command (m/s)
            omega: Angular velocity command (rad/s)
            dt: Time step (seconds)
        """
        # Clamp velocities
        v = np.clip(v, -self.max_speed, self.max_speed)
        omega = np.clip(omega, -self.max_omega, self.max_omega)
        
        self.v = v
        self.omega = omega
        
        # Update pose using simple Euler integration
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        
        # Normalize theta to [-pi, pi]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        # Record trajectory
        self.trajectory.append((self.x, self.y))
    
    def move_to_point(self, target_x: float, target_y: float, 
                      k_rho: float = 0.5, k_alpha: float = 1.5,
                      dt: float = 0.1) -> Tuple[float, float]:
        """
        Simple proportional control to move toward a target point.
        
        Args:
            target_x, target_y: Target position
            k_rho: Gain for distance
            k_alpha: Gain for heading
            dt: Time step
            
        Returns:
            (v, omega): Velocity commands applied
        """
        dx = target_x - self.x
        dy = target_y - self.y
        
        rho = np.hypot(dx, dy)  # Distance to target
        alpha = np.arctan2(dy, dx) - self.theta  # Heading error
        
        # Normalize alpha to [-pi, pi]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        # Proportional control
        v = k_rho * rho
        omega = k_alpha * alpha
        
        # Apply motion
        self.step(v, omega, dt)
        
        return v, omega
    
    def distance_to(self, x: float, y: float) -> float:
        """Calculate distance to a point."""
        return np.hypot(x - self.x, y - self.y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test robot motion
    robot = DifferentialDriveRobot(x=5, y=5, theta=0)
    target = (40, 40)
    
    # Simulate motion
    for _ in range(200):
        if robot.distance_to(*target) < 0.5:
            break
        robot.move_to_point(*target)
    
    # Plot trajectory
    traj = np.array(robot.trajectory)
    plt.figure(figsize=(8, 8))
    plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory')
    plt.plot(5, 5, 'go', markersize=15, label='Start')
    plt.plot(target[0], target[1], 'r*', markersize=20, label='Goal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Differential Drive Robot - Simple Navigation')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
