# visualize.py
"""
Enhanced visualization utilities for path planning algorithms.
Features modern, visually appealing plots with gradients and animations.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Wedge
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib import colors as mcolors
from typing import List, Tuple, Optional
import matplotlib.patheffects as path_effects

# Set modern style
plt.style.use('dark_background')

# Color schemes for different algorithms
ALGORITHM_COLORS = {
    'A*': {'primary': '#00D4FF', 'secondary': '#0088AA', 'gradient': ['#00D4FF', '#0088AA']},
    'RRT': {'primary': '#FF6B6B', 'secondary': '#CC4444', 'gradient': ['#FF6B6B', '#FF3333']},
    'RRT*': {'primary': '#A855F7', 'secondary': '#7C3AED', 'gradient': ['#A855F7', '#7C3AED']},
    'PSO': {'primary': '#FFA500', 'secondary': '#FF8C00', 'gradient': ['#FFD700', '#FFA500']},
    'APF': {'primary': '#00FF88', 'secondary': '#00CC66', 'gradient': ['#00FF88', '#00CC66']},
    'Dijkstra': {'primary': '#FFD700', 'secondary': '#DAA520', 'gradient': ['#FFD700', '#FFA500']},
    'PRM': {'primary': '#FF69B4', 'secondary': '#FF1493', 'gradient': ['#FF69B4', '#FF1493']},
}


def create_gradient_colormap(color1, color2, n=256):
    """Create a gradient colormap between two colors."""
    c1 = np.array(mcolors.to_rgb(color1))
    c2 = np.array(mcolors.to_rgb(color2))
    gradient = np.linspace(c1, c2, n)
    return mcolors.ListedColormap(gradient)


def plot_environment(ax, env, title: str = "Environment", style: str = 'dark'):
    """Plot environment with enhanced styling."""
    # Create custom colormap for obstacles
    if style == 'dark':
        obstacle_color = '#2D3748'
        bg_color = '#1A202C'
        grid_color = '#4A5568'
    else:
        obstacle_color = '#718096'
        bg_color = '#F7FAFC'
        grid_color = '#CBD5E0'
    
    ax.set_facecolor(bg_color)
    
    # Plot obstacles with gradient effect
    obstacle_cmap = create_gradient_colormap('#4A5568', '#2D3748')
    ax.imshow(env.grid, origin='lower',
              extent=[0, env.width, 0, env.height],
              cmap=obstacle_cmap, alpha=0.9, interpolation='nearest')
    
    # Add grid lines
    ax.set_xlabel('X (meters)', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=12, color='white', fontweight='bold')
    ax.set_title(title, fontsize=14, color='white', fontweight='bold', pad=15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, color=grid_color, linestyle='--', linewidth=0.5)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    
    # Style tick labels
    ax.tick_params(colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#4A5568')
        spine.set_linewidth(2)


def plot_start_goal(ax, start: Tuple[float, float], goal: Tuple[float, float]):
    """Plot start and goal with glowing effects."""
    # Start marker with glow
    glow_start = Circle(start, 2.5, color='#00FF00', alpha=0.2, zorder=4)
    ax.add_patch(glow_start)
    ax.plot(start[0], start[1], 'o', markersize=18, color='#00FF00', 
            markeredgecolor='white', markeredgewidth=2, zorder=5, label='Start')
    ax.annotate('START', start, textcoords="offset points", xytext=(0, 20),
                ha='center', fontsize=10, color='#00FF00', fontweight='bold')
    
    # Goal marker with glow
    glow_goal = Circle(goal, 2.5, color='#FF4444', alpha=0.2, zorder=4)
    ax.add_patch(glow_goal)
    ax.plot(goal[0], goal[1], '*', markersize=25, color='#FF4444',
            markeredgecolor='white', markeredgewidth=2, zorder=5, label='Goal')
    ax.annotate('GOAL', goal, textcoords="offset points", xytext=(0, 20),
                ha='center', fontsize=10, color='#FF4444', fontweight='bold')


def plot_path(ax, path: List[Tuple[float, float]], algorithm: str = 'default',
              linewidth: float = 3.0, label: str = None, alpha: float = 0.95):
    """Plot path with gradient coloring and glow effect."""
    if len(path) < 2:
        return
    
    # Get colors for algorithm
    colors = ALGORITHM_COLORS.get(algorithm, 
                                   {'primary': '#00D4FF', 'secondary': '#0088AA'})
    primary = colors['primary']
    secondary = colors['secondary']
    
    path_arr = np.array(path)
    
    # Create gradient along path
    points = path_arr.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Color gradient based on position along path
    norm = plt.Normalize(0, len(segments))
    cmap = create_gradient_colormap(secondary, primary)
    
    # Glow effect (wider, transparent line behind)
    glow = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.3, linewidths=linewidth*3)
    glow.set_array(np.arange(len(segments)))
    ax.add_collection(glow)
    
    # Main path
    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha, linewidths=linewidth)
    lc.set_array(np.arange(len(segments)))
    ax.add_collection(lc)
    
    # Add arrow markers along path
    step = max(1, len(path) // 8)
    for i in range(0, len(path)-1, step):
        if i + 1 < len(path):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            if np.hypot(dx, dy) > 0.1:
                ax.annotate('', xy=(path[i+1][0], path[i+1][1]), 
                           xytext=(path[i][0], path[i][1]),
                           arrowprops=dict(arrowstyle='->', color=primary, lw=1.5, alpha=0.6))
    
    # Add to legend
    if label:
        ax.plot([], [], color=primary, linewidth=linewidth, label=label, alpha=alpha)


def plot_robot(ax, robot, algorithm: str = 'default'):
    """Plot robot with enhanced styling."""
    colors = ALGORITHM_COLORS.get(algorithm, {'primary': '#00D4FF'})
    color = colors['primary']
    
    # Robot body with glow
    glow = Circle((robot.x, robot.y), robot.radius * 1.5, 
                  color=color, alpha=0.2, zorder=5)
    ax.add_patch(glow)
    
    body = Circle((robot.x, robot.y), robot.radius, 
                  color=color, alpha=0.9, zorder=6,
                  edgecolor='white', linewidth=2)
    ax.add_patch(body)
    
    # Direction indicator
    arrow_len = robot.radius * 2
    dx = arrow_len * np.cos(robot.theta)
    dy = arrow_len * np.sin(robot.theta)
    ax.arrow(robot.x, robot.y, dx, dy, head_width=0.3, head_length=0.15,
             fc='white', ec='white', zorder=7, alpha=0.9)


def plot_trajectory(ax, trajectory: List[Tuple[float, float]], 
                   algorithm: str = 'default', linewidth: float = 2.0):
    """Plot robot trajectory with fade effect."""
    if len(trajectory) < 2:
        return
    
    colors = ALGORITHM_COLORS.get(algorithm, {'primary': '#00D4FF'})
    color = colors['primary']
    
    traj_arr = np.array(trajectory)
    
    # Create fading effect (older points more transparent)
    n = len(traj_arr)
    for i in range(n - 1):
        alpha = 0.1 + 0.6 * (i / n)  # Fade from old to new
        ax.plot([traj_arr[i, 0], traj_arr[i+1, 0]], 
               [traj_arr[i, 1], traj_arr[i+1, 1]],
               color=color, alpha=alpha, linewidth=linewidth, zorder=3)


def plot_tree(ax, nodes: dict, parent: dict, algorithm: str = 'RRT', alpha: float = 0.4):
    """Plot RRT tree with enhanced styling."""
    colors = ALGORITHM_COLORS.get(algorithm, {'primary': '#FF6B6B', 'secondary': '#CC4444'})
    
    lines = []
    for node_id, node_pos in nodes.items():
        if node_id in parent and parent[node_id] is not None:
            parent_id = parent[node_id]
            if parent_id in nodes:
                parent_pos = nodes[parent_id]
                lines.append([parent_pos, node_pos])
    
    if lines:
        lc = LineCollection(lines, colors=colors['secondary'], 
                           alpha=alpha, linewidths=0.8, zorder=2)
        ax.add_collection(lc)
        
        # Plot nodes as small dots
        node_positions = np.array(list(nodes.values()))
        ax.scatter(node_positions[:, 0], node_positions[:, 1], 
                  c=colors['primary'], s=3, alpha=0.3, zorder=2)


def plot_explored_nodes(ax, explored: List[Tuple[float, float]], algorithm: str = 'A*'):
    """Plot explored nodes with heat map effect."""
    if not explored or len(explored) < 2:
        return
    
    colors = ALGORITHM_COLORS.get(algorithm, {'primary': '#00D4FF', 'secondary': '#0088AA'})
    
    exp_arr = np.array(explored)
    
    # Create density-based coloring
    ax.scatter(exp_arr[:, 0], exp_arr[:, 1], 
              c=range(len(exp_arr)), cmap=create_gradient_colormap(colors['secondary'], colors['primary']),
              s=8, alpha=0.4, zorder=2)


def plot_pso_particles(ax, particles: np.ndarray, iteration: int, best_path: List = None):
    """Plot PSO particles for animation."""
    colors = ALGORITHM_COLORS['PSO']
    
    # Plot particles
    for i, particle in enumerate(particles):
        control_pts = particle.reshape(-1, 2)
        ax.scatter(control_pts[:, 0], control_pts[:, 1], 
                  c=colors['primary'], s=20, alpha=0.5, marker='o')
    
    # Plot best path if available
    if best_path:
        plot_path(ax, best_path, algorithm='PSO', label=f'Best (iter {iteration})')


def compare_algorithms(env, paths: dict, start: Tuple[float, float], 
                       goal: Tuple[float, float], times: dict = None,
                       save_path: str = None):
    """Create enhanced comparison visualization."""
    n_algorithms = len(paths) + 1  # +1 for combined view
    
    fig = plt.figure(figsize=(20, 12), facecolor='#1A202C')
    
    # Create grid for subplots
    if n_algorithms <= 3:
        rows, cols = 1, n_algorithms
    elif n_algorithms <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 2, 4
    
    algorithm_names = list(paths.keys())
    
    # Individual algorithm plots
    for idx, (name, path) in enumerate(paths.items()):
        ax = fig.add_subplot(rows, cols, idx + 1)
        plot_environment(ax, env, f"{name}", style='dark')
        
        if path:
            length = sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) 
                        for i in range(len(path)-1))
            time_str = f", {times[name]:.3f}s" if times and name in times else ""
            plot_path(ax, path, algorithm=name, label=f'{length:.1f}m{time_str}')
        
        plot_start_goal(ax, start, goal)
        ax.legend(loc='upper right', fontsize=9, facecolor='#2D3748', 
                 edgecolor='#4A5568', labelcolor='white')
    
    # Combined comparison plot
    ax = fig.add_subplot(rows, cols, len(paths) + 1)
    plot_environment(ax, env, "All Algorithms", style='dark')
    
    for name, path in paths.items():
        if path:
            length = sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) 
                        for i in range(len(path)-1))
            plot_path(ax, path, algorithm=name, linewidth=2.5, 
                     label=f'{name}: {length:.1f}m', alpha=0.85)
    
    plot_start_goal(ax, start, goal)
    ax.legend(loc='upper right', fontsize=9, facecolor='#2D3748',
             edgecolor='#4A5568', labelcolor='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, facecolor='#1A202C', 
                   edgecolor='none', bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()


def animate_path_following(env, robot, path: List[Tuple[float, float]], 
                           algorithm_name: str = "Path Planning",
                           speed: float = 1.0):
    """Animate robot following path with enhanced visuals."""
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#1A202C')
    plt.ion()
    
    robot.set_pose(path[0][0], path[0][1], 0)
    
    waypoint_idx = 1
    goal_threshold = 0.5
    dt = 0.1
    
    try:
        while waypoint_idx < len(path):
            ax.clear()
            ax.set_facecolor('#1A202C')
            
            plot_environment(ax, env, f"{algorithm_name} - Navigation", style='dark')
            plot_path(ax, path, algorithm=algorithm_name, label='Planned Path')
            plot_start_goal(ax, path[0], path[-1])
            
            # Current target
            target = path[waypoint_idx]
            ax.plot(target[0], target[1], 'o', markersize=12, 
                   color='#FFD700', alpha=0.8, markeredgecolor='white')
            
            robot.move_to_point(target[0], target[1], dt=dt)
            
            if robot.distance_to(target[0], target[1]) < goal_threshold:
                waypoint_idx += 1
            
            plot_trajectory(ax, robot.trajectory, algorithm=algorithm_name)
            plot_robot(ax, robot, algorithm=algorithm_name)
            
            ax.legend(loc='upper right', fontsize=10, facecolor='#2D3748',
                     edgecolor='#4A5568', labelcolor='white')
            
            plt.draw()
            plt.pause(0.03 / speed)
        
        ax.set_title(f"{algorithm_name} - Goal Reached! ✓", fontsize=14, 
                    color='#00FF88', fontweight='bold')
        plt.draw()
        plt.pause(1.5)
        
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    from environment import create_demo_environment
    from robot import DifferentialDriveRobot
    
    env = create_demo_environment()
    robot = DifferentialDriveRobot(x=5, y=5, theta=0)
    
    # Demo path
    test_path = [(5, 5), (15, 8), (20, 25), (35, 30), (45, 45)]
    
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#1A202C')
    plot_environment(ax, env, "Enhanced Visualization Demo", style='dark')
    plot_start_goal(ax, test_path[0], test_path[-1])
    plot_path(ax, test_path, algorithm='A*', label='Demo Path')
    plot_robot(ax, robot, algorithm='A*')
    ax.legend(loc='upper right', fontsize=10, facecolor='#2D3748',
             edgecolor='#4A5568', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig('visualization_demo.png', dpi=150, facecolor='#1A202C')
    plt.show()
