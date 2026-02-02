# demo_all.py
"""
Complete demonstration of all 7 path planning algorithms.
Runs each algorithm on the same environment and compares results.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import create_demo_environment, Environment
from robot import DifferentialDriveRobot
from visualize import (plot_environment, plot_start_goal, plot_path, 
                        animate_path_following, compare_algorithms, ALGORITHM_COLORS)

from planners.astar_planner import AStarPlanner
from planners.rrt_planner import RRTPlanner
from planners.rrt_star_planner import RRTStarPlanner
from planners.pso_planner import PSOPlanner
from planners.apf_planner import APFPlanner
from planners.dijkstra_planner import DijkstraPlanner
from planners.prm_planner import PRMPlanner


def calculate_path_length(path):
    """Calculate total path length."""
    if not path or len(path) < 2:
        return 0
    return sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) 
               for i in range(len(path)-1))


def run_comparison(env, start, goal, robot_radius=0.5):
    """Run all planners and compare results."""
    results = {}
    
    print("\n" + "="*60)
    print("MOBILE ROBOT PATH PLANNING - 7 ALGORITHM COMPARISON")
    print("="*60)
    print(f"Start: {start}")
    print(f"Goal:  {goal}")
    print(f"Environment: {env.width}m x {env.height}m")
    print("="*60 + "\n")
    
    planners = [
        ("A*", AStarPlanner(env, robot_radius=robot_radius)),
        ("RRT", RRTPlanner(env, robot_radius=robot_radius, step_size=2.0, max_iterations=3000)),
        ("RRT*", RRTStarPlanner(env, robot_radius=robot_radius, step_size=2.0, max_iterations=2000)),
        ("PSO", PSOPlanner(env, robot_radius=robot_radius, num_particles=50, max_iterations=80)),
        ("APF", APFPlanner(env, robot_radius=robot_radius, attractive_gain=5.0, repulsive_gain=150.0)),
        ("Dijkstra", DijkstraPlanner(env, robot_radius=robot_radius)),
        ("PRM", PRMPlanner(env, robot_radius=robot_radius, num_samples=250, k_neighbors=8)),
    ]
    
    for i, (name, planner) in enumerate(planners, 1):
        print(f"{i}. {name.upper()} PLANNER")
        print("-" * 40)
        
        t0 = time.time()
        path = planner.plan(start, goal)
        t1 = time.time()
        
        if path:
            # Simplify APF path
            if name == "APF" and hasattr(planner, 'simplify_path'):
                path = planner.simplify_path(path)
            
            results[name] = {
                'path': path,
                'time': t1 - t0,
                'length': calculate_path_length(path),
                'points': len(path)
            }
            print(f"   Time: {results[name]['time']:.4f}s")
            print(f"   Path length: {results[name]['length']:.2f}m")
            print(f"   Waypoints: {results[name]['points']}")
        else:
            print("   FAILED to find path")
        print()
    
    return results


def print_summary(results):
    """Print summary table."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Algorithm':<12} {'Time (s)':<12} {'Length (m)':<12} {'Waypoints':<10}")
    print("-"*60)
    
    for name in ['A*', 'RRT', 'RRT*', 'PSO', 'APF', 'Dijkstra', 'PRM']:
        if name in results:
            r = results[name]
            print(f"{name:<12} {r['time']:<12.4f} {r['length']:<12.2f} {r['points']:<10}")
        else:
            print(f"{name:<12} {'FAILED':<12} {'-':<12} {'-':<10}")
    
    print("="*60)
    
    if results:
        best_length = min(results.items(), key=lambda x: x[1]['length'])
        best_time = min(results.items(), key=lambda x: x[1]['time'])
        print(f"\nShortest path: {best_length[0]} ({best_length[1]['length']:.2f}m)")
        print(f"Fastest planning: {best_time[0]} ({best_time[1]['time']:.4f}s)")


def visualize_comparison(env, results, start, goal, save_path=None):
    """Create comparison visualization."""
    paths = {name: data['path'] for name, data in results.items()}
    times = {name: data['time'] for name, data in results.items()}
    compare_algorithms(env, paths, start, goal, times=times, save_path=save_path)


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print("   MOBILE ROBOT PATH PLANNING - 7 ALGORITHM DEMONSTRATION")
    print("="*70)
    print("\nAlgorithms: A*, Dijkstra, RRT, RRT*, PRM, PSO, APF")
    print()
    
    env = create_demo_environment()
    start = (3, 3)
    goal = (45, 45)
    
    results = run_comparison(env, start, goal)
    print_summary(results)
    visualize_comparison(env, results, start, goal, save_path='comparison.png')
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
