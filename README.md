# Mobile Robot Path Planning

A comprehensive implementation of **7 path planning algorithms** for differential-drive mobile robots, with enhanced visualization.

## 🤖 Algorithms

| Algorithm | Type | Optimal | Description |
|-----------|------|---------|-------------|
| **A*** | Grid | ✅ | Optimal graph search with heuristic |
| **Dijkstra** | Grid | ✅ | Uniform cost search (no heuristic) |
| **RRT** | Sampling | ❌ | Rapidly-exploring Random Tree |
| **RRT*** | Sampling | ✅ | Optimal RRT with rewiring |
| **PRM** | Sampling | Local | Probabilistic Roadmap |
| **PSO** | Optimization | Local | Particle Swarm Optimization |
| **APF** | Reactive | ❌ | Artificial Potential Fields |

## 📁 Project Structure

```
Mobile robotics path planning/
├── planners/
│   ├── __init__.py
│   ├── astar_planner.py      # A* grid search
│   ├── dijkstra_planner.py   # Dijkstra's algorithm
│   ├── rrt_planner.py        # Basic RRT
│   ├── rrt_star_planner.py   # Optimal RRT*
│   ├── prm_planner.py        # Probabilistic Roadmap
│   ├── pso_planner.py        # Particle Swarm Optimization
│   └── apf_planner.py        # Artificial Potential Fields
├── environment.py            # Map and obstacle management
├── robot.py                  # Differential drive robot model
├── visualize.py              # Enhanced plotting utilities
├── demo_all.py               # Full comparison demo
└── README.md
```

## 🚀 Quick Start

```bash
# Install requirements
pip install numpy matplotlib scipy

# Run full comparison (all 7 algorithms)
python demo_all.py

# Run individual algorithms
python planners/astar_planner.py
python planners/dijkstra_planner.py
python planners/rrt_planner.py
python planners/rrt_star_planner.py
python planners/prm_planner.py
python planners/pso_planner.py
python planners/apf_planner.py
```

## 📊 Algorithm Details

### Grid-Based Planners
- **A***: Uses Euclidean heuristic for faster optimal search
- **Dijkstra**: Explores uniformly without heuristic bias

### Sampling-Based Planners
- **RRT**: Fast exploration with random sampling
- **RRT***: Asymptotically optimal with tree rewiring
- **PRM**: Builds reusable roadmap for multiple queries

### Optimization & Reactive
- **PSO**: Smooth spline paths via particle optimization
- **APF**: Real-time reactive with potential fields

## 🎯 Features

- ✅ 7 different planning algorithms
- ✅ Enhanced dark-theme visualization
- ✅ Algorithm-specific color schemes
- ✅ Path gradient effects and glow
- ✅ Robot simulation with trajectory tracking
- ✅ Performance metrics (time, path length)
- ✅ Modular, easy-to-extend design

## 📝 License

MIT License - Educational use welcome.
