"""Self-balancing Segway: LQR/MPC/SMC/pole-placement/RL control, dynamics, ROA, RL training."""
import os


def model_path(name="segway.xml"):
    """Absolute path to a bundled MuJoCo model / asset under balancing/models/."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", name)
