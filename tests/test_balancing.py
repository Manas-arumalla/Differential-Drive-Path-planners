"""Integration tests for the self-balancing Segway.

Skip automatically where the heavy control/RL deps are absent so CI on a
minimal install stays green.
"""
import numpy as np
import pytest

PHYS = {"m_base": 1.0, "mw": 0.432, "mp": 5.0, "l": 0.4, "r": 0.1, "Ip": 0.2,
        "damping_x": 0.05, "damping_theta": 0.2}
STATE = np.array([0.0, 0.0, 0.1, 0.0])  # small forward tilt


def _controllers():
    pytest.importorskip("control")
    pytest.importorskip("cvxpy")
    from navstack.balancing import controllers as C
    return C


def test_classical_controllers_produce_finite_torque():
    C = _controllers()
    ctrls = [
        C.LQRController(PHYS, [980, 1, 1, 1], 0.01),
        C.PolePlacementController(PHYS, [-60, -5, -2, -1.5]),
        C.SlidingModeController(PHYS, [2.0, 1.0, 5.0, 0.8], 20.0, 0.5),
    ]
    for c in ctrls:
        assert np.isfinite(c.update(STATE, 0.0, 0.0))


def test_mpc_controller_produces_finite_torque():
    C = _controllers()
    mpc = C.MPCController(PHYS, [100, 1, 1000, 1], 0.01, 20, 0.5,
                          control_horizon=None, dt=0.05, u_max=200.0)
    assert np.isfinite(mpc.update(STATE, 0.0, 0.0))


def test_segway_model_loads_and_steps():
    mujoco = pytest.importorskip("mujoco")
    from navstack.balancing import model_path
    model = mujoco.MjModel.from_xml_path(model_path())
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))


def test_gym_env_resets_and_steps():
    pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    from navstack.balancing.gym_env import SegwayEnv
    env = SegwayEnv()
    obs, info = env.reset()
    obs2, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (4,)
    assert np.all(np.isfinite(obs2)) and np.isfinite(reward)


def test_rl_controller_runs():
    pytest.importorskip("stable_baselines3")
    C = _controllers()
    rl = C.RLController(PHYS)  # loads the bundled model if present
    assert np.isfinite(rl.update(STATE, 0.0, 0.0))


def test_lidar_constructs_on_segway_model():
    mujoco = pytest.importorskip("mujoco")
    from navstack.balancing import model_path
    from navstack.perception.lidar import Lidar
    model = mujoco.MjModel.from_xml_path(model_path())
    lidar = Lidar(model, max_range=5.0, n_rays=36)
    assert lidar is not None
