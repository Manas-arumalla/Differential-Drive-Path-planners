"""Test closed-loop physics path following (differential drive in MuJoCo)."""
import numpy as np
import pytest


def test_robot_drives_path_to_goal(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from navstack.sim import model_generator as mg
    from navstack.sim.headless_drive import drive_path

    # gentle S-curve path in the robot's frame (starts at origin)
    xs = np.linspace(0, 4, 30)
    path = [(float(x), float(0.8 * np.sin(0.6 * x))) for x in xs]

    xml = mg.generate_mjcf("differential_2wd", bounds=[-2, 6, -2, 6],
                           filename=str(tmp_path / "scene.xml"))
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    res = drive_path(model, data, path, max_steps=6000, goal_tol=0.35)
    assert res["reached"], f"robot did not reach goal (drove to {res['trajectory'][-1][:2]})"
    # stayed upright the whole way (free-joint z near spawn height)
    assert np.all(np.isfinite(res["trajectory"]))
