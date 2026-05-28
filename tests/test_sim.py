"""Integration tests for the MuJoCo multi-drive simulator.

These skip automatically where the heavy deps (mujoco / PySide6) are absent,
so CI on a minimal install stays green while local runs are fully covered.
"""
import os

import numpy as np
import pytest

from navstack.controllers.navigation import get_controller

DRIVE_TYPES = [
    "differential_2wd", "differential_4wd", "mecanum",
    "omni_3wheel", "ackermann", "bicycle",
]


@pytest.mark.parametrize("rtype", DRIVE_TYPES)
def test_drive_type_generates_and_steps(rtype, tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from navstack.sim import model_generator as mg

    xml = mg.generate_mjcf(rtype, filename=str(tmp_path / f"{rtype}.xml"))
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos)), f"{rtype} physics diverged"
    assert model.nu >= 2, f"{rtype} has too few actuators"


@pytest.mark.parametrize("name", ["PurePursuit", "Stanley", "Proportional", "DWA"])
def test_controller_produces_finite_command(name):
    path = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (5, 3)]
    v, w, idx = get_controller(name).compute_control((0.0, 0.0, 0.0), path, 1)
    assert np.isfinite(v) and np.isfinite(w)
    assert isinstance(idx, int)


def test_gui_module_imports():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import importlib
    mod = importlib.import_module("navstack.gui.control_center")
    assert hasattr(mod, "MobileRobotGUI")
