"""Cinematic MuJoCo render: a differential-drive robot follows an A* path.

Plans a path around an obstacle with A*, then drives a *physics-simulated* robot
along it (closed-loop turn-then-go control) while an orbiting camera renders
frames offscreen. Saves media/mujoco_drive.gif.

    python examples/mujoco_drive_gif.py
"""
import os

import imageio.v2 as imageio
import mujoco
import numpy as np

from navstack.environment import Environment
from navstack.planners.astar_planner import AStarPlanner
from navstack.sim import model_generator as mg
from navstack.sim.headless_drive import drive_path

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")


def main(fps=20):
    # 1. plan with A* (6x6 m world, one circular obstacle); shift so start = origin
    env = Environment(width=6.0, height=6.0, resolution=0.15)
    env.add_circular_obstacle(3.0, 3.0, 0.8)
    raw = AStarPlanner(env, robot_radius=0.25).plan((1.0, 1.0), (5.0, 5.0))
    if not raw:
        print("A* failed"); return
    path = [(p[0] - 1.0, p[1] - 1.0) for p in raw]

    # 2. matching MuJoCo scene (obstacle at (3,3)-shift = (2,2))
    os.makedirs(OUT, exist_ok=True)
    obstacles = [{"type": "cylinder", "pos": [2.0, 2.0], "size": [0.8, 0.5], "color": "red"}]
    xml = mg.generate_mjcf("differential_2wd", obstacles=obstacles, bounds=[-2, 6, -2, 6],
                           filename=os.path.join(OUT, "_drive_scene.xml"))
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=360, width=480)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 5.0, -35.0
    frames = []
    render_every = max(1, int(0.11 / model.opt.timestep))  # ~120 frames over the run

    def on_step(m, d, step):
        if step % render_every == 0:
            cam.lookat[:] = [float(d.qpos[0]), float(d.qpos[1]), 0.1]
            cam.azimuth = 110 + 40 * min(1.0, step / 3000)
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())

    res = drive_path(model, data, path, on_step=on_step, max_steps=4000)
    print(f"reached={res['reached']} steps={res['steps']} frames={len(frames)}")

    out = os.path.join(OUT, "mujoco_drive.gif")
    imageio.mimsave(out, frames, fps=fps)
    try:
        os.remove(os.path.join(OUT, "_drive_scene.xml"))
    except OSError:
        pass
    print(f"Saved {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
