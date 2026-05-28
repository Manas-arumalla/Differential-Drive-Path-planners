# Simulation

`navstack.sim` is a MuJoCo-based physics simulator. `model_generator` builds an
MJCF model procedurally for the chosen drive type; `mujoco_sim` runs the physics
loop and dispatches per-drive-type control.

```python
from navstack.sim import model_generator as mg
import mujoco

xml = mg.generate_mjcf("ackermann", filename="scene.xml")
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
```

## Drive types

| Type | Description |
|------|-------------|
| `differential_2wd` | Two driven wheels + caster |
| `differential_4wd` | Skid-steer, four wheels |
| `mecanum` | Holonomic via mecanum wheels |
| `omni_3wheel` | Three omni wheels at 120° |
| `ackermann` | Car-like, front steering |
| `bicycle` | Single-track, front steering |

Adding or changing a drive type means touching **both** the generator function
in `model_generator.py` and the control branch in `mujoco_sim.py`.

Launch the interactive control center:

```bash
python -m navstack.gui.control_center   # needs the [sim] extra
```
