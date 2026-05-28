# Balancing & RL

`navstack.balancing` is a self-balancing Segway stack. Every controller shares
one `(state, t, target) -> torque` interface, so they are swappable.

| Controller | Method |
|------------|--------|
| `LQRController` | Infinite-horizon LQR |
| `PolePlacementController` | Explicit closed-loop poles |
| `SlidingModeController` | Robust sliding-mode control |
| `MPCController` | CVXPY receding-horizon MPC |
| `RLController` | Pre-trained PPO policy |

```python
from navstack.balancing import controllers as C, model_path
import mujoco

phys = dict(m_base=1.0, mw=0.432, mp=5.0, l=0.4, r=0.1, Ip=0.2,
            damping_x=0.05, damping_theta=0.2)
lqr = C.LQRController(phys, [980, 1, 1, 1], 0.01)
torque = lqr.update([0, 0, 0.1, 0], t=0.0, target_pos=0.0)

model = mujoco.MjModel.from_xml_path(model_path())  # bundled segway.xml
```

## Reinforcement learning

`gym_env.SegwayEnv` is a Gymnasium environment; `train.py` trains a PPO agent
and `RLController` loads the result (`balancing/models/ppo_segway.zip`).

```bash
python -m navstack.balancing.train --train --steps 100000
python -m navstack.balancing.train --test
```

## Analysis

- `dynamics.py` / `derive_dynamics.py` — analytical + symbolic linearized dynamics.
- `roa.py` — region-of-attraction stability sweep.
- `optimize.py` — genetic-algorithm gain tuning.
