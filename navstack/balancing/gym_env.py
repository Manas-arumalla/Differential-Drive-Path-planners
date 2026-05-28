import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco as mj
import os

class SegwayEnv(gym.Env):
    """
    Custom Gym Environment for Segway Balancing.
    Replicates the physics of 'segway_control_lqr.py'.
    """
    
    def __init__(self, xml_path=None):
        super(SegwayEnv, self).__init__()
        if xml_path is None:
            from navstack.balancing import model_path
            xml_path = model_path()
        # 1. Load Model
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML {xml_path} not found.")
            
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        
        # 2. Physics Calibration (From segway_control_lqr.py)
        # Assuming constants: m_base=1.0, mw=0.432, mp=5.0, Ip=0.2
        # Modify model directly
        m_base, mw, mp = 1.0, 0.432, 5.0
        l, Ip = 0.4, 0.2
        d_x, d_theta = 0.05, 0.2
        
        # Mass
        M = m_base + 2 * mw
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base")
        if bid != -1: self.model.body_mass[bid] = M
        pid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "pendulum")
        if pid != -1:
            self.model.body_mass[pid] = mp
            self.model.body_inertia[pid][:] = [Ip, Ip, Ip/20.0]
            self.model.body_ipos[pid][2] = l
            
        # Damping
        for jn, d in [("slide_x", d_x), ("slide_y", d_x), ("pend_hinge", d_theta)]:
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jn)
            if jid != -1: self.model.dof_damping[self.model.jnt_dofadr[jid]] = d
            
        # 3. Action Space (Torque: +/- 1.0 Normalized)
        # We will scale this to +/- 50.0 Nm inside step()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 4. Observation Space
        # [x, vx, theta, w] -> 4 values
        # Limits: x (+/- 5m), vx (+/- 10m/s), theta (+/- 1 rad), w (+/- 10 rad/s)
        high = np.array([5.0, 10.0, 1.0, 20.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mj.mj_resetData(self.model, self.data)
        
        # Randomize Start slightly
        self.data.qpos[0] = np.random.uniform(-0.1, 0.1) # x
        
        # Random initial tilt (+/- 5 degrees)
        angle = np.random.uniform(-0.08, 0.08)
        self.data.qpos[3] = np.cos(angle/2)
        self.data.qpos[5] = np.sin(angle/2)
        
        mj.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
    
    def step(self, action):
        # Scale Normalized Action (-1, 1) to Max Torque (50 Nm)
        # 20Nm was too weak for recovery in early training
        act_val = float(action[0])
        torque = act_val * 50.0 
        
        # Apply Torque to both wheels (1D Balance)
        self.data.ctrl[0] = torque / 2.0
        self.data.ctrl[1] = torque / 2.0
        
        # Step Physics
        # Run 5 steps for stable frequency (Assuming dt=0.002 -> 0.01s step)
        for _ in range(5):
            mj.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        x, vx, theta, w = obs
        
        # Reward Function (Bellman-style Balance)
        # 1.0 Alive Bonus
        # Penalize Tilt squared
        # Penalize Velocity (Drift)
        r_survive = 1.0
        r_angle = -10.0 * (theta ** 2)
        r_pos = -0.1 * (x ** 2)
        r_vel = -0.05 * (w ** 2) 
        r_ctrl = -0.01 * (torque ** 2)
        
        reward = float(r_survive + r_angle + r_pos + r_vel + r_ctrl)
        
        # Termination
        terminated = False
        truncated = False
        
        if abs(theta) > 0.8: # Fallen (~45 deg)
            terminated = True
            reward -= 100.0 # Crash Penalty
            
        if abs(x) > 4.5: # Off map
            terminated = True
            reward -= 100.0
            
        return obs, float(reward), terminated, truncated, {}
    
    def _get_obs(self):
        # State Extraction
        x = self.data.qpos[0]
        
        # Quaternion to Pitch (Simplified for 2D)
        # q = [w, x, y, z] -> qpos[3,4,5,6] (MuJoCo order)
        qw = self.data.qpos[3]
        qy = self.data.qpos[5]
        # pitch ~ 2 * atan2(y, w)
        theta = 2 * np.arctan2(qy, qw)
        
        vx = self.data.qvel[0]
        w = self.data.qvel[2] # Hinge velocity
        
        return np.array([x, vx, theta, w], dtype=np.float32)
        
    def close(self):
        pass
