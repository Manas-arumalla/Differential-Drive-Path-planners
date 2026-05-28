import numpy as np
import control as ctrl
import cvxpy as cp
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
from navstack.balancing import dynamics as segway_dynamics

class BaseController:
    def __init__(self, params):
        self.params = params
        self.A, self.B = segway_dynamics.get_linearized_dynamics(
            m_base=params.get('m_base', 1.0),
            mw=params.get('mw', 0.432),
            mp=params.get('mp', 5.0),
            l=params.get('l', 0.4),
            r=params.get('r', 0.1),
            Ip=params.get('Ip', 0.2),
            damping_x=params.get('damping_x', 0.05),
            damping_theta=params.get('damping_theta', 0.2)
        )

    def reset(self):
        pass

class LQRController(BaseController):
    def __init__(self, params, Q_diag, R_val):
        super().__init__(params)
        Q = np.diag(Q_diag)
        R = np.array([[R_val]])
        try:
            self.K, _, _ = ctrl.lqr(self.A, self.B, Q, R)
            self.K = np.asarray(self.K)
            print(f"[LQR] K = {self.K}")
        except Exception as e:
            print(f"[LQR] Failed: {e}")
            self.K = np.zeros((1, 4))
            
    def update(self, state, t=0, target_pos=0.0):
        # Saturation: Limit perceived distance to 1.0m
        pos_err = np.clip(state[0] - target_pos, -1.0, 1.0)
        error = np.array([pos_err, state[1], state[2], state[3]])
        u = -self.K @ error
        return u.item()
    
    def reset(self):
        pass

class PolePlacementController(BaseController):
    def __init__(self, params, poles):
        super().__init__(params)
        try:
            print(f"[PP] Placing poles at: {poles}")
            self.K = ctrl.place(self.A, self.B, poles)
            self.K = np.asarray(self.K)
            print(f"[PP] K = {self.K}")
        except Exception as e:
            print(f"[PP] Failed: {e}")
            self.K = np.zeros((1, 4))
            
    def update(self, state, t=0, target_pos=0.0):
        pos_err = np.clip(state[0] - target_pos, -1.0, 1.0)
        error = np.array([pos_err, state[1], state[2], state[3]])
        u = -self.K @ error
        return u.item()
    
    def reset(self):
        pass

class SlidingModeController(BaseController):
    def __init__(self, params, lambda_vec, K_smc, phi):
        super().__init__(params)
        self.coeffs = np.array(lambda_vec) 
        self.K_smc = K_smc
        self.phi = phi
        
        # Pre-compute Equivalent Control Terms
        self.CB_inv = 1.0 / (self.coeffs @ self.B)
        self.CA = self.coeffs @ self.A
        
    def sat(self, s):
        return np.clip(s / self.phi, -1.0, 1.0)

    def update(self, state, t=0, target_pos=0.0):
        pos_err = np.clip(state[0] - target_pos, -1.0, 1.0)
        error = np.array([pos_err, state[1], state[2], state[3]])
        
        # Sliding Surface
        s = self.coeffs @ error
        
        # Control Law: u = u_eq + u_sw
        term = self.CA @ error + self.K_smc * self.sat(s)
        u = -self.CB_inv * term
        return u.item()
    
    def reset(self):
        pass

class MPCController(BaseController):
    def __init__(self, params, Q_diag, R_val, horizon=10, r_rate=0.5, 
                 control_horizon=None, dt=0.05, u_max=2000.0):
        super().__init__(params)
        self.N = int(horizon) 
        self.dt = float(dt)
        self.u_max = float(u_max)
        
        # Control Horizon
        if control_horizon and int(control_horizon) > 0:
            self.Nc = min(int(control_horizon), self.N)
        else:
            self.Nc = self.N
            
        # Discretize
        sysd = cont2discrete((self.A, self.B, np.eye(4), np.zeros((4,1))), self.dt)
        self.Ad = sysd[0]
        self.Bd = sysd[1]
        
        # Weights
        self.Q_state_cost = np.diag(Q_diag)
        self.R_control_cost = np.array([[R_val]])
        self.R_rate_cost = r_rate
        
        # Terminal Cost
        try:
            self.P_terminal_cost = solve_discrete_are(self.Ad, self.Bd, self.Q_state_cost, self.R_control_cost)
        except:
            print("[MPC] Warning: DARE failed, using Q terminal.")
            self.P_terminal_cost = self.Q_state_cost

        # CVXPY Setup
        self.initial_state_param = cp.Parameter(4)
        self.previous_u_param = cp.Parameter(1)
        
        self.control_trajectory = cp.Variable((1, self.N))
        self.state_trajectory = cp.Variable((4, self.N + 1))
        
        cost = 0
        constraints = [self.state_trajectory[:, 0] == self.initial_state_param]
        
        for k in range(self.N):
            # Stage Cost
            cost += cp.quad_form(self.state_trajectory[:, k], self.Q_state_cost) + cp.quad_form(self.control_trajectory[:, k], self.R_control_cost)
            
            # Rate Cost
            if k == 0:
                cost += self.R_rate_cost * cp.sum_squares(self.control_trajectory[:, k] - self.previous_u_param)
            else:
                cost += self.R_rate_cost * cp.sum_squares(self.control_trajectory[:, k] - self.control_trajectory[:, k-1])
            
            # Dynamics
            constraints += [self.state_trajectory[:, k+1] == self.Ad @ self.state_trajectory[:, k] + self.Bd @ self.control_trajectory[:, k]]
            
            # Input Constraints
            constraints += [self.control_trajectory[:, k] <= self.u_max, self.control_trajectory[:, k] >= -self.u_max]
            
        cost += cp.quad_form(self.state_trajectory[:, self.N], self.P_terminal_cost)
        
        self.prob = cp.Problem(cp.Minimize(cost), constraints)
        self.last_update_time = -1
        self.last_u = 0

    def update(self, state, t=0, target_pos=0.0):
        if t - self.last_update_time < self.dt and self.last_update_time >= 0:
             return self.last_u

        try:
            # Saturation
            pos_error = np.clip(state[0] - target_pos, -1.0, 1.0)
            
            # Params
            self.initial_state_param.value = np.array([pos_error, state[1], state[2], state[3]])
            self.previous_u_param.value = [self.last_u]
            
            # Solve (Warm Start -> Cold Start -> ECOS)
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            
            if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                self.prob.solve(solver=cp.OSQP, warm_start=False)
                
            if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                print(f"[MPC] OSQP failed ({self.prob.status}), switching to ECOS...")
                self.prob.solve(solver=cp.ECOS)
            
            if self.control_trajectory.value is not None:
                self.last_u = self.control_trajectory[:, 0].value.item()
            else:
                print(f"[MPC CRITICAL] Solver failed. Status: {self.prob.status}")
                self.last_u = 0.0 
                
        except Exception as e:
             print(f"[MPC Exception] {e}")
             pass
             
        self.last_update_time = t
        return self.last_u
    
    def reset(self):
        self.last_update_time = -1
        self.last_u = 0

class RLController(BaseController):
    def __init__(self, params, model_path=None):
        super().__init__(params)
        self.model = None
        import os
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ppo_segway.zip")
        try:
            from stable_baselines3 import PPO
            if os.path.exists(model_path):
                self.model = PPO.load(model_path)
                print(f"[RL] Loaded PPO model from {model_path}")
            else:
                print(f"[RL] Model not found at {model_path}")
        except ImportError:
            print("[RL] stable_baselines3 not installed. RL Control unavailable.")
            
    def update(self, state, t=0, target_pos=0.0):
        if self.model is None:
            return 0.0
            
        # RL Obs: [x, vx, theta, w]
        # state from LQR loop: [x, vx, theta, w]
        # But gym was trained on Absolute X. 
        # For tracking target, we invert the logic: 
        # If target is +5.0 (Ahead), we want to mock 'Error' such that Agent moves Forward.
        # User reported previous logic moved Backward. Flipping sign.
        
        pos_err = target_pos - state[0]
        
        # Clip observation to training bounds roughly to avoid OOD (Out of Distribution)
        # Gym limit: x +/- 5.0
        pos_err = np.clip(pos_err, -5.0, 5.0)
        
        obs = np.array([pos_err, state[1], state[2], state[3]], dtype=np.float32)
        
        # PPO Predict
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Un-Normalize Action: [-1, 1] -> [-50, 50]
        torque = float(action[0]) * 50.0
        
        return torque

    def reset(self):
        pass
