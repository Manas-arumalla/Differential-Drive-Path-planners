import mujoco as mj
import numpy as np

class Lidar:
    def __init__(self, model, max_range=10.0, n_rays=360):
        self.model = model
        self.max_range = max_range
        self.n_rays = n_rays
        
        # Precompute ray vectors (Body Frame)
        angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
        self.vectors = np.zeros((n_rays, 3))
        self.vectors[:, 0] = np.cos(angles)
        self.vectors[:, 1] = np.sin(angles)
        # Z is 0 (Planar scan)
        
        # Buffers for mj_ray
        self.geom_id = np.zeros(1, dtype=np.int32) # Out
        self.dist = np.float64(0.0)
        self.pnt = np.zeros(3, dtype=np.float64)
        self.vec = np.zeros(3, dtype=np.float64)
        self.root_filter = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "chassis")

    def scan(self, data):
        """
        Perform 360 raycast.
        Returns: list of (x, y) points in WORLD frame that were hit.
        """
        hits = []
        robot_pos = data.qpos[:3] # x,y,z (z might be offset)
        robot_quat = data.qpos[3:7]
        
        # Convert quat to rotation matrix for transforming ray vectors
        # Just use built-in MuJoCo math?
        # Or simple 2D yaw rotation since we know physics.
        # Let's rely on data.site_xmat if we had a site, or just qpos quaternion.
        
        # Helper: Rotate vector by quat? 
        # Actually easier: Ray start is body position.
        # Ray dir is Body Frame -> World Frame.
        
        # Get Body Rotation Matrix from Data
        # body_id of chassis?
        # data.xmat[body_id]
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "chassis")
        mat = data.xmat[bid].reshape(3,3)
        pos = data.xpos[bid]
        
        start_p = pos.copy()
        
        for i in range(self.n_rays):
            # Transform local vector to world
            # vec_world = R * vec_local
            v_local = self.vectors[i]
            v_world = mat @ v_local
            
            # Cast Ray
            # mj_ray(m, d, pnt, vec, geomgroup, flg_static, bodyexclude, out_geomID)
            # geomgroup=None (all), flg_static=1 (include static), bodyexclude=chassis
            
            dist = mj.mj_ray(
                self.model, data,
                start_p, v_world,
                None, 1, self.root_filter,
                self.geom_id
            )
            
            if dist > 0 and dist < self.max_range:
                # Hit point
                hit_p = start_p + v_world * dist
                hits.append(hit_p[:2]) # Keep X,Y
                
        return np.array(hits)

    def get_min_dist(self, data, angle_range=np.pi/3):
        """
        Returns minimum distance in the front sector (+/- angle_range/2).
        """
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "chassis")
        mat = data.xmat[bid].reshape(3,3)
        start_p = data.xpos[bid]
        
        min_d = self.max_range
        
        # Check rays within angle range (Front is 0)
        # Vector 0 is [1, 0] = Front in Body Frame
        
        for i in range(self.n_rays):
            # Calculate angle of this ray
            # We generated rays from 0 to 2pi. 0 is Front.
            angle = (i / self.n_rays) * 2 * np.pi
            # Normalize to -pi, pi
            if angle > np.pi: angle -= 2 * np.pi
            
            if abs(angle) < angle_range / 2.0:
                v_world = mat @ self.vectors[i]
                dist = mj.mj_ray(
                    self.model, data,
                    start_p, v_world,
                    None, 1, self.root_filter,
                    self.geom_id
                )
                if dist > 0 and dist < min_d:
                    min_d = dist
                    
        return min_d

    def scan_sectors(self, data):
        """
        Returns (min_front, min_left, min_right) distances.
        Front: +/- 30 deg
        Left: +30 to +90 deg
        Right: -30 to -90 deg
        """
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "chassis")
        mat = data.xmat[bid].reshape(3,3)
        start_p = data.xpos[bid]
        
        d_f, d_l, d_r = self.max_range, self.max_range, self.max_range
        
        for i in range(self.n_rays):
            angle = (i / self.n_rays) * 2 * np.pi
            if angle > np.pi: angle -= 2 * np.pi
            
            # Cast
            v_world = mat @ self.vectors[i]
            dist = mj.mj_ray(self.model, data, start_p, v_world, None, 1, self.root_filter, self.geom_id)
            if dist <= 0: dist = self.max_range
            
            # Classify
            if abs(angle) < np.radians(30): # Front
                d_f = min(d_f, dist)
            elif angle > np.radians(30) and angle < np.radians(100): # Left
                d_l = min(d_l, dist)
            elif angle < np.radians(-30) and angle > np.radians(-100): # Right
                d_r = min(d_r, dist)
                
        return d_f, d_l, d_r
