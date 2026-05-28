import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import matplotlib.pyplot as plt
import time
from navstack.balancing import controllers, model_path
from navstack.perception import lidar as sensors
from navstack.perception import vision

def run_segway_simulation(
    # Physics
    m_base=1.0, mw=0.432, mp=5.0, l=0.4, r=0.1, Ip=0.2,
    damping_x=0.05, damping_theta=0.2,
    # Controller
    strat_name="LQR", strat_params=None, 
    # Simulation
    initial_tilt=1.0, initial_pos=0.0, initial_pos_y=0.0,
    target_pos=0.0, target_pos_y=0.0,
    disturbances=[], path=None, duration=30.0,
    sim_speed_factor=1.25, show_plots=True,
    heading_strat="PID", heading_params=None,
    planner_name="Manual", xml_path=None, task_mode="Nav",
    target_color=None, planner=None, search_bounds=None
):
    """Run Segway Simulation with Navigation"""
    if strat_params is None: strat_params = {}
    if xml_path is None: xml_path = model_path()
    
    # 1. Controller Init
    print(f"Initializing {strat_name} Controller...")
    phys_params = {
        'm_base': m_base, 'mw': mw, 'mp': mp, 'l': l, 'r': r, 'Ip': Ip,
        'damping_x': damping_x, 'damping_theta': damping_theta
    }

    if strat_name == "LQR":
        q = strat_params.get("Q_diag", [980, 1, 1, 1])
        r_val = strat_params.get("R_val", 0.01)
        controller = controllers.LQRController(phys_params, q, r_val)
        
    elif strat_name == "PolePlacement":
        poles = strat_params.get("poles", [-60, -5, -2, -1.5]) # LQR-like dynamics
        controller = controllers.PolePlacementController(phys_params, poles)
        
    elif strat_name == "SMC":
        lam = strat_params.get("lambda", [2.0, 1.0, 5.0, 0.8])
        K = strat_params.get("K_smc", 20.0)
        phi = strat_params.get("phi", 0.5) # Increased for smoother control
        controller = controllers.SlidingModeController(phys_params, lam, K, phi)
        
    elif strat_name == "MPC":
        q = strat_params.get("Q_diag", [100, 1, 1000, 1])
        r_val = strat_params.get("R_val", 0.01)
        horizon = strat_params.get("horizon", 20)
        r_rate = strat_params.get("r_delta", 0.5)
        nc = strat_params.get("nc", None)
        dt_mpc = strat_params.get("dt_mpc", 0.05)
        u_max = strat_params.get("u_max", 200.0)
        
        controller = controllers.MPCController(phys_params, q, r_val, horizon, r_rate, 
                                               control_horizon=nc, dt=dt_mpc, u_max=u_max)
    elif strat_name == "RL (PPO)":
        controller = controllers.RLController(phys_params)

    else:
        print(f"Unknown strategy {strat_name}, defaulting to LQR")
        controller = controllers.LQRController(phys_params, [1,1,1,1], 1)

    # 2. Physics & Model
    model = mj.MjModel.from_xml_path(xml_path)
    
    # Update Mass/Inertia
    M = m_base + 2 * mw
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base")
    if base_id != -1: model.body_mass[base_id] = M

    pend_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pendulum")
    if pend_id != -1:
        model.body_mass[pend_id] = mp
        model.body_inertia[pend_id][:] = [Ip, Ip, Ip/20.0]
        model.body_ipos[pend_id][2] = l 
        
    # Update Damping
    for j_name, d_val in [("slide_x", damping_x), ("slide_y", damping_x), ("pend_hinge", damping_theta)]:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, j_name)
        if jid != -1: model.dof_damping[model.jnt_dofadr[jid]] = d_val

    data = mj.MjData(model)
    
    # Initial State
    data.qpos[0] = initial_pos
    data.qpos[1] = initial_pos_y
    data.qpos[2] = 0.1
    
    # Initial Tilt (Quaternion)
    ha = initial_tilt / 2.0
    data.qpos[3] = np.cos(ha)
    data.qpos[5] = np.sin(ha)
    mj.mj_forward(model, data)

    # 3. Visualization
    if not glfw.init(): return
    window = glfw.create_window(1600, 1200, f"Segway - {strat_name}", None, None)
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "chassis") 
    
    # Front Camera
    cam_front = mj.MjvCamera()
    cam_front.type = mj.mjtCamera.mjCAMERA_FIXED
    cam_front.fixedcamid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "front_cam") 
    
    # Sensors
    lidar = sensors.Lidar(model)
    lidar = sensors.Lidar(model)
    lidar_hits = [] 
    
    # Vision System
    detector = vision.ColorDetector()
    vision_steering = 0.0
    vision_active = False
    vision_area = 0.0
    vision_search_dir = 1.0 
    
    # Track Target Loss & Stability
    vision_was_active = False
    lost_target_timer = 0.0
    last_slide_dir = 1.0
    smoothed_head_err = 0.0

                # ... (Path Following)
    rgb_buffer = np.zeros((300, 400, 3), dtype=np.uint8) 
    
    if task_mode == "Balance":
        cam.distance = 4.0
        cam.elevation = -10
        cam.azimuth = 90
    else:
        cam.distance = 12.0
        cam.elevation = -30
        cam.azimuth = 135
    
    opt = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    # buffers
    time_data, x_data, y_data, theta_data, ctrl_data = [], [], [], [], []
    tau_left_data, tau_right_data = [], []
    dt = model.opt.timestep
    
    last_render_time = time.time()
    stability_duration = 0.0
    i = 0
    
    # Path Setup
    if path is None or len(path) == 0:
        full_path = [(initial_pos, initial_pos_y), (target_pos, target_pos_y)]
    else:
        full_path = path

    current_wp_idx = 1 if len(full_path) > 1 else 0
    last_wp_idx = 0 # Track last valid waypoint for smooth path following
    start_wall_time = time.time()
    
    # Helper: Quat to Euler
    def quat2euler(q):
        w, x, y, z = q
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = np.sign(sinp) * np.pi / 2 if abs(sinp) >= 1 else np.arcsin(sinp)
        return yaw, pitch

    # Heading State
    yaw_integral = 0.0
    if heading_params is None: heading_params = {}
    
    k_turn_p = heading_params.get('kp', 30.0)
    k_turn_i = heading_params.get('ki', 0.1)
    k_turn_d = heading_params.get('kd', 5.0)
    
    smc_lambda = heading_params.get('lambda', 3.0)
    smc_k = heading_params.get('k', 80.0)
    smc_phi = heading_params.get('phi', 0.1)

    # Controllers
    def calc_smc_torque(error, rate, lam, K, phi):
        s = rate + lam * error
        return K * np.tanh(s / phi)

    def calc_stanley_control(cte, v_fwd, psi_e, k_gain):
        # Steer = HeadingErr + arctan(k*CTE / (v+1))
        v_abs = abs(v_fwd) + 1.0
        delta = psi_e + np.arctan2(k_gain * cte, v_abs)
        return 50.0 * delta # Torque Scaling

    # --- Main Loop ---
    while True:
        sim_t = i * dt
        if sim_t >= duration: break
        i += 1
        
        # Disturbances
        for d in disturbances:
            if abs(sim_t - d['time']) < dt/2:
                print(f"Kick at t={sim_t:.2f}!")
                data.qvel[1] += d['impulse']
        
        # State
        x_curr, y_curr = data.qpos[0], data.qpos[1]
        yaw_curr, theta_curr = quat2euler(data.qpos[3:7])
        vx_world, vy_world = data.qvel[0], data.qvel[1]
        vx_world, vy_world = data.qvel[0], data.qvel[1]
        wz_curr, w_theta_curr = data.qvel[5], data.qvel[4]
        
        # Sensor Scan (5Hz)
        if i % 20 == 0:
            hits = lidar.scan(data)
            if len(hits) > 0:
                lidar_hits.append(hits)
        
        # Body Frame Velocity
        vx_curr = vx_world * np.cos(yaw_curr) + vy_world * np.sin(yaw_curr)
        vy_curr = -vx_world * np.sin(yaw_curr) + vy_world * np.cos(yaw_curr)
        
        is_final_goal = (len(full_path) > 0 and last_wp_idx >= len(full_path) - 2)
        dist_to_target = 100.0 # Default infinite
        if len(full_path) > 0:
             # Simple distance to final goal
             d_final = np.hypot(x_curr - full_path[-1][0], y_curr - full_path[-1][1])
             dist_to_target = d_final
        
        # Deadband Logic
        base_deadband = 0.0 if task_mode == "Balance" else (0.3 if is_final_goal else 0.1)
        
        # unified Control Logic
        if task_mode == "Hunt" and vision_active:
             vision_was_active = True
             lost_target_timer = 0.0 # Reset
             
             # --- 1. CHASE with POTENTIAL FIELDS ---
             head_err = -vision_steering * 1.0 
             drive_sign = 1.0
             
             # --- 1. CHASE with POTENTIAL FIELDS ---
             d_f, d_l, d_r = lidar.scan_sectors(data)
             
             # Repulsive Forces (Inverse Distance Law)
             # Higher Force if closer. Max range 2.0m for influence.
             k_rep = 2.0
             f_rep_front = k_rep / (d_f**2 + 0.1) if d_f < 2.0 else 0.0
             f_rep_left  = k_rep / (d_l**2 + 0.1) if d_l < 2.0 else 0.0
             f_rep_right = k_rep / (d_r**2 + 0.1) if d_r < 2.0 else 0.0
             
             # Net Repulsion ( Lateral )
             # Left Obstacle pushes Right (-ve), Right Obstacle pushes Left (+ve)
             f_lat_rep = f_rep_right - f_rep_left
             
             # Front Obstacle Ramping (Smooth Force)
             avoid_threshold = 1.5
             intensity = 0.0 # Default
             
             if d_f < avoid_threshold:
                 # Hysteresis for Slide Direction
                 if d_l > d_r + 0.2:
                     last_slide_dir = 1.0 # Left
                 elif d_r > d_l + 0.2:
                     last_slide_dir = -1.0 # Right
                 
                 # Ramp Force: 0.0 at 1.5m -> 1.0 at 0.5m
                 intensity = np.clip((avoid_threshold - d_f) / 1.0, 0.0, 1.0)
                 
                 # Add smooth lateral push
                 f_lat_rep += last_slide_dir * intensity * 3.0
                 
                 # Slow down smoothly
                 drive_sign = 1.0 - (0.8 * intensity) 
                 drive_sign = max(0.2, drive_sign)
             else:
                 drive_sign = 1.0
             
             # Attraction (Vision)
             f_att = -vision_steering 
             
             # Dynamic Weighting: Suppress Attraction if avoiding obstacle
             # This prevents "fighting" (shaking) when target is behind obstacle
             suppression = 1.0 - (0.9 * intensity) 
             w_att = 1.0 * suppression
             w_rep = 1.5 
             
             raw_head_err = w_att * f_att + w_rep * f_lat_rep
             
             # Low Pass Filter (Smoothing)
             smoothed_head_err = 0.7 * smoothed_head_err + 0.3 * raw_head_err
             
             head_err = np.clip(smoothed_head_err, -1.5, 1.5)
             
             if vision_area > 0.20:
                drive_sign = 0.0
                head_err = 0.0
             
             yaw_err = head_err
             target_dist_signed = 1.0 
             cte = 0.0 # No path following in Chase
             
        else:
             # --- 2. NAVIGATE / PATROL ---
             
             # Re-Plan Logic if Target Lost
             if task_mode == "Hunt" and vision_was_active:
                 lost_target_timer += dt
                 if lost_target_timer > 1.0: # Lost for > 1s
                     print("Target Lost! Re-generating Search Pattern from current location...")
                     if planner and search_bounds:
                         try:
                             current_pos = (x_curr, y_curr)
                             # Re-plan from current pos
                             new_path = planner.generate_coverage_path(search_bounds, stride=1.5, start_pos=current_pos)
                             if new_path:
                                 full_path = new_path
                                 last_wp_idx = 0 # Reset progress
                                 print(f"Re-planned {len(full_path)} Waypoints.")
                             else:
                                 print("Re-plan returned empty path!")
                         except Exception as e:
                             print(f"Re-plan failed: {e}")
                     else:
                        print("Planner not available.")

                     # Always reset to prevent infinite loop
                     vision_was_active = False 
                     lost_target_timer = 0.0

             if full_path and len(full_path) > 1:
                # Path Following Logic
                closest_dist = float('inf')
                closest_idx = 0
                for idx in range(last_wp_idx, len(full_path)-1):
                    p1 = np.array(full_path[idx])
                    p2 = np.array(full_path[idx+1])
                    seg = p2 - p1
                    l2 = np.sum(seg**2)
                    if l2 == 0: continue
                    t = max(0, min(1, np.dot(np.array([x_curr, y_curr]) - p1, seg) / l2))
                    proj = p1 + t * seg
                    d = np.hypot(x_curr - proj[0], y_curr - proj[1])
                    if d < closest_dist:
                        closest_dist = d
                        closest_idx = idx
                
                last_wp_idx = closest_idx
                p1 = np.array(full_path[closest_idx])
                p2 = np.array(full_path[closest_idx+1])
                seg = p2 - p1
                seg_len = np.linalg.norm(seg)
                lookahead = 0.5 + 0.5 * abs(vx_curr)
                t_proj = max(0, min(1, np.dot(np.array([x_curr, y_curr]) - p1, seg) / (seg_len**2 + 1e-6)))
                dist_on_seg = t_proj * seg_len
                dist_remain = seg_len - dist_on_seg
                
                
                drive_sign = 1.0 # Default to driving
                if dist_remain > lookahead:
                    tx, ty = p1 + (seg/seg_len) * (dist_on_seg + lookahead)
                elif closest_idx + 1 < len(full_path) - 1:
                    rem_l = lookahead - dist_remain
                    p3 = np.array(full_path[closest_idx+2])
                    seg2 = p3 - p2
                    seg2_len = np.linalg.norm(seg2)
                    tx, ty = p2 + (seg2/seg2_len) * min(rem_l, seg2_len)
                else:
                    # Final Waypoint Logic
                    tx, ty = p2
                    dist_to_end = np.linalg.norm([x_curr - p2[0], y_curr - p2[1]])
                    if dist_to_end < 0.3: # Close enough to goal
                         tx, ty = x_curr + np.cos(yaw_curr), y_curr + np.sin(yaw_curr) # Fake target ahead
                         drive_sign = 0.0 # Stop
                    
                
                if drive_sign > 0.1:
                    target_yaw = np.arctan2(ty - y_curr, tx - x_curr)
                else:
                    target_yaw = yaw_curr # Maintain heading if stopped
                
                yaw_err = target_yaw - yaw_curr
                while yaw_err > np.pi: yaw_err -= 2*np.pi
                while yaw_err < -np.pi: yaw_err += 2*np.pi
                
                # Check for Reversing (If target is behind, don't turn 180, just reverse)
                # Only apply if we are driving 
                if drive_sign > 0.1 and abs(yaw_err) > (np.pi / 2):
                    yaw_err -= np.sign(yaw_err) * np.pi 
                    while yaw_err > np.pi: yaw_err -= 2*np.pi
                    while yaw_err < -np.pi: yaw_err += 2*np.pi
                    drive_sign = -1.0 # Reverse

                
                # CTE for Stanley
                p_robot = np.array([x_curr, y_curr])
                p_start, p_end = p1, p2 # Approximate segment
                line_vec = p_end - p_start
                len_sq = np.sum(line_vec**2)
                cte = 0.0
                if len_sq > 0.01:
                    cte = ((p_robot[0]-p_start[0])*line_vec[1] - (p_robot[1]-p_start[1])*line_vec[0]) / np.sqrt(len_sq)

             else:
                 # No Path (or Finished)
                 cte = 0.0
                 if task_mode == "Hunt":
                     # --- 3. SCAN (Fallback) ---
                     # Spin to find target
                     head_err = vision_search_dir * 1.5
                     yaw_err = head_err
                     drive_sign = 0.0
                 else:
                     # Balance in place
                     yaw_err = 0.0
                     drive_sign = 0.0

        # Apply Heading Control (Unified)
        if heading_strat == "Stanley" and (full_path and not (task_mode=="Hunt" and vision_active)):
             k_stanley = heading_params.get('k', 2.5)
             turn_torque = calc_stanley_control(cte, vx_curr, yaw_err, k_stanley)
             turn_torque -= 5.0 * wz_curr # Damping
        elif heading_strat == "SMC":
             s_head = yaw_err + 0.1 * wz_curr 
             turn_torque = 20.0 * np.tanh(s_head/0.5) 
        else: # PID
             yaw_integral = np.clip(yaw_integral + yaw_err*dt, -5.0, 5.0)
             turn_torque = (k_turn_p * yaw_err) + (k_turn_i * yaw_integral) - (k_turn_d * wz_curr) 

        # Balance Control
        mpc_state = np.array([0.0, vx_curr, theta_curr, w_theta_curr])
        
        # Adaptive Cornering (Pivot priority)
        # Smoothly reduce speed as error increases, rather than hard cut
        align_fac = max(0.0, 1.0 - (abs(yaw_err) / 1.0)) # Linear falloff over 1 radian
        align_fac = align_fac ** 2 # Convex (stays slow for larger errors)
        
        target_dist_signed = dist_to_target * drive_sign * align_fac
        
        # RL HACK: Boost target distance to encourage speed
        if strat_name == "RL (PPO)":
             # If we want to move, fake the target to be 2.0m away to force acceleration
             if abs(target_dist_signed) > 0.1:
                 target_dist_signed = 2.0 * np.sign(target_dist_signed) * align_fac

        # LQR Update
        u_fwd = controller.update(mpc_state, sim_t, target_pos=target_dist_signed)
        
        # Actuation
        
        # Stability Priority: Suppress turning if tilting too much
        # If tilt > 0.15 rad (~8 deg), Turn Torque -> 0.0
        tilt_suppress = max(0.0, 1.0 - (abs(theta_curr) / 0.15))
        turn_torque *= tilt_suppress
        
        turn_torque = np.clip(turn_torque, -10.0, 10.0) # Conservative limit 
        tau_left = (u_fwd / 2.0) - (turn_torque / 2.0)
        tau_right = (u_fwd / 2.0) + (turn_torque / 2.0)
        
        # Clip to Physical Limits (Modified to 200)
        tau_left = np.clip(tau_left, -200, 200)
        tau_right = np.clip(tau_right, -200, 200)
        
        data.ctrl[0] = tau_left
        data.ctrl[1] = tau_right
        
        mj.mj_step(model, data)

        # Time Sync
        target_wall_time = sim_t / sim_speed_factor
        current_wall_time = time.time() - start_wall_time
        drift = target_wall_time - current_wall_time
        if drift > 0: time.sleep(drift)
        
        # Recording
        if i % 10 == 0:
            time_data.append(sim_t)
            x_data.append(x_curr)
            y_data.append(y_curr)
            theta_data.append(theta_curr)
            ctrl_data.append(u_fwd)
            tau_left_data.append(tau_left)
            tau_right_data.append(tau_right)
            
        # Rendering
        if time.time() - last_render_time > 0.033:
            # Camera Logic (Third Person Follow)
            if task_mode in ["Nav", "Hunt"]:
                # ISO View (User Requested: Beside/Parallel/Top)
                # -45 deg = Back-Right Isometric View (Looking Forward)
                cam.azimuth = np.degrees(yaw_curr) - 45 
                cam.elevation = -45 
                cam.distance = 9.0
            
            # Main View
            viewport = mj.MjrRect(0, 0, 1600, 1200)
            mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
            mj.mjr_render(viewport, scene, context)
            
            # PiP View (Front Camera)
            vp_front = mj.MjrRect(1180, 20, 400, 300) # Bottom Right
            mj.mjv_updateScene(model, data, opt, None, cam_front, mj.mjtCatBit.mjCAT_ALL.value, scene)
            mj.mjr_render(vp_front, scene, context)
            
            # Vision Processing
            if task_mode == "Hunt" and target_color:
                mj.mjr_readPixels(rgb_buffer, None, vp_front, context)
                # Flip vertically (OpenGl is bottom-up)
                rgb_img = np.flipud(rgb_buffer) 
                
                res = detector.detect(rgb_img, target_color)
                if res:
                    cx, cy, area = res
                    vision_steering = cx # -1(Left) to 1(Right)
                    vision_active = True
                    vision_area = area
                    # Stop if close
                    if area > 0.2: 
                        print("Target Reached (Vision)!")
                else:
                    vision_active = False
                    vision_steering = 0.0
                    vision_area = 0.0
            
            title = f"Segway | T: {sim_t:.1f} | Pos: ({x_curr:.1f}, {y_curr:.1f}) | Tilt: {theta_curr:.3f}"
            glfw.set_window_title(window, title)
            glfw.swap_buffers(window)
            glfw.poll_events()
            last_render_time = time.time()
            if glfw.window_should_close(window): break
            
        # Stability Check
        lin_vel = np.sqrt(vx_curr**2 + vy_curr**2)
        th_dist = 0.05 if task_mode == "Balance" else 0.3
        th_vel = 0.05 if task_mode == "Balance" else 0.2
        th_tilt = 0.02 if task_mode == "Balance" else 0.05

        if dist_to_target < th_dist and lin_vel < th_vel and abs(theta_curr) < th_tilt: 
            stability_duration += dt
        else:
            stability_duration = 0.0
            
        if stability_duration > 2.0:
            print(f"Goal Reached! (Stable {stability_duration:.1f}s)")
            break
            
    glfw.terminate()
    
    # 4. Plots
    if show_plots and len(time_data) > 0:
        plt.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=(16, 12))
        
        if task_mode == "Balance":
            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(time_data, theta_data, 'r', linewidth=2)
            ax1.axhline(0, color='k', linestyle='--')
            ax1.set_ylabel("Tilt (rad)")
            ax1.set_title("Stability")
            ax1.grid(True)
            
            ax2 = plt.subplot(2, 2, 2)
            ax2.plot(time_data, x_data, 'b', linewidth=2)
            ax2.set_ylabel("X (m)")
            ax2.set_title("Position")
            ax2.grid(True)

            ax3 = plt.subplot(2, 1, 2)
            ax3.plot(time_data, tau_left_data, 'g-', label="Left", linewidth=1.5)
            ax3.plot(time_data, tau_right_data, 'y--', label="Right", linewidth=1.5)
            ax3.set_ylabel("Torque (Nm)")
            ax3.set_xlabel("Time (s)")
            ax3.legend()
            ax3.grid(True)
        else:
            ax1 = plt.subplot(2, 2, (1, 3))
            ax1.plot(x_data, y_data, 'b-', linewidth=3, label="Actual")
            
            if full_path:
                px = [p[0] for p in full_path]
                py = [p[1] for p in full_path]
                ax1.plot(px, py, 'c--', linewidth=2, label=f"Plan ({planner_name})")
                
            ax1.plot(initial_pos, initial_pos_y, 'go', ms=10, label="Start")
            if task_mode != "Hunt":
                ax1.plot(target_pos, target_pos_y, 'r*', ms=20, label="Target")
            ax1.set_title(f"Trajectory ({strat_name})")
            ax1.axis('equal')
            ax1.grid(True)
            ax1.axis('equal')
            ax1.grid(True)
            
            # Plot LiDAR
            if len(lidar_hits) > 0:
                all_hits = np.vstack(lidar_hits)
                # Downsample for plotting speed
                if len(all_hits) > 5000: all_hits = all_hits[::10]
                ax1.scatter(all_hits[:,0], all_hits[:,1], c='r', s=1, alpha=0.3, label="LiDAR")
                
            ax1.legend()
            
            ax2 = plt.subplot(2, 2, 2)
            ax2.plot(time_data, theta_data, 'r', label="Tilt")
            ax2.axhline(0, color='k', linestyle='--')
            ax2.set_title("Balance")
            ax2.grid(True)
            
            ax3 = plt.subplot(2, 2, 4)
            ax3.plot(time_data, ctrl_data, 'g', label="Force")
            ax3.set_ylabel("Force (N)")
            ax3.set_xlabel("Time (s)")
            ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Default Run")
    run_segway_simulation()
