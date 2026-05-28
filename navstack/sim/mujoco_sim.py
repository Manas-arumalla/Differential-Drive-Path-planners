"""
Mobile Robot MuJoCo Simulation Runner
Runs physics simulation with GLFW visualization.
Supports goal-based termination and path following.
"""
import mujoco
from mujoco.glfw import glfw
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Local imports
from navstack.sim import model_generator as model_gen
from navstack.controllers import navigation as nav_ctrl


def quaternion_to_yaw(quat):
    """Convert quaternion [w, x, y, z] to yaw angle."""
    r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x,y,z,w]
    euler = r.as_euler('xyz')
    return euler[2]  # Yaw


def run_simulation(
    robot_type='differential_2wd',
    robot_params=None,
    obstacles=None,
    start=(0, 0, 0),       # (x, y, theta)
    goal=(5, 5),           # (x, y)
    path=None,             # List of (x, y) waypoints
    nav_controller='PurePursuit',
    nav_params=None,
    duration=60.0,
    show_plots=True,
    realtime=True,
    goal_threshold=0.3
):
    """
    Run MuJoCo simulation for mobile robot path following.
    
    Args:
        robot_type: Robot type string
        robot_params: Dict of robot parameters (None for defaults)
        obstacles: List of obstacle dicts
        start: (x, y, theta) start pose
        goal: (x, y) goal position
        path: List of (x, y) waypoints (uses [start, goal] if None)
        nav_controller: Controller name or instance
        nav_params: Controller parameters dict
        duration: Max simulation time (seconds)
        show_plots: Show matplotlib plots after simulation
        realtime: Run at realtime speed
        goal_threshold: Distance to consider goal reached
    
    Returns:
        dict: Simulation data (time, positions, velocities, etc.)
    """
    print(f"\n{'='*60}")
    print(f"Mobile Robot Simulation")
    print(f"Robot: {robot_type}")
    print(f"Controller: {nav_controller}")
    print(f"Start: {start} -> Goal: {goal}")
    print(f"{'='*60}\n")
    
    # Generate path if not provided
    if path is None or len(path) == 0:
        path = [(start[0], start[1]), goal]
    
    # Ensure goal is in path
    if path[-1] != goal:
        path.append(goal)
    
    # Generate MJCF model
    if obstacles is None:
        obstacles = []
    
    bounds = [
        min(start[0], goal[0]) - 5,
        max(start[0], goal[0]) + 5,
        min(start[1], goal[1]) - 5,
        max(start[1], goal[1]) + 5
    ]
    
    xml_file = model_gen.generate_mjcf(
        robot_type=robot_type,
        robot_params=robot_params,
        obstacles=obstacles,
        bounds=bounds,
        filename="mobile_robot_scene.xml"
    )
    
    # Load model
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)
    
    # Reset to initial state from XML first
    mujoco.mj_resetData(model, data)
    
    # Set initial XY position (keep Z from XML - robot is placed at correct height)
    data.qpos[0] = start[0]  # x
    data.qpos[1] = start[1]  # y
    # Don't touch qpos[2] (z) - let XML define the initial height
    
    # Set initial orientation (yaw)
    yaw = float(start[2]) if len(start) > 2 else 0.0
    quat = Rotation.from_euler('xyz', [0, 0, yaw]).as_quat()  # [x,y,z,w]
    data.qpos[3] = quat[3]  # w
    data.qpos[4] = quat[0]  # x
    data.qpos[5] = quat[1]  # y
    data.qpos[6] = quat[2]  # z
    
    # Reset velocities to zero for stability
    data.qvel[:] = 0
    data.ctrl[:] = 0
    
    # Forward dynamics to update state
    mujoco.mj_forward(model, data)
    
    # Create controller
    if isinstance(nav_controller, str):
        ctrl_params = nav_params or {}
        controller = nav_ctrl.get_controller(nav_controller, **ctrl_params)
        if hasattr(controller, 'set_obstacles'):
            controller.set_obstacles(obstacles)
    else:
        controller = nav_controller
    
    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    width, height = 1280, 720
    window = glfw.create_window(width, height, f"Mobile Robot - {robot_type}", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # Camera setup
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    cam.distance = 5.0
    cam.elevation = -45
    cam.azimuth = 135
    
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    
    # Data logging
    log_time = []
    log_x, log_y, log_theta = [], [], []
    log_v, log_omega = [], []
    log_target_x, log_target_y = [], []
    log_ctrl = []
    
    # Simulation parameters
    dt = model.opt.timestep
    sim_time = 0.0
    current_wp_idx = 1  # Start aiming at first waypoint (index 0 is start)
    
    print("Starting simulation... Press ESC to exit early.")
    start_wall_time = time.time()
    
    # Main simulation loop
    while not glfw.window_should_close(window):
        step_start = time.time()
        
        # Get robot state from sensors
        pos = data.sensordata[0:3]  # robot_pos
        quat = data.sensordata[3:7]  # robot_quat [w,x,y,z]
        vel = data.sensordata[7:10]  # robot_vel
        angvel = data.sensordata[10:13]  # robot_angvel
        
        x, y = pos[0], pos[1]
        theta = quaternion_to_yaw(quat)
        
        # Current target waypoint
        if current_wp_idx < len(path):
            target = path[current_wp_idx]
        else:
            target = path[-1]
        
        # Distance to current waypoint
        dist_to_wp = np.sqrt((target[0] - x)**2 + (target[1] - y)**2)
        
        # Waypoint switching
        if dist_to_wp < 0.4 and current_wp_idx < len(path) - 1:
            current_wp_idx += 1
            print(f"Reached waypoint {current_wp_idx-1}, heading to {path[current_wp_idx]}")
        
        # Distance to final goal
        final_goal = path[-1]
        dist_to_goal = np.sqrt((final_goal[0] - x)**2 + (final_goal[1] - y)**2)
        
        # Check if goal reached
        robot_vel = np.sqrt(vel[0]**2 + vel[1]**2)
        if dist_to_goal < goal_threshold and robot_vel < 0.1:
            print(f"\n✓ GOAL REACHED! Time: {sim_time:.2f}s, Final Error: {dist_to_goal:.3f}m")
            break
        
        # Check timeout
        if sim_time >= duration:
            print(f"\n✗ Timeout! Distance to goal: {dist_to_goal:.2f}m")
            break
        
        # Get control commands
        pose = (x, y, float(theta))
        v_cmd, omega_cmd, current_wp_idx = controller.compute_control(
            pose, path, current_wp_idx
        )
        
        # Safety: clamp control values to prevent instability
        v_cmd = np.clip(float(v_cmd), -2.0, 2.0)
        omega_cmd = np.clip(float(omega_cmd), -3.0, 3.0)
        
        # Apply control based on robot type
        if robot_type in ['differential_2wd', 'differential_4wd']:
            # Differential drive: convert (v, omega) to wheel angular velocities
            wheel_base = robot_params.get('wheel_separation', 0.22) if robot_params else 0.22
            wheel_radius = robot_params.get('wheel_radius', 0.05) if robot_params else 0.05
            
            # Calculate target wheel angular velocities (rad/s)
            # For v=0.5 m/s, omega_wheel = v/r = 0.5/0.05 = 10 rad/s
            omega_left = (v_cmd - omega_cmd * wheel_base / 2) / wheel_radius
            omega_right = (v_cmd + omega_cmd * wheel_base / 2) / wheel_radius
            
            # Motor actuators: convert desired angular velocity to torque
            # 4WD skid-steer needs higher gain because wheels must slide to turn
            if robot_type == 'differential_4wd':
                k_motor = 0.3  # Gain for 4WD
                max_torque = 4.0
            else:
                k_motor = 1.0  # Max gain for 2WD to prevent stall
                max_torque = 3.0
            
            data.ctrl[0] = np.clip(omega_left * k_motor, -max_torque, max_torque)   # FL / Left
            data.ctrl[1] = np.clip(omega_right * k_motor, -max_torque, max_torque)  # FR / Right
            
            if robot_type == 'differential_4wd':
                data.ctrl[2] = np.clip(omega_left * k_motor, -max_torque, max_torque)   # RL
                data.ctrl[3] = np.clip(omega_right * k_motor, -max_torque, max_torque)  # RR
        
        elif robot_type == 'mecanum':
            # Mecanum drive with Sphere wheels = Skid Steer Approximation
            # We want Differential Steering for rotation (Left -, Right +)
            # v_cmd (Forward): All +
            
            wheel_radius = robot_params.get('wheel_radius', 0.05) if robot_params else 0.05
            half_width = (robot_params.get('wheel_separation', 0.24) if robot_params else 0.24) / 2
            
            k_motor = 0.3  # Gain for Mecanum
            max_torque = 4.0
            
            # Use Differential Drive logic for Sphere-based Mecanum
            omega_left = (v_cmd - omega_cmd * half_width) / wheel_radius
            omega_right = (v_cmd + omega_cmd * half_width) / wheel_radius
            
            data.ctrl[0] = np.clip(omega_left * k_motor, -max_torque, max_torque)  # FL
            data.ctrl[1] = np.clip(omega_right * k_motor, -max_torque, max_torque)  # FR
            data.ctrl[2] = np.clip(omega_left * k_motor, -max_torque, max_torque)  # RL
            data.ctrl[3] = np.clip(omega_right * k_motor, -max_torque, max_torque)  # RR
        
        elif robot_type == 'omni_3wheel':
            # 3-wheel omni
            body_radius = robot_params.get('body_radius', 0.15) if robot_params else 0.15
            wheel_radius = robot_params.get('wheel_radius', 0.04) if robot_params else 0.04
            
            k_motor = 0.5  # High gain for Omni (cylinder friction is high)
            max_torque = 4.0
            
            # Cylinders roll in 'angle' direction. Standard sign.
            # Correction: Positive rotation causes backward motion. Invert sign.
            for i in range(3):
                angle = i * 2 * np.pi / 3
                wheel_vel = -(v_cmd * np.cos(angle) + omega_cmd * body_radius)
                data.ctrl[i] = np.clip(wheel_vel / wheel_radius * k_motor, -max_torque, max_torque)
        
        elif robot_type in ['ackermann', 'bicycle']:
            # Ackermann steering
            wheelbase = robot_params.get('wheelbase', 0.35) if robot_params else 0.35
            wheel_radius = robot_params.get('wheel_radius', 0.06) if robot_params else 0.06
            max_steer = np.radians(robot_params.get('max_steer_angle', 30)) if robot_params else 0.5
            
            # Calculate steering angle from omega and velocity
            if abs(v_cmd) > 0.01:
                steer_angle = np.arctan(omega_cmd * wheelbase / v_cmd)
                steer_angle = np.clip(steer_angle, -max_steer, max_steer)
            else:
                steer_angle = 0.0
            
            wheel_omega = v_cmd / wheel_radius
            k_motor = 0.6 # High gain for largeAckermann/Bicycle
            max_torque = 5.0
            
            if robot_type == 'ackermann':
                data.ctrl[0] = np.clip(wheel_omega * k_motor, -max_torque, max_torque) # RL
                data.ctrl[1] = np.clip(wheel_omega * k_motor, -max_torque, max_torque) # RR
                data.ctrl[2] = steer_angle # FL
                data.ctrl[3] = steer_angle # FR
            else: # bicycle
                data.ctrl[0] = steer_angle # Front Steer
                data.ctrl[1] = np.clip(wheel_omega * k_motor, -max_torque, max_torque) # Rear Drive
        
        # Step simulation with stability check
        mujoco.mj_step(model, data)
        sim_time += dt
        
        # Check for simulation instability (NaN detection)
        if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
            print(f"\n⚠ Simulation became unstable at t={sim_time:.2f}s")
            print("Try reducing v_max or omega_max in controller parameters")
            break
        
        # Log data
        log_time.append(sim_time)
        log_x.append(x)
        log_y.append(y)
        log_theta.append(theta)
        log_v.append(v_cmd)
        log_omega.append(omega_cmd)
        log_target_x.append(target[0])
        log_target_y.append(target[1])
        
        # Render at ~30Hz
        if int(sim_time * 100) % 3 == 0:
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
            mujoco.mjr_render(viewport, scene, context)
            
            title = f"T: {sim_time:.1f}s | Pos: ({x:.2f}, {y:.2f}) | θ: {np.degrees(theta):.1f}° | Dist: {dist_to_goal:.2f}m"
            glfw.set_window_title(window, title)
            
            glfw.swap_buffers(window)
            glfw.poll_events()
        
        # Realtime sync
        if realtime:
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    # Cleanup
    glfw.terminate()
    
    # Prepare results
    sim_data = {
        'time': np.array(log_time),
        'x': np.array(log_x),
        'y': np.array(log_y),
        'theta': np.array(log_theta),
        'v': np.array(log_v),
        'omega': np.array(log_omega),
        'target_x': np.array(log_target_x),
        'target_y': np.array(log_target_y),
        'path': path,
        'goal_reached': dist_to_goal < goal_threshold,
        'final_distance': dist_to_goal,
        'total_time': sim_time
    }
    
    # Show plots (save to file to avoid thread issues, then show)
    if show_plots and len(log_time) > 0:
        try:
            plot_results(sim_data, robot_type, controller.name)
        except Exception as e:
            print(f"Note: Could not display plots ({e}). Data still available.")
    
    return sim_data


def plot_results(sim_data, robot_type, controller_name):
    """Plot simulation results and save to file."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Mobile Robot Simulation Results\n{robot_type} with {controller_name}", 
                 fontsize=14, fontweight='bold')
    
    # 1. 2D Trajectory
    ax1 = plt.subplot(2, 2, (1, 3))
    path = sim_data['path']
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    
    ax1.plot(path_x, path_y, 'c--', linewidth=2, alpha=0.7, label='Planned Path')
    ax1.plot(sim_data['x'], sim_data['y'], 'lime', linewidth=2, label='Actual Path')
    ax1.plot(path_x[0], path_y[0], 'go', markersize=12, label='Start')
    ax1.plot(path_x[-1], path_y[-1], 'r*', markersize=15, label='Goal')
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('2D Trajectory', fontsize=12)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # 2. Velocity Profile
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(sim_data['time'], sim_data['v'], 'cyan', linewidth=1.5, label='Linear Velocity')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Angular Velocity / Heading
    ax3 = plt.subplot(2, 2, 4)
    ax3.plot(sim_data['time'], np.degrees(sim_data['theta']), 'orange', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heading (degrees)')
    ax3.set_title('Robot Heading')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to file
    plot_file = 'simulation_results.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {plot_file}")


def run_demo():
    """Run a demo simulation with default settings."""
    # Sample path
    demo_path = [
        (0, 0), (2, 0), (4, 1), (5, 3), (4, 5), (2, 6), (0, 5)
    ]
    
    # Sample obstacles
    demo_obstacles = [
        {'type': 'cylinder', 'pos': [3, 2], 'size': [0.3, 1.0], 'color': 'red'},
        {'type': 'box', 'pos': [1.5, 4], 'size': [0.5, 0.5, 0.3], 'color': 'blue'},
    ]
    
    # Run simulation
    result = run_simulation(
        robot_type='differential_2wd',
        start=(0, 0, 0),
        goal=(0, 5),
        path=demo_path,
        obstacles=demo_obstacles,
        nav_controller='PurePursuit',
        nav_params={'lookahead': 0.6, 'v_max': 0.8},
        duration=60.0,
        show_plots=True
    )
    
    print(f"\nSimulation Complete!")
    print(f"Goal Reached: {result['goal_reached']}")
    print(f"Final Distance to Goal: {result['final_distance']:.3f}m")
    print(f"Total Time: {result['total_time']:.2f}s")


if __name__ == "__main__":
    run_demo()
