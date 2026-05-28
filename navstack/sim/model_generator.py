"""
Mobile Robot Model Generator
Generates MuJoCo MJCF XML files for various mobile robot configurations.
Supports: Differential (2WD, 4WD), Mecanum, Omni, Ackermann, Bicycle
"""
import numpy as np

# Default robot configurations
ROBOT_CONFIGS = {
    'differential_2wd': {
        'body_length': 0.5, 'body_width': 0.15, 'body_height': 0.1, # Reduced width further to 0.15
        'body_mass': 2.0,
        'wheel_radius': 0.05, 'wheel_width': 0.02, 'wheel_mass': 0.1,
        'wheel_separation': 0.22
    },
    'differential_4wd': {
        'body_length': 0.5, 'body_width': 0.25, 'body_height': 0.1,
        'body_mass': 5.0,
        'wheel_radius': 0.05, 'wheel_width': 0.03, 'wheel_mass': 0.2,
        'wheel_separation': 0.28, 'axle_separation': 0.35
    },
    'mecanum': {
        'body_length': 0.5, 'body_width': 0.25, 'body_height': 0.1,
        'body_mass': 5.0,
        'wheel_radius': 0.05, 'wheel_width': 0.03, 'wheel_mass': 0.2,
        'wheel_separation': 0.28, 'axle_separation': 0.35
    },
    'omni_3wheel': {
        'body_radius': 0.15, 'body_height': 0.1,
        'body_mass': 5.0,
        'wheel_radius': 0.04, 'wheel_width': 0.02, 'wheel_mass': 0.1
    },
    'ackermann': {
        'body_length': 0.5, 'body_width': 0.20, 'body_height': 0.1,  # Reduced width
        'body_mass': 5.0,
        'wheel_radius': 0.06, 'wheel_width': 0.03, 'wheel_mass': 0.2,
        'wheel_separation': 0.35, 'wheelbase': 0.35,  # Increased separation
        'max_steer_angle': 30
    },
    'bicycle': {
        'body_length': 0.4, 'body_width': 0.08, 'body_height': 0.08,  # Reduced width
        'body_mass': 3.0,
        'wheel_radius': 0.08, 'wheel_width': 0.025, 'wheel_mass': 0.25, # Larger wheels
        'wheelbase': 0.35, 'max_steer_angle': 35
    }
}


def get_rgba(color_name):
    """Convert color name to RGBA string."""
    colors = {
        'red': '0.8 0.1 0.1 1', 'green': '0.1 0.8 0.1 1',
        'blue': '0.1 0.1 0.8 1', 'yellow': '0.8 0.8 0.1 1',
        'orange': '0.8 0.4 0.1 1', 'purple': '0.8 0.1 0.8 1',
        'cyan': '0.1 0.8 0.8 1', 'white': '0.9 0.9 0.9 1',
        'gray': '0.5 0.5 0.5 1', 'dark_gray': '0.2 0.2 0.2 1',
        'black': '0.1 0.1 0.1 1'
    }
    return colors.get(color_name, '0.5 0.5 0.5 1')


def generate_differential_2wd(params):
    """Generate 2-wheel differential drive robot."""
    bl, bw, bh = params['body_length'], params['body_width'], params['body_height']
    bm = params['body_mass']
    wr, ww, wm = params['wheel_radius'], params['wheel_width'], params['wheel_mass']
    ws = params['wheel_separation']
    
    # Caster wheel radius (smaller than drive wheels)
    caster_radius = wr * 0.4  # 0.02m for 0.05m main wheels
    
    # Body center height: must be high enough for both drive wheels and caster
    # Drive wheel center at ground level: z = wr  
    # Body bottom should clear ground: body_z - bh/2 > 0
    # We want drive wheels and caster to both touch ground
    body_z = wr + bh/2  # = 0.05 + 0.05 = 0.10m
    
    # Drive wheel offset from body center (they should be at z = wr above ground)
    # wheel_world_z = body_z + wheel_z_offset = wr
    # so wheel_z_offset = wr - body_z = wr - (wr + bh/2) = -bh/2
    wheel_z_offset = -bh/2  # = -0.05m, wheel center at 0.10 - 0.05 = 0.05 = wr ✓
    
    # Caster needs to touch ground: caster_world_z - caster_radius = 0
    # caster_world_z = body_z + caster_z_offset = caster_radius
    # caster_z_offset = caster_radius - body_z
    caster_z_offset = caster_radius - body_z  # = 0.02 - 0.10 = -0.08m
    
    # Wheel X positions: drive wheels near REAR, caster at FRONT
    drive_wheel_x = -bl/4  # Rear of body (-0.075m)
    caster_x = bl/3        # Front of body (0.1m)
    
    xml = f'''
        <!-- Differential 2WD Robot -->
        <body name="robot" pos="0 0 {body_z}">
            <joint name="root" type="free"/>
            <geom name="chassis" type="box" size="{bl/2} {bw/2} {bh/2}" 
                  rgba="{get_rgba('dark_gray')}" mass="{bm}"/>
            
            <!-- Caster Wheel (Sphere) -->
            <geom name="caster" type="sphere" pos="{caster_x} 0 {caster_z_offset}" size="{caster_radius}" 
                  rgba="{get_rgba('white')}" mass="0.1" friction="0.0 0.0 0.0"/>
            
            <!-- Left Wheel (rear) -->
            <body name="wheel_left" pos="{drive_wheel_x} {ws/2} {wheel_z_offset}">
                <joint name="joint_left" type="hinge" axis="0 1 0" damping="0.05" frictionloss="0.001"/>
                <geom name="wheel_l" type="cylinder" size="{wr} {ww/2}" 
                      euler="90 0 0" rgba="{get_rgba('orange')}" mass="{wm}"
                      friction="1.5 0.005 0.0001" condim="3"/>
            </body>
            
            <!-- Right Wheel (rear) -->
            <body name="wheel_right" pos="{drive_wheel_x} {-ws/2} {wheel_z_offset}">
                <joint name="joint_right" type="hinge" axis="0 1 0" damping="0.05" frictionloss="0.001"/>
                <geom name="wheel_r" type="cylinder" size="{wr} {ww/2}" 
                      euler="90 0 0" rgba="{get_rgba('orange')}" mass="{wm}"
                      friction="1.5 0.005 0.0001" condim="3"/>
            </body>
            
            <!-- Direction indicator -->
            <geom name="arrow" type="box" pos="{bl/3} 0 {bh/2+0.01}" 
                  size="0.03 0.01 0.005" rgba="0 1 0 1"/>
        </body>
    '''
    return xml


def generate_differential_4wd(params):
    """Generate 4-wheel differential (skid-steer) robot."""
    bl, bw, bh = params['body_length'], params['body_width'], params['body_height']
    bm = params['body_mass']
    wr, ww, wm = params['wheel_radius'], params['wheel_width'], params['wheel_mass']
    ws = params['wheel_separation']
    axle_sep = params['axle_separation']
    
    body_z = wr + bh/2
    wheel_z_offset = -(bh/2)
    front_x, rear_x = axle_sep/2, -axle_sep/2
    
    xml = f'''
        <!-- Differential 4WD Robot (Skid-Steer) -->
        <body name="robot" pos="0 0 {body_z}">
            <joint name="root" type="free"/>
            <geom name="chassis" type="box" size="{bl/2} {bw/2} {bh/2}" 
                  rgba="{get_rgba('dark_gray')}" mass="{bm}"/>
            
            <!-- Front Left -->
            <body name="wheel_fl" pos="{front_x} {ws/2} {wheel_z_offset}">
                <joint name="joint_fl" type="hinge" axis="0 1 0" damping="0.05" frictionloss="0.001"/>
                <geom type="cylinder" size="{wr} {ww/2}" euler="90 0 0" 
                      rgba="{get_rgba('orange')}" mass="{wm}" friction="0.3 0.005 0.0001"/>
            </body>
            <!-- Front Right -->
            <body name="wheel_fr" pos="{front_x} {-ws/2} {wheel_z_offset}">
                <joint name="joint_fr" type="hinge" axis="0 1 0" damping="0.05" frictionloss="0.001"/>
                <geom type="cylinder" size="{wr} {ww/2}" euler="90 0 0" 
                      rgba="{get_rgba('orange')}" mass="{wm}" friction="0.3 0.005 0.0001"/>
            </body>
            <!-- Rear Left -->
            <body name="wheel_rl" pos="{rear_x} {ws/2} {wheel_z_offset}">
                <joint name="joint_rl" type="hinge" axis="0 1 0" damping="0.05" frictionloss="0.001"/>
                <geom type="cylinder" size="{wr} {ww/2}" euler="90 0 0" 
                      rgba="{get_rgba('orange')}" mass="{wm}" friction="0.3 0.005 0.0001"/>
            </body>
            <!-- Rear Right -->
            <body name="wheel_rr" pos="{rear_x} {-ws/2} {wheel_z_offset}">
                <joint name="joint_rr" type="hinge" axis="0 1 0" damping="0.05" frictionloss="0.001"/>
                <geom type="cylinder" size="{wr} {ww/2}" euler="90 0 0" 
                      rgba="{get_rgba('orange')}" mass="{wm}" friction="0.3 0.005 0.0001"/>
            </body>
            
            <geom name="arrow" type="box" pos="{bl/3} 0 {bh/2+0.01}" 
                  size="0.03 0.01 0.005" rgba="0 1 0 1"/>
        </body>
    '''
    return xml


def generate_mecanum(params):
    """Generate 4-wheel mecanum robot (using spheres for simplified omni physics)."""
    bl, bw, bh = params['body_length'], params['body_width'], params['body_height']
    bm = params['body_mass']
    wr, ww, wm = params['wheel_radius'], params['wheel_width'], params['wheel_mass']
    ws = params['wheel_separation']
    axle_sep = params['axle_separation']
    
    body_z = wr + bh/2
    wheel_z_offset = -(bh/2)
    front_x, rear_x = axle_sep/2, -axle_sep/2
    
    # Mecanum wheels have rollers at 45 degrees
    xml = f'''
        <!-- Mecanum Robot -->
        <body name="robot" pos="0 0 {body_z}">
            <joint name="root" type="free"/>
            <geom name="chassis" type="box" size="{bl/2} {bw/2} {bh/2}" 
                  rgba="{get_rgba('dark_gray')}" mass="{bm}"/>
            
            <!-- Use SPHERES for wheels to approximate omni motion with directional force -->
            <!-- Front Left -->
            <body name="wheel_fl" pos="{front_x} {ws/2} {wheel_z_offset}">
                <joint name="joint_fl" type="hinge" axis="0 1 0" damping="0.05" frictionloss="0.001"/>
                <geom type="sphere" size="{wr}" rgba="{get_rgba('gray')}" mass="{wm}" friction="0.8 0.005 0.0001"/>
            </body>
            <!-- Front Right -->
            <body name="wheel_fr" pos="{front_x} {-ws/2} {wheel_z_offset}">
                <joint name="joint_fr" type="hinge" axis="0 1 0" damping="0.05" frictionloss="0.001"/>
                <geom type="sphere" size="{wr}" rgba="{get_rgba('gray')}" mass="{wm}" friction="0.8 0.005 0.0001"/>
            </body>
            <!-- Rear Left -->
            <body name="wheel_rl" pos="{rear_x} {ws/2} {wheel_z_offset}">
                <joint name="joint_rl" type="hinge" axis="0 1 0" damping="0.05" frictionloss="0.001"/>
                <geom type="sphere" size="{wr}" rgba="{get_rgba('gray')}" mass="{wm}" friction="0.8 0.005 0.0001"/>
            </body>
            <!-- Rear Right -->
            <body name="wheel_rr" pos="{rear_x} {-ws/2} {wheel_z_offset}">
                <joint name="joint_rr" type="hinge" axis="0 1 0" damping="0.05" frictionloss="0.001"/>
                <geom type="sphere" size="{wr}" rgba="{get_rgba('gray')}" mass="{wm}" friction="0.8 0.005 0.0001"/>
            </body>
            
            <geom name="arrow" type="box" pos="{bl/3} 0 {bh/2+0.01}" 
                  size="0.03 0.01 0.005" rgba="0 1 0 1"/>
        </body>
    '''
    return xml


def generate_omni_3wheel(params):
    """Generate 3-wheel omnidirectional robot."""
    br = params['body_radius']
    bh = params['body_height']
    bm = params['body_mass']
    wr, ww, wm = params['wheel_radius'], params['wheel_width'], params['wheel_mass']
    
    body_z = wr + bh/2
    wheel_z_offset = -(bh/2)
    wheel_dist = br * 0.9  # Wheels placed near edge
    
    # 3 wheels at 120 degree intervals
    angles = [0, 120, 240]
    
    xml = f'''
        <!-- Omni 3-Wheel Robot -->
        <body name="robot" pos="0 0 {body_z}">
            <joint name="root" type="free"/>
            <geom name="chassis" type="cylinder" size="{br - 0.05} {bh/2}"
                  rgba="{get_rgba('cyan')}" mass="{bm}"/>
    '''
    
    for i, angle in enumerate(angles):
        rad = np.radians(angle)
        wx = wheel_dist * np.cos(rad)
        wy = wheel_dist * np.sin(rad)
        wheel_euler = f"90 0 {angle + 90}"  # Align wheel direction
        
        xml += f'''
            <body name="wheel_{i}" pos="{wx:.4f} {wy:.4f} {wheel_z_offset}">
                <joint name="joint_{i}" type="hinge" axis="0 1 0" damping="0.1" frictionloss="0.01"/>
                <geom type="cylinder" size="{wr} {ww/2}" euler="{wheel_euler}" 
                      rgba="{get_rgba('green')}" mass="{wm}" friction="0.5 0.005 0.0001"/>
            </body>
        '''
    
    xml += f'''
            <geom name="arrow" type="box" pos="{br*0.5} 0 {bh/2+0.01}" 
                  size="0.03 0.01 0.005" rgba="1 0 0 1"/>
        </body>
    '''
    return xml


def generate_ackermann(params):
    """Generate 4-wheel Ackermann steering robot."""
    bl, bw, bh = params['body_length'], params['body_width'], params['body_height']
    bm = params['body_mass']
    wr, ww, wm = params['wheel_radius'], params['wheel_width'], params['wheel_mass']
    ws = params['wheel_separation']
    wb = params['wheelbase']
    max_steer = np.radians(params['max_steer_angle'])
    
    body_z = wr + bh/2
    wheel_z_offset = -(bh/2)
    front_x, rear_x = wb/2, -wb/2
    
    # Steering hub mass (Increased for stability)
    hub_mass = 0.1
    
    xml = f'''
        <!-- Ackermann Robot -->
        <body name="robot" pos="0 0 {body_z}">
            <joint name="root" type="free"/>
            <geom name="chassis" type="box" size="{bl/2} {bw/2} {bh/2}" 
                  rgba="{get_rgba('purple')}" mass="{bm}"/>
            
            <!-- Front Left (steerable) -->
            <body name="steer_fl" pos="{front_x} {ws/2} {wheel_z_offset}">
                <inertial pos="0 0 0" mass="{hub_mass}" diaginertia="0.001 0.001 0.001"/>
                <joint name="steer_joint_fl" type="hinge" axis="0 0 1" 
                       range="{-max_steer:.3f} {max_steer:.3f}" damping="0.5"/>
                <body name="wheel_fl">
                    <joint name="joint_fl" type="hinge" axis="0 1 0" damping="0.1" frictionloss="0.01"/>
                    <geom type="cylinder" size="{wr} {ww/2}" euler="90 0 0" 
                          rgba="{get_rgba('gray')}" mass="{wm}" friction="1.0 0.005 0.0001"/>
                </body>
            </body>
            
            <!-- Front Right (steerable) -->
            <body name="steer_fr" pos="{front_x} {-ws/2} {wheel_z_offset}">
                <inertial pos="0 0 0" mass="{hub_mass}" diaginertia="0.001 0.001 0.001"/>
                <joint name="steer_joint_fr" type="hinge" axis="0 0 1" 
                       range="{-max_steer:.3f} {max_steer:.3f}" damping="0.5"/>
                <body name="wheel_fr">
                    <joint name="joint_fr" type="hinge" axis="0 1 0" damping="0.1" frictionloss="0.01"/>
                    <geom type="cylinder" size="{wr} {ww/2}" euler="90 0 0" 
                          rgba="{get_rgba('gray')}" mass="{wm}" friction="1.0 0.005 0.0001"/>
                </body>
            </body>
            
            <!-- Rear Left (fixed, drive) -->
            <body name="wheel_rl" pos="{rear_x} {ws/2} {wheel_z_offset}">
                <joint name="joint_rl" type="hinge" axis="0 1 0" damping="0.1" frictionloss="0.01"/>
                <geom type="cylinder" size="{wr} {ww/2}" euler="90 0 0" 
                      rgba="{get_rgba('gray')}" mass="{wm}" friction="1.0 0.005 0.0001"/>
            </body>
            
            <!-- Rear Right (fixed) -->
            <body name="wheel_rr" pos="{rear_x} {-ws/2} {wheel_z_offset}">
                <joint name="joint_rr" type="hinge" axis="0 1 0" damping="0.1" frictionloss="0.01"/>
                <geom type="cylinder" size="{wr} {ww/2}" euler="90 0 0" 
                      rgba="{get_rgba('gray')}" mass="{wm}" friction="1.0 0.005 0.0001"/>
            </body>
            
            <geom name="arrow" type="box" pos="{bl/3} 0 {bh/2+0.01}" 
                  size="0.03 0.01 0.005" rgba="0 1 0 1"/>
        </body>
    '''
    return xml


def generate_bicycle(params):
    """Generate bicycle model (2-wheel, front steering)."""
    bl, bw, bh = params['body_length'], params['body_width'], params['body_height']
    bm = params['body_mass']
    wr, ww, wm = params['wheel_radius'], params['wheel_width'], params['wheel_mass']
    wb = params['wheelbase']
    max_steer = np.radians(params['max_steer_angle'])
    
    # RAISE CHASSIS to avoid collision with inline wheels
    body_z = wr + bh + 0.02  # extra clearance
    wheel_z_offset = -(bh/2 + 0.02)
    front_x, rear_x = wb/2, -wb/2
    
    # Steering hub mass
    hub_mass = 0.1
    
    xml = f'''
        <!-- Bicycle Model -->
        <body name="robot" pos="0 0 {body_z}">
            <joint name="root" type="free"/>
            <geom name="chassis" type="box" size="{bl/2} {bw/2} {bh/2}" 
                  rgba="{get_rgba('green')}" mass="{bm}"/>
            
            <!-- Front Wheel (steerable) -->
            <body name="steer_front" pos="{front_x} 0 {wheel_z_offset}">
                <inertial pos="0 0 0" mass="{hub_mass}" diaginertia="0.001 0.001 0.001"/>
                <joint name="steer_joint" type="hinge" axis="0 0 1" 
                       range="{-max_steer:.3f} {max_steer:.3f}" damping="0.5"/>
                <body name="wheel_front">
                    <joint name="joint_front" type="hinge" axis="0 1 0" damping="0.1" frictionloss="0.01"/>
                    <geom type="cylinder" size="{wr} {ww/2}" euler="90 0 0" 
                          rgba="{get_rgba('orange')}" mass="{wm}" friction="1.0 0.005 0.0001"/>
                </body>
            </body>
            
            <!-- Rear Wheel (drive) -->
            <body name="wheel_rear" pos="{rear_x} 0 {wheel_z_offset}">
                <joint name="joint_rear" type="hinge" axis="0 1 0" damping="0.1" frictionloss="0.01"/>
                <geom type="cylinder" size="{wr} {ww/2}" euler="90 0 0" 
                      rgba="{get_rgba('orange')}" mass="{wm}" friction="1.0 0.005 0.0001"/>
            </body>
            
            <geom name="arrow" type="box" pos="{bl/3} 0 {bh/2+0.01}" 
                  size="0.03 0.01 0.005" rgba="0 1 0 1"/>
        </body>
    '''
    return xml


def generate_actuators(robot_type):
    """Generate actuators based on robot type."""
    if robot_type == 'differential_2wd':
        # Motor actuators: gear converts ctrl to torque
        # For a 2kg robot at 0.22m wheel base, we need ~0.5Nm per wheel
        return '''
    <actuator>
        <motor name="motor_left" joint="joint_left" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <motor name="motor_right" joint="joint_right" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
        '''
    elif robot_type == 'differential_4wd':
        return '''
    <actuator>
        <motor name="motor_fl" joint="joint_fl" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <motor name="motor_fr" joint="joint_fr" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <motor name="motor_rl" joint="joint_rl" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <motor name="motor_rr" joint="joint_rr" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
        '''
    elif robot_type == 'mecanum':
        return '''
    <actuator>
        <motor name="motor_fl" joint="joint_fl" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <motor name="motor_fr" joint="joint_fr" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <motor name="motor_rl" joint="joint_rl" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <motor name="motor_rr" joint="joint_rr" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
        '''
    elif robot_type == 'omni_3wheel':
        return '''
    <actuator>
        <motor name="motor_0" joint="joint_0" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <motor name="motor_1" joint="joint_1" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <motor name="motor_2" joint="joint_2" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
        '''
    elif robot_type == 'ackermann':
        return '''
    <actuator>
        <motor name="drive_rl" joint="joint_rl" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <motor name="drive_rr" joint="joint_rr" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <position name="steer_fl" joint="steer_joint_fl" kp="50" ctrllimited="true" ctrlrange="-0.6 0.6"/>
        <position name="steer_fr" joint="steer_joint_fr" kp="50" ctrllimited="true" ctrlrange="-0.6 0.6"/>
    </actuator>
        '''
    elif robot_type == 'bicycle':
        return '''
    <actuator>
        <position name="steer_front" joint="steer_joint" kp="50" ctrllimited="true" ctrlrange="-0.6 0.6"/>
        <motor name="drive_rear" joint="joint_rear" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
        '''
    return ''


def generate_obstacles(obstacles):
    """Generate obstacle geometries."""
    xml = ''
    for i, obs in enumerate(obstacles):
        if isinstance(obs, dict):
            otype = obs.get('type', 'box')
            pos = obs['pos']
            size = obs['size']
            color = obs.get('color', 'gray')
            
            if otype == 'box':
                xml += f'        <geom name="obs_{i}" type="box" pos="{pos[0]} {pos[1]} {size[2]/2}" size="{size[0]/2} {size[1]/2} {size[2]/2}" rgba="{get_rgba(color)}"/>\n'
            elif otype == 'cylinder':
                xml += f'        <geom name="obs_{i}" type="cylinder" pos="{pos[0]} {pos[1]} {size[1]/2}" size="{size[0]} {size[1]/2}" rgba="{get_rgba(color)}"/>\n'
            elif otype == 'sphere':
                xml += f'        <geom name="obs_{i}" type="sphere" pos="{pos[0]} {pos[1]} {size[0]}" size="{size[0]}" rgba="{get_rgba(color)}"/>\n'
    return xml


def generate_mjcf(robot_type, robot_params=None, obstacles=None, bounds=None, filename="mobile_robot_scene.xml"):
    """
    Generate complete MJCF XML file for mobile robot simulation.
    
    Args:
        robot_type: One of 'differential_2wd', 'differential_4wd', 'mecanum', 
                    'omni_3wheel', 'ackermann', 'bicycle'
        robot_params: Dict of robot parameters (uses defaults if None)
        obstacles: List of obstacle dicts [{'type': 'box', 'pos': [x,y], 'size': [l,w,h]}]
        bounds: [x_min, x_max, y_min, y_max] for floor size
        filename: Output XML filename
    
    Returns:
        str: Path to generated XML file
    """
    # Use defaults if not provided
    if robot_params is None:
        robot_params = ROBOT_CONFIGS.get(robot_type, ROBOT_CONFIGS['differential_2wd']).copy()
    else:
        # Merge with defaults
        defaults = ROBOT_CONFIGS.get(robot_type, {}).copy()
        defaults.update(robot_params)
        robot_params = defaults
    
    if obstacles is None:
        obstacles = []
    
    if bounds is None:
        bounds = [-10, 10, -10, 10]
    
    floor_size = max(abs(bounds[1] - bounds[0]), abs(bounds[3] - bounds[2])) / 2 + 2
    
    # Generate robot body
    robot_generators = {
        'differential_2wd': generate_differential_2wd,
        'differential_4wd': generate_differential_4wd,
        'mecanum': generate_mecanum,
        'omni_3wheel': generate_omni_3wheel,
        'ackermann': generate_ackermann,
        'bicycle': generate_bicycle
    }
    
    robot_xml = robot_generators.get(robot_type, generate_differential_2wd)(robot_params)
    actuator_xml = generate_actuators(robot_type)
    obstacles_xml = generate_obstacles(obstacles)
    
    # Complete MJCF with improved simulation settings
    xml = f'''<?xml version="1.0" ?>
<mujoco model="mobile_robot_{robot_type}">
    <!-- Simulation: timestep=0.005 for 200Hz physics (faster, less lag) -->
    <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler" solver="Newton" iterations="20"/>
    
    <!-- Default contact and friction settings -->
    <default>
        <geom friction="1.0 0.005 0.0001" condim="3" contype="1" conaffinity="1"/>
        <joint damping="0.05" frictionloss="0.001"/>
    </default>
    
    <visual>
        <map force="0.1" zfar="50"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global offwidth="1280" offheight="720"/>
    </visual>
    
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.1 0.1 0.2" width="512" height="3072"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="512" height="512"/>
        <material name="grid_mat" texture="grid" texrepeat="10 10" reflectance="0.1"/>
    </asset>
    
    <worldbody>
        <light directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" pos="0 0 10" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="{floor_size} {floor_size} 0.1" material="grid_mat" friction="1.0 0.005 0.0001"/>
        
        <!-- Start marker -->
        <geom name="start_marker" type="cylinder" pos="0 0 0.001" size="0.15 0.002" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
        
        <!-- Goal marker (will be updated by simulation) -->
        <geom name="goal_marker" type="cylinder" pos="5 5 0.001" size="0.2 0.002" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
        
        <!-- Obstacles -->
{obstacles_xml}
        
        <!-- Robot -->
{robot_xml}
    </worldbody>
    
{actuator_xml}
    
    <sensor>
        <framepos name="robot_pos" objtype="body" objname="robot"/>
        <framequat name="robot_quat" objtype="body" objname="robot"/>
        <framelinvel name="robot_vel" objtype="body" objname="robot"/>
        <frameangvel name="robot_angvel" objtype="body" objname="robot"/>
    </sensor>
</mujoco>
'''
    
    with open(filename, 'w') as f:
        f.write(xml)
    
    print(f"Generated MJCF: {filename} (Robot: {robot_type})")
    return filename


if __name__ == "__main__":
    # Test: Generate each robot type
    for rtype in ROBOT_CONFIGS.keys():
        generate_mjcf(rtype, filename=f"test_{rtype}.xml")
    print("All robot models generated successfully!")
