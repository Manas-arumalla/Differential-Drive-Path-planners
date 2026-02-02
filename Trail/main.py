"""
Animated runner for hierarchical Hybrid A* + DWA simulation (no ROS).
- Smooth, continuous animation (faster refresh, artists update)
- Enhanced dynamic obstacles: patrol, orbit, and random-walk behaviors
  + simple reactive avoidance (slow repulsion) when near robot or each other
All original planning/execution behavior preserved.
If your Simulator / DynamicObstacle classes don't expose attributes
(x, y, vx, vy, r) or a modifiable list, adapt the "enhanced obstacles" section.
"""
import heapq
import time
from math import hypot, pi, sin, cos, atan2
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from simulator import Simulator, DynamicObstacle
from vehicle import KinematicBicycle
from hybrid_astar import HybridAStar
from dwa_local_planner import DWAPlanner
from utils import euclidean, line_collision, to_grid

# ------------------ Map & helpers (unchanged from simplified version) ------------------

def create_test_map(w=80, h=60, res=0.5, start=None, goal=None, clearance_m=1.2):
    grid = np.zeros((h, w), dtype=bool)
    grid[6:9, 6:42] = True
    grid[22:24, 12:72] = True
    grid[36:39, 12:62] = True
    grid[11:19, 46:49] = True
    grid[41:56, 31:34] = True

    def clear_circle(center):
        if center is None:
            return
        cx, cy = center
        gx = int(round(cx / res)); gy = int(round(cy / res))
        r_cells = int(np.ceil(clearance_m / res))
        for yy in range(max(0, gy - r_cells), min(grid.shape[0], gy + r_cells + 1)):
            for xx in range(max(0, gx - r_cells), min(grid.shape[1], gx + r_cells + 1)):
                xw = xx * res; yw = yy * res
                if hypot(xw - cx, yw - cy) <= clearance_m:
                    grid[yy, xx] = False
    clear_circle(start); clear_circle(goal)
    return grid

def inflate_grid(occ, inflation_cells):
    if inflation_cells <= 0:
        return occ.copy()
    h, w = occ.shape
    out = occ.copy()
    for gy, gx in np.argwhere(occ):
        y0 = max(0, gy - inflation_cells); y1 = min(h, gy + inflation_cells + 1)
        x0 = max(0, gx - inflation_cells); x1 = min(w, gx + inflation_cells + 1)
        out[y0:y1, x0:x1] = True
    return out

def grid_neighbors(gx, gy, w, h):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < w and 0 <= ny < h:
                cost = 1.0 if dx == 0 or dy == 0 else 1.41421356
                yield nx, ny, cost

def grid_a_star(occ, start_w, goal_w, res):
    h, w = occ.shape
    sx, sy = start_w; gx, gy = goal_w
    start = (int(round(sx / res)), int(round(sy / res)))
    goal = (int(round(gx / res)), int(round(gy / res)))
    def h_cost(a, b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
    open_heap = []
    heapq.heappush(open_heap, (h_cost(start, goal), 0.0, start, None))
    came_from = {}
    gscore = {start: 0.0}
    closed = set()
    while open_heap:
        f, g, cur, parent = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        came_from[cur] = parent
        if cur == goal:
            path = []
            node = cur
            while node is not None:
                path.append((node[0], node[1])); node = came_from.get(node, None)
            path.reverse()
            return [(c[0] * res, c[1] * res) for c in path]
        for nx, ny, cost in grid_neighbors(cur[0], cur[1], w, h):
            if occ[ny, nx]:
                continue
            tentative = g + cost
            nbr = (nx, ny)
            if tentative < gscore.get(nbr, 1e9):
                gscore[nbr] = tentative
                heapq.heappush(open_heap, (tentative + h_cost(nbr, goal), tentative, nbr, cur))
    return None

def downsample_waypoints(path_pts, downsample=6):
    if path_pts is None or len(path_pts) == 0:
        return []
    if downsample <= 1:
        return path_pts[:]
    return path_pts[::downsample] + [path_pts[-1]]

def smooth_path(path):
    pts = np.array([[p[0], p[1]] for p in path])
    if pts.shape[0] < 4:
        return [(float(x), float(y), 0.0) for x, y in pts]
    t = np.linspace(0, 1, pts.shape[0])
    csx = CubicSpline(t, pts[:, 0]); csy = CubicSpline(t, pts[:, 1])
    ts = np.linspace(0, 1, int((pts.shape[0]-1) * 5))
    return [(float(csx(tt)), float(csy(tt)), 0.0) for tt in ts]

def adjust_goal_if_invalid(goal_xy, occ_grid, res, max_shift=1.0, step=0.2):
    gx, gy = to_grid(goal_xy[0], goal_xy[1], res)
    h, w = occ_grid.shape
    if 0 <= gx < w and 0 <= gy < h and not occ_grid[gy, gx]:
        return goal_xy
    for r in np.arange(step, max_shift + 1e-9, step):
        for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            nx = goal_xy[0] + r * np.cos(ang)
            ny = goal_xy[1] + r * np.sin(ang)
            gx2, gy2 = to_grid(nx, ny, res)
            if 0 <= gx2 < w and 0 <= gy2 < h and not occ_grid[gy2, gx2]:
                return (nx, ny)
    return None

def try_hybrid(planner_obj, src_pose, tgt_xy, max_iter_local=60000, tol_local=0.6):
    return planner_obj.plan(src_pose, tgt_xy, max_iter=max_iter_local, goal_tolerance=tol_local, verbose=True)

# ------------------ Enhanced dynamic obstacle helpers ------------------

class EnhancedObstacle:
    """
    Local wrapper that augments DynamicObstacle with higher-level motion patterns.
    We keep a reference to the original DynamicObstacle instance (so sim uses it for collision checks),
    and we modify its vx/vy attributes each tick.
    """
    def __init__(self, dyn_obs, pattern='random', params=None):
        self.obs = dyn_obs
        self.pattern = pattern
        self.params = params or {}
        # pattern-specific state
        if self.pattern == 'patrol':
            # params: waypoints list, speed
            self.waypoints = self.params.get('waypoints', [(dyn_obs.x, dyn_obs.y)])
            self.speed = self.params.get('speed', 0.25)
            self.idx = 0
        elif self.pattern == 'orbit':
            self.center = self.params.get('center', (dyn_obs.x, dyn_obs.y))
            self.radius = self.params.get('radius', 2.0)
            self.omega = self.params.get('omega', 0.6)  # rad/s
            self.theta = random.random() * 2 * pi
        else:  # 'random'
            self.speed = self.params.get('speed', 0.15)
            # random direction
            self.theta = random.random() * 2 * pi
            self.change_prob = self.params.get('change_prob', 0.02)

    def step(self, dt, robot_pos, neighbors):
        """Update obs.obs.vx & vy according to pattern + simple repulsion from robot/neighbors."""
        ox, oy = self.obs.x, self.obs.y

        if self.pattern == 'patrol':
            tx, ty = self.waypoints[self.idx]
            dx, dy = tx - ox, ty - oy
            dist = hypot(dx, dy)
            if dist < 0.2:
                self.idx = (self.idx + 1) % len(self.waypoints)
                tx, ty = self.waypoints[self.idx]
                dx, dy = tx - ox, ty - oy
                dist = hypot(dx, dy)
            if dist > 1e-6:
                vx = (dx / dist) * self.speed
                vy = (dy / dist) * self.speed
            else:
                vx = vy = 0.0

        elif self.pattern == 'orbit':
            self.theta += self.omega * dt
            tx = self.center[0] + self.radius * cos(self.theta)
            ty = self.center[1] + self.radius * sin(self.theta)
            # velocity tangent to circle
            vx = -self.radius * self.omega * sin(self.theta)
            vy =  self.radius * self.omega * cos(self.theta)

        else:  # random walk with occasional direction change
            if random.random() < self.change_prob:
                self.theta = random.random() * 2 * pi
            vx = cos(self.theta) * self.speed
            vy = sin(self.theta) * self.speed

        # simple repulsion from robot
        rx, ry = robot_pos
        rdx, rdy = ox - rx, oy - ry
        rdist = hypot(rdx, rdy)
        if rdist < 2.0 and rdist > 1e-6:
            # repel proportionally when closer than 2m
            scale = (2.0 - rdist) / 2.0 * 0.8  # strength
            vx += (rdx / rdist) * scale
            vy += (rdy / rdist) * scale

        # mild pairwise repulsion from neighbors (to avoid obstacle clustering)
        for n in neighbors:
            nx, ny = n.obs.x, n.obs.y
            ddx, ddy = ox - nx, oy - ny
            d2 = hypot(ddx, ddy)
            if d2 < 1.0 and d2 > 1e-6:
                strength = (1.0 - d2) * 0.35
                vx += (ddx / d2) * strength
                vy += (ddy / d2) * strength

        # optionally clamp speed to a safe max for obstacles
        max_speed = 0.6
        speed = hypot(vx, vy)
        if speed > max_speed:
            vx *= max_speed / speed; vy *= max_speed / speed

        # write back to underlying dynamic obstacle attributes (assumes these exist)
        try:
            self.obs.vx = vx
            self.obs.vy = vy
        except Exception:
            # fallback attribute names commonly: ux, uy or simply set methods - user may tweak here
            setattr(self.obs, 'vx', vx); setattr(self.obs, 'vy', vy)

# ------------------ Main (with animation & enhanced obstacles) ------------------

def main():
    res = 0.5
    start = (2.0, 2.0, 0.0)
    goal = (34.0, 22.0)
    occ = create_test_map(res=res, start=(start[0], start[1]), goal=(goal[0], goal[1]), clearance_m=1.2)
    inflation_m = 1.0
    inflation_cells = int(np.ceil(inflation_m / res))
    occ_inflated = inflate_grid(occ, inflation_cells)

    sim = Simulator(occ, resolution=res, dt=0.1)
    veh = KinematicBicycle(x=start[0], y=start[1], theta=start[2], L=0.7, dt=0.1)
    sim.set_vehicle(veh)

    # ----- create dynamic obstacles and wrap them with EnhancedObstacle -----
    # Keep references so we can update their vx/vy each iteration.
    base_obs = []
    enh_obs = []

    # obstacle A: patrol between two points
    oA = DynamicObstacle(12.0, 2.0, 0.2, 0.0, r=0.4)
    sim.add_dynamic_obstacle(oA); base_obs.append(oA)
    enh_obs.append(EnhancedObstacle(oA, pattern='patrol', params={'waypoints': [(12.0,2.0),(16.0,6.0)], 'speed':0.28}))

    # obstacle B: orbit around a center
    oB = DynamicObstacle(20.0, 10.0, -0.15, 0.05, r=0.4)
    sim.add_dynamic_obstacle(oB); base_obs.append(oB)
    enh_obs.append(EnhancedObstacle(oB, pattern='orbit', params={'center':(20.0,10.0),'radius':2.0,'omega':0.8}))

    # obstacle C: random walker
    oC = DynamicObstacle(30.0, 18.0, 0.0, -0.25, r=0.4)
    sim.add_dynamic_obstacle(oC); base_obs.append(oC)
    enh_obs.append(EnhancedObstacle(oC, pattern='random', params={'speed':0.2, 'change_prob':0.03}))

    # additional small random obstacles to create a busier scene
    for i in range(3):
        x = 8.0 + i*6.0
        y = 30.0 + (i%2)*3.0
        o = DynamicObstacle(x, y, 0.0, 0.0, r=0.35)
        sim.add_dynamic_obstacle(o); base_obs.append(o)
        # give each a different random walk profile
        enh_obs.append(EnhancedObstacle(o, pattern='random', params={'speed':0.12 + 0.05*i, 'change_prob':0.04}))

    # ----- hierarchical planning (same as before) -----
    print("Running coarse grid A* for long-range routing...")
    grid_path = grid_a_star(occ_inflated, (start[0], start[1]), (goal[0], goal[1]), res)
    if grid_path is None:
        print("Coarse A* failed. Exiting."); return
    waypoints = downsample_waypoints(grid_path, downsample=6)
    planner = HybridAStar(occ, resolution=res, vehicle_length=0.7, step_size=0.18, theta_res_deg=10, radius=0.18)

    plan_segments = []
    cur_pose = start
    success = True
    for i, wp in enumerate(waypoints):
        wx, wy = wp
        if euclidean((cur_pose[0], cur_pose[1]), (wx, wy)) < 0.6:
            continue
        safe_wp = adjust_goal_if_invalid((wx, wy), occ, res, max_shift=1.0, step=0.2)
        if safe_wp is None:
            print(f"Waypoint {wp} invalid & cannot be adjusted. Skipping."); continue
        wx, wy = safe_wp
        print(f"Planning segment {i+1}/{len(waypoints)} to waypoint {wx:.2f},{wy:.2f} ...")
        seg = try_hybrid(planner, cur_pose, (wx, wy), max_iter_local=60000, tol_local=0.6)
        if seg is None:
            planner_tmp = HybridAStar(occ, resolution=res, vehicle_length=0.7, step_size=0.14, theta_res_deg=10, radius=0.16)
            seg = try_hybrid(planner_tmp, cur_pose, (wx, wy), max_iter_local=90000, tol_local=0.8)
        if seg is None:
            sx, sy = cur_pose[0], cur_pose[1]
            mid1 = ((2 * sx + wx) / 3.0, (2 * sy + wy) / 3.0)
            mid2 = ((sx + 2 * wx) / 3.0, (sy + 2 * wy) / 3.0)
            sub_goals = [mid1, mid2, (wx, wy)]
            segmented_success = True
            for sg in sub_goals:
                safe_sg = adjust_goal_if_invalid(sg, occ, res, max_shift=0.8, step=0.15)
                if safe_sg is None:
                    segmented_success = False; break
                seg_try = try_hybrid(planner, cur_pose, safe_sg, max_iter_local=50000, tol_local=0.5)
                if seg_try is None:
                    seg_try = try_hybrid(planner_tmp, cur_pose, safe_sg, max_iter_local=80000, tol_local=0.6)
                if seg_try is None:
                    segmented_success = False; break
                if len(plan_segments) and plan_segments[-1] == seg_try[0]:
                    plan_segments.extend(seg_try[1:])
                else:
                    plan_segments.extend(seg_try)
                cur = plan_segments[-1]; cur_pose = (cur[0], cur[1], cur[2])
            if segmented_success:
                continue
            else:
                seg = None
        if seg is None:
            if not line_collision((cur_pose[0], cur_pose[1]), (wx, wy), occ, res, radius=0.16):
                seg = [(cur_pose[0], cur_pose[1], cur_pose[2]), (wx, wy, 0.0)]
            else:
                success = False; break
        if len(plan_segments) and plan_segments[-1] == seg[0]:
            plan_segments.extend(seg[1:])
        else:
            plan_segments.extend(seg)
        last = plan_segments[-1]; cur_pose = (last[0], last[1], last[2])

    if not success or len(plan_segments) == 0:
        print("Hierarchical planning failed. Exiting."); return
    plan = plan_segments
    print("Hierarchical planning succeeded. Total hybrid path length:", len(plan))
    smoothed = smooth_path(plan)
    if not smoothed:
        print("Smoothed path empty. Exiting."); return

    dwa = DWAPlanner({'dt': 0.1, 'predict_time': 1.5, 'robot_radius': 0.35})

    # ------------------ Animated plotting (artists) ------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.ion(); plt.show()
    sim.plot(ax, start, goal, path=plan, smoothed=smoothed)  # initial draw (map + static obstacles)
    # plot robot pose as a marker (we'll update coords)
    robot_marker, = ax.plot([], [], 'ro', markersize=6, label='robot')
    traj_line, = ax.plot([], [], '-b', linewidth=1.5, alpha=0.8)
    # obstacles scatter (we'll update positions & sizes)
    obs_scatter = ax.scatter([], [], s=[], c='orange', alpha=0.9, edgecolors='k', zorder=5)

    MAX_STEPS = 5000
    WALLCLOCK_TIMEOUT = 180.0
    stuck_window = 25
    stuck_distance_threshold = 0.05
    zero_cmd_window = 12

    dist_history = []
    zero_cmd_count = 0
    traj = []
    start_time = time.time()

    # helper to get current obstacle states
    def obstacles_as_list():
        # prefer sim.get_obstacles if it returns objects with x,y,r
        obs = sim.get_obstacles()
        # normalize to list of dicts
        out = []
        for o in obs:
            # try attribute access first
            try:
                ox, oy, rr = o.x, o.y, o.r
            except Exception:
                # if sim returns tuples, assume (x,y,r)
                try:
                    ox, oy, rr = o[0], o[1], o[2]
                except Exception:
                    continue
            out.append({'x': ox, 'y': oy, 'r': rr})
        return out

    try:
        for step in range(MAX_STEPS):
            if time.time() - start_time > WALLCLOCK_TIMEOUT:
                print("Wall-clock timeout. Exiting."); break

            # update enhanced obstacles BEFORE DWA so planner sees their new vx/vy if sim uses them
            robot_pos = (veh.x, veh.y)
            for e in enh_obs:
                # neighbors = all other enh_obs except itself
                neighbors = [n for n in enh_obs if n is not e]
                e.step(sim.dt, robot_pos, neighbors)

            # step simulation and get obstacles for planner
            # sim.step only moves vehicle with control; here we haven't commanded yet,
            # but we want obstacles to integrate their velocities into sim's state.
            # If Simulator updates dynamic obstacles on sim.step, call a zero-step with no motion:
            # We'll not call sim.step here because later we call sim.step after commanding the robot.
            # Instead we'll rely on sim to update obstacle positions when sim.step is called.
            # (if your simulator requires explicit obstacle update, modify it accordingly)
            obstacles = obstacles_as_list()

            px = veh.x; py = veh.y

            # choose lookahead target on smoothed path
            dists = [euclidean((px, py), (p[0], p[1])) for p in smoothed]
            idx = int(np.argmin(dists))
            if idx >= len(smoothed) - 2:
                goal_point = (smoothed[-1][0], smoothed[-1][1])
            else:
                lookahead_idx = min(idx + 8, len(smoothed) - 1)
                goal_point = (smoothed[lookahead_idx][0], smoothed[lookahead_idx][1])

            # DWA plan
            # The DWA planner expects obstacles list; convert to (x,y,r) tuples if needed.
            obs_for_dwa = [(o['x'], o['y'], o['r']) for o in obstacles]
            v_cmd, om_cmd = dwa.plan((px, py, veh.theta), veh.v, veh.omega, goal_point, obs_for_dwa)

            # zero-command stall detection and nudge
            if abs(v_cmd) < 1e-3 and abs(om_cmd) < 1e-3:
                zero_cmd_count += 1
            else:
                zero_cmd_count = 0
            if zero_cmd_count >= zero_cmd_window:
                v_cmd = 0.2; om_cmd = 0.2; zero_cmd_count = 0

            if abs(v_cmd) < 1e-5:
                delta = 0.0
            else:
                delta = np.arctan(om_cmd * veh.L / max(1e-4, v_cmd))

            # Step simulation: this will (should) update vehicle and dynamic obstacles inside sim
            sim.step(v_cmd, delta)

            # append route & update visuals
            traj.append((veh.x, veh.y))
            cur_dist = euclidean((veh.x, veh.y), goal)
            dist_history.append(cur_dist)

            # stuck detection and replan
            if len(dist_history) > stuck_window:
                improvement = (dist_history[-stuck_window] - dist_history[-1]) / stuck_window
                if improvement < stuck_distance_threshold:
                    print(f"[WARN] Low progress ({improvement:.4f} m/step). Triggering replan...")
                    new_start = (veh.x, veh.y, veh.theta)
                    new_plan = planner.plan(new_start, goal, max_iter=100000, goal_tolerance=0.8, verbose=True)
                    if new_plan:
                        plan = new_plan; smoothed = smooth_path(plan); dist_history = []
                        print("Replan succeeded")
                    else:
                        print("Replan failed; performing small escape.")
                        sim.step(0.2, 0.0)

            # update obstacle positions for visualization (read from sim)
            obstacles = obstacles_as_list()
            oxs = [o['x'] for o in obstacles]
            oys = [o['y'] for o in obstacles]
            sizes = [(o['r']*200.0)**2 for o in obstacles]  # scale radii -> scatter size

            # update scatter artist
            if len(oxs) > 0:
                obs_scatter.set_offsets(np.column_stack([oxs, oys]))
                obs_scatter.set_sizes(sizes)
            else:
                obs_scatter.set_offsets(np.empty((0, 2)))
                obs_scatter.set_sizes([])

            # update robot marker and trajectory
            robot_marker.set_data([veh.x], [veh.y])
            xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
            traj_line.set_data(xs, ys)

            # redraw (fast)
            ax.set_title(f"Step {step} | Dist to goal: {cur_dist:.2f} m")
            fig.canvas.draw_idle()
            plt.pause(0.02)  # ~50 FPS drawing; simulation dt is 0.1 so visual smoothing looks good

            if cur_dist < 0.8:
                print("Reached goal at step", step, "distance:", cur_dist)
                break

        else:
            print("MAX_STEPS reached without reaching goal.")

    except KeyboardInterrupt:
        print("User interrupted.")
    finally:
        plt.ioff()
        print("Simulation ended. Final pose:", (veh.x, veh.y, veh.theta))
        plt.show()

if __name__ == "__main__":
    main()
