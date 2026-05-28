"""
Mobile Robot GUI - Advanced Control Center
PySide6 GUI for configuring and running mobile robot simulations.
Features: 6 robot types, 7 planners, 4 nav controllers, GA auto-tuning
"""
import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QFormLayout, QLineEdit, QComboBox,
    QPushButton, QListWidget, QListWidgetItem, QLabel, QSpinBox,
    QDoubleSpinBox, QCheckBox, QScrollArea, QFrame, QMessageBox,
    QSplitter, QTextEdit, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPalette, QColor, QPainter, QPen, QBrush

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from navstack.sim import model_generator as model_gen
from navstack.controllers import navigation as nav_ctrl

# Import planners
from navstack.planners.astar_planner import AStarPlanner
from navstack.planners.rrt_planner import RRTPlanner
from navstack.planners.rrt_star_planner import RRTStarPlanner
from navstack.planners.pso_planner import PSOPlanner
from navstack.planners.apf_planner import APFPlanner
from navstack.planners.dijkstra_planner import DijkstraPlanner
from navstack.planners.prm_planner import PRMPlanner
from navstack.environment import Environment


# ========================= DARK THEME STYLING =========================
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #eaeaea;
}
QTabWidget::pane {
    border: 1px solid #3a3a5a;
    background-color: #16213e;
}
QTabBar::tab {
    background-color: #1a1a2e;
    color: #aaa;
    padding: 10px 20px;
    border: 1px solid #3a3a5a;
}
QTabBar::tab:selected {
    background-color: #0f3460;
    color: #00e5ff;
    border-bottom: 2px solid #00e5ff;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #3a3a5a;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 10px;
}
QGroupBox::title {
    color: #00e5ff;
    subcontrol-origin: margin;
    left: 10px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #16213e;
    border: 1px solid #3a3a5a;
    border-radius: 4px;
    padding: 6px;
    color: #eaeaea;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #00e5ff;
}
QPushButton {
    background-color: #0f3460;
    color: #eaeaea;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #00e5ff;
    color: #1a1a2e;
}
QPushButton:pressed {
    background-color: #00b8d4;
}
QPushButton#runBtn {
    background-color: #00c853;
    font-size: 14px;
}
QPushButton#runBtn:hover {
    background-color: #00e676;
}
QListWidget {
    background-color: #16213e;
    border: 1px solid #3a3a5a;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #0f3460;
    color: #00e5ff;
}
QScrollArea {
    border: none;
}
QLabel#statusLabel {
    color: #00e5ff;
    font-size: 12px;
}
QProgressBar {
    border: 1px solid #3a3a5a;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #00e5ff;
}
"""


# ========================= MAP PREVIEW WIDGET =========================
class MapPreviewWidget(QWidget):
    """2D visualization of environment, obstacles, and path."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.obstacles = []
        self.path = []
        self.start = (0, 0)
        self.goal = (5, 5)
        self.bounds = [-10, 10, -10, 10]  # x_min, x_max, y_min, y_max
        self.scale = 20  # pixels per meter
    
    def set_data(self, obstacles, path, start, goal, bounds=None):
        """Update visualization data."""
        self.obstacles = obstacles or []
        self.path = path or []
        self.start = start
        self.goal = goal
        if bounds:
            self.bounds = bounds
        self.update()
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen pixels."""
        w = self.width()
        h = self.height()
        
        # Calculate scale to fit bounds
        world_w = self.bounds[1] - self.bounds[0]
        world_h = self.bounds[3] - self.bounds[2]
        
        scale_x = (w - 40) / world_w if world_w > 0 else 1
        scale_y = (h - 40) / world_h if world_h > 0 else 1
        scale = min(scale_x, scale_y)
        
        # Center offset
        cx = w / 2
        cy = h / 2
        world_cx = (self.bounds[0] + self.bounds[1]) / 2
        world_cy = (self.bounds[2] + self.bounds[3]) / 2
        
        sx = cx + (x - world_cx) * scale
        sy = cy - (y - world_cy) * scale  # Flip y
        
        return int(sx), int(sy), scale
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(22, 33, 62))
        
        # Draw grid
        painter.setPen(QPen(QColor(60, 60, 90), 1, Qt.DotLine))
        _, _, scale = self.world_to_screen(0, 0)
        grid_step = 1  # 1 meter grid
        for x in np.arange(self.bounds[0], self.bounds[1] + 1, grid_step):
            sx, _, _ = self.world_to_screen(x, 0)
            painter.drawLine(sx, 0, sx, self.height())
        for y in np.arange(self.bounds[2], self.bounds[3] + 1, grid_step):
            _, sy, _ = self.world_to_screen(0, y)
            painter.drawLine(0, sy, self.width(), sy)
        
        # Draw obstacles
        for obs in self.obstacles:
            if isinstance(obs, dict):
                ox, oy = obs['pos'][0], obs['pos'][1]
                sx, sy, sc = self.world_to_screen(ox, oy)
                
                if obs.get('type') == 'cylinder':
                    r = obs['size'][0] * sc
                    painter.setBrush(QBrush(QColor(200, 80, 80)))
                    painter.setPen(QPen(QColor(255, 100, 100), 2))
                    painter.drawEllipse(int(sx - r), int(sy - r), int(2*r), int(2*r))
                else:
                    w = obs['size'][0] * sc
                    h = obs['size'][1] * sc if len(obs['size']) > 1 else w
                    painter.setBrush(QBrush(QColor(80, 80, 200)))
                    painter.setPen(QPen(QColor(100, 100, 255), 2))
                    painter.drawRect(int(sx - w/2), int(sy - h/2), int(w), int(h))
        
        # Draw path
        if len(self.path) > 1:
            painter.setPen(QPen(QColor(0, 229, 255), 2))
            for i in range(len(self.path) - 1):
                x1, y1, _ = self.world_to_screen(self.path[i][0], self.path[i][1])
                x2, y2, _ = self.world_to_screen(self.path[i+1][0], self.path[i+1][1])
                painter.drawLine(x1, y1, x2, y2)
        
        # Draw start
        sx, sy, sc = self.world_to_screen(self.start[0], self.start[1])
        painter.setBrush(QBrush(QColor(0, 200, 0)))
        painter.setPen(QPen(QColor(100, 255, 100), 2))
        painter.drawEllipse(sx - 8, sy - 8, 16, 16)
        
        # Draw goal
        gx, gy, _ = self.world_to_screen(self.goal[0], self.goal[1])
        painter.setBrush(QBrush(QColor(255, 50, 50)))
        painter.setPen(QPen(QColor(255, 150, 150), 2))
        points = [
            (gx, gy - 12), (gx + 10, gy + 8), (gx - 10, gy + 8)
        ]
        from PySide6.QtGui import QPolygon
        from PySide6.QtCore import QPoint
        polygon = QPolygon([QPoint(int(p[0]), int(p[1])) for p in points])
        painter.drawPolygon(polygon)
        
        # Info text
        painter.setPen(QColor(150, 150, 150))
        painter.drawText(10, 20, f"Obstacles: {len(self.obstacles)}")
        painter.drawText(10, 35, f"Path: {len(self.path)} waypoints")


# ========================= SIMULATION THREAD =========================
class SimulationThread(QThread):
    """Background thread for running simulation."""
    finished = Signal(dict)
    progress = Signal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def run(self):
        try:
            from navstack.sim import mujoco_sim as sim
            
            self.progress.emit("Generating robot model...")
            result = sim.run_simulation(
                robot_type=self.config['robot_type'],
                robot_params=self.config['robot_params'],
                obstacles=self.config['obstacles'],
                start=self.config['start'],
                goal=self.config['goal'],
                path=self.config['path'],
                nav_controller=self.config['nav_controller'],
                nav_params=self.config['nav_params'],
                duration=self.config['duration'],
                show_plots=self.config.get('show_plots', True),
                goal_threshold=self.config.get('goal_threshold', 0.3)
            )
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit({'error': str(e)})


# ========================= MAIN GUI =========================
class MobileRobotGUI(QMainWindow):
    """Main GUI Application for Mobile Robot Simulation."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mobile Robot Control Center")
        self.setGeometry(100, 50, 1200, 900)
        
        # Data
        self.obstacles = []
        self.path = []
        self.environment = None
        self.sim_thread = None
        
        # Setup
        self.setStyleSheet(DARK_STYLE)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left: Tabs
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(500)
        
        # Create tabs
        self.tab_config = QWidget()
        self.tab_env = QWidget()
        self.tab_robot = QWidget()
        self.tab_planner = QWidget()
        self.tab_sim = QWidget()
        
        self.tabs.addTab(self.tab_config, "Configuration")
        self.tabs.addTab(self.tab_env, "Environment")
        self.tabs.addTab(self.tab_robot, "Robot Builder")
        self.tabs.addTab(self.tab_planner, "Planner")
        self.tabs.addTab(self.tab_sim, "Simulation")
        
        # Build tabs
        self.build_config_tab()
        self.build_env_tab()
        self.build_robot_tab()
        self.build_planner_tab()
        self.build_sim_tab()
        
        # Right: Map Preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        preview_label = QLabel("MAP PREVIEW")
        preview_label.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 14px;")
        right_layout.addWidget(preview_label)
        
        self.map_preview = MapPreviewWidget()
        right_layout.addWidget(self.map_preview)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        right_layout.addWidget(self.status_label)
        
        # Main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tabs)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 500])
        
        main_layout.addWidget(splitter)
        
        # Initial preview update
        self.update_preview()
    
    def build_config_tab(self):
        """Build Configuration tab."""
        layout = QVBoxLayout(self.tab_config)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        clayout = QVBoxLayout(container)
        
        # Start Position
        gb_start = QGroupBox("START POSITION")
        form_start = QFormLayout(gb_start)
        self.start_x = QDoubleSpinBox()
        self.start_x.setRange(-50, 50)
        self.start_x.setValue(0)
        self.start_y = QDoubleSpinBox()
        self.start_y.setRange(-50, 50)
        self.start_y.setValue(0)
        self.start_theta = QDoubleSpinBox()
        self.start_theta.setRange(-180, 180)
        self.start_theta.setValue(0)
        form_start.addRow("X (m):", self.start_x)
        form_start.addRow("Y (m):", self.start_y)
        form_start.addRow("Theta (deg):", self.start_theta)
        clayout.addWidget(gb_start)
        
        # Goal Position
        gb_goal = QGroupBox("GOAL POSITION")
        form_goal = QFormLayout(gb_goal)
        self.goal_x = QDoubleSpinBox()
        self.goal_x.setRange(-50, 50)
        self.goal_x.setValue(5)
        self.goal_y = QDoubleSpinBox()
        self.goal_y.setRange(-50, 50)
        self.goal_y.setValue(5)
        form_goal.addRow("X (m):", self.goal_x)
        form_goal.addRow("Y (m):", self.goal_y)
        clayout.addWidget(gb_goal)
        
        # Simulation Settings
        gb_sim = QGroupBox("SIMULATION SETTINGS")
        form_sim = QFormLayout(gb_sim)
        self.max_duration = QDoubleSpinBox()
        self.max_duration.setRange(10, 300)
        self.max_duration.setValue(60)
        self.goal_threshold = QDoubleSpinBox()
        self.goal_threshold.setRange(0.1, 2.0)
        self.goal_threshold.setValue(0.3)
        self.goal_threshold.setSingleStep(0.1)
        form_sim.addRow("Max Duration (s):", self.max_duration)
        form_sim.addRow("Goal Threshold (m):", self.goal_threshold)
        clayout.addWidget(gb_sim)
        
        # Update button
        btn_update = QPushButton("Update Preview")
        btn_update.clicked.connect(self.update_preview)
        clayout.addWidget(btn_update)
        
        clayout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
    
    def build_env_tab(self):
        """Build Environment tab."""
        layout = QVBoxLayout(self.tab_env)
        
        # Obstacle List
        gb_obs = QGroupBox("OBSTACLES")
        obs_layout = QVBoxLayout(gb_obs)
        
        self.obs_list = QListWidget()
        obs_layout.addWidget(self.obs_list)
        
        # Add obstacle form
        add_layout = QHBoxLayout()
        self.obs_type = QComboBox()
        self.obs_type.addItems(["Box", "Cylinder"])
        self.obs_x = QDoubleSpinBox()
        self.obs_x.setRange(-50, 50)
        self.obs_y = QDoubleSpinBox()
        self.obs_y.setRange(-50, 50)
        self.obs_size = QDoubleSpinBox()
        self.obs_size.setRange(0.1, 5)
        self.obs_size.setValue(0.5)
        
        add_layout.addWidget(QLabel("Type:"))
        add_layout.addWidget(self.obs_type)
        add_layout.addWidget(QLabel("X:"))
        add_layout.addWidget(self.obs_x)
        add_layout.addWidget(QLabel("Y:"))
        add_layout.addWidget(self.obs_y)
        add_layout.addWidget(QLabel("Size:"))
        add_layout.addWidget(self.obs_size)
        obs_layout.addLayout(add_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self.add_obstacle)
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self.remove_obstacle)
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self.clear_obstacles)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_remove)
        btn_layout.addWidget(btn_clear)
        obs_layout.addLayout(btn_layout)
        
        layout.addWidget(gb_obs)
        
        # Presets
        gb_preset = QGroupBox("PRESET ENVIRONMENTS")
        preset_layout = QVBoxLayout(gb_preset)
        btn_maze = QPushButton("Load Maze")
        btn_maze.clicked.connect(lambda: self.load_preset("maze"))
        btn_open = QPushButton("Load Open Field")
        btn_open.clicked.connect(lambda: self.load_preset("open"))
        btn_dense = QPushButton("Load Dense Obstacles")
        btn_dense.clicked.connect(lambda: self.load_preset("dense"))
        preset_layout.addWidget(btn_maze)
        preset_layout.addWidget(btn_open)
        preset_layout.addWidget(btn_dense)
        layout.addWidget(gb_preset)
        
        layout.addStretch()
    
    def build_robot_tab(self):
        """Build Robot Builder tab with full parameter customization."""
        layout = QVBoxLayout(self.tab_robot)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        clayout = QVBoxLayout(container)
        
        # Robot Type Selection
        gb_type = QGroupBox("ROBOT TYPE")
        type_layout = QFormLayout(gb_type)
        self.robot_type = QComboBox()
        self.robot_type.addItems([
            "differential_2wd", "differential_4wd", "mecanum",
            "omni_3wheel", "ackermann", "bicycle"
        ])
        self.robot_type.currentTextChanged.connect(self.on_robot_type_changed)
        type_layout.addRow("Model:", self.robot_type)
        clayout.addWidget(gb_type)
        
        # Body Parameters
        gb_body = QGroupBox("BODY PARAMETERS")
        body_layout = QFormLayout(gb_body)
        
        self.body_length = QDoubleSpinBox()
        self.body_length.setRange(0.1, 2.0)
        self.body_length.setValue(0.3)
        self.body_length.setSingleStep(0.05)
        
        self.body_width = QDoubleSpinBox()
        self.body_width.setRange(0.1, 2.0)
        self.body_width.setValue(0.2)
        self.body_width.setSingleStep(0.05)
        
        self.body_height = QDoubleSpinBox()
        self.body_height.setRange(0.05, 1.0)
        self.body_height.setValue(0.1)
        self.body_height.setSingleStep(0.02)
        
        self.body_mass = QDoubleSpinBox()
        self.body_mass.setRange(0.5, 50.0)
        self.body_mass.setValue(2.0)
        self.body_mass.setSingleStep(0.5)
        
        body_layout.addRow("Length (m):", self.body_length)
        body_layout.addRow("Width (m):", self.body_width)
        body_layout.addRow("Height (m):", self.body_height)
        body_layout.addRow("Mass (kg):", self.body_mass)
        clayout.addWidget(gb_body)
        
        # Wheel Parameters
        gb_wheel = QGroupBox("WHEEL PARAMETERS")
        wheel_layout = QFormLayout(gb_wheel)
        
        self.wheel_radius = QDoubleSpinBox()
        self.wheel_radius.setRange(0.02, 0.3)
        self.wheel_radius.setValue(0.05)
        self.wheel_radius.setSingleStep(0.01)
        
        self.wheel_width = QDoubleSpinBox()
        self.wheel_width.setRange(0.01, 0.1)
        self.wheel_width.setValue(0.02)
        self.wheel_width.setSingleStep(0.005)
        
        self.wheel_mass = QDoubleSpinBox()
        self.wheel_mass.setRange(0.05, 2.0)
        self.wheel_mass.setValue(0.1)
        self.wheel_mass.setSingleStep(0.05)
        
        self.wheel_separation = QDoubleSpinBox()
        self.wheel_separation.setRange(0.1, 1.0)
        self.wheel_separation.setValue(0.22)
        self.wheel_separation.setSingleStep(0.02)
        
        wheel_layout.addRow("Radius (m):", self.wheel_radius)
        wheel_layout.addRow("Width (m):", self.wheel_width)
        wheel_layout.addRow("Mass (kg):", self.wheel_mass)
        wheel_layout.addRow("Separation (m):", self.wheel_separation)
        clayout.addWidget(gb_wheel)
        
        # Type-specific parameters
        gb_special = QGroupBox("TYPE-SPECIFIC PARAMETERS")
        special_layout = QFormLayout(gb_special)
        
        self.wheelbase = QDoubleSpinBox()
        self.wheelbase.setRange(0.1, 1.0)
        self.wheelbase.setValue(0.35)
        self.wheelbase.setSingleStep(0.05)
        
        self.max_steer = QDoubleSpinBox()
        self.max_steer.setRange(10, 60)
        self.max_steer.setValue(30)
        
        self.axle_separation = QDoubleSpinBox()
        self.axle_separation.setRange(0.1, 1.0)
        self.axle_separation.setValue(0.25)
        
        special_layout.addRow("Wheelbase (m):", self.wheelbase)
        special_layout.addRow("Max Steer Angle (°):", self.max_steer)
        special_layout.addRow("Axle Separation (m):", self.axle_separation)
        self.gb_special = gb_special
        clayout.addWidget(gb_special)
        
        clayout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Initialize visibility
        self.on_robot_type_changed(self.robot_type.currentText())
    
    def build_planner_tab(self):
        """Build Planner tab with all 7 algorithms."""
        layout = QVBoxLayout(self.tab_planner)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        clayout = QVBoxLayout(container)
        
        # Planner Selection
        gb_planner = QGroupBox("PATH PLANNER")
        planner_layout = QFormLayout(gb_planner)
        
        self.planner_type = QComboBox()
        self.planner_type.addItems([
            "A*", "Dijkstra", "RRT", "RRT*", "PRM", "PSO", "APF"
        ])
        self.planner_type.currentTextChanged.connect(self.on_planner_changed)
        planner_layout.addRow("Algorithm:", self.planner_type)
        clayout.addWidget(gb_planner)
        
        # Common Parameters
        gb_common = QGroupBox("COMMON PARAMETERS")
        common_layout = QFormLayout(gb_common)
        
        self.robot_radius_plan = QDoubleSpinBox()
        self.robot_radius_plan.setRange(0.1, 1.0)
        self.robot_radius_plan.setValue(0.3)
        
        self.env_width = QSpinBox()
        self.env_width.setRange(10, 100)
        self.env_width.setValue(50)
        
        self.env_height = QSpinBox()
        self.env_height.setRange(10, 100)
        self.env_height.setValue(50)
        
        self.env_resolution = QDoubleSpinBox()
        self.env_resolution.setRange(0.1, 2.0)
        self.env_resolution.setValue(0.5)
        
        common_layout.addRow("Robot Radius (m):", self.robot_radius_plan)
        common_layout.addRow("Env Width (m):", self.env_width)
        common_layout.addRow("Env Height (m):", self.env_height)
        common_layout.addRow("Resolution (m):", self.env_resolution)
        clayout.addWidget(gb_common)
        
        # Algorithm-specific parameters
        gb_algo = QGroupBox("ALGORITHM PARAMETERS")
        algo_layout = QFormLayout(gb_algo)
        
        self.rrt_step = QDoubleSpinBox()
        self.rrt_step.setRange(0.5, 5.0)
        self.rrt_step.setValue(2.0)
        
        self.rrt_iters = QSpinBox()
        self.rrt_iters.setRange(100, 10000)
        self.rrt_iters.setValue(3000)
        
        self.pso_particles = QSpinBox()
        self.pso_particles.setRange(10, 100)
        self.pso_particles.setValue(30)
        
        self.pso_iters = QSpinBox()
        self.pso_iters.setRange(50, 500)
        self.pso_iters.setValue(100)
        
        self.apf_att_gain = QDoubleSpinBox()
        self.apf_att_gain.setRange(0.1, 10.0)
        self.apf_att_gain.setValue(1.0)
        
        self.apf_rep_gain = QDoubleSpinBox()
        self.apf_rep_gain.setRange(0.1, 10.0)
        self.apf_rep_gain.setValue(1.0)
        
        self.prm_samples = QSpinBox()
        self.prm_samples.setRange(50, 2000)
        self.prm_samples.setValue(500)
        
        self.prm_k = QSpinBox()
        self.prm_k.setRange(3, 20)
        self.prm_k.setValue(6)
        
        algo_layout.addRow("RRT Step Size:", self.rrt_step)
        algo_layout.addRow("RRT Iterations:", self.rrt_iters)
        algo_layout.addRow("PSO Particles:", self.pso_particles)
        algo_layout.addRow("PSO Iterations:", self.pso_iters)
        algo_layout.addRow("APF Attractive Gain:", self.apf_att_gain)
        algo_layout.addRow("APF Repulsive Gain:", self.apf_rep_gain)
        algo_layout.addRow("PRM Samples:", self.prm_samples)
        algo_layout.addRow("PRM K-Neighbors:", self.prm_k)
        
        self.gb_algo = gb_algo
        clayout.addWidget(gb_algo)
        
        # Plan button
        btn_plan = QPushButton("🔍 PLAN PATH")
        btn_plan.setStyleSheet("font-size: 14px; padding: 15px;")
        btn_plan.clicked.connect(self.plan_path)
        clayout.addWidget(btn_plan)
        
        # Path info
        self.path_info = QLabel("No path planned")
        self.path_info.setStyleSheet("color: #aaa;")
        clayout.addWidget(self.path_info)
        
        clayout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
    
    def build_sim_tab(self):
        """Build Simulation tab with nav controller and GA tuning."""
        layout = QVBoxLayout(self.tab_sim)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        clayout = QVBoxLayout(container)
        
        # Navigation Controller
        gb_nav = QGroupBox("NAVIGATION CONTROLLER")
        nav_layout = QFormLayout(gb_nav)
        
        self.nav_controller = QComboBox()
        self.nav_controller.addItems(["PurePursuit", "Stanley", "Proportional", "DWA", "MPC"])
        self.nav_controller.currentTextChanged.connect(self.on_nav_changed)
        nav_layout.addRow("Controller:", self.nav_controller)
        clayout.addWidget(gb_nav)
        
        # Controller Parameters
        gb_ctrl = QGroupBox("CONTROLLER PARAMETERS")
        ctrl_layout = QFormLayout(gb_ctrl)
        
        self.ctrl_lookahead = QDoubleSpinBox()
        self.ctrl_lookahead.setRange(0.1, 3.0)
        self.ctrl_lookahead.setValue(0.5)
        
        self.ctrl_k_gain = QDoubleSpinBox()
        self.ctrl_k_gain.setRange(0.1, 10.0)
        self.ctrl_k_gain.setValue(2.5)
        
        self.ctrl_v_max = QDoubleSpinBox()
        self.ctrl_v_max.setRange(0.1, 5.0)
        self.ctrl_v_max.setValue(1.0)
        
        self.ctrl_w_max = QDoubleSpinBox()
        self.ctrl_w_max.setRange(0.5, 5.0)
        self.ctrl_w_max.setValue(2.0)
        
        self.ctrl_k_rho = QDoubleSpinBox()
        self.ctrl_k_rho.setRange(0.1, 5.0)
        self.ctrl_k_rho.setValue(1.0)
        
        self.ctrl_k_alpha = QDoubleSpinBox()
        self.ctrl_k_alpha.setRange(0.5, 10.0)
        self.ctrl_k_alpha.setValue(3.0)
        
        ctrl_layout.addRow("Lookahead (m):", self.ctrl_lookahead)
        ctrl_layout.addRow("K Gain:", self.ctrl_k_gain)
        ctrl_layout.addRow("V Max (m/s):", self.ctrl_v_max)
        ctrl_layout.addRow("ω Max (rad/s):", self.ctrl_w_max)
        ctrl_layout.addRow("K_rho:", self.ctrl_k_rho)
        ctrl_layout.addRow("K_alpha:", self.ctrl_k_alpha)
        
        self.gb_ctrl = gb_ctrl
        clayout.addWidget(gb_ctrl)
        
        # GA Auto-Tuning
        gb_ga = QGroupBox("GA AUTO-TUNING (Optional)")
        ga_layout = QVBoxLayout(gb_ga)
        
        self.ga_enabled = QCheckBox("Enable GA Optimization")
        ga_layout.addWidget(self.ga_enabled)
        
        ga_params = QFormLayout()
        self.ga_pop = QSpinBox()
        self.ga_pop.setRange(10, 100)
        self.ga_pop.setValue(20)
        self.ga_gen = QSpinBox()
        self.ga_gen.setRange(5, 100)
        self.ga_gen.setValue(30)
        ga_params.addRow("Population:", self.ga_pop)
        ga_params.addRow("Generations:", self.ga_gen)
        ga_layout.addLayout(ga_params)
        
        btn_tune = QPushButton("🧬 Run GA Tuning")
        btn_tune.clicked.connect(self.run_ga_tuning)
        ga_layout.addWidget(btn_tune)
        
        clayout.addWidget(gb_ga)
        
        # Run Simulation Button
        btn_run = QPushButton("▶ RUN SIMULATION")
        btn_run.setObjectName("runBtn")
        btn_run.setStyleSheet("font-size: 16px; padding: 20px;")
        btn_run.clicked.connect(self.run_simulation)
        clayout.addWidget(btn_run)
        
        clayout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
    
    # ==================== EVENT HANDLERS ====================
    
    def on_robot_type_changed(self, robot_type):
        """Handle robot type change."""
        # Load defaults
        defaults = model_gen.ROBOT_CONFIGS.get(robot_type, {})
        
        if 'body_length' in defaults:
            self.body_length.setValue(defaults['body_length'])
        if 'body_width' in defaults:
            self.body_width.setValue(defaults['body_width'])
        if 'body_height' in defaults:
            self.body_height.setValue(defaults['body_height'])
        if 'body_mass' in defaults:
            self.body_mass.setValue(defaults['body_mass'])
        if 'wheel_radius' in defaults:
            self.wheel_radius.setValue(defaults['wheel_radius'])
        if 'wheel_separation' in defaults:
            self.wheel_separation.setValue(defaults['wheel_separation'])
        if 'wheelbase' in defaults:
            self.wheelbase.setValue(defaults['wheelbase'])
        if 'max_steer_angle' in defaults:
            self.max_steer.setValue(defaults['max_steer_angle'])
        
        # Show/hide special parameters
        show_wheelbase = robot_type in ['ackermann', 'bicycle']
        show_steer = robot_type in ['ackermann', 'bicycle']
        show_axle = robot_type in ['differential_4wd', 'mecanum']
        
        self.wheelbase.setVisible(show_wheelbase)
        self.max_steer.setVisible(show_steer)
        self.axle_separation.setVisible(show_axle)
    
    def on_planner_changed(self, planner):
        """Handle planner type change."""
        # Could show/hide relevant parameters
        pass
    
    def on_nav_changed(self, nav):
        """Handle nav controller change."""
        # Show relevant parameters
        is_pp = nav == "PurePursuit"
        is_stanley = nav == "Stanley"
        is_prop = nav == "Proportional"
        is_dwa = nav == "DWA"
        
        self.ctrl_lookahead.setEnabled(is_pp)
        self.ctrl_k_gain.setEnabled(is_stanley)
        self.ctrl_k_rho.setEnabled(is_prop)
        self.ctrl_k_alpha.setEnabled(is_prop)
        self.ctrl_w_max.setEnabled(is_dwa)
    
    def update_preview(self):
        """Update map preview."""
        start = (self.start_x.value(), self.start_y.value())
        goal = (self.goal_x.value(), self.goal_y.value())
        
        # Calculate bounds
        all_x = [start[0], goal[0]] + [o['pos'][0] for o in self.obstacles if isinstance(o, dict)]
        all_y = [start[1], goal[1]] + [o['pos'][1] for o in self.obstacles if isinstance(o, dict)]
        
        margin = 3
        bounds = [
            min(all_x) - margin, max(all_x) + margin,
            min(all_y) - margin, max(all_y) + margin
        ]
        
        self.map_preview.set_data(self.obstacles, self.path, start, goal, bounds)
    
    def add_obstacle(self):
        """Add obstacle to list."""
        obs = {
            'type': 'cylinder' if self.obs_type.currentText() == "Cylinder" else 'box',
            'pos': [self.obs_x.value(), self.obs_y.value()],
            'size': [self.obs_size.value(), self.obs_size.value(), 0.5],
            'color': 'red' if self.obs_type.currentText() == "Cylinder" else 'blue'
        }
        self.obstacles.append(obs)
        self.obs_list.addItem(f"{obs['type']} at ({obs['pos'][0]}, {obs['pos'][1]})")
        self.update_preview()
    
    def remove_obstacle(self):
        """Remove selected obstacle."""
        idx = self.obs_list.currentRow()
        if idx >= 0:
            self.obstacles.pop(idx)
            self.obs_list.takeItem(idx)
            self.update_preview()
    
    def clear_obstacles(self):
        """Clear all obstacles."""
        self.obstacles = []
        self.obs_list.clear()
        self.update_preview()
    
    def load_preset(self, preset):
        """Load preset environment."""
        self.clear_obstacles()
        
        if preset == "maze":
            self.obstacles = [
                {'type': 'box', 'pos': [2, 0], 'size': [0.3, 4, 0.5], 'color': 'gray'},
                {'type': 'box', 'pos': [4, 3], 'size': [0.3, 4, 0.5], 'color': 'gray'},
                {'type': 'box', 'pos': [6, 0], 'size': [0.3, 3, 0.5], 'color': 'gray'},
            ]
        elif preset == "dense":
            for i in range(5):
                for j in range(5):
                    if (i + j) % 2 == 0:
                        self.obstacles.append({
                            'type': 'cylinder', 'pos': [i*2, j*2],
                            'size': [0.3, 0.5], 'color': 'red'
                        })
        # Update list and preview
        for obs in self.obstacles:
            self.obs_list.addItem(f"{obs['type']} at ({obs['pos'][0]}, {obs['pos'][1]})")
        self.update_preview()
    
    def plan_path(self):
        """Run path planning."""
        self.status_label.setText("Planning path...")
        
        try:
            # Create environment
            env = Environment(
                width=self.env_width.value(),
                height=self.env_height.value(),
                resolution=self.env_resolution.value()
            )
            
            # Add obstacles
            for obs in self.obstacles:
                if obs['type'] == 'box':
                    env.add_rectangular_obstacle(
                        obs['pos'][0] - obs['size'][0]/2,
                        obs['pos'][1] - obs['size'][1]/2,
                        obs['size'][0], obs['size'][1]
                    )
                else:
                    env.add_circular_obstacle(obs['pos'][0], obs['pos'][1], obs['size'][0])
            
            # Get start/goal
            start = (self.start_x.value(), self.start_y.value())
            goal = (self.goal_x.value(), self.goal_y.value())
            radius = self.robot_radius_plan.value()
            
            # Create planner
            planner_name = self.planner_type.currentText()
            
            if planner_name == "A*":
                planner = AStarPlanner(env, robot_radius=radius)
            elif planner_name == "Dijkstra":
                planner = DijkstraPlanner(env, robot_radius=radius)
            elif planner_name == "RRT":
                planner = RRTPlanner(env, robot_radius=radius,
                                    step_size=self.rrt_step.value(),
                                    max_iterations=self.rrt_iters.value())
            elif planner_name == "RRT*":
                planner = RRTStarPlanner(env, robot_radius=radius,
                                        step_size=self.rrt_step.value(),
                                        max_iterations=self.rrt_iters.value())
            elif planner_name == "PRM":
                planner = PRMPlanner(env, robot_radius=radius,
                                    num_samples=self.prm_samples.value(),
                                    k_neighbors=self.prm_k.value())
            elif planner_name == "PSO":
                planner = PSOPlanner(env, robot_radius=radius,
                                    n_particles=self.pso_particles.value(),
                                    n_iterations=self.pso_iters.value())
            elif planner_name == "APF":
                planner = APFPlanner(env, robot_radius=radius,
                                    attractive_gain=self.apf_att_gain.value(),
                                    repulsive_gain=self.apf_rep_gain.value())
            
            # Plan
            import time
            t0 = time.time()
            self.path = planner.plan(start, goal)
            t1 = time.time()
            
            if self.path:
                self.path_info.setText(f"✓ Path found: {len(self.path)} waypoints in {t1-t0:.3f}s")
                self.path_info.setStyleSheet("color: #00e5ff;")
                self.status_label.setText("Path planning complete!")
            else:
                self.path_info.setText("✗ No path found!")
                self.path_info.setStyleSheet("color: #ff5555;")
                self.status_label.setText("Path planning failed!")
            
            self.update_preview()
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            QMessageBox.warning(self, "Planning Error", str(e))
    
    def run_ga_tuning(self):
        """Run GA auto-tuning for controller."""
        if not self.path or len(self.path) < 2:
            QMessageBox.warning(self, "No Path", "Please plan a path first!")
            return
        
        self.status_label.setText("Running GA optimization...")
        
        try:
            ctrl_name = self.nav_controller.currentText()
            tuner = nav_ctrl.GAControllerTuner(
                ctrl_name,
                population_size=self.ga_pop.value(),
                generations=self.ga_gen.value()
            )
            tuner.set_test_scenario(self.path, self.obstacles)
            
            best_params = tuner.optimize(verbose=True)
            
            # Apply best parameters
            if ctrl_name == "PurePursuit":
                self.ctrl_lookahead.setValue(best_params.get('lookahead', 0.5))
                self.ctrl_v_max.setValue(best_params.get('v_max', 1.0))
            elif ctrl_name == "Stanley":
                self.ctrl_k_gain.setValue(best_params.get('k_gain', 2.5))
            elif ctrl_name == "Proportional":
                self.ctrl_k_rho.setValue(best_params.get('k_rho', 1.0))
                self.ctrl_k_alpha.setValue(best_params.get('k_alpha', 3.0))
            
            self.status_label.setText("GA optimization complete!")
            QMessageBox.information(self, "GA Complete", f"Best params: {best_params}")
            
        except Exception as e:
            self.status_label.setText(f"GA Error: {str(e)}")
    
    def get_robot_params(self):
        """Get current robot parameters from GUI."""
        params = {
            'body_length': self.body_length.value(),
            'body_width': self.body_width.value(),
            'body_height': self.body_height.value(),
            'body_mass': self.body_mass.value(),
            'wheel_radius': self.wheel_radius.value(),
            'wheel_width': self.wheel_width.value(),
            'wheel_mass': self.wheel_mass.value(),
            'wheel_separation': self.wheel_separation.value(),
        }
        
        robot_type = self.robot_type.currentText()
        if robot_type in ['ackermann', 'bicycle']:
            params['wheelbase'] = self.wheelbase.value()
            params['max_steer_angle'] = self.max_steer.value()
        if robot_type in ['differential_4wd', 'mecanum']:
            params['axle_separation'] = self.axle_separation.value()
        
        return params
    
    def get_nav_params(self):
        """Get current nav controller parameters."""
        ctrl = self.nav_controller.currentText()
        
        if ctrl == "PurePursuit":
            return {'lookahead': self.ctrl_lookahead.value(), 'v_max': self.ctrl_v_max.value()}
        elif ctrl == "Stanley":
            return {'k_gain': self.ctrl_k_gain.value(), 'v_max': self.ctrl_v_max.value()}
        elif ctrl == "Proportional":
            return {'k_rho': self.ctrl_k_rho.value(), 'k_alpha': self.ctrl_k_alpha.value(),
                   'v_max': self.ctrl_v_max.value()}
        elif ctrl == "DWA":
            return {'v_max': self.ctrl_v_max.value(), 'w_max': self.ctrl_w_max.value()}
        return {}
    
    def run_simulation(self):
        """Start the MuJoCo simulation."""
        if not self.path or len(self.path) < 2:
            QMessageBox.warning(self, "No Path", "Please plan a path first!")
            return
        
        self.status_label.setText("Starting simulation...")
        QApplication.processEvents()  # Update UI before blocking
        
        config = {
            'robot_type': self.robot_type.currentText(),
            'robot_params': self.get_robot_params(),
            'obstacles': self.obstacles,
            'start': (self.start_x.value(), self.start_y.value(), 
                     float(np.radians(self.start_theta.value()))),
            'goal': (self.goal_x.value(), self.goal_y.value()),
            'path': self.path,
            'nav_controller': self.nav_controller.currentText(),
            'nav_params': self.get_nav_params(),
            'duration': self.max_duration.value(),
            'goal_threshold': self.goal_threshold.value(),
            'show_plots': True
        }
        
        # Run simulation directly (MuJoCo GLFW requires main thread)
        try:
            from navstack.sim import mujoco_sim as sim
            result = sim.run_simulation(**config)
            self.on_simulation_finished(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.on_simulation_finished({'error': str(e)})
    
    def on_simulation_finished(self, result):
        """Handle simulation completion."""
        if 'error' in result:
            self.status_label.setText(f"Simulation error: {result['error']}")
            QMessageBox.warning(self, "Simulation Error", result['error'])
        else:
            msg = "Goal reached!" if result.get('goal_reached') else "Timeout"
            self.status_label.setText(f"Simulation complete: {msg}")


def main():
    app = QApplication(sys.argv)
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.WindowText, QColor(234, 234, 234))
    app.setPalette(palette)
    
    gui = MobileRobotGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
