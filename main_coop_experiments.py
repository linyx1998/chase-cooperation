import numpy as np
import time
import os
import csv
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cooperation import IntentInferenceSystem, DroneCooperationAgent, IndependentDroneAgent
from recommendation import RecommendationManager

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
EXPERIMENT_MODE = 'proposed'  # Options: 'proposed', 'heuristic', 'independent'

print("="*80)
print(f"EXPERIMENT MODE: {EXPERIMENT_MODE.upper()}")
print("="*80)

# ============================================================================
# Task & Agent Configuration
# ============================================================================
drone_initial_position = np.array([3, -2.5, 0.0])
car_initial_position = np.array([3, -2.5, 0.0])

tasks_real = {
    "Black toolbox": np.array([5.7105512619018555, 0.8945932388305664, 0.0]),
    "Blue toolbox": np.array([7.514193534851074, -0.8613284826278687, 0.0]),
    "Dummy": np.array([0.22931592166423798, 0.10323812067508698, 0.0]),
}

tasks_1 = {
    "Black toolbox": np.array([-4, 0.5, 0.0]),
    "Blue toolbox": np.array([8, 0.5, 0.0]),
    "Dummy": np.array([2, 1.0, 0.0]),
}

tasks_2 = {
    "Black toolbox": np.array([-3.5, 0.5, 0.0]),
    "Blue toolbox": np.array([8.5, -0.5, 0.0]),
    "Dummy": np.array([8.5, -2.0, 0.0]),
}

tasks_3 = {
    "Black toolbox": np.array([8.5, 1.0, 0.0]),
    "Blue toolbox": np.array([-4, 0, 0.0]),
    "Dummy": np.array([4.5, -2.0, 0.0]),
}

ENV_NAME = "tasks_3"   # options: tasks_1 / tasks_2 / tasks_3

if ENV_NAME == "tasks_1":
    tasks = tasks_1
elif ENV_NAME == "tasks_2":
    tasks = tasks_2
elif ENV_NAME == "tasks_3":
    tasks = tasks_3

task_types = {
    "Black toolbox": "independent",
    "Blue toolbox": "independent",
    "Dummy": "cooperative",
}

task_rewards_dict = {
    "Black toolbox": 5.0,
    "Blue toolbox": 5.0,
    "Dummy": 10.0
}

task_names = sorted(tasks.keys())

# Create log directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"./logs/{EXPERIMENT_MODE}_{ENV_NAME}_{timestamp}"
os.makedirs(log_path, exist_ok=True)
print(f"Log directory: {log_path}\n")

# ============================================================================
# Initialize Intent Inference System
# ============================================================================
completion_radius = 0.6

intent_system = IntentInferenceSystem(
    task_positions=tasks,
    task_types=task_types,
    lambda_dist=0.3,
    gamma_dir=1.0,
    task_completion_radius=completion_radius,
    distance_scale_factor=1.0,
    inference_interval=0.2,
    enable_csv=True,
    csv_file_path=log_path
)
print("IntentInferenceSystem initialized.\n")

# ============================================================================
# Initialize Distance Tracking CSV
# ============================================================================
distance_csv_filename = os.path.join(log_path, f"distance_tracking_{timestamp}.csv")
distance_csv_file = open(distance_csv_filename, 'w', newline='')
distance_csv_writer = csv.writer(distance_csv_file)

distance_csv_writer.writerow([
    'Time', 'Human_X', 'Human_Y', 'Drone_X', 'Drone_Y',
    'Human_Cumulative_Distance', 'Drone_Cumulative_Distance',
    'Expected_Future_Distance',
    'Drone_Target', 'Completed_Tasks'
])
distance_csv_file.flush()
csv_record_interval = 0.2
last_csv_record_time = 0.0

# Distance tracking
human_cumulative_distance = 0.0
drone_cumulative_distance = 0.0
last_human_position = None
last_drone_position = None

# Human waiting time at dummy
human_dummy_arrival_time = None
human_wait_timeout = 5.0  # seconds

# ============================================================================
# Initialize Drone Agent
# ============================================================================
# drone_initial_position = np.array([3, -2.5, 0.0])

drone_agent = DroneCooperationAgent(
    task_positions=tasks,
    task_types=task_types,
    initial_position=drone_initial_position,
    speed=0.8,
    intent_threshold=0.5,
    commitment_distance=3.0,
    dummy_wait_timeout=5.0,
    task_completion_radius=completion_radius
)

print(f"DroneCooperationAgent initialized for {EXPERIMENT_MODE} mode")
if EXPERIMENT_MODE == 'proposed':
    print("    Uses: Real intent + Full completion info")
elif EXPERIMENT_MODE == 'heuristic':
    print("    Uses: Heuristic intent (nearest task) + No completion info")
elif EXPERIMENT_MODE == 'independent':
    # Use simple independent agent for independent mode
    drone_agent = IndependentDroneAgent(
        task_positions=tasks,
        task_types=task_types,
        initial_position=drone_initial_position,
        speed=0.8,
        cooperative_wait_timeout=5.0,
        task_completion_radius=completion_radius
    )
    print("    Uses: Uniform intent + No completion info")
print()

# ============================================================================
# Initialize Recommendation Manager
# ============================================================================
rec_manager = RecommendationManager(
    task_positions=tasks,
    task_types=task_types,
    task_rewards=task_rewards_dict,
    distance_penalty=1.5,
    intent_threshold=-0.1   # allow 0 goes through
)
print("RecommendationManager initialized.\n")

# ============================================================================
# Human State
# ============================================================================
car_position = car_initial_position
car_angle = np.pi / 2
car_radius = 0.2
current_speed = 0.0
max_speed = 1.5
acceleration = 2.0
friction = 1.0
turn_speed = np.pi * 0.85

start_time = time.time()
last_frame_time = start_time

throttle_input = 0.0
turn_rate = 0.0
keys_pressed = {}

# ============================================================================
# Initialize Drone Target
# ============================================================================
# Force initial inference
intent_system.update(car_position, 0.0, 'car', force_inference=True)

if EXPERIMENT_MODE == 'proposed':
    init_probs = intent_system.get_probabilities()
    init_completed = intent_system.completed_tasks
elif EXPERIMENT_MODE == 'heuristic':
    # Heuristic: nearest task from ALL tasks
    min_dist = float('inf')
    nearest = None
    for name in task_names:
        dist = np.linalg.norm(car_position[:2] - tasks[name][:2])
        if dist < min_dist:
            min_dist = dist
            nearest = name
    init_probs = {name: (1 if name == nearest else 0.) for name in task_names}
    init_completed = set()
elif EXPERIMENT_MODE == 'independent':
    # Independent: 0.0 over ALL tasks
    init_probs = {name: 0.0 for name in task_names}
    init_completed = set()

drone_agent.initialize_target(init_probs, init_completed)
print(f"Drone initialized with target: {drone_agent.get_target()}\n")

# ============================================================================
# Setup Matplotlib Animation
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_aspect('equal')
ax.set_xlim(-6, 10)
ax.set_ylim(-4, 4)
ax.set_title(f"Multi-Agent Cooperation - {EXPERIMENT_MODE.upper()} Mode")
ax.set_xlabel("X coordinate (m)")
ax.set_ylabel("Y coordinate (m)")

# Car
car_artist = plt.Circle((car_position[0], car_position[1]), car_radius, fc='b', zorder=10, label="Human")
ax.add_patch(car_artist)
car_heading_line, = ax.plot([], [], 'b-', linewidth=2, zorder=11)

# Drone
drone_radius = 0.15
drone_pos = drone_agent.get_position()
drone_artist = plt.Circle((drone_pos[0], drone_pos[1]), drone_radius, fc='orange', zorder=10, label="Drone")
ax.add_patch(drone_artist)
drone_target_line, = ax.plot([], [], '--', color='orange', linewidth=2, alpha=0.6, zorder=9)

# Tasks
task_radius_plots = {}
task_labels = {}
for name, pos in tasks.items():
    circle = plt.Circle((pos[0], pos[1]), completion_radius, color='r', alpha=0.3, fill=True)
    ax.add_artist(circle)
    task_radius_plots[name] = circle
    label = ax.text(pos[0], pos[1] + completion_radius + 0.2, name.upper(), 
                    ha='center', va='bottom', fontweight='bold', color='#b91c1c')
    task_labels[name] = label
    
# Add timeout text for human at dummy (initially hidden)
cooperative_task_name = [k for k in tasks.keys() if task_types[k] == 'cooperative'][0]
dummy_pos = tasks[cooperative_task_name]
timeout_text = ax.text(dummy_pos[0], dummy_pos[1] - completion_radius - 0.5, 
                       '', 
                       ha='center', va='top', 
                       fontsize=10, color='darkorange',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, edgecolor='darkorange', linewidth=1),
                       visible=False, zorder=15)

# Probability visualization (only for proposed)
prob_lines = {}
if EXPERIMENT_MODE == 'proposed':
    for name in task_names:
        line, = ax.plot([], [], '--', color='#3b82f6', alpha=0.1)
        prob_lines[name] = line

# Text displays
time_text = ax.text(0.1, 0.85, '', transform=ax.transAxes)
mode_text = ax.text(0.02, 0.95, f'Mode: {EXPERIMENT_MODE.upper()}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
distance_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=10)

# Recommendation text (only for proposed)
# recommendation_text = None
# if EXPERIMENT_MODE == 'proposed':
#     recommendation_text = ax.text(-6.9, -1.5, '', fontsize=10,
#                                   bbox=dict(boxstyle='square', facecolor='lightgreen', 
#                                            alpha=0.8, edgecolor='none'),
#                                   verticalalignment='top', horizontalalignment='left')

# Probability bars (only for proposed)
ax_bar = None
prob_bars = None
if EXPERIMENT_MODE == 'proposed':
    ax_bar = fig.add_axes([0.09, 0.65, 0.2, 0.2])
    ax_bar.set_xlim(0, 1)
    ax_bar.set_yticks(np.arange(len(task_names)))
    ax_bar.set_yticklabels(task_names)
    ax_bar.set_title("Human Intent")
    ax_bar.set_xticks([0, 0.5, 1])
    prob_bars = ax_bar.barh(task_names, [0]*len(task_names), color='#3b82f6')

ax.legend(handles=[car_artist, drone_artist], loc='lower right')

# ============================================================================
# Keyboard Control
# ============================================================================
def on_key_press(event):
    keys_pressed[event.key] = True

def on_key_release(event):
    keys_pressed[event.key] = False

fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_release_event', on_key_release)

# ============================================================================
# Animation Functions
# ============================================================================
def init():
    return []

def update(frame):
    global car_position, car_angle, last_frame_time, current_speed
    global throttle_input, turn_rate
    global human_cumulative_distance, drone_cumulative_distance
    global last_human_position, last_drone_position
    global last_csv_record_time
    global human_dummy_arrival_time 
    
    # ========================================================================
    # Check if all tasks completed - stop updates
    # ========================================================================
    # if len(intent_system.completed_tasks) >= len(tasks):
    #     return []
    
    # ========================================================================
    # Check if all tasks completed
    # ========================================================================
    all_completed = len(intent_system.completed_tasks) >= len(tasks)
    
    # Time
    current_time = time.time()
    delta_time = current_time - last_frame_time
    if delta_time == 0:
        delta_time = 0.0001
    last_frame_time = current_time
    total_time = current_time - start_time
    
    # Only update total_time if not all completed
    if not all_completed:
        total_time = current_time - start_time
    else:
        # Use the time when last task was completed
        if not hasattr(update, 'final_time'):
            update.final_time = current_time - start_time
        total_time = update.final_time
    
    # ========================================================================
    # 1. Update Human Position
    # ========================================================================
    throttle_input = 0.0
    turn_rate = 0.0
    
    if keys_pressed.get('up'):
        throttle_input = 1.0
    if keys_pressed.get('down'):
        throttle_input = -0.5
    if keys_pressed.get('left'):
        turn_rate = turn_speed
    if keys_pressed.get('right'):
        turn_rate = -turn_speed
    
    car_angle += turn_rate * delta_time
    car_angle = car_angle % (2 * np.pi)
    
    if throttle_input > 0:
        current_speed = min(max_speed, current_speed + acceleration * delta_time)
    elif throttle_input < 0:
        current_speed = max(-max_speed/2, current_speed - acceleration * delta_time)
    else:
        if current_speed > 0:
            current_speed = max(0, current_speed - friction * delta_time)
        elif current_speed < 0:
            current_speed = min(0, current_speed + friction * delta_time)
    
    # Track human distance (only if not all completed)
    if not all_completed:
        prev_human_position = car_position.copy()
        car_position[0] += np.cos(car_angle) * current_speed * delta_time
        car_position[1] += np.sin(car_angle) * current_speed * delta_time

        if last_human_position is None:
            last_human_position = prev_human_position.copy()

        human_cumulative_distance += np.linalg.norm(car_position[:2] - last_human_position[:2])
        last_human_position = car_position.copy()
    else:
        # Still allow movement but don't count distance
        car_position[0] += np.cos(car_angle) * current_speed * delta_time
        car_position[1] += np.sin(car_angle) * current_speed * delta_time
    
    # ========================================================================
    # 2. Calculate Intent and Completion Info for Drone
    # ========================================================================
    
    # Update intent system (tracks real completion)
    intent_system.update(car_position, total_time, 'car')
    
    # Calculate intent probabilities based on mode
    if EXPERIMENT_MODE == 'proposed':
        probabilities = intent_system.get_probabilities()
        if probabilities is None:
            probabilities = {name: 0.0 for name in task_names}
        
    elif EXPERIMENT_MODE == 'heuristic':
        # Find nearest task from UNCOMPLETED tasks
        min_dist = float('inf')
        nearest = None
        remaining = [n for n in task_names if n not in intent_system.completed_tasks]

        for name in remaining:
            dist = np.linalg.norm(car_position[:2] - tasks[name][:2])
            if dist < min_dist:
                min_dist = dist
                nearest = name

        probabilities = {name: 0.0 for name in task_names}

        if nearest is not None:
            if len(remaining) == 1:
                probabilities[remaining[0]] = 1.0
            else:
                for name in remaining:
                    probabilities[name] = 1.0 if name == nearest else 0.0
        
    elif EXPERIMENT_MODE == 'independent':
        # 0.0 over ALL tasks (including uncompleted ones)
        probabilities = {name: 0.0 for name in task_names}
    
    completed_for_drone = intent_system.completed_tasks
    
    # ========================================================================
    # 3. Update Drone
    # ========================================================================
    
    drone_agent.update_target(probabilities, completed_for_drone, total_time)
    
    drone_agent.move_towards_target(delta_time)
    drone_pos = drone_agent.get_position()

    if not all_completed:
        if last_drone_position is not None:
            drone_distance_delta = np.linalg.norm(drone_pos[:2] - last_drone_position[:2])
            drone_cumulative_distance += drone_distance_delta

        last_drone_position = drone_pos.copy()
    
    # Check drone completion
    intent_system.update(drone_pos, total_time, 'drone')
    
    # ========================================================================
    # 4. Calculate Expected Future Distances (for all modes)
    # ========================================================================
    
    recommendation = None
    expected_h_distance = 0.0
    expected_d_distance = 0.0
    
    if probabilities:
        recommendation = rec_manager.get_recommendation(
            human_intent=probabilities,
            completed_tasks=intent_system.completed_tasks,
            human_position=car_position,
            drone_position=drone_agent.get_position()
        )
        
        if recommendation:
            expected_h_distance = recommendation['h_distance']
            expected_d_distance = recommendation['d_distance']
    
    # ========================================================================
    # 5. Write to CSV
    # ========================================================================
    
    if total_time - last_csv_record_time < csv_record_interval:
        pass
    else:
        last_csv_record_time = total_time
        
        completed_str = ','.join(sorted(intent_system.completed_tasks))
        
        distance_csv_writer.writerow([
            f"{total_time:.2f}",
            f"{car_position[0]:.4f}", f"{car_position[1]:.4f}",
            f"{drone_pos[0]:.4f}", f"{drone_pos[1]:.4f}",
            f"{human_cumulative_distance:.4f}",
            f"{drone_cumulative_distance:.4f}",
            f"{expected_h_distance + expected_d_distance:.4f}",
            drone_agent.get_target() or '',
            completed_str
        ])
        distance_csv_file.flush()
    
    # ========================================================================
    # 6. Update Visualization
    # ========================================================================
    
    # Car
    car_artist.center = (car_position[0], car_position[1])
    arrow_start_x = car_position[0] + np.cos(car_angle) * car_radius
    arrow_start_y = car_position[1] + np.sin(car_angle) * car_radius
    arrow_end_x = car_position[0] + np.cos(car_angle) * (car_radius + 0.5)
    arrow_end_y = car_position[1] + np.sin(car_angle) * (car_radius + 0.5)
    car_heading_line.set_data([arrow_start_x, arrow_end_x], [arrow_start_y, arrow_end_y])
    
    # Drone
    drone_artist.center = (drone_pos[0], drone_pos[1])
    
    # Drone target line
    drone_target = drone_agent.get_target()
    if drone_target and drone_target not in intent_system.completed_tasks:
        target_pos = tasks[drone_target][:2]
        drone_target_line.set_data([drone_pos[0], target_pos[0]], [drone_pos[1], target_pos[1]])
        drone_target_line.set_visible(True)
    else:
        drone_target_line.set_visible(False)
    
    # Time
    time_text.set_text(f"Time: {total_time:.1f}s")
    
    # Distance
    distance_text.set_text(
        f"Human: {human_cumulative_distance:.2f}m\n"
        f"Drone: {drone_cumulative_distance:.2f}m\n"
        f"Total: {human_cumulative_distance + drone_cumulative_distance:.2f}m"
    )
    
    # Probability visualization (only for proposed)
    if EXPERIMENT_MODE == 'proposed':
        for i, name in enumerate(task_names):
            prob = probabilities[name]
            
            # Lines
            if name not in intent_system.completed_tasks:
                pos = tasks[name][:2]
                prob_lines[name].set_data([car_position[0], pos[0]], [car_position[1], pos[1]])
                prob_lines[name].set_alpha(prob * 0.8)
                prob_lines[name].set_linewidth(1 + prob * 4)
                prob_lines[name].set_visible(True)
            else:
                prob_lines[name].set_visible(False)
            
            # Bars
            if prob_bars:
                prob_bars[i].set_width(prob)
    
    # Task status (always show real completion)
    for name in task_names:
        if name in intent_system.completed_tasks:
            task_radius_plots[name].set_color('g')
            task_labels[name].set_color('#15803d')
        else:
            task_radius_plots[name].set_color('r')
            task_labels[name].set_color('#b91c1c')
    
    # ========================================================================
    # Show timeout warning for human at dummy
    # ========================================================================
    global human_dummy_arrival_time
    
    cooperative_task = [k for k in tasks.keys() if task_types[k] == 'cooperative'][0]
    if cooperative_task not in intent_system.completed_tasks:
        dummy_pos = tasks[cooperative_task][:2]
        human_dist_to_dummy = np.linalg.norm(car_position[:2] - dummy_pos)
        
        # Check if human is at dummy
        if human_dist_to_dummy <= completion_radius:
            # Human just arrived
            if human_dummy_arrival_time is None:
                human_dummy_arrival_time = total_time
            
            # Calculate wait time
            wait_time = total_time - human_dummy_arrival_time
            remaining_time = human_wait_timeout - wait_time
            
            # Check if drone is also here
            agents_at_dummy = intent_system.agents_at_cooperative_tasks.get(cooperative_task, set())
            if len(agents_at_dummy) < 2:  # Human alone, waiting
                if remaining_time > 0:
                    # Still waiting
                    timeout_text.set_text(f'Waiting... {remaining_time:.1f}s')
                    timeout_text.set_color('darkorange')
                    timeout_text.get_bbox_patch().set_facecolor('yellow')
                    timeout_text.get_bbox_patch().set_edgecolor('darkorange')
                    timeout_text.set_visible(True)
                else:
                    # Timed out - warning to leave
                    timeout_text.set_text('Timeout.')
                    timeout_text.set_color('red')
                    timeout_text.get_bbox_patch().set_facecolor('#ffcccc')
                    timeout_text.get_bbox_patch().set_edgecolor('red')
                    timeout_text.set_visible(True)
            else:
                # Both agents here
                timeout_text.set_visible(False)
        else:
            # Human left dummy
            human_dummy_arrival_time = None
            timeout_text.set_visible(False)
    else:
        # Dummy completed
        human_dummy_arrival_time = None
        timeout_text.set_visible(False)
    
    return []

# ============================================================================
# Run Animation
# ============================================================================
try:
    print("="*80)
    print("Starting Simulation")
    print("="*80)
    print("Controls: Arrow keys to move")
    print("Close window to stop\n")
    
    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=20, repeat=False)
    plt.show()
    
except Exception as e:
    print(f"Error: {e}")

finally:
    intent_system.close()
    distance_csv_file.close()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Mode: {EXPERIMENT_MODE}")
    print(f"Time: {time.time() - start_time:.2f}s")
    print(f"Human distance: {human_cumulative_distance:.2f}m")
    print(f"Drone distance: {drone_cumulative_distance:.2f}m")
    print(f"Total: {human_cumulative_distance + drone_cumulative_distance:.2f}m")
    print(f"Tasks completed: {len(intent_system.completed_tasks)}/{len(tasks)}")
    print(f"\nLogs saved to: {log_path}")
    print("="*80)