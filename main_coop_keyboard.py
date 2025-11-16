# --- Import Necessary Libraries ---

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cooperation import IntentInferenceSystem, DroneCooperationAgent

# --- 1. Define Tasks and Create Log Directory ---

# Task positions
tasks = {
    "box1": np.array([5.0, 5.0, 0.0]),
    "box2": np.array([-5.0, 5.0, 0.0]),
    "dummy": np.array([0.0, 10.0, 0.0]),
}

# Define task types
task_types = {
    "box1": "independent",
    "box2": "independent",
    "dummy": "cooperative",
}

task_names = sorted(tasks.keys())

log_path = "./logs"
os.makedirs(log_path, exist_ok=True)

# --- 2. Initialize the Intent Inference System ---

try:
    completion_radius = 1.0 
    
    intent_system = IntentInferenceSystem(
        task_positions=tasks,
        task_types=task_types,
        lambda_dist=0.3,
        gamma_dir=1.0,
        task_completion_radius=completion_radius,
        distance_scale_factor=1.0,
        inference_interval=0.5,
        enable_csv=True,
        csv_file_path=log_path
    )
    print("IntentInferenceSystem initialized successfully.\n")
except Exception as e:
    print(f"Error initializing IntentInferenceSystem: {e}")
    exit()

# --- 3. Simulation State ---

# Car state
car_position = np.array([0.0, 0.0, 0.0])
car_angle = np.pi / 2
car_radius = 0.2
current_speed = 0.0
max_speed = 2.5
acceleration = 2.0
friction = 1.0
turn_speed = np.pi * 0.85

# Time tracking
start_time = time.time()
last_frame_time = start_time

throttle_input = 0.0
turn_rate = 0.0
keys_pressed = {}

# --- 4. Initialize Drone Cooperation Agent ---

try:
    drone_initial_position = np.array([0.0, -1.0, 0.0])
    
    drone_agent = DroneCooperationAgent(
        task_positions=tasks,
        task_types=task_types,
        initial_position=drone_initial_position,
        speed=1.5,
        intent_threshold=0.5,  # Can adjust this (must be >= 0.5)
        commitment_distance=3.0
    )
    
    # Wait for first intent before initializing target
    drone_initialized = False
    
    print("DroneCooperationAgent created.\n")
    
    # Force an initial inference to get starting intent probabilities
    intent_system.update(
        current_position=car_position,
        current_time=0.0,
        agent_id='car',
        force_inference=True
    )
    initial_probs = intent_system.get_probabilities()
    if initial_probs:
        drone_agent.initialize_target(initial_probs, intent_system.completed_tasks)
        drone_initialized = True
        print()
    
except Exception as e:
    print(f"Error initializing DroneCooperationAgent: {e}")
    exit()

# --- 5. Setup the Matplotlib Animation ---

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.set_xlim(-8, 8)
ax.set_ylim(-2, 12)
ax.set_title("Multi-Agent Intent Inference & Cooperation")
ax.set_xlabel("X coordinate (m)")
ax.set_ylabel("Y coordinate (m)")

# --- Plot Artists ---

# Car (blue circle)
car_artist = plt.Circle((car_position[0], car_position[1]), 
                         car_radius, fc='b', zorder=10, label="Car (Human)")
ax.add_patch(car_artist)

# Car heading arrow
car_heading_line, = ax.plot([], [], 'b-', linewidth=2, zorder=11)

# Drone (orange circle)
drone_radius = 0.15
drone_pos = drone_agent.get_position()
drone_artist = plt.Circle((drone_pos[0], drone_pos[1]), 
                          drone_radius, fc='orange', zorder=10, label="Drone (Agent)")
ax.add_patch(drone_artist)

# Drone target line (dashed orange)
drone_target_line, = ax.plot([], [], '--', color='orange', linewidth=2, alpha=0.6, zorder=9)

# Task completion radius plots
task_radius_plots = {}
task_labels = {}
for name, pos in tasks.items():
    circle = plt.Circle((pos[0], pos[1]), completion_radius, 
                          color='r', alpha=0.3, fill=True, visible=True)
    ax.add_artist(circle)
    task_radius_plots[name] = circle
    
    label = ax.text(pos[0], pos[1] + completion_radius + 0.2, name.upper(), 
            ha='center', va='bottom', fontweight='bold', color='#b91c1c')
    task_labels[name] = label

# Probability lines from car
prob_lines = {}
for name, pos in tasks.items():
    line, = ax.plot([], [], '--', color='#3b82f6', alpha=0.1) 
    prob_lines[name] = line

# Time text
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Drone status text
drone_status_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=9, color='darkorange')

# Probability bars
ax_bar = fig.add_axes([0.08, 0.7, 0.2, 0.2])
ax_bar.set_xlim(0, 1)
ax_bar.set_yticks(np.arange(len(task_names)))
ax_bar.set_yticklabels(task_names)
ax_bar.set_title("Human Intent")
ax_bar.set_xticks([0, 0.5, 1])
ax_bar.set_xticklabels(['0', '0.5', '1'])
prob_bars = ax_bar.barh(task_names, [0] * len(task_names), color='#3b82f6')

ax.legend(handles=[car_artist, drone_artist], loc='lower left')

# --- 6. Keyboard Control Functions ---

def on_key_press(event):
    keys_pressed[event.key] = True

def on_key_release(event):
    keys_pressed[event.key] = False

fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_release_event', on_key_release)

# --- 7. Animation Functions ---

def init():
    car_artist.center = (car_position[0], car_position[1])
    car_heading_line.set_data([], [])
    
    drone_pos = drone_agent.get_position()
    drone_artist.center = (drone_pos[0], drone_pos[1])
    drone_target_line.set_data([], [])
    
    time_text.set_text('')
    drone_status_text.set_text('')
    
    for name in task_names:
        prob_lines[name].set_data([], [])
        task_radius_plots[name].set_color('r')
        task_radius_plots[name].set_visible(True)
        task_labels[name].set_color('#b91c1c')
    
    artists = [car_artist, car_heading_line, drone_artist, drone_target_line, 
               time_text, drone_status_text] + \
              list(prob_lines.values()) + \
              list(task_radius_plots.values()) + \
              list(task_labels.values())
              
    return artists

def update(frame):
    global car_position, car_angle, last_frame_time, current_speed, throttle_input, turn_rate
    global drone_initialized
    
    # Calculate delta time
    current_time = time.time()
    delta_time = current_time - last_frame_time
    if delta_time == 0: delta_time = 0.0001
    last_frame_time = current_time
    total_time = current_time - start_time
    
    # --- 1. Update Car Position ---
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
        current_speed = max(-max_speed / 2, current_speed - acceleration * delta_time)
    else:
        if current_speed > 0:
            current_speed = max(0, current_speed - friction * delta_time)
        elif current_speed < 0:
            current_speed = min(0, current_speed + friction * delta_time)
        
    car_position[0] += np.cos(car_angle) * current_speed * delta_time
    car_position[1] += np.sin(car_angle) * current_speed * delta_time
            
    # --- 2. Call Intent System for Car ---
    intent_system.update(
        current_position=car_position,
        current_time=total_time,
        agent_id='car'
    )
    
    probabilities = intent_system.get_probabilities()
    if probabilities is None:
        probabilities = {name: 0.0 for name in task_names}
    
    # --- 3. Initialize Drone Target (once we have first intent) ---
    if not drone_initialized and probabilities:
        drone_agent.initialize_target(probabilities)
        drone_initialized = True
    
    # --- 4. Update Drone Based on Intent ---
    if drone_initialized:
        drone_agent.update_target(probabilities, intent_system.completed_tasks)
        drone_agent.move_towards_target(delta_time)
    
    # --- 5. Update Intent System for Drone (to check completion) ---
    drone_pos = drone_agent.get_position()
    intent_system.update(
        current_position=drone_pos,
        current_time=total_time,
        agent_id='drone'
    )
    
    # --- 6. Update Plot Artists ---
    
    # Car
    car_artist.center = (car_position[0], car_position[1])
    
    arrow_start_x = car_position[0] + np.cos(car_angle) * car_radius
    arrow_start_y = car_position[1] + np.sin(car_angle) * car_radius
    arrow_length = 0.5
    arrow_end_x = car_position[0] + np.cos(car_angle) * (car_radius + arrow_length)
    arrow_end_y = car_position[1] + np.sin(car_angle) * (car_radius + arrow_length)
    car_heading_line.set_data([arrow_start_x, arrow_end_x], [arrow_start_y, arrow_end_y])
    
    # Drone
    drone_artist.center = (drone_pos[0], drone_pos[1])
    
    # Drone target line
    drone_target = drone_agent.get_target()
    if drone_target and drone_target not in intent_system.completed_tasks:
        target_pos = tasks[drone_target][:2]
        drone_target_line.set_data([drone_pos[0], target_pos[0]], 
                                    [drone_pos[1], target_pos[1]])
        drone_target_line.set_visible(True)
    else:
        drone_target_line.set_visible(False)
    
    # Time text
    time_text.set_text(f"Time: {total_time:.1f}s")
    
    # Probability lines and bars
    for i, name in enumerate(task_names):
        pos = tasks[name][:2]
        prob = probabilities[name]
        
        if name not in intent_system.completed_tasks:
            prob_lines[name].set_data([car_position[0], pos[0]], [car_position[1], pos[1]])
            prob_lines[name].set_alpha(prob * 0.8)
            prob_lines[name].set_linewidth(1 + prob * 4)
            prob_lines[name].set_visible(True)
        else:
            prob_lines[name].set_visible(False)
            
        prob_bars[i].set_width(prob)
        
        if name in intent_system.completed_tasks:
            task_radius_plots[name].set_color('g')
            task_labels[name].set_color('#15803d')
        else:
            task_radius_plots[name].set_color('r')
            task_labels[name].set_color('#b91c1c')
            
    artists = [car_artist, car_heading_line, drone_artist, drone_target_line,
               time_text, drone_status_text] + \
              list(prob_lines.values()) + \
              list(task_radius_plots.values()) + \
              list(prob_bars) + \
              list(task_labels.values())
              
    return artists

# --- 8. Run the Animation ---
try:
    print("=" * 60)
    print("Starting Multi-Agent Cooperation Simulation")
    print("=" * 60)
    print("Controls: Use ARROW KEYS to control the car")
    print("The drone will cooperate based on your intent!")
    print("Close the plot window to stop.\n")
    
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=None,
        init_func=init,
        blit=True,
        interval=20,
        repeat=False
    )
    plt.show()

except Exception as e:
    print(f"An error occurred during animation: {e}")

finally:
    intent_system.close()
    print("\nSimulation stopped. CSV file saved.")