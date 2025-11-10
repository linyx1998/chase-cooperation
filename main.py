import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from intent_inference import IntentInferenceSystem  # Import Yuxin's class

# --- 1. Define Tasks and Create Log Directory ---

tasks = {
    "box1": np.array([5.0, 5.0, 0.0]),
    "box2": np.array([-5.0, 5.0, 0.0]),
    "dummy": np.array([0.0, 10.0, 0.0]),
}
task_names = sorted(tasks.keys())

log_path = "./logs"
os.makedirs(log_path, exist_ok=True)

# --- 2. Initialize the Intent Inference System ---

try:
    intent_system = IntentInferenceSystem(
        task_positions=tasks,
        lambda_dist=0.3,
        gamma_dir=1.0,
        task_completion_radius=1.5,
        distance_scale_factor=1.0,
        inference_interval=0.5, # Run inference every 0.5 seconds
        enable_csv=True,
        csv_file_path=log_path
    )
    print("IntentInferenceSystem initialized.")
except Exception as e:
    print(f"Error initializing IntentInferenceSystem: {e}")
    exit()

# --- 3. Simulation State ---

car_position = np.array([0.0, 0.0, 0.0])
start_time = time.time()
velocity = 2.0  # meters per second

# --- 4. Setup the Matplotlib Animation ---

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.set_xlim(-8, 8)
ax.set_ylim(-2, 12)
ax.set_title("Intent Inference Simulation")
ax.set_xlabel("X coordinate (m)")
ax.set_ylabel("Y coordinate (m)")
ax.grid(True)

# --- Plot Artists (the elements we will update) ---

# a) Car plot
car_plot, = ax.plot([car_position[0]], [car_position[1]], 'bo', markersize=10, label="Car")

# b) Task plots (uncompleted)
task_plots = {}
for name, pos in tasks.items():
    plot, = ax.plot([pos[0]], [pos[1]], 'ro', markersize=15, alpha=0.6, label=f"Task: {name}")
    ax.text(pos[0], pos[1] + 0.5, name.upper(), ha='center', va='bottom', fontweight='bold')
    task_plots[name] = plot

# c) Task completion circles
completion_plots = {}
for name, pos in tasks.items():
    circle = plt.Circle((pos[0], pos[1]), intent_system.task_completion_radius, 
                          color='g', alpha=0.3, fill=True, visible=False)
    ax.add_artist(circle)
    completion_plots[name] = circle

# d) Probability lines from car to task
prob_lines = {}
for name, pos in tasks.items():
    line, = ax.plot([car_position[0], pos[0]], [car_position[1], pos[1]], 
                     'k--', alpha=0.1) # Start invisible
    prob_lines[name] = line

# e) Time text
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# f) Probability bars (using a separate, inset axes)
ax_bar = fig.add_axes([0.02, 0.7, 0.2, 0.2]) # [left, bottom, width, height]
ax_bar.set_xlim(0, 1)
ax_bar.set_yticks(np.arange(len(task_names)))
ax_bar.set_yticklabels(task_names)
ax_bar.set_title("Probabilities")
ax_bar.set_xticks([0, 0.5, 1])
ax_bar.set_xticklabels(['0', '0.5', '1'])

prob_bars = ax_bar.barh(task_names, [0] * len(task_names), color='orange')

ax.legend(loc='lower left')


def init():
    """Initializes the animation."""
    car_plot.set_data([], [])
    time_text.set_text('')
    for name in task_names:
        task_plots[name].set_data([], [])
        prob_lines[name].set_data([], [])
        completion_plots[name].set_visible(False)
    
    artists = [car_plot, time_text] + \
              list(task_plots.values()) + \
              list(prob_lines.values()) + \
              list(completion_plots.values())
              
    return artists

def update(frame):
    """Called each frame to update the animation."""
    global car_position # Use global car_position
    
    current_time = time.time() - start_time
    
    # --- 1. Update Car Position (Simulation) ---
    if not intent_system._all_tasks_completed():
        if "dummy" not in intent_system.completed_tasks:
            target = tasks["dummy"]
        elif "box1" not in intent_system.completed_tasks:
            target = tasks["box1"]
        else:
            target = tasks["box2"]
            
        direction = target[:2] - car_position[:2]
        norm = np.linalg.norm(direction)
        
        # Simple time-based movement
        # (A real loop would use a fixed delta-time)
        if norm > 0.1:
            move_vec = (direction / norm) * velocity * 0.05 # 0.05 is ~timestep
            car_position[0] += move_vec[0]
            car_position[1] += move_vec[1]
    
    # --- 2. Call Yuxin's Intent System ---
    intent_system.update(
        current_position=car_position,
        current_time=current_time
    )
    
    probabilities = intent_system.get_probabilities()
    if probabilities is None:
        probabilities = {name: 0.0 for name in task_names}

    # --- 3. Update Plot Artists ---
    
    # Update car
    car_plot.set_data([car_position[0]], [car_position[1]])
    
    # Update time
    time_text.set_text(f"Time: {current_time:.1f}s")
    
    # Update tasks, lines, and bars
    for i, name in enumerate(task_names):
        pos = tasks[name][:2]
        prob = probabilities[name]
        
        # Update probability line
        prob_lines[name].set_data([car_position[0], pos[0]], [car_position[1], pos[1]])
        prob_lines[name].set_alpha(prob * 0.8)
        prob_lines[name].set_linewidth(1 + prob * 4)
        
        # Update probability bar
        prob_bars[i].set_width(prob)
        
        # Show/hide completion
        if name in intent_system.completed_tasks:
            task_plots[name].set_alpha(0.1) # Fade out original task
            completion_plots[name].set_visible(True) # Show green circle
        
    # Return all updated artists
    artists = [car_plot, time_text] + \
              list(prob_lines.values()) + \
              list(completion_plots.values()) + \
              list(prob_bars)
              
    return artists

# --- 5. Run the Animation ---
try:
    print("Starting animation... Close the plot window to stop.")
    # interval=50ms -> ~20 FPS. 
    # blit=True makes it run faster by only redrawing changed parts.
    ani = animation.FuncAnimation(
        fig, update, 
        frames=None, 
        init_func=init, 
        blit=True, 
        interval=50, 
        repeat=False
    )
    plt.show()

except Exception as e:
    print(f"An error occurred during animation: {e}")

finally:
    # This runs after the plot window is closed
    intent_system.close()
    print("Animation stopped. CSV file saved.")

