# --- Import Necessary Libraries ---

import numpy as np  # For numerical operations (vectors, positions)
import time         # To get the current time for the simulation
import os           # To create directories (for the 'logs' folder)
import matplotlib.pyplot as plt # The main library for plotting
import matplotlib.animation as animation # The library to create the animation
from intent_inference import IntentInferenceSystem  # <-- This is how we import Yuxin's class from the other file

# --- 1. Define Tasks and Create Log Directory ---

# This dictionary holds the real-world 3D coordinates (x, y, z) for each target.
# Yuxin's class will use these to calculate distances.
tasks = {
    "box1": np.array([5.0, 5.0, 0.0]),
    "box2": np.array([-5.0, 5.0, 0.0]),
    "dummy": np.array([0.0, 10.0, 0.0]),
}
# Get a sorted list of names, which helps us plot things in a consistent order
task_names = sorted(tasks.keys())

# Define the path for the log files Yuxin's class will create
log_path = "./logs"
# Create the 'logs' directory if it doesn't already exist
os.makedirs(log_path, exist_ok=True)

# --- 2. Initialize the Intent Inference System ---

try:
    # This is the radius (in meters) for a task to be "complete".
    # Yuxin's code will check the car's distance against this.
    completion_radius = 1.0 
    
    # This is the most important step: We are creating an "instance" of Yuxin's class.
    # This 'intent_system' object is the "brain" we will be testing.
    intent_system = IntentInferenceSystem(
        task_positions=tasks,              # Tell the brain where the targets are
        lambda_dist=0.3,                   # Parameter from Yuxin's code (distance weight)
        gamma_dir=1.0,                     # Parameter from Yuxin's code (direction weight)
        task_completion_radius=completion_radius, # Tell the brain how close to get
        distance_scale_factor=1.0,         # Parameter from Yuxin's code
        inference_interval=0.5,            # How often to run the (slower) inference logic
        enable_csv=True,                   # Tell the brain to save a CSV
        csv_file_path=log_path             # Tell the brain where to save it
    )
    print("IntentInferenceSystem (Yuxin's code) initialized successfully.")
except Exception as e:
    print(f"Error initializing IntentInferenceSystem: {e}")
    exit()

# --- 3. Simulation State ---

# These variables are for OUR car, which we will control
car_position = np.array([0.0, 0.0, 0.0]) # The car's starting position
car_angle = np.pi / 2                    # Start facing "up" (90 degrees, or pi/2 radians)
start_time = time.time()                 # The wall-clock time when the script starts
last_frame_time = start_time             # For calculating delta-time

# --- NEW Car Physics Parameters (with Acceleration/Friction) ---
current_speed = 0.0                      # Car's current speed, starts at 0
max_speed = 2.5                          # Slightly reduced top speed (m/s)
acceleration = 2.0                       # How fast the car speeds up (m/s^2)
friction = 1.0                           # How fast the car slows down (m/s^2)
turn_speed = np.pi * 0.85                # Slightly reduced turning speed (rad/s)
car_radius = 0.2                         # Define a radius for the car circle

throttle_input = 0.0                     # -1 for reverse, 0 for none, 1 for forward
turn_rate = 0.0                          # Current turn rate (set by left/right)

# This dictionary will store the state of our keyboard keys
keys_pressed = {}

# --- 4. Setup the Matplotlib Animation ---

# Create the main window ('fig') and the main plotting area ('ax')
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal') # Make the X and Y axes have the same scale
ax.set_xlim(-8, 8)     # Set the visible X-axis range
ax.set_ylim(-2, 12)    # Set the visible Y-axis range
ax.set_title("Intent Inference Simulation (Use Arrow Keys to Steer and Drive)")
ax.set_xlabel("X coordinate (m)")
ax.set_ylabel("Y coordinate (m)")
# ax.grid(True)          # Add a grid for readability

# --- Plot Artists (the elements we will update) ---
# "Artists" are the plot elements. We create them as empty placeholders
# and then just update their data in the 'update' loop. This is *much*
# faster than redrawing everything from scratch every frame.

# a) Car plot: A blue circle
# Create the circle artist
car_artist = plt.Circle((car_position[0], car_position[1]), 
                         car_radius, fc='b', zorder=10, label="Car")
ax.add_patch(car_artist) # Add the car circle to the plot

# b) --- NEW Heading Arrow ---
# A black line to clearly show the car's heading
heading_line, = ax.plot([], [], 'k-', linewidth=2, zorder=11)

# c) Task completion radius plots
# We use dictionaries to keep track of the plot objects for each task
task_radius_plots = {} # Holds the circles
task_labels = {}       # Holds the text labels
for name, pos in tasks.items():
    # Create a red circle (for "incomplete") at the task's position
    circle = plt.Circle((pos[0], pos[1]), completion_radius, 
                          color='r', alpha=0.3, fill=True, visible=True)
    ax.add_artist(circle)
    task_radius_plots[name] = circle
    
    # Add a text label above the circle
    label = ax.text(pos[0], pos[1] + completion_radius + 0.2, name.upper(), 
            ha='center', va='bottom', fontweight='bold', color='#b91c1c') # Red text
    task_labels[name] = label

# d) Probability lines: Dashed blue lines from the car to each task
prob_lines = {}
for name, pos in tasks.items():
    # Create a faint, thin, dashed line as a placeholder
    line, = ax.plot([], [], '--', color='#3b82f6', alpha=0.1) 
    prob_lines[name] = line

# e) Time text: A text object in the top-left corner
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# f) Probability bars: A new, small "inset" plot area for the bar chart
ax_bar = fig.add_axes([0.08, 0.7, 0.2, 0.2]) # [left, bottom, width, height]
ax_bar.set_xlim(0, 1) # Probabilities are from 0 to 1
ax_bar.set_yticks(np.arange(len(task_names)))
ax_bar.set_yticklabels(task_names)
ax_bar.set_title("Probabilities")
ax_bar.set_xticks([0, 0.5, 1])
ax_bar.set_xticklabels(['0', '0.5', '1'])
# Create the horizontal bars. 'prob_bars' is a list of bar objects.
prob_bars = ax_bar.barh(task_names, [0] * len(task_names), color='#3b82f6')

# Add a legend for the 'Car' plot
ax.legend(handles=[car_artist], loc='lower left')


# --- 5. Keyboard Control Functions ---

def on_key_press(event):
    """Handles when a key is pressed down."""
    keys_pressed[event.key] = True

def on_key_release(event):
    """Handles when a key is released."""
    keys_pressed[event.key] = False

# Connect the key event functions to the plot window
fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_release_event', on_key_release)


def init():
    """
    This function is called once at the start of the animation.
    It sets all the artists to their "empty" default state.
    """
    # Set the car's initial position
    car_artist.center = (car_position[0], car_position[1])
    
    # Initialize heading line
    heading_line.set_data([], [])
    
    time_text.set_text('')
    for name in task_names:
        prob_lines[name].set_data([], [])
        # Reset circles to red
        task_radius_plots[name].set_color('r')
        task_radius_plots[name].set_visible(True)
        task_labels[name].set_color('#b91c1c') # Reset label text to red
    
    # Return all the artists that the 'init' function has touched
    artists = [car_artist, time_text, heading_line] + \
              list(prob_lines.values()) + \
              list(task_radius_plots.values()) + \
              list(task_labels.values())
              
    return artists

def update(frame):
    """
    This is the main animation function! It's called for every single frame.
    It (1) updates position from keyboard, (2) calls Yuxin's code, and (3) updates the plots.
    """
    global car_position, car_angle, last_frame_time, current_speed, throttle_input, turn_rate
    
    # Calculate time elapsed since the last frame
    current_time = time.time()
    delta_time = current_time - last_frame_time
    if delta_time == 0: delta_time = 0.0001 # Avoid divide by zero
    last_frame_time = current_time
    
    # Get the total running time
    total_time = current_time - start_time
    
    # --- 1. Update Car Position (OUR NEW keyboard logic) ---
    
    # Reset throttle and steering inputs
    throttle_input = 0.0
    turn_rate = 0.0
    
    # Check which keys are pressed and update throttle/steering
    if keys_pressed.get('up'):
        throttle_input = 1.0  # Go forward
    if keys_pressed.get('down'):
        throttle_input = -0.5 # Go reverse (slower)
    if keys_pressed.get('left'):
        turn_rate = turn_speed # Turn counter-clockwise
    if keys_pressed.get('right'):
        turn_rate = -turn_speed # Turn clockwise
        
    # --- Apply new physics ---
    
    # a) Update car's angle based on steering
    car_angle += turn_rate * delta_time
    car_angle = car_angle % (2 * np.pi) # Keep angle between 0 and 2*pi
    
    # b) Apply acceleration or friction
    if throttle_input > 0:
        # Accelerate forward
        current_speed = min(max_speed, current_speed + acceleration * delta_time)
    elif throttle_input < 0:
        # Accelerate/Brake in reverse
        current_speed = max(-max_speed / 2, current_speed - acceleration * delta_time)
    else:
        # Apply friction if no throttle
        if current_speed > 0:
            current_speed = max(0, current_speed - friction * delta_time)
        elif current_speed < 0:
            current_speed = min(0, current_speed + friction * delta_time)
        
    # c) Update car's 2D position based on angle and new speed
    # We use trigonometry: x = cos(angle), y = sin(angle)
    car_position[0] += np.cos(car_angle) * current_speed * delta_time
    car_position[1] += np.sin(car_angle) * current_speed * delta_time
            
    # --- 2. Call Yuxin's Intent System (The "Brain") ---
    # We feed the car's *new* position (from keyboard) and the *current* time.
    intent_system.update(
        current_position=car_position,
        current_time=total_time
    )
    
    # Get the results (the probabilities) back from the class
    probabilities = intent_system.get_probabilities()
    if probabilities is None:
        # If the class hasn't run inference yet, just use 0s
        probabilities = {name: 0.0 for name in task_names}

    # --- 3. Update Plot Artists (The Visualization) ---
    
    # a) Update the car's circle position
    car_artist.center = (car_position[0], car_position[1])
    
    # b) Update the heading arrow
    # Make the arrow start from the edge of the circle
    arrow_start_x = car_position[0] + np.cos(car_angle) * car_radius
    arrow_start_y = car_position[1] + np.sin(car_angle) * car_radius
    arrow_length = 0.5 # Make the arrow 0.5m long (in addition to radius)
    arrow_end_x = car_position[0] + np.cos(car_angle) * (car_radius + arrow_length)
    arrow_end_y = car_position[1] + np.sin(car_angle) * (car_radius + arrow_length)
    
    heading_line.set_data([arrow_start_x, arrow_end_x], [arrow_start_y, arrow_end_y])
    
    # c) Update the time text
    time_text.set_text(f"Time: {total_time:.1f}s")
    
    # d) Loop through each task to update its line, bar, and circle
    for i, name in enumerate(task_names):
        pos = tasks[name][:2]
        prob = probabilities[name]
        
        # Update the probability line from the car to the task
        if name not in intent_system.completed_tasks:
            prob_lines[name].set_data([car_position[0], pos[0]], [car_position[1], pos[1]])
            prob_lines[name].set_alpha(prob * 0.8) # Set fade based on probability
            prob_lines[name].set_linewidth(1 + prob * 4) # Set thickness
            prob_lines[name].set_visible(True)
        else:
            # Hide the line if the task is done
            prob_lines[name].set_visible(False)
            
        # Update the width of the probability bar
        prob_bars[i].set_width(prob)
        
        # Update the radius circle and label color
        if name in intent_system.completed_tasks:
            task_radius_plots[name].set_color('g') # Change to green
            task_labels[name].set_color('#15803d') # Green text
        else:
            task_radius_plots[name].set_color('r') # Keep it red
            task_labels[name].set_color('#b91c1c') # Red text
            
    # Return all the artists that were changed in this frame
    artists = [car_artist, heading_line, time_text] + \
              list(prob_lines.values()) + \
              list(task_radius_plots.values()) + \
              list(prob_bars) + \
              list(task_labels.values())
              
    return artists

# --- 6. Run the Animation ---
try:
    print("Starting animation... Click the plot window and use ARROW KEYS to move.")
    print("Close the plot window to stop.")
    
    # This is the command that starts the animation
    ani = animation.FuncAnimation(
        fig,          # The figure to animate
        update,       # The function to call for each frame
        frames=None,  # Run indefinitely
        init_func=init, # The function to call at theS start
        blit=True,    # Use 'blitting' for faster, smoother animation
        interval=20,  # Try to run every 20ms (for ~50fps)
        repeat=False
    )
    # This command actually opens the window. It blocks until you close it.
    plt.show()

except Exception as e:
    print(f"An error occurred during animation: {e}")

finally:
    # This code runs *after* you close the plot window
    intent_system.close() # Tell Yuxin's class to close its CSV file
    print("Animation stopped. CSV file saved.")

