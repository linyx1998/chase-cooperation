This module infers a human’s current target (intent) from the robot’s motion using distance and direction cues:
$$
p(j)\ \propto\ \exp(-\lambda \cdot \text{dist}(j)) \times \exp(\gamma \cdot \cos \theta_j)
$$

- **Distance evidence** decreases with the (scaled) distance to each task.
- **Direction evidence** increases when the robot’s movement heading aligns with the direction of a task.
- Probabilities are **renormalized** at every inference step.
- When the robot reaches a task within a **completion radius**, that task is marked completed and excluded from future inference.



Minimal Usage:

```python
# 1) Define task coordinates (x, y, z)
tasks = {
    "A": np.array([5.0,  0.0, 0.0]),
    "B": np.array([0.0, 10.0, 0.0]),
    "C": np.array([-5.0, 2.0, 0.0]),
}

# 2) Create the inference system
intent = IntentInferenceSystem(
    task_positions=tasks,
    lambda_dist=0.3,            # distance weight
    gamma_dir=1.0,              # direction weight
    task_completion_radius=2.0, # completion check uses raw (unscaled) distance
    distance_scale_factor=1.0,  # set to your world scaling, by default using 1.0
    inference_interval=1.0,     # seconds between inferences
    enable_csv=True,					  # output to a csv file
    csv_file_path="./logs"
)

# 3) Feed positions over time
result = self.intent_system.update(
    current_position=jackal_position,
    current_time=self.current_time
)

if result is not None:
    self.intent_system.print_status()
```