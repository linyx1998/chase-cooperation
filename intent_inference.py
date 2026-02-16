import numpy as np
import csv
import os
import datetime
from typing import Dict, Set, Optional, Tuple, List

class IntentInferenceSystem:
    """
    - Reset: p_k(j) ∝ exp(-λ dist(j))
    - Update: p_k(j) ∝ exp(-λ dist(j)) x exp(γ cos(θ_j))
    """
    
    def __init__(
        self,
        task_positions: Dict[str, np.ndarray],
        lambda_dist: float = 0.3,
        gamma_dir: float = 1.0,
        task_completion_radius: float = 2.0,
        distance_scale_factor: float = 1.0,
        inference_interval: float = 1.0,
        enable_csv: bool = True,
        csv_file_path: Optional[str] = None
    ):
        
        self.task_positions = task_positions
        self.lambda_dist = lambda_dist
        self.gamma_dir = gamma_dir
        self.task_completion_radius = task_completion_radius
        self.distance_scale_factor = distance_scale_factor
        self.inference_interval = inference_interval
        
        # State variables
        self.intention_probs: Optional[Dict[str, float]] = None
        self.completed_tasks: Set[str] = set()
        self.previous_position: Optional[np.ndarray] = None
        self.last_inference_time: float = 0.0
        self.current_time: float = 0.0
        
        # History record
        self.inference_history: List[Dict] = []
        
        # CSV export
        self.enable_csv = enable_csv
        self.csv_file = None
        self.csv_writer = None
        if csv_file_path is not None:
            self.csv_file_path = csv_file_path
        else:
            self.csv_file_path = '.'
        
        if self.enable_csv:
            self._init_csv()
        
    def _init_csv(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.csv_file_path, f"intent_inference_{timestamp}.csv")
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        header = ['Time']
        
        # Add intent probability columns for each task
        for task_name in sorted(self.task_positions.keys()):
            header.append(f'Intent_{task_name}')
        
        # Add completion status columns for each task
        for task_name in sorted(self.task_positions.keys()):
            header.append(f'Completed_{task_name}')
        
        self.csv_writer.writerow(header)
        self.csv_file.flush()
        
    def update(
        self,
        current_position: np.ndarray,
        current_time: float,
        force_inference: bool = False
    ) -> Optional[Dict[str, float]]:
        self.current_time = current_time
        
        # Check if any task is completed (may trigger reset and inference)
        task_just_completed = self._check_task_completion(current_position)
        
        # Determine whether to perform inference
        should_infer = (
            force_inference or 
            task_just_completed or  # Force inference if a task is just completed
            (self.current_time - self.last_inference_time >= self.inference_interval)
        )
        
        # If all tasks are completed, perform a final record
        if self._all_tasks_completed():
            if not hasattr(self, '_final_record_written'):
                self._perform_final_inference(current_position)
                self._final_record_written = True
            # Save the current position
            if self.previous_position is None:
                self.previous_position = current_position.copy()
            return None
        
        if should_infer:
            self._perform_inference(current_position)
            self.last_inference_time = self.current_time
            
            # Save current position for next direction calculation
            self.previous_position = current_position.copy()
            
            return self.intention_probs
        
        # Even if no inference is made, still update position
        if self.previous_position is None:
            self.previous_position = current_position.copy()
        
        return None
    
    def _perform_inference(self, current_position: np.ndarray):
        # Perform reset on first inference or after a task is completed
        if self.intention_probs is None:
            self._reset_intention_probs(current_position)
        else:
            self._update_intention_probs(current_position)
        
        # Compute distances
        distances = self._calculate_distances(current_position)
        
        # Write to CSV
        if self.enable_csv:
            self._write_to_csv()
        
        # Save to history
        self.inference_history.append({
            'time': self.current_time,
            'position': current_position.copy(),
            'distances': distances.copy(),
            'intention_probs': self.intention_probs.copy(),
            'completed_tasks': self.completed_tasks.copy()
        })
    
    def _write_to_csv(self):
        if self.csv_writer is None:
            return
        
        row = [f"{self.current_time:.2f}"]
        
        # Add intent probabilities (sorted by task name)
        for task_name in sorted(self.task_positions.keys()):
            prob = self.intention_probs.get(task_name, 0.0)
            row.append(f"{prob:.4f}")
        
        # Add completion status (sorted by task name)
        for task_name in sorted(self.task_positions.keys()):
            is_completed = 1 if task_name in self.completed_tasks else 0
            row.append(str(is_completed))
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # Write immediately to disk
    
    def _reset_intention_probs(self, current_position: np.ndarray):
        distances = self._calculate_distances(current_position)
        
        unnormalized_probs = {}
        for task_name, dist in distances.items():
            if task_name in self.completed_tasks:
                unnormalized_probs[task_name] = 0.0
            else:
                unnormalized_probs[task_name] = np.exp(-self.lambda_dist * dist)
        
        # Normalize probabilities
        total = sum(unnormalized_probs.values())
        if total > 0:
            self.intention_probs = {k: v/total for k, v in unnormalized_probs.items()}
        else:
            self.intention_probs = {k: 0.0 for k in self.task_positions.keys()}
    
    def _update_intention_probs(self, current_position: np.ndarray):
        distances = self._calculate_distances(current_position)
        
        if self.previous_position is None:
            self._reset_intention_probs(current_position)
            return
        
        # Compute movement direction
        movement_vector = current_position[:2] - self.previous_position[:2]
        movement_norm = np.linalg.norm(movement_vector)
        
        if movement_norm < 0.01:
            # Movement too small; keep previous probabilities
            return
        
        movement_direction = movement_vector / movement_norm
        
        # Compute probability for each task
        unnormalized_probs = {}
        for task_name, task_pos in self.task_positions.items():
            if task_name in self.completed_tasks:
                unnormalized_probs[task_name] = 0.0
            else:
                # Distance evidence
                dist = distances[task_name]
                distance_evidence = np.exp(-self.lambda_dist * dist)
                
                # Direction evidence
                to_task_vector = task_pos[:2] - current_position[:2]
                to_task_norm = np.linalg.norm(to_task_vector)
                
                if to_task_norm > 0.01:
                    to_task_direction = to_task_vector / to_task_norm
                    cos_theta = np.dot(movement_direction, to_task_direction)
                    direction_evidence = np.exp(self.gamma_dir * cos_theta)
                else:
                    direction_evidence = 1.0
                
                # Combine evidences
                unnormalized_probs[task_name] = distance_evidence * direction_evidence
        
        # Normalize probabilities
        total = sum(unnormalized_probs.values())
        if total > 0:
            self.intention_probs = {k: v/total for k, v in unnormalized_probs.items()}
    
    def _calculate_distances(self, current_position: np.ndarray) -> Dict[str, float]:
        distances = {}
        for task_name, task_pos in self.task_positions.items():
            # 2D Euclidean distance
            raw_dist = np.linalg.norm(current_position[:2] - task_pos[:2])
            # Apply scale factor
            distances[task_name] = raw_dist * self.distance_scale_factor
        return distances
    
    def _check_task_completion(self, current_position: np.ndarray) -> bool:
        task_just_completed = False
        
        for task_name, task_pos in self.task_positions.items():
            if task_name not in self.completed_tasks:
                # Note: use raw distance (no scaling) for completion check
                dist = np.linalg.norm(current_position[:2] - task_pos[:2])
                if dist <= self.task_completion_radius:
                    print(f"Reached {task_name}, reset intent probabilities.\n")
                    self.completed_tasks.add(task_name)
                    self._reset_intention_probs(current_position)
                    task_just_completed = True
        
        return task_just_completed
    
    def _perform_final_inference(self, current_position: np.ndarray):
        # Ensure there is an intention_probs dictionary (all zeros)
        if self.intention_probs is None:
            self.intention_probs = {k: 0.0 for k in self.task_positions.keys()}
        
        # Compute distances
        distances = self._calculate_distances(current_position)
        
        # Write to CSV (final record, all tasks completed)
        if self.enable_csv:
            self._write_to_csv()
        
        # Save to history
        self.inference_history.append({
            'time': self.current_time,
            'position': current_position.copy(),
            'distances': distances.copy(),
            'intention_probs': self.intention_probs.copy(),
            'completed_tasks': self.completed_tasks.copy()
        })
    
    def _all_tasks_completed(self) -> bool:
        return len(self.completed_tasks) >= len(self.task_positions)
    
    def get_predicted_intent(self) -> Optional[str]:
        if self.intention_probs is None:
            return None
        return max(self.intention_probs, key=self.intention_probs.get)
    
    def get_probabilities(self) -> Optional[Dict[str, float]]:
        return self.intention_probs
    
    def print_status(self):
        if self._all_tasks_completed():
            if not hasattr(self, '_all_done'):
                print("All subtasks completed.")
                self._all_done = True
            return
        
        if self.intention_probs is None:
            return
        
        # Output current status
        print(f"{'='*60}")
        
        intended_task = self.get_predicted_intent()
        print(f"Intent Estimation: {intended_task}")
        for task_name, prob in self.intention_probs.items():
            print(f"    {task_name}: {prob:.3f}")
        
        print(f"{'='*60}\n")
    
    def reset(self):
        self.intention_probs = None
        self.completed_tasks = set()
        self.previous_position = None
        self.last_inference_time = 0.0
        self.current_time = 0.0
        self.inference_history = []
    
    def get_history(self) -> List[Dict]:
        return self.inference_history
    
    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()
            print(f"CSV file saved: {self.csv_filename}")
            print(f"   Total records: {len(self.inference_history)}")
    
    def __del__(self):
        self.close()