import numpy as np
import csv
import os
import datetime
from typing import Dict, Set, Optional, Tuple, List

class IntentInferenceSystem:
    """
    Intent inference system for human agents.
    - Reset: p_k(j) ∝ exp(-λ dist(j))
    - Update: p_k(j) ∝ exp(-λ dist(j)) × exp(γ cos(θ_j))
    """
    
    def __init__(
        self,
        task_positions: Dict[str, np.ndarray],
        task_types: Dict[str, str],  # 'independent' or 'cooperative'
        lambda_dist: float = 0.3,
        gamma_dir: float = 1.0,
        task_completion_radius: float = 2.0,
        distance_scale_factor: float = 1.0,
        inference_interval: float = 1.0,
        enable_csv: bool = True,
        csv_file_path: Optional[str] = None
    ):
        
        self.task_positions = task_positions
        self.task_types = task_types
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
        
        # Track which agents are at cooperative tasks
        self.agents_at_cooperative_tasks: Dict[str, Set[str]] = {
            task: set() for task, typ in task_types.items() if typ == 'cooperative'
        }
        
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
        
        header = ['Time']
        for task_name in sorted(self.task_positions.keys()):
            header.append(f'Intent_{task_name}')
        for task_name in sorted(self.task_positions.keys()):
            header.append(f'Completed_{task_name}')
        
        self.csv_writer.writerow(header)
        self.csv_file.flush()
        
    def update(
        self,
        current_position: np.ndarray,
        current_time: float,
        agent_id: str = 'car',
        force_inference: bool = False
    ) -> Optional[Dict[str, float]]:
        self.current_time = current_time
        
        task_just_completed = self._check_task_completion(current_position, agent_id)
        
        should_infer = (
            force_inference or 
            task_just_completed or
            (self.current_time - self.last_inference_time >= self.inference_interval)
        )
        
        if self._all_tasks_completed():
            if not hasattr(self, '_final_record_written'):
                self._perform_final_inference(current_position)
                self._final_record_written = True
            if self.previous_position is None:
                self.previous_position = current_position.copy()
            return None
        
        if should_infer:
            self._perform_inference(current_position)
            self.last_inference_time = self.current_time
            self.previous_position = current_position.copy()
            return self.intention_probs
        
        if self.previous_position is None:
            self.previous_position = current_position.copy()
        
        return None
    
    def _perform_inference(self, current_position: np.ndarray):
        if self.intention_probs is None:
            self._reset_intention_probs(current_position)
        else:
            self._update_intention_probs(current_position)
        
        distances = self._calculate_distances(current_position)
        
        if self.enable_csv:
            self._write_to_csv()
        
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
        for task_name in sorted(self.task_positions.keys()):
            prob = self.intention_probs.get(task_name, 0.0)
            row.append(f"{prob:.4f}")
        for task_name in sorted(self.task_positions.keys()):
            is_completed = 1 if task_name in self.completed_tasks else 0
            row.append(str(is_completed))
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
    
    def _reset_intention_probs(self, current_position: np.ndarray):
        distances = self._calculate_distances(current_position)
        
        unnormalized_probs = {}
        for task_name, dist in distances.items():
            if task_name in self.completed_tasks:
                unnormalized_probs[task_name] = 0.0
            else:
                unnormalized_probs[task_name] = np.exp(-self.lambda_dist * dist)
        
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
        
        movement_vector = current_position[:2] - self.previous_position[:2]
        movement_norm = np.linalg.norm(movement_vector)
        
        if movement_norm < 0.01:
            return
        
        movement_direction = movement_vector / movement_norm
        
        unnormalized_probs = {}
        for task_name, task_pos in self.task_positions.items():
            if task_name in self.completed_tasks:
                unnormalized_probs[task_name] = 0.0
            else:
                dist = distances[task_name]
                distance_evidence = np.exp(-self.lambda_dist * dist)
                
                to_task_vector = task_pos[:2] - current_position[:2]
                to_task_norm = np.linalg.norm(to_task_vector)
                
                if to_task_norm > 0.01:
                    to_task_direction = to_task_vector / to_task_norm
                    cos_theta = np.dot(movement_direction, to_task_direction)
                    direction_evidence = np.exp(self.gamma_dir * cos_theta)
                else:
                    direction_evidence = 1.0
                
                unnormalized_probs[task_name] = distance_evidence * direction_evidence
        
        total = sum(unnormalized_probs.values())
        if total > 0:
            self.intention_probs = {k: v/total for k, v in unnormalized_probs.items()}
    
    def _calculate_distances(self, current_position: np.ndarray) -> Dict[str, float]:
        distances = {}
        for task_name, task_pos in self.task_positions.items():
            raw_dist = np.linalg.norm(current_position[:2] - task_pos[:2])
            distances[task_name] = raw_dist * self.distance_scale_factor
        return distances
    
    def _check_task_completion(self, current_position: np.ndarray, agent_id: str) -> bool:
        """Check task completion with support for cooperative tasks"""
        task_just_completed = False
        
        for task_name, task_pos in self.task_positions.items():
            if task_name not in self.completed_tasks:
                dist = np.linalg.norm(current_position[:2] - task_pos[:2])
                
                if dist <= self.task_completion_radius:
                    task_type = self.task_types[task_name]
                    
                    if task_type == 'independent':
                        print(f"{agent_id} reached {task_name} (independent), task completed!\n")
                        self.completed_tasks.add(task_name)
                        self._reset_intention_probs(current_position)
                        task_just_completed = True
                        
                    elif task_type == 'cooperative':
                        self.agents_at_cooperative_tasks[task_name].add(agent_id)
                        
                        if len(self.agents_at_cooperative_tasks[task_name]) >= 2:
                            print(f"Both agents at {task_name}, task completed!\n")
                            self.completed_tasks.add(task_name)
                            self._reset_intention_probs(current_position)
                            task_just_completed = True
                else:
                    # Agent left the cooperative task area
                    if task_name in self.agents_at_cooperative_tasks:
                        if agent_id in self.agents_at_cooperative_tasks[task_name]:
                            self.agents_at_cooperative_tasks[task_name].remove(agent_id)
        
        return task_just_completed
    
    def _perform_final_inference(self, current_position: np.ndarray):
        if self.intention_probs is None:
            self.intention_probs = {k: 0.0 for k in self.task_positions.keys()}
        
        distances = self._calculate_distances(current_position)
        
        if self.enable_csv:
            self._write_to_csv()
        
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
        self.agents_at_cooperative_tasks = {
            task: set() for task, typ in self.task_types.items() if typ == 'cooperative'
        }
    
    def get_history(self) -> List[Dict]:
        return self.inference_history
    
    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()
            print(f"CSV file saved: {self.csv_filename}")
            print(f"   Total records: {len(self.inference_history)}")
    
    def __del__(self):
        self.close()


class DroneCooperationAgent:
    """
    Drone cooperation agent that responds to human intent.
    
    Logic (simplified for 2 boxes + 1 dummy scenario):
    - If human's max intent > threshold and it's dummy -> go to dummy
    - If human's max intent > threshold and it's a box -> go to the OTHER box
    - Threshold must be >= 0.5 to avoid ambiguity
    """
    
    def __init__(
        self,
        task_positions: Dict[str, np.ndarray],
        task_types: Dict[str, str],
        initial_position: np.ndarray,
        speed: float = 1.5,
        intent_threshold: float = 0.5,
        commitment_distance: float = 4.0,
        dummy_wait_timeout: float = 10.0
    ):
        """
        Args:
            task_positions: Dictionary of task names to positions
            task_types: Dictionary of task names to types ('independent' or 'cooperative')
            initial_position: Starting position of the drone
            speed: Movement speed (m/s)
            intent_threshold: Threshold for reacting to human intent (must be >= 0.5)
            commitment_distance: Distance within which drone commits to completing task (m)
            dummy_wait_timeout: Max time to wait at dummy before choosing freely (seconds)
        """
        if intent_threshold < 0.5:
            raise ValueError(f"intent_threshold must be >= 0.5, got {intent_threshold}")
        
        self.task_positions = task_positions
        self.task_types = task_types
        self.position = initial_position.copy()
        self.speed = speed
        self.intent_threshold = intent_threshold
        self.commitment_distance = commitment_distance
        self.dummy_wait_timeout = dummy_wait_timeout
        
        # Identify box tasks (independent tasks)
        self.box_tasks = [name for name, typ in task_types.items() if typ == 'independent']
        self.cooperative_task = [name for name, typ in task_types.items() if typ == 'cooperative'][0]
        
        if len(self.box_tasks) != 2:
            raise ValueError(f"Expected exactly 2 box tasks, got {len(self.box_tasks)}")
        
        # Current target
        self.target: Optional[str] = None
        self.dummy_arrival_time: Optional[float] = None
        self.dummy_timeout_cooldown_until: Optional[float] = None
        
        print(f"DroneCooperationAgent initialized:")
        print(f"  - Speed: {speed} m/s")
        print(f"  - Intent threshold: {intent_threshold}")
        print(f"  - Commitment distance: {commitment_distance} m")
        print(f"  - Dummy wait timeout: {dummy_wait_timeout} s")
        print(f"  - Box tasks: {self.box_tasks}")
        print(f"  - Cooperative task: {self.cooperative_task}")
    
    def initialize_target(self, human_intent_probs: Dict[str, float], completed_tasks: Set[str] = None) -> str:
        """
        Initialize drone target.
        
        Args:
            human_intent_probs: Human's intent probabilities
            completed_tasks: Set of already completed tasks (usually empty at init)
            
        Returns:
            Initial target task name
        """
        if completed_tasks is None:
            completed_tasks = set()
        
        # Default: go to dummy (waiting for cooperation)
        self.target = self.cooperative_task
        print(f"Drone initial target: {self.target} (default)")
        
        return self.target
    
    def update_target(
        self, 
        human_intent_probs: Dict[str, float],
        completed_tasks: Set[str],
        current_time: float
    ) -> Optional[str]:
        """
        Update drone target based on current human intent.
        """
        if not human_intent_probs:
            return self.target
        
        # Check if current target is completed
        if self.target in completed_tasks:
            uncompleted = [name for name in self.task_positions.keys() 
                          if name not in completed_tasks]
            if not uncompleted:
                self.target = None
                self.dummy_arrival_time = None
                return None
            
            self.target = np.random.choice(uncompleted)
            self.dummy_arrival_time = None
            print(f"Target completed, switching to: {self.target}")
            return self.target
        
        # Track arrival at dummy
        if self.target == self.cooperative_task:
            target_pos = self.task_positions[self.target][:2]
            dist = np.linalg.norm(self.position[:2] - target_pos)
            if dist < 0.5:
                if self.dummy_arrival_time is None:
                    self.dummy_arrival_time = current_time
            else:
                self.dummy_arrival_time = None
        else:
            self.dummy_arrival_time = None
        
        # Check if drone is committed to current box
        is_committed = False
        if self.target and self.task_types.get(self.target) == 'independent':
            target_pos = self.task_positions[self.target][:2]
            dist = np.linalg.norm(self.position[:2] - target_pos)
            if dist < self.commitment_distance:
                is_committed = True
        
        # Find human's max intent
        max_intent_task = max(human_intent_probs, key=human_intent_probs.get)
        max_intent_prob = human_intent_probs[max_intent_task]
        
        # Check dummy timeout
        if (self.target == self.cooperative_task and 
            self.dummy_arrival_time is not None and 
            current_time - self.dummy_arrival_time > self.dummy_wait_timeout):
            available_boxes = [b for b in self.box_tasks if b not in completed_tasks]
            if available_boxes:
                if len(available_boxes) == 1:
                    self.target = available_boxes[0]
                else:
                    human_intended_boxes = [b for b in available_boxes 
                                          if human_intent_probs.get(b, 0) > self.intent_threshold]
                    if human_intended_boxes:
                        other_boxes = [b for b in available_boxes if b not in human_intended_boxes]
                        self.target = other_boxes[0] if other_boxes else np.random.choice(available_boxes)
                    else:
                        self.target = np.random.choice(available_boxes)
                
                self.dummy_arrival_time = None
                self.dummy_timeout_cooldown_until = current_time + 5.0
                return self.target
        
        # Only react if above threshold
        if max_intent_prob > self.intent_threshold:
            
            # Case 1: Human wants dummy
            if max_intent_task == self.cooperative_task:
                if is_committed:
                    return self.target
                
                # Check cooldown
                if (self.dummy_timeout_cooldown_until is not None and 
                    current_time < self.dummy_timeout_cooldown_until):
                    return self.target
                
                if self.target != self.cooperative_task:
                    print(f"Human -> {max_intent_task}, Drone: {self.target} -> {self.cooperative_task}")
                    self.target = self.cooperative_task
                    self.dummy_timeout_cooldown_until = None
                return self.target
            
            # Case 2: Human wants a box
            elif max_intent_task in self.box_tasks:
                
                # If committed and not conflicting, keep target
                if is_committed and self.target != max_intent_task:
                    return self.target
                
                # Find other available boxes (not what human wants, not completed)
                other_boxes = [b for b in self.box_tasks 
                              if b != max_intent_task and b not in completed_tasks]
                
                if other_boxes:
                    # Has other boxes, pick random one
                    new_target = np.random.choice(other_boxes)
                    if self.target != new_target:
                        print(f"Human -> {max_intent_task}, Drone: {self.target} -> {new_target}")
                        if self.target == self.cooperative_task:
                            self.dummy_arrival_time = None
                        self.target = new_target
                
                elif self.cooperative_task not in completed_tasks:
                    if (self.dummy_timeout_cooldown_until is not None and 
                        current_time < self.dummy_timeout_cooldown_until):
                        pass
                    elif self.target != self.cooperative_task:
                        self.target = self.cooperative_task
                
                else:
                    # Only one task left (the box human wants), go compete
                    if self.target != max_intent_task:
                        if self.target == self.cooperative_task:
                            self.dummy_arrival_time = None
                        self.target = max_intent_task
                
                return self.target
        
        return self.target
    
    def get_target_position(self) -> Optional[np.ndarray]:
        """
        Get the position coordinates of the current target.
        
        Returns:
            Target position as numpy array, or None if no target
        """
        if self.target is None:
            return None
        return self.task_positions[self.target].copy()
    
    def move_towards_target(self, delta_time: float) -> np.ndarray:
        """
        Move drone towards its target at constant speed.
        
        Args:
            delta_time: Time elapsed since last update (seconds)
            
        Returns:
            Updated position
        """
        if self.target is None:
            return self.position
        
        target_pos = self.task_positions[self.target][:2]
        to_target = target_pos - self.position[:2]
        dist_to_target = np.linalg.norm(to_target)
        
        # Move if not already at target
        if dist_to_target > 0.1:
            direction = to_target / dist_to_target
            move_distance = min(self.speed * delta_time, dist_to_target)
            self.position[0] += direction[0] * move_distance
            self.position[1] += direction[1] * move_distance
        
        return self.position
    
    def get_position(self) -> np.ndarray:
        """Get current position"""
        return self.position.copy()
    
    def get_target(self) -> Optional[str]:
        """Get current target task name"""
        return self.target