import numpy as np
from typing import Dict, Set, List, Optional, FrozenSet, Tuple
from itertools import permutations, product, combinations


class RecommendationManager:
    def __init__(
        self,
        task_positions: Dict[str, np.ndarray],
        task_types: Dict[str, str],
        task_rewards: Dict[str, float],
        distance_penalty: float = 0.5,
        intent_threshold: float = 0.3
    ):
        """
        Args:
            task_positions: Dictionary mapping task names to positions
            task_types: Dictionary mapping task names to types ('independent' or 'cooperative')
            task_rewards: Dictionary mapping task names to base rewards
            distance_penalty: Penalty coefficient for distance
            intent_threshold: Minimum intent probability to consider a task
        """
        self.task_positions = task_positions
        self.task_types = task_types
        self.task_rewards = task_rewards
        self.distance_penalty = distance_penalty
        self.intent_threshold = intent_threshold
        
        # Offline: generate all possible sequences for each completion status
        self.sequence_map = self._generate_sequence_map()
        
        print(f"RecommendationManager initialized:")
        print(f"  - Distance penalty: {distance_penalty}")
        print(f"  - Intent threshold: {intent_threshold}")
        print(f"  - Generated sequences for {len(self.sequence_map)} completion states")
    
    def _generate_sequence_map(self) -> Dict[FrozenSet[str], List[List[Dict]]]:
        """
        Generate all possible task sequences for each completion status.
        
        Returns:
            Dictionary mapping completion status (frozenset) to list of possible sequences
            
        Example:
            {
                frozenset(): [all sequences when nothing is completed],
                frozenset({'box1'}): [sequences when box1 is completed],
                ...
            }
        """
        sequence_map = {}
        all_tasks = list(self.task_positions.keys())
        
        # Generate all possible completion statuses (power set)
        all_statuses = []
        for r in range(len(all_tasks) + 1):
            for subset in combinations(all_tasks, r):
                all_statuses.append(frozenset(subset))
        
        # For each completion status, generate possible sequences for remaining tasks
        for completed in all_statuses:
            remaining = set(all_tasks) - set(completed)
            
            if not remaining:
                # All tasks completed
                sequence_map[completed] = []
                continue
            
            sequences = self._generate_sequences_for_tasks(remaining)
            sequence_map[completed] = sequences
            
        return sequence_map
    
    def _generate_sequences_for_tasks(self, tasks: Set[str]) -> List[List[Dict]]:
        """
        Generate all possible sequences for a given set of tasks.
        
        Args:
            tasks: Set of task names to complete
            
        Returns:
            List of sequences, where each sequence is a list of steps
            Each step is {"task": task_name, "agent": "H"/"D"/"H+D"}
        """
        sequences = []
        
        # Separate independent and cooperative tasks
        independent = [t for t in tasks if self.task_types[t] == "independent"]
        cooperative = [t for t in tasks if self.task_types[t] == "cooperative"]
        
        # Generate all agent assignments for independent tasks
        if independent:
            # Each independent task can be assigned to H or D
            agent_assignments = product(["H", "D"], repeat=len(independent))
        else:
            agent_assignments = [()]
        
        for assignment in agent_assignments:
            # Build task list with agent assignments
            task_list = []
            
            for i, task in enumerate(independent):
                task_list.append({"task": task, "agent": assignment[i]})
            
            for task in cooperative:
                # Cooperative tasks require both agents
                task_list.append({"task": task, "agent": "H+D"})
            
            # Generate all permutations (different orderings)
            for perm in permutations(task_list):
                sequences.append(list(perm))
        
        return sequences
    
    def get_recommendation(
        self,
        human_intent: Dict[str, float],
        completed_tasks: Set[str],
        human_position: np.ndarray,
        drone_position: np.ndarray
    ) -> Optional[Dict]:
        """
        Get task recommendation based on current state.
        
        Args:
            human_intent: Dictionary of intent probabilities for each task
            completed_tasks: Set of already completed task names
            human_position: Current human position
            drone_position: Current drone position
            
        Returns:
            Dictionary containing:
                - "task": recommended task name
                - "expected_reward": expected cooperative reward
                - "intent": human intent probability for this task
            Returns None if no recommendation can be made
        """
        # Query sequences for current completion status
        completed_key = frozenset(completed_tasks)
        possible_sequences = self.sequence_map.get(completed_key, [])
        
        if not possible_sequences:
            return None
        
        # Evaluate each possible sequence
        evaluations = []
        for seq in possible_sequences:
            # Find the next task for human in this sequence
            next_h_task = self._find_next_human_task(seq)
            
            if next_h_task is None:
                continue
            
            # Check intent threshold
            if human_intent.get(next_h_task, 0) <= self.intent_threshold:
                continue
            
            # Calculate expected reward for this sequence
            reward = self._calculate_sequence_reward_imm(
                seq, human_position, drone_position
            )
            
            # Calculate expected future distances for both agents
            _, h_distance, d_distance = self._calculate_sequence_reward(
                seq, human_position, drone_position
            )
            
            evaluations.append({
                "sequence": seq,
                "next_human_task": next_h_task,
                "reward": reward,
                "intent": human_intent[next_h_task],
                "h_distance": h_distance,
                "d_distance": d_distance
            })
        
        if not evaluations:
            return None
        
        # Select sequence with highest expected reward
        # best = max(evaluations, key=lambda x: x["reward"])
        
        # Select best: prioritize reward, use intent as tie-breaker
        best = max(evaluations, key=lambda x: (x["reward"], x["intent"], x['d_distance']))
        
        return {
            "task": best["next_human_task"],
            "expected_reward": best["reward"],
            "intent": best["intent"],
            "h_distance": best["h_distance"],
            "d_distance": best["d_distance"]
        }
    
    def _find_next_human_task(self, sequence: List[Dict]) -> Optional[str]:
        """
        Find the next task that involves the human in a sequence.
        
        Args:
            sequence: List of steps (each step is {"task": ..., "agent": ...})
            
        Returns:
            Task name or None if no human task found
        """
        for step in sequence:
            if step["agent"] in ["H", "H+D"]:
                return step["task"]
        return None
    
    def _calculate_sequence_reward_imm(
        self,
        sequence: List[Dict],
        human_position: np.ndarray,
        drone_position: np.ndarray
    ) -> float:
        """
        Calculate the immediate reward for the first step of a sequence (greedy).
        
        Reward = base_reward - alpha * distance_for_first_step
        
        Args:
            sequence: List of steps to complete
            human_position: Starting position of human
            drone_position: Starting position of drone
            
        Returns:
            Immediate reward for the first step (float)
        """
        if not sequence:
            return 0.0
        
        # Only consider the first step
        first_step = sequence[0]
        task = first_step["task"]
        agent = first_step["agent"]
        task_pos = self.task_positions[task][:2]
        
        base_reward = self.task_rewards[task]
        
        if agent == "H":
            # Human completes this task
            distance = np.linalg.norm(human_position[:2] - task_pos)
            return base_reward - self.distance_penalty * distance
            
        elif agent == "D":
            # Drone completes this task
            distance = np.linalg.norm(drone_position[:2] - task_pos)
            return base_reward - self.distance_penalty * distance
            
        elif agent == "H+D":
            # Cooperative task: both agents move to location
            h_distance = np.linalg.norm(human_position[:2] - task_pos)
            d_distance = np.linalg.norm(drone_position[:2] - task_pos)
            total_distance = h_distance + d_distance
            return base_reward - self.distance_penalty * total_distance
        
        return 0.0
    
    def _calculate_sequence_reward(
        self,
        sequence: List[Dict],
        human_position: np.ndarray,
        drone_position: np.ndarray
    ) -> float:
        """
        Calculate the cooperative reward for completing a sequence.
        
        Reward = Sum(base_rewards) - alpha * (human_distance + drone_distance)
        
        Args:
            sequence: List of steps to complete
            human_position: Starting position of human
            drone_position: Starting position of drone
            
        Returns:
            Total cooperative reward (float)
        """
        h_pos = human_position[:2].copy()
        d_pos = drone_position[:2].copy()
        
        h_distance = 0.0
        d_distance = 0.0
        base_reward = 0.0
        
        for step in sequence:
            task = step["task"]
            agent = step["agent"]
            task_pos = self.task_positions[task][:2]
            
            if agent == "H":
                # Human completes this task
                h_distance += np.linalg.norm(h_pos - task_pos)
                h_pos = task_pos
                base_reward += self.task_rewards[task]
                
            elif agent == "D":
                # Drone completes this task
                d_distance += np.linalg.norm(d_pos - task_pos)
                d_pos = task_pos
                base_reward += self.task_rewards[task]
                
            elif agent == "H+D":
                # Cooperative task: both agents move to location
                h_distance += np.linalg.norm(h_pos - task_pos)
                d_distance += np.linalg.norm(d_pos - task_pos)
                h_pos = task_pos
                d_pos = task_pos
                base_reward += self.task_rewards[task]
        
        # Apply distance penalty
        total_reward = base_reward - self.distance_penalty * (h_distance + d_distance)
        
        return total_reward, h_distance, d_distance
    
    def get_sequence_count(self, completed_tasks: Set[str] = None) -> int:
        """
        Get the number of possible sequences for a given completion status.
        
        Args:
            completed_tasks: Set of completed task names (None for initial state)
            
        Returns:
            Number of possible sequences
        """
        if completed_tasks is None:
            completed_tasks = set()
        
        completed_key = frozenset(completed_tasks)
        sequences = self.sequence_map.get(completed_key, [])
        return len(sequences)