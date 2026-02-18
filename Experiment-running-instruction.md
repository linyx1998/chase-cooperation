# IROS Experiment Running Instruction

## **1. How to Run the Experiment**

To run the experiment, use:

```bash
python main_coop_experiment.py \
--mode proposed \
--tasks_id 1 \
--case R1 \
--commit_distance 3.0
```

---

### **Parameter Description**

- mode
  
    Available options:
    
    - proposed
    - independent → Greedy nearest-task drone, which does not use human intent
- tasks_id (layout indicator)
  
    Available options:
    
    - 1
    - 2
    - 3
    - real
    
    The real option corresponds to Cheng’s real-world environment layout.
    
- case
  
    Available options:
    
    - R1
    - R2
    - A
    
    These correspond to:
    
    - Two rational human behavior cases (R1, R2)
    - One ambiguous case (A)
    
    The detailed execution protocol for each case is explained in the next section.
    
- commit_distance
  
    This parameter controls the commitment threshold for the drone. 
    For layouts 1, 2, and 3, we use:
    
    ```bash
    commit_distance = 1.5
    ```
    
    For the real layout, we use:
    
    ```bash
    commit_distance = 1.0
    ```
    
    The reason is that the real-world environment is more compact, so a smaller commitment distance is more appropriate.
    

---

### **Logging**

The selected mode, tasks_id, and case will be reflected in the generated CSV filenames.

Each run automatically creates a timestamped folder under ./logs/.

## 2. **Human Behavior Protocols**

### **General Principles (Applicable to All Cases)**

For both proposed and independent modes, it would be appreciated if the human operator could follow the predefined target order as strictly as possible, while allowing the drone to adapt autonomously.

The following rules apply to all cases:

1. The human should attempt to execute the target order specified by the case.
2. If the human is heading toward a task and the drone also approaches it, the human should continue moving toward the task until it is completed.
3. When the human arrives at the dummy location, a timeout indicator will be displayed. The expected behavior is that the human does not wait unnecessarily. If the timeout is triggered (i.e., the drone does not arrive in time), the human should leave the dummy and proceed to the next predefined task.
4. The human should avoid switching targets unless explicitly specified in the case definition (e.g., ambiguous cases).
5. For ambiguous cases, if there are remaining tasks that are not explicitly specified in the predefined sequence, the human should complete them according to the nearest-task principle.

---

### Case Definitions

1. tasks 1
    1. R1: Black toolbox → Dummy → Blue toolbox
    2. R2: Dummy → Black toolbox → Blue toolbox
    3. A: Dummy → Halfway redirect to Black toolbox → Back to Dummy → Blue toolbox
2. tasks 2
    1. R1: Blue toolbox → Dummy → Black toolbox
    2. R2: Dummy → Blue toolbox → Black toolbox
    3. A: Black toolbox → Halfway redirect to Blue toolbox → Dummy
3. tasks 3
    1. R1: Blue toolbox → Dummy → Black toolbox
    2. R2: Dummy → Black toolbox → Blue toolbox
    3. A: Blue toolbox → Halfway redirect to Dummy → Back to Blue toolbox
4. tasks 4
    1. R1: Black toolbox → Dummy → Blue toolbox
    2. R2: Dummy → Blue toolbox → Black toolbox
    3. A: Black toolbox → Halfway redirect to Dummy → Blue toolbox → Black toolbox