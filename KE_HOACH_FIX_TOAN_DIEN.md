# 🔧 KẾ HOẠCH FIX TOÀN DIỆN - HỘI TỤ TARDINESS

## 🚨 **VẤN ĐỀ GỐC RỄ PHÁT HIỆN!**

### **Bug #1: REWARD FUNCTION KHÔNG BAO GỒM TARDINESS!**

```python
# scheduling_env.py, line 357-359 - HIỆN TẠI:
# Reward = -makespan (simple version)
cost = makespan
reward = -cost  # ❌ KHÔNG CÓ TARDINESS!
```

**Hậu quả**: PPO CHỈ tối ưu makespan, hoàn toàn KHÔNG HỌC để giảm tardiness!
Tardiness được TÍNH nhưng KHÔNG được sử dụng → PPO không có motivation để giảm tardiness.

---

## 📋 KẾ HOẠCH FIX TOÀN DIỆN - 7 BƯỚC

### **PHASE 1: CRITICAL FIXES (Bắt buộc)**

| # | Task | File | Impact |
|---|------|------|--------|
| 1 | **Fix Reward Function** | `scheduling_env.py` | 🔴 Critical |
| 2 | **Add Tardiness to State** | `scheduling_env.py` | 🔴 Critical |
| 3 | **Dynamic Entropy Schedule** | `lgp_coevolution_trainer.py` | 🟡 Important |

### **PHASE 2: OPTIMIZATION (Cải thiện)**

| # | Task | File | Impact |
|---|------|------|--------|
| 4 | **Protected Elite (Hall of Fame)** | `lgp_coevolution_trainer.py` | 🟡 Important |
| 5 | **Fix Learning Rate Decay** | `lgp_coevolution_trainer.py` | 🟢 Moderate |
| 6 | **Extend Forced Exploration** | `lgp_coevolution_trainer.py` | 🟢 Moderate |

### **PHASE 3: VALIDATION (Kiểm tra)**

| # | Task | File | Impact |
|---|------|------|--------|
| 7 | **Update Config & Metrics** | `config.py`, `visualize_metrics.py` | 🟢 Moderate |

---

## 📝 CHI TIẾT TỪNG BƯỚC

---

## **STEP 1: FIX REWARD FUNCTION** 🔴 CRITICAL

### **File**: `environment/scheduling_env.py`

### **Hiện tại** (Bug):
```python
# Line 357-359
# Reward = -makespan (simple version)
cost = makespan
reward = -cost
```

### **Đề xuất Fix**:
```python
# NEW REWARD FUNCTION - Multi-objective with tardiness focus
# Weights from EnvironmentConfig
w_makespan = 1.0
w_tardiness_normal = self.lambda_tardiness  # default 1.0
w_tardiness_urgent = self.lambda_tardiness * 2.0  # Urgent jobs more important

# Normalize tardiness by number of jobs (scale-invariant)
num_normal_jobs = sum(1 for j in self.all_jobs_info if isinstance(j, int))
num_urgent_jobs = len(self.all_jobs_info) - num_normal_jobs

avg_tardiness_normal = total_tardiness_normal / max(1, num_normal_jobs)
avg_tardiness_urgent = total_tardiness_urgent / max(1, num_urgent_jobs)

# Combined cost
cost = (w_makespan * makespan + 
        w_tardiness_normal * total_tardiness_normal + 
        w_tardiness_urgent * total_tardiness_urgent)

reward = -cost
```

### **Alternative - Bonus/Penalty System**:
```python
# Bonus for zero tardiness, penalty for late jobs
base_reward = -makespan

# Tardiness penalty (scaled)
tardiness_penalty = -(total_tardiness_normal + 2 * total_tardiness_urgent)

# Bonus for on-time delivery
on_time_bonus = 0
for job, info in self.all_jobs_info.items():
    job_events = [e for e in merged if e['job'] == job]
    if job_events:
        comp_time = max(e['finish'] for e in job_events)
        if comp_time <= info['due_date']:
            on_time_bonus += 10  # +10 for each on-time job

reward = base_reward + tardiness_penalty + on_time_bonus
```

---

## **STEP 2: ADD TARDINESS TO STATE** 🔴 CRITICAL

### **File**: `environment/scheduling_env.py`

### **Hiện tại**:
```python
# observation_space = (3,): [current_time, num_unfinished, avg_pt]
```

### **Đề xuất Fix**:
```python
# NEW STATE SPACE - Include tardiness info
# observation_space = (6,): 
#   [current_time, num_unfinished, avg_pt, 
#    slack_ratio, num_late_jobs, urgency_ratio]

def _get_state(self):
    """Get current state observation with tardiness info."""
    finished_events, unfinished_jobs = split_schedule_list(
        self.current_schedule_events, 
        self.current_time, 
        self.all_jobs_info
    )
    num_unfinished = sum(len(info['operations']) for info in unfinished_jobs.values())
    
    # Average processing time
    total_pt = 0
    count = 0
    for info in unfinished_jobs.values():
        for op in info['operations']:
            total_pt += op['processing_time']
            count += 1
    avg_pt = total_pt / count if count > 0 else 0
    
    # NEW: Tardiness-related features
    # 1. Slack ratio = (due_date - estimated_completion) / due_date
    total_slack_ratio = 0
    num_late = 0
    num_urgent = 0
    
    for job, info in unfinished_jobs.items():
        remaining_pt = sum(op['processing_time'] for op in info['operations'])
        estimated_completion = self.current_time + remaining_pt
        due_date = info['due_date']
        
        slack = due_date - estimated_completion
        slack_ratio = slack / max(1, due_date)
        total_slack_ratio += slack_ratio
        
        if slack < 0:
            num_late += 1
        
        if info.get('job_type', 'Normal') == 'Urgent':
            num_urgent += 1
    
    avg_slack_ratio = total_slack_ratio / max(1, len(unfinished_jobs))
    urgency_ratio = num_urgent / max(1, len(unfinished_jobs))
    
    return np.array([
        self.current_time,
        num_unfinished,
        avg_pt,
        avg_slack_ratio,      # NEW: negative = will be late
        num_late,             # NEW: how many jobs already late
        urgency_ratio         # NEW: proportion of urgent jobs
    ], dtype=np.float32)
```

### **Update observation_space**:
```python
# In __init__
self.observation_space = spaces.Box(low=-10, high=1000, shape=(6,), dtype=np.float32)
```

### **Update EnvironmentConfig**:
```python
# config.py
class EnvironmentConfig:
    obs_dim = 6  # Updated from 3
```

### **Update PPO model input**:
```python
# ppo_model.py - ActorCritic.__init__
# Automatically uses EnvironmentConfig.obs_dim
```

---

## **STEP 3: DYNAMIC ENTROPY SCHEDULE** 🟡 IMPORTANT

### **File**: `training/lgp_coevolution_trainer.py`

### **Đề xuất**:
```python
def get_dynamic_entropy_coef(generation, num_generations):
    """
    Dynamic entropy coefficient schedule.
    High at start (exploration), low at end (exploitation).
    """
    progress = generation / num_generations
    
    if progress <= 0.1:  # First 10% (Gen 1-2 for 20 gens)
        return 0.5  # High exploration for forced mode
    elif progress <= 0.5:  # 10-50% (Gen 3-10)
        return 0.3  # Maintain diversity
    elif progress <= 0.75:  # 50-75% (Gen 11-15)
        return 0.2  # Start converging
    else:  # Last 25% (Gen 16-20)
        return 0.15  # Final convergence
```

### **Integration**:
```python
# In train_with_coevolution_lgp(), before PPO update:
current_entropy = get_dynamic_entropy_coef(generation, num_generations)
# Temporarily override entropy_coef for this generation
original_entropy = PPOConfig.entropy_coef
PPOConfig.entropy_coef = current_entropy
# ... PPO update ...
PPOConfig.entropy_coef = original_entropy  # Restore

print(f"  Entropy coefficient: {current_entropy:.3f}")
```

---

## **STEP 4: PROTECTED ELITE (Hall of Fame)** 🟡 IMPORTANT

### **File**: `training/lgp_coevolution_trainer.py`

### **Đề xuất**:
```python
class HallOfFame:
    """
    Keeps track of best-ever programs across all generations.
    These programs are NEVER replaced.
    """
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.programs = []  # List of (fitness, program, generation)
    
    def update(self, program, fitness, generation):
        """Try to add a program to Hall of Fame."""
        # Check if this fitness is good enough
        if len(self.programs) < self.max_size:
            self.programs.append((fitness, copy.deepcopy(program), generation))
            self.programs.sort(key=lambda x: x[0], reverse=True)
            return True
        
        # Replace worst if better
        if fitness > self.programs[-1][0]:
            self.programs[-1] = (fitness, copy.deepcopy(program), generation)
            self.programs.sort(key=lambda x: x[0], reverse=True)
            return True
        
        return False
    
    def get_protected_indices(self, action_library):
        """Get indices of Hall of Fame programs in current library."""
        protected = []
        for fitness, program, gen in self.programs:
            # Find matching program in library (by gene equality)
            for i, lib_prog in enumerate(action_library):
                if self._programs_equal(program, lib_prog):
                    protected.append(i)
                    break
        return protected
    
    def _programs_equal(self, p1, p2):
        """Check if two programs are functionally equivalent."""
        if len(p1.genes) != len(p2.genes):
            return False
        for g1, g2 in zip(p1.genes, p2.genes):
            if g1.kind != g2.kind or g1.name != g2.name:
                return False
            if abs(g1.w_raw - g2.w_raw) > 0.01:
                return False
        return True
```

### **Integration trong evolution step**:
```python
# Initialize Hall of Fame at start
hall_of_fame = HallOfFame(max_size=5)

# In evolution step, BEFORE replacement:
# Update Hall of Fame with best program from this generation
best_idx = int(np.argmax(all_fitness))
best_fitness = all_fitness[best_idx]
hall_of_fame.update(action_library[best_idx], best_fitness, generation)

# Get protected indices
protected = set(elite_idx[:elite_size])  # Normal elite
hof_protected = set(hall_of_fame.get_protected_indices(action_library))
protected = protected.union(hof_protected)

# When selecting replacement targets, EXCLUDE protected
candidates_for_replacement = [i for i in range(pool_size) if i not in protected]
```

---

## **STEP 5: FIX LEARNING RATE DECAY** 🟢 MODERATE

### **File**: `training/lgp_coevolution_trainer.py`

### **Hiện tại**:
```python
decay_factor = 0.95  # Gen 20: LR = 0.000038 (TOO LOW!)
```

### **Đề xuất**:
```python
# Option A: Slower decay
decay_factor = 0.98  # Gen 20: LR = 0.000067

# Option B: Minimum LR floor
decay_factor = 0.95
min_lr = 5e-5  # Never go below this
current_lr = max(min_lr, initial_lr * (decay_factor ** generation))

# Option C: Cosine annealing
import math
min_lr = 1e-5
max_lr = initial_lr
current_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * generation / num_generations))
```

### **Recommended: Option B (simplest)**:
```python
# In train_with_coevolution_lgp()
min_lr = 5e-5  # Floor learning rate
decay_factor = 0.95
current_lr = max(min_lr, PPOConfig.learning_rate * (decay_factor ** generation))

for param_group in ppo_optimizer.param_groups:
    param_group['lr'] = current_lr
```

---

## **STEP 6: EXTEND FORCED EXPLORATION** 🟢 MODERATE

### **File**: `training/lgp_coevolution_trainer.py`

### **Hiện tại**:
```python
forced_exploration_gens = 2  # Only Gen 1-2
```

### **Đề xuất**:
```python
# Dynamic based on total generations
forced_exploration_gens = max(3, num_generations // 5)  # 10-20% of training
# For 20 gens: forced_exploration_gens = 4
```

### **Alternative - Probabilistic Forced**:
```python
# After initial forced period, occasionally force exploration
if generation <= forced_exploration_gens:
    # Full forced exploration
    force_explore = True
elif generation <= forced_exploration_gens * 2:
    # 50% chance forced
    force_explore = random.random() < 0.5
else:
    # 10% chance forced (for diversity maintenance)
    force_explore = random.random() < 0.1
```

---

## **STEP 7: UPDATE CONFIG & METRICS** 🟢 MODERATE

### **File**: `config.py`

### **Updates**:
```python
class EnvironmentConfig:
    obs_dim = 6  # Updated for new state space
    
    # NEW: Tardiness weights for reward function
    w_makespan = 1.0
    w_tardiness_normal = 1.0
    w_tardiness_urgent = 2.0  # Urgent jobs more important
    
    # NEW: On-time bonus
    on_time_bonus = 10.0
```

### **File**: `analysis/visualize_metrics.py`

### **Add tardiness convergence plot**:
```python
def plot_tardiness_convergence(metrics_data):
    """Plot tardiness over generations to show convergence."""
    generations = []
    tardiness_normal = []
    tardiness_urgent = []
    
    for gen, data in sorted(metrics_data.items()):
        generations.append(gen)
        tardiness_normal.append(data['aggregated_metrics']['avg_tardiness_normal'])
        tardiness_urgent.append(data['aggregated_metrics']['avg_tardiness_urgent'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, tardiness_normal, 'b-o', label='Normal Jobs', linewidth=2)
    plt.plot(generations, tardiness_urgent, 'r-s', label='Urgent Jobs', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Average Tardiness')
    plt.title('Tardiness Convergence Over Generations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add convergence indicator
    if len(tardiness_normal) > 3:
        recent_avg = np.mean(tardiness_normal[-3:])
        if recent_avg < 1.0:
            plt.axhline(y=recent_avg, color='g', linestyle='--', 
                       label=f'Converged: {recent_avg:.2f}')
    
    plt.savefig('results/plots/tardiness_convergence.png', dpi=150)
    plt.close()
```

---

## 📊 EXPECTED RESULTS

### **Before Fix (Current)**:
| Metric | Gen 1 | Gen 20 | Trend |
|--------|-------|--------|-------|
| Makespan | 166.31 | 158.08 | ✅ -5% |
| Tardiness Normal | 0.0175 | 0.0935 | ❌ +434% |
| Tardiness Urgent | 0.0150 | 0.0130 | ➡️ Flat |

### **After Fix (Expected)**:
| Metric | Gen 1 | Gen 20 | Trend |
|--------|-------|--------|-------|
| Makespan | ~170 | ~155 | ✅ -9% |
| Tardiness Normal | ~0.02 | **<0.01** | ✅ -50% |
| Tardiness Urgent | ~0.02 | **<0.01** | ✅ -50% |

---

## 🎯 IMPLEMENTATION ORDER

### **Day 1: Critical Fixes**
1. ✅ STEP 1: Fix Reward Function
2. ✅ STEP 2: Add Tardiness to State
3. ⏳ Run quick test (3 generations)

### **Day 2: Optimization**
4. ✅ STEP 3: Dynamic Entropy
5. ✅ STEP 5: Fix LR Decay
6. ⏳ Run medium test (10 generations)

### **Day 3: Advanced Features**
7. ✅ STEP 4: Hall of Fame
8. ✅ STEP 6: Extend Forced Exploration
9. ✅ STEP 7: Update Config & Metrics
10. ⏳ Run full training (20 generations)

---

## 🛡️ SAFETY CHECKLIST

Before implementing, verify:

- [ ] Backup current `scheduling_env.py`
- [ ] Backup current `lgp_coevolution_trainer.py`
- [ ] Backup current `config.py`
- [ ] Backup current `ppo_model.py`
- [ ] Clear `results/` folder (or create new folder)

After implementing:

- [ ] Test with 3 generations first
- [ ] Check reward values are reasonable (not too large/small)
- [ ] Check state values are normalized (~0-1 range)
- [ ] Verify tardiness appears in training logs
- [ ] Check PolicyLoss is NOT spiking
- [ ] Monitor GPU/CPU memory

---

## 📝 CODE SNIPPETS READY TO COPY

### **snippet_1_reward_function.py**
```python
# Replace in scheduling_env.py, step() method, after line 355

# Multi-objective reward with tardiness
w_makespan = 1.0
w_tardiness_normal = self.lambda_tardiness
w_tardiness_urgent = self.lambda_tardiness * 2.0

cost = (w_makespan * makespan + 
        w_tardiness_normal * total_tardiness_normal + 
        w_tardiness_urgent * total_tardiness_urgent)

reward = -cost
```

### **snippet_2_state_space.py**
```python
# Replace _get_state() method in scheduling_env.py

def _get_state(self):
    """Get current state observation with tardiness info."""
    finished_events, unfinished_jobs = split_schedule_list(
        self.current_schedule_events, 
        self.current_time, 
        self.all_jobs_info
    )
    num_unfinished = sum(len(info['operations']) for info in unfinished_jobs.values())
    
    total_pt = 0
    count = 0
    for info in unfinished_jobs.values():
        for op in info['operations']:
            total_pt += op['processing_time']
            count += 1
    avg_pt = total_pt / count if count > 0 else 0
    
    # Tardiness features
    total_slack_ratio = 0
    num_late = 0
    num_urgent = 0
    
    for job, info in unfinished_jobs.items():
        remaining_pt = sum(op['processing_time'] for op in info['operations'])
        estimated_completion = self.current_time + remaining_pt
        due_date = info['due_date']
        
        slack = due_date - estimated_completion
        slack_ratio = slack / max(1, due_date)
        total_slack_ratio += slack_ratio
        
        if slack < 0:
            num_late += 1
        
        if info.get('job_type', 'Normal') == 'Urgent':
            num_urgent += 1
    
    n_jobs = max(1, len(unfinished_jobs))
    avg_slack_ratio = total_slack_ratio / n_jobs
    urgency_ratio = num_urgent / n_jobs
    
    return np.array([
        self.current_time,
        num_unfinished,
        avg_pt,
        avg_slack_ratio,
        num_late,
        urgency_ratio
    ], dtype=np.float32)
```

---

## 🎬 BẮT ĐẦU?

Bạn muốn tôi implement theo thứ tự nào?

**Recommend**: Bắt đầu với **STEP 1 + STEP 2** (Critical) trước, chạy test nhanh, rồi tiếp tục các bước còn lại.

Xác nhận để tôi bắt đầu implement! 🚀
