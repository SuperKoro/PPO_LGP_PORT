# GIẢI THÍCH CHI TIẾT PROJECT PPO + LGP - PHẦN 4: PPO & COEVOLUTION

## 📋 MỤC LỤC PHẦN NÀY
1. PPO Agent là gì?
2. PPO Architecture  
3. Coevolution - PPO + LGP cùng tiến hóa
4. Training Loop chi tiết
5. Tổng kết toàn bộ pipeline

---

## 4.1. PPO AGENT LÀ GÌ?

### **Reinforcement Learning cơ bản:**

**Mục tiêu:** Train một "agent" (AI) học **chọn action tốt nhất** cho mỗi state

```
State → [AGENT] → Action → Reward
```

**Ví dụ game Mario:**
```
State: Mario đứng trước hố
  ↓
Agent quyết định: NHẢY!
  ↓
Action: Jump
  ↓  
Reward: +10 (qua được hố)
```

**Trong project này:**
```
State: [current_time=120, num_jobs=35, avg_pt=8.5]
  ↓
PPO Agent quyết định: Chọn Portfolio #23
  ↓
Action: Run Portfolio #23 (EDD | SA:60%, GA:30%, PSO:10%)
  ↓
Reward: -makespan (vd: -180)
```

---

### **PPO (Proximal Policy Optimization)**

**PPO = Một thuật toán RL hiện đại, ổn định, hiệu quả**

**Tại sao dùng PPO?**
- ✅ Stable (ổn định hơn vanilla policy gradient)
- ✅ Sample efficient (học nhanh)
- ✅ Dễ implement
- ✅ Widely used (OpenAI, DeepMind dùng)

---

## 4.2. PPO ARCHITECTURE

### **File:** `training/ppo_model.py`

```python
class PPOActorCritic(nn.Module):
    """
    Neural network cho PPO
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        
        # Shared layers (Actor và Critic dùng chung)
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # Actor head (chọn action)
        self.actor = nn.Linear(64, act_dim)
        
        # Critic head (đánh giá state)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, state):
        # Forward pass
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor: logits cho mỗi action
        logits = self.actor(x)
        
        # Critic: value của state
        value = self.critic(x)
        
        return logits, value
```

**Giải thích kiến trúc:**

```
Input: State [3 dimensions]
   ↓
Layer 1: Linear(3 → 64) + ReLU
   ↓
Layer 2: Linear(64 → 64) + ReLU
   ↓         ↓
Actor      Critic
(64 → 64)  (64 → 1)
   ↓         ↓
Logits    Value
(action)  (state)
```

---

### **Actor vs Critic:**

| Component | Mục đích | Output |
|-----------|----------|--------|
| **Actor** | Chọn action | Logits (64 numbers) → probabilities |
| **Critic** | Đánh giá state | Value (1 number) = expected return |

**Ví dụ:**

```python
state = torch.tensor([120.0, 35.0, 8.5])  # [time, jobs, avg_pt]
logits, value = model(state)

# Actor output:
logits = [0.5, -0.3, 1.2, ..., 0.8]  # 64 values
probs = softmax(logits) = [0.02, 0.01, 0.05, ..., 0.03]
# → Portfolio 2 có probability 5% được chọn

# Critic output:
value = -250.5  # Expected total reward từ state này
```

---

### **Select Action:**

```python
# File: training/ppo_model.py
def select_action(model, state):
    """
    Chọn action từ state
    """
    state_t = torch.FloatTensor(state).unsqueeze(0)  # [1, 3]
    
    with torch.no_grad():
        logits, value = model(state_t)
    
    # Sample action từ categorical distribution
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()  # Random theo probability
    log_prob = dist.log_prob(action)
    
    return action.item(), log_prob, value
```

**Step-by-step:**

```
State: [120.0, 35.0, 8.5]
   ↓
Model forward
   ↓
Logits: [0.5, -0.3, 1.2, 0.8, ...]
   ↓
Softmax → Probabilities: [0.02, 0.01, 0.05, 0.03, ...]
   ↓
Sample (random): Action = 23 (chọn portfolio #23)
   ↓
Return: (action=23, log_prob=-3.2, value=-250.5)
```

---

## 4.3. COEVOLUTION - PPO + LGP CÙNG TIẾN HÓA

### **Khái niệm Coevolution:**

**Coevolution = 2 populations tiến hóa cùng nhau và ảnh hưởng lẫn nhau**

```
PPO Population: 1 PPO model
  ↕ (interact)
LGP Population: 64 LGP programs
```

**Vòng lặp Coevolution:**

```
Generation N:
  1. LGP programs → sinh portfolios
  2. PPO train với portfolios này
  3. PPO reward → LGP fitness
  4. Evolve LGP programs
  5. Update PPO model
  ↓
Generation N+1: Repeat
```

---

### **Tại sao Coevolution?**

**So sánh các approaches:**

| Approach | PPO | LGP | Kết quả |
|----------|-----|-----|---------|
| **Only PPO** | Learn | Fixed portfolios | PPO tốt nhưng bị giới hạn bởi portfolios |
| **Only LGP** | Fixed policy | Evolve | LGP evolve nhưng không biết state nào dùng gì |
| **Coevolution** | Learn | Evolve | ✅ **BEST**: PPO học chọn, LGP evolve portfolios |

**Ví dụ:**
```
Gen 1:
  LGP: Portfolios tệ
  PPO: Học chọn portfolio ít tệ nhất
  
Gen 5:
  LGP: Portfolios tốt hơn (nhờ PPO reward)
  PPO: Học chọn portfolio tốt trong số tốt hơn
  
Gen 10:
  LGP: Portfolios rất tốt
  PPO: Expert ở việc chọn đúng portfolio cho đúng state
```

---

## 4.4. TRAINING LOOP CHI TIẾT

### **File:** `training/lgp_coevolution_trainer.py`

### **Main Loop:**

```python
def train_with_coevolution_lgp(env, lgp_programs, model, optimizer, cfg):
    """
    Coevolution training
    """
    K = len(lgp_programs)  # 64 programs
    
    for gen in range(cfg.num_generations):  # 10 generations
        print(f"Generation {gen+1}/10")
        
        # ============================================
        # STEP 1: LGP PROGRAMS → PORTFOLIOS
        # ============================================
        lgp_inputs = build_lgp_inputs_for_env(env)
        
        action_library = []
        for prog in lgp_programs:
            portfolio = prog.execute(lgp_inputs)
            action_library.append(portfolio)
        
        env.action_library = action_library  # Gắn vào env
        
        # ============================================
        # STEP 2: PPO TRAINING
        # ============================================
        usage = np.zeros(K)      # Đếm bao nhiêu lần mỗi portfolio được dùng
        sum_reward = np.zeros(K)  # Tổng reward mỗi portfolio
        
        for ep in range(cfg.episodes_per_gen):  # 500 episodes
            state = env.reset()
            
            states, actions, log_probs, values, rewards, masks = [], [], [], [], [], []
            ep_return = 0.0
            
            # --- EPISODE LOOP ---
            for step in range(cfg.max_steps_per_episode):
                # PPO chọn action
                action, log_prob, value = select_action(model, state)
                
                # Environment step
                next_state, reward, done, _ = env.step(action)
                
                # Save trajectory
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append(0.0 if done else 1.0)
                
                # Track usage & reward
                usage[action] += 1
                sum_reward[action] += reward
                ep_return += reward
                
                state = next_state
                if done:
                    break
            
            # --- PPO UPDATE ---
            returns = compute_returns(rewards, masks, gamma=0.9)
            advantages = returns - values
            
            for _ in range(4):  # 4 PPO epochs
                policy_loss, value_loss = compute_ppo_loss(
                    states, actions, log_probs, returns, advantages
                )
                loss = policy_loss + 0.5 * value_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # ============================================
        # STEP 3: COMPUTE LGP FITNESS
        # ============================================
        avg_reward = np.zeros(K)
        for i in range(K):
            if usage[i] > 0:
                avg_reward[i] = sum_reward[i] / usage[i]
            else:
                avg_reward[i] = -1e9  # Penalty cho unused
        
        # ============================================
        # STEP 4: EVOLVE LGP PROGRAMS
        # ============================================
        # Selection
        elite_indices = np.argsort(avg_reward)[-16:]  # Top 16
        elite = [lgp_programs[i] for i in elite_indices]
        
        # Crossover + Mutation
        children = []
        for _ in range(4):  # 4 children
            p1, p2 = random.sample(elite, 2)
            child = linear_crossover(p1, p2)
            child = mutate_program(child)
            children.append(child)
        
        # Replace worst programs
        worst_indices = np.argsort(avg_reward)[:4]  # Bottom 4
        for idx, child in zip(worst_indices, children):
            lgp_programs[idx] = child
        
        print(f"Best fitness: {avg_reward.max():.2f}")
    
    return lgp_programs, action_library
```

---

### **Detailed Breakdown:**

#### **STEP 1: LGP → Portfolios**

```python
lgp_inputs = {
    "num_jobs": 20.0,
    "avg_processing_time": 8.0,
    "avg_ops_per_job": 2.5
}

action_library = []
for prog in lgp_programs:  # 64 programs
    portfolio = prog.execute(lgp_inputs)
    action_library.append(portfolio)

# Kết quả:
# action_library = [Portfolio0, Portfolio1, ..., Portfolio63]
```

---

#### **STEP 2: PPO Training**

**Episode loop:**

```python
# Episode 1:
state = [0.0, 50.0, 10.0]  # Initial state

Step 1:
  PPO chọn: action=23 (Portfolio #23)
  Env.step(23) → reward=-180
  Next state: [45.0, 35.0, 8.5]

Step 2:
  PPO chọn: action=12 (Portfolio #12)  
  Env.step(12) → reward=-210
  Next state: [78.0, 20.0, 7.2]

Step 3:
  PPO chọn: action=45 (Portfolio #45)
  Env.step(45) → reward=-120
  Done = True

Total return = -180 + -210 + -120 = -510
```

**PPO Update:**

```python
# Compute returns (discounted)
returns = [-510, -330, -120]  # Simplified

# Compute advantages
advantages = returns - values
advantages = [-510 - (-400), -330 - (-250), -120 - (-100)]
          = [-110, -80, -20]

# Policy loss
ratio = exp(new_log_prob - old_log_prob)
policy_loss = -min(ratio * advantages, clipped_ratio * advantages)

# Value loss
value_loss = (returns - values)^2

# Update
loss = policy_loss + 0.5 * value_loss
optimizer.step()
```

---

#### **STEP 3: LGP Fitness**

```python
# Sau 500 episodes:
usage = [5, 3, 0, 12, 8, ..., 7]  # Số lần mỗi portfolio được dùng
sum_reward = [-250, -180, 0, -480, -320, ..., -280]

# Tính average reward = fitness
avg_reward = sum_reward / usage
avg_reward = [-50, -60, -inf, -40, -40, ..., -40]
#                                ↑
#                          Portfolio #3 tốt nhất!
```

---

#### **STEP 4: Evolve LGP**

```python
# Selection
elite_indices = [3, 4, 63, 23, ...]  # Top 16 theo fitness
elite = [prog3, prog4, prog63, prog23, ...]

# Crossover
child1 = crossover(prog3, prog23)
child2 = crossover(prog4, prog63)
...

# Mutation
child1 = mutate(child1)
...

# Replace worst
worst_indices = [2, 17, 45, 55]  # Bottom 4
lgp_programs[2] = child1
lgp_programs[17] = child2
...
```

---

## 4.5. TỔNG KẾT TOÀN BỘ PIPELINE

### **Full Pipeline Diagram:**

```
┌─────────────────────────────────────────────────────────┐
│                   INITIALIZATION                        │
├─────────────────────────────────────────────────────────┤
│ 1. Create Environment (DynamicSchedulingEnv)           │
│ 2. Initialize PPO Model (3→64→64→[64,1])              │
│ 3. Generate 64 random LGP programs                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              GENERATION LOOP (10 times)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────────────────────────────────┐    │
│  │ PHASE 1: LGP → PORTFOLIOS                     │    │
│  ├───────────────────────────────────────────────┤    │
│  │ For each LGP program:                         │    │
│  │   inputs = build_lgp_inputs(env)              │    │
│  │   portfolio = program.execute(inputs)         │    │
│  │ → Get 64 portfolios                           │    │
│  └───────────────────────────────────────────────┘    │
│                        ↓                                │
│  ┌───────────────────────────────────────────────┐    │
│  │ PHASE 2: PPO TRAINING (# episodes)          │    │
│  ├───────────────────────────────────────────────┤    │
│  │ For each episode:                             │    │
│  │   state = env.reset()                         │    │
│  │   For each step:                              │    │
│  │     action = PPO.select(state)                │    │
│  │     next_state, reward = env.step(action)     │    │
│  │     track usage[action] += 1                  │    │
│  │     track sum_reward[action] += reward        │    │
│  │   PPO.update()                                │    │
│  └───────────────────────────────────────────────┘    │
│                        ↓                                │
│  ┌───────────────────────────────────────────────┐    │
│  │ PHASE 3: COMPUTE FITNESS                      │    │
│  ├───────────────────────────────────────────────┤    │
│  │ For each LGP program:                         │    │
│  │   fitness[i] = sum_reward[i] / usage[i]       │    │
│  └───────────────────────────────────────────────┘    │
│                        ↓                                │
│  ┌───────────────────────────────────────────────┐    │
│  │ PHASE 4: EVOLVE LGP                           │    │
│  ├───────────────────────────────────────────────┤    │
│  │ 1. Select elite (top 16)                      │    │
│  │ 2. Crossover → children                       │    │
│  │ 3. Mutation                                   │    │
│  │ 4. Replace worst programs                     │    │
│  └───────────────────────────────────────────────┘    │
│                        ↓                                │
│  Save: metrics, programs, model                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                     FINAL OUTPUT                        │
├─────────────────────────────────────────────────────────┤
│ • Trained PPO Model                                     │
│ • Evolved LGP Programs                                  │
│ • Training Metrics                                      │
│ • Visualization Plots                                   │
└─────────────────────────────────────────────────────────┘
```

---

### **Key Insights:**

1. **LGP creates diversity:**
   - 64 different programs → 64 different portfolios
   - PPO có nhiều options để chọn

2. **PPO provides selection pressure:**
   - PPO chọn programs tốt thường xuyên hơn
   - Programs tốt có fitness cao → được giữ lại

3. **Coevolution creates specialization:**
   - PPO học: "State X → dùng Program Y"
   - LGP evolve: "Program Y tối ưu cho State X"

4. **Result:**
   - PPO là "expert selector"
   - LGP programs là "specialized tools"

---

**⏭️ PHẦN 5 sẽ là Code Walkthrough với ví dụ cụ thể từ đầu đến cuối!**
