# 📊 PHÂN TÍCH TẠI SAO MAKESPAN KHÔNG HỘI TỤ

## 📈 TÌNH TRẠNG HIỆN TẠI

```
Gen | Makespan | Std    | Change
---------------------------------
  1 |   166.31 |  45.30 | ---
  5 |   174.57 |  44.47 | +8.26 ❌
 10 |   164.39 |  40.51 | -10.18 ✅
 15 |   157.72 |  43.01 | -6.67 ✅ (BEST!)
 20 |   158.08 |  43.71 | +0.36 →

Range: 157.72 - 174.57 (16.85 dao động!)
Average Std: 43.98 (RẤT CAO!)
```

**Observation**: Makespan dao động lên xuống, KHÔNG có xu hướng giảm ổn định!

---

## 🔍 **6 NGUYÊN NHÂN GỐC RỄ**

---

### **1️⃣ ENVIRONMENT STOCHASTICITY (Chủ yếu)**

**Bằng chứng**:
```
Gen  1: min=78, max=305, range=227
Gen 10: min=104, max=374, range=270
Gen 20: min=81, max=330, range=250
```

**Nguyên nhân**: Mỗi episode là một bài toán KHÁC NHAU!

```python
# scheduling_env.py - Dynamic jobs are RANDOM:
- num_dynamic = 2-4 jobs/episode (random)
- 25% Urgent, 75% Normal (random)
- 1-5 operations per job (random)
- Processing time: 5-50 (random)
- Arrival time: Exponential distribution (random)
```

**Hậu quả**:
- Episode có ít dynamic jobs → makespan thấp (78-100)
- Episode có nhiều urgent jobs phức tạp → makespan cao (300-374)
- **VARIANCE CỰC CAO (std ~44) LÀ UNAVOIDABLE!**

**💡 Insight**: Đây KHÔNG phải bug - là đặc tính của bài toán DYNAMIC scheduling!

---

### **2️⃣ POLICY KHÔNG HỌC ĐƯỢC (Nghiêm trọng)**

**Bằng chứng - Within-generation learning**:
```
Gen  1: First 50 eps avg=-303.15, Last 50 eps avg=-309.56, Change=-6.41 ❌
Gen  5: First 50 eps avg=-315.47, Last 50 eps avg=-325.84, Change=-10.37 ❌
Gen 10: First 50 eps avg=-294.59, Last 50 eps avg=-309.06, Change=-14.47 ❌
Gen 20: First 50 eps avg=-297.63, Last 50 eps avg=-310.44, Change=-12.81 ❌
```

**Policy đang TỆ ĐI trong mỗi generation!** Return giảm từ đầu đến cuối mỗi gen.

**Nguyên nhân có thể**:
1. **Forced exploration gây nhiễu**: Gen 1-2 bị force random → PPO học sai signal
2. **Stale log_probs**: PPO update với forced actions nhưng dùng log_probs từ policy khác
3. **Learning rate decay quá nhanh**: Gen 20 LR = 0.000038 (gần như không học)

---

### **3️⃣ LGP EVOLUTION DESTABILIZES LEARNING**

**Bằng chứng - Best program changes**:
```
Gen | Best | Same as prev?
  1 |  #6  | NO
  2 | #38  | NO  (BEST fitness = -85.96)
  3 | #45  | NO  ← LOST best program!
  4 | #61  | NO
  5 | #14  | NO
...
Best program changed 9/19 times (47.4% instability)
```

**Vấn đề**: 
- Gen 2 đạt BEST fitness (-85.96) với program #38
- Gen 3: Program #38 bị mutate/replace → fitness drop về -136.97
- PPO phải học lại từ đầu với program mới!

**💡 LGP evolution phá hoại PPO learning!**

---

### **4️⃣ POLICY COLLAPSE VẪN XẢY RA**

**Bằng chứng**:
```
Gen  1: 64/64 programs used (forced)
Gen  5: 6/64 programs used  (9.4%)
Gen 10: 5/64 programs used  (7.8%)
Gen 20: 5/64 programs used  (7.8%)
```

**Vấn đề**:
- PPO chỉ dùng 5/64 programs (7.8%)
- Top program chiếm 60% usage
- **Không explore các programs có thể tốt hơn!**

**Entropy 0.2 KHÔNG ĐỦ cho action space 64!**

---

### **5️⃣ STATE SPACE THIẾU THÔNG TIN**

**Hiện tại**:
```python
state = [current_time, num_unfinished, avg_pt]  # Only 3 features!
```

**Vấn đề**:
- PPO không biết job nào urgent
- Không biết slack (thời gian dư)
- Không biết có bao nhiêu jobs đang late
- **→ Không đủ thông tin để chọn action tối ưu!**

---

### **6️⃣ REWARD FUNCTION KHÔNG INCENTIVIZE CONVERGENCE**

**Hiện tại**:
```python
reward = -makespan  # Only makespan!
```

**Vấn đề**:
- Reward = -makespan dao động theo dynamic jobs
- Không có bonus cho consistent performance
- Không penalize cho variance cao
- **→ PPO không được reward cho việc STABLE!**

---

## 📊 SO SÁNH: VÌ SAO V2 (10 gen) HỘI TỤ TỐT HƠN?

| Factor | V2 (old) | V4 (current) | Impact |
|--------|----------|--------------|--------|
| Entropy | 0.01 | 0.2 | V2 exploit nhanh hơn |
| Forced explore | No | Yes (Gen 1-2) | V4 mất 2 gen không học |
| LR decay | No | Yes (0.95^gen) | V4 LR quá thấp cuối |
| Best makespan | 143.16 | 157.72 | V2 tốt hơn 10% |

**V2 "may mắn" converge vào 1 program tốt sớm và exploit 99%!**
**V4 quá diverse → không exploit được program tốt nhất!**

---

## 🎯 **GIẢI PHÁP TOÀN DIỆN**

### **FIX 1: Reduce Environment Variance (Trung hạn)**

```python
# Option A: Fix số dynamic jobs
self._generate_dynamic_jobs(num_dynamic=2)  # Always 2, không random

# Option B: Seed-based episodes
def reset(self, seed=None):
    if seed:
        random.seed(seed)
    # ...deterministic dynamic jobs
```

**Expected**: Std giảm từ 44 → 20-25

---

### **FIX 2: Fix Forced Exploration (Quan trọng)**

**Vấn đề hiện tại**:
```python
if forced_exploration:
    action = programs_to_explore[forced_idx]  # Random action
    _, log_prob, value = select_action_fn(model, state)  # BUT uses policy's log_prob!
```

**Fix**:
```python
if forced_exploration:
    action = programs_to_explore[forced_idx]
    # DO NOT use this for PPO update!
    # Just collect metrics, don't train
    skip_ppo_update = True
else:
    action, log_prob, value = select_action_fn(model, state)
    skip_ppo_update = False

# Later...
if not skip_ppo_update:
    # PPO update
```

---

### **FIX 3: Hall of Fame - Protect Best Programs (Quan trọng)**

```python
# NEVER mutate/replace programs with best-ever fitness
class HallOfFame:
    def __init__(self, size=5):
        self.best_programs = []  # (fitness, program_copy, gen)
    
    def try_add(self, program, fitness, gen):
        if len(self.best_programs) < self.size:
            self.best_programs.append((fitness, deepcopy(program), gen))
            self.best_programs.sort(reverse=True)
        elif fitness > self.best_programs[-1][0]:
            self.best_programs[-1] = (fitness, deepcopy(program), gen)
            self.best_programs.sort(reverse=True)
    
    def get_protected_indices(self, current_library):
        # Return indices that should NEVER be replaced
        ...
```

**Expected**: Gen 2 best (-85.96) sẽ được maintain!

---

### **FIX 4: Adaptive Entropy Schedule**

```python
def get_entropy(gen, num_gens, base_entropy=0.2):
    """
    High entropy early (explore), low late (exploit)
    """
    progress = gen / num_gens
    if progress < 0.2:
        return 0.4  # High exploration
    elif progress < 0.5:
        return 0.25  # Moderate
    else:
        return 0.15  # Converge
```

---

### **FIX 5: Better Reward for Stability**

```python
# Track rolling average
if not hasattr(self, 'makespan_history'):
    self.makespan_history = []

self.makespan_history.append(makespan)
if len(self.makespan_history) > 10:
    self.makespan_history.pop(0)

# Bonus for beating average
avg_recent = np.mean(self.makespan_history)
if makespan < avg_recent:
    stability_bonus = 10  # Reward for improvement
else:
    stability_bonus = 0

reward = -makespan + stability_bonus
```

---

### **FIX 6: Minimum Learning Rate Floor**

```python
min_lr = 5e-5
current_lr = max(min_lr, initial_lr * (0.95 ** gen))
```

---

## 📋 **IMPLEMENTATION PRIORITY**

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| 🔴 P0 | Hall of Fame (protect best programs) | HIGH | Medium |
| 🔴 P0 | Fix forced exploration (skip PPO update) | HIGH | Low |
| 🟡 P1 | Minimum LR floor | Medium | Low |
| 🟡 P1 | Adaptive entropy | Medium | Low |
| 🟢 P2 | Reduce env variance | Low | Medium |
| 🟢 P2 | Better reward | Low | Medium |

---

## 🎯 **KẾT LUẬN**

**Makespan không hội tụ vì 3 lý do chính**:

1. **Environment variance CAO** (std ~44): Mỗi episode là bài toán khác nhau → KHÔNG THỂ đạt makespan stable hoàn toàn

2. **PPO không học được**: Forced exploration gây nhiễu, LR decay quá nhanh, policy tệ đi trong mỗi generation

3. **LGP evolution phá hoại**: Best program bị mutate/replace, PPO phải học lại từ đầu mỗi generation

**Realistic expectation**:
- Với environment variance hiện tại: Best possible std ~30-35
- Với fixes: Makespan có thể xuống 140-150 (thay vì 158)
- Perfect convergence (std < 10) là KHÔNG KHẢ THI với dynamic scheduling

---

## ✅ **RECOMMENDED ACTION**

Implement **FIX 1 + FIX 2 + FIX 3** trước:
1. Hall of Fame để protect Gen 2 best program (-85.96)
2. Skip PPO update trong forced exploration
3. Minimum LR = 5e-5

**Expected result**:
- Makespan: 145-155 (cải thiện 5-10%)
- Best fitness maintained: -85.96
- More stable training curve

Bạn muốn tôi implement các fixes này không?
