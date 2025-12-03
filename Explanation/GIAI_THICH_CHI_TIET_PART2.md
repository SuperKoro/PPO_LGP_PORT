# GIẢI THÍCH CHI TIẾT PROJECT PPO + LGP - PHẦN 2: DISPATCHING RULES & METAHEURISTICS

## 📋 MỤC LỤC PHẦN NÀY
1. Registry Pattern - Hệ thống đăng ký rules
2. Dispatching Rules (DR) chi tiết
3. Metaheuristics (MH) chi tiết  
4. Action Individual & Portfolio
5. Cách chạy một portfolio

---

## 2.1. REGISTRY PATTERN - HỆ THỐNG ĐĂNG KÝ

### **Vấn đề:**
- Có nhiều Dispatching Rules: EDD, SPT, LPT, FCFS, ...
- Có nhiều Metaheuristics: SA, GA, PSO, ...
- Làm sao để **quản lý** và **truy xuất** dễ dàng?

### **Giải pháp: Registry Pattern**

Tạo một "từ điển" (dictionary) lưu tất cả functions:

```python
# Dispatching Rules Registry
DR_REGISTRY = {
    "EDD": function_edd,
    "SPT": function_spt,
    "LPT": function_lpt,
    # ...
}

# Metaheuristics Registry
MH_REGISTRY = {
    "SA": function_sa,
    "GA": function_ga,
    "PSO": function_pso,
    # ...
}
```

Khi cần dùng:
```python
# Thay vì:
if dr_name == "EDD":
    result = function_edd(...)
elif dr_name == "SPT":
    result = function_spt(...)
# ... rất dài!

# Chỉ cần:
result = DR_REGISTRY[dr_name](...)
```

---

### **Code triển khai:**

#### **File:** `registries/dispatching_registry.py`

```python
"""Global registry for dispatching rules"""

# Dictionary lưu tất cả DR functions
_DISPATCHING_REGISTRY = {}

def register_dr(name: str):
    """Decorator để đăng ký một dispatching rule"""
    def decorator(func):
        _DISPATCHING_REGISTRY[name] = func
        return func
    return decorator

def get_dr(name: str):
    """Lấy DR function theo tên"""
    if name not in _DISPATCHING_REGISTRY:
        raise ValueError(f"Unknown dispatching rule: {name}")
    return _DISPATCHING_REGISTRY[name]

def list_drs():
    """List tất cả DR đã đăng ký"""
    return list(_DISPATCHING_REGISTRY.keys())
```

**Giải thích:**

1. **`_DISPATCHING_REGISTRY = {}`**: Dictionary trống để lưu
2. **`@register_dr("EDD")`**: Decorator để đăng ký
3. **`get_dr("EDD")`**: Lấy function ra khi cần

---

#### **File:** `registries/dispatching_rules.py`

```python
# File: registries/dispatching_rules.py

from registries.dispatching_registry import register_dr
from environment.env_utils import reschedule_unfinished_jobs_earliest_due_date

@register_dr("EDD")
def dr_earliest_due_date(env, finished_events, unfinished_jobs):
    """
    Earliest Due Date: Ưu tiên job có deadline sớm nhất
    """
    return reschedule_unfinished_jobs_earliest_due_date(
        env,
        finished_events,
        unfinished_jobs
    )
```

**Giải thích:**

1. **`@register_dr("EDD")`**:
   - Khi Python load file này
   - Nó tự động gọi `register_dr("EDD")(dr_earliest_due_date)`
   - Thêm `{"EDD": dr_earliest_due_date}` vào `_DISPATCHING_REGISTRY`

2. **Function wrapper:**
   - `dr_earliest_due_date` là wrapper function
   - Nó gọi function thực tế `reschedule_unfinished_jobs_earliest_due_date`

---

## 2.2. DISPATCHING RULES (DR) CHI TIẾT

### **DR là gì?**

**Dispatching Rule** = Quy tắc để **chọn job nào làm trước** khi có nhiều jobs chờ.

**Ví dụ tình huống:**
```
Máy 1 đang rảnh, có 3 jobs đang chờ:
  - Job A: due_date = 100, processing_time = 10
  - Job B: due_date = 80,  processing_time = 5
  - Job C: due_date = 120, processing_time = 15

→ Chọn job nào?
```

Mỗi DR cho câu trả lời khác nhau!

---

### **2.2.1. EDD (Earliest Due Date)**

**Ý tưởng:** Chọn job có **deadline sớm nhất** trước

```python
@register_dr("EDD")
def dr_earliest_due_date(env, finished_events, unfinished_jobs):
    return reschedule_unfinished_jobs_earliest_due_date(...)
```

**Logic trong `reschedule_unfinished_jobs_earliest_due_date`:**

```python
# File: environment/env_utils.py
def reschedule_unfinished_jobs_earliest_due_date(env, finished_events, unfinished_jobs):
    # Sắp xếp jobs theo due_date tăng dần
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: x[1]['due_date']  # Sort by due_date
    )
    
    # Schedule từng job theo thứ tự
    new_schedule = []
    for job_id, job_info in sorted_jobs:
        # Schedule operations của job này
        for op in job_info['operations']:
            # Tìm máy rảnh sớm nhất
            machine = find_earliest_available_machine(op['candidate_machines'], ...)
            # Thêm vào schedule
            new_schedule.append({
                'job': job_id,
                'op_id': op['op_id'],
                'machine': machine,
                'start': earliest_time,
                'finish': earliest_time + op['processing_time']
            })
    
    return new_schedule
```

**Ví dụ:**
```
Jobs:
  A: due=100
  B: due=80   ← Chọn trước
  C: due=120

Kết quả: B → A → C
```

---

### **2.2.2. SPT (Shortest Processing Time)**

**Ý tưởng:** Chọn job có **thời gian ngắn nhất** trước

```python
@register_dr("SPT")
def dr_shortest_processing_time(env, finished_events, unfinished_jobs):
    return reschedule_unfinished_jobs_shortest_processing_time(...)
```

**Logic:**
```python
def reschedule_unfinished_jobs_shortest_processing_time(...):
    # Sắp xếp theo tổng processing time
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: sum(op['processing_time'] for op in x[1]['operations'])
    )
    # ... tương tự EDD
```

**Ví dụ:**
```
Jobs:
  A: total_pt=25
  B: total_pt=15  ← Chọn trước
  C: total_pt=40

Kết quả: B → A → C
```

---

### **2.2.3. Các DR khác:**

| DR | Ý nghĩa | Sort key |
|----|---------|----------|
| **LPT** | Longest Processing Time | `-total_pt` (dài nhất trước) |
| **FCFS** | First Come First Served | `arrival_time` (đến trước làm trước) |
| **CR** | Critical Ratio | `(due_date - current_time) / total_pt` |

---

## 2.3. METAHEURISTICS (MH) CHI TIẾT

### **MH là gì?**

**Metaheuristic** = Thuật toán **tối ưu hóa** schedule đã có.

**Flow:**
```
Initial Schedule (từ DR)
    ↓
Apply MH (SA/GA/PSO/...)
    ↓
Improved Schedule (hopefully!)
```

---

### **2.3.1. SA (Simulated Annealing)**

**Ý tưởng:** Giống "ủ kim loại"
- Bắt đầu "nóng" → chấp nhận giải pháp tệ
- Dần "nguội" → chỉ chấp nhận giải pháp tốt

**Code:**

```python
# File: registries/metaheuristics_impl.py
@register_mh("SA")
def mh_simulated_annealing(env, finished_events, unfinished_jobs, time_budget_s):
    return reschedule_unfinished_jobs_sa(
        env,
        finished_events,
        unfinished_jobs,
        time_budget_s=time_budget_s
    )
```

**Logic trong `reschedule_unfinished_jobs_sa`:**

```python
# File: environment/env_utils.py (simplified)
def reschedule_unfinished_jobs_sa(env, finished_events, unfinished_jobs, 
                                   time_budget_s=3.0):
    # 1. Tạo initial solution (dùng EDD)
    current_solution = reschedule_unfinished_jobs_earliest_due_date(
        env, finished_events, unfinished_jobs
    )
    current_cost = calculate_makespan(current_solution)
    
    best_solution = current_solution
    best_cost = current_cost
    
    # 2. SA parameters
    temperature = 100.0
    cooling_rate = 0.95
    iterations = 100
    
    # 3. SA loop
    for i in range(iterations):
        # Tạo neighbor (đổi chỗ 2 operations random)
        neighbor = swap_random_operations(current_solution)
        neighbor_cost = calculate_makespan(neighbor)
        
        # Quyết định accept hay không
        delta = neighbor_cost - current_cost
        
        if delta < 0:  # Tốt hơn → accept luôn
            current_solution = neighbor
            current_cost = neighbor_cost
        else:  # Tệ hơn → accept với xác suất
            acceptance_prob = math.exp(-delta / temperature)
            if random.random() < acceptance_prob:
                current_solution = neighbor
                current_cost = neighbor_cost
        
        # Update best
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost
        
        # Cool down
        temperature *= cooling_rate
    
    return best_solution
```

**Giải thích:**

1. **Initial solution:** Dùng EDD tạo schedule ban đầu
2. **Loop nhiều lần:**
   - Tạo **neighbor** (schedule tương tự nhưng khác 1 chút)
   - Nếu **tốt hơn** → chấp nhận
   - Nếu **tệ hơn** → chấp nhận với **xác suất** (giảm dần theo temperature)
3. **Temperature giảm dần:** Càng về sau càng khó chấp nhận giải pháp tệ

**So sánh trực quan:**
```
Iteration 1 (T=100):  Bad solution với ΔCost=+50 → 60% accept
Iteration 50 (T=50):  Bad solution với ΔCost=+50 → 30% accept  
Iteration 100 (T=10): Bad solution với ΔCost=+50 → 5% accept
```

---

### **2.3.2. GA (Genetic Algorithm)**

**Ý tưởng:** Mô phỏng tiến hóa
- Có "population" nhiều schedules
- "Crossover" (lai ghép) các schedules tốt
- "Mutation" (đột biến) random

```python
@register_mh("GA")
def mh_genetic_algorithm(env, finished_events, unfinished_jobs, time_budget_s):
    return reschedule_unfinished_jobs_ga(...)
```

**Logic (simplified):**
```python
def reschedule_unfinished_jobs_ga(...):
    # 1. Init population
    population = [create_random_schedule() for _ in range(50)]
    
    # 2. Evolution
    for generation in range(20):
        # Đánh giá fitness
        fitness = [calculate_makespan(s) for s in population]
        
        # Selection (chọn tốt nhất)
        parents = select_best(population, fitness, k=20)
        
        # Crossover (lai ghép)
        children = []
        for i in range(0, len(parents), 2):
            child1, child2 = crossover(parents[i], parents[i+1])
            children.extend([child1, child2])
        
        # Mutation (đột biến)
        for child in children:
            if random.random() < 0.1:  # 10% mutation rate
                mutate(child)
        
        # New population
        population = parents + children
    
    # Return best
    best_idx = np.argmin(fitness)
    return population[best_idx]
```

---

### **2.3.3. PSO (Particle Swarm Optimization)**

**Ý tưởng:** Mô phỏng đàn chim tìm thức ăn
- Mỗi "particle" = 1 schedule
- Di chuyển về phía best solution

```python
@register_mh("PSO")
def mh_particle_swarm(env, finished_events, unfinished_jobs, time_budget_s):
    return reschedule_unfinished_jobs_pso(...)
```

---

## 2.4. ACTION INDIVIDUAL & PORTFOLIO

### **Gene Data Structure**

```python
# File: training/portfolio_types.py
@dataclass
class Gene:
    kind: str      # "DR" hoặc "MH"
    name: str      # Tên: "EDD", "SA", ...
    w_raw: float   # Weight (trọng số)
```

**Ví dụ:**
```python
gene1 = Gene(kind="DR", name="EDD", w_raw=1.0)
gene2 = Gene(kind="MH", name="SA", w_raw=2.5)
gene3 = Gene(kind="MH", name="GA", w_raw=1.0)
```

---

### **ActionIndividual (Portfolio)**

```python
@dataclass
class ActionIndividual:
    genes: List[Gene]
    
    @property
    def dr_gene(self) -> Gene:
        """Gene đầu tiên luôn là DR"""
        return self.genes[0]
    
    @property
    def mh_genes(self) -> List[Gene]:
        """Các gene còn lại là MH"""
        return self.genes[1:]
```

**Ví dụ portfolio:**
```python
portfolio = ActionIndividual(genes=[
    Gene(kind="DR", name="EDD", w_raw=1.0),         # DR
    Gene(kind="MH", name="SA", w_raw=2.5),          # MH 1
    Gene(kind="MH", name="GA", w_raw=1.0),          # MH 2
    Gene(kind="MH", name="PSO", w_raw=0.5),         # MH 3
])
```

**Giải thích:**
- **1 DR gene**: EDD
- **3 MH genes**: SA (weight=2.5), GA (weight=1.0), PSO (weight=0.5)

---

### **Normalized Weights**

Weights được **normalize** để tổng = 1.0:

```python
def individual_normalized_weights(ind: ActionIndividual):
    mh_weights_raw = [g.w_raw for g in ind.mh_genes]
    total = sum(mh_weights_raw)
    
    if total == 0:
        # Nếu tất cả = 0 → chia đều
        return [1.0 / len(mh_weights_raw)] * len(mh_weights_raw)
    
    # Normalize
    return [w / total for w in mh_weights_raw]
```

**Ví dụ:**
```python
Raw weights: [2.5, 1.0, 0.5]
Total: 4.0
Normalized: [0.625, 0.25, 0.125]
```

**Ý nghĩa:** SA chiếm 62.5% time budget, GA 25%, PSO 12.5%

---

## 2.5. CÁCH CHẠY MỘT PORTFOLIO

### **Code:**

```python
# File: training/typed_action_adapter.py
def run_action_individual(
    env,
    individual: ActionIndividual,
    finished_events: list,
    unfinished_jobs: dict,
    total_budget_s: float = 3.0
):
    """
    Chạy 1 portfolio = DR + nhiều MH
    """
    # 1. Lấy DR và MH functions
    dr_func = get_dr(individual.dr_gene.name)
    mh_funcs = [get_mh(g.name) for g in individual.mh_genes]
    
    # 2. Chạy DR trước
    schedule_after_dr = dr_func(env, finished_events, unfinished_jobs)
    
    # 3. Normalize MH weights
    mh_weights = individual_normalized_weights(individual)
    
    # 4. Chạy từng MH với time budget tương ứng
    current_schedule = schedule_after_dr
    
    for mh_func, weight in zip(mh_funcs, mh_weights):
        time_budget_for_this_mh = total_budget_s * weight
        
        if time_budget_for_this_mh > 0.01:  # Chỉ chạy nếu có đủ time
            current_schedule = mh_func(
                env,
                finished_events,
                unfinished_jobs,
                time_budget_s=time_budget_for_this_mh
            )
    
    return current_schedule
```

**Flow chi tiết:**

```
Portfolio: EDD | SA:62.5%, GA:25%, PSO:12.5%
Total budget: 3.0 seconds

Step 1: Chạy DR (EDD)
    → Tạo initial schedule

Step 2: Chạy SA với budget = 3.0 * 0.625 = 1.875s
    → Improve schedule

Step 3: Chạy GA với budget = 3.0 * 0.25 = 0.75s
    → Improve thêm

Step 4: Chạy PSO với budget = 3.0 * 0.125 = 0.375s
    → Final schedule

Return: Final schedule
```

---

**⏭️  TIẾP TỤC PHẦN 3 để tìm hiểu Linear Genetic Programming (LGP)!**
