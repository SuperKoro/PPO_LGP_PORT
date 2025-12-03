# GIẢI THÍCH CHI TIẾT PROJECT PPO + LGP - PHẦN 1: TỔNG QUAN & MÔI TRƯỜNG

## 📋 MỤC LỤC TOÀN BỘ SERIES
- **PHẦN 1** (file này): Tổng quan + Môi trường Job Shop Scheduling
- **PHẦN 2**: Dispatching Rules & Metaheuristics Integration
- **PHẦN 3**: Linear Genetic Programming (LGP) chi tiết
- **PHẦN 4**: PPO Agent & Coevolution Pipeline
- **PHẦN 5**: Code Walkthrough & Examples

---

## 🎯 PHẦN 1: TỔNG QUAN & MÔI TRƯỜNG

### 1.1. BÀI TOÁN CẦN GIẢI QUYẾT

#### **Bài toán Job Shop Scheduling là gì?**

Tưởng tượng bạn có một nhà máy với:
- **Nhiều máy móc** (machines): Máy 1, Máy 2, ..., Máy N
- **Nhiều công việc** (jobs): Job A, Job B, Job C, ...
- Mỗi job có **nhiều công đoạn** (operations) phải làm **tuần tự**

**Ví dụ cụ thể:**
```
Job 1: Op1 (máy 1, 10 phút) → Op2 (máy 3, 5 phút) → Op3 (máy 2, 8 phút)
Job 2: Op1 (máy 2, 12 phút) → Op2 (máy 1, 7 phút)
Job 3: Op1 (máy 3, 6 phút) → Op2 (máy 1, 9 phút) → Op3 (máy 3, 4 phút)
```

**Mục tiêu:** Sắp xếp các operations lên máy sao cho:
- ✅ Mỗi máy chỉ làm 1 việc tại 1 thời điểm
- ✅ Mỗi job làm đúng thứ tự operations
- ✅ **Minimize Makespan** = Thời gian hoàn thành tất cả jobs

#### **Bài toán DYNAMIC Job Shop Scheduling**

Project này giải quyết bài toán **KHÓ HƠN**:
- Jobs **không xuất hiện cùng lúc** mà **đến dần dần** (dynamic arrivals)
- Mỗi khi có job mới đến → phải **reschedule** (sắp xếp lại)
- Có jobs **Urgent** (khẩn cấp) và **Normal**

**Ví dụ timeline:**
```
t=0:   Job 1, 2, 3 bắt đầu
t=45:  Job 4 đến (dynamic!) → Phải reschedule
t=78:  Job 5 đến (urgent!)  → Phải reschedule
t=120: Job 6 đến            → Phải reschedule
```

---

### 1.2. MÔI TRƯỜNG (ENVIRONMENT) - `DynamicSchedulingEnv`

Đây là "thế giới" mà AI agent sống và tương tác.

#### **File:** `environment/scheduling_env.py`

#### **Khái niệm cơ bản:**

Environment (môi trường) trong Reinforcement Learning giống như một **game**:
- **State** (trạng thái): Thông tin hiện tại (bao nhiêu jobs, máy nào đang bận, ...)
- **Action** (hành động): Agent chọn làm gì (chọn DR nào, MH nào)
- **Reward** (phần thưởng): Điểm số sau khi làm (+10, -50, ...)
- **Done** (kết thúc): Game kết thúc chưa?

---

### 1.3. CẤU TRÚC DỮ LIỆU TRONG ENVIRONMENT

#### **1.3.1. Jobs Data Structure**

```python
# File: environment/scheduling_env.py, line 25-78
jobs_initial = {
    1: [{'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12}],
    2: [{'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12}],
    3: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1},
        {'op_id': 2, 'candidate_machines': [6], 'processing_time': 8},
        {'op_id': 3, 'candidate_machines': [6], 'processing_time': 8}
    ],
    # ... more jobs
}
```

**Giải thích:**
- **Key** (1, 2, 3, ...): Job ID
- **Value**: List các operations của job đó
  - `op_id`: Thứ tự operation (phải làm tuần tự 1→2→3)
  - `candidate_machines`: Máy nào có thể làm operation này
  - `processing_time`: Thời gian cần để hoàn thành (phút)

**Ví dụ đọc:**
- Job 1: Có 1 operation, làm trên máy 1 hoặc 2, mất 12 phút
- Job 3: Có 3 operations:
  - Op1: Máy 3 hoặc 4, mất 1 phút
  - Op2: Máy 6, mất 8 phút (sau khi Op1 xong)
  - Op3: Máy 6, mất 8 phút (sau khi Op2 xong)

#### **1.3.2. Due Dates (Deadline)**

```python
# File: environment/scheduling_env.py, line 81
due_dates_initial = {i: 1200 for i in range(1, 51)}
```

Mỗi job có **deadline** (due date):
- Job 1 phải hoàn thành trước thời điểm 1200
- Nếu trễ → **tardiness** (phạt điểm)

#### **1.3.3. Schedule Events**

Mỗi khi lập lịch xong, ta có list các **events**:

```python
# Ví dụ một event:
{
    'job': 1,           # Job ID
    'op_id': 1,         # Operation ID
    'machine': 2,       # Máy được chọn
    'start': 10,        # Thời điểm bắt đầu
    'finish': 22,       # Thời điểm kết thúc
}
```

**Giải thích:**
- Job 1, operation 1 được lên lịch
- Chạy trên máy 2
- Bắt đầu lúc t=10, kết thúc lúc t=22

**Current schedule:**
```python
self.current_schedule_events = [
    {'job': 1, 'op_id': 1, 'machine': 1, 'start': 0, 'finish': 12},
    {'job': 2, 'op_id': 1, 'machine': 2, 'start': 0, 'finish': 12},
    {'job': 3, 'op_id': 1, 'machine': 3, 'start': 0, 'finish': 1},
    # ... many more events
]
```

---

### 1.4. KHỞI TẠO MÔI TRƯỜNG

#### **Code:**

```python
# File: environment/scheduling_env.py, line 93-121
class DynamicSchedulingEnv(gym.Env):
    def __init__(self,
                 lambda_tardiness: float = 1.0,
                 action_library: list = None,
                 action_budget_s: float = 3.0):
        super(DynamicSchedulingEnv, self).__init__()
        
        # Lưu parameters
        self.lambda_tardiness = lambda_tardiness  # Trọng số penalty cho tardiness
        self.machine_pool = machine_pool          # List các máy: [1,2,3,4,5,6,7,8,12,13]
        self.jobs_initial = jobs_initial          # Jobs ban đầu
        self.due_dates_initial = due_dates_initial
        
        # Tạo unified jobs info (kết hợp jobs + due dates)
        self.all_jobs_info = create_unified_jobs_info(
            self.jobs_initial, 
            self.due_dates_initial
        )
        
        # TẠO INITIAL SCHEDULE bằng Simulated Annealing
        _, schedule, _, _, _, _ = simulated_annealing(
            self.jobs_initial,
            self.due_dates_initial,
            lambda_tardiness=self.lambda_tardiness
        )
        
        # Convert schedule sang list events
        self.initial_schedule_events = schedule_dict_to_list(
            schedule, 
            self.all_jobs_info
        )
        
        # Copy để dùng
        self.current_schedule_events = copy.deepcopy(self.initial_schedule_events)
        self.current_time = 0
        
        # Sinh dynamic jobs (jobs sẽ đến sau)
        self._generate_dynamic_jobs(num_dynamic=4)
        self.current_dynamic_index = 0
        
        # Action library (portfolios PPO có thể chọn)
        self.action_library = action_library if action_library is not None \
                             else self._build_default_action_library()
        self.action_budget_s = float(action_budget_s)  # Thời gian tối ưu cho mỗi action
        
        # Định nghĩa observation space và action space
        self.observation_space = spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_library))
```

**Giải thích từng bước:**

1. **Lưu thông tin cơ bản:**
   - `lambda_tardiness`: Penalty weight cho jobs trễ deadline
   - `machine_pool`: List máy có sẵn
   - `jobs_initial`, `due_dates_initial`: Jobs và deadlines

2. **Tạo initial schedule:**
   ```python
   _, schedule, _, _, _, _ = simulated_annealing(...)
   ```
   - Dùng **Simulated Annealing** (một metaheuristic) để tạo lịch ban đầu
   - Schedule này sẽ được dùng làm "baseline"

3. **Generate dynamic jobs:**
   ```python
   self._generate_dynamic_jobs(num_dynamic=4)
   ```
   - Tạo 4 jobs sẽ đến **sau** trong quá trình agent chạy
   - Mỗi job có `arrival_time` khác nhau

4. **Action library:**
   - Đây là **danh sách portfolios** mà PPO agent có thể chọn
   - Mỗi portfolio = 1 cách kết hợp DR + MH
   - Sẽ giải thích chi tiết ở Phần 2

5. **Observation & Action spaces:**
   ```python
   self.observation_space = spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)
   self.action_space = spaces.Discrete(len(self.action_library))
   ```
   - **Observation space**: Vector 3 chiều (sẽ giải thích ở section 1.5)
   - **Action space**: Số lượng portfolios có thể chọn (thường là 64)

---

### 1.5. STATE (OBSERVATION) - Agent nhìn thấy gì?

Agent cần biết "hiện tại thế nào" để quyết định action.

#### **Code:**

```python
# File: environment/scheduling_env.py, line 228-245
def _get_state(self):
    """Get current state observation."""
    # Tách schedule thành finished và unfinished
    finished_events, unfinished_jobs = split_schedule_list(
        self.current_schedule_events, 
        self.current_time, 
        self.all_jobs_info
    )
    
    # Đếm số operations chưa làm
    num_unfinished = sum(len(info['operations']) for info in unfinished_jobs.values())
    
    # Tính processing time trung bình
    total_pt = 0
    count = 0
    for info in unfinished_jobs.values():
        for op in info['operations']:
            total_pt += op['processing_time']
            count += 1
    avg_pt = total_pt / count if count > 0 else 0
    
    # Trả về state vector 3 chiều
    return np.array([self.current_time, num_unfinished, avg_pt], dtype=np.float32)
```

**State gồm 3 thông tin:**

1. **current_time** (thời gian hiện tại):
   - VD: 120 (đang ở thời điểm t=120)

2. **num_unfinished** (số operations chưa hoàn thành):
   - VD: 35 (còn 35 operations chưa làm)

3. **avg_pt** (processing time trung bình):
   - VD: 8.5 (trung bình mỗi operation mất 8.5 phút)

**Ví dụ state:**
```python
state = [120.0, 35.0, 8.5]
```

Agent nhận được vector này và quyết định: "Với tình hình này, tôi nên chọn portfolio nào?"

---

### 1.6. DYNAMIC JOBS GENERATION

Jobs không đến cùng lúc, mà **đến dần dần** trong quá trình scheduling.

#### **Code:**

```python
# File: environment/scheduling_env.py, line 144-180
def _generate_dynamic_job(self, job_id, arrival_time, 
                          min_ops=1, max_ops=5, min_pt=5, max_pt=50):
    """Generate a single dynamic job."""
    # 25% là urgent, 75% là normal
    if random.random() < 0.25:
        job_type = "Urgent"
        etuf = 1.2  # Due date sát hơn (ít thời gian hơn)
    else:
        job_type = "Normal"
        etuf = 1.8  # Due date rộng hơn
    
    # Random số operations (1-5)
    num_ops = random.randint(min_ops, max_ops)
    operations = []
    total_pt = 0
    
    for i in range(num_ops):
        # Random máy candidates
        candidate_machines = random.sample(
            self.machine_pool, 
            k=random.randint(1, min(5, len(self.machine_pool)))
        )
        # Random processing time
        pt = random.randint(min_pt, max_pt)
        total_pt += pt
        
        op = {
            'op_id': i+1,
            'candidate_machines': candidate_machines,
            'processing_time': pt
        }
        operations.append(op)
    
    # Tính due date
    due_date = math.ceil(arrival_time + total_pt * etuf)
    
    dynamic_job = {
        'job_id': job_id,
        'arrival_time': arrival_time,
        'due_date': due_date,
        'operations': operations,
        'job_type': job_type
    }
    return dynamic_job
```

**Giải thích:**

1. **Job type (25% Urgent, 75% Normal):**
   ```python
   if random.random() < 0.25:
       job_type = "Urgent"
       etuf = 1.2  # Expected Time Until Finish
   ```
   - Urgent jobs: `etuf = 1.2` → deadline sát hơn
   - Normal jobs: `etuf = 1.8` → deadline rộng hơn

2. **Operations generation:**
   - Random 1-5 operations
   - Mỗi operation:
     - Random máy candidates (1-5 máy)
     - Random processing time (5-50 phút)

3. **Due date calculation:**
   ```python
   due_date = math.ceil(arrival_time + total_pt * etuf)
   ```
   - `total_pt`: Tổng thời gian cần thiết
   - Nhân với `etuf` để có thêm "buffer time"

**Ví dụ:**
```
Job D1 (Urgent):
  - Arrival: t=45
  - Operations: 3 ops, total_pt=25 phút
  - Due date: 45 + 25*1.2 = 75
  - → Chỉ có 30 phút để hoàn thành!

Job D2 (Normal):
  - Arrival: t=78  
  - Operations: 2 ops, total_pt=20 phút
  - Due date: 78 + 20*1.8 = 114
  - → Có 36 phút để hoàn thành (rộng hơn)
```

---

**⏭️  TIẾP TỤC PHẦN 2 để tìm hiểu về Dispatching Rules & Metaheuristics!**
