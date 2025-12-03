# GIẢI THÍCH CHI TIẾT PROJECT PPO + LGP - PHẦN 3: LINEAR GENETIC PROGRAMMING

## 📋 MỤC LỤC PHẦN NÀY
1. LGP là gì? Tại sao cần LGP?
2. Cấu trúc LGP Program
3. Registers & Instructions
4. Execution Flow (chạy program)
5. Evolution (tiến hóa programs)

---

## 3.1. LGP LÀ GÌ? TẠI SAO CẦN LGP?

### **Vấn đề ban đầu:**

Trong Phần 2, ta biết portfolio = DR + MH weights:
```python
Portfolio: EDD | SA:0.6, GA:0.3, PSO:0.1
```

**Câu hỏi:** Làm sao tạo ra các portfolios tốt?

**Cách thông thường:** Human expert thiết kế thủ công
- ❌ Mất thời gian
- ❌ Không optimal
- ❌ Không adapt với problem mới

**Giải pháp:** Dùng **LGP tự động sinh portfolios**!

---

### **LGP (Linear Genetic Programming) là gì?**

**Định nghĩa đơn giản:**

**LGP = Một chương trình tự sinh portfolios bằng cách chạy các instructions trên registers**

**So sánh:**

| Khái niệm | Tương đương với |
|-----------|-----------------|
| **LGP Program** | Một công thức tính toán |
| **Instructions** | Các bước tính toán (cộng, trừ, if-else, ...) |
| **Registers** | Biến lưu trữ (R0, R1, R2, ...) |
| **Execute** | Chạy công thức → ra kết quả = Portfolio |

**Ví dụ tương tự:**

```python
# Python code bình thường:
def create_portfolio(num_jobs, avg_pt):
    if num_jobs > 30:
        sa_weight = 2.5
    else:
        sa_weight = 1.0
    
    ga_weight = avg_pt / 10.0
    pso_weight = 0.5
    
    return Portfolio(EDD, [sa_weight, ga_weight, pso_weight])
```

**LGP làm gì:** SINH RA chính code trên bằng **evolution**!

---

### **Tại sao LGP hiệu quả?**

| Traditional | LGP |
|-------------|-----|
| Fixed portfolios | **Adaptive** portfolios |
| Same for all states | **Different** cho mỗi state |
| Human design | **Auto-generated** |

**Ví dụ:**
```
State 1: num_jobs=10, avg_pt=5  → Portfolio A
State 2: num_jobs=50, avg_pt=12 → Portfolio B (khác!)
```

LGP học được: "Khi nhiều jobs → tăng SA weight, giảm GA weight"

---

## 3.2. CẤU TRÚC LGP PROGRAM

### **File:** `core/lgp_program.py`

```python
@dataclass
class LGPProgram:
    """
    Linear GP program = danh sách instruction chạy trên dãy registers.
    """
    instructions: List[Instruction]
    num_registers: int = 20
    
    def execute(self, inputs: Dict[str, float]) -> ActionIndividual:
        """
        Chạy program với inputs → ra portfolio
        """
        # 1. Khởi tạo registers
        registers = [0.0] * self.num_registers
        
        # 2. Load inputs vào registers
        registers[0] = inputs.get("num_jobs", 0.0)
        registers[1] = inputs.get("avg_processing_time", 0.0)
        registers[2] = inputs.get("avg_ops_per_job", 0.0)
        
        # 3. Chạy từng instruction
        for instruction in self.instructions:
            instruction.execute(registers)
        
        # 4. Build portfolio từ registers cuối
        portfolio = PortfolioBuilder.build_from_registers(registers)
        
        return portfolio
```

**Giải thích:**

1. **Instructions:** List các lệnh (add, multiply, if, ...)
2. **num_registers:** Số lượng biến (R0, R1, ..., R19)
3. **execute():** Hàm chính để chạy program

---

### **Ví dụ LGP Program cụ thể:**

```python
program = LGPProgram(
    instructions=[
        # Instruction 1: R5 = R0 + R1
        ArithmeticInstruction(op="+", dest=5, src1=0, src2=1),
        
        # Instruction 2: R6 = R5 * 2.0
        ArithmeticConstInstruction(op="*", dest=6, src=5, const=2.0),
        
        # Instruction 3: if R0 > 30 then skip next line
        ConditionalSkip(cond=">", src1=0, src2_or_const=30.0, use_const=True),
        
        # Instruction 4: R7 = 1.5 (chỉ chạy nếu R0 <= 30)
        SetConstInstruction(dest=7, value=1.5),
        
        # Instruction 5: Set portfolio weights
        SetPortfolioInstruction(
            dr_name="EDD",
            mh1_reg=6, mh2_reg=7, mh3_reg=8
        )
    ],
    num_registers=20
)
```

**Đọc program:**
```
Line 1: R5 = R0 + R1          // R5 = num_jobs + avg_pt
Line 2: R6 = R5 * 2.0         // R6 = (num_jobs + avg_pt) * 2.0
Line 3: if R0 > 30: skip 1    // Nếu num_jobs > 30 thì skip line 4
Line 4: R7 = 1.5              // R7 = 1.5 (chỉ chạy khi num_jobs <= 30)
Line 5: Portfolio(EDD | SA:R6, GA:R7, PSO:R8)
```

---

## 3.3. REGISTERS & INSTRUCTIONS

### **3.3.1. Registers**

**Registers = Các biến lưu trữ số**

```python
registers = [
    0.0,   # R0: num_jobs (input)
    0.0,   # R1: avg_processing_time (input)
    0.0,   # R2: avg_ops_per_job (input)
    0.0,   # R3: (tính toán)
    0.0,   # R4: (tính toán)
    # ... R5-R19
]
```

**Rules:**
- **R0-R2:** Reserved cho inputs
- **R3-R19:** Dùng cho tính toán
- Mỗi register chứa 1 số float

---

### **3.3.2. Arithmetic Instructions**

#### **ArithmeticInstruction:**

```python
# File: core/lgp_instructions.py
@dataclass
class ArithmeticInstruction(Instruction):
    op: str      # "+", "-", "*", "/"
    dest: int    # Register đích
    src1: int    # Register nguồn 1
    src2: int    # Register nguồn 2
    
    def execute(self, registers: List[float]):
        if self.op == "+":
            registers[self.dest] = registers[self.src1] + registers[self.src2]
        elif self.op == "-":
            registers[self.dest] = registers[self.src1] - registers[self.src2]
        elif self.op == "*":
            registers[self.dest] = registers[self.src1] * registers[self.src2]
        elif self.op == "/":
            # Tránh chia 0
            if abs(registers[self.src2]) > 1e-9:
                registers[self.dest] = registers[self.src1] / registers[self.src2]
```

**Ví dụ:**
```python
# R5 = R0 + R1
instr = ArithmeticInstruction(op="+", dest=5, src1=0, src2=1)

# Trước:
registers = [10.0, 5.0, ..., 0.0, ...]
                ↑     ↑          ↑
               R0    R1         R5

# Sau:
registers = [10.0, 5.0, ..., 15.0, ...]
                              ↑
                             R5 = 10+5
```

---

#### **ArithmeticConstInstruction:**

```python
@dataclass
class ArithmeticConstInstruction(Instruction):
    op: str      # "+", "-", "*", "/"
    dest: int    
    src: int     # Register nguồn
    const: float # Hằng số
    
    def execute(self, registers: List[float]):
        if self.op == "+":
            registers[self.dest] = registers[self.src] + self.const
        elif self.op == "*":
            registers[self.dest] = registers[self.src] * self.const
        # ...
```

**Ví dụ:**
```python
# R6 = R5 * 2.0
instr = ArithmeticConstInstruction(op="*", dest=6, src=5, const=2.0)

# Trước: R5=15.0, R6=0.0
# Sau:   R5=15.0, R6=30.0
```

---

### **3.3.3. Conditional Instructions**

#### **ConditionalSkip:**

```python
@dataclass
class ConditionalSkip(Instruction):
    cond: str     # ">", "<", ">=", "<=", "=="
    src1: int     # Register so sánh 1
    src2_or_const: float  # Register hoặc hằng số
    use_const: bool       # True = dùng const, False = dùng register
    
    def execute(self, registers: List[float]):
        """
        Nếu điều kiện ĐÚNG → skip instruction tiếp theo
        """
        val1 = registers[self.src1]
        val2 = self.src2_or_const if self.use_const else registers[int(self.src2_or_const)]
        
        if self.cond == ">":
            return val1 > val2
        elif self.cond == "<":
            return val1 < val2
        # ...
```

**Ví dụ:**
```python
# if R0 > 30.0: skip next
instr = ConditionalSkip(cond=">", src1=0, src2_or_const=30.0, use_const=True)

# Case 1: R0=50
#   → 50 > 30 = True
#   → SKIP instruction tiếp theo

# Case 2: R0=20  
#   → 20 > 30 = False
#   → KHÔNG skip
```

**Trong execute loop:**
```python
for i, instruction in enumerate(self.instructions):
    if isinstance(instruction, ConditionalSkip):
        should_skip = instruction.execute(registers)
        if should_skip:
            # Skip instruction kế tiếp
            continue  # (thực tế phức tạp hơn)
    else:
        instruction.execute(registers)
```

---

### **3.3.4. Set Instructions**

#### **SetConstInstruction:**

```python
@dataclass
class SetConstInstruction(Instruction):
    dest: int
    value: float
    
    def execute(self, registers: List[float]):
        registers[self.dest] = self.value
```

**Ví dụ:**
```python
# R7 = 1.5
instr = SetConstInstruction(dest=7, value=1.5)
# R7 trước: 0.0
# R7 sau:   1.5
```

---

#### **SetPortfolioInstruction:**

```python
@dataclass
class SetPortfolioInstruction(Instruction):
    dr_name: str    # Tên DR ("EDD", "SPT", ...)
    mh1_reg: int    # Register cho MH1 weight
    mh2_reg: int    # Register cho MH2 weight
    mh3_reg: int    # Register cho MH3 weight
    
    def execute(self, registers: List[float]):
        """
        KHÔNG thực sự execute tại đây.
        Chỉ đánh dấu "portfolio sẽ được build từ những registers này"
        """
        pass  # Chỉ để PortfolioBuilder đọc
```

**PortfolioBuilder sẽ đọc:**
```python
class PortfolioBuilder:
    @staticmethod
    def build_from_registers(registers, set_portfolio_instr):
        dr_name = set_portfolio_instr.dr_name
        
        # Lấy weights từ registers
        w1 = max(0, registers[set_portfolio_instr.mh1_reg])  # Không âm
        w2 = max(0, registers[set_portfolio_instr.mh2_reg])
        w3 = max(0, registers[set_portfolio_instr.mh3_reg])
        
        # Tạo portfolio
        genes = [
            Gene(kind="DR", name=dr_name, w_raw=1.0),
            Gene(kind="MH", name="SA", w_raw=w1),
            Gene(kind="MH", name="GA", w_raw=w2),
            Gene(kind="MH", name="PSO", w_raw=w3),
        ]
        
        return ActionIndividual(genes=genes)
```

---

## 3.4. EXECUTION FLOW (CHẠY PROGRAM)

### **Ví dụ đầy đủ:**

```python
# === PROGRAM ===
program = LGPProgram(
    instructions=[
        ArithmeticInstruction(op="+", dest=5, src1=0, src2=1),     # Line 0
        ArithmeticConstInstruction(op="*", dest=6, src=5, const=2.0),  # Line 1
        ConditionalSkip(cond=">", src1=0, src2_or_const=30.0, use_const=True),  # Line 2
        SetConstInstruction(dest=7, value=1.5),                    # Line 3
        SetPortfolioInstruction(dr_name="EDD", mh1_reg=6, mh2_reg=7, mh3_reg=8)  # Line 4
    ],
    num_registers=20
)

# === INPUTS ===
inputs = {
    "num_jobs": 50.0,
    "avg_processing_time": 8.0,
    "avg_ops_per_job": 2.5
}

# === EXECUTE ===
portfolio = program.execute(inputs)
```

**Step-by-step:**

```
INITIAL STATE:
registers = [50.0, 8.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]
             R0    R1   R2   R3   R4   R5   R6   R7   R8   ...

---

LINE 0: R5 = R0 + R1
  → R5 = 50.0 + 8.0 = 58.0
registers = [50.0, 8.0, 2.5, 0.0, 0.0, 58.0, 0.0, 0.0, 0.0, ...]
                                        ↑
                                       R5

---

LINE 1: R6 = R5 * 2.0
  → R6 = 58.0 * 2.0 = 116.0
registers = [50.0, 8.0, 2.5, 0.0, 0.0, 58.0, 116.0, 0.0, 0.0, ...]
                                              ↑
                                             R6

---

LINE 2: if R0 > 30.0: skip next
  → R0 = 50.0 > 30.0 → TRUE
  → SKIP LINE 3!

---

LINE 3: (SKIPPED)
  R7 vẫn = 0.0

---

LINE 4: SetPortfolio(EDD | SA:R6, GA:R7, PSO:R8)
  Portfolio = {
      DR: "EDD",
      MH weights: [R6=116.0, R7=0.0, R8=0.0]
  }
  → Normalized: [1.0, 0.0, 0.0]  (vì R7=R8=0)
  → Final: EDD | SA:100%, GA:0%, PSO:0%
```

**Kết quả:**
```python
portfolio = ActionIndividual(genes=[
    Gene(kind="DR", name="EDD", w_raw=1.0),
    Gene(kind="MH", name="SA", w_raw=116.0),  # 100% sau normalize
    Gene(kind="MH", name="GA", w_raw=0.0),
    Gene(kind="MH", name="PSO", w_raw=0.0),
])
```

---

### **Thử với input khác:**

```python
inputs2 = {
    "num_jobs": 20.0,  # Nhỏ hơn 30!
    "avg_processing_time": 5.0,
    "avg_ops_per_job": 2.0
}
```

**Execute:**

```
registers = [20.0, 5.0, 2.0, ...]

LINE 0: R5 = 20 + 5 = 25.0
LINE 1: R6 = 25 * 2.0 = 50.0
LINE 2: if 20 > 30: skip
  → FALSE → KHÔNG skip
LINE 3: R7 = 1.5  (CHẠY!)
LINE 4: Portfolio(EDD | SA:R6, GA:R7, PSO:R8)
  → weights: [50.0, 1.5, 0.0]
  → normalized: [0.97, 0.03, 0.0]
  → EDD | SA:97%, GA:3%, PSO:0%
```

**KẾT LUẬN:**
- Input khác → Portfolio khác!
- Program học được: "Khi num_jobs > 30 → chỉ dùng SA, không dùng GA"

---

## 3.5. EVOLUTION (TIẾN HÓA PROGRAMS)

### **Genetic Algorithm cho LGP**

LGP programs được **tiến hóa** như DNA!

#### **3.5.1. Initial Population**

```python
# File: core/lgp_generator.py
class LGPGenerator:
    def generate_random_program(self):
        """Tạo program ngẫu nhiên"""
        length = random.randint(self.min_length, self.max_length)
        instructions = []
        
        for _ in range(length):
            # Random chọn loại instruction
            instr_type = random.choice([
                "arithmetic", 
                "arithmetic_const", 
                "conditional", 
                "set_const"
            ])
            
            if instr_type == "arithmetic":
                instructions.append(ArithmeticInstruction(
                    op=random.choice(["+", "-", "*", "/"]),
                    dest=random.randint(3, 19),
                    src1=random.randint(0, 19),
                    src2=random.randint(0, 19)
                ))
            # ... tương tự cho các loại khác
        
        # Thêm SetPortfolio ở cuối
        instructions.append(SetPortfolioInstruction(
            dr_name=random.choice(["EDD", "SPT", "LPT", ...]),
            mh1_reg=random.randint(3, 19),
            mh2_reg=random.randint(3, 19),
            mh3_reg=random.randint(3, 19)
        ))
        
        return LGPProgram(instructions=instructions)
```

**Tạo pool ban đầu:**
```python
pool = [generator.generate_random_program() for _ in range(64)]
```

---

#### **3.5.2. Fitness Evaluation**

```python
# Mỗi program được đánh giá:
for program in pool:
    # Chạy program nhiều lần với inputs khác nhau
    total_reward = 0
    for inputs in test_cases:
        portfolio = program.execute(inputs)
        reward = evaluate_portfolio(portfolio)  # Reward từ PPO
        total_reward += reward
    
    fitness[program] = total_reward / len(test_cases)
```

**Fitness cao = Program tốt = Sinh portfolios tốt**

---

#### **3.5.3. Selection**

```python
# Chọn elite (top programs)
sorted_programs = sorted(pool, key=lambda p: fitness[p], reverse=True)
elite = sorted_programs[:16]  # 16 tốt nhất
```

---

#### **3.5.4. Crossover (Lai ghép)**

```python
# File: core/lgp_evolution.py
def linear_crossover(parent1: LGPProgram, parent2: LGPProgram, rng):
    """
    Lai ghép 2 programs
    """
    # Chọn cutpoint random
    len1 = len(parent1.instructions)
    len2 = len(parent2.instructions)
    
    cut1 = rng.randint(0, len1)
    cut2 = rng.randint(0, len2)
    
    # Child = đầu P1 + đuôi P2
    child_instructions = (
        parent1.instructions[:cut1] + 
        parent2.instructions[cut2:]
    )
    
    return LGPProgram(instructions=child_instructions)
```

**Ví dụ:**
```
Parent1: [I0, I1, I2, I3, I4]
Parent2: [J0, J1, J2, J3]

Cut1=2, Cut2=1

Child: [I0, I1] + [J1, J2, J3]
     = [I0, I1, J1, J2, J3]
```

---

#### **3.5.5. Mutation (Đột biến)**

```python
def mutate_program(program: LGPProgram, generator, rng, mutation_rate=0.3):
    """
    Đột biến program
    """
    new_instructions = []
    
    for instr in program.instructions:
        if rng.random() < mutation_rate:
            # Mutate: thay bằng instruction mới
            new_instr = generator.generate_random_instruction()
            new_instructions.append(new_instr)
        else:
            # Giữ nguyên
            new_instructions.append(instr)
    
    return LGPProgram(instructions=new_instructions)
```

**Ví dụ:**
```
Original: [I0, I1, I2, I3]
Mutate I1: [I0, X, I2, I3]  (X = instruction mới)
```

---

#### **3.5.6. Evolution Loop**

```python
for generation in range(num_generations):
    # 1. Evaluate fitness
    for program in pool:
        fitness[program] = evaluate(program)
    
    # 2. Selection
    elite = select_elite(pool, fitness)
    
    # 3. Crossover
    children = []
    for _ in range(num_children):
        p1, p2 = random.sample(elite, 2)
        child = linear_crossover(p1, p2)
        children.append(child)
    
    # 4. Mutation
    for child in children:
        if random.random() < 0.3:
            child = mutate_program(child)
    
    # 5. New population
    pool = elite + children
```

---

**⏭️  TIẾP TỤC PHẦN 4 để tìm hiểu PPO Agent & Coevolution Pipeline!**
