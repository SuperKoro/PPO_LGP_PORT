# 📊 UPDATE THÔNG SỐ METAHEURISTIC - OPTION A

## ✅ **ĐÃ APPLY: ADOPT NOTEBOOK PARAMETERS**

---

## 🔄 **THAY ĐỔI THÔNG SỐ**

### **File: `registries/metaheuristics_impl.py`**

---

### **1️⃣ SIMULATED ANNEALING (SA)**

**Before (Project Original):**
```python
iterations = max(10, int(time_budget_s * 10.0))
# With time_budget_s = 3.0 → iterations = 30
```

**After (Adopt Notebook):**
```python
iterations = max(30, int(time_budget_s * 16.7))
# With time_budget_s = 3.0 → iterations = 50 ✅
```

**Change:**
- **+20 iterations** (+67% more exploration)
- Matches notebook proven parameters

---

### **2️⃣ PARTICLE SWARM OPTIMIZATION (PSO)**

**Before (Project Original):**
```python
num_particles = max(5, int(time_budget_s * 5.0))  # = 15
iterations = max(5, int(time_budget_s * 5.0))     # = 15
# Total: 15 particles × 15 iterations = 225 evaluations
```

**After (Adopt Notebook):**
```python
num_particles = max(5, int(time_budget_s * 3.3))  # = 10
iterations = max(10, int(time_budget_s * 6.7))    # = 20
# Total: 10 particles × 20 iterations = 200 evaluations
```

**Change:**
- **-5 particles** (better focus)
- **+5 iterations** (better convergence)
- Total evaluations: 225 → 200 (-11% but more effective)
- Matches notebook proven parameters

---

### **3️⃣ GENETIC ALGORITHM (GA)**

**No Change (Keep Project Parameters):**
```python
num_candidates = max(5, int(time_budget_s * 5.0))  # = 15
generations = max(3, int(time_budget_s * 3.0))     # = 9
# Total: 15 candidates × 9 generations = 135 evaluations
```

**Reason:**
- Project GA (135 evals) > Notebook GA (50 evals)
- More powerful, better exploration
- **Keep as is** ✅

---

## 📊 **SO SÁNH TỔNG QUAN**

| Metric | Notebook | Old Project | **New Project** | Change |
|--------|----------|-------------|-----------------|--------|
| **SA Iterations** | 50 | 30 | **50** | +67% ✅ |
| **PSO Particles** | 10 | 15 | **10** | -33% |
| **PSO Iterations** | 20 | 15 | **20** | +33% ✅ |
| **PSO Total Evals** | 200 | 225 | **200** | -11% |
| **GA Pop × Gens** | 10×5 | 15×9 | **15×9** | Keep |
| **GA Total Evals** | 50 | 135 | **135** | Keep |

---

## 🎯 **EXPECTED IMPROVEMENTS**

### **1. Better SA Performance**
- 50 iterations vs 30 → More thorough search
- Temperature decay: 100 × 0.95^50 = 7.7 (vs 4.8 before)
- **→ Better exploration in early iterations**
- **→ Better exploitation in late iterations**

### **2. Better PSO Convergence**
- 20 iterations vs 15 → +33% convergence time
- 10 particles (more focused) vs 15 (spread thin)
- **→ Better global best convergence**
- **→ Less premature convergence**

### **3. Proven in Notebook**
- These parameters worked well in notebook
- Tested on same dataset (20 jobs)
- **→ Higher confidence in performance**

---

## 🔬 **THEORETICAL ANALYSIS**

### **SA: Why 50 > 30?**

Temperature schedule:
```python
Gen  | T (iter=30) | T (iter=50) | Acceptance Prob (ΔC=10)
-----|-------------|-------------|-------------------------
 0   |   100.00    |   100.00    | 0.905
10   |    59.87    |    59.87    | 0.847
20   |    35.85    |    35.85    | 0.766
30   |    21.46    |    21.46    | 0.632
40   |      -      |    12.85    | 0.438
50   |      -      |     7.69    | 0.281
```

**→ 50 iterations allows better fine-tuning in low temperature!**

### **PSO: Why 20 iters > 15 iters?**

Convergence dynamics:
```python
# PSO velocity decay: v *= w = 0.5
Iter | Effective Velocity Contribution
-----|-----------------------------------
  1  | 100%
  5  | 3.1%  (0.5^5)
 10  | 0.1%  (0.5^10)
 15  | 0.003% (0.5^15) ← Old stopping point
 20  | 0.0001% (0.5^20) ← New stopping point
```

**→ 20 iterations ensures full convergence to global best!**

---

## 🚀 **NEXT STEPS**

### **1. Test with Current Config**
```bash
python scripts/train_lgp.py
```

Expected:
- SA: Better makespan in early gens
- PSO: Better convergence in dominant portfolios
- Overall: 2-5% improvement in makespan

### **2. Monitor Key Metrics**
- Best program performance (Gen 2-20)
- Makespan evolution
- Which MH is used most (should favor PSO/SA more)

### **3. Compare Results**
- Old best: Makespan 143.16 (Gen 20)
- New target: **<140** with better SA/PSO

---

## 📝 **CONFIGURATION SUMMARY**

**With `time_budget_s = 3.0`:**

| Metaheuristic | Iterations | Population/Particles | Total Evaluations |
|---------------|------------|----------------------|-------------------|
| **SA** | **50** ✅ | 1 | **50** |
| **GA** | 9 | 15 | 135 |
| **PSO** | **20** ✅ | **10** ✅ | **200** |

**Total computational effort:**
- SA: 50 evals (powerful single-thread)
- GA: 135 evals (exploration-focused)
- PSO: 200 evals (balanced exploration-exploitation)

---

## ✅ **STATUS: READY FOR TESTING**

All changes applied to `registries/metaheuristics_impl.py`

Next action: Run training to validate improvements! 🚀
