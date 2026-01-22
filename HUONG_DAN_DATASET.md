# 📚 HƯỚNG DẪN SỬ DỤNG DATASET KHÁC

## 🎯 TÓM TẮT

Project hiện có **7 datasets khác nhau** (từ 20 đến 50 jobs). Bạn có thể dễ dàng chuyển đổi giữa các datasets bằng cách thay đổi **1 dòng trong `config.py`**.

---

## 📂 DATASETS CÓ SẴN

```
data/
  ├── Set20.json  → 20 jobs (default)
  ├── Set25.json  → 25 jobs
  ├── Set30.json  → 30 jobs
  ├── Set35.json  → 35 jobs
  ├── Set40.json  → 40 jobs
  ├── Set45.json  → 45 jobs
  └── Set50.json  → 50 jobs
```

### Cấu trúc mỗi dataset:
- **Jobs**: Danh sách công việc với các operations
- **Machines**: Pool các máy móc khả dụng
- **Due dates**: Deadline cho mỗi job
- **Processing times**: Thời gian xử lý mỗi operation

---

## ⚡ CÁCH SỬ DỤNG (3 BƯỚC)

### **Bước 1: Mở file `config.py`**

Tìm dòng này trong class `EnvironmentConfig` (khoảng dòng 150):

```python
class EnvironmentConfig:
    """Job Shop Scheduling Environment settings"""
    
    # Dataset selection
    dataset_name = None  # ← ĐÂY!
```

---

### **Bước 2: Thay đổi `dataset_name`**

#### **Option A: Dùng dataset mặc định (hardcoded - 20 jobs)**
```python
dataset_name = None  # Default
```

#### **Option B: Dùng Set20 (20 jobs)**
```python
dataset_name = "Set20"
```

#### **Option C: Dùng Set25 (25 jobs)**
```python
dataset_name = "Set25"
```

#### **Option D: Dùng Set30 (30 jobs)**
```python
dataset_name = "Set30"
```

#### **Option E: Dùng Set35 (35 jobs)**
```python
dataset_name = "Set35"
```

#### **Option F: Dùng Set40 (40 jobs)**
```python
dataset_name = "Set40"
```

#### **Option G: Dùng Set45 (45 jobs)**
```python
dataset_name = "Set45"
```

#### **Option H: Dùng Set50 (50 jobs)**
```python
dataset_name = "Set50"
```

---

### **Bước 3: Chạy training như bình thường**

```bash
python run_training.py
```

Hoặc:

```bash
python scripts/train_lgp.py
```

---

## 📊 XEM THÔNG TIN DATASET

### **List tất cả datasets:**

```bash
python environment/dataset_loader.py
```

Output:
```
📚 Available Datasets:
==================================================
  - Set20
  - Set25
  - Set30
  - Set35
  - Set40
  - Set45
  - Set50
==================================================
```

---

### **Xem chi tiết 1 dataset:**

Mở Python console:

```python
from environment.dataset_loader import print_dataset_info

# Xem thông tin Set20
print_dataset_info("Set20")
```

Output:
```
📊 Dataset Info: Set20
==================================================
  Total Jobs:        20
  Total Machines:    12
  Machine IDs:       [0, 1, 3, 4, 5, 6, 7, 8, 12, 25, 43, 1200]
  Total Operations:  31
  Avg Ops/Job:       1.55
  Unique Due Dates:  1
==================================================
```

---

## 🔧 ADVANCED: LOAD DATASET TRONG CODE

Nếu bạn muốn load dataset trực tiếp trong code:

```python
from environment.dataset_loader import load_dataset

# Load Set30
jobs, due_dates, machine_pool = load_dataset("Set30")

print(f"Loaded {len(jobs)} jobs")
print(f"Machine pool: {machine_pool}")
```

---

## ⚠️ LƯU Ý QUAN TRỌNG

### **1. Dataset Size vs Training Time**

| Dataset | Jobs | Approx. Training Time | Complexity |
|---------|------|----------------------|------------|
| Set20   | 20   | Baseline (1x)        | Low        |
| Set25   | 25   | 1.3x                 | Low-Med    |
| Set30   | 30   | 1.5x                 | Medium     |
| Set35   | 35   | 1.8x                 | Med-High   |
| Set40   | 40   | 2.0x                 | High       |
| Set45   | 45   | 2.3x                 | High       |
| Set50   | 50   | 2.5x                 | Very High  |

⚠️ **Datasets lớn hơn = thời gian training lâu hơn!**

---

### **2. Hyperparameter Tuning**

Khi chuyển sang dataset lớn hơn, bạn NÊN điều chỉnh:

#### **Cho Set30-35:**
```python
# config.py
CoevolutionConfig.episodes_per_gen = 500  # Tăng từ 400
LGPConfig.action_budget_s = 4.0  # Tăng từ 3.0 (MH cần thời gian hơn)
```

#### **Cho Set40-50:**
```python
# config.py
CoevolutionConfig.episodes_per_gen = 600  # Tăng nhiều hơn
LGPConfig.action_budget_s = 5.0  # Cho MH đủ thời gian
CoevolutionConfig.num_generations = 25  # Tăng số generations
```

---

### **3. Fallback Safety**

Nếu file dataset **không tồn tại** hoặc **có lỗi**, hệ thống sẽ tự động:
- ⚠️ In cảnh báo
- ✅ Fallback về default hardcoded dataset (20 jobs)
- ✅ Tiếp tục training bình thường

**Ví dụ:**
```python
dataset_name = "Set99"  # File không tồn tại

# Output:
# ⚠️ Dataset file not found: data/Set99.json
#    Falling back to hardcoded default dataset
# ✅ Using hardcoded default dataset (20 jobs)
```

---

## 🧪 TEST DATASETS

### **Quick Test với tất cả datasets:**

```bash
python environment/dataset_loader.py
```

Script sẽ:
1. List tất cả datasets
2. Load từng dataset
3. Verify dữ liệu hợp lệ
4. In thông tin

---

## 📝 VÍ DỤ THỰC TẾ

### **Scenario 1: Training với Set25**

```python
# config.py
class EnvironmentConfig:
    dataset_name = "Set25"  # ← Thay đổi dòng này
    lambda_tardiness = 1.0
    num_dynamic_jobs = 2
```

```bash
python run_training.py
```

Output:
```
✅ Loaded dataset: Set25
🏭 Creating scheduling environment...
✓ Environment created with 25 initial jobs
...
```

---

### **Scenario 2: So sánh performance giữa datasets**

```python
# Test 1: Set20
EnvironmentConfig.dataset_name = "Set20"
# Run training → Save results as "results_set20/"

# Test 2: Set30
EnvironmentConfig.dataset_name = "Set30"
# Run training → Save results as "results_set30/"

# Compare makespan, tardiness, etc.
```

---

### **Scenario 3: Progressive training**

Train từ dataset nhỏ → lớn:

```python
# Week 1: Set20 (learn basics)
dataset_name = "Set20"
num_generations = 20

# Week 2: Set30 (scale up)
dataset_name = "Set30"
num_generations = 25

# Week 3: Set50 (final challenge)
dataset_name = "Set50"
num_generations = 30
```

---

## 🎯 KHUYẾN NGHỊ

### **Cho nghiên cứu/testing:**
- ✅ Dùng **Set20** hoặc **Set25** (nhanh, dễ debug)

### **Cho experiments:**
- ✅ Dùng **Set30** hoặc **Set35** (balance tốt)

### **Cho final results/paper:**
- ✅ Dùng **Set40-50** (thách thức, impressive)

### **Cho quick debugging:**
- ✅ Dùng **None** (default hardcoded, fastest)

---

## 🐛 TROUBLESHOOTING

### **Problem 1: Dataset không load được**

```
❌ Error loading dataset: [Errno 2] No such file or directory
```

**Solution:**
- Kiểm tra file tồn tại trong `data/` directory
- Kiểm tra tên file đúng format: `SetXX.json`
- Dùng `dataset_name = None` để dùng default

---

### **Problem 2: JSON format error**

```
❌ Error loading dataset: Expecting property name enclosed in double quotes
```

**Solution:**
- File JSON bị lỗi format
- Kiểm tra syntax JSON (dùng jsonlint.com)
- Hoặc fallback về default: `dataset_name = None`

---

### **Problem 3: Training quá chậm**

**Solution:**
- Giảm `episodes_per_gen`
- Giảm `num_generations`
- Chuyển sang dataset nhỏ hơn
- Giảm `action_budget_s`

---

## 📚 THÊM DATASET MỚI

Nếu bạn muốn thêm dataset riêng:

### **Bước 1: Tạo file JSON**

```json
{
  "name": "MyCustomSet",
  "machine_pool": [1, 2, 3, 4, 5],
  "jobs": {
    "1": [
      {
        "op_id": 1,
        "candidate_machines": [1, 2],
        "processing_time": 10
      }
    ],
    "2": [...]
  },
  "due_dates": {
    "1": 1000,
    "2": 1000
  }
}
```

### **Bước 2: Lưu vào `data/MyCustomSet.json`**

### **Bước 3: Sử dụng**

```python
# config.py
EnvironmentConfig.dataset_name = "MyCustomSet"
```

---

## 🎉 TÓM LẠI

1. ✅ **Chỉ cần thay đổi 1 dòng trong `config.py`**
2. ✅ **7 datasets sẵn có (20-50 jobs)**
3. ✅ **Auto fallback nếu có lỗi**
4. ✅ **Easy to add custom datasets**
5. ✅ **Backward compatible với code cũ**

**Happy experimenting! 🚀**
