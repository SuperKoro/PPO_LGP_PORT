#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detailed analysis of makespan performance issues
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

print("=" * 80)
print("📊 PHÂN TÍCH VẤN ĐỀ MAKESPAN")
print("=" * 80)

# Load all generation metrics
metrics_dir = Path("results/metrics")
generations = []

for gen_file in sorted(metrics_dir.glob("generation_*.json")):
    with open(gen_file, 'r') as f:
        data = json.load(f)
        gen_num = data['generation']
        agg = data['aggregated_metrics']
        
        generations.append({
            'gen': gen_num,
            'avg_makespan': agg['avg_makespan'],
            'std_makespan': agg['std_makespan'],
            'avg_return': agg['avg_return'],
            'avg_policy_loss': agg['avg_policy_loss'],
            'avg_value_loss': agg['avg_value_loss']
        })

# Convert to arrays
gen_nums = [g['gen'] for g in generations]
avg_makespans = [g['avg_makespan'] for g in generations]
std_makespans = [g['std_makespan'] for g in generations]

print(f"\n📈 MAKESPAN QUA CÁC GENERATIONS:")
print("=" * 80)
print(f"{'Gen':<6} {'Avg Makespan':<15} {'Std':<10} {'Change':<10} {'Status'}")
print("-" * 80)

for i, g in enumerate(generations):
    if i == 0:
        change = 0
        status = "Baseline"
    else:
        change = g['avg_makespan'] - generations[i-1]['avg_makespan']
        if change < -1:
            status = "✅ Better"
        elif change > 1:
            status = "❌ Worse"
        else:
            status = "~ Stable"
    
    print(f"{g['gen']:<6} {g['avg_makespan']:<15.2f} {g['std_makespan']:<10.2f} {change:>+9.2f} {status}")

# Statistics
initial_makespan = avg_makespans[0]
final_makespan = avg_makespans[-1]
best_makespan = min(avg_makespans)
worst_makespan = max(avg_makespans)
improvement = initial_makespan - final_makespan

print("\n" + "=" * 80)
print("📊 SUMMARY STATISTICS:")
print("=" * 80)
print(f"  Initial (Gen 1):     {initial_makespan:.2f}")
print(f"  Final (Gen 20):      {final_makespan:.2f}")
print(f"  Best (any gen):      {best_makespan:.2f}")
print(f"  Worst (any gen):     {worst_makespan:.2f}")
print(f"  Total improvement:   {improvement:+.2f} ({improvement/initial_makespan*100:+.1f}%)")
print(f"  Avg std deviation:   {np.mean(std_makespans):.2f}")

# Trend analysis
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(gen_nums, avg_makespans)
print(f"\n  Trend (slope):       {slope:+.3f} per generation")
print(f"  R²:                  {r_value**2:.3f}")
if abs(slope) < 0.1:
    trend = "❌ NO CONVERGENCE (flat)"
elif slope < 0:
    trend = f"✅ Improving (slope: {slope:.3f})"
else:
    trend = f"❌ DEGRADING (slope: {slope:.3f})"
print(f"  Status:              {trend}")

print("\n" + "=" * 80)
print("🔍 VẤNĐỀ PHÁT HIỆN:")
print("=" * 80)

issues = []

# Issue 1: No improvement
if abs(improvement) < 5:
    issues.append({
        'severity': 'CRITICAL',
        'issue': 'Không có cải thiện makespan',
        'detail': f'Chỉ {improvement:.2f} improvement sau 20 generations',
        'impact': 'PPO không học được cách chọn portfolio tốt hơn'
    })

# Issue 2: High variance
if np.mean(std_makespans) > 40:
    issues.append({
        'severity': 'HIGH',
        'issue': 'Variance quá cao',
        'detail': f'Avg std = {np.mean(std_makespans):.2f} (>40)',
        'impact': 'Environment quá stochastic, fitness không reliable'
    })

# Issue 3: Trend
if slope > 0:
    issues.append({
        'severity': 'CRITICAL',
        'issue': 'Makespan đang TỒI LÊN',
        'detail': f'Slope = {slope:+.3f} (dương)',
        'impact': 'Model đang học sai hướng hoặc overfitting'
    })
elif abs(slope) < 0.1:
    issues.append({
        'severity': 'HIGH',
        'issue': 'Không có convergence',
        'detail': f'Slope = {slope:+.3f} (gần 0)',
        'impact': 'Learning rate quá thấp hoặc policy collapse'
    })

# Issue 4: Policy loss
policy_losses = [g['avg_policy_loss'] for g in generations]
if np.mean(policy_losses[-5:]) < 0.01:
    issues.append({
        'severity': 'HIGH',
        'issue': 'Policy loss gần 0',
        'detail': f'Avg last 5 gens: {np.mean(policy_losses[-5:]):.4f}',
        'impact': 'Policy đã collapse, không explore nữa'
    })

for i, issue in enumerate(issues, 1):
    print(f"\n❌ Issue {i}: {issue['issue']} ({issue['severity']})")
    print(f"   Detail: {issue['detail']}")
    print(f"   Impact: {issue['impact']}")

print("\n" + "=" * 80)
print("💡 NGUYÊN NHÂN GỐC RỄ:")
print("=" * 80)

root_causes = [
    "1. Reward function chỉ optimize makespan, bỏ qua tardiness",
    "2. Environment variance quá cao (random dynamic jobs)",
    "3. Policy collapse - PPO chỉ dùng 1-2 portfolios",
    "4. LGP evolution phá hỏng good programs",
    "5. Learning rate decay quá nhanh",
    "6. Observation space quá đơn giản (chỉ 3 features)",
    "7. Entropy coefficient quá thấp → không explore",
    "8. Episodes per generation không đủ để estimate fitness"
]

for cause in root_causes:
    print(f"  {cause}")

print("\n" + "=" * 80)
print("🎯 SAVED ANALYSIS TO: makespan_analysis_result.txt")
print("=" * 80)

# Save to file
with open("makespan_analysis_result.txt", 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("MAKESPAN ANALYSIS RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Initial makespan: {initial_makespan:.2f}\n")
    f.write(f"Final makespan: {final_makespan:.2f}\n")
    f.write(f"Improvement: {improvement:+.2f} ({improvement/initial_makespan*100:+.1f}%)\n")
    f.write(f"Trend slope: {slope:+.3f}\n")
    f.write(f"Status: {trend}\n\n")
    f.write("Issues:\n")
    for i, issue in enumerate(issues, 1):
        f.write(f"{i}. {issue['issue']} ({issue['severity']})\n")
        f.write(f"   {issue['detail']}\n\n")
