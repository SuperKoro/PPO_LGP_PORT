"""Plot makespan over iterations for Phase 2."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load both phase 2 results
with open('results/phase2/training_metrics.json', 'r') as f:
    data1 = json.load(f)

with open('results/phase2_improved/metrics.json', 'r') as f:
    data2 = json.load(f)

fig, ax = plt.subplots(figsize=(14, 6))

# Phase 2 original
ms1 = data1['makespans']
ax.plot(ms1, alpha=0.2, color='blue', linewidth=0.5)
window = 50
smoothed1 = np.convolve(ms1, np.ones(window)/window, mode='valid')
ax.plot(range(window-1, len(ms1)), smoothed1, 'b-', linewidth=2, 
        label=f'Phase 2 (2000 eps) - Avg: {np.mean(ms1):.1f}')

# Phase 2 improved  
ms2 = data2['makespans']
ax.plot(range(len(ms1), len(ms1)+len(ms2)), ms2, alpha=0.2, color='orange', linewidth=0.5)
smoothed2 = np.convolve(ms2, np.ones(window)/window, mode='valid')
ax.plot(range(len(ms1)+window-1, len(ms1)+len(ms2)), smoothed2, 'orange', linewidth=2, 
        label=f'Phase 2 Improved (5000 eps) - Avg: {np.mean(ms2):.1f}')

# Best lines
best1 = data1['best_makespan']
best2 = data2['best_makespan']
ax.axhline(y=best1, color='blue', linestyle='--', alpha=0.7, label=f'Best Phase 2: {best1}')
ax.axhline(y=best2, color='green', linestyle='--', alpha=0.7, label=f'Best Improved: {best2}')

ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Makespan', fontsize=12)
ax.set_title('Phase 2: Makespan over Training Iterations', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/phase2_makespan_plot.png', dpi=150)
print('Saved: results/phase2_makespan_plot.png')
plt.close()
