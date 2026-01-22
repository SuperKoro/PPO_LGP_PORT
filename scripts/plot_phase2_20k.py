"""Plot Phase 2 Improved (20000 episodes) results."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load Phase 2 improved metrics (20000 eps)
with open('results/phase2_improved/metrics.json', 'r') as f:
    data = json.load(f)

returns = data['returns']
makespans = data['makespans']
best_return = data['best_return']
best_makespan = data['best_makespan']
best_ep = data['best_episode']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Return over episodes
window = 200
axes[0,0].plot(returns, alpha=0.2, color='blue', linewidth=0.5)
smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
axes[0,0].plot(range(window-1, len(returns)), smoothed, 'r-', linewidth=2, label=f'Moving Avg({window})')
axes[0,0].axhline(y=best_return, color='g', linestyle='--', linewidth=2, label=f'Best: {best_return:.1f}')
axes[0,0].set_xlabel('Episode')
axes[0,0].set_ylabel('Return')
axes[0,0].set_title('Phase 2 Improved (20000 eps): Return')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# Makespan over episodes
axes[0,1].plot(makespans, alpha=0.2, color='blue', linewidth=0.5)
smoothed_ms = np.convolve(makespans, np.ones(window)/window, mode='valid')
axes[0,1].plot(range(window-1, len(makespans)), smoothed_ms, 'r-', linewidth=2, label=f'Moving Avg({window})')
axes[0,1].axhline(y=best_makespan, color='g', linestyle='--', linewidth=2, label=f'Best: {best_makespan} (Ep {best_ep})')
axes[0,1].axhline(y=np.mean(makespans), color='orange', linestyle=':', label=f'Mean: {np.mean(makespans):.1f}')
axes[0,1].set_xlabel('Episode')
axes[0,1].set_ylabel('Makespan')
axes[0,1].set_title('Phase 2 Improved (20000 eps): Makespan')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# Histogram of makespans
axes[1,0].hist(makespans, bins=60, edgecolor='black', alpha=0.7, color='steelblue')
axes[1,0].axvline(x=np.mean(makespans), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(makespans):.1f}')
axes[1,0].axvline(x=best_makespan, color='g', linestyle='--', linewidth=2, label=f'Best: {best_makespan}')
axes[1,0].set_xlabel('Makespan')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Distribution of Makespans')
axes[1,0].legend()

# Summary text
summary_text = f'''TRAINING SUMMARY
================
Episodes: {len(returns):,}
Best Makespan: {best_makespan}
Best Episode: {best_ep}
Best Return: {best_return:.2f}

Average Makespan: {np.mean(makespans):.2f}
Std Makespan: {np.std(makespans):.2f}
Min Makespan: {min(makespans)}
Max Makespan: {max(makespans)}

Average Return: {np.mean(returns):.2f}
Std Return: {np.std(returns):.2f}
'''

axes[1,1].text(0.1, 0.5, summary_text, transform=axes[1,1].transAxes, 
               fontsize=12, verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
axes[1,1].axis('off')
axes[1,1].set_title('Training Statistics')

plt.suptitle('Phase 2 Improved: 20,000 Episodes Training Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/phase2_improved/training_results_20k.png', dpi=150, bbox_inches='tight')
print('Saved: results/phase2_improved/training_results_20k.png')

print(f'\nSummary:')
print(f'  Episodes: {len(returns):,}')
print(f'  Best Makespan: {best_makespan} (Episode {best_ep})')
print(f'  Avg Makespan: {np.mean(makespans):.2f}')
print(f'  Best Return: {best_return:.2f}')
