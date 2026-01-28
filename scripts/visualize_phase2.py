"""
Visualize Phase 2 training results.
Creates plots for returns, makespans, tardiness, and distributions.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # Load data
    with open('results/phase2/training_metrics.json', 'r') as f:
        data = json.load(f)

    returns = data['returns']
    makespans = data['makespans']
    best_return = data['best_return']
    best_makespan = data['best_makespan']
    policy_losses = data.get('policy_losses', [])
    value_losses = data.get('value_losses', [])
    total_losses = data.get('total_losses', [])
    
    # Load tardiness data (if available)
    tardiness_normal = data.get('tardiness_normal', [])
    tardiness_urgent = data.get('tardiness_urgent', [])
    has_tardiness = len(tardiness_normal) > 0 and len(tardiness_urgent) > 0
    
    if has_tardiness:
        total_tardiness = [n + u for n, u in zip(tardiness_normal, tardiness_urgent)]
        best_tardiness = min(total_tardiness)

    window = 50
    
    # Determine layout based on available data
    if has_tardiness:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Return plot
    axes[0,0].plot(returns, alpha=0.3, linewidth=0.5, color='blue')
    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
    axes[0,0].plot(range(window-1, len(returns)), smoothed, 'r-', linewidth=2, label=f'Moving Avg({window})')
    axes[0,0].axhline(y=best_return, color='g', linestyle='--', label=f'Best: {best_return:.1f}')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Return')
    axes[0,0].set_title('Return over Episodes')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)

    # Makespan plot
    axes[0,1].plot(makespans, alpha=0.3, linewidth=0.5, color='blue')
    smoothed_ms = np.convolve(makespans, np.ones(window)/window, mode='valid')
    axes[0,1].plot(range(window-1, len(makespans)), smoothed_ms, 'r-', linewidth=2, label=f'Moving Avg({window})')
    axes[0,1].axhline(y=best_makespan, color='g', linestyle='--', label=f'Best: {best_makespan}')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Makespan')
    axes[0,1].set_title('Makespan over Episodes')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)

    if has_tardiness:
        # Total tardiness plot
        axes[0,2].plot(total_tardiness, alpha=0.3, linewidth=0.5, color='red')
        smoothed_tard = np.convolve(total_tardiness, np.ones(window)/window, mode='valid')
        axes[0,2].plot(range(window-1, len(total_tardiness)), smoothed_tard, 'b-', linewidth=2, label=f'Moving Avg({window})')
        axes[0,2].axhline(y=best_tardiness, color='g', linestyle='--', label=f'Best: {best_tardiness:.1f}')
        axes[0,2].set_xlabel('Episode')
        axes[0,2].set_ylabel('Total Tardiness')
        axes[0,2].set_title('Total Tardiness over Episodes')
        axes[0,2].legend()
        axes[0,2].grid(alpha=0.3)

    # Histogram of returns
    axes[1,0].hist(returns, bins=50, edgecolor='black', alpha=0.7)
    axes[1,0].axvline(x=np.mean(returns), color='r', linestyle='--', label=f'Mean: {np.mean(returns):.1f}')
    axes[1,0].set_xlabel('Return')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Distribution of Returns')
    axes[1,0].legend()

    # Histogram of makespans
    axes[1,1].hist(makespans, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1,1].axvline(x=np.mean(makespans), color='r', linestyle='--', label=f'Mean: {np.mean(makespans):.1f}')
    axes[1,1].axvline(x=best_makespan, color='g', linestyle='--', label=f'Best: {best_makespan}')
    axes[1,1].set_xlabel('Makespan')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Makespans')
    axes[1,1].legend()

    if has_tardiness:
        # Histogram of tardiness
        axes[1,2].hist(total_tardiness, bins=50, edgecolor='black', alpha=0.7, color='red')
        axes[1,2].axvline(x=np.mean(total_tardiness), color='b', linestyle='--', label=f'Mean: {np.mean(total_tardiness):.1f}')
        axes[1,2].axvline(x=best_tardiness, color='g', linestyle='--', label=f'Best: {best_tardiness:.1f}')
        axes[1,2].set_xlabel('Total Tardiness')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Distribution of Tardiness')
        axes[1,2].legend()

    plt.suptitle('Phase 2 Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/phase2/phase2_results_overview.png', dpi=150, bbox_inches='tight')
    print('Saved: results/phase2/phase2_results_overview.png')
    plt.close()

    # Loss plots (separate figures for each loss type)
    if policy_losses and value_losses:
        # Policy Loss plot
        fig_policy, ax_policy = plt.subplots(1, 1, figsize=(10, 5))
        ax_policy.plot(policy_losses, alpha=0.5, linewidth=0.5, color='blue')
        smoothed_policy = np.convolve(policy_losses, np.ones(window)/window, mode='valid')
        ax_policy.plot(range(window-1, len(policy_losses)), smoothed_policy, 'r-', linewidth=2, label=f'Moving Avg({window})')
        ax_policy.set_xlabel('Episode')
        ax_policy.set_ylabel('Policy Loss')
        ax_policy.set_title('PPO Policy Loss over Episodes')
        ax_policy.grid(alpha=0.3)
        ax_policy.legend()
        plt.tight_layout()
        plt.savefig('results/phase2/phase2_policy_loss.png', dpi=150, bbox_inches='tight')
        print('Saved: results/phase2/phase2_policy_loss.png')
        plt.close()

        # Value Loss plot
        fig_value, ax_value = plt.subplots(1, 1, figsize=(10, 5))
        ax_value.plot(value_losses, alpha=0.5, linewidth=0.5, color='orange')
        smoothed_value = np.convolve(value_losses, np.ones(window)/window, mode='valid')
        ax_value.plot(range(window-1, len(value_losses)), smoothed_value, 'r-', linewidth=2, label=f'Moving Avg({window})')
        ax_value.set_xlabel('Episode')
        ax_value.set_ylabel('Value Loss')
        ax_value.set_title('PPO Value Loss over Episodes')
        ax_value.grid(alpha=0.3)
        ax_value.legend()
        plt.tight_layout()
        plt.savefig('results/phase2/phase2_value_loss.png', dpi=150, bbox_inches='tight')
        print('Saved: results/phase2/phase2_value_loss.png')
        plt.close()
    
    # Separate tardiness breakdown plot (if available)
    if has_tardiness:
        fig_tard, axes_tard = plt.subplots(1, 2, figsize=(14, 5))
        
        # Normal vs Urgent tardiness over episodes
        axes_tard[0].plot(tardiness_normal, alpha=0.3, linewidth=0.5, color='blue', label='Normal (raw)')
        axes_tard[0].plot(tardiness_urgent, alpha=0.3, linewidth=0.5, color='red', label='Urgent (raw)')
        smoothed_n = np.convolve(tardiness_normal, np.ones(window)/window, mode='valid')
        smoothed_u = np.convolve(tardiness_urgent, np.ones(window)/window, mode='valid')
        axes_tard[0].plot(range(window-1, len(tardiness_normal)), smoothed_n, 'b-', linewidth=2, label=f'Normal Avg({window})')
        axes_tard[0].plot(range(window-1, len(tardiness_urgent)), smoothed_u, 'r-', linewidth=2, label=f'Urgent Avg({window})')
        axes_tard[0].set_xlabel('Episode')
        axes_tard[0].set_ylabel('Tardiness')
        axes_tard[0].set_title('Tardiness Breakdown over Episodes')
        axes_tard[0].legend()
        axes_tard[0].grid(alpha=0.3)
        
        # Stacked histogram
        axes_tard[1].hist([tardiness_normal, tardiness_urgent], bins=50, stacked=True, 
                         edgecolor='black', alpha=0.7, color=['blue', 'red'], 
                         label=['Normal', 'Urgent'])
        axes_tard[1].set_xlabel('Tardiness')
        axes_tard[1].set_ylabel('Frequency')
        axes_tard[1].set_title('Distribution of Normal vs Urgent Tardiness')
        axes_tard[1].legend()
        
        plt.tight_layout()
        plt.savefig('results/phase2/phase2_tardiness_breakdown.png', dpi=150, bbox_inches='tight')
        print('Saved: results/phase2/phase2_tardiness_breakdown.png')
        plt.close()

    print(f'\nSummary:')
    print(f'  Episodes: {len(returns)}')
    print(f'  Best Return: {best_return:.2f}')
    print(f'  Avg Return: {np.mean(returns):.2f}')
    print(f'  Best Makespan: {best_makespan}')
    print(f'  Avg Makespan: {np.mean(makespans):.2f}')
    if has_tardiness:
        print(f'  Best Total Tardiness: {best_tardiness:.2f}')
        print(f'  Avg Total Tardiness: {np.mean(total_tardiness):.2f}')
        print(f'  Avg Normal Tardiness: {np.mean(tardiness_normal):.2f}')
        print(f'  Avg Urgent Tardiness: {np.mean(tardiness_urgent):.2f}')
    if policy_losses and value_losses and total_losses:
        print(f'  Avg Policy Loss: {np.mean(policy_losses):.4f}')
        print(f'  Avg Value Loss: {np.mean(value_losses):.4f}')
        print(f'  Avg Total Loss: {np.mean(total_losses):.4f}')


if __name__ == "__main__":
    main()
