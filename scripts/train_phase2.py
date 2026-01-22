"""
Phase 2: PPO Training with Fixed Portfolios (No LGP)
Uses the top portfolios extracted from Phase 1 training.
Now includes metrics logging and best schedule tracking for Gantt chart.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import RANDOM_SEED, PPOConfig, EnvironmentConfig
from training.ppo_model import PPOActorCritic, select_action, compute_returns, ppo_update
from training.portfolio_types import ActionIndividual, Gene
from environment.scheduling_env import DynamicSchedulingEnv
import registries.dispatching_rules
import registries.metaheuristics_impl


def load_top_portfolios(filepath="results/top_portfolios_phase2.json"):
    """Load top portfolios from Phase 1."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    action_library = []
    for p in data['portfolios']:
        genes = [Gene(kind='DR', name=p['dr'], w_raw=1.0)]
        for mh in p['mh_genes']:
            genes.append(Gene(kind='MH', name=mh['name'], w_raw=mh['weight']))
        action_library.append(ActionIndividual(genes=genes))
    
    return action_library


def plot_gantt(schedule_events, title="Gantt Chart", save_path=None):
    """Plot Gantt chart from schedule events."""
    colors = plt.cm.tab20.colors
    job_colors = {}
    
    def get_job_color(job):
        job_str = str(job)
        if job_str not in job_colors:
            job_colors[job_str] = colors[len(job_colors) % len(colors)]
        return job_colors[job_str]
    
    # Group by machine
    machines = {}
    for event in schedule_events:
        m = event['machine']
        machines.setdefault(m, []).append(event)
    
    machine_ids = sorted(machines.keys())
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, m in enumerate(machine_ids):
        events = sorted(machines[m], key=lambda e: e['start'])
        for event in events:
            start = event['start']
            duration = event['finish'] - event['start']
            job = event['job']
            color = get_job_color(job)
            
            ax.barh(i, duration, left=start, height=0.6, align='center', 
                   color=color, edgecolor='black', linewidth=0.5)
            
            if duration > 5:
                ax.text(start + duration/2, i, str(job), 
                       ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    ax.set_yticks(range(len(machine_ids)))
    ax.set_yticklabels([f"M{m}" for m in machine_ids])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def train_phase2(num_episodes=2000, save_every=200):
    """Train PPO with fixed portfolios."""
    
    print("=" * 60)
    print("PHASE 2: PPO TRAINING WITH FIXED PORTFOLIOS")
    print("=" * 60)
    
    # Set seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Load portfolios
    action_library = load_top_portfolios()
    print(f"\n✓ Loaded {len(action_library)} portfolios from Phase 1")
    
    # Create environment
    env = DynamicSchedulingEnv(
        lambda_tardiness=1.0,
        action_library=action_library,
        dataset_name=EnvironmentConfig.dataset_name
    )
    
    obs_dim = env.observation_space.shape[0]
    act_dim = len(action_library)
    print(f"✓ Environment: obs_dim={obs_dim}, act_dim={act_dim}")
    
    # Create model
    model = PPOActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=PPOConfig.learning_rate)
    print(f"✓ PPO model created")
    
    # Training metrics
    print(f"\n🚀 Starting training for {num_episodes} episodes...")
    
    all_returns = []
    all_makespans = []
    best_return = float('-inf')
    best_makespan = float('inf')
    best_schedule = None
    best_schedule_ep = 0
    
    os.makedirs("results/phase2", exist_ok=True)
    
    for ep in range(num_episodes):
        env.seed(RANDOM_SEED + ep)
        state = env.reset()
        
        states, actions, log_probs, values, rewards, masks = [], [], [], [], [], []
        
        done = False
        while not done:
            action, log_prob, value = select_action(model, state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(1 - float(done))
            
            state = next_state
        
        ep_return = sum(rewards)
        all_returns.append(ep_return)
        
        # Get makespan
        metrics = env.get_metrics()
        makespan = metrics['makespan']
        all_makespans.append(makespan)
        
        # Track best
        if ep_return > best_return:
            best_return = ep_return
            torch.save(model.state_dict(), "results/phase2/best_model.pth")
        
        # Track best schedule (minimum makespan)
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = [dict(e) for e in env.current_schedule_events]  # Deep copy
            best_schedule_ep = ep + 1
        
        # PPO update
        returns = compute_returns(rewards, masks, gamma=PPOConfig.gamma)
        advantages = [r - v.item() for r, v in zip(returns, values)]
        
        ppo_update(model, optimizer, states, actions, log_probs, returns, advantages,
                   clip_epsilon=PPOConfig.clip_epsilon, ppo_epochs=PPOConfig.ppo_epochs)
        
        if (ep + 1) % 50 == 0:
            avg_return = np.mean(all_returns[-50:])
            avg_makespan = np.mean(all_makespans[-50:])
            print(f"[Ep {ep+1}] Return: {ep_return:.2f} | Makespan: {makespan} | "
                  f"Avg Return: {avg_return:.2f} | Best Makespan: {best_makespan}")
    
    # Save metrics
    metrics_data = {
        'returns': all_returns,
        'makespans': all_makespans,
        'best_return': best_return,
        'best_makespan': best_makespan,
        'best_schedule_episode': best_schedule_ep
    }
    
    with open("results/phase2/training_metrics.json", 'w') as f:
        json.dump(metrics_data, f)
    
    # Save best schedule
    with open("results/phase2/best_schedule.json", 'w') as f:
        json.dump({
            'episode': best_schedule_ep,
            'makespan': best_makespan,
            'schedule': best_schedule
        }, f, indent=2)
    
    # Plot results
    print("\n📊 Generating plots...")
    
    # Return plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(all_returns, alpha=0.5, linewidth=0.5)
    window = 50
    smoothed = np.convolve(all_returns, np.ones(window)/window, mode='valid')
    axes[0].plot(range(window-1, len(all_returns)), smoothed, 'r-', linewidth=2, label=f'Avg({window})')
    axes[0].axhline(y=best_return, color='g', linestyle='--', label=f'Best: {best_return:.2f}')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].set_title('Phase 2: Return over Episodes')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Makespan plot
    axes[1].plot(all_makespans, alpha=0.5, linewidth=0.5)
    smoothed_ms = np.convolve(all_makespans, np.ones(window)/window, mode='valid')
    axes[1].plot(range(window-1, len(all_makespans)), smoothed_ms, 'r-', linewidth=2, label=f'Avg({window})')
    axes[1].axhline(y=best_makespan, color='g', linestyle='--', label=f'Best: {best_makespan}')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Makespan')
    axes[1].set_title('Phase 2: Makespan over Episodes')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/phase2/training_curves.png", dpi=150)
    print("✓ Saved: results/phase2/training_curves.png")
    plt.close()
    
    # Gantt chart of best schedule
    plot_gantt(best_schedule, 
               title=f"Best Schedule (Makespan={best_makespan}, Episode={best_schedule_ep})",
               save_path="results/phase2/best_gantt_chart.png")
    
    # Final save
    torch.save(model.state_dict(), "results/phase2/final_model.pth")
    
    print("\n" + "=" * 60)
    print("✅ PHASE 2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best return: {best_return:.2f}")
    print(f"Best makespan: {best_makespan} (Episode {best_schedule_ep})")
    print(f"Final avg return (last 100): {np.mean(all_returns[-100:]):.2f}")
    print(f"\nOutputs saved to results/phase2/")


if __name__ == "__main__":
    train_phase2(num_episodes=2000)
