"""
Phase 2 Improved: PPO Training with Fixed Portfolios
Improvements:
1. Larger network (128 neurons)
2. Higher entropy coefficient (0.05)
3. More episodes (5000)
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import RANDOM_SEED, PPOConfig, EnvironmentConfig
from training.portfolio_types import ActionIndividual, Gene
from environment.scheduling_env import DynamicSchedulingEnv
import registries.dispatching_rules
import registries.metaheuristics_impl


# Improved PPO Model with larger network
class PPOActorCriticLarge(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_size // 2, act_dim)
        self.value_head = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x)


def load_top_portfolios(filepath="results/top_portfolios_phase2.json"):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    action_library = []
    for p in data['portfolios']:
        genes = [Gene(kind='DR', name=p['dr'], w_raw=1.0)]
        for mh in p['mh_genes']:
            genes.append(Gene(kind='MH', name=mh['name'], w_raw=mh['weight']))
        action_library.append(ActionIndividual(genes=genes))
    
    return action_library


def select_action(model, state, deterministic=False):
    state_t = torch.FloatTensor(state).unsqueeze(0)
    logits, value = model(state_t)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    
    if deterministic:
        action = probs.argmax(dim=-1)
    else:
        action = dist.sample()
    
    return action.item(), dist.log_prob(action), value


def compute_returns(rewards, masks, gamma=0.99):
    returns = []
    R = 0
    for r, mask in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * mask
        returns.insert(0, R)
    return returns


def plot_gantt(schedule_events, title, save_path):
    colors = plt.cm.tab20.colors
    job_colors = {}
    
    machines = {}
    for event in schedule_events:
        m = event['machine']
        machines.setdefault(m, []).append(event)
    
    machine_ids = sorted(machines.keys())
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, m in enumerate(machine_ids):
        for event in sorted(machines[m], key=lambda e: e['start']):
            job = str(event['job'])
            if job not in job_colors:
                job_colors[job] = colors[len(job_colors) % len(colors)]
            
            duration = event['finish'] - event['start']
            ax.barh(i, duration, left=event['start'], height=0.6, 
                   color=job_colors[job], edgecolor='black', linewidth=0.5)
            if duration > 5:
                ax.text(event['start'] + duration/2, i, job, 
                       ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_yticks(range(len(machine_ids)))
    ax.set_yticklabels([f"M{m}" for m in machine_ids])
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_phase2_improved(num_episodes=5000):
    print("=" * 60)
    print("PHASE 2 IMPROVED: PPO WITH LARGER NETWORK")
    print("=" * 60)
    
    # Improved hyperparameters
    HIDDEN_SIZE = 128
    ENTROPY_COEF = 0.05  # Increased from 0.01
    GAMMA = 0.99
    LR = 3e-4
    CLIP_EPSILON = 0.2
    PPO_EPOCHS = 4
    
    print(f"Improvements:")
    print(f"  - Hidden size: 64 -> {HIDDEN_SIZE}")
    print(f"  - Entropy coef: 0.01 -> {ENTROPY_COEF}")
    print(f"  - Episodes: 2000 -> {num_episodes}")
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    action_library = load_top_portfolios()
    print(f"\n✓ Loaded {len(action_library)} portfolios")
    
    env = DynamicSchedulingEnv(
        lambda_tardiness=1.0,
        action_library=action_library,
        dataset_name=EnvironmentConfig.dataset_name
    )
    
    obs_dim = env.observation_space.shape[0]
    act_dim = len(action_library)
    
    model = PPOActorCriticLarge(obs_dim, act_dim, hidden_size=HIDDEN_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"✓ Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    all_returns, all_makespans = [], []
    best_return = float('-inf')
    best_makespan = float('inf')
    best_schedule = None
    best_ep = 0
    
    os.makedirs("results/phase2_improved", exist_ok=True)
    
    print(f"\n🚀 Training {num_episodes} episodes...")
    
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
        metrics = env.get_metrics()
        makespan = metrics['makespan']
        
        all_returns.append(ep_return)
        all_makespans.append(makespan)
        
        if ep_return > best_return:
            best_return = ep_return
            torch.save(model.state_dict(), "results/phase2_improved/best_model.pth")
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = [dict(e) for e in env.current_schedule_events]
            best_ep = ep + 1
        
        # PPO Update
        returns = compute_returns(rewards, masks, GAMMA)
        
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        old_log_probs = torch.stack(log_probs).detach()
        returns_t = torch.FloatTensor(returns)
        values_t = torch.stack(values).squeeze().detach()
        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(PPO_EPOCHS):
            logits, new_values = model(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns_t - new_values.squeeze()).pow(2).mean()
            entropy = dist.entropy().mean()
            
            loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        if (ep + 1) % 100 == 0:
            avg_ret = np.mean(all_returns[-100:])
            avg_ms = np.mean(all_makespans[-100:])
            print(f"[Ep {ep+1}] Return: {ep_return:.1f} | MS: {makespan} | "
                  f"Avg Ret: {avg_ret:.1f} | Avg MS: {avg_ms:.1f} | Best MS: {best_makespan}")
    
    # Save results
    torch.save(model.state_dict(), "results/phase2_improved/final_model.pth")
    
    with open("results/phase2_improved/metrics.json", 'w') as f:
        json.dump({
            'returns': all_returns, 'makespans': all_makespans,
            'best_return': best_return, 'best_makespan': best_makespan,
            'best_episode': best_ep
        }, f)
    
    with open("results/phase2_improved/best_schedule.json", 'w') as f:
        json.dump({'makespan': best_makespan, 'episode': best_ep, 'schedule': best_schedule}, f)
    
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(all_returns, alpha=0.3)
    smoothed = np.convolve(all_returns, np.ones(100)/100, mode='valid')
    axes[0].plot(range(99, len(all_returns)), smoothed, 'r-', linewidth=2)
    axes[0].axhline(y=best_return, color='g', linestyle='--')
    axes[0].set_title('Phase 2 Improved: Returns')
    axes[0].set_xlabel('Episode')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(all_makespans, alpha=0.3)
    smoothed_ms = np.convolve(all_makespans, np.ones(100)/100, mode='valid')
    axes[1].plot(range(99, len(all_makespans)), smoothed_ms, 'r-', linewidth=2)
    axes[1].axhline(y=best_makespan, color='g', linestyle='--')
    axes[1].set_title('Phase 2 Improved: Makespan')
    axes[1].set_xlabel('Episode')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/phase2_improved/training_curves.png", dpi=150)
    plt.close()
    
    plot_gantt(best_schedule, f"Best Schedule (Makespan={best_makespan}, Ep={best_ep})",
               "results/phase2_improved/best_gantt.png")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Return: {best_return:.2f}")
    print(f"Best Makespan: {best_makespan} (Episode {best_ep})")
    print(f"Avg Makespan (last 100): {np.mean(all_makespans[-100:]):.2f}")
    print(f"\nOutputs: results/phase2_improved/")


if __name__ == "__main__":
    train_phase2_improved(num_episodes=20000)
