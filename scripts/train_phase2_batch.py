"""
Phase 2 with Batch PPO: Collect data from multiple episodes before updating.
This reduces variance and helps PPO generalize across different problem instances.
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

from config import RANDOM_SEED, EnvironmentConfig
from training.portfolio_types import ActionIndividual, Gene
from environment.scheduling_env import DynamicSchedulingEnv
import registries.dispatching_rules
import registries.metaheuristics_impl


class PPOActorCriticLarge(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x)


def load_portfolios():
    with open('results/top_portfolios_phase2.json', 'r') as f:
        data = json.load(f)
    
    action_library = []
    for p in data['portfolios']:
        genes = [Gene(kind='DR', name=p['dr'], w_raw=1.0)]
        for mh in p['mh_genes']:
            genes.append(Gene(kind='MH', name=mh['name'], w_raw=mh['weight']))
        action_library.append(ActionIndividual(genes=genes))
    return action_library


def train_batch_ppo(num_episodes=10000, batch_size=10):
    """
    Batch PPO: Collect data from batch_size episodes, then do one PPO update.
    """
    print("=" * 60)
    print("PHASE 2 BATCH PPO")
    print("=" * 60)
    
    # Hyperparameters
    HIDDEN_SIZE = 128
    ENTROPY_COEF = 0.05
    GAMMA = 0.99
    LR = 3e-4
    CLIP_EPSILON = 0.2
    PPO_EPOCHS = 4
    
    print(f"Config:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Batch size: {batch_size} episodes/update")
    print(f"  Total updates: {num_episodes // batch_size}")
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    action_library = load_portfolios()
    
    env = DynamicSchedulingEnv(
        lambda_tardiness=1.0,
        action_library=action_library,
        dataset_name=EnvironmentConfig.dataset_name
    )
    
    obs_dim = env.observation_space.shape[0]
    act_dim = len(action_library)
    
    model = PPOActorCriticLarge(obs_dim, act_dim, HIDDEN_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"✓ Model: {sum(p.numel() for p in model.parameters())} params")
    print(f"✓ Action space: {act_dim}")
    
    all_returns, all_makespans = [], []
    best_return = float('-inf')
    best_makespan = float('inf')
    best_schedule = None
    best_ep = 0
    
    os.makedirs("results/phase2_batch", exist_ok=True)
    
    print(f"\n🚀 Training...")
    
    # Batch buffers
    batch_states = []
    batch_actions = []
    batch_log_probs = []
    batch_returns = []
    batch_advantages = []
    
    for ep in range(num_episodes):
        env.seed(RANDOM_SEED + ep)
        state = env.reset()
        
        ep_states, ep_actions, ep_log_probs, ep_values, ep_rewards, ep_masks = [], [], [], [], [], []
        
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            logits, value = model(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            next_state, reward, done, _ = env.step(action.item())
            
            ep_states.append(state)
            ep_actions.append(action.item())
            ep_log_probs.append(dist.log_prob(action))
            ep_values.append(value)
            ep_rewards.append(reward)
            ep_masks.append(1 - float(done))
            
            state = next_state
        
        # Compute returns for this episode
        returns = []
        R = 0
        for r, m in zip(reversed(ep_rewards), reversed(ep_masks)):
            R = r + GAMMA * R * m
            returns.insert(0, R)
        
        returns_t = torch.FloatTensor(returns)
        values_t = torch.stack(ep_values).squeeze()
        advantages = returns_t - values_t.detach()
        
        # Add to batch
        batch_states.extend(ep_states)
        batch_actions.extend(ep_actions)
        batch_log_probs.extend(ep_log_probs)
        batch_returns.extend(returns)
        batch_advantages.extend(advantages.tolist())
        
        # Track metrics
        ep_return = sum(ep_rewards)
        metrics = env.get_metrics()
        makespan = metrics['makespan']
        
        all_returns.append(ep_return)
        all_makespans.append(makespan)
        
        if ep_return > best_return:
            best_return = ep_return
            torch.save(model.state_dict(), "results/phase2_batch/best_model.pth")
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = [dict(e) for e in env.current_schedule_events]
            best_ep = ep + 1
        
        # PPO update every batch_size episodes
        if (ep + 1) % batch_size == 0:
            # Prepare tensors
            states_t = torch.FloatTensor(np.array(batch_states))
            actions_t = torch.LongTensor(batch_actions)
            old_log_probs = torch.stack(batch_log_probs).detach()
            returns_t = torch.FloatTensor(batch_returns)
            advantages_t = torch.FloatTensor(batch_advantages)
            
            # Normalize advantages
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
            # PPO epochs
            for _ in range(PPO_EPOCHS):
                logits, new_values = model(states_t)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantages_t
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (returns_t - new_values.squeeze()).pow(2).mean()
                entropy = dist.entropy().mean()
                
                loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            # Clear batch
            batch_states, batch_actions, batch_log_probs = [], [], []
            batch_returns, batch_advantages = [], []
        
        if (ep + 1) % 500 == 0:
            avg_ret = np.mean(all_returns[-500:])
            avg_ms = np.mean(all_makespans[-500:])
            print(f"[Ep {ep+1}] Avg Ret: {avg_ret:.1f} | Avg MS: {avg_ms:.1f} | Best MS: {best_makespan}")
    
    # Save results
    torch.save(model.state_dict(), "results/phase2_batch/final_model.pth")
    
    with open("results/phase2_batch/metrics.json", 'w') as f:
        json.dump({
            'returns': all_returns, 'makespans': all_makespans,
            'best_return': best_return, 'best_makespan': best_makespan,
            'best_episode': best_ep
        }, f)
    
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    window = 100
    smoothed = np.convolve(all_returns, np.ones(window)/window, mode='valid')
    axes[0].plot(all_returns, alpha=0.2)
    axes[0].plot(range(window-1, len(all_returns)), smoothed, 'r-', linewidth=2)
    axes[0].axhline(y=best_return, color='g', linestyle='--')
    axes[0].set_title(f'Batch PPO: Returns (Best: {best_return:.1f})')
    axes[0].set_xlabel('Episode')
    axes[0].grid(alpha=0.3)
    
    smoothed_ms = np.convolve(all_makespans, np.ones(window)/window, mode='valid')
    axes[1].plot(all_makespans, alpha=0.2)
    axes[1].plot(range(window-1, len(all_makespans)), smoothed_ms, 'r-', linewidth=2)
    axes[1].axhline(y=best_makespan, color='g', linestyle='--')
    axes[1].set_title(f'Batch PPO: Makespan (Best: {best_makespan})')
    axes[1].set_xlabel('Episode')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/phase2_batch/training_curves.png", dpi=150)
    plt.close()
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Return: {best_return:.2f}")
    print(f"Best Makespan: {best_makespan} (Episode {best_ep})")
    print(f"Avg Makespan (last 500): {np.mean(all_makespans[-500:]):.2f}")
    print(f"\nOutputs: results/phase2_batch/")


if __name__ == "__main__":
    train_batch_ppo(num_episodes=10000, batch_size=32)
