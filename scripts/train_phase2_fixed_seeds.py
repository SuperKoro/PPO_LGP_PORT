"""
Phase 2 with Fixed Seeds: Train on 100 fixed problem instances.
This allows PPO to learn and converge by seeing the same problems repeatedly.
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


class PPOActorCritic(nn.Module):
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


def train_fixed_seeds(num_epochs=200, num_train_seeds=100, num_test_seeds=50):
    """
    Train on fixed seeds:
    - Each epoch cycles through all num_train_seeds problems
    - PPO sees same problems repeatedly and can learn optimal actions
    """
    print("=" * 60)
    print("PHASE 2 FIXED SEEDS: PPO CAN CONVERGE!")
    print("=" * 60)
    
    # Config
    HIDDEN_SIZE = 128
    ENTROPY_COEF = 0.02
    GAMMA = 0.99
    LR = 3e-4
    CLIP_EPSILON = 0.2
    PPO_EPOCHS = 4
    
    print(f"Config:")
    print(f"  Training seeds: {num_train_seeds}")
    print(f"  Test seeds: {num_test_seeds}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Total episodes: {num_epochs * num_train_seeds}")
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Fixed seeds for training and testing
    train_seeds = list(range(RANDOM_SEED, RANDOM_SEED + num_train_seeds))
    test_seeds = list(range(RANDOM_SEED + 1000, RANDOM_SEED + 1000 + num_test_seeds))
    
    action_library = load_portfolios()
    
    env = DynamicSchedulingEnv(
        lambda_tardiness=1.0,
        action_library=action_library,
        dataset_name=EnvironmentConfig.dataset_name
    )
    
    obs_dim = env.observation_space.shape[0]
    act_dim = len(action_library)
    
    model = PPOActorCritic(obs_dim, act_dim, HIDDEN_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"✓ Model: {sum(p.numel() for p in model.parameters())} params")
    print(f"✓ Action space: {act_dim}")
    
    os.makedirs("results/phase2_fixed_seeds", exist_ok=True)
    
    # Training metrics
    epoch_train_makespans = []
    epoch_test_makespans = []
    best_test_makespan = float('inf')
    
    print(f"\n🚀 Training...")
    
    for epoch in range(num_epochs):
        # Shuffle training order each epoch
        shuffled_seeds = train_seeds.copy()
        random.shuffle(shuffled_seeds)
        
        # Collect data from all training seeds
        all_states, all_actions, all_log_probs, all_returns, all_advantages = [], [], [], [], []
        train_makespans = []
        
        for seed in shuffled_seeds:
            env.seed(seed)
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
            
            # Compute returns
            returns = []
            R = 0
            for r, m in zip(reversed(ep_rewards), reversed(ep_masks)):
                R = r + GAMMA * R * m
                returns.insert(0, R)
            
            returns_t = torch.FloatTensor(returns)
            values_t = torch.stack(ep_values).squeeze()
            advantages = returns_t - values_t.detach()
            
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            all_log_probs.extend(ep_log_probs)
            all_returns.extend(returns)
            all_advantages.extend(advantages.tolist())
            
            train_makespans.append(env.get_metrics()['makespan'])
        
        # PPO update on all collected data
        states_t = torch.FloatTensor(np.array(all_states))
        actions_t = torch.LongTensor(all_actions)
        old_log_probs = torch.stack(all_log_probs).detach()
        returns_t = torch.FloatTensor(all_returns)
        advantages_t = torch.FloatTensor(all_advantages)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
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
        
        avg_train_ms = np.mean(train_makespans)
        epoch_train_makespans.append(avg_train_ms)
        
        # Test on held-out seeds
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            test_makespans = []
            
            with torch.no_grad():
                for seed in test_seeds:
                    env.seed(seed)
                    state = env.reset()
                    done = False
                    
                    while not done:
                        state_t = torch.FloatTensor(state).unsqueeze(0)
                        logits, _ = model(state_t)
                        action = logits.argmax(dim=-1).item()  # Greedy
                        state, _, done, _ = env.step(action)
                    
                    test_makespans.append(env.get_metrics()['makespan'])
            
            avg_test_ms = np.mean(test_makespans)
            epoch_test_makespans.append((epoch, avg_test_ms))
            
            if avg_test_ms < best_test_makespan:
                best_test_makespan = avg_test_ms
                torch.save(model.state_dict(), "results/phase2_fixed_seeds/best_model.pth")
            
            model.train()
            
            print(f"[Epoch {epoch+1:3d}] Train MS: {avg_train_ms:.1f} | Test MS: {avg_test_ms:.1f} | Best Test: {best_test_makespan:.1f}")
    
    # Save results
    torch.save(model.state_dict(), "results/phase2_fixed_seeds/final_model.pth")
    
    with open("results/phase2_fixed_seeds/metrics.json", 'w') as f:
        json.dump({
            'train_makespans': epoch_train_makespans,
            'test_makespans': epoch_test_makespans,
            'best_test_makespan': best_test_makespan
        }, f)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epoch_train_makespans, 'b-', alpha=0.7, label='Train Makespan')
    test_epochs = [x[0] for x in epoch_test_makespans]
    test_values = [x[1] for x in epoch_test_makespans]
    ax.plot(test_epochs, test_values, 'ro-', markersize=6, label='Test Makespan')
    ax.axhline(y=best_test_makespan, color='g', linestyle='--', label=f'Best Test: {best_test_makespan:.1f}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Makespan')
    ax.set_title('Phase 2 Fixed Seeds: PPO Convergence')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/phase2_fixed_seeds/convergence.png", dpi=150)
    plt.close()
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Test Makespan: {best_test_makespan:.2f}")
    print(f"Final Train Makespan: {epoch_train_makespans[-1]:.2f}")
    print(f"\nOutputs: results/phase2_fixed_seeds/")


if __name__ == "__main__":
    train_fixed_seeds(num_epochs=200, num_train_seeds=100, num_test_seeds=50)
