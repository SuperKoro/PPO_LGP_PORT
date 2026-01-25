"""
PPO (Proximal Policy Optimization) Model for PPO+LGP.
Neural network architecture and action selection for PPO agent.
"""

import numpy as np
import torch
import torch.nn as nn


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    - Actor: Policy network (outputs action probabilities)
    - Critic: Value network (estimates state values)
    """
    
    def __init__(self, obs_dim, act_dim):
        super(PPOActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


def select_action(model, state, deterministic=False):
    """
    Select action using policy network.
    
    Args:
        model: PPOActorCritic network
        state: Current state observation
        deterministic: If True, select action with highest probability (for evaluation)
                      If False, sample from distribution (for training)
        
    Returns:
        action: Selected action index
        log_prob: Log probability of selected action
        value: State value estimate
    """
    device = next(model.parameters()).device
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, value = model(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            # ⭐ FIX: For evaluation, use argmax (no randomness)
            action = torch.argmax(logits, dim=-1)
        else:
            # For training, sample from distribution (allows exploration)
            action = dist.sample()

        log_prob = dist.log_prob(action)

    # Keep log_prob/value as 1D tensors to avoid (T,1) vs (T) broadcast issues.
    return action.item(), log_prob.squeeze(-1), value.squeeze(-1)


def compute_returns(rewards, masks, gamma=0.9):
    """
    Compute discounted returns.
    
    Args:
        rewards: List of rewards
        masks: List of masks (1 = continue, 0 = terminal)
        gamma: Discount factor
        
    Returns:
        List of discounted returns
    """
    returns = []
    R = 0
    for r, mask in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * mask
        returns.insert(0, R)
    return returns


def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages,
               clip_epsilon=0.2, ppo_epochs=4, entropy_coef=0.01, vf_coef=0.01,
               normalize_returns=True, returns_mean=None, returns_std=None):
    """
    Perform PPO update step.
    
    Args:
        model: PPOActorCritic network
        optimizer: PyTorch optimizer
        states: Batch of states
        actions: Batch of actions taken
        old_log_probs: Log probs of actions under old policy
        returns: Computed returns
        advantages: Advantage estimates
        clip_epsilon: PPO clipping parameter
        ppo_epochs: Number of PPO update epochs
        entropy_coef: Entropy bonus coefficient
        
    Returns:
        Tuple of (policy_loss, value_loss, total_loss)
    """
    device = next(model.parameters()).device

    # Convert list of numpy arrays to a single contiguous array for speed.
    if isinstance(states, torch.Tensor):
        states = states
    else:
        states = torch.from_numpy(np.asarray(states, dtype=np.float32))
    states = states.to(device)

    actions = torch.as_tensor(actions, dtype=torch.long, device=device)

    if isinstance(old_log_probs, torch.Tensor):
        old_log_probs = old_log_probs.detach()
    else:
        old_log_probs = torch.stack(old_log_probs).detach()
    old_log_probs = old_log_probs.squeeze(-1).to(device)

    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
    if normalize_returns:
        if returns_mean is None:
            returns_mean = returns.mean().item()
        if returns_std is None:
            returns_std = returns.std().item() + 1e-8
        returns = (returns - returns_mean) / returns_std

    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss_val = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clip_frac = 0.0
    
    for _ in range(ppo_epochs):
        logits, values = model(states)
        values = values.squeeze()
        # NOTE: Do NOT normalize values here - critic learns to predict
        # normalized returns directly. Previous normalization was incorrect.
        
        # Policy loss
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (on normalized returns if enabled)
        value_loss = 0.5 * (returns - values).pow(2).mean()
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        approx_kl = (old_log_probs - new_log_probs).mean()
        clip_frac = ((ratio < (1 - clip_epsilon)) | (ratio > (1 + clip_epsilon))).float().mean()
        
        # Total loss (vf_coef scales value loss to be comparable with policy loss)
        loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss_val += loss.item()
        total_entropy += entropy.item()
        total_approx_kl += approx_kl.item()
        total_clip_frac += clip_frac.item()
    
    # Return average losses
    avg_policy_loss = total_policy_loss / ppo_epochs
    avg_value_loss = total_value_loss / ppo_epochs
    avg_total_loss = total_loss_val / ppo_epochs
    avg_entropy = total_entropy / ppo_epochs
    avg_approx_kl = total_approx_kl / ppo_epochs
    avg_clip_frac = total_clip_frac / ppo_epochs

    metrics = {
        "entropy": avg_entropy,
        "approx_kl": avg_approx_kl,
        "clip_frac": avg_clip_frac,
    }

    return avg_policy_loss, avg_value_loss, avg_total_loss, metrics
