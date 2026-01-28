"""
PPO (Proximal Policy Optimization) Model for PPO+LGP.
Neural network architecture and action selection for PPO agent.
Optimized with GAE (Generalized Advantage Estimation).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    """
    def __init__(self, obs_dim, act_dim):
        super(PPOActorCritic, self).__init__()
        # Shared features
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),  # Tanh thường ổn định hơn ReLU cho RL
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # Actor head (Policy)
        self.policy_head = nn.Linear(64, act_dim)
        
        # Critic head (Value)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

    def get_value(self, x):
        """Helper to get only value"""
        x = self.fc(x)
        return self.value_head(x)


def select_action(model, state, deterministic=False):
    """
    Select action using policy network.
    """
    device = next(model.parameters()).device
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
    if state_tensor.ndim == 1:
        state_tensor = state_tensor.unsqueeze(0)

    with torch.no_grad():
        logits, value = model(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

    return action.item(), log_prob.squeeze(), value.squeeze()


def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).
    Tính toán Advantage và Return chuẩn xác hơn Monte Carlo truyền thống.
    
    Args:
        rewards: List rewards
        values: List values từ Critic (bao gồm cả value của state cuối cùng)
        masks: List masks (1 nếu continue, 0 nếu done)
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        returns: List of computed returns (targets for critic)
        advantages: List of advantages (for policy update)
    """
    gae = 0
    returns = []
    advantages = []
    
    # values thường có độ dài len(rewards) + 1 (để tính delta bước cuối)
    # Nếu values chỉ bằng len(rewards), ta giả định next_value = 0
    if len(values) == len(rewards):
        values = list(values) + [0.0]
        
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        
        # Advantage tại bước t
        advantages.insert(0, gae)
        
        # Return = Advantage + Value (Target cho Critic)
        returns.insert(0, gae + values[step])
        
    return returns, advantages


# Keep old compute_returns for backward compatibility
def compute_returns(rewards, masks, gamma=0.99):
    """
    Compute discounted returns (Monte Carlo style).
    Kept for backward compatibility.
    """
    returns = []
    R = 0
    for r, mask in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * mask
        returns.insert(0, R)
    return returns


def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages,
               clip_epsilon=0.2, ppo_epochs=4, entropy_coef=0.01, vf_coef=0.01,
               max_grad_norm=0.5):
    """
    Hàm update PPO chuẩn.
    NOTE: advantages và returns phải được tính TOÀN BỘ trước khi gọi hàm này.
    
    Returns:
        Tuple of (policy_loss, value_loss, entropy, metrics_dict)
    """
    device = next(model.parameters()).device

    # 1. Chuyển dữ liệu sang Tensor
    if isinstance(states, torch.Tensor):
        states = states.to(device)
    else:
        states = torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)
    
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    
    if isinstance(old_log_probs, torch.Tensor):
        old_log_probs = old_log_probs.detach().to(device)
    else:
        old_log_probs = torch.stack(old_log_probs).detach().squeeze(-1).to(device)
    
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)

    # 2. Normalize Advantage (BẮT BUỘC cho Policy Loss ổn định)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.detach()

    # 3. Training Loop
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clip_frac = 0.0
    
    for _ in range(ppo_epochs):
        # Forward pass lấy policy và value mới
        logits, values = model(states)
        values = values.squeeze()
        
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # --- POLICY LOSS ---
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # --- VALUE LOSS ---
        value_loss = F.mse_loss(values, returns)

        # --- TOTAL LOSS ---
        loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

        # Metrics
        approx_kl = (old_log_probs - new_log_probs).mean()
        clip_frac = ((ratio < (1 - clip_epsilon)) | (ratio > (1 + clip_epsilon))).float().mean()

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # Logging
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
        total_approx_kl += approx_kl.item()
        total_clip_frac += clip_frac.item()

    avg_policy_loss = total_policy_loss / ppo_epochs
    avg_value_loss = total_value_loss / ppo_epochs
    avg_entropy = total_entropy / ppo_epochs
    avg_total_loss = avg_policy_loss + vf_coef * avg_value_loss - entropy_coef * avg_entropy
    
    metrics = {
        "entropy": avg_entropy,
        "approx_kl": total_approx_kl / ppo_epochs,
        "clip_frac": total_clip_frac / ppo_epochs,
    }

    return avg_policy_loss, avg_value_loss, avg_total_loss, metrics
