"""
PPO (Proximal Policy Optimization) Model for PPO+LGP.
Neural network architecture and action selection for PPO agent.
"""

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
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    logits, value = model(state_tensor)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    
    if deterministic:
        # ⭐ FIX: For evaluation, use argmax (no randomness)
        action = probs.argmax(dim=-1)
    else:
        # For training, sample from distribution (allows exploration)
        action = dist.sample()
    
    return action.item(), dist.log_prob(action), value


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
               clip_epsilon=0.2, ppo_epochs=4, entropy_coef=0.01):
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
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    old_log_probs = torch.stack(old_log_probs).detach()
    returns = torch.FloatTensor(returns)
    advantages = torch.FloatTensor(advantages)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_policy_loss = 0
    total_value_loss = 0
    total_loss_val = 0
    
    for _ in range(ppo_epochs):
        logits, values = model(states)
        values = values.squeeze()
        
        # Policy loss
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = 0.5 * (returns - values).pow(2).mean()
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + value_loss - entropy_coef * entropy
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss_val += loss.item()
    
    # Return average losses
    return (
        total_policy_loss / ppo_epochs,
        total_value_loss / ppo_epochs,
        total_loss_val / ppo_epochs
    )
