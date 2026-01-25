"""
Quick validation for PPO model + environment.
Checks shapes, determinism (seed), numeric stability, and a PPO update pass.
"""

import json
import math
import os
import random
import sys
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import RANDOM_SEED, PPOConfig, EnvironmentConfig, LGPConfig
from environment.dataset_loader import load_dataset
from environment.scheduling_env import DynamicSchedulingEnv
from training.ppo_model import PPOActorCritic, select_action, compute_returns, ppo_update
from training.portfolio_types import ActionIndividual, Gene
import registries.dispatching_rules  # noqa: F401
import registries.metaheuristics_impl  # noqa: F401
from registries.dispatching_registry import has_dr
from registries.mh_registry import has_mh


def build_action_library():
    genes = [Gene(kind="DR", name="EDD", w_raw=1.0)]
    # 3 MH genes to match pipeline
    genes.append(Gene(kind="MH", name="SA", w_raw=1.0))
    genes.append(Gene(kind="MH", name="SA", w_raw=0.0))
    genes.append(Gene(kind="MH", name="SA", w_raw=0.0))
    return [ActionIndividual(genes=genes) for _ in range(4)]


def load_phase2_action_library(filepath="results/top_portfolios_phase2.json"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"phase2 portfolios missing: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    portfolios = data.get("portfolios")
    if not isinstance(portfolios, list) or len(portfolios) == 0:
        raise ValueError("phase2 portfolios list is missing or empty")

    action_library = []
    errors = []

    for i, p in enumerate(portfolios):
        if not isinstance(p, dict):
            errors.append(f"[{i}] portfolio is not a dict")
            continue

        dr = p.get("dr")
        mh_genes = p.get("mh_genes")

        if not isinstance(dr, str) or not dr.strip():
            errors.append(f"[{i}] invalid dr: {dr}")

        if not isinstance(mh_genes, list) or len(mh_genes) == 0:
            errors.append(f"[{i}] mh_genes missing or empty")
            mh_genes = []

        if len(mh_genes) != LGPConfig.n_mh_genes:
            errors.append(
                f"[{i}] mh_genes len {len(mh_genes)} != {LGPConfig.n_mh_genes}"
            )

        genes = []
        if isinstance(dr, str) and dr.strip():
            genes.append(Gene(kind="DR", name=dr, w_raw=1.0))

        for j, mh in enumerate(mh_genes):
            if not isinstance(mh, dict):
                errors.append(f"[{i}] mh_genes[{j}] not a dict")
                continue
            name = mh.get("name")
            weight = mh.get("weight")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"[{i}] mh_genes[{j}] invalid name: {name}")
                continue
            try:
                weight_f = float(weight)
                if not math.isfinite(weight_f):
                    raise ValueError("non-finite weight")
            except Exception:
                errors.append(f"[{i}] mh_genes[{j}] invalid weight: {weight}")
                continue
            genes.append(Gene(kind="MH", name=name, w_raw=weight_f))

        try:
            action_library.append(ActionIndividual(genes=genes))
        except Exception as e:
            errors.append(f"[{i}] ActionIndividual error: {e}")

    if len(action_library) != len(portfolios):
        errors.append(
            f"action_library size {len(action_library)} != portfolios {len(portfolios)}"
        )

    # Registry validation
    for i, ind in enumerate(action_library):
        dr_name = ind.dr_gene.name
        if not has_dr(dr_name):
            errors.append(f"[{i}] DR not registered: {dr_name}")
        for j, g in enumerate(ind.mh_genes):
            if not has_mh(g.name):
                errors.append(f"[{i}] MH not registered: {g.name} (idx {j})")

    if errors:
        msg = "Phase2 portfolio validation failed:\n" + "\n".join(errors[:20])
        if len(errors) > 20:
            msg += f"\n... {len(errors) - 20} more"
        raise ValueError(msg)

    return action_library


def check_json_serializable(obj, label="obj"):
    try:
        json.dumps(obj)
        return True
    except TypeError as e:
        print(f"[json] {label} not serializable: {e}")
        return False


def run_episode(env, model=None, seed=0, policy="random", fixed_action=0):
    env.seed(seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    state = env.reset()
    done = False
    rewards = []
    actions = []
    while not done:
        if model is None:
            if policy == "fixed":
                action = int(fixed_action)
            else:
                action = int(np.random.randint(0, env.action_space.n))
        else:
            action, _, _ = select_action(model, state)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
        state = next_state
    metrics = env.get_metrics()
    return sum(rewards), actions, metrics


def main():
    print("=== PPO + ENV VALIDATION (DEEP) ===")

    # 1) Dataset sanity
    jobs, due_dates, machine_pool, *_ = load_dataset(EnvironmentConfig.dataset_name, return_meta=True)
    print(f"[dataset] name={EnvironmentConfig.dataset_name} jobs={len(jobs)} machines={len(machine_pool)}")

    # 2) Phase 2 portfolios
    action_library = load_phase2_action_library()
    print(f"[phase2] portfolios={len(action_library)}")
    if len(action_library) != LGPConfig.pool_size:
        print(f"[phase2] WARNING: portfolios != pool_size ({LGPConfig.pool_size})")

    # 3) Env shapes
    env = DynamicSchedulingEnv(
        lambda_tardiness=1.0,
        action_library=action_library,
        dataset_name=EnvironmentConfig.dataset_name,
    )
    obs = env.reset()
    print(f"[env] obs_shape={obs.shape} act_dim={env.action_space.n}")
    assert obs.shape[0] == env.observation_space.shape[0], "obs_dim mismatch"
    assert env.action_space.n == len(action_library), "action space mismatch"

    # 4) Determinism (same seed -> same metrics)
    r1, a1, m1 = run_episode(env, model=None, seed=RANDOM_SEED + 1, policy="fixed", fixed_action=0)
    r2, a2, m2 = run_episode(env, model=None, seed=RANDOM_SEED + 1, policy="fixed", fixed_action=0)
    deterministic = (r1 == r2) and (a1 == a2) and (m1 == m2)
    print(f"[seed] deterministic={deterministic}")

    # 5) PPO forward + action sampling
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = PPOActorCritic(obs_dim, act_dim)
    state = env.reset()
    with torch.no_grad():
        action, logp, value = select_action(model, state)
    print(f"[ppo] action={action} logp={float(logp):.4f} value={float(value):.4f}")
    assert 0 <= action < act_dim, "action out of range"
    assert math.isfinite(float(logp)) and math.isfinite(float(value)), "non-finite logp/value"
    with torch.no_grad():
        logits, _ = model(torch.FloatTensor(state).unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
    prob_sum = float(probs.sum().item())
    print(f"[ppo] prob_sum={prob_sum:.6f} min_prob={float(probs.min()):.6f}")
    assert abs(prob_sum - 1.0) < 1e-4, "policy probs do not sum to 1"
    assert float(probs.min()) >= 0.0, "negative policy probability"

    # 6) PPO update sanity (single episode)
    env.seed(RANDOM_SEED + 7)
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
    returns = compute_returns(rewards, masks, gamma=PPOConfig.gamma)
    values_rollout = [v.item() for v in values]
    returns_arr = np.array(returns, dtype=np.float32)
    returns_mean = float(returns_arr.mean()) if len(returns_arr) else 0.0
    returns_std = float(returns_arr.std() + 1e-8) if len(returns_arr) else 1.0
    advantages = [r - v for r, v in zip(returns, values_rollout)]
    if returns_std < 1e-6:
        print("[ppo] WARNING: returns_std is near zero; advantages may be unstable")
    optimizer = torch.optim.Adam(model.parameters(), lr=PPOConfig.learning_rate)
    ppo_update(
        model,
        optimizer,
        states,
        actions,
        log_probs,
        returns,
        advantages,
        clip_epsilon=PPOConfig.clip_epsilon,
        ppo_epochs=2,
        normalize_returns=True,
        returns_mean=returns_mean,
        returns_std=returns_std,
    )
    # Check for NaNs after update
    with torch.no_grad():
        for p in model.parameters():
            assert torch.isfinite(p).all(), "non-finite param after update"
    print("[ppo] update ok, params finite")

    # 7) Basic runtime sanity (random vs model)
    r_rand, _, m_rand = run_episode(env, model=None, seed=RANDOM_SEED + 9)
    r_model, _, m_model = run_episode(env, model=model, seed=RANDOM_SEED + 9)
    print(f"[runtime] return random={r_rand:.2f} model={r_model:.2f}")
    print(f"[runtime] makespan random={m_rand['makespan']:.2f} model={m_model['makespan']:.2f}")

    # 8) Metrics + JSON sanity
    metrics = env.get_metrics()
    assert isinstance(metrics, dict), "metrics is not a dict"
    assert math.isfinite(float(metrics.get("makespan", 0.0))), "non-finite makespan"
    if env.current_schedule_events:
        check_json_serializable(env.current_schedule_events, label="schedule_events")

    print("=== VALIDATION DONE ===")


if __name__ == "__main__":
    main()
