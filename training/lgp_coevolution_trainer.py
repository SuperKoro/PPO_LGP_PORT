# LGP Coevolution Training Module

from typing import List, Dict, Any
import os
import json
import random
import copy
import numpy as np
import torch

from config import LGPConfig
from core.lgp_program import LGPProgram
from core.lgp_generator import LGPGenerator
from core.lgp_evolution import linear_crossover, mutate_program
from training.portfolio_types import ActionIndividual, describe_individual, Gene


# ============================================================================
# FIX 1: Hall of Fame - Protect best programs across ALL generations
# ============================================================================
class HallOfFame:
    """
    Keeps track of best-ever programs across all generations.
    These programs are NEVER replaced during LGP evolution.
    """
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.entries = []  # List of (fitness, program_copy, generation, original_idx)
    
    def try_add(self, program: 'LGPProgram', fitness: float, generation: int, original_idx: int) -> bool:
        """
        Try to add a program to Hall of Fame.
        Returns True if added, False otherwise.
        """
        # Don't add programs with invalid fitness
        if fitness <= -1e8:
            return False
        
        # Check if program is already in HoF (by fitness similarity)
        for f, p, g, idx in self.entries:
            if abs(f - fitness) < 0.01:  # Same fitness = probably same program
                return False
        
        if len(self.entries) < self.max_size:
            # Room available - just add
            self.entries.append((fitness, copy.deepcopy(program), generation, original_idx))
            self.entries.sort(key=lambda x: x[0], reverse=True)  # Sort by fitness (best first)
            return True
        
        # Check if better than worst in HoF
        if fitness > self.entries[-1][0]:
            self.entries[-1] = (fitness, copy.deepcopy(program), generation, original_idx)
            self.entries.sort(key=lambda x: x[0], reverse=True)
            return True
        
        return False
    
    def get_best_fitness(self) -> float:
        """Get the best fitness ever recorded."""
        if not self.entries:
            return -1e9
        return self.entries[0][0]
    
    def get_best_program(self) -> 'LGPProgram':
        """Get copy of the best program ever."""
        if not self.entries:
            return None
        return copy.deepcopy(self.entries[0][1])
    
    def find_matching_indices(self, lgp_programs: List['LGPProgram']) -> List[int]:
        """
        Find indices in lgp_programs that match HoF programs.
        Used to protect these indices from replacement.
        """
        protected = []
        for hof_fitness, hof_program, hof_gen, hof_idx in self.entries:
            # Try to find matching program by checking if same instructions
            for i, prog in enumerate(lgp_programs):
                if self._programs_similar(hof_program, prog):
                    protected.append(i)
                    break
        return protected
    
    def _programs_similar(self, p1: 'LGPProgram', p2: 'LGPProgram') -> bool:
        """Check if two programs have same instructions."""
        if len(p1.instructions) != len(p2.instructions):
            return False
        for i1, i2 in zip(p1.instructions, p2.instructions):
            if str(i1) != str(i2):
                return False
        return True
    
    def restore_best_to_pool(self, lgp_programs: List['LGPProgram'], target_idx: int) -> bool:
        """
        Restore the best HoF program to the pool at target_idx.
        Useful if best program was lost due to mutation.
        """
        if not self.entries:
            return False
        
        best_program = copy.deepcopy(self.entries[0][1])
        lgp_programs[target_idx] = best_program
        return True
    
    def print_status(self):
        """Print current Hall of Fame status."""
        print(f"  📜 Hall of Fame ({len(self.entries)}/{self.max_size}):")
        for i, (fitness, prog, gen, idx) in enumerate(self.entries):
            print(f"      #{i+1}: fitness={fitness:.2f} (Gen {gen}, idx={idx})")


# CoevolutionConfig definition
from dataclasses import dataclass

@dataclass
class CoevolutionConfig:
    """Configuration for coevolution training."""
    num_generations: int = 20
    episodes_per_gen: int = 10
    max_steps_per_episode: int = 200
    gamma: float = 0.9
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    elite_size: int = 16
    n_replace: int = 4
    warmup_episodes: int = 2
    mutation_sigma: float = 0.3
    dr_mutation_prob: float = 0.1
    mh_name_mutation_prob: float = 0.2


# Helper functions for metrics collection
def collect_episode_metrics(env):
    """Collect metrics from environment after episode."""
    return env.get_metrics()


def save_generation_metrics(generation: int, episodes_metrics: List[Dict[str, Any]], output_dir: str):
    """Save aggregated metrics for a generation."""
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Aggregate metrics
    avg_metrics = {}
    if episodes_metrics:
        keys = episodes_metrics[0].keys()
        for key in keys:
            values = [m[key] for m in episodes_metrics if key in m]
            if values:
                avg_metrics[f"avg_{key}"] = float(np.mean(values))
                avg_metrics[f"std_{key}"] = float(np.std(values))
    
    # Save to file
    filepath = os.path.join(metrics_dir, f"generation_{generation}.json")
    with open(filepath, 'w') as f:
        json.dump({
            "generation": generation,
            "num_episodes": len(episodes_metrics),
            "aggregated_metrics": avg_metrics,
            "per_episode": episodes_metrics
        }, f, indent=2)


def portfolio_to_dict(individual: ActionIndividual, index: int = None, fitness: float = None, usage: int = None):
    """Convert ActionIndividual to dictionary format."""
    result = {
        "index": int(index) if index is not None else None,
        "fitness": float(fitness) if fitness is not None else None,
        "usage": int(usage) if usage is not None else 0,
        "dr": individual.dr_gene.name,
        "mh_genes": [{"name": g.name, "weight": float(g.w_raw)} for g in individual.mh_genes]
    }
    return result


def save_portfolios_json(portfolios_data, filepath):
    """Save portfolios data to JSON file with numpy type handling."""
    # Custom encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open(filepath, 'w') as f:
        json.dump(portfolios_data, f, indent=2, cls=NumpyEncoder)


def save_final_results(env, output_dir: str, generation: int):
    """Save final schedule and metrics."""
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    metrics = env.get_metrics()
    filepath = os.path.join(final_dir, "final_metrics.json")
    with open(filepath, 'w') as f:
        json.dump({
            "generation": generation,
            "metrics": metrics
        }, f, indent=2)


def _select_elite_indices(fitness: np.ndarray, elite_size: int):
    """Select top elite indices based on fitness."""
    return np.argsort(fitness)[::-1][:elite_size]


def build_lgp_inputs_for_env(env) -> Dict[str, float]:
    """
    Xây macro-state cho LGP.
    Có thể chỉnh lại tuỳ dạng dữ liệu jobs của bạn.
    """
    num_jobs = 0
    total_pt = 0.0
    total_ops = 0

    jobs = getattr(env, "jobs_initial", None)
    if isinstance(jobs, dict):
        num_jobs = len(jobs)
        for job_id, job_info in jobs.items():
            ops = job_info.get("operations") if isinstance(job_info, dict) else job_info
            if ops is None:
                continue
            for op in ops:
                pt = op.get("processing_time", 0.0) if isinstance(op, dict) else 0.0
                total_pt += float(pt)
                total_ops += 1
    elif isinstance(jobs, list):
        num_jobs = len(jobs)
        for job in jobs:
            ops = job.get("operations") if isinstance(job, dict) else None
            if ops is None:
                continue
            for op in ops:
                pt = op.get("processing_time", 0.0) if isinstance(op, dict) else 0.0
                total_pt += float(pt)
                total_ops += 1

    avg_pt = total_pt / total_ops if total_ops > 0 else 0.0
    avg_ops_per_job = total_ops / num_jobs if num_jobs > 0 else 0.0

    return {
        "num_jobs": float(num_jobs),
        "avg_processing_time": float(avg_pt),
        "avg_ops_per_job": float(avg_ops_per_job),
    }


def make_fallback_individual() -> ActionIndividual:
    """
    Portfolio fallback nếu LGPProgram bị lỗi runtime.
    First available DR (typically EDD) + first available MH (typically SA) with weight 1.0.
    """
    genes = [
        Gene(kind="DR", name=LGPConfig.available_dr[0], w_raw=1.0),
        Gene(kind="MH", name=LGPConfig.available_mh[0], w_raw=1.0),
    ]
    while len(genes) < 1 + LGPConfig.n_mh_genes:
        genes.append(Gene(kind="MH", name=LGPConfig.available_mh[0], w_raw=0.0))
    return ActionIndividual(genes=genes)


def train_with_coevolution_lgp(
    env,
    lgp_programs: List[LGPProgram],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    select_action_fn,
    compute_returns_fn,
    cfg: CoevolutionConfig,
    output_dir: str = "results_lgp",
):
    """
    Coevolution training: evolve LGP programs, PPO vẫn chọn index action như cũ.
    """
    os.makedirs(output_dir, exist_ok=True)

    K = len(lgp_programs)
    assert K == LGPConfig.pool_size, "Pool size must match number of LGP programs"

    rng_np = np.random.default_rng(seed=0)
    rng_py = random.Random(0)

    lgp_gen = LGPGenerator(
        max_length=LGPConfig.max_program_length,
        min_length=LGPConfig.min_program_length,
        num_registers=LGPConfig.num_registers,
        rng=rng_py,
    )

    # ===================================================================
    # FIX 1: Initialize Hall of Fame
    # ===================================================================
    hall_of_fame = HallOfFame(max_size=5)
    print("  📜 Hall of Fame initialized (max_size=5)")
    
    # ===================================================================
    # FIX 3: Learning Rate Decay with MINIMUM FLOOR
    # ===================================================================
    initial_lr = optimizer.param_groups[0]['lr']
    min_lr = 5e-5  # NEVER go below this!
    decay_factor = 0.95
    
    for gen in range(cfg.num_generations):
        print(f"\n========== LGP Generation {gen+1}/{cfg.num_generations} ==========")
        
        # FIX 3: Apply learning rate decay WITH MINIMUM FLOOR
        raw_lr = initial_lr * (decay_factor ** gen)
        current_lr = max(min_lr, raw_lr)  # Never go below min_lr!
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        lr_status = "✓" if current_lr > min_lr else "⚠️ AT FLOOR"
        print(f"  Learning rate: {current_lr:.6f} (decay: {decay_factor**gen:.4f}) {lr_status}")

        # 1) Sinh portfolios từ programs
        lgp_inputs = build_lgp_inputs_for_env(env)

        action_library: List[ActionIndividual] = []
        for idx, prog in enumerate(lgp_programs):
            try:
                ind = prog.execute(lgp_inputs)
            except Exception as e:
                print(f"[WARN] Program {idx} crashed during execute(): {e}")
                ind = make_fallback_individual()
            action_library.append(ind)

        # gắn vào env
        env.action_library = action_library

        usage = np.zeros(K, dtype=np.int32)
        sum_reward = np.zeros(K, dtype=np.float32)
        episodes_metrics = []
        
        # ===================================================================
        # FIX 2: Forced Exploration WITHOUT PPO update in Early Generations
        # ===================================================================
        # Force each program to be sampled at least once, but DON'T train on forced actions
        forced_exploration = (gen < 2)
        if forced_exploration:
            programs_to_explore = list(range(K))
            random.shuffle(programs_to_explore)
            forced_idx = 0
            print(f"  🔍 FORCED EXPLORATION MODE: Sampling all programs (NO PPO update on forced)")

        # 2) PPO training trong generation này
        for ep in range(cfg.episodes_per_gen):
            state = env.reset()

            # Separate lists for forced vs policy actions
            states_list = []
            actions_list = []
            log_probs_list = []
            values_list = []
            rewards = []
            masks = []
            is_forced_action = []  # FIX 2: Track which actions were forced

            ep_return = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            count_updates = 0

            for step in range(cfg.max_steps_per_episode):
                # FIX 2: Forced exploration - sample but mark as forced
                if forced_exploration and forced_idx < len(programs_to_explore):
                    action = programs_to_explore[forced_idx]
                    forced_idx += 1
                    # Get policy's action for comparison (but don't use it)
                    policy_action, log_prob, value = select_action_fn(model, state)
                    was_forced = True
                else:
                    action, log_prob, value = select_action_fn(model, state)
                    was_forced = False
                
                next_state, reward, done, info = env.step(action)

                states_list.append(state)
                actions_list.append(action)
                log_probs_list.append(log_prob)
                values_list.append(value.squeeze(0))
                rewards.append(reward)
                masks.append(0.0 if done else 1.0)
                is_forced_action.append(was_forced)  # FIX 2: Track

                usage[action] += 1
                sum_reward[action] += reward
                ep_return += reward

                state = next_state
                if done:
                    break

            # FIX 2: SKIP PPO update if ALL actions were forced
            num_forced = sum(is_forced_action)
            num_policy = len(is_forced_action) - num_forced
            
            if num_policy == 0:
                # All forced - skip PPO update entirely
                avg_pl = 0.0
                avg_vl = 0.0
            else:
                # Filter to only policy actions for PPO update
                policy_indices = [i for i, forced in enumerate(is_forced_action) if not forced]
                
                if len(policy_indices) > 0:
                    # Extract only non-forced data
                    states_policy = [states_list[i] for i in policy_indices]
                    actions_policy = [actions_list[i] for i in policy_indices]
                    log_probs_policy = [log_probs_list[i] for i in policy_indices]
                    values_policy = [values_list[i] for i in policy_indices]
                    rewards_policy = [rewards[i] for i in policy_indices]
                    masks_policy = [masks[i] for i in policy_indices]
                    
                    # PPO update on policy actions only
                    returns = compute_returns_fn(rewards_policy, masks_policy, gamma=cfg.gamma)
                    returns_t = torch.tensor(returns, dtype=torch.float32)
                    values_t = torch.stack(values_policy)
                    log_probs_old = torch.stack(log_probs_policy).detach()
                    states_np = np.array(states_policy, dtype=np.float32)
                    states_t = torch.from_numpy(states_np)
                    actions_t = torch.tensor(actions_policy, dtype=torch.int64)

                    advantages = returns_t - values_t.detach()
                    if len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    value_coef = 0.5
                    for _ in range(cfg.ppo_epochs):
                        logits, values = model(states_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        log_probs = dist.log_prob(actions_t)
                        ratio = torch.exp(log_probs - log_probs_old)

                        surr1 = ratio * advantages
                        surr2 = torch.clamp(
                            ratio,
                            1.0 - cfg.clip_epsilon,
                            1.0 + cfg.clip_epsilon,
                        ) * advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = (returns_t - values.squeeze(-1)).pow(2).mean()
                        entropy = dist.entropy().mean()

                        loss = policy_loss + value_coef * value_loss - cfg.entropy_coef * entropy

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_policy_loss += policy_loss.item()
                        total_value_loss += value_loss.item()
                        count_updates += 1
                    
                    avg_pl = total_policy_loss / max(count_updates, 1)
                    avg_vl = total_value_loss / max(count_updates, 1)
                else:
                    avg_pl = 0.0
                    avg_vl = 0.0
            
            # Collect episode metrics
            ep_metrics = collect_episode_metrics(env)
            ep_metrics["policy_loss"] = avg_pl
            ep_metrics["value_loss"] = avg_vl
            ep_metrics["return"] = ep_return
            ep_metrics["num_forced_actions"] = num_forced  # FIX 2: Track for analysis
            ep_metrics["num_policy_actions"] = num_policy
            episodes_metrics.append(ep_metrics)
            
            if (ep + 1) % 50 == 0:
                forced_info = f" (forced={num_forced})" if num_forced > 0 else ""
                print(f"[Gen {gen+1} Ep {ep+1}] Return={ep_return:.2f} | PolicyLoss={avg_pl:.4f} | ValueLoss={avg_vl:.4f}{forced_info}")

        # 3) Fitness cho mỗi program
        avg_reward = np.full(K, -1e9, dtype=np.float32)
        for i in range(K):
            if usage[i] > 0:
                avg_reward[i] = sum_reward[i] / max(1, usage[i])

        # 4) Evolve LGP programs
        elite_indices = _select_elite_indices(avg_reward, cfg.elite_size)

        best_idx = int(elite_indices[0])
        best_fitness = avg_reward[best_idx]
        print(f"\n[Gen {gen+1}] Best program idx={best_idx}, avg_reward={best_fitness:.3f}")
        print("  Example portfolio from best program:")
        print("   ", describe_individual(action_library[best_idx]))
        
        # ===================================================================
        # FIX 1: Update Hall of Fame with best program from this generation
        # ===================================================================
        if hall_of_fame.try_add(lgp_programs[best_idx], best_fitness, gen + 1, best_idx):
            print(f"  🏆 NEW ENTRY in Hall of Fame! fitness={best_fitness:.2f}")
        
        # Check if current best is worse than HoF best (program was lost!)
        hof_best_fitness = hall_of_fame.get_best_fitness()
        if best_fitness < hof_best_fitness - 10:  # Lost more than 10 fitness points
            print(f"  ⚠️ WARNING: Current best ({best_fitness:.2f}) is worse than HoF best ({hof_best_fitness:.2f})")
            print(f"  🔄 Restoring HoF best program to pool...")
            # Find a low-performing slot to restore the best program
            worst_idx = int(np.argmin(avg_reward))
            if hall_of_fame.restore_best_to_pool(lgp_programs, worst_idx):
                print(f"  ✓ Restored HoF best to index {worst_idx}")
        
        hall_of_fame.print_status()

        # Save generation metrics
        save_generation_metrics(gen + 1, episodes_metrics, output_dir)
        
        # Save portfolios
        portfolios_data = {
            "generation": gen + 1,
            "elite": [portfolio_to_dict(action_library[i], i, avg_reward[i], usage[i]) for i in elite_indices],
            "all_fitness": avg_reward.tolist(),
            "all_usage": usage.tolist()
        }
        portfolios_dir = os.path.join(output_dir, "portfolios")
        os.makedirs(portfolios_dir, exist_ok=True)
        save_portfolios_json(portfolios_data, os.path.join(portfolios_dir, f"generation_{gen+1}_final.json"))

        # ===================================================================
        # FIX 1: Get Hall of Fame protected indices
        # ===================================================================
        hof_protected_indices = set(hall_of_fame.find_matching_indices(lgp_programs))
        if hof_protected_indices:
            print(f"  🛡️ Hall of Fame protected indices: {sorted(hof_protected_indices)}")
        
        # ===================================================================
        # DIVERSITY MECHANISM 1: Protect unused programs in early generations
        # ===================================================================
        def _should_protect_program(prog_idx, gen_num):
            """Protect programs with low usage to prevent PPO bias from eliminating unexplored programs"""
            # FIX 1: Always protect HoF programs
            if prog_idx in hof_protected_indices:
                return True
            
            prog_usage = usage[prog_idx]
            
            if gen_num <= 3:
                # First 3 generations: protect completely unused
                return prog_usage == 0
            elif gen_num <= 6:
                # Generations 4-6: protect rarely used (< 3 times)
                return prog_usage < 3
            else:
                # After gen 6: normal selection pressure
                return False
        
        # Select candidates for replacement (non-elite programs AND not in HoF)
        candidate_indices = [i for i in range(K) if i not in elite_indices and i not in hof_protected_indices]
        
        # Separate into protected and unprotected
        protected_programs = [i for i in candidate_indices if _should_protect_program(i, gen + 1)]
        unprotected_programs = [i for i in candidate_indices if not _should_protect_program(i, gen + 1)]
        
        # Select losers only from unprotected pool
        if len(unprotected_programs) >= cfg.n_replace:
            # Enough unprotected programs - use them
            unprotected_programs.sort(key=lambda i: avg_reward[i])  # Sort by fitness (worst first)
            loser_indices = unprotected_programs[:cfg.n_replace]
        else:
            # Not enough unprotected - use all candidates (protection overridden)
            candidate_indices.sort(key=lambda i: avg_reward[i])
            loser_indices = candidate_indices[:cfg.n_replace]
            if protected_programs:
                print(f"  [WARNING] Gen {gen+1}: Had to replace {len([i for i in loser_indices if i in protected_programs])} protected programs")

        # ===================================================================
        # DIVERSITY MECHANISM 2: Rank-based parent selection
        # ===================================================================
        def _sample_parents_rank_based(num_pairs: int):
            """
            Sample parents using rank-based selection instead of fitness-proportional.
            This reduces selection pressure and gives lower-fitness programs more chance.
            """
            # Sort programs by fitness (descending order - best first)
            sorted_indices = np.argsort(avg_reward)[::-1]
            
            # Assign ranks: best program = rank K, worst = rank 1
            ranks = np.arange(len(sorted_indices), 0, -1)
            
            # Selection probability proportional to rank (not exponential like softmax)
            # This is MUCH less biased than fitness-proportional
            rank_probs = ranks / ranks.sum()
            
            pairs = []
            for _ in range(num_pairs):
                # Sample two parents based on rank probabilities
                try:
                    selected = rng_np.choice(sorted_indices, size=2, p=rank_probs, replace=False)
                    p1, p2 = int(selected[0]), int(selected[1])
                except ValueError:
                    # Fallback if something goes wrong
                    p1 = int(sorted_indices[0])
                    p2 = int(sorted_indices[min(1, len(sorted_indices)-1)])
                
                pairs.append((p1, p2))
            
            return pairs

        # Sample parent pairs using rank-based selection
        parent_pairs = _sample_parents_rank_based(cfg.n_replace)
        
        # ===================================================================
        # DIVERSITY MECHANISM 3: Limit best program copies
        # ===================================================================
        best_program_idx = int(elite_indices[0])
        max_copies_from_best = K // 4  # Maximum 25% of population from same parent
        
        # Count how many children would come from best program
        best_program_usage_in_pairs = sum(
            1 for p1, p2 in parent_pairs 
            if p1 == best_program_idx or p2 == best_program_idx
        )
        
        # If too many pairs use best program, redistribute some
        if best_program_usage_in_pairs > max_copies_from_best:
            print(f"  [DIVERSITY] Gen {gen+1}: Limiting best program #{best_program_idx} from {best_program_usage_in_pairs} to {max_copies_from_best} children")
            
            # Get other top-performing programs to use instead
            other_good_programs = [int(i) for i in elite_indices[1:6] if i != best_program_idx]  # Top 2-6
            
            if len(other_good_programs) >= 2:
                new_pairs = []
                best_count = 0
                
                for p1, p2 in parent_pairs:
                    uses_best = (p1 == best_program_idx or p2 == best_program_idx)
                    
                    if uses_best and best_count >= max_copies_from_best:
                        # Replace this pair with pair from other good programs
                        new_p1, new_p2 = rng_np.choice(other_good_programs, size=2, replace=False)
                        new_pairs.append((int(new_p1), int(new_p2)))
                    else:
                        new_pairs.append((p1, p2))
                        if uses_best:
                            best_count += 1
                
                parent_pairs = new_pairs

        new_programs = list(lgp_programs)
        for li, (p1, p2) in zip(loser_indices, parent_pairs):
            child = linear_crossover(lgp_programs[p1], lgp_programs[p2], rng_py)
            child = mutate_program(
                child,
                generator=lgp_gen,
                rng=rng_py,
                mutation_rate=LGPConfig.mutation_rate,
            )
            new_programs[li] = child

        lgp_programs = new_programs

    # Save final results
    save_final_results(env, output_dir, cfg.num_generations)
    
    return lgp_programs, action_library
