"""
Train PPO with LGP Coevolution for Dynamic Job Shop Scheduling.
Simplified entry point for PPO_LGP_Clean project.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.optim as optim

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import configuration
from config import (
    RANDOM_SEED, 
    PPOConfig, 
    LGPConfig, 
    CoevolutionConfig,
    EnvironmentConfig
)

# Import core LGP components
from core.lgp_generator import LGPGenerator
from core.lgp_program import LGPProgram

# Import registries (auto-register functions)
import registries.dispatching_rules
import registries.metaheuristics_impl

# Import training components
from training.ppo_model import PPOActorCritic, select_action, compute_returns
from training.lgp_coevolution_trainer import train_with_coevolution_lgp, CoevolutionConfig as CoevoCfg
from environment.scheduling_env import DynamicSchedulingEnv


def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_lgp_programs(pool_size, max_length, min_length, num_registers, seed):
    """Initialize a pool of random LGP programs."""
    lgp_rng = random.Random(seed)
    lgp_gen = LGPGenerator(
        max_length=max_length,
        min_length=min_length,
        num_registers=num_registers,
        rng=lgp_rng,
    )
    programs = [
        lgp_gen.generate_random_program()
        for _ in range(pool_size)
    ]
    return programs


def main():
    """Main training entry point."""
    print("=" * 70)
    print("PPO + LGP COEVOLUTION TRAINING")
    print("=" * 70)
    
    # Set random seeds
    set_random_seeds(RANDOM_SEED)
    print(f"✓ Random seed set to: {RANDOM_SEED}")
    
    # Initialize LGP programs
    print(f"\n📋 Initializing {LGPConfig.pool_size} LGP programs...")
    lgp_programs = initialize_lgp_programs(
        pool_size=LGPConfig.pool_size,
        max_length=LGPConfig.max_program_length,
        min_length=LGPConfig.min_program_length,
        num_registers=LGPConfig.num_registers,
        seed=RANDOM_SEED
    )
    print(f"✓ Initialized {len(lgp_programs)} programs")
    
    # Initialize environment
    print(f"\n🏭 Creating scheduling environment...")
    env = DynamicSchedulingEnv(
        lambda_tardiness=EnvironmentConfig.lambda_tardiness,
        action_library=None,  # Will be set by LGP
        action_budget_s=LGPConfig.action_budget_s
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"✓ Environment created: obs_dim={obs_dim}, act_dim={act_dim}")
    
    # Initialize PPO model
    print(f"\n🧠 Creating PPO model...")
    model = PPOActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=PPOConfig.learning_rate)
    print(f"✓ PPO model created with lr={PPOConfig.learning_rate}")
    
    # Configure coevolution
    print(f"\n⚙️  Configuring coevolution...")
    cfg = CoevoCfg(
        num_generations=CoevolutionConfig.num_generations,
        episodes_per_gen=CoevolutionConfig.episodes_per_gen,
        max_steps_per_episode=CoevolutionConfig.max_steps_per_episode,
        gamma=PPOConfig.gamma,
        ppo_epochs=PPOConfig.ppo_epochs,
        clip_epsilon=PPOConfig.clip_epsilon,
        entropy_coef=PPOConfig.entropy_coef,
        elite_size=CoevolutionConfig.elite_size,
        n_replace=CoevolutionConfig.n_replace,
        warmup_episodes=CoevolutionConfig.warmup_episodes,
        mutation_sigma=CoevolutionConfig.mutation_sigma,
        dr_mutation_prob=CoevolutionConfig.dr_mutation_prob,
        mh_name_mutation_prob=CoevolutionConfig.mh_name_mutation_prob
    )
    print(f"✓ Configuration: {cfg.num_generations} generations, {cfg.episodes_per_gen} episodes/gen")
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "programs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "portfolios"), exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}/")
    
    # Start training
    print(f"\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    
    lgp_programs, final_action_library = train_with_coevolution_lgp(
        env=env,
        lgp_programs=lgp_programs,
        model=model,
        optimizer=optimizer,
        select_action_fn=select_action,
        compute_returns_fn=compute_returns,
        cfg=cfg,
        output_dir=output_dir
    )
    
    # Save results
    model_path = os.path.join(output_dir, "models", "trained_policy.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save program info
    programs_data = {
        "num_programs": len(lgp_programs),
        "programs": [prog.to_dict() for prog in lgp_programs]
    }
    lgp_path = os.path.join(output_dir, "programs", "lgp_programs_final.json")
    with open(lgp_path, 'w') as f:
        json.dump(programs_data, f, indent=2)
    print(f"✓ LGP programs saved to: {lgp_path}")
    
    print(f"\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved in: {output_dir}/")
    print(f"  - Model: {model_path}")
    print(f"  - Programs: {lgp_path}")
    print(f"  - Metrics: {output_dir}/metrics/")
    print(f"  - Portfolios: {output_dir}/portfolios/")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
