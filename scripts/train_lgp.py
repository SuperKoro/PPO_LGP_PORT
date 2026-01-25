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

# Workaround: OpenCV (cv2) tries to execute any config.py found on sys.path.
# If run from project root, it may pick up this project's config.py via gym import.
current_dir = os.path.dirname(os.path.abspath(__file__))
orig_cwd = os.getcwd()
try:
    os.chdir(current_dir)
    import gym  # Preload to avoid cv2 reading project config.py
finally:
    os.chdir(orig_cwd)

# Add parent directory to path for imports
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
from training.lgp_coevolution_trainer import train_with_coevolution_lgp
from training.portfolio_types import ActionIndividual, Gene
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


def create_dummy_action_library(pool_size: int):
    """
    Create dummy action library with correct size for environment initialization.
    This ensures action_space.n matches pool_size from the start.
    
    The actual portfolios will be replaced by LGP-generated ones during training,
    but we need the correct size for PPO model initialization.
    """
    dummy_library = []
    for i in range(pool_size):
        # Create a simple portfolio: EDD + SA (3 MH genes)
        genes = [
            Gene(kind="DR", name=LGPConfig.available_dr[i % len(LGPConfig.available_dr)], w_raw=1.0),
        ]
        # Add MH genes
        for j in range(LGPConfig.n_mh_genes):
            mh_name = LGPConfig.available_mh[(i + j) % len(LGPConfig.available_mh)]
            genes.append(Gene(kind="MH", name=mh_name, w_raw=random.uniform(0.1, 1.0)))
        
        dummy_library.append(ActionIndividual(genes=genes))
    
    return dummy_library


def main():
    """Main training entry point."""
    print("=" * 70)
    print("PPO + LGP COEVOLUTION TRAINING")
    print("=" * 70)
    
    # Set random seeds
    set_random_seeds(RANDOM_SEED)
    print(f"OK: Random seed set to: {RANDOM_SEED}")
    
    # Initialize LGP programs FIRST
    print(f"\nInitializing {LGPConfig.pool_size} LGP programs...")
    lgp_programs = initialize_lgp_programs(
        pool_size=LGPConfig.pool_size,
        max_length=LGPConfig.max_program_length,
        min_length=LGPConfig.min_program_length,
        num_registers=LGPConfig.num_registers,
        seed=RANDOM_SEED
    )
    print(f"OK: Initialized {len(lgp_programs)} programs")
    
    # Create dummy action library with CORRECT SIZE (= pool_size)
    # This ensures environment's action_space matches the number of LGP programs
    print(f"\nCreating action library with {LGPConfig.pool_size} slots...")
    dummy_action_library = create_dummy_action_library(LGPConfig.pool_size)
    print(f"OK: Action library created with {len(dummy_action_library)} portfolios")
    
    # Initialize environment WITH CORRECT ACTION LIBRARY SIZE
    print(f"\nCreating scheduling environment...")
    env = DynamicSchedulingEnv(
        lambda_tardiness=1.0,  # Used for initial schedule creation only
        action_library=dummy_action_library,  # FIX: Pass dummy library with correct size!
        action_budget_s=LGPConfig.action_budget_s,
        dataset_name=EnvironmentConfig.dataset_name  # Load dataset from config
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # Now act_dim = pool_size = 64
    print(f"OK: Environment created: obs_dim={obs_dim}, act_dim={act_dim}")
    
    # Verify action space matches pool size
    assert act_dim == LGPConfig.pool_size, \
        f"ACTION SPACE MISMATCH! act_dim={act_dim} but pool_size={LGPConfig.pool_size}"
    print(f"OK: Action space verified: {act_dim} actions = {LGPConfig.pool_size} LGP programs")
    
    # Initialize PPO model
    print(f"\nCreating PPO model...")
    model = PPOActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=PPOConfig.learning_rate)
    print(f"OK: PPO model created with lr={PPOConfig.learning_rate}")
    
    # Configure coevolution - use CoevolutionConfig directly
    print(f"\nConfiguring coevolution...")
    print(f"OK: Configuration: {CoevolutionConfig.num_generations} generations, {CoevolutionConfig.episodes_per_gen} episodes/gen")
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "programs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "portfolios"), exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")
    
    # Start training
    print(f"\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    lgp_programs, final_action_library = train_with_coevolution_lgp(
        env=env,
        lgp_programs=lgp_programs,
        model=model,
        optimizer=optimizer,
        select_action_fn=select_action,
        compute_returns_fn=compute_returns,
        cfg=CoevolutionConfig,  # Use config class directly
        output_dir=output_dir
    )
    
    # Save results
    model_path = os.path.join(output_dir, "models", "trained_policy.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nOK: Model saved to: {model_path}")
    
    # Save program info
    programs_data = {
        "num_programs": len(lgp_programs),
        "programs": [prog.to_dict() for prog in lgp_programs]
    }
    lgp_path = os.path.join(output_dir, "programs", "lgp_programs_final.json")
    with open(lgp_path, 'w') as f:
        json.dump(programs_data, f, indent=2)
    print(f"OK: LGP programs saved to: {lgp_path}")
    
    print(f"\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved in: {output_dir}/")
    print(f"  - Model: {model_path}")
    print(f"  - Programs: {lgp_path}")
    print(f"  - Metrics: {output_dir}/metrics/")
    print(f"  - Portfolios: {output_dir}/portfolios/")


if __name__ == "__main__":
    main()
