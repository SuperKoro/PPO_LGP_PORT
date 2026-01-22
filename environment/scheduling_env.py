"""
Dynamic Scheduling Environment for PPO+LGP.
Gym environment for dynamic job shop scheduling with disruptions.
"""

import numpy as np
import random
import copy
import math
import gym
from gym import spaces

from environment.env_utils import (
    simulated_annealing,
    schedule_dict_to_list,
    split_schedule_list,
    create_unified_jobs_info,
    machine_pool
)
from environment.dataset_loader import load_dataset
from training.portfolio_types import Gene, ActionIndividual
from config import EnvironmentConfig
# NOTE: run_action_individual is imported inside step() to avoid circular import


class DynamicSchedulingEnv(gym.Env):
    """
    Gym environment for dynamic job shop scheduling.
    
    - Initial jobs are scheduled
    - Dynamic jobs arrive during execution
    - Agent selects portfolios (DR + MH) to reschedule
    """
    
    def __init__(self,
                 lambda_tardiness: float = 1.0,
                 action_library: list = None,
                 action_budget_s: float = 3.0,
                 dataset_name: str = None):
        super(DynamicSchedulingEnv, self).__init__()
        self.lambda_tardiness = lambda_tardiness
        
        # Load dataset (either from file or use default hardcoded)
        jobs_initial, due_dates_initial, machine_pool_loaded = load_dataset(dataset_name)
        
        self.machine_pool = machine_pool_loaded
        self.jobs_initial = jobs_initial
        self.due_dates_initial = due_dates_initial
        self.all_jobs_info = create_unified_jobs_info(self.jobs_initial, self.due_dates_initial)
        
        # Create initial schedule once offline
        _, schedule, _, _, _, _ = simulated_annealing(
            self.jobs_initial,
            self.due_dates_initial,
            lambda_tardiness=1.0  # Initial schedule parameter
        )
        self.initial_schedule_events = schedule_dict_to_list(schedule, self.all_jobs_info)
        self.current_schedule_events = copy.deepcopy(self.initial_schedule_events)
        self.current_time = 0
        self._generate_dynamic_jobs(num_dynamic=EnvironmentConfig.num_dynamic_jobs)
        self.current_dynamic_index = 0
        
        # Action library + budget
        self.action_library = action_library if action_library is not None else self._build_default_action_library()
        self.action_budget_s = float(action_budget_s)
        
        self.observation_space = spaces.Box(low=-1000, high=10000, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_library))

    def _build_default_action_library(self):
        """
        Create 4 default actions: SA, GA, EDD, PSO
        for backward-compatibility when not using LGP.
        """
        def make_one(mh_name: str) -> ActionIndividual:
            genes = [Gene(kind="DR", name="EDD", w_raw=1.0)]
            # 3 MH genes, only first has weight, others are 0
            genes.append(Gene(kind="MH", name=mh_name, w_raw=1.0))
            genes.append(Gene(kind="MH", name=mh_name, w_raw=0.0))
            genes.append(Gene(kind="MH", name=mh_name, w_raw=0.0))
            return ActionIndividual(genes=genes)

        actions = [
            make_one("SA"),
            make_one("GA"),
            make_one("EDD"),
            make_one("PSO"),
        ]
        return actions
    
    def seed(self, seed=None):
        """
        ⭐ FIX 1: Set random seed for reproducible episodes.
        This ensures dynamic job generation is deterministic.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _generate_dynamic_job(self, job_id, arrival_time, min_ops=1, max_ops=5, min_pt=5, max_pt=50):
        """Generate a single dynamic job."""
        # 25% chance urgent, 75% normal
        if random.random() < 0.25:
            job_type = "Urgent"
            etuf = 1.2
        else:
            job_type = "Normal"
            etuf = 1.8
        
        num_ops = random.randint(min_ops, max_ops)
        operations = []
        total_pt = 0
        
        for i in range(num_ops):
            candidate_machines = random.sample(
                self.machine_pool, 
                k=random.randint(1, min(5, len(self.machine_pool)))
            )
            pt = random.randint(min_pt, max_pt)
            total_pt += pt
            op = {
                'op_id': i+1,
                'candidate_machines': candidate_machines,
                'processing_time': pt
            }
            operations.append(op)
        
        due_date = math.ceil(arrival_time + total_pt * etuf)
        dynamic_job = {
            'job_id': job_id,
            'arrival_time': arrival_time,
            'due_date': due_date,
            'operations': operations,
            'job_type': job_type
        }
        return dynamic_job

    def _generate_dynamic_jobs(self, num_dynamic=4):
        """Generate multiple dynamic jobs with staggered arrival times."""
        dynamic_jobs_events = []
        Eave = 20  # Average interarrival time
        
        # Calculate time bounds
        max_finish = max(e['finish'] for e in self.current_schedule_events)
        margin = 5
        T_max = int(max_finish - margin)
        T_min = self.current_time + 5
        if T_min >= T_max:
            T_max = T_min + 10

        # Generate first arrival time
        arrival_time = self.current_time + int(np.random.exponential(scale=Eave))
        arrival_time = max(T_min, min(arrival_time, T_max - (num_dynamic - 1)))
        
        for i in range(num_dynamic):
            if i > 0:
                arrival_time += int(np.random.exponential(scale=Eave))
            
            # Ensure within bounds
            max_allowed = T_max - (num_dynamic - i - 1)
            if arrival_time > max_allowed:
                arrival_time = max_allowed
            
            temp_id = "Temp" + str(i+1)
            dyn_job = self._generate_dynamic_job(temp_id, arrival_time)
            dynamic_jobs_events.append((arrival_time, dyn_job))
        
        # Sort and assign proper IDs
        dynamic_jobs_events.sort(key=lambda x: x[0])
        for i, (arrival_time, dj) in enumerate(dynamic_jobs_events):
            dj['job_id'] = "D" + str(i+1)
        
        self.dynamic_jobs_events = dynamic_jobs_events

    def reset(self):
        """Reset environment to initial state."""
        self.current_time = 0
        self.all_jobs_info = create_unified_jobs_info(self.jobs_initial, self.due_dates_initial)
        self.current_schedule_events = copy.deepcopy(self.initial_schedule_events)
        self._generate_dynamic_jobs(num_dynamic=EnvironmentConfig.num_dynamic_jobs)
        self.current_dynamic_index = 0
        return self._get_state()

    def _get_state(self):
        """
        Get current state observation.
        
        Enhanced observation space (10D):
        [0] current_time - Current simulation time
        [1] num_unfinished_ops - Number of unfinished operations
        [2] avg_processing_time - Average processing time of remaining ops
        [3] min_slack - Minimum slack time (due_date - current_time - remaining_pt)
        [4] max_slack - Maximum slack time
        [5] urgent_ratio - Ratio of urgent jobs to total jobs
        [6] total_remaining_pt - Total remaining processing time
        [7] num_jobs - Number of unfinished jobs
        [8] avg_due_date - Average due date of unfinished jobs
        [9] machine_load_std - Standard deviation of machine load (utilization variance)
        """
        finished_events, unfinished_jobs = split_schedule_list(
            self.current_schedule_events, 
            self.current_time, 
            self.all_jobs_info
        )
        
        # Basic features
        num_jobs = len(unfinished_jobs)
        num_unfinished_ops = sum(len(info['operations']) for info in unfinished_jobs.values())
        
        # Processing time features
        total_pt = 0
        count = 0
        for info in unfinished_jobs.values():
            for op in info['operations']:
                total_pt += op['processing_time']
                count += 1
        avg_pt = total_pt / count if count > 0 else 0
        
        # Slack time features (due_date - current_time - remaining_pt)
        slacks = []
        urgent_count = 0
        due_dates = []
        
        for job, info in unfinished_jobs.items():
            remaining_pt = sum(op['processing_time'] for op in info['operations'])
            slack = info['due_date'] - self.current_time - remaining_pt
            slacks.append(slack)
            due_dates.append(info['due_date'])
            
            # Check if urgent (from dynamic jobs)
            if info.get('job_type', 'Normal') == 'Urgent':
                urgent_count += 1
        
        min_slack = min(slacks) if slacks else 0
        max_slack = max(slacks) if slacks else 0
        avg_due_date = sum(due_dates) / len(due_dates) if due_dates else self.current_time
        urgent_ratio = urgent_count / num_jobs if num_jobs > 0 else 0
        
        # Machine load variance (how evenly distributed is the work)
        machine_loads = {}
        for event in self.current_schedule_events:
            if event['finish'] > self.current_time:  # Only future events
                m = event['machine']
                duration = event['finish'] - max(event['start'], self.current_time)
                machine_loads[m] = machine_loads.get(m, 0) + duration
        
        if machine_loads:
            loads = list(machine_loads.values())
            mean_load = sum(loads) / len(loads)
            machine_load_std = (sum((l - mean_load)**2 for l in loads) / len(loads)) ** 0.5
        else:
            machine_load_std = 0
        
        # =====================================================
        # NORMALIZATION: Scale all features to similar ranges
        # This helps neural network training significantly
        # =====================================================
        # Scale factors (approximate max values for normalization)
        TIME_SCALE = 200.0        # Max expected time
        OPS_SCALE = 50.0          # Max expected operations
        PT_SCALE = 50.0           # Max processing time
        SLACK_SCALE = 500.0       # Max slack time (can be negative)
        JOBS_SCALE = 30.0         # Max jobs
        LOAD_SCALE = 100.0        # Max load std
        
        return np.array([
            self.current_time / TIME_SCALE,              # [0] Normalized time (0-1+)
            num_unfinished_ops / OPS_SCALE,              # [1] Normalized ops (0-1)
            avg_pt / PT_SCALE,                           # [2] Normalized avg PT (0-1)
            min_slack / SLACK_SCALE,                     # [3] Normalized min slack (-1 to 1)
            max_slack / SLACK_SCALE,                     # [4] Normalized max slack (-1 to 1)
            urgent_ratio,                                # [5] Already 0-1
            total_pt / (PT_SCALE * OPS_SCALE),          # [6] Normalized total PT (0-1)
            num_jobs / JOBS_SCALE,                       # [7] Normalized num jobs (0-1)
            avg_due_date / (TIME_SCALE * 10),           # [8] Normalized due date (0-1)
            machine_load_std / LOAD_SCALE                # [9] Normalized load std (0-1)
        ], dtype=np.float32)

    def get_metrics(self):
        """Calculate current scheduling metrics."""
        merged = self.current_schedule_events
        makespan = max(e['finish'] for e in merged) if merged else 0
        
        total_tardiness_normal = 0
        total_tardiness_urgent = 0
        
        for job, info in self.all_jobs_info.items():
            job_events = [e for e in merged if e['job'] == job]
            if job_events:
                comp_time = max(e['finish'] for e in job_events)
                tardiness = max(0, comp_time - info['due_date'])
                
                if isinstance(job, int):
                    total_tardiness_normal += tardiness
                else:
                    if info.get('job_type', 'Normal') == 'Urgent':
                        total_tardiness_urgent += tardiness
                    else:
                        total_tardiness_normal += tardiness
        
        return {
            "makespan": makespan, 
            "tardiness_normal": total_tardiness_normal, 
            "tardiness_urgent": total_tardiness_urgent
        }

    def step(self, action):
        """
        Execute one step.
        
        Args:
            action: Index into self.action_library
            
        Returns:
            next_state, reward, done, info
        """
        # Lazy import to avoid circular dependency
        from training.typed_action_adapter import run_action_individual
        
        # Check if all dynamic jobs processed
        if self.current_dynamic_index >= len(self.dynamic_jobs_events):
            return self._get_state(), 0.0, True, {}

        # Get current dynamic job and update time
        arrival_time, dyn_job = self.dynamic_jobs_events[self.current_dynamic_index]
        self.current_time = arrival_time

        # Split schedule
        finished_events, unfinished_jobs = split_schedule_list(
            self.current_schedule_events,
            self.current_time,
            self.all_jobs_info
        )

        # Add new dynamic job to unfinished
        ops_list = []
        for i, op in enumerate(dyn_job['operations']):
            ops_list.append({
                'op_index': i,
                'op_id': op['op_id'],
                'candidate_machines': op['candidate_machines'],
                'processing_time': op['processing_time'],
            })
        
        dyn_info = {
            'job_ready': self.current_time,
            'due_date': dyn_job['due_date'],
            'operations': ops_list,
            'job_type': dyn_job.get('job_type', 'Normal'),
        }
        job_id = dyn_job['job_id']
        self.all_jobs_info[job_id] = dyn_info
        unfinished_jobs[job_id] = dyn_info

        # Execute action (portfolio with DR + MH)
        individual = self.action_library[action]
        
        new_unfinished_events = run_action_individual(
            env=self,
            individual=individual,
            finished_events=finished_events,
            unfinished_jobs=unfinished_jobs,
            total_budget_s=self.action_budget_s
        )

        self.current_schedule_events = finished_events + new_unfinished_events

        # Calculate reward
        merged = self.current_schedule_events
        makespan = max(e['finish'] for e in merged) if merged else 0

        total_tardiness_normal = 0.0
        total_tardiness_urgent = 0.0
        for job, info in self.all_jobs_info.items():
            job_events = [e for e in merged if e['job'] == job]
            if not job_events:
                continue
            comp_time = max(e['finish'] for e in job_events)
            tardiness = max(0, comp_time - info['due_date'])
            
            if isinstance(job, int):
                total_tardiness_normal += tardiness
            else:
                if info.get('job_type', 'Normal') == 'Urgent':
                    total_tardiness_urgent += tardiness
                else:
                    total_tardiness_normal += tardiness

        # Reward function using config parameters
        # Reward = -(alpha * makespan + (1-alpha) * (tardiness_normal + beta * tardiness_urgent))
        alpha = EnvironmentConfig.reward_alpha
        beta = EnvironmentConfig.reward_beta
        
        # Combined cost: weighted sum of makespan and tardiness
        cost = alpha * makespan + (1 - alpha) * (total_tardiness_normal + beta * total_tardiness_urgent)
        reward = -cost  # Negative because we want to minimize

        self.current_dynamic_index += 1
        done = self.current_dynamic_index >= len(self.dynamic_jobs_events)
        next_state = self._get_state()
        
        return next_state, reward, done, {}
