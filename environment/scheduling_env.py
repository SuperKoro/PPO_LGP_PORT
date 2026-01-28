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
    get_op_processing_time,
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
        jobs_all, due_dates_all, machine_pool_loaded, arrival_times, initial_job_ids, job_types = load_dataset(
            dataset_name, return_meta=True
        )
        
        self.machine_pool = machine_pool_loaded

        # If dataset provides arrival_times + initial_job_ids, use predefined dynamic jobs
        self._predefined_dynamic_jobs = None
        if arrival_times is not None and initial_job_ids is not None:
            jobs_initial = {jid: jobs_all[jid] for jid in initial_job_ids}
            due_dates_initial = {jid: due_dates_all[jid] for jid in initial_job_ids}

            dynamic_jobs_events = []
            for jid, ops in jobs_all.items():
                if jid in jobs_initial:
                    continue
                dyn_job = {
                    'job_id': jid,
                    'arrival_time': arrival_times.get(jid, 0),
                    'due_date': due_dates_all[jid],
                    'operations': ops,
                    'job_type': (job_types.get(jid) if job_types else 'Normal'),
                }
                dynamic_jobs_events.append((dyn_job['arrival_time'], dyn_job))

            dynamic_jobs_events.sort(key=lambda x: (x[0], x[1]['job_id']))
            self._predefined_dynamic_jobs = dynamic_jobs_events
        else:
            jobs_initial = jobs_all
            due_dates_initial = due_dates_all
        
        self.jobs_initial = jobs_initial
        self.due_dates_initial = due_dates_initial
        self._job_types = job_types
        self.all_jobs_info = create_unified_jobs_info(self.jobs_initial, self.due_dates_initial)
        if self._job_types:
            for jid in self.jobs_initial.keys():
                self.all_jobs_info[jid]['job_type'] = self._job_types.get(jid, 'Normal')
        
        # Create initial schedule once offline
        _, schedule, _, _, _, _ = simulated_annealing(
            self.jobs_initial,
            self.due_dates_initial,
            lambda_tardiness=1.0  # Initial schedule parameter
        )
        self.initial_schedule_events = schedule_dict_to_list(schedule, self.all_jobs_info)
        self.current_schedule_events = copy.deepcopy(self.initial_schedule_events)
        self.current_time = 0
        # Use predefined dynamic jobs if available, else generate randomly
        if self._predefined_dynamic_jobs is not None:
            self.dynamic_jobs_events = copy.deepcopy(self._predefined_dynamic_jobs)
        else:
            self._generate_dynamic_jobs(num_dynamic=EnvironmentConfig.num_dynamic_jobs)
        self.current_dynamic_index = 0
        
        # Action library + budget
        self.action_library = action_library if action_library is not None else self._build_default_action_library()
        self.action_budget_s = float(action_budget_s)
        
        # Number of machines for state representation
        self.num_machines = len(self.machine_pool)
        
        # Observation space: 12 global features + 2 per-machine features
        # Global: 6 original + 6 queue-based features
        # Total = 12 + 2 * num_machines
        obs_dim = 12 + 2 * self.num_machines
        self.observation_space = spaces.Box(low=-10, high=10, shape=(obs_dim,), dtype=np.float32)
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
        if hasattr(self, "_last_cost"):
            self._last_cost = None
        self.all_jobs_info = create_unified_jobs_info(self.jobs_initial, self.due_dates_initial)
        if self._job_types:
            for jid in self.jobs_initial.keys():
                self.all_jobs_info[jid]['job_type'] = self._job_types.get(jid, 'Normal')
        self.current_schedule_events = copy.deepcopy(self.initial_schedule_events)
        if self._predefined_dynamic_jobs is not None:
            self.dynamic_jobs_events = copy.deepcopy(self._predefined_dynamic_jobs)
        else:
            self._generate_dynamic_jobs(num_dynamic=EnvironmentConfig.num_dynamic_jobs)
        self.current_dynamic_index = 0
        return self._get_state()

    def _get_state(self):
        """
        Enhanced State Representation for JSSP.
        
        Features:
        - 12 global features (6 original + 6 queue-based)
        - 2 * num_machines per-machine features (finish_time, queue_load)
        
        Total: 12 + 2 * num_machines dimensions
        """
        finished_events, unfinished_jobs = split_schedule_list(
            self.current_schedule_events, 
            self.current_time, 
            self.all_jobs_info
        )
        
        # Scale factors - adjusted for typical JSSP values
        TIME_SCALE = 2000.0       # Max expected makespan
        SLACK_SCALE = 1000.0      # Max slack time
        PT_SCALE = 100.0          # Max processing time per operation
        QUEUE_SCALE = 200.0       # Max queue load per machine
        
        # ============ GLOBAL FEATURES (Original 6) ============
        
        # 1. Time progress (0-1)
        time_progress = self.current_time / TIME_SCALE
        
        # 2. Number of unfinished jobs (normalized)
        num_jobs = len(unfinished_jobs)
        norm_num_jobs = num_jobs / 50.0
        
        # 3-4. Slack features (all jobs)
        slacks = []
        urgent_count = 0
        for job, info in unfinished_jobs.items():
            remaining_pt = sum(get_op_processing_time(op) for op in info['operations'])
            slack = info['due_date'] - self.current_time - remaining_pt
            slacks.append(slack)
            if info.get('job_type', 'Normal') == 'Urgent':
                urgent_count += 1
        
        min_slack = min(slacks) / SLACK_SCALE if slacks else 0
        avg_slack = (sum(slacks) / len(slacks) / SLACK_SCALE) if slacks else 0
        
        # 5. Urgent ratio (0-1)
        urgent_ratio = urgent_count / num_jobs if num_jobs > 0 else 0
        
        # 6. Total remaining work (normalized)
        total_remaining_pt = 0
        for info in unfinished_jobs.values():
            for op in info['operations']:
                total_remaining_pt += get_op_processing_time(op)
        norm_remaining_pt = total_remaining_pt / (PT_SCALE * 50)
        
        # ============ QUEUE-BASED FEATURES (New 6) ============
        # Features about operations that are READY to be scheduled now
        
        ready_ops_pt = []       # Processing time of ready operations
        ready_ops_slack = []    # Slack of jobs with ready operations
        num_ready_urgent = 0
        
        for job_id, info in unfinished_jobs.items():
            # Check if job is ready (arrived and has operations)
            job_ready_time = info.get('job_ready', 0)
            if job_ready_time <= self.current_time and info['operations']:
                # Get next operation to be scheduled
                next_op = info['operations'][0]
                pt = get_op_processing_time(next_op)
                ready_ops_pt.append(pt)
                
                # Calculate slack for this job
                remaining_pt_job = sum(get_op_processing_time(op) for op in info['operations'])
                slack = info['due_date'] - self.current_time - remaining_pt_job
                ready_ops_slack.append(slack)
                
                if info.get('job_type', 'Normal') == 'Urgent':
                    num_ready_urgent += 1
        
        # Queue statistics
        if ready_ops_pt:
            min_pt_waiting = min(ready_ops_pt) / PT_SCALE
            max_pt_waiting = max(ready_ops_pt) / PT_SCALE
            std_pt_waiting = float(np.std(ready_ops_pt)) / PT_SCALE if len(ready_ops_pt) > 1 else 0.0
        else:
            min_pt_waiting = max_pt_waiting = std_pt_waiting = 0.0
        
        if ready_ops_slack:
            min_slack_waiting = min(ready_ops_slack) / SLACK_SCALE
            avg_slack_waiting = float(np.mean(ready_ops_slack)) / SLACK_SCALE
        else:
            min_slack_waiting = avg_slack_waiting = 0.0
        
        ratio_urgent_waiting = num_ready_urgent / len(ready_ops_pt) if ready_ops_pt else 0.0
        
        # ============ PER-MACHINE FEATURES ============
        
        # Initialize per-machine arrays
        machine_finish_times = np.zeros(self.num_machines)
        machine_queue_loads = np.zeros(self.num_machines)
        
        # Create machine_id to index mapping
        machine_to_idx = {m: i for i, m in enumerate(sorted(self.machine_pool))}
        
        # Calculate when each machine will finish current work
        for event in self.current_schedule_events:
            if event['finish'] > self.current_time:
                m = event['machine']
                if m in machine_to_idx:
                    m_idx = machine_to_idx[m]
                    machine_finish_times[m_idx] = max(
                        machine_finish_times[m_idx], 
                        event['finish']
                    )
                    # Add remaining work on this machine
                    remaining_work = event['finish'] - max(event['start'], self.current_time)
                    machine_queue_loads[m_idx] += remaining_work
        
        # Normalize machine features
        rel_finish_times = (machine_finish_times - self.current_time) / TIME_SCALE
        rel_finish_times = np.clip(rel_finish_times, 0, 1)
        
        norm_queue_loads = machine_queue_loads / QUEUE_SCALE
        norm_queue_loads = np.clip(norm_queue_loads, 0, 1)
        
        # ============ ASSEMBLE STATE VECTOR ============
        state_features = [
            # Original 6 global features
            time_progress,           # [0] Time progress
            norm_num_jobs,           # [1] Number of jobs
            min_slack,               # [2] Min slack (all jobs)
            avg_slack,               # [3] Avg slack (all jobs)
            urgent_ratio,            # [4] Overall urgent ratio
            norm_remaining_pt,       # [5] Total remaining work
            # New 6 queue-based features
            min_pt_waiting,          # [6] Min PT of ready ops (helps decide SPT)
            max_pt_waiting,          # [7] Max PT of ready ops
            std_pt_waiting,          # [8] Std of PT (high = SPT effective)
            min_slack_waiting,       # [9] Min slack of ready jobs (helps decide EDD)
            avg_slack_waiting,       # [10] Avg slack of ready jobs
            ratio_urgent_waiting,    # [11] Ratio of urgent ready jobs
        ]
        
        # Add per-machine features
        state_features.extend(rel_finish_times.tolist())   # [12 : 12+M]
        state_features.extend(norm_queue_loads.tolist())   # [12+M : 12+2M]
        
        state = np.array(state_features, dtype=np.float32)
        
        # Replace NaN/Inf with 0 for stability
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state

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
            op_entry = {
                'op_index': i,
                'op_id': op['op_id'],
                'candidate_machines': op['candidate_machines'],
                'processing_time': op['processing_time'],
            }
            if 'processing_times' in op:
                op_entry['processing_times'] = op['processing_times']
            ops_list.append(op_entry)
        
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

        # Use delta-cost reward with step penalty
        if getattr(self, "_last_cost", None) is None:
            self._last_cost = cost
        
        # Improvement reward: positive if cost decreased
        improvement = self._last_cost - cost
        
        # Step penalty: encourage agent to finish faster
        step_penalty = 0.1
        
        # Total reward = improvement - penalty
        reward = improvement - step_penalty
        
        self._last_cost = cost

        self.current_dynamic_index += 1
        done = self.current_dynamic_index >= len(self.dynamic_jobs_events)
        next_state = self._get_state()
        
        return next_state, reward, done, {}
