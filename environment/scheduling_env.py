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
from training.portfolio_types import Gene, ActionIndividual
from training.typed_action_adapter import run_action_individual


# Initial jobs data
jobs_initial = {
    1: [{'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12}],
    2: [{'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12}],
    3: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1},
        {'op_id': 2, 'candidate_machines': [6], 'processing_time': 8},
        {'op_id': 3, 'candidate_machines': [6], 'processing_time': 8}
    ],
    4: [{'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 7}],
    5: [{'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1}],
    6: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1},
        {'op_id': 2, 'candidate_machines': [6], 'processing_time': 8},
        {'op_id': 3, 'candidate_machines': [6], 'processing_time': 8}
    ],
    7: [{'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 7}],
    8: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1},
        {'op_id': 2, 'candidate_machines': [6], 'processing_time': 8},
        {'op_id': 3, 'candidate_machines': [6], 'processing_time': 8}
    ],
    9: [{'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 7}],
    10: [{'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 7}],
    11: [{'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 7}],
    12: [{'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12}],
    13: [{'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12}],
    14: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1},
        {'op_id': 2, 'candidate_machines': [6], 'processing_time': 8},
        {'op_id': 3, 'candidate_machines': [6], 'processing_time': 8}
    ],
    15: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1},
        {'op_id': 2, 'candidate_machines': [7], 'processing_time': 43},
        {'op_id': 3, 'candidate_machines': [5], 'processing_time': 43}
    ],
    16: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12},
        {'op_id': 2, 'candidate_machines': [8], 'processing_time': 8}
    ],
    17: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12},
        {'op_id': 2, 'candidate_machines': [8], 'processing_time': 12}
    ],
    18: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12},
        {'op_id': 2, 'candidate_machines': [8], 'processing_time': 4}
    ],
    19: [{'op_id': 1, 'candidate_machines': [8], 'processing_time': 3}],
    20: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12},
        {'op_id': 2, 'candidate_machines': [12, 13], 'processing_time': 25}
    ]
}

# Due dates for all jobs
due_dates_initial = {i: 1200 for i in range(1, 51)}


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
                 action_budget_s: float = 3.0):
        super(DynamicSchedulingEnv, self).__init__()
        self.lambda_tardiness = lambda_tardiness
        self.machine_pool = machine_pool
        self.jobs_initial = jobs_initial
        self.due_dates_initial = due_dates_initial
        self.all_jobs_info = create_unified_jobs_info(self.jobs_initial, self.due_dates_initial)
        
        # Create initial schedule once offline
        _, schedule, _, _, _, _ = simulated_annealing(
            self.jobs_initial,
            self.due_dates_initial,
            lambda_tardiness=self.lambda_tardiness
        )
        self.initial_schedule_events = schedule_dict_to_list(schedule, self.all_jobs_info)
        self.current_schedule_events = copy.deepcopy(self.initial_schedule_events)
        self.current_time = 0
        self._generate_dynamic_jobs(num_dynamic=4)
        self.current_dynamic_index = 0
        
        # Action library + budget
        self.action_library = action_library if action_library is not None else self._build_default_action_library()
        self.action_budget_s = float(action_budget_s)
        
        self.observation_space = spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)
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
        self._generate_dynamic_jobs(num_dynamic=2)
        self.current_dynamic_index = 0
        return self._get_state()

    def _get_state(self):
        """Get current state observation."""
        finished_events, unfinished_jobs = split_schedule_list(
            self.current_schedule_events, 
            self.current_time, 
            self.all_jobs_info
        )
        num_unfinished = sum(len(info['operations']) for info in unfinished_jobs.values())
        
        total_pt = 0
        count = 0
        for info in unfinished_jobs.values():
            for op in info['operations']:
                total_pt += op['processing_time']
                count += 1
        avg_pt = total_pt / count if count > 0 else 0
        
        return np.array([self.current_time, num_unfinished, avg_pt], dtype=np.float32)

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

        # Reward = -makespan (simple version)
        cost = makespan
        reward = -cost

        self.current_dynamic_index += 1
        done = self.current_dynamic_index >= len(self.dynamic_jobs_events)
        next_state = self._get_state()
        
        return next_state, reward, done, {}
