"""
Environment utility functions for PPO+LGP Dynamic Scheduling.
Includes scheduling helpers and dispatching heuristics.
"""

import numpy as np
import random
import copy
import math


# Global machine pool definition
machine_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15]


def simulated_annealing(jobs, due_dates, lambda_tardiness=1.0, **kwargs):
    """Dummy simulated_annealing: tạo lịch trình ban đầu dưới dạng dictionary."""
    # Khởi tạo thời gian sẵn sàng của các máy
    machine_ready = {m: 0 for m in machine_pool}
    schedule = {}
    # Với mỗi job, duyệt các operation theo thứ tự
    for job, ops in jobs.items():
        job_ready = 0  # Thời gian job sẵn sàng cho operation tiếp theo
        for i, op in enumerate(ops):
            best_machine = None
            best_start = None
            best_finish = float('inf')
            # Chọn máy trong candidate có thể bắt đầu sớm nhất
            for m in op['candidate_machines']:
                st = max(job_ready, machine_ready[m])
                ft = st + op['processing_time']
                if ft < best_finish:
                    best_finish = ft
                    best_start = st
                    best_machine = m
            schedule[(job, i)] = (best_start, best_finish, best_machine)
            job_ready = best_finish
            machine_ready[best_machine] = best_finish
    makespan = max(ft for (st, ft, m) in schedule.values())
    total_tardiness = sum(max(0, schedule[(job, i)][1] - due_dates[job])
                          for job in jobs for i in range(len(jobs[job])))
    cost = makespan + lambda_tardiness * total_tardiness
    return None, schedule, makespan, total_tardiness, cost, None


def schedule_dict_to_list(schedule_dict, jobs_info):
    """Chuyển schedule dictionary sang list các event."""
    events = []
    for (job, op_index), (s, f, m) in schedule_dict.items():
        op_info = jobs_info[job]['operations'][op_index]
        event = {
            'job': job,
            'op_index': op_index,
            'start': s,
            'finish': f,
            'machine': m,
            'op_id': op_info['op_id'],
            'candidate_machines': op_info['candidate_machines']
        }
        events.append(event)
    events = sorted(events, key=lambda e: (str(e['job']), e['op_index'], e['start']))
    return events


def split_schedule_list(event_list, current_time, jobs_info):
    """Hàm tách schedule (list event) theo current_time."""
    finished_events = []
    unfinished_jobs = {}
    jobs_events = {}
    for event in event_list:
        jobs_events.setdefault(event['job'], []).append(event)
    for job, events in jobs_events.items():
        events = sorted(events, key=lambda e: (e['op_index'], e['start']))
        ops_list = []
        job_ready = None
        for event in events:
            if event['finish'] <= current_time:
                finished_events.append(event)
                job_ready = event['finish']
            elif event['start'] < current_time < event['finish']:
                finished_part = event.copy()
                finished_part['finish'] = current_time
                finished_events.append(finished_part)
                remaining_time = event['finish'] - current_time
                unfinished_op = {
                    'op_index': event['op_index'],
                    'op_id': event['op_id'],
                    'candidate_machines': event['candidate_machines'],
                    'processing_time': remaining_time
                }
                ops_list.append(unfinished_op)
                job_ready = current_time
                total_ops = len(jobs_info[job]['operations'])
                for op_index in range(event['op_index']+1, total_ops):
                    op = jobs_info[job]['operations'][op_index]
                    new_op = {
                        'op_index': op_index,
                        'op_id': op['op_id'],
                        'candidate_machines': op['candidate_machines'],
                        'processing_time': op['processing_time']
                    }
                    ops_list.append(new_op)
                break
            else:
                unfinished_op = {
                    'op_index': event['op_index'],
                    'op_id': event['op_id'],
                    'candidate_machines': event['candidate_machines'],
                    'processing_time': event['finish'] - event['start']
                }
                ops_list.append(unfinished_op)
                if job_ready is None:
                    job_ready = current_time
        if ops_list:
            unfinished_jobs[job] = {
                'job_ready': job_ready,
                'due_date': jobs_info[job]['due_date'],
                'operations': ops_list
            }
    return finished_events, unfinished_jobs


def create_unified_jobs_info(jobs_initial, due_dates_initial):
    """Hàm tạo unified job info cho các job ban đầu."""
    info = {}
    for job, ops in jobs_initial.items():
        info[job] = {
            'operations': ops,
            'due_date': due_dates_initial[job]
        }
    return info


# ==================== Dispatching Heuristics ====================

def reschedule_unfinished_jobs_edd(unfinished_jobs, current_time, finished_events, machine_pool):
    """Earliest Due Date (EDD) dispatching."""
    sorted_jobs = sorted(unfinished_jobs.items(), 
                        key=lambda x: (x[1]['due_date'], 
                                     sum(op['processing_time'] for op in x[1]['operations'])))
    new_events = []
    machine_ready = {m: current_time for m in machine_pool}
    for job, info in sorted_jobs:
        job_ready = info['job_ready']
        for op in sorted(info['operations'], key=lambda op: op['op_index']):
            pt = op['processing_time']
            best_start = float('inf')
            best_finish = float('inf')
            best_machine = None
            for m in op['candidate_machines']:
                st = max(job_ready, machine_ready.get(m, current_time))
                ft = st + pt
                if ft < best_finish:
                    best_finish = ft
                    best_start = st
                    best_machine = m
            event = {
                'job': job,
                'op_index': op['op_index'],
                'start': best_start,
                'finish': best_finish,
                'machine': best_machine,
                'op_id': op['op_id'],
                'candidate_machines': op['candidate_machines']
            }
            new_events.append(event)
            job_ready = best_finish
            machine_ready[best_machine] = best_finish
    return new_events


def reschedule_unfinished_jobs_spt(unfinished_jobs, current_time, finished_events, machine_pool):
    """Shortest Processing Time (SPT)."""
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: sum(op['processing_time'] for op in x[1]['operations'])
    )
    new_events = []
    machine_ready = {m: current_time for m in machine_pool}
    for job, info in sorted_jobs:
        job_ready = info['job_ready']
        for op in sorted(info['operations'], key=lambda op: op['op_index']):
            pt = op['processing_time']
            best_start = float('inf')
            best_finish = float('inf')
            best_machine = None
            for m in op['candidate_machines']:
                st = max(job_ready, machine_ready.get(m, current_time))
                ft = st + pt
                if ft < best_finish:
                    best_finish = ft
                    best_start = st
                    best_machine = m
            event = {
                'job': job,
                'op_index': op['op_index'],
                'start': best_start,
                'finish': best_finish,
                'machine': best_machine,
                'op_id': op['op_id'],
                'candidate_machines': op['candidate_machines']
            }
            new_events.append(event)
            job_ready = best_finish
            machine_ready[best_machine] = best_finish
    return new_events


def reschedule_unfinished_jobs_lpt(unfinished_jobs, current_time, finished_events, machine_pool):
    """Longest Processing Time (LPT)."""
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: sum(op['processing_time'] for op in x[1]['operations']),
        reverse=True
    )
    new_events = []
    machine_ready = {m: current_time for m in machine_pool}
    for job, info in sorted_jobs:
        job_ready = info['job_ready']
        for op in sorted(info['operations'], key=lambda op: op['op_index']):
            pt = op['processing_time']
            best_start = float('inf')
            best_finish = float('inf')
            best_machine = None
            for m in op['candidate_machines']:
                st = max(job_ready, machine_ready.get(m, current_time))
                ft = st + pt
                if ft < best_finish:
                    best_finish = ft
                    best_start = st
                    best_machine = m
            event = {
                'job': job,
                'op_index': op['op_index'],
                'start': best_start,
                'finish': best_finish,
                'machine': best_machine,
                'op_id': op['op_id'],
                'candidate_machines': op['candidate_machines']
            }
            new_events.append(event)
            job_ready = best_finish
            machine_ready[best_machine] = best_finish
    return new_events


def reschedule_unfinished_jobs_fcfs(unfinished_jobs, current_time, finished_events, machine_pool):
    """First Come First Served (FCFS/FIFO)."""
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: x[1]['job_ready']
    )
    new_events = []
    machine_ready = {m: current_time for m in machine_pool}
    for job, info in sorted_jobs:
        job_ready = info['job_ready']
        for op in sorted(info['operations'], key=lambda op: op['op_index']):
            pt = op['processing_time']
            best_start = float('inf')
            best_finish = float('inf')
            best_machine = None
            for m in op['candidate_machines']:
                st = max(job_ready, machine_ready.get(m, current_time))
                ft = st + pt
                if ft < best_finish:
                    best_finish = ft
                    best_start = st
                    best_machine = m
            event = {
                'job': job,
                'op_index': op['op_index'],
                'start': best_start,
                'finish': best_finish,
                'machine': best_machine,
                'op_id': op['op_id'],
                'candidate_machines': op['candidate_machines']
            }
            new_events.append(event)
            job_ready = best_finish
            machine_ready[best_machine] = best_finish
    return new_events


def reschedule_unfinished_jobs_cr(unfinished_jobs, current_time, finished_events, machine_pool):
    """Critical Ratio (CR)."""
    def calculate_cr(info):
        remaining_pt = sum(op['processing_time'] for op in info['operations'])
        slack = info['due_date'] - current_time
        if remaining_pt <= 0:
            return float('inf')
        return slack / remaining_pt
    
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: calculate_cr(x[1])
    )
    new_events = []
    machine_ready = {m: current_time for m in machine_pool}
    for job, info in sorted_jobs:
        job_ready = info['job_ready']
        for op in sorted(info['operations'], key=lambda op: op['op_index']):
            pt = op['processing_time']
            best_start = float('inf')
            best_finish = float('inf')
            best_machine = None
            for m in op['candidate_machines']:
                st = max(job_ready, machine_ready.get(m, current_time))
                ft = st + pt
                if ft < best_finish:
                    best_finish = ft
                    best_start = st
                    best_machine = m
            event = {
                'job': job,
                'op_index': op['op_index'],
                'start': best_start,
                'finish': best_finish,
                'machine': best_machine,
                'op_id': op['op_id'],
                'candidate_machines': op['candidate_machines']
            }
            new_events.append(event)
            job_ready = best_finish
            machine_ready[best_machine] = best_finish
    return new_events


def reschedule_unfinished_jobs_sa(unfinished_jobs, current_time, finished_events, machine_pool, iterations=50):
    """Simulated Annealing for rescheduling."""
    current_solution = reschedule_unfinished_jobs_edd(unfinished_jobs, current_time, finished_events, machine_pool)
    current_cost = max(e['finish'] for e in (finished_events + current_solution))
    T = 100
    cooling_rate = 0.95
    best_solution = current_solution
    best_cost = current_cost
    for i in range(iterations):
        neighbor = copy.deepcopy(current_solution)
        if neighbor and len(neighbor) > 0:
            idx = random.randint(0, len(neighbor)-1)
            neighbor[idx]['finish'] *= random.uniform(1.0, 1.05)
        merged = finished_events + neighbor
        makespan = max(e['finish'] for e in merged) if merged else 0
        new_cost = makespan
        if new_cost < best_cost or random.random() < math.exp(-(new_cost - current_cost)/T):
            current_solution = neighbor
            current_cost = new_cost
            if new_cost < best_cost:
                best_solution = neighbor
                best_cost = new_cost
        T *= cooling_rate
    return best_solution


def reschedule_unfinished_jobs_ga(unfinished_jobs, current_time, finished_events, machine_pool, 
                                  num_candidates=10, generations=5):
    """Genetic Algorithm for rescheduling."""
    population = [reschedule_unfinished_jobs_edd(unfinished_jobs, current_time, finished_events, machine_pool) 
                 for _ in range(num_candidates)]
    
    def evaluate(solution):
        merged = finished_events + solution
        return max(e['finish'] for e in merged)
    
    for gen in range(generations):
        population = sorted(population, key=evaluate)[:max(1, num_candidates//2)]
        new_population = []
        while len(new_population) < num_candidates:
            parent1, parent2 = random.sample(population, 2)
            child = []
            for e1, e2 in zip(parent1, parent2):
                child.append(e1 if random.random() < 0.5 else e2)
            new_population.append(child)
        for solution in new_population:
            if random.random() < 0.3 and len(solution) > 0:
                idx = random.randint(0, len(solution)-1)
                solution[idx]['finish'] *= random.uniform(0.95, 1.05)
        population = new_population
    best_solution = min(population, key=evaluate)
    return best_solution


def reschedule_unfinished_jobs_pso(unfinished_jobs, current_time, finished_events, machine_pool, 
                                   num_particles=10, iterations=20):
    """Particle Swarm Optimization for rescheduling."""
    def cost_function(candidate):
        merged = finished_events + candidate
        return max(e['finish'] for e in merged) if merged else 0

    particles = []
    velocities = []
    base_candidate = reschedule_unfinished_jobs_edd(unfinished_jobs, current_time, finished_events, machine_pool)
    for i in range(num_particles):
        candidate = copy.deepcopy(base_candidate)
        for event in candidate:
            event['finish'] *= random.uniform(0.95, 1.05)
        particles.append(candidate)
        velocities.append([0]*len(candidate))

    pbest = copy.deepcopy(particles)
    pbest_costs = [cost_function(p) for p in particles]
    gbest = min(particles, key=cost_function)
    gbest_cost = cost_function(gbest)

    w = 0.5
    c1 = 1.0
    c2 = 1.0

    for it in range(iterations):
        for i in range(num_particles):
            for j in range(len(particles[i])):
                current_finish = particles[i][j]['finish']
                pbest_finish = pbest[i][j]['finish']
                gbest_finish = gbest[j]['finish']
                r1 = random.random()
                r2 = random.random()
                new_velocity = w * velocities[i][j] + c1 * r1 * (pbest_finish - current_finish) + c2 * r2 * (gbest_finish - current_finish)
                velocities[i][j] = new_velocity
                particles[i][j]['finish'] = current_finish + new_velocity
            cost_candidate = cost_function(particles[i])
            if cost_candidate < pbest_costs[i]:
                pbest[i] = copy.deepcopy(particles[i])
                pbest_costs[i] = cost_candidate
        candidate_costs = [cost_function(p) for p in particles]
        min_cost = min(candidate_costs)
        if min_cost < gbest_cost:
            gbest = copy.deepcopy(particles[candidate_costs.index(min_cost)])
            gbest_cost = min_cost
    return gbest
