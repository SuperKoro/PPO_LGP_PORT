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

def get_op_processing_time(op, machine=None):
    """
    Return processing time for an operation.
    If machine is provided and op has per-machine times, use that.
    Otherwise fallback to op['processing_time'] or average of processing_times.
    """
    pt_map = op.get('processing_times')
    if machine is not None and isinstance(pt_map, dict):
        if machine in pt_map:
            return float(pt_map[machine])
        if str(machine) in pt_map:
            return float(pt_map[str(machine)])

    if 'processing_time' in op:
        try:
            return float(op['processing_time'])
        except Exception:
            pass

    if isinstance(pt_map, dict) and pt_map:
        return float(sum(pt_map.values()) / len(pt_map))

    if 'start' in op and 'finish' in op:
        try:
            return float(op['finish'] - op['start'])
        except Exception:
            pass

    return 0.0


def simulated_annealing(jobs, due_dates, lambda_tardiness=1.0, **kwargs):
    """Dummy simulated_annealing: tạo lịch trình ban đầu dưới dạng dictionary."""
    # Khởi tạo thời gian sẵn sàng của các máy
    # Extract all unique machines from jobs' candidate_machines
    all_machines = set()
    for job, ops in jobs.items():
        for op in ops:
            all_machines.update(op['candidate_machines'])
    machine_ready = {m: 0 for m in all_machines}
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
                pt = get_op_processing_time(op, m)
                ft = st + pt
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
        if 'processing_times' in op_info:
            event['processing_times'] = op_info['processing_times']
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
                    pt_avg = get_op_processing_time(op)
                    new_op = {
                        'op_index': op_index,
                        'op_id': op['op_id'],
                        'candidate_machines': op['candidate_machines'],
                        'processing_time': pt_avg
                    }
                    if 'processing_times' in op:
                        new_op['processing_times'] = op['processing_times']
                    ops_list.append(new_op)
                break
            else:
                pt_avg = get_op_processing_time(event)
                unfinished_op = {
                    'op_index': event['op_index'],
                    'op_id': event['op_id'],
                    'candidate_machines': event['candidate_machines'],
                    'processing_time': pt_avg
                }
                if 'processing_times' in event:
                    unfinished_op['processing_times'] = event['processing_times']
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

def _init_machine_ready(finished_events, current_time, machine_pool):
    """Initialize machine ready times from finished events."""
    machine_ready = {m: current_time for m in machine_pool}
    for e in finished_events:
        m = e.get('machine')
        if m is None:
            continue
        machine_ready[m] = max(machine_ready.get(m, current_time), e.get('finish', current_time))
    return machine_ready


def _schedule_jobs_in_order(jobs_order, unfinished_jobs, current_time, finished_events, machine_pool):
    """Build a feasible schedule for unfinished jobs following a given job order."""
    new_events = []
    machine_ready = _init_machine_ready(finished_events, current_time, machine_pool)
    for job in jobs_order:
        info = unfinished_jobs[job]
        job_ready = info['job_ready']
        for op in sorted(info['operations'], key=lambda op: op['op_index']):
            best_start = float('inf')
            best_finish = float('inf')
            best_machine = None
            for m in op['candidate_machines']:
                st = max(job_ready, machine_ready.get(m, current_time))
                pt = get_op_processing_time(op, m)
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
            if 'processing_times' in op:
                event['processing_times'] = op['processing_times']
            new_events.append(event)
            job_ready = best_finish
            machine_ready[best_machine] = best_finish
    return new_events


def _compute_cost(finished_events, scheduled_events, due_dates):
    """Compute makespan + total tardiness for merged events."""
    merged = finished_events + scheduled_events
    if not merged:
        return 0.0, 0.0, 0.0
    makespan = max(e['finish'] for e in merged)
    completion = {}
    for e in merged:
        completion[e['job']] = max(completion.get(e['job'], 0), e['finish'])
    total_tardiness = 0.0
    for job, due in due_dates.items():
        if job in completion:
            total_tardiness += max(0.0, completion[job] - due)
    cost = makespan + total_tardiness
    return makespan, total_tardiness, cost


def reschedule_unfinished_jobs_edd(unfinished_jobs, current_time, finished_events, machine_pool):
    """Earliest Due Date (EDD) dispatching."""
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: (x[1]['due_date'],
                       sum(get_op_processing_time(op) for op in x[1]['operations']))
    )
    jobs_order = [j for j, _ in sorted_jobs]
    return _schedule_jobs_in_order(jobs_order, unfinished_jobs, current_time, finished_events, machine_pool)


def reschedule_unfinished_jobs_spt(unfinished_jobs, current_time, finished_events, machine_pool):
    """Shortest Processing Time (SPT)."""
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: sum(get_op_processing_time(op) for op in x[1]['operations'])
    )
    jobs_order = [j for j, _ in sorted_jobs]
    return _schedule_jobs_in_order(jobs_order, unfinished_jobs, current_time, finished_events, machine_pool)


def reschedule_unfinished_jobs_lpt(unfinished_jobs, current_time, finished_events, machine_pool):
    """Longest Processing Time (LPT)."""
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: sum(get_op_processing_time(op) for op in x[1]['operations']),
        reverse=True
    )
    jobs_order = [j for j, _ in sorted_jobs]
    return _schedule_jobs_in_order(jobs_order, unfinished_jobs, current_time, finished_events, machine_pool)


def reschedule_unfinished_jobs_fcfs(unfinished_jobs, current_time, finished_events, machine_pool):
    """First Come First Served (FCFS/FIFO)."""
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: x[1]['job_ready']
    )
    jobs_order = [j for j, _ in sorted_jobs]
    return _schedule_jobs_in_order(jobs_order, unfinished_jobs, current_time, finished_events, machine_pool)


def reschedule_unfinished_jobs_cr(unfinished_jobs, current_time, finished_events, machine_pool):
    """Critical Ratio (CR)."""
    def calculate_cr(info):
        remaining_pt = sum(get_op_processing_time(op) for op in info['operations'])
        slack = info['due_date'] - info['job_ready']
        if remaining_pt <= 0:
            return float('inf')
        return slack / remaining_pt
    
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: calculate_cr(x[1])
    )
    jobs_order = [j for j, _ in sorted_jobs]
    return _schedule_jobs_in_order(jobs_order, unfinished_jobs, current_time, finished_events, machine_pool)


def reschedule_unfinished_jobs_sa(unfinished_jobs, current_time, finished_events, machine_pool, iterations=50):
    """Simulated Annealing for rescheduling."""
    jobs_order = [j for j, _ in sorted(
        unfinished_jobs.items(),
        key=lambda x: (x[1]['due_date'],
                       sum(get_op_processing_time(op) for op in x[1]['operations']))
    )]
    due_dates = {j: info['due_date'] for j, info in unfinished_jobs.items()}
    current_events = _schedule_jobs_in_order(jobs_order, unfinished_jobs, current_time, finished_events, machine_pool)
    _, _, current_cost = _compute_cost(finished_events, current_events, due_dates)
    best_events = current_events
    best_cost = current_cost
    T = 100.0
    cooling_rate = 0.95
    for _ in range(iterations):
        if len(jobs_order) >= 2:
            i, j = random.sample(range(len(jobs_order)), 2)
            neighbor_order = jobs_order[:]
            neighbor_order[i], neighbor_order[j] = neighbor_order[j], neighbor_order[i]
        else:
            neighbor_order = jobs_order[:]
        neighbor_events = _schedule_jobs_in_order(neighbor_order, unfinished_jobs, current_time, finished_events, machine_pool)
        _, _, new_cost = _compute_cost(finished_events, neighbor_events, due_dates)
        if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / max(T, 1e-6)):
            jobs_order = neighbor_order
            current_events = neighbor_events
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_events = neighbor_events
        T *= cooling_rate
    return best_events


def reschedule_unfinished_jobs_ga(unfinished_jobs, current_time, finished_events, machine_pool, 
                                  num_candidates=10, generations=5):
    """Genetic Algorithm for rescheduling."""
    base_order = [j for j, _ in sorted(
        unfinished_jobs.items(),
        key=lambda x: (x[1]['due_date'],
                       sum(get_op_processing_time(op) for op in x[1]['operations']))
    )]
    due_dates = {j: info['due_date'] for j, info in unfinished_jobs.items()}
    population = []
    for _ in range(num_candidates):
        order = base_order[:]
        random.shuffle(order)
        population.append(order)
    
    def evaluate(order):
        events = _schedule_jobs_in_order(order, unfinished_jobs, current_time, finished_events, machine_pool)
        _, _, cost = _compute_cost(finished_events, events, due_dates)
        return cost
    
    for gen in range(generations):
        population = sorted(population, key=evaluate)[:max(1, num_candidates//2)]
        new_population = []
        while len(new_population) < num_candidates:
            parent1, parent2 = random.sample(population, 2)
            cut = random.randint(1, len(parent1) - 1) if len(parent1) > 1 else 1
            head = parent1[:cut]
            tail = [j for j in parent2 if j not in head]
            child = head + tail
            if len(child) != len(parent1):
                child = parent1[:]
            # Mutation: swap two jobs
            if len(child) >= 2 and random.random() < 0.3:
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            new_population.append(child)
        population = new_population
    best_order = min(population, key=evaluate)
    return _schedule_jobs_in_order(best_order, unfinished_jobs, current_time, finished_events, machine_pool)


def reschedule_unfinished_jobs_pso(unfinished_jobs, current_time, finished_events, machine_pool, 
                                   num_particles=10, iterations=20):
    """Particle Swarm Optimization for rescheduling."""
    base_order = [j for j, _ in sorted(
        unfinished_jobs.items(),
        key=lambda x: (x[1]['due_date'],
                       sum(get_op_processing_time(op) for op in x[1]['operations']))
    )]
    due_dates = {j: info['due_date'] for j, info in unfinished_jobs.items()}

    def cost_function(order):
        events = _schedule_jobs_in_order(order, unfinished_jobs, current_time, finished_events, machine_pool)
        _, _, cost = _compute_cost(finished_events, events, due_dates)
        return cost

    particles = []
    velocities = []
    for _ in range(num_particles):
        order = base_order[:]
        random.shuffle(order)
        particles.append(order)
        velocities.append([0.0] * len(order))

    pbest = copy.deepcopy(particles)
    pbest_costs = [cost_function(p) for p in particles]
    gbest = min(particles, key=cost_function)
    gbest_cost = cost_function(gbest)

    w = 0.5
    c1 = 1.0
    c2 = 1.0

    for _ in range(iterations):
        for i in range(num_particles):
            # Discrete "velocity": swap towards pbest/gbest
            if len(particles[i]) >= 2:
                if random.random() < c1:
                    a, b = random.sample(range(len(particles[i])), 2)
                    particles[i][a], particles[i][b] = particles[i][b], particles[i][a]
                if random.random() < c2:
                    a, b = random.sample(range(len(particles[i])), 2)
                    particles[i][a], particles[i][b] = particles[i][b], particles[i][a]
            cost_candidate = cost_function(particles[i])
            if cost_candidate < pbest_costs[i]:
                pbest[i] = copy.deepcopy(particles[i])
                pbest_costs[i] = cost_candidate
        gbest = min(particles, key=cost_function)
        gbest_cost = min(gbest_cost, cost_function(gbest))
    return _schedule_jobs_in_order(gbest, unfinished_jobs, current_time, finished_events, machine_pool)
