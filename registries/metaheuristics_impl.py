# metaheuristics_impl.py
"""
Proper Metaheuristic implementations for Dynamic Job Shop Scheduling.
These implementations actually reschedule jobs by:
1. Reordering job sequence
2. Swapping machine assignments
3. Rebuilding the schedule from scratch
"""

from typing import Any, Dict, List
import copy
import random
import math
from registries.mh_registry import register_mh


def _build_schedule_from_order(
    job_order: List[Any],
    unfinished_jobs: Dict[Any, Any],
    current_time: float,
    machine_pool: List[int]
) -> List[Dict[str, Any]]:
    """
    Build a valid schedule from a given job ordering.
    This is the core function that constructs a feasible schedule.
    
    Args:
        job_order: List of job IDs in the order they should be scheduled
        unfinished_jobs: Dictionary of unfinished jobs with their operations
        current_time: Current simulation time
        machine_pool: List of available machines
        
    Returns:
        List of scheduled events
    """
    machine_ready = {m: current_time for m in machine_pool}
    job_ready = {}
    
    # Initialize job ready times
    for job in job_order:
        if job in unfinished_jobs:
            job_ready[job] = unfinished_jobs[job]['job_ready']
    
    new_events = []
    
    for job in job_order:
        if job not in unfinished_jobs:
            continue
            
        info = unfinished_jobs[job]
        
        for op in sorted(info['operations'], key=lambda x: x['op_index']):
            pt = op['processing_time']
            best_start = float('inf')
            best_finish = float('inf')
            best_machine = None
            
            for m in op['candidate_machines']:
                if m not in machine_ready:
                    machine_ready[m] = current_time
                st = max(job_ready.get(job, current_time), machine_ready[m])
                ft = st + pt
                if ft < best_finish:
                    best_finish = ft
                    best_start = st
                    best_machine = m
            
            if best_machine is None:
                # Fallback: use first candidate machine
                m = op['candidate_machines'][0]
                if m not in machine_ready:
                    machine_ready[m] = current_time
                best_start = max(job_ready.get(job, current_time), machine_ready[m])
                best_finish = best_start + pt
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
            job_ready[job] = best_finish
            machine_ready[best_machine] = best_finish
    
    return new_events


def _calculate_makespan(events: List[Dict[str, Any]], finished_events: List[Dict[str, Any]]) -> float:
    """Calculate makespan of a schedule."""
    all_events = finished_events + events
    if not all_events:
        return 0.0
    return max(e['finish'] for e in all_events)


def _swap_jobs(order: List[Any]) -> List[Any]:
    """Swap two random jobs in the order."""
    if len(order) < 2:
        return order
    new_order = order.copy()
    i, j = random.sample(range(len(new_order)), 2)
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order


def _insert_job(order: List[Any]) -> List[Any]:
    """Remove a job and insert it at a random position."""
    if len(order) < 2:
        return order
    new_order = order.copy()
    i = random.randint(0, len(new_order) - 1)
    job = new_order.pop(i)
    j = random.randint(0, len(new_order))
    new_order.insert(j, job)
    return new_order


@register_mh("SA")
def mh_sa(env,
          finished_events: List[Dict[str, Any]],
          unfinished_jobs: Dict[Any, Any],
          time_budget_s: float):
    """
    Simulated Annealing for job shop rescheduling.
    
    Neighbor generation:
    - Swap two jobs in the sequence
    - Insert a job at a different position
    - (Machine assignment is handled optimally in schedule building)
    """
    current_time = env.current_time
    machine_pool = env.machine_pool
    
    if not unfinished_jobs:
        return []
    
    # Initial solution: EDD ordering
    job_list = list(unfinished_jobs.keys())
    current_order = sorted(job_list, key=lambda j: (
        unfinished_jobs[j]['due_date'],
        sum(op['processing_time'] for op in unfinished_jobs[j]['operations'])
    ))
    
    current_schedule = _build_schedule_from_order(current_order, unfinished_jobs, current_time, machine_pool)
    current_cost = _calculate_makespan(current_schedule, finished_events)
    
    best_order = current_order.copy()
    best_schedule = current_schedule
    best_cost = current_cost
    
    # SA parameters
    iterations = max(30, int(time_budget_s * 16.7))  # ~50 iterations for 3s budget
    T = 100.0  # Initial temperature
    cooling_rate = 0.95
    
    for i in range(iterations):
        # Generate neighbor (50% swap, 50% insert)
        if random.random() < 0.5:
            neighbor_order = _swap_jobs(current_order)
        else:
            neighbor_order = _insert_job(current_order)
        
        neighbor_schedule = _build_schedule_from_order(neighbor_order, unfinished_jobs, current_time, machine_pool)
        neighbor_cost = _calculate_makespan(neighbor_schedule, finished_events)
        
        # Acceptance criterion
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_order = neighbor_order
            current_schedule = neighbor_schedule
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_order = current_order.copy()
                best_schedule = current_schedule
                best_cost = current_cost
        
        T *= cooling_rate
    
    return best_schedule


@register_mh("GA")
def mh_ga(env,
          finished_events: List[Dict[str, Any]],
          unfinished_jobs: Dict[Any, Any],
          time_budget_s: float):
    """
    Genetic Algorithm for job shop rescheduling.
    
    Chromosome: Job ordering (permutation)
    Crossover: Order crossover (OX)
    Mutation: Swap mutation
    """
    current_time = env.current_time
    machine_pool = env.machine_pool
    
    if not unfinished_jobs:
        return []
    
    job_list = list(unfinished_jobs.keys())
    n_jobs = len(job_list)
    
    if n_jobs <= 1:
        schedule = _build_schedule_from_order(job_list, unfinished_jobs, current_time, machine_pool)
        return schedule
    
    # GA parameters
    pop_size = max(5, int(time_budget_s * 5.0))  # ~15 for 3s budget
    generations = max(3, int(time_budget_s * 3.0))  # ~9 for 3s budget
    mutation_rate = 0.3
    elite_size = 2
    
    # Initialize population with different orderings
    population = []
    
    # EDD ordering
    edd_order = sorted(job_list, key=lambda j: unfinished_jobs[j]['due_date'])
    population.append(edd_order)
    
    # SPT ordering
    spt_order = sorted(job_list, key=lambda j: sum(op['processing_time'] for op in unfinished_jobs[j]['operations']))
    population.append(spt_order)
    
    # Random orderings
    while len(population) < pop_size:
        order = job_list.copy()
        random.shuffle(order)
        population.append(order)
    
    def evaluate(order):
        schedule = _build_schedule_from_order(order, unfinished_jobs, current_time, machine_pool)
        return _calculate_makespan(schedule, finished_events)
    
    def order_crossover(p1, p2):
        """Order Crossover (OX) operator."""
        size = len(p1)
        if size <= 2:
            return p1.copy()
        
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end+1] = p1[start:end+1]
        
        pos = (end + 1) % size
        for gene in p2:
            if gene not in child:
                child[pos] = gene
                pos = (pos + 1) % size
        
        return child
    
    for gen in range(generations):
        # Evaluate
        fitness = [(order, evaluate(order)) for order in population]
        fitness.sort(key=lambda x: x[1])
        
        # Selection (elitism + tournament)
        new_population = [f[0] for f in fitness[:elite_size]]
        
        while len(new_population) < pop_size:
            # Tournament selection
            candidates = random.sample(fitness, min(3, len(fitness)))
            parent1 = min(candidates, key=lambda x: x[1])[0]
            
            candidates = random.sample(fitness, min(3, len(fitness)))
            parent2 = min(candidates, key=lambda x: x[1])[0]
            
            # Crossover
            child = order_crossover(parent1, parent2)
            
            # Mutation
            if random.random() < mutation_rate:
                child = _swap_jobs(child)
            
            new_population.append(child)
        
        population = new_population
    
    # Return best solution
    best_order = min(population, key=evaluate)
    return _build_schedule_from_order(best_order, unfinished_jobs, current_time, machine_pool)


@register_mh("PSO")
def mh_pso(env,
           finished_events: List[Dict[str, Any]],
           unfinished_jobs: Dict[Any, Any],
           time_budget_s: float):
    """
    Particle Swarm Optimization for job shop rescheduling.
    
    For combinatorial optimization, we use a discrete PSO variant
    where particles are job orderings and velocity is applied as swap operations.
    """
    current_time = env.current_time
    machine_pool = env.machine_pool
    
    if not unfinished_jobs:
        return []
    
    job_list = list(unfinished_jobs.keys())
    n_jobs = len(job_list)
    
    if n_jobs <= 1:
        schedule = _build_schedule_from_order(job_list, unfinished_jobs, current_time, machine_pool)
        return schedule
    
    # PSO parameters
    num_particles = max(5, int(time_budget_s * 3.3))  # ~10 for 3s budget
    iterations = max(10, int(time_budget_s * 6.7))     # ~20 for 3s budget
    
    def evaluate(order):
        schedule = _build_schedule_from_order(order, unfinished_jobs, current_time, machine_pool)
        return _calculate_makespan(schedule, finished_events)
    
    # Initialize particles
    particles = []
    
    # EDD and SPT as initial particles
    edd_order = sorted(job_list, key=lambda j: unfinished_jobs[j]['due_date'])
    spt_order = sorted(job_list, key=lambda j: sum(op['processing_time'] for op in unfinished_jobs[j]['operations']))
    
    particles.append(edd_order.copy())
    particles.append(spt_order.copy())
    
    while len(particles) < num_particles:
        order = job_list.copy()
        random.shuffle(order)
        particles.append(order)
    
    # Personal best
    pbest = [p.copy() for p in particles]
    pbest_costs = [evaluate(p) for p in pbest]
    
    # Global best
    gbest_idx = pbest_costs.index(min(pbest_costs))
    gbest = pbest[gbest_idx].copy()
    gbest_cost = pbest_costs[gbest_idx]
    
    # PSO coefficients
    w = 0.4   # inertia (probability of keeping current move)
    c1 = 0.3  # cognitive (probability of moving toward pbest)
    c2 = 0.3  # social (probability of moving toward gbest)
    
    for iteration in range(iterations):
        for i in range(num_particles):
            # Apply "velocity" as swap operations
            new_order = particles[i].copy()
            
            # Random move (inertia)
            if random.random() < w:
                new_order = _swap_jobs(new_order)
            
            # Move toward personal best
            if random.random() < c1:
                # Find positions that differ from pbest and try to fix one
                for j in range(len(new_order)):
                    if new_order[j] != pbest[i][j]:
                        # Find where pbest[i][j] is in new_order
                        k = new_order.index(pbest[i][j])
                        new_order[j], new_order[k] = new_order[k], new_order[j]
                        break
            
            # Move toward global best
            if random.random() < c2:
                for j in range(len(new_order)):
                    if new_order[j] != gbest[j]:
                        k = new_order.index(gbest[j])
                        new_order[j], new_order[k] = new_order[k], new_order[j]
                        break
            
            particles[i] = new_order
            cost = evaluate(new_order)
            
            # Update personal best
            if cost < pbest_costs[i]:
                pbest[i] = new_order.copy()
                pbest_costs[i] = cost
                
                # Update global best
                if cost < gbest_cost:
                    gbest = new_order.copy()
                    gbest_cost = cost
    
    return _build_schedule_from_order(gbest, unfinished_jobs, current_time, machine_pool)
