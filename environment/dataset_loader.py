# dataset_loader.py
"""
Dataset loader for Dynamic Job Shop Scheduling Environment.
Loads job data and due dates from JSON files in the data/ directory.
"""

import json
import os
import re
import random
from typing import Dict, Any, List, Tuple, Optional

import numpy as np


def load_dataset(dataset_name: Optional[str] = None, return_meta: bool = False):
    """
    Load dataset from JSON file in data/ directory.
    
    Args:
        dataset_name: Name of dataset file (e.g., "Set20", "Set25", etc.)
                     If None, returns hardcoded default data
    
    Returns:
        Tuple of (jobs_dict, due_dates_dict, machine_pool)
        
    Example:
        jobs, due_dates, machines = load_dataset("Set20")
    """
    
    # If no dataset specified, return hardcoded default
    if dataset_name is None:
        jobs, due_dates, machine_pool = _get_default_dataset()
        if return_meta:
            return jobs, due_dates, machine_pool, None, None, None
        return jobs, due_dates, machine_pool
    
    # Construct file path
    raw_name = dataset_name
    if not dataset_name.endswith('.json'):
        dataset_file = f"{dataset_name}.json"
    else:
        dataset_file = dataset_name
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    file_path = os.path.join(data_dir, dataset_file)
    
    # Try to load from file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded dataset: {data.get('name', dataset_name)}")
        
        # Parse jobs (convert string keys to int, normalize processing times)
        jobs = {}
        for job_id_str, operations in data['jobs'].items():
            job_id = int(job_id_str)
            norm_ops = []
            for op in operations:
                op_copy = dict(op)
                if isinstance(op_copy.get('candidate_machines'), list):
                    op_copy['candidate_machines'] = [int(m) for m in op_copy['candidate_machines']]
                pt_map = op_copy.get('processing_times')
                if isinstance(pt_map, dict):
                    normalized_map = {}
                    for k, v in pt_map.items():
                        try:
                            mk = int(k)
                        except Exception:
                            mk = k
                        normalized_map[mk] = float(v)
                    op_copy['processing_times'] = normalized_map
                    if op_copy.get('processing_time') is None:
                        op_copy['processing_time'] = (
                            sum(normalized_map.values()) / len(normalized_map) if normalized_map else 0.0
                        )
                if 'processing_time' in op_copy and op_copy['processing_time'] is not None:
                    op_copy['processing_time'] = float(op_copy['processing_time'])
                norm_ops.append(op_copy)
            jobs[job_id] = norm_ops
        
        # Validate: check for null processing times
        null_count = sum(1 for job_ops in jobs.values()
                        for op in job_ops if op.get('processing_time') is None)
        if null_count > 0:
            print(f"Warning: {null_count} operations have null processing_time in {dataset_name}")
            print(f"   This should not happen with proper datasets from Excel!")
        
        # Parse due dates (convert string keys to int)
        due_dates = {}
        for job_id_str, due_date in data['due_dates'].items():
            job_id = int(job_id_str)
            due_dates[job_id] = due_date

        # Optional arrival times (for predefined dynamic jobs)
        arrival_times = None
        if isinstance(data.get('arrival_times'), dict):
            arrival_times = {}
            for job_id_str, at in data['arrival_times'].items():
                arrival_times[int(job_id_str)] = int(at)

        # Optional initial job ids
        initial_job_ids = None
        if isinstance(data.get('initial_job_ids'), list):
            initial_job_ids = [int(j) for j in data['initial_job_ids']]

        # Optional job types (urgent/normal)
        job_types = None
        if isinstance(data.get('job_types'), dict):
            job_types = {}
            for job_id_str, jt in data['job_types'].items():
                job_types[int(job_id_str)] = str(jt)
        
        # Get machine pool
        machine_pool = data.get('machine_pool', [1, 2, 3, 4, 5, 6, 7, 8, 12, 13])
        
        if return_meta:
            return jobs, due_dates, machine_pool, arrival_times, initial_job_ids, job_types
        return jobs, due_dates, machine_pool
    
    except FileNotFoundError:
        print(f"Dataset file not found: {file_path}")
        # If dataset name follows Ref_M{M}_E{E}_N{N}, generate on the fly
        ref_name = raw_name or dataset_name or ""
        if ref_name.endswith(".json"):
            ref_name = ref_name[:-5]
        ref_match = re.match(r"^Ref_M(\d+)_E(\d+)_N(\d+)$", ref_name)
        if ref_match:
            M_num = int(ref_match.group(1))
            E_ave = int(ref_match.group(2))
            New_insert = int(ref_match.group(3))
            jobs, due_dates, machine_pool, arrival_times, initial_job_ids, job_types = _generate_ref_dataset(
                M_num=M_num,
                E_ave=E_ave,
                New_insert=New_insert,
                seed=42,
            )
            if return_meta:
                return jobs, due_dates, machine_pool, arrival_times, initial_job_ids, job_types
            return jobs, due_dates, machine_pool

        print("   Falling back to hardcoded default dataset")
        jobs, due_dates, machine_pool = _get_default_dataset()
        if return_meta:
            return jobs, due_dates, machine_pool, None, None, None
        return jobs, due_dates, machine_pool
    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("   Falling back to hardcoded default dataset")
        jobs, due_dates, machine_pool = _get_default_dataset()
        if return_meta:
            return jobs, due_dates, machine_pool, None, None, None
        return jobs, due_dates, machine_pool


def _get_default_dataset() -> Tuple[Dict, Dict, List]:
    """
    Return the hardcoded default dataset (original 20 jobs).
    This is the same data that was previously hardcoded in scheduling_env.py
    """
    
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
    
    due_dates_initial = {i: 1200 for i in range(1, 51)}
    
    machine_pool = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
    
    print("Using hardcoded default dataset (20 jobs)")
    
    return jobs_initial, due_dates_initial, machine_pool


def _generate_ref_dataset(M_num: int, E_ave: int, New_insert: int, seed: int = 42, Initial_Job_num: int = 5):
    """
    Generate dataset using Ref/code/DQN.py::Instance_Generator logic.
    Returns (jobs, due_dates, machine_pool, arrival_times, initial_job_ids, job_types).

    Note: We store per-op processing_time as the average across candidate machines,
    and keep the full per-machine map in 'processing_times' for reference.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # operations per job
    Op_num = [rng.randint(1, 20) for _ in range(New_insert + Initial_Job_num)]

    jobs: Dict[int, List[Dict[str, Any]]] = {}
    processing_time_full: Dict[int, List[Dict[int, int]]] = {}

    for job_id in range(Initial_Job_num + New_insert):
        ops = []
        ops_full = []
        for op_index in range(Op_num[job_id]):
            k = rng.randint(1, M_num - 2)
            machines = list(range(M_num))
            rng.shuffle(machines)
            machines = machines[:k + 1]
            pt_map: Dict[int, int] = {}
            for m in machines:
                pt_map[m] = rng.randint(1, 50)
            avg_pt = sum(pt_map.values()) / len(pt_map)
            ops.append({
                "op_id": op_index + 1,
                "candidate_machines": machines,
                "processing_time": float(avg_pt),
                "processing_times": {m: int(pt) for m, pt in pt_map.items()},
            })
            ops_full.append(pt_map)
        jobs[job_id] = ops
        processing_time_full[job_id] = ops_full

    # Arrival times
    A1 = [0 for _ in range(Initial_Job_num)]
    A = np.random.exponential(E_ave, size=New_insert)
    A = [int(a) for a in A]
    A1.extend(A)

    # Emergency levels
    EL = [rng.randint(1, 3) for _ in range(len(A1))]

    # Average processing time per job
    T_ijave = []
    for job_id in range(Initial_Job_num + New_insert):
        Tad = []
        for op_map in processing_time_full[job_id]:
            Tad.append(sum(op_map.values()) / len(op_map))
        T_ijave.append(sum(Tad))

    # Due dates
    D1 = [int((0.2 + 0.5 * EL[i]) * T_ijave[i]) for i in range(Initial_Job_num)]
    D = [
        int(A1[i] + (0.2 + 0.5 * EL[i]) * T_ijave[i])
        for i in range(Initial_Job_num, Initial_Job_num + New_insert)
    ]
    D1.extend(D)

    arrival_times = {i: int(A1[i]) for i in range(len(A1))}
    due_dates = {i: int(D1[i]) for i in range(len(D1))}
    job_types = {i: ("Urgent" if EL[i] >= 3 else "Normal") for i in range(len(EL))}
    initial_job_ids = list(range(Initial_Job_num))
    machine_pool = list(range(M_num))

    return jobs, due_dates, machine_pool, arrival_times, initial_job_ids, job_types


def list_available_datasets() -> List[str]:
    """
    List all available datasets in the data/ directory.
    
    Returns:
        List of dataset names (without .json extension)
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    if not os.path.exists(data_dir):
        return []
    
    datasets = []
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            datasets.append(file.replace('.json', ''))
    
    return sorted(datasets)


def print_dataset_info(dataset_name: Optional[str] = None):
    """
    Print information about a dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., "Set20")
                     If None, shows available datasets
    """
    if dataset_name is None:
        available = list_available_datasets()
        print("\nAvailable Datasets:")
        print("=" * 50)
        if available:
            for ds in available:
                print(f"  - {ds}")
        else:
            print("  No datasets found in data/ directory")
        print("=" * 50)
        return
    
    # Load and show info
    jobs, due_dates, machine_pool = load_dataset(dataset_name)
    
    print(f"\nDataset Info: {dataset_name}")
    print("=" * 50)
    print(f"  Total Jobs:        {len(jobs)}")
    print(f"  Total Machines:    {len(machine_pool)}")
    print(f"  Machine IDs:       {machine_pool}")
    
    # Calculate total operations
    total_ops = sum(len(ops) for ops in jobs.values())
    print(f"  Total Operations:  {total_ops}")
    print(f"  Avg Ops/Job:       {total_ops / len(jobs):.2f}")
    
    # Due date info
    unique_due_dates = set(due_dates.values())
    print(f"  Unique Due Dates:  {len(unique_due_dates)}")
    print("=" * 50)


# ============================================================================
# 🧪 TEST FUNCTIONALITY
# ============================================================================

if __name__ == "__main__":
    # Show available datasets
    print_dataset_info()
    
    # Test loading each dataset
    for dataset in list_available_datasets():
        print(f"\n🔍 Testing {dataset}...")
        jobs, due_dates, machines = load_dataset(dataset)
        print(f"   ✅ Loaded {len(jobs)} jobs, {len(machines)} machines")
