# dataset_loader.py
"""
Dataset loader for Dynamic Job Shop Scheduling Environment.
Loads job data and due dates from JSON files in the data/ directory.
"""

import json
import os
from typing import Dict, Any, List, Tuple, Optional


def load_dataset(dataset_name: Optional[str] = None) -> Tuple[Dict, Dict, List]:
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
        return _get_default_dataset()
    
    # Construct file path
    if not dataset_name.endswith('.json'):
        dataset_name = f"{dataset_name}.json"
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    file_path = os.path.join(data_dir, dataset_name)
    
    # Try to load from file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded dataset: {data.get('name', dataset_name)}")
        
        # Parse jobs (convert string keys to int)
        jobs = {}
        for job_id_str, operations in data['jobs'].items():
            job_id = int(job_id_str)
            jobs[job_id] = operations
        
        # Validate: check for null processing times
        null_count = sum(1 for job_ops in jobs.values() 
                        for op in job_ops if op['processing_time'] is None)
        if null_count > 0:
            print(f"⚠️  Warning: {null_count} operations have null processing_time in {dataset_name}")
            print(f"   This should not happen with proper datasets from Excel!")
        
        # Parse due dates (convert string keys to int)
        due_dates = {}
        for job_id_str, due_date in data['due_dates'].items():
            job_id = int(job_id_str)
            due_dates[job_id] = due_date
        
        # Get machine pool
        machine_pool = data.get('machine_pool', [1, 2, 3, 4, 5, 6, 7, 8, 12, 13])
        
        return jobs, due_dates, machine_pool
    
    except FileNotFoundError:
        print(f"⚠️ Dataset file not found: {file_path}")
        print(f"   Falling back to hardcoded default dataset")
        return _get_default_dataset()
    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"   Falling back to hardcoded default dataset")
        return _get_default_dataset()


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
    
    print("✅ Using hardcoded default dataset (20 jobs)")
    
    return jobs_initial, due_dates_initial, machine_pool


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
        print("\n📚 Available Datasets:")
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
    
    print(f"\n📊 Dataset Info: {dataset_name}")
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
