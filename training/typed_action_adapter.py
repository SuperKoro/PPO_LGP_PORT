"""
Typed Action Adapter for PPO+LGP.
Executes ActionIndividual portfolios (DR + MH pipeline).
"""

from __future__ import annotations
from typing import Any, Dict, List

from training.portfolio_types import ActionIndividual, Gene
from registries.dispatching_registry import has_dr, get_dr
from registries.mh_registry import has_mh, get_mh
from environment.env_utils import split_schedule_list


def run_action_individual(env,
                          individual: ActionIndividual,
                          finished_events: List[Dict[str, Any]],
                          unfinished_jobs: Dict[Any, Any],
                          total_budget_s: float) -> List[Dict[str, Any]]:
    """
    Execute an ActionIndividual (portfolio) on environment:
    - Stage 0: Apply DR to reorder unfinished jobs
    - Stage 1..n: Apply MH sequentially according to weight vector
    
    Args:
        env: Scheduling environment
        individual: Portfolio to execute
        finished_events: Already finished operations
        unfinished_jobs: Jobs needing rescheduling
        total_budget_s: Total time budget in seconds
        
    Returns:
        List of rescheduled events
    """
    current_time = env.current_time
    all_jobs_info = env.all_jobs_info

    # --------- Stage 0: Dispatching Rule ----------
    dr_gene: Gene = individual.dr_gene
    dr_name = dr_gene.name.upper()

    if has_dr(dr_name):
        dr_fn = get_dr(dr_name)
    else:
        # Fallback to EDD if DR not found
        if has_dr("EDD"):
            dr_fn = get_dr("EDD")
        else:
            raise RuntimeError("No valid dispatching rule found (EDD).")

    cand_unfinished = dr_fn(
        env,
        finished_events=finished_events,
        unfinished_jobs=unfinished_jobs,
        time_budget_s=0.0
    )
    
    virtual_events = finished_events + cand_unfinished
    finished, unfinished = split_schedule_list(virtual_events, current_time, all_jobs_info)

    # --------- Stage 1..n: Metaheuristics ----------
    mh_genes: List[Gene] = individual.mh_genes

    raw_ws = [max(0.0, g.w_raw) for g in mh_genes]
    if sum(raw_ws) <= 0.0:
        raw_ws = [1.0] * len(mh_genes)
    total_w = float(sum(raw_ws))

    last_unfinished = cand_unfinished

    for gene, w_raw in zip(mh_genes, raw_ws):
        mh_name = gene.name.upper()
        if not has_mh(mh_name):
            continue

        mh_fn = get_mh(mh_name)
        stage_budget = (w_raw / total_w) * float(total_budget_s)

        if stage_budget <= 1e-9:
            continue

        cand_unfinished = mh_fn(
            env,
            finished_events=finished,
            unfinished_jobs=unfinished,
            time_budget_s=stage_budget
        )
        last_unfinished = cand_unfinished

        virtual_events = finished + cand_unfinished
        finished, unfinished = split_schedule_list(virtual_events, current_time, all_jobs_info)

    return last_unfinished
