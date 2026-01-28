"""
Extract top portfolios from training results for Phase 2.
Reads elite portfolios from all generations to create a curated action library.

Supports two extraction modes:
- fitness: Original mode, select top K by fitness only
- diverse: Stratified selection ensuring DR and MH diversity
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import Counter

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from training.portfolio_types import ActionIndividual, Gene
from config import LGPConfig


def load_all_elite_portfolios(results_dir="results"):
    """Load all elite portfolios from all generations."""
    portfolios_dir = os.path.join(results_dir, "portfolios")
    
    all_portfolios = []
    
    for filename in os.listdir(portfolios_dir):
        if filename.startswith("generation_") and filename.endswith("_final.json"):
            filepath = os.path.join(portfolios_dir, filename)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            gen = data['generation']
            
            # Read elite portfolios
            for elite in data.get('elite', []):
                all_portfolios.append({
                    'generation': gen,
                    'index': elite['index'],
                    'fitness': elite['fitness'],
                    'usage': elite['usage'],
                    'dr': elite['dr'],
                    'mh_genes': elite['mh_genes']
                })
    
    # Sort by fitness (best first)
    all_portfolios.sort(key=lambda x: x['fitness'], reverse=True)
    
    return all_portfolios


def portfolio_signature(p):
    """Create unique signature."""
    mh_parts = [f"{m['name']}:{round(m['weight'], 1)}" for m in p['mh_genes']]
    return f"{p['dr']}|{'|'.join(mh_parts)}"


def extract_top_unique(all_portfolios, top_k=10):
    """Extract top K unique portfolios (original fitness-based method)."""
    unique = []
    seen = set()
    
    for p in all_portfolios:
        sig = portfolio_signature(p)
        if sig not in seen:
            unique.append(p)
            seen.add(sig)
            if len(unique) >= top_k:
                break
    
    return unique


# =============================================================================
# NEW: Diversity-aware selection functions
# =============================================================================

def compute_diversity_score(portfolio):
    """
    Compute diversity score for a single portfolio.
    Higher score = more diverse/active genes.
    """
    score = 0.0
    mh_genes = portfolio['mh_genes']
    
    # Count unique MH types
    mh_types = set(g['name'] for g in mh_genes)
    score += len(mh_types) * 10  # Bonus for multiple MH types
    
    # Count active genes (weight > 0.1)
    active_count = sum(1 for g in mh_genes if abs(g['weight']) > 0.1)
    score += active_count * 5
    
    # Bonus for weight diversity
    weights = [abs(g['weight']) for g in mh_genes]
    if max(weights) > 0:
        weight_std = float(np.std(weights))
        score += min(weight_std, 5) * 2
    
    return score


def get_primary_mh(portfolio):
    """Get the MH with highest weight (primary MH)."""
    mh_genes = portfolio['mh_genes']
    if not mh_genes:
        return None
    return max(mh_genes, key=lambda g: abs(g['weight']))['name']


def get_mh_signature(portfolio):
    """Get a signature based on MH combination."""
    mh_names = sorted([g['name'] for g in portfolio['mh_genes']])
    return "_".join(mh_names)


def select_with_mh_diversity(portfolios, quota, available_mh):
    """
    Select portfolios ensuring MH diversity within a group.
    Prioritizes portfolios using underrepresented MHs.
    """
    if not portfolios:
        return []
    
    selected = []
    mh_counts = {mh: 0 for mh in available_mh}
    seen_sigs = set()
    
    # Round 1: Ensure each MH has representation
    for mh in available_mh:
        for p in portfolios:
            if p in selected:
                continue
            primary_mh = get_primary_mh(p)
            if primary_mh == mh:
                sig = portfolio_signature(p)
                if sig not in seen_sigs:
                    selected.append(p)
                    seen_sigs.add(sig)
                    mh_counts[mh] += 1
                    break
    
    # Round 2: Fill remaining quota with diversity preference
    # Sort remaining by diversity score
    remaining = [p for p in portfolios if p not in selected]
    remaining.sort(key=lambda p: compute_diversity_score(p), reverse=True)
    
    for p in remaining:
        if len(selected) >= quota:
            break
        sig = portfolio_signature(p)
        if sig not in seen_sigs:
            selected.append(p)
            seen_sigs.add(sig)
    
    return selected


def extract_diverse_portfolios(all_portfolios, target_size=64):
    """
    Stratified Diversity Selection Algorithm.
    
    Steps:
    1. Distribute quota across DR types
    2. Within each DR group, ensure MH diversity
    3. Prioritize portfolios with active weights
    4. Fill remaining with top fitness
    5. Generate synthetic portfolios for underrepresented types
    """
    AVAILABLE_DR = ["EDD", "SPT", "LPT", "FCFS", "CR"]
    AVAILABLE_MH = ["SA", "GA", "PSO"]
    
    # Step 1: Calculate quota per DR
    base_quota = target_size // len(AVAILABLE_DR)
    extra = target_size % len(AVAILABLE_DR)
    
    dr_quotas = {dr: base_quota for dr in AVAILABLE_DR}
    # Distribute extra slots
    for i, dr in enumerate(AVAILABLE_DR[:extra]):
        dr_quotas[dr] += 1
    
    print(f"\n📊 Quota per DR: {dr_quotas}")
    
    # Step 2: Group portfolios by DR
    by_dr = {dr: [] for dr in AVAILABLE_DR}
    for p in all_portfolios:
        if p['dr'] in by_dr:
            by_dr[p['dr']].append(p)
    
    print(f"📊 Available portfolios per DR:")
    for dr, group in by_dr.items():
        print(f"   {dr}: {len(group)}")
    
    # Step 3: Select diverse portfolios from each DR group
    selected = []
    unfilled_quota = 0
    synthetic_needed = {}  # Track how many synthetic we need per DR
    
    for dr in AVAILABLE_DR:
        quota = dr_quotas[dr]
        group = by_dr[dr]
        
        if len(group) == 0:
            print(f"   ⚠️ No portfolios for {dr}, will generate {quota} synthetic")
            synthetic_needed[dr] = quota
            continue
        
        # Sort by diversity score, then fitness
        group.sort(key=lambda p: (compute_diversity_score(p), p['fitness']), reverse=True)
        
        # Select with MH diversity
        selected_from_group = select_with_mh_diversity(group, quota, AVAILABLE_MH)
        
        actual = len(selected_from_group)
        if actual < quota:
            shortfall = quota - actual
            synthetic_needed[dr] = shortfall
            print(f"   ⚠️ {dr}: only {actual}/{quota} available, will generate {shortfall} synthetic")
        
        selected.extend(selected_from_group)
    
    print(f"\n📊 Selected from source: {len(selected)}")
    
    # Step 4: Generate synthetic portfolios for underrepresented types
    if synthetic_needed:
        print(f"📊 Generating synthetic portfolios for: {synthetic_needed}")
        synthetic = generate_synthetic_portfolios(synthetic_needed, AVAILABLE_MH)
        selected.extend(synthetic)
        print(f"📊 Added {len(synthetic)} synthetic portfolios")
    
    # Step 5: Fill remaining slots with top fitness (avoiding duplicates)
    if len(selected) < target_size:
        needed = target_size - len(selected)
        print(f"📊 Filling {needed} remaining slots with top fitness...")
        
        seen_sigs = {portfolio_signature(p) for p in selected}
        remaining = [p for p in all_portfolios if portfolio_signature(p) not in seen_sigs]
        remaining.sort(key=lambda x: x['fitness'], reverse=True)
        
        for p in remaining:
            if len(selected) >= target_size:
                break
            sig = portfolio_signature(p)
            if sig not in seen_sigs:
                selected.append(p)
                seen_sigs.add(sig)
    
    return selected[:target_size]


def generate_synthetic_portfolios(needs_by_dr, available_mh):
    """
    Generate synthetic portfolios for underrepresented DR types.
    Creates diverse MH combinations with reasonable weights.
    """
    synthetic = []
    
    # Weight templates for variety
    weight_templates = [
        [5.0, 10.0, 15.0],    # All active
        [10.0, 5.0, 0.0],     # Two active
        [15.0, 0.0, 0.0],     # One strong
        [8.0, 8.0, 8.0],      # Balanced
        [20.0, 5.0, 2.0],     # Dominant first
        [0.0, 15.0, 10.0],    # Dominant middle
        [0.0, 0.0, 20.0],     # Dominant last
        [12.0, 6.0, 3.0],     # Decreasing
        [3.0, 6.0, 12.0],     # Increasing
    ]
    
    # MH combinations for variety
    mh_combinations = [
        ["SA", "GA", "PSO"],
        ["GA", "PSO", "SA"],
        ["PSO", "SA", "GA"],
        ["SA", "SA", "GA"],
        ["GA", "GA", "PSO"],
        ["PSO", "PSO", "SA"],
        ["SA", "GA", "GA"],
        ["GA", "PSO", "PSO"],
        ["SA", "PSO", "SA"],
    ]
    
    idx = 0
    for dr, count in needs_by_dr.items():
        for i in range(count):
            # Cycle through combinations
            weights = weight_templates[idx % len(weight_templates)]
            mh_combo = mh_combinations[idx % len(mh_combinations)]
            
            portfolio = {
                'generation': 0,  # Mark as synthetic
                'index': -1,      # Mark as synthetic
                'fitness': 0.0,   # Neutral fitness (will be evaluated)
                'usage': 0,
                'dr': dr,
                'mh_genes': [
                    {'name': mh_combo[j], 'weight': weights[j]}
                    for j in range(len(mh_combo))
                ]
            }
            synthetic.append(portfolio)
            idx += 1
    
    return synthetic


def print_diversity_report(portfolios):
    """Print a summary of portfolio diversity."""
    dr_counts = Counter(p['dr'] for p in portfolios)
    mh_counts = Counter()
    active_mh_counts = Counter()
    
    for p in portfolios:
        for gene in p['mh_genes']:
            mh_counts[gene['name']] += 1
            if abs(gene['weight']) > 0.1:
                active_mh_counts[gene['name']] += 1
    
    total = len(portfolios)
    total_genes = sum(len(p['mh_genes']) for p in portfolios)
    active_genes = sum(active_mh_counts.values())
    
    print("\n" + "=" * 60)
    print("📊 DIVERSITY REPORT")
    print("=" * 60)
    
    print("\nDispatching Rules:")
    for dr in ["EDD", "SPT", "LPT", "FCFS", "CR"]:
        count = dr_counts.get(dr, 0)
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {dr:6s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print("\nMetaheuristics (all genes):")
    for mh in ["SA", "GA", "PSO"]:
        count = mh_counts.get(mh, 0)
        pct = count / total_genes * 100 if total_genes > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {mh:4s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print(f"\nActive genes: {active_genes}/{total_genes} ({active_genes/total_genes*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Extract top portfolios for Phase 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_top_portfolios.py                    # Use diverse mode (default)
  python extract_top_portfolios.py --mode fitness     # Original fitness-only mode
  python extract_top_portfolios.py --mode diverse     # Stratified diversity mode
        """
    )
    parser.add_argument('--mode', choices=['fitness', 'diverse'], default='diverse',
                       help='Selection mode: fitness (original) or diverse (recommended)')
    parser.add_argument('--size', type=int, default=None,
                       help=f'Target pool size (default: {LGPConfig.pool_size})')
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXTRACTING TOP PORTFOLIOS FOR PHASE 2")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    
    all_portfolios = load_all_elite_portfolios()
    print(f"\n✓ Loaded {len(all_portfolios)} elite portfolios")
    
    top_k = args.size if args.size else LGPConfig.pool_size
    print(f"✓ Target size: {top_k}")
    
    if args.mode == 'diverse':
        top_portfolios = extract_diverse_portfolios(all_portfolios, target_size=top_k)
    else:
        top_portfolios = extract_top_unique(all_portfolios, top_k=top_k)
    
    print(f"\n✓ Selected {len(top_portfolios)} portfolios")
    
    # Print diversity report
    print_diversity_report(top_portfolios)
    
    # Print top portfolios
    print("\n" + "=" * 60)
    print("TOP PORTFOLIOS (first 10)")
    print("=" * 60)
    
    for i, p in enumerate(top_portfolios[:10], 1):
        print(f"\n#{i} | Fitness: {p['fitness']:.2f} | Gen {p['generation']} | Usage: {p['usage']}")
        print(f"   DR: {p['dr']}")
        for j, mh in enumerate(p['mh_genes'], 1):
            status = "✓" if abs(mh['weight']) > 0.1 else "○"
            print(f"   MH{j}: {mh['name']} (weight={mh['weight']:.2f}) {status}")
    
    # Save
    output_file = "results/top_portfolios_phase2.json"
    with open(output_file, 'w') as f:
        json.dump({'portfolios': top_portfolios}, f, indent=2)
    
    print(f"\n✓ Saved to: {output_file}")
    print("\n✅ DONE - Use these portfolios for Phase 2!")


if __name__ == "__main__":
    main()
