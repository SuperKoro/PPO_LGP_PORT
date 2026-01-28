"""
Analyze portfolio diversity for Phase 2 training.
Reports distribution of Dispatching Rules (DR) and Metaheuristics (MH).
"""

import os
import sys
import json
import argparse
from collections import Counter

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def load_portfolios(filepath):
    """Load portfolios from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['portfolios']


def analyze_dr_distribution(portfolios):
    """Analyze distribution of Dispatching Rules."""
    dr_counts = Counter(p['dr'] for p in portfolios)
    total = len(portfolios)
    
    print("\n" + "=" * 50)
    print("📊 DISPATCHING RULE (DR) DISTRIBUTION")
    print("=" * 50)
    
    for dr, count in sorted(dr_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {dr:6s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    return dr_counts


def analyze_mh_distribution(portfolios):
    """Analyze distribution of Metaheuristics across all genes."""
    mh_counts = Counter()
    active_mh_counts = Counter()  # weight > 0.1
    total_genes = 0
    active_genes = 0
    
    for p in portfolios:
        for gene in p['mh_genes']:
            mh_name = gene['name']
            weight = abs(gene['weight'])
            
            mh_counts[mh_name] += 1
            total_genes += 1
            
            if weight > 0.1:
                active_mh_counts[mh_name] += 1
                active_genes += 1
    
    print("\n" + "=" * 50)
    print("📊 METAHEURISTIC (MH) DISTRIBUTION (All Genes)")
    print("=" * 50)
    
    for mh, count in sorted(mh_counts.items(), key=lambda x: -x[1]):
        pct = count / total_genes * 100
        bar = "█" * int(pct / 2)
        print(f"  {mh:4s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print(f"\n  Total genes: {total_genes}")
    print(f"  Active genes (weight > 0.1): {active_genes} ({active_genes/total_genes*100:.1f}%)")
    
    print("\n" + "-" * 50)
    print("📊 ACTIVE MH DISTRIBUTION (weight > 0.1)")
    print("-" * 50)
    
    if active_genes > 0:
        for mh, count in sorted(active_mh_counts.items(), key=lambda x: -x[1]):
            pct = count / active_genes * 100
            bar = "█" * int(pct / 2)
            print(f"  {mh:4s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    return mh_counts, active_mh_counts


def analyze_weight_distribution(portfolios):
    """Analyze distribution of weights."""
    weights = []
    zero_count = 0
    
    for p in portfolios:
        for gene in p['mh_genes']:
            w = abs(gene['weight'])
            weights.append(w)
            if w < 0.1:
                zero_count += 1
    
    print("\n" + "=" * 50)
    print("📊 WEIGHT DISTRIBUTION")
    print("=" * 50)
    
    total = len(weights)
    print(f"  Total weights: {total}")
    print(f"  Zero/near-zero (< 0.1): {zero_count} ({zero_count/total*100:.1f}%)")
    print(f"  Active (>= 0.1): {total - zero_count} ({(total-zero_count)/total*100:.1f}%)")
    
    if weights:
        print(f"\n  Min: {min(weights):.2f}")
        print(f"  Max: {max(weights):.2f}")
        print(f"  Mean: {sum(weights)/len(weights):.2f}")
    
    # Histogram buckets
    buckets = [0, 0.1, 1, 5, 10, 20, float('inf')]
    bucket_counts = [0] * (len(buckets) - 1)
    
    for w in weights:
        for i in range(len(buckets) - 1):
            if buckets[i] <= w < buckets[i + 1]:
                bucket_counts[i] += 1
                break
    
    print("\n  Weight ranges:")
    labels = ["[0, 0.1)", "[0.1, 1)", "[1, 5)", "[5, 10)", "[10, 20)", "[20+)"]
    for label, count in zip(labels, bucket_counts):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:12s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    return weights


def compute_diversity_score(portfolios):
    """Compute overall diversity score."""
    # DR diversity (ideal: equal distribution)
    dr_counts = Counter(p['dr'] for p in portfolios)
    n_dr_types = len(dr_counts)
    dr_balance = min(dr_counts.values()) / max(dr_counts.values()) if dr_counts else 0
    
    # MH diversity
    mh_counts = Counter()
    for p in portfolios:
        for gene in p['mh_genes']:
            if abs(gene['weight']) > 0.1:
                mh_counts[gene['name']] += 1
    n_mh_types = len(mh_counts)
    mh_balance = min(mh_counts.values()) / max(mh_counts.values()) if mh_counts else 0
    
    # Active genes ratio
    total_genes = sum(len(p['mh_genes']) for p in portfolios)
    active_genes = sum(
        1 for p in portfolios 
        for gene in p['mh_genes'] 
        if abs(gene['weight']) > 0.1
    )
    active_ratio = active_genes / total_genes if total_genes > 0 else 0
    
    # Combined score (0-100)
    score = (
        n_dr_types * 10 +      # Max 50 (5 types)
        dr_balance * 20 +       # Max 20
        n_mh_types * 5 +        # Max 15 (3 types)
        mh_balance * 15 +       # Max 15
        active_ratio * 20       # Max 20 (based on active genes)
    )
    
    print("\n" + "=" * 50)
    print("📊 DIVERSITY SCORE")
    print("=" * 50)
    print(f"  DR types used: {n_dr_types}/5")
    print(f"  DR balance (min/max): {dr_balance:.2f}")
    print(f"  MH types used: {n_mh_types}/3")
    print(f"  MH balance (min/max): {mh_balance:.2f}")
    print(f"  Active gene ratio: {active_ratio:.2f}")
    print(f"\n  ⭐ OVERALL DIVERSITY SCORE: {score:.1f}/100")
    
    if score < 50:
        print("  ⚠️  LOW DIVERSITY - Consider using --mode diverse in extract_top_portfolios.py")
    elif score < 70:
        print("  ⚡ MODERATE DIVERSITY")
    else:
        print("  ✅ GOOD DIVERSITY")
    
    return score


def main():
    parser = argparse.ArgumentParser(description='Analyze portfolio diversity')
    parser.add_argument('filepath', nargs='?', 
                       default='results/top_portfolios_phase2.json',
                       help='Path to portfolios JSON file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("📈 PORTFOLIO DIVERSITY ANALYSIS")
    print("=" * 60)
    print(f"File: {args.filepath}")
    
    portfolios = load_portfolios(args.filepath)
    print(f"Total portfolios: {len(portfolios)}")
    
    analyze_dr_distribution(portfolios)
    analyze_mh_distribution(portfolios)
    analyze_weight_distribution(portfolios)
    compute_diversity_score(portfolios)
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
