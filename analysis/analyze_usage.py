"""
Analyze program usage distribution across generations.
Helps diagnose PPO policy collapse and exploration issues.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_portfolio_data(results_dir="results"):
    """Load all portfolio data from JSON files"""
    portfolios_dir = os.path.join(results_dir, "portfolios")
    
    gen_files = sorted([f for f in os.listdir(portfolios_dir) 
                       if f.startswith("generation_") and f.endswith("_final.json")])
    
    generations_data = []
    for gen_file in gen_files:
        filepath = os.path.join(portfolios_dir, gen_file)
        with open(filepath, 'r') as f:
            data = json.load(f)
        generations_data.append(data)
    
    return generations_data


def analyze_usage_distribution(generations_data):
    """Analyze usage distribution across generations"""
    print("\n" + "="*70)
    print("📊 PROGRAM USAGE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    for gen_data in generations_data:
        gen_num = gen_data['generation']
        all_usage = np.array(gen_data['all_usage'])
        
        total_usage = all_usage.sum()
        nonzero_programs = (all_usage > 0).sum()
        max_usage = all_usage.max()
        max_idx = all_usage.argmax()
        
        # Concentration metrics
        top1_pct = (max_usage / total_usage * 100) if total_usage > 0 else 0
        top5_usage = np.sort(all_usage)[-5:].sum()
        top5_pct = (top5_usage / total_usage * 100) if total_usage > 0 else 0
        
        # Gini coefficient (measure of inequality)
        sorted_usage = np.sort(all_usage)
        n = len(sorted_usage)
        cumsum = np.cumsum(sorted_usage)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_usage)) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] > 0 else 0
        
        print(f"\n🔍 Generation {gen_num}:")
        print(f"  Total usages: {total_usage}")
        print(f"  Programs used: {nonzero_programs}/64 ({nonzero_programs/64*100:.1f}%)")
        print(f"  Top program: #{max_idx} with {max_usage} usages ({top1_pct:.1f}%)")
        print(f"  Top 5 programs: {top5_pct:.1f}% of total usage")
        print(f"  Gini coefficient: {gini:.3f} (0=equal, 1=concentrated)")
        
        if top1_pct > 80:
            print(f"  ⚠️  WARNING: Severe concentration! Top program = {top1_pct:.1f}%")
        elif top1_pct > 50:
            print(f"  ⚠️  CAUTION: High concentration! Top program = {top1_pct:.1f}%")


def plot_usage_heatmap(generations_data, output_dir="results/plots"):
    """Plot heatmap of program usage across generations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    num_gens = len(generations_data)
    num_programs = 64
    usage_matrix = np.zeros((num_gens, num_programs))
    
    for i, gen_data in enumerate(generations_data):
        usage_matrix[i, :] = gen_data['all_usage']
    
    # Normalize by row (each generation)
    row_sums = usage_matrix.sum(axis=1, keepdims=True)
    usage_matrix_norm = np.divide(usage_matrix, row_sums, 
                                   where=row_sums!=0, out=np.zeros_like(usage_matrix))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Heatmap 1: Absolute usage
    im1 = ax1.imshow(usage_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_xlabel('Program Index', fontsize=12)
    ax1.set_ylabel('Generation', fontsize=12)
    ax1.set_title('Program Usage (Absolute Counts)', fontsize=14, fontweight='bold')
    ax1.set_yticks(range(num_gens))
    ax1.set_yticklabels([f"Gen {i+1}" for i in range(num_gens)])
    plt.colorbar(im1, ax=ax1, label='Usage Count')
    
    # Heatmap 2: Normalized usage
    im2 = ax2.imshow(usage_matrix_norm, aspect='auto', cmap='YlOrRd', 
                     interpolation='nearest', vmin=0, vmax=1)
    ax2.set_xlabel('Program Index', fontsize=12)
    ax2.set_ylabel('Generation', fontsize=12)
    ax2.set_title('Program Usage (Normalized by Generation)', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(num_gens))
    ax2.set_yticklabels([f"Gen {i+1}" for i in range(num_gens)])
    plt.colorbar(im2, ax=ax2, label='Usage Fraction')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'usage_heatmap.png'), dpi=150)
    print(f"\n✓ Saved: {output_dir}/usage_heatmap.png")
    plt.close()


def plot_concentration_metrics(generations_data, output_dir="results/plots"):
    """Plot concentration metrics over generations"""
    os.makedirs(output_dir, exist_ok=True)
    
    generations = []
    top1_pcts = []
    top5_pcts = []
    gini_coeffs = []
    num_used = []
    
    for gen_data in generations_data:
        gen_num = gen_data['generation']
        all_usage = np.array(gen_data['all_usage'])
        
        total_usage = all_usage.sum()
        if total_usage == 0:
            continue
        
        generations.append(gen_num)
        
        # Top 1 percentage
        max_usage = all_usage.max()
        top1_pcts.append(max_usage / total_usage * 100)
        
        # Top 5 percentage
        top5_usage = np.sort(all_usage)[-5:].sum()
        top5_pcts.append(top5_usage / total_usage * 100)
        
        # Number of programs used
        num_used.append((all_usage > 0).sum())
        
        # Gini coefficient
        sorted_usage = np.sort(all_usage)
        n = len(sorted_usage)
        cumsum = np.cumsum(sorted_usage)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_usage)) / (n * cumsum[-1]) - (n + 1) / n
        gini_coeffs.append(gini)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top 1 concentration
    axes[0, 0].plot(generations, top1_pcts, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    axes[0, 0].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    axes[0, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    axes[0, 0].set_xlabel('Generation', fontsize=12)
    axes[0, 0].set_ylabel('Top Program Usage %', fontsize=12)
    axes[0, 0].set_title('Top 1 Program Concentration', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top 5 concentration
    axes[0, 1].plot(generations, top5_pcts, marker='s', linewidth=2, markersize=8, color='#3498db')
    axes[0, 1].axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
    axes[0, 1].set_xlabel('Generation', fontsize=12)
    axes[0, 1].set_ylabel('Top 5 Programs Usage %', fontsize=12)
    axes[0, 1].set_title('Top 5 Programs Concentration', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Number of programs used
    axes[1, 0].plot(generations, num_used, marker='D', linewidth=2, markersize=8, color='#2ecc71')
    axes[1, 0].axhline(y=64, color='green', linestyle='--', alpha=0.5, label='All 64 programs')
    axes[1, 0].axhline(y=32, color='orange', linestyle='--', alpha=0.5, label='Half (32 programs)')
    axes[1, 0].set_xlabel('Generation', fontsize=12)
    axes[1, 0].set_ylabel('Number of Programs Used', fontsize=12)
    axes[1, 0].set_title('Program Diversity', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 70])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gini coefficient
    axes[1, 1].plot(generations, gini_coeffs, marker='^', linewidth=2, markersize=8, color='#9b59b6')
    axes[1, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate inequality')
    axes[1, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='High inequality')
    axes[1, 1].set_xlabel('Generation', fontsize=12)
    axes[1, 1].set_ylabel('Gini Coefficient', fontsize=12)
    axes[1, 1].set_title('Usage Inequality (Gini)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('PPO Exploration & Program Usage Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concentration_metrics.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/concentration_metrics.png")
    plt.close()


def main():
    """Main analysis entry point"""
    print("\n🔬 Starting usage distribution analysis...")
    print("="*70)
    
    try:
        # Load data
        generations_data = load_portfolio_data()
        print(f"✓ Loaded data for {len(generations_data)} generations")
        
        # Analyze
        analyze_usage_distribution(generations_data)
        
        # Plot
        plot_usage_heatmap(generations_data)
        plot_concentration_metrics(generations_data)
        
        print("\n" + "="*70)
        print("✅ Analysis complete!")
        print("📁 Check results/plots/ for visualizations")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure training has completed and portfolio files exist in results/portfolios/")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
