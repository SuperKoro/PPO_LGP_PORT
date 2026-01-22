"""
Extract top portfolios from training results for Phase 2.
Reads elite portfolios from all generations to create a curated action library.
"""

import os
import sys
import json

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from training.portfolio_types import ActionIndividual, Gene


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
    all_portfolios.sort(key=lambda x: x['fitness'])
    
    return all_portfolios


def portfolio_signature(p):
    """Create unique signature."""
    mh_parts = [f"{m['name']}:{round(m['weight'], 1)}" for m in p['mh_genes']]
    return f"{p['dr']}|{'|'.join(mh_parts)}"


def extract_top_unique(all_portfolios, top_k=10):
    """Extract top K unique portfolios."""
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


def main():
    print("=" * 60)
    print("EXTRACTING TOP PORTFOLIOS FOR PHASE 2")
    print("=" * 60)
    
    all_portfolios = load_all_elite_portfolios()
    print(f"\n✓ Loaded {len(all_portfolios)} elite portfolios")
    
    top_portfolios = extract_top_unique(all_portfolios, top_k=10)
    print(f"✓ Found {len(top_portfolios)} unique portfolios")
    
    print("\n" + "=" * 60)
    print("TOP PORTFOLIOS")
    print("=" * 60)
    
    for i, p in enumerate(top_portfolios, 1):
        print(f"\n#{i} | Fitness: {p['fitness']:.2f} | Gen {p['generation']} | Usage: {p['usage']}")
        print(f"   DR: {p['dr']}")
        for j, mh in enumerate(p['mh_genes'], 1):
            print(f"   MH{j}: {mh['name']} (weight={mh['weight']:.2f})")
    
    # Save
    output_file = "results/top_portfolios_phase2.json"
    with open(output_file, 'w') as f:
        json.dump({'portfolios': top_portfolios}, f, indent=2)
    
    print(f"\n✓ Saved to: {output_file}")
    print("\n✅ DONE - Use these portfolios for Phase 2!")


if __name__ == "__main__":
    main()
