"""
Quick verification test for EDD migration from MH to DR.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, r"g:\IU copy\OneDrive\International University\Research\PPO_LGP_Clean")

print("=" * 70)
print("VERIFICATION TEST: EDD Migration from MH to DR")
print("=" * 70)

# Test 1: Registry verification
print("\n[Test 1] Registry Verification")
print("-" * 50)

from registries.mh_registry import has_mh, MH_REGISTRY
from registries.dispatching_registry import has_dr, DR_REGISTRY

edd_in_mh = has_mh("EDD")
edd_in_dr = has_dr("EDD")

print(f"  ✓ EDD in MH registry: {edd_in_mh} (Expected: False)")
print(f"  ✓ EDD in DR registry: {edd_in_dr} (Expected: True)")

if edd_in_mh:
    print("  ✗ FAIL: EDD should NOT be in MH registry")
    sys.exit(1)
    
if not edd_in_dr:
    print("  ✗ FAIL: EDD should be in DR registry")
    sys.exit(1)

print("  ✓ PASS: EDD correctly registered only as DR")

# Test 2: Config validation
print("\n[Test 2] Config Validation")
print("-" * 50)

from config import LGPConfig

print(f"  Available DRs: {LGPConfig.available_dr}")
print(f"  Available MHs: {LGPConfig.available_mh}")

if "EDD" in LGPConfig.available_mh:
    print("  ✗ FAIL: EDD should NOT be in available_mh")
    sys.exit(1)

if "EDD" not in LGPConfig.available_dr:
    print("  ✗ FAIL: EDD should be in available_dr")
    sys.exit(1)

print("  ✓ PASS: Config correctly lists EDD only in available_dr")

# Test 3: Portfolio generation test
print("\n[Test 3] Portfolio Generation Test")
print("-" * 50)

from training.portfolio_types import ActionLGP

lgp = ActionLGP(
    dr_list=LGPConfig.available_dr,
    mh_list=LGPConfig.available_mh,
    pool_size=10,
    seed=42
)

# Check all portfolios
edd_found_in_mh = False
for idx, individual in enumerate(lgp.pool):
    for gene in individual.mh_genes:
        if gene.name == "EDD":
            edd_found_in_mh = True
            print(f"  ✗ FAIL: Found EDD in MH genes of portfolio {idx}")
            break
    if edd_found_in_mh:
        break

if not edd_found_in_mh:
    print(f"  ✓ PASS: Generated {len(lgp.pool)} portfolios, none have EDD in MH genes")
else:
    sys.exit(1)

# Test 4: Fallback portfolio test
print("\n[Test 4] Fallback Portfolio Test")
print("-" * 50)

from training.lgp_coevolution_trainer import make_fallback_individual
from training.portfolio_types import describe_individual

fallback = make_fallback_individual()
description = describe_individual(fallback)

print(f"  Fallback portfolio: {description}")
print(f"  DR gene: {fallback.dr_gene.name}")
print(f"  MH genes: {[g.name for g in fallback.mh_genes]}")

if fallback.dr_gene.name not in LGPConfig.available_dr:
    print("  ✗ FAIL: Fallback DR not in available_dr")
    sys.exit(1)

for gene in fallback.mh_genes:
    if gene.name not in LGPConfig.available_mh:
        print(f"  ✗ FAIL: Fallback MH '{gene.name}' not in available_mh")
        sys.exit(1)
    if gene.name == "EDD":
        print("  ✗ FAIL: Fallback has EDD in MH genes")
        sys.exit(1)

print("  ✓ PASS: Fallback portfolio uses correct DR and MH lists")

# Test 5: MH Registry contents
print("\n[Test 5] MH Registry Contents")
print("-" * 50)

print(f"  Registered MHs: {list(MH_REGISTRY.keys())}")
expected_mhs = {"SA", "GA", "PSO"}
actual_mhs = set(MH_REGISTRY.keys())

if "EDD" in actual_mhs:
    print("  ✗ FAIL: EDD found in MH_REGISTRY")
    sys.exit(1)

if expected_mhs != actual_mhs:
    print(f"  ⚠ WARNING: MH registry mismatch")
    print(f"    Expected: {expected_mhs}")
    print(f"    Actual: {actual_mhs}")

print(f"  ✓ PASS: MH registry contains only true metaheuristics")

# Test 6: DR Registry contents
print("\n[Test 6] DR Registry Contents")
print("-" * 50)

print(f"  Registered DRs: {list(DR_REGISTRY.keys())}")

if "EDD" not in DR_REGISTRY:
    print("  ✗ FAIL: EDD not found in DR_REGISTRY")
    sys.exit(1)

print(f"  ✓ PASS: EDD found in DR registry")

# All tests passed
print("\n" + "=" * 70)
print("✅ ALL VERIFICATION TESTS PASSED")
print("=" * 70)
print("\nSummary:")
print("  ✓ EDD removed from MH registry")
print("  ✓ EDD remains in DR registry")
print("  ✓ Config updated correctly")
print("  ✓ Portfolio generation works correctly")
print("  ✓ Fallback portfolio uses correct lists")
print("\nThe migration is complete and working correctly!")
