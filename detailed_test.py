"""
Detailed import test to find the exact error
"""

import sys
import traceback

def test_import(module_name, item_name=None):
    """Test importing a module or item"""
    try:
        if item_name:
            print(f"  Importing {item_name} from {module_name}...", end=" ")
            exec(f"from {module_name} import {item_name}")
        else:
            print(f"  Importing {module_name}...", end=" ")
            exec(f"import {module_name}")
        print("✓")
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        traceback.print_exc()
        return False

print("=" * 70)
print("DETAILED IMPORT TEST")
print("=" * 70)

print("\n1. Testing config...")
test_import("config", "LGPConfig")
test_import("config", "PPOConfig")

print("\n2. Testing training.portfolio_types...")
test_import("training.portfolio_types", "Gene")
test_import("training.portfolio_types", "ActionIndividual")

print("\n3. Testing core.lgp_instructions...")
test_import("core.lgp_instructions", "LoadConstInstruction")

print("\n4. Testing core.lgp_program...")
test_import("core.lgp_program", "LGPProgram")

print("\n5. Testing core.lgp_generator...")
test_import("core.lgp_generator", "LGPGenerator")

print("\n6. Testing registries...")
test_import("registries.dispatching_registry")
test_import("registries.mh_registry")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
