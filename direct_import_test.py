#!/usr/bin/env python
"""Direct import test for ActionIndividual"""

import sys
print(f"Python path: {sys.path[:3]}")

print("\n1. Testing direct import of training.portfolio_types...")
try:
    import training.portfolio_types as pt
    print(f"   Module loaded: {pt}")
    print(f"   Module file: {pt.__file__}")
    print(f"   Module attributes: {[a for a in dir(pt) if not a.startswith('_')]}")
    
    if hasattr(pt, 'ActionIndividual'):
        print(f"   ✓ ActionIndividual found in module!")
        print(f"   ActionIndividual: {pt.ActionIndividual}")
    else:
        print(f"   ✗ ActionIndividual NOT found in module!")
        
except Exception as e:
    print(f"   ✗ ERROR importing: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing import ActionIndividual directly...")
try:
    from training.portfolio_types import ActionIndividual
    print(f"   ✓ Successfully imported Action Individual: {ActionIndividual}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
