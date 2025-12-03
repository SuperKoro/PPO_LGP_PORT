#!/usr/bin/env python
"""
Run test_lgp with detailed error reporting
"""

import sys
import traceback

print("=" * 70)
print("RUNNING TEST_LGP WITH DETAILED ERRORS")
print("=" * 70)

try:
    # Import all the test functions
    print("\n1. Importing test module...")
    sys.path.insert(0, '.')
    from scripts import test_lgp
    
    print("\n2. Running tests...")
    
    # Run each test individually
    print("\n--- Test 1: Simple Program ---")
    try:
        result1 = test_lgp.test_1_simple_program()
    except Exception as e:
        print(f"Test 1 failed with error: {e}")
        traceback.print_exc()
        result1 = False
    
    print("\n--- Test 2: Conditional Skip ---")
    try:
        result2 = test_lgp.test_2_conditional_skip()
    except Exception as e:
        print(f"Test 2 failed with error: {e}")
        traceback.print_exc()
        result2 = False
    
    print("\n--- Test 3: Random Generation ---")
    try:
        result3 = test_lgp.test_3_random_generation()
    except Exception as e:
        print(f"Test 3 failed with error: {e}")
        traceback.print_exc()
        result3 = False
    
    print("\n--- Test 4: Serialization ---")
    try:
        result4 = test_lgp.test_4_program_serialization()
    except Exception as e:
        print(f"Test 4 failed with error: {e}")
        traceback.print_exc()
        result4 = False
    
    results = [result1, result2, result3, result4]
    passed = sum(results)
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: {passed}/4 tests passed")
    print(f"{'='*70}")
    
except ImportError as e:
    print(f"\n✗ Failed to import test_lgp module!")
    print(f"Error: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"\n✗ Unexpected error!")
    print(f"Error: {e}")
    traceback.print_exc()
