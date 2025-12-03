"""Test import lgp_instructions directly without going through __init__.py"""

print("1. Testing direct import of lgp_instructions module...")
try:
    import core.lgp_instructions
    print("   OK - Module imported successfully")
    print(f"   Module file: {core.lgp_instructions.__file__}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing import specific classes from lgp_instructions...")
try:
    from core.lgp_instructions import LoadConstInstruction, SetPortfolioInstruction
    print("   OK - Classes imported successfully")
    print(f"   LoadConstInstruction: {LoadConstInstruction}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
