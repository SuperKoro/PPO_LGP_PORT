"""Test imports one by one to find the problem"""

print("Testing imports step by step...")

print("\n1. Importing config...")
from config import LGPConfig
print("   OK")

print("\n2. Importing core.lgp_instructions...")
from core.lgp_instructions import LoadConstInstruction, ArithmeticInstruction, SetPortfolioInstruction
print("   OK")

print("\n3. Importing core.lgp_program (this may fail)...")
try:
    from core.lgp_program import LGPProgram, PortfolioBuilder
    print("   OK")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Importing core.lgp_generator...")
try:
    from core.lgp_generator import LGPGenerator  
    print("   OK")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nAll imports completed!")
