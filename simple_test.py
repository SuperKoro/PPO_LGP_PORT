"""
Simple test to check imports and basic functionality
"""

print("Testing imports...")

try:
    print("1. Importing config...")
    from config import LGPConfig, PPOConfig
    print("   ✓ Config imported")
    
    print("2. Importing portfolio types...")
    from training.portfolio_types import ActionIndividual, Gene
    print("   ✓ Portfolio types imported")
    
    print("3. Importing LGP instructions...")
    from core.lgp_instructions import LoadConstInstruction, ArithmeticInstruction
    print("   ✓ LGP instructions imported")
    
    print("4. Importing LGP program...")
    from core.lgp_program import LGPProgram, PortfolioBuilder
    print("   ✓ LGP program imported")
    
    print("5. Importing LGP generator...")
    from core.lgp_generator import LGPGenerator
    print("   ✓ LGP generator imported")
    
    print("\n✅ All imports successful!")
    
    print("\n6. Creating a simple LGP program...")
    instructions = [
        LoadConstInstruction(dest=0, value=5.0),
        LoadConstInstruction(dest=1, value=10.0),
    ]
    program = LGPProgram(instructions=instructions, num_registers=20)
    print(f"   ✓ Created program with {len(program.instructions)} instructions")
    
    print("\n🎉 Basic tests PASSED!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
