"""
Test imports step by step
"""

print("=" * 70)
print("STEP-BY-STEP IMPORT TEST")
print("=" * 70)

def test_step(step_num, description, code):
    print(f"\n{step_num}. {description}")
    try:
        exec(code, globals())
        print("   ✓ Success")
        return True
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test imports one by one
test_step(1, "Import config", "from config import LGPConfig")
test_step(2, "Import portfolio_types.Gene", "from training.portfolio_types import Gene")  
test_step(3, "Import portfolio_types.ActionIndividual", "from training.portfolio_types import ActionIndividual")
test_step(4, "Import lgp_instructions.LoadConstInstruction", "from core.lgp_instructions import LoadConstIns import LGPProgram")
test_step(6, "Import lgp_generator", "from core.lgp_generator import LGPGenerator")

print("\n" + "=" * 70)
print("If all passed, the project structure is OK!")
print("=" * 70)
