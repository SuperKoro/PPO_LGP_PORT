"""
Simple test without special characters
"""

from typing import Dict
from core.lgp_instructions import *
from core.lgp_program import LGPProgram, PortfolioBuilder
from core.lgp_generator import LGPGenerator
from config import LGPConfig
import random


def test_1_simple_program():
    """Test 1: Simple program with LoadConst + Arithmetic + SET_* instructions"""
    print("\n" + "="*70)
    print("TEST 1: Simple Program Execution")
    print("="*70)
    
    instructions = [
        # Load some constants
        LoadConstInstruction(dest=0, value=5.0),
        LoadConstInstruction(dest=1, value=10.0),
        
        # Arithmetic
        ArithmeticInstruction(dest=2, op="+", src1=0, src2=1, src2_is_const=False),
        ArithmeticInstruction(dest=3, op="*", src1=2, src2=2.0, src2_is_const=True),
        
        # Set portfolio
        SetPortfolioInstruction(component="DR", reg_name=0),
        SetPortfolioInstruction(component="MH1", reg_name=1, reg_weight=2),
        SetPortfolioInstruction(component="MH2", reg_name=2, reg_weight=3),
        SetPortfolioInstruction(component="MH3", reg_name=3, reg_weight=2),
    ]
    
    program = LGPProgram(instructions=instructions, num_registers=20)
    
    # Execute
    inputs = {"num_jobs": 10, "avg_processing_time": 8.5, "avg_ops_per_job": 3.0}
    try:
        portfolio = program.execute(inputs)
        
        # Validate
        dr_name = portfolio.genes[0].name
        mh_genes = portfolio.genes[1:]
        
        assert dr_name in LGPConfig.available_dr, f"DR {dr_name} not in available_dr"
        assert len(mh_genes) == LGPConfig.n_mh_genes, f"Expected {LGPConfig.n_mh_genes} MH genes, got {len(mh_genes)}"
        
        for mh in mh_genes:
            assert mh.name in LGPConfig.available_mh, f"MH {mh.name} not in available_mh"
        
        print(f"[PASSED] Portfolio = {dr_name} | " + ", ".join([f"{g.name}:{g.w_raw:.2f}" for g in mh_genes]))
        return True
        
    except Exception as e:
        print(f"[FAILED] {e}")
        return False


def test_2_random_generation():
    """Test 2: Generate 10 random programs and execute all"""
    print("\n" + "="*70)
    print("TEST 2: Random Program Generation (10 programs)")
    print("="*70)
    
    rng = random.Random(42)
    generator = LGPGenerator(
        max_length=LGPConfig.max_program_length,
        min_length=LGPConfig.min_program_length,
        num_registers=LGPConfig.num_registers,
        rng=rng
    )
    
    inputs = {
        "num_jobs": 12.0,
        "avg_processing_time": 8.5,
        "avg_ops_per_job": 3.2
    }
    
    success_count = 0
    fail_count = 0
    
    for i in range(10):
        try:
            program = generator.generate_random_program()
            portfolio = program.execute(inputs)
            
            # Quick validation
            assert len(portfolio.genes) == 1 + LGPConfig.n_mh_genes
            assert portfolio.genes[0].name in LGPConfig.available_dr
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            print(f"  Program {i+1} failed: {e}")
    
    print(f"\n[SUCCESS] {success_count}/10 programs executed correctly")
    if fail_count > 0:
        print(f"[FAILED] {fail_count}/10 programs crashed")
        return False
    else:
        print("[ALL PASSED] ALL 10 PROGRAMS PASSED!")
        return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LGP UNIT TESTS")
    print("="*70)
    
    # Import config to validate
    from config import validate_config
    
    print("\nValidating configuration...")
    validate_config()
    
    # Run tests
    results = []
    results.append(("Simple Program", test_1_simple_program()))
    results.append(("Random Generation", test_2_random_generation()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum([1 for name, result in results if result])
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! LGP System is ready!")
        exit(0)
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Please debug before running main.py")
        exit(1)
