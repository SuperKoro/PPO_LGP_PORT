"""Debug training module imports"""

print("Testing training module imports...")

print("\n1. Import training.ppo_model...")
try:
    from training.ppo_model import PPOActorCritic
    print("   OK")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Import training.portfolio_types...")
try:
    from training.portfolio_types import Gene, ActionIndividual
    print("   OK")
    print(f"   Gene: {Gene}")
    print(f"   ActionIndividual: {ActionIndividual}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Import training.typed_action_adapter...")
try:
    from training.typed_action_adapter import run_action_individual
    print("   OK")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Import from training package...")
try:
    from training import ActionIndividual
    print("   OK - imported ActionIndividual from training package")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
