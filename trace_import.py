"""
Trace the exact import error and save to file
"""
import sys
import traceback
import io

output = io.StringIO()

output.write("=" * 70 + "\n")
output.write("TRACING CIRCULAR IMPORT ISSUE\n")
output.write("=" * 70 + "\n\n")

output.write("Attempting: from core.lgp_program import LGPProgram\n")
output.write("-" * 70 + "\n")

try:
    from core.lgp_program import LGPProgram
    output.write("\nSUCCESS! LGPProgram imported successfully\n")
    output.write(f"LGPProgram: {LGPProgram}\n")
except ImportError as e:
    output.write(f"\nIMPORT ERROR: {e}\n")
    output.write("\nFull traceback:\n")
    output.write("-" * 70 + "\n")
    output.write(traceback.format_exc())
    output.write("-" * 70 + "\n")
except Exception as e:
    output.write(f"\nUNEXPECTED ERROR: {e}\n")
    output.write(traceback.format_exc())

output.write("\n" + "=" * 70 + "\n")

# Save to file
result = output.getvalue()
with open("import_trace_result.txt", "w", encoding="utf-8") as f:
    f.write(result)

print(result)
print("\nResult saved to: import_trace_result.txt")
