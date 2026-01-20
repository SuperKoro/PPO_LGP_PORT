# -*- coding: utf-8 -*-
"""
Wrapper script to run training with proper UTF-8 encoding
"""
import sys
import io

# Force UTF-8 encoding for stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Now import and run the actual training script
from scripts.train_lgp import main

if __name__ == "__main__":
    main()
