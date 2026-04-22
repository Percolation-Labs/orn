import os
import sys
from pathlib import Path

# Put the package root on sys.path so `pip install -e .` is not required.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "orn" / "src"))

# numpy BLAS emits spurious "divide by zero" warnings on subnormal matmul inputs;
# we catch them in result modules but keep default filter here.
