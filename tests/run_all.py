"""Run every test file. Kept dependency-free so CI needs nothing but Python."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
files = sorted(f for f in here.glob("test_*.py"))
if not files:
    sys.exit("no test files found")

failed = []
for f in files:
    print(f"\n── {f.name} " + "─" * (60 - len(f.name)))
    if subprocess.run([sys.executable, str(f)]).returncode != 0:
        failed.append(f.name)

print("\n" + "=" * 62)
if failed:
    print(f"FAILED: {', '.join(failed)}")
    sys.exit(1)
print(f"all {len(files)} test files passed")
