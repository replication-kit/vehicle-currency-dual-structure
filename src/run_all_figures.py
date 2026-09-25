import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

scripts = [
    "fig1.py",
    "fig2.py",
    "fig3.py",
    "fig4.py",
    "fig5.py",
    "figC1.py",
    "figC2.py",
]

for name in scripts:
    print(f"Running {name} ...")
    subprocess.run([sys.executable, str(HERE / name)], check=True)

print("All figure scripts completed.")
