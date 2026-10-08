"""Regenerate every results/*.json, every figure and REPORT.md, in dependency order.

Wall time on an Apple M3 (8 GB) is roughly 1.5-2 hours; the Monte Carlo and envelope dominate.
timing.py must run with nothing else on the machine, so it runs alone and first.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["timing.py", "solve_once.py", "verify_battery.py", "sweep_tof.py", "nonconvex_compare.py",
         "pointing_gap.py", "lcvx_conditions.py",
         "convergence.py", "baselines.py", "montecarlo.py", "ablation.py", "envelope.py",
         "report.py"]


def main():
    for step in STEPS:
        print(f"== {step}", flush=True)
        subprocess.run([sys.executable, str(HERE / step)], check=True)


if __name__ == "__main__":
    main()
