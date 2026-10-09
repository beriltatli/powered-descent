import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _clean(o):
    # JSON has no inf/nan; write null so a missing number can never be read as a value.
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def save(name: str, data: dict) -> Path:
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(_clean(json.loads(json.dumps(data, default=_default))), indent=1))
    return p


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def machine() -> dict:
    import clarabel
    import cvxpy
    import ecos
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
    return {"cpu": chip or platform.processor(), "memory_gb": round(int(mem) / 2**30) if mem else None,
            "os": platform.platform(), "python": platform.python_version(),
            "cvxpy": cvxpy.__version__, "clarabel": clarabel.__version__, "ecos": ecos.__version__,
            "numpy": np.__version__}
