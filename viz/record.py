"""Record closed-loop flights for the static (GitHub Pages) build of the 3-D view.

Without viz/server.py there is no solver, so the page replays these files instead: the same
hello / plan / step / end events the server streams, with steps thinned to STEP_HZ of sim time.
One file per (start, eps) on the page's eps slider; the random start uses RANDOM_SEEDS.
Run: .venv/bin/python viz/record.py   (writes viz/flights/*.json and viz/flights/index.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import KW, P, R, VEH, end_event, fly, hello, jsonable, scenario_state  # noqa: E402

OUT = Path(__file__).resolve().parent / "flights"
EPS_PCT = [x / 2 for x in range(-10, 11)]   # the page's slider: -5 % to +5 % in 0.5 % steps
RANDOM_SEEDS = [11, 23, 37, 41]             # NOT IN PAPER — a few random starts for the static page
STEP_HZ = 20.0


def rounded(x, nd):
    if isinstance(x, dict):
        return {k: rounded(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v, nd) for v in x]
    return round(x, nd) if isinstance(x, float) else x


def record(query: dict, eps: float) -> dict:
    label, r0, v0 = scenario_state(query)
    events, last = [], {"t": -1.0, "step": None}

    def observer(kind: str, data: dict) -> None:
        if kind == "plan":
            plan = rounded(jsonable(data), 2)
            plan["r"] = rounded(plan["r"], 1)   # drawn as a dashed line; 10 cm is invisible
            events.append(["plan", plan])
            return
        last["step"] = data
        if data["t"] - last["t"] >= 1 / STEP_HZ - 1e-9:
            last["t"] = data["t"]
            events.append(["step", rounded(jsonable(data), 2)])

    res = fly(P, VEH, r0, v0, VEH.m_wet, eps, R=R, observer=observer, **KW)
    if last["step"] is not None and last["step"]["t"] > last["t"]:
        events.append(["step", rounded(jsonable(last["step"]), 2)])
    head = hello("replay", label, r0, v0, eps)
    return {"hello": jsonable(head), "events": events, "end": rounded(jsonable(end_event(res)), 4)}


def main():
    OUT.mkdir(exist_ok=True)
    starts = [("mc", {"scenario": "mc"}), ("section_v", {"scenario": "section_v"})]
    starts += [(f"random{i}", {"scenario": "random", "seed": str(s)}) for i, s in enumerate(RANDOM_SEEDS)]
    for name, q in starts:
        for pct in EPS_PCT:
            flight = record(q, pct / 100)
            path = OUT / f"{name}_{pct:+.1f}.json"
            path.write_text(json.dumps(flight, separators=(",", ":"), allow_nan=False))
            e = flight["end"]
            print(f"{path.name}: {e['status']}, err {e['landing_error']}, {path.stat().st_size // 1024} KB", flush=True)
    index = {"eps_pct": EPS_PCT, "random": len(RANDOM_SEEDS), "step_hz": STEP_HZ}
    (OUT / "index.json").write_text(json.dumps(index))


if __name__ == "__main__":
    main()
