"""Local server for the 3-D view: serves viz/ and flies the closed-loop guidance live.

GET /api/fly?scenario=mc|section_v|random|custom&eps=0.03&speed=1[&seed=..][&r0=a,b,c&v0=a,b,c]
streams Server-Sent Events while guidance.mpc.fly runs in real time (paced to the wall clock):
  hello  - vehicle, scenario and guidance settings
  plan   - every planning step: which fallback produced it, solve time, planned positions
  step   - every plant step (decimated to the display rate at high speed)
  end    - FlightResult summary
POST /api/control?id=..&speed=..&paused=0|1 changes pacing of a running flight.

Binds to 127.0.0.1 only. Run: .venv/bin/python viz/server.py [port]
"""
import json
import sys
import threading
import time
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dynamics.model import load_config, vehicle_from_config  # noqa: E402
from guidance.config import guidance_kwargs, guidance_problems  # noqa: E402
from guidance.mpc import fly  # noqa: E402
from montecarlo.sampling import draw_instance  # noqa: E402

CFG = load_config()
VEH = vehicle_from_config(CFG)
P, R = guidance_problems(CFG)
KW = guidance_kwargs(CFG)
FLIGHT_LOCK = threading.Lock()
RUNS: dict[str, dict] = {}
DISPLAY_HZ = 60.0


class FlightAborted(Exception):
    pass


def jsonable(x):
    # numpy scalars to Python; NaN/inf to null, which JSON.parse in the browser accepts.
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def scenario_state(q: dict) -> tuple[str, np.ndarray, np.ndarray]:
    name = q.get("scenario", "mc")
    if name == "section_v":
        sc = CFG["scenario"]
        return "Section V state (2007 paper, no fuel margin)", np.array(sc["r0"], float), np.array(sc["v0"], float)
    if name == "random":
        seed = int(q.get("seed", time.time_ns() % 2**31))
        r0, v0 = draw_instance(np.random.default_rng(seed))
        return f"Random start (Monte Carlo box, seed {seed})", r0, v0
    if name == "custom":
        r0 = np.array([float(x) for x in q["r0"].split(",")])
        v0 = np.array([float(x) for x in q["v0"].split(",")])
        return "Custom start", r0, v0
    mc = CFG["montecarlo"]
    return "Monte Carlo state", np.array(mc["r0"], float), np.array(mc["v0"], float)


def hello(run_id: str, label: str, r0, v0, eps: float) -> dict:
    v = CFG["vehicle"]
    return {"id": run_id, "scenario": label, "r0": list(r0), "v0": list(v0), "eps": eps,
            "vehicle": {"m_dry": VEH.m_dry, "m_wet": VEH.m_wet, "rho1": VEH.rho1, "rho2": VEH.rho2,
                        "n_engines": v["n_thrusters"], "engine_max_N": v["thrust_per_engine"],
                        "cant_deg": v["cant_deg"], "throttle_min": v["throttle_min"],
                        "throttle_max": v["throttle_max"], "isp": v["isp"], "g": VEH.g.tolist()},
            "guidance": {"N": P.N, "glideslope_deg": P.opts.glideslope_deg, "dt_resolve": KW["dt_resolve"],
                         "t_stop": KW["t_stop"], "plan_mass_margin": KW["plan_mass_margin"],
                         "rho2_margin": KW["rho2_margin"], "solver": CFG["solver"]["backend"]}}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".gltf": "model/gltf+json", ".bin": "application/octet-stream", ".json": "application/json"}

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/fly":
            return self.fly(parse_qs(url.query))
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/control":
            return self.send_error(404)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        run = RUNS.get(q.get("id", ""))
        if run is None:
            return self.send_error(404, "no such flight")
        if "speed" in q:
            run["speed"] = min(max(float(q["speed"]), 0.1), 20.0)
        if "paused" in q:
            run["paused"] = q["paused"] == "1"
        self.send_response(204)
        self.end_headers()

    def send_event(self, name: str, data: dict) -> None:
        self.wfile.write(f"event: {name}\ndata: {json.dumps(jsonable(data), separators=(',', ':'), allow_nan=False)}\n\n".encode())
        self.wfile.flush()

    def fly(self, qs: dict) -> None:
        q = {k: v[-1] for k, v in qs.items()}
        eps = float(q.get("eps", 0.0))
        if abs(eps) > 0.2:
            return self.send_error(400, "eps outside [-0.2, 0.2]")
        label, r0, v0 = scenario_state(q)
        run_id = uuid.uuid4().hex[:8]
        run = {"speed": float(q.get("speed", 1.0)), "paused": False, "abort": False}
        for other in RUNS.values():
            other["abort"] = True        # a new flight pre-empts the current one
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        with FLIGHT_LOCK:
            RUNS[run_id] = run
            try:
                self.send_event("hello", hello(run_id, label, r0, v0, eps))
                clock = {"wall": time.perf_counter(), "sim": 0.0, "sent": -1.0}

                def observer(kind: str, data: dict) -> None:
                    if run["abort"]:
                        raise FlightAborted
                    if kind == "plan":
                        self.send_event("plan", data)
                        return
                    # Pace: wall time advances by sim dt / speed; a pause shifts the wall origin.
                    clock["wall"] += (data["t"] - clock["sim"]) / run["speed"]
                    clock["sim"] = data["t"]
                    while run["paused"] and not run["abort"]:
                        time.sleep(0.05)
                        clock["wall"] = time.perf_counter()
                    delay = clock["wall"] - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    if data["t"] - clock["sent"] >= run["speed"] / DISPLAY_HZ - 1e-9:
                        clock["sent"] = data["t"]
                        self.send_event("step", data)

                res = fly(P, VEH, r0, v0, VEH.m_wet, eps, R=R, observer=observer, **KW)
                self.send_event("end", {"status": res.status, "t_end": res.t_end,
                                        "r_end": None if res.r_end is None else res.r_end.tolist(),
                                        "v_end": None if res.v_end is None else res.v_end.tolist(),
                                        "landing_error": res.landing_error, "touchdown_speed": res.touchdown_speed,
                                        "fuel_used": res.fuel_used, "fuel_exhausted": res.fuel_exhausted,
                                        "n_resolves": res.n_resolves, "n_feasible": res.n_feasible,
                                        "recursively_feasible": res.recursively_feasible,
                                        "solve_ms_p50": float(np.median(res.solve_ms)) if res.solve_ms else None,
                                        "solve_ms_max": float(np.max(res.solve_ms)) if res.solve_ms else None})
            except (FlightAborted, BrokenPipeError, ConnectionResetError):
                pass
            finally:
                RUNS.pop(run_id, None)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    # Pay CVXPY's one-time canonicalization before the first flight.
    mc = CFG["montecarlo"]
    P.solve(np.array(mc["r0"]), np.array(mc["v0"]), VEH.m_wet, 60.0)
    R.solve(np.array(mc["r0"]), np.array(mc["v0"]), VEH.m_wet, 60.0)
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT / "viz")))
    print(f"http://127.0.0.1:{port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
