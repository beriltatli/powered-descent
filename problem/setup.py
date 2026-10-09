from dynamics.model import load_config, vehicle_from_config
from problem.pdg import Options, PDGProblem


def options_from_config(cfg: dict, **overrides) -> Options:
    sc = cfg["scenario"]
    kw = dict(glideslope_deg=sc["glideslope_deg"] if sc["glideslope"] else None,
              vmax=sc["vmax"], final_thrust_dir=tuple(sc["final_thrust_dir"]),
              pointing_axis=tuple(cfg["pointing"]["v"]))
    kw.update(overrides)
    return Options(**kw)


def make_problem(cfg: dict | None = None, N: int | None = None, **overrides) -> PDGProblem:
    cfg = cfg or load_config()
    veh = vehicle_from_config(cfg)
    return PDGProblem(N or cfg["discretization"]["N"], veh, options_from_config(cfg, **overrides))
