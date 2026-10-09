from dynamics.model import load_config
from problem.setup import make_problem


def guidance_kwargs(cfg: dict | None = None) -> dict:
    g = (cfg or load_config())["guidance"]
    return {"dt_resolve": g["dt_resolve"], "t_stop": g["t_stop"], "sim_dt": g["sim_dt"],
            "plan_mass_margin": g["plan_mass_margin"], "rho2_margin": g["rho2_margin"]}


def guidance_problems(cfg: dict | None = None):
    """Nominal Problem 4 and the soft-state recovery problem, both compiled once per process."""
    cfg = cfg or load_config()
    return make_problem(cfg), make_problem(cfg, soft_state=True)
