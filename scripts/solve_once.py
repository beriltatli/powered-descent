"""Nominal Section V scenario: Algorithm 1, the full verification battery, one trajectory figure."""
import numpy as np

from common import FIGURES, save
from figstyle import AXIS, INK2, MUTED, SERIES, plt, savefig

from dynamics.model import load_config, vehicle_from_config
from problem.setup import make_problem
from tof.search import algorithm1
from verify.bangbang import classify_thrust
from verify.exact32 import exact32_margins
from verify.internode import continuous_trajectory, internode_violation
from verify.slack import mass_tightness, slack_gaps, slack_report


def main():
    cfg = load_config()
    veh = vehicle_from_config(cfg)
    sc = cfg["scenario"]
    P = make_problem(cfg)
    res = algorithm1(P, sc["r0"], sc["v0"], veh.m_wet, veh.m_fuel,
                     coarse=cfg["tof"]["coarse_points"], tol=cfg["tof"]["golden_tol"])
    sol = res.sol
    out = {
        "N": P.N, "tf": res.tf, "fuel_kg": res.fuel, "classification": res.classification,
        "n_probes": len(res.probes), "rho1": veh.rho1, "rho2": veh.rho2, "alpha": veh.alpha,
        "slack": slack_report(sol, cfg["solver"]["slack_tol"]),
        "mass13": mass_tightness(sol, veh.alpha),
        "exact32": exact32_margins(sol, veh),
        "bangbang": classify_thrust(sol, veh, P.opts, veh.m_wet),
        "internode": internode_violation(sol, veh, P.opts.glideslope_deg, P.opts.vmax),
        "paper_fig7": {"tf": 81.0, "fuel_kg": 399.5},
        "trajectory": {"t": (sol.dt * np.arange(P.N + 1)).tolist(), "Y": sol.Y.tolist(),
                       "U": sol.U.tolist(), "S": sol.S.tolist(), "slack_gap": slack_gaps(sol).tolist()},
    }
    save("solve_once", out)

    t = sol.dt * np.arange(P.N)
    tc, Yc = continuous_trajectory(sol, veh)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    hz = np.linspace(0, Yc[:, 2].max() * 1.05, 50)
    ax[0].fill_between(hz, 0, hz / np.tan(np.deg2rad(sc["glideslope_deg"])), color=AXIS, alpha=0.5,
                       lw=0, label="outside glide slope (86°)")
    ax[0].plot(Yc[:, 2], Yc[:, 0], color=SERIES[0], label="continuous")
    ax[0].plot(sol.Y[:, 2], sol.Y[:, 0], "o", ms=3, color=SERIES[0], label="nodes")
    ax[0].set(xlabel="downrange r₃ (m)", ylabel="altitude r₁ (m)",
              title=f"Trajectory, tf = {res.tf:.1f} s, fuel = {res.fuel:.1f} kg")
    ax[0].legend(loc="upper left")
    ax[1].step(np.append(t, sol.tf), np.append(sol.thrust, sol.thrust[-1]) / 1e3, where="post",
               color=SERIES[0], label="‖Tc‖ = ‖u‖m")
    ax[1].axhline(veh.rho1 / 1e3, color=MUTED, lw=1, ls="--")
    ax[1].axhline(veh.rho2 / 1e3, color=MUTED, lw=1, ls="--")
    ax[1].text(1, veh.rho1 / 1e3 + 0.2, "ρ1", color=INK2, fontsize=8)
    ax[1].text(1, veh.rho2 / 1e3 - 0.6, "ρ2", color=INK2, fontsize=8)
    ax[1].set(xlabel="time (s)", ylabel="thrust (kN)", title=f"Thrust magnitude, N = {P.N}",
              ylim=(0, veh.rho2 / 1e3 * 1.12))
    savefig(fig, FIGURES / "trajectory_nominal.png")
    print({k: out[k] for k in ("tf", "fuel_kg", "classification", "slack", "mass13", "exact32", "internode")})
    print("bang-bang:", out["bangbang"]["counts"], "rho1", out["bangbang"]["n_rho1"], "rho2", out["bangbang"]["n_rho2"])


if __name__ == "__main__":
    main()
