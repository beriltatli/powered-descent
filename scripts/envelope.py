"""Landable envelope over initial downrange position and horizontal velocity, with and without mass error."""
import os
from multiprocessing import Pool

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from common import FIGURES, save
from figstyle import AXIS, CRITICAL, GOOD, INK2, SERIOUS, WARNING, plt, savefig

from dynamics.model import load_config
from metrics.dispersion import success
from montecarlo.harness import run_envelope_cell

WORKERS = max(1, min(6, (os.cpu_count() or 2) - 2))
CODES = {"landable": 0, "failed_in_flight": 1, "insufficient_fuel": 2, "insufficient_thrust": 3,
         "initial_state_infeasible": 4, "solver_failure": 5}


def cell_codes(cell: dict, radius: float, speed: float) -> tuple[int, int]:
    cls = cell["class"]
    if cls in ("insufficient_thrust", "initial_state_infeasible", "solver_failure"):
        return CODES[cls], CODES[cls]
    no_err = "landable" if cls == "valid" and success(cell["no_error"], radius, speed) else (
        "insufficient_fuel" if cls == "insufficient_fuel" else "failed_in_flight")
    ok = all(success(c, radius, speed) for c in cell["corners"].values())
    with_err = "landable" if ok else ("insufficient_fuel" if any(c["fuel_exhausted"] for c in cell["corners"].values())
                                      or cls == "insufficient_fuel" else "failed_in_flight")
    return CODES[no_err], CODES[with_err]


def plot(xs, vs, no_err, with_err, cfg) -> None:
    sc = cfg["scenario"]
    n0, n1 = int((np.asarray(no_err) == 0).sum()), int((np.asarray(with_err) == 0).sum())
    colors = [GOOD, SERIOUS, WARNING, CRITICAL, AXIS, "#4a3aa7"]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    dx, dv = xs[1] - xs[0], vs[1] - vs[0]
    ext = [xs[0] - dx / 2, xs[-1] + dx / 2, vs[0] - dv / 2, vs[-1] + dv / 2]
    for a, grid, title in ((ax[0], no_err, f"Perfect mass knowledge: {n0} landable cells"),
                           (ax[1], with_err, f"ε = ±5 % (both corners), with margin: {n1} landable")):
        a.imshow(np.asarray(grid), origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=-0.5,
                 vmax=len(colors) - 0.5, interpolation="nearest")
        a.plot(sc["r0"][2], sc["v0"][2], "s", mfc="none", mec="#0b0b0b", ms=9, mew=1.5)
        a.plot(cfg["montecarlo"]["r0"][2], cfg["montecarlo"]["v0"][2], "o", mfc="none", mec="#0b0b0b", ms=9, mew=1.5)
        a.set(xlabel="initial downrange r₃ (m)", title=title)
        a.grid(False)
    ax[0].set_ylabel("initial horizontal velocity ṙ₃ (m/s)")
    labels = ["landable", "feasible, failed in flight", "insufficient fuel (plan or in flight)",
              "insufficient thrust", "starts outside glide slope", "solver failure"]
    handles = [Patch(color=c, label=lab) for c, lab in zip(colors, labels)]
    handles += [plt.Line2D([], [], marker="s", ls="", mfc="none", mec="#0b0b0b", label="Section V state"),
                plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#0b0b0b", label="Monte Carlo state")]
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(FIGURES / "envelope.png", dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    ev, sc = cfg["envelope"], cfg["scenario"]
    xs = np.linspace(*ev["downrange"][:2], int(ev["downrange"][2]))
    vs = np.linspace(*ev["horizontal_velocity"][:2], int(ev["horizontal_velocity"][2]))
    args = [([sc["r0"][0], 0.0, float(x)], [sc["v0"][0], 0.0, float(v)], ev["mass_error_cases"])
            for v in vs for x in xs]
    with Pool(WORKERS) as pool:
        cells = pool.map(run_envelope_cell, args, chunksize=2)
    radius, speed = cfg["success"]["landing_radius"], cfg["success"]["touchdown_speed"]
    no_err = np.zeros((len(vs), len(xs)), int)
    with_err = np.zeros_like(no_err)
    for i, c in enumerate(cells):
        a, b = cell_codes(c, radius, speed)
        no_err.flat[i], with_err.flat[i] = a, b
    n0, n1 = int((no_err == 0).sum()), int((with_err == 0).sum())
    summary = {"grid": [len(vs), len(xs)], "landable_no_error": n0, "landable_with_error": n1,
               "shrink_fraction": 1 - n1 / n0 if n0 else None,
               "counts_no_error": {k: int((no_err == v).sum()) for k, v in CODES.items()},
               "counts_with_error": {k: int((with_err == v).sum()) for k, v in CODES.items()},
               "lost_cells": [{"downrange": float(xs[j]), "v_horizontal": float(vs[i]),
                               "with_error": [k for k, v in CODES.items() if v == with_err[i, j]][0]}
                              for i in range(len(vs)) for j in range(len(xs)) if no_err[i, j] == 0 and with_err[i, j] != 0]}
    save("envelope", {"summary": summary, "downrange": xs, "v_horizontal": vs, "no_error": no_err,
                      "with_error": with_err, "codes": CODES, "cells": cells})

    plot(xs, vs, no_err, with_err, cfg)
    print(summary)


if __name__ == "__main__":
    main()
