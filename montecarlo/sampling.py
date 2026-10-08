import numpy as np


def draw_instance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # NOT IN PAPER — a box around the Section V example, chosen to include diverts toward and away
    # from the target and near-hover starts. Speed is capped below vmax so (10) holds at t = 0.
    while True:
        r0 = np.array([rng.uniform(500, 2500), rng.uniform(-1000, 1000), rng.uniform(-3000, 3000)])
        v0 = np.array([rng.uniform(-90, 10), rng.uniform(-40, 40), rng.uniform(-100, 100)])
        if np.linalg.norm(v0) < 140:
            return r0, v0


def draw_pointing_instance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # NOT IN PAPER — high, slow starts like the Section VI example (5000 m, at rest), where the
    # fuel-optimal thrust can point downward early on and the pointing cone binds.
    r0 = np.array([rng.uniform(1500, 5000), rng.uniform(-1500, 1500), rng.uniform(-1500, 1500)])
    v0 = np.array([rng.uniform(-40, 30), rng.uniform(-30, 30), rng.uniform(-30, 30)])
    return r0, v0
