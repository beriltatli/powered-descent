from guidance.mpc import fly


def fly_open_loop(P, veh, r0, v0, m0, eps, **kw):
    """Same optimizer, same mass belief and planning margin as the closed loop, solved once at t = 0 and
    flown blind. The gap to the closed loop isolates the value of re-solving."""
    return fly(P, veh, r0, v0, m0, eps, mode="open", **kw)
