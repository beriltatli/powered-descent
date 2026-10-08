**Question: can a fuel-optimal powered-descent guidance law built from lossless convexification (Açıkmeşe & Ploen 2007) be verified against its own theory, solved within a flight-software latency budget, and fly a soft pinpoint landing in closed loop when the vehicle mass is wrong by up to 5 %?**

Every number below is rendered from `results/*.json` by `scripts/report.py`; `tests/test_report.py` fails if this file and the results disagree. Hardware for every timing: ${hw}, Python ${py}, CVXPY ${cvxpy}, Clarabel ${clarabel}.

## Headline

Monte Carlo state ${mc_r0} m, ${mc_v0} m/s; ${mc_runs} runs, mass error ε ~ U[−5 %, +5 %] (the guidance believes m̂ = (1 + ε) m); success means landing error < 5 m **and** touchdown speed < 1 m/s. Solve time: N = ${N}, Clarabel, p99 over ${tm_n} solves.

| Method (ε ~ U[−5 %, +5 %]) | Mean fuel (kg) | Within 5 m | Below 1 m/s | p99 solve | Recursive feasibility |
|:--|--:|--:|--:|--:|--:|
| Constant T/W gravity turn | ${constant_tw_fuel_mean} | ${constant_tw_within_radius} | ${constant_tw_below_speed} | none (closed form) | n/a |
| Polynomial (Apollo-style), engine-saturated | ${polynomial_fuel_mean} | ${polynomial_within_radius} | ${polynomial_below_speed} | none (closed form) | n/a |
| Open-loop convex optimum, flown blind | ${open_loop_fuel_mean} | ${open_loop_within_radius} | ${open_loop_below_speed} | ${tm_p99} ms (once) | n/a (one solve) |
| **Convex MPC, re-solved every 1 s (this repo)** | **${closed_fuel_mean}** | **${closed_within_radius}** | **${closed_below_speed}** | **${tm_p99} ms** | **${rf_res}** of re-solves, ${rf_runs} of runs |

Not flyable, shown for reference: the polynomial law with an *ideal* engine (no saturation) lands every run (${polynomial_ideal_within_radius} within 5 m) on ${polynomial_ideal_fuel_mean} kg, but spends ${polyi_outside} of its flight commanding thrust outside [ρ1, ρ2].

Recursive feasibility is reported two ways (section 6): above is "a hard-constraint solution existed at every re-solve, at some tf"; the stricter "the re-solve at the originally planned landing time was feasible" is ${strict_res} of re-solves and ${strict_runs} of runs.

## 1. Theory in brief

Full derivation, with equation numbers, in [THEORY.md](THEORY.md). In one paragraph: the thrust-magnitude annulus ρ1 ≤ ‖Tc‖ ≤ ρ2 (3) is nonconvex; Problem 2 lifts it to a cone with a slack Γ, (14)–(15), and Lemma 1 proves the slack is tight at the optimum, so the relaxation is lossless. The change of variables σ = Γ/m, u = Tc/m, z = ln m, (25)–(31), makes the dynamics linear; the remaining nonconvex bound (32) is replaced by Taylor bounds (34) that Lemma 2 proves conservative. Zero-order-hold discretization gives Problem 4, an SOCP, solved for the time of flight by the line search of Algorithm 1.

Vehicle and scenario are the paper's Section V example (config.yaml cites page and equation for each value): ρ1 = ${rho1} N, ρ2 = ${rho2} N, α = ${alpha} s/m.

**One deviation from the paper, argued here.** The paper fixes Δt and searches the integer N. This repo fixes N at compile time and makes Δt = tf/N a cvxpy Parameter, so one compiled problem serves every tf and every initial state. It is the same line search over tf; it is what makes the per-solve latency numbers meaningful for a guidance loop, and it lets closed-loop guidance refine Δt as time-to-go shrinks.

## 2. Losslessness verification (no pointing constraint)

**Reproduction of the paper.** At N = ${N} the nominal Section V case (glide slope on) gives tf = ${nom_tf} s and ${nom_fuel} kg; the paper's Fig. 7 reports 81 s and 399.5 kg with an unstated Δt. At N = ${cv_ref} this repo gets ${cv_sv_fuel_ref} kg (`tests/test_problem.py` asserts the N = 200 case is within 1 kg).

**The paper's example has no fuel margin.** It carries 400 kg. Algorithm 1 at N = ${N} needs ${cv_sv_fuel_N} kg and classifies the case as **${nom_class}**; it first becomes valid at N = ${cv_first_valid}, where the p99 solve time is ${cv_first_valid_p99} ms, over the 50 ms budget. This is why the Monte Carlo uses a nearby state with margin (section 6) and also flies the paper's state as given.

| N | Section V fuel (kg) | Section V class | MC-state fuel (kg) | p50 ms | p99 ms |
|--:|--:|:--|--:|--:|--:|
${cv_table}

Fuel converges monotonically with N. At the chosen N = ${N} the gap to N = ${cv_ref} is ${cv_mc_gap} kg (${cv_mc_gap_pct} %) on the Monte Carlo state and ${cv_sv_gap} kg on the Section V state. N = ${N} was chosen over N = 80 (p99 ${cv_p99_80} ms) and N = 120 (p99 ${cv_p99_120} ms) for headroom: a closed-loop flight makes ${resolves_per_flight} re-solves on average.

**Battery on ${bat_n} random feasible instances** (${bat_draws} draws: ${bat_valid} valid, ${bat_fuel} insufficient fuel, ${bat_thrust} insufficient thrust; none dropped). `results/lemma1_battery.json`, `figures/lemma1_battery.png`.

| Theory | Test | Result |
|:--|:--|:--|
| Exact ZOH | discrete vs `solve_ivp`, 12 random intervals | < 1e-9 relative (`tests/test_dynamics.py`); ballistic case to 1e-9 m |
| Lemma 1, slack tight | σ_k − ‖u_k‖ at every node, tolerance 1e-4 m/s² | median ${bat_q50}, 99th pct ${bat_q99}, 99.9th ${bat_q999}, max ${bat_qmax} m/s². **${bat_loose} instance with a loose node** (#${bat_loose_i}, node 0, gap ${bat_loose_gap}); re-solved at 1e-10 tolerances it is tight to ${bat_loose_tight}, i.e. an interior-point residual, not a losslessness violation |
| Lemma 1, bang-bang | Γ_k on the (34) bounds | ${bb_switch} nodes at a switch inside an interval, ${bb_short_arc} on an arc shorter than one or two intervals, ${bb_final_dir} pinned by the final thrust direction (37), ${bb_state_active} on an active state constraint, **${bb_unexplained} unexplained** |
| Lemma 2 | exact (32) at every node | holds at ${bat_e32}; worst upper margin ${bat_e32_up} N (solver tolerance) |
| (32) between nodes | lower bound at the interval end | worst ${bat_e32_end_abs} N below ρ1: Γ = σm falls inside a ZOH interval, and (51) is imposed at the interval start only |
| (13) tight | fuel booked through σ vs burned through ‖u‖ | max difference ${bat_mass13} kg |
| Inter-node | continuous re-propagation, 40 samples per interval | glide slope: max ${bat_gs_max} m outside the cone, 95th pct ${bat_gs_p95} m; speed bound ${bat_speed} m/s; subsurface ${bat_sub} m |

The paper's (13) is an equality, ṁ = −αΓ; the relaxation is (14). In terms of the actual thrust the mass law is ṁ ≤ −α‖Tc‖, tight exactly when the slack is, so "(13) tight" is the same fact as "slack tight", measured here in kilograms.

**Does Lemma 1 cover active state constraints?** No. The proof's Hamiltonian (16) and costate equations (19)–(20) have no multiplier terms for (7) or (8); with a state constraint active the costate need not have λ1 constant and λ2 affine, which the proof relies on. The classifier above found no intermediate thrust attributable to an active constraint at N = ${N}, but that is an observation, not a theorem. The citing list in the 2007 PDF includes Harris & Açıkmeşe (2014) on state-constrained lossless convexification; that paper is not in `refs/` and nothing here relies on it.

**Remark 10, unimodality** (`figures/tof_sweep.png`). The nominal fuel-vs-tf curve is ${tof_uni}; ${tof_rand_uni} of ${tof_rand} feasible random instances are unimodal, and on ${tof_alg_ok} of ${tof_rand} Algorithm 1's golden-section result is at least as good as the 80-point grid. The search strategy is valid on everything tested. The remark's second claim, that the optimum sits very close to the minimum feasible tf, does not hold here: feasible from ${tof_min} s, optimal at ${tof_arg} s.

**Nonconvex comparison** (`results/nonconvex.json`, `figures/nonconvex_compare.png`). ${nc_n} instances, N = ${nc_N}, SLSQP from ${nc_starts} random starts each on Problem 1 at the convex tf.

- With the same Taylor bounds as Problem 4 (isolates Lemma 1): the convex optimum is at least as good as the best start on **${nc_t_ok} of ${nc_n}**; median relative gap of the best start ${nc_t_gap}; only ${nc_t_feas} of ${nc_t_total} starts reached a feasible point. The convex optimum, plugged into Problem 1 with σ = ‖u‖, violates its constraints by at most ${nc_t_viol}. Because Problem 4 is a relaxation, "convex ≤ best start" is guaranteed whenever the convex solve is global; the content of the test is that the convex point is Problem-1 feasible and that no start beats it.
- With the exact bound (32), which is the literal original Problem 1: SLSQP beats the convex optimum on **every one of the ${nc_e_beaten} instances where a start reached a feasible point** (${nc_e_none} had none), by up to ${nc_e_gap} % in β. **This criterion is missed as literally stated.** The cause is Lemma 2's conservatism (the Taylor interval is strictly inside the exact one), not a losslessness failure: warm-started from the convex optimum, SLSQP on the exact problem improves β by at most ${nc_e_warm} %.

**Infeasibility classification** is Algorithm 1's (fuel vs thrust) plus a separate class for initial states that already violate a state constraint and one for solver failure (any non-certificate status). `tests/test_tof.py` exercises each.

## 3. The pointing-constraint gap, and what the 2013 paper adds

Section VI's claim reproduced (`results/pointing.json`, `figures/pointing_example.png`). On the paper's 5 km hover-start example: without pointing, tf = ${pt_off_tf} s and ${pt_off_fuel} kg (paper: 69 s, 293.6 kg); with (61) exact, tf = ${pt_ex_tf} s and ${pt_ex_fuel} kg (paper, with its heuristic: 78 s, 303 kg), and **${pt_ex_loose} nodes have u ≈ 0 while σ > 0**: the vehicle is told to thrust below ρ1, exactly the mechanism Section VI describes.

Over ${pt_n} random high, slow starts (thrust cone about +altitude):

| Cone | Formulation | Instances with material loose slack (gap > 1e-3) | Commanded thrust < 0.99 ρ1 | Pointing violated |
|:--|:--|--:|--:|--:|
| 90° (γ = 0, paper) | no pointing | ${pt_90deg_off_mat} | ${pt_90deg_off_thr} | ${pt_90deg_off_viol} |
| 90° | (61) exact | ${pt_90deg_exact61_mat} | ${pt_90deg_exact61_thr} | ${pt_90deg_exact61_viol} |
| 90° | heuristic (62) | ${pt_90deg_heuristic62_mat} | ${pt_90deg_heuristic62_thr} | ${pt_90deg_heuristic62_viol} |
| 45° | no pointing | ${pt_45deg_off_mat} | ${pt_45deg_off_thr} | ${pt_45deg_off_viol} |
| 45° | (61) exact | ${pt_45deg_exact61_mat} | ${pt_45deg_exact61_thr} | ${pt_45deg_exact61_viol} |
| 45° | heuristic (62) | ${pt_45deg_heuristic62_mat} | ${pt_45deg_heuristic62_thr} | ${pt_45deg_heuristic62_viol} (up to ${pt_45deg_maxviol}°) |

Why it happens, in my words: Lemma 1 works because, with no limit on direction, the Hamiltonian-maximizing thrust always points along R2 = λ2/m and uses the whole slack. A pointing cone can exclude every direction with a positive component along R2; then the best the relaxed problem can do is u = 0 and pay σ ≥ ρ1 e^{−z} anyway, which the original problem forbids. The relaxed problem's optimum is also non-unique in these arcs (sideways thrust at ρ1 on the cone edge costs the same σ); the paper's Fig. 11 shows horizontal thrust at about 5 kN for the first seconds, one such tight point, while the interior-point solver here returns the centre of the optimal face, u ≈ 0.

Three findings beyond the paper:

1. **Loose slack without any pointing constraint** (${pt_90deg_off_mat} of ${pt_n}; ${pt_off_loose} nodes on the paper's example, thrust down to ${pt_off_min} ρ1). These are direction reversals: R2 passes through zero at one instant (the one point Lemma 1's proof allows), the optimal thrust flips from down to up, and zero-order hold averages the flip over a whole interval. Lossless in continuous time, not at the nodes.
2. **The heuristic as printed is a no-op at γ = 0**, the paper's own example: κ2 is a norm, so κ2 ≥ γ = 0 always holds and case 1 (exact) is always taken (`tests/test_pointing_nonconvex.py`). The paper's Fig. 11 cannot have come from case 2 of the rule as printed.
3. At 45°, where the heuristic does relax, it does not restore tightness here (material loose ${pt_45deg_heuristic62_mat} vs ${pt_45deg_exact61_mat}) and it violates the pointing cone, at a median fuel cost of ${pt_45deg_pen} kg.

### What the 2013 result requires, and whether this model meets it

**The 2013 paper (Açıkmeşe, Carson & Blackmore, IEEE TCST) is still not in `refs/`.** Its result is stated
here from a secondary source: Malyuta et al., *Convex Optimization for Trajectory Generation* (2022 tutorial,
`refs/malyuta_2022_tutorial.pdf`). The tutorial's Theorem 2 (p.8) cites the 2013 paper jointly with Carson
et al. 2011. The NASA Tech Brief in `refs/` only asserts the result. THEORY.md §8 has the exact statements.

**What the relaxation is.** The 2013 relaxation of the pointing constraint is the same half-space as the
2007 paper's (61). What 2013 adds is a set of hypotheses under which that relaxation is lossless:
Conditions 1–3 of the tutorial.

**How this repo's model fares** (`results/lcvx_conditions.json`):

- **Condition 1** (controllability of {A, B}) holds.
- **Condition 2** (transversality, here σ(tf) ≠ 0) holds on every solve: on ${lc_none_2007_90deg_c2} of ${lc_none_2007_90deg_solved} with the 90° cone, and it is guaranteed by (51).
- **Condition 3** ({A, BN} controllable, with N the thrust directions orthogonal to the cone axis) **fails**.
  - The 2007 dynamics have no planet rotation, and the vertical channel cannot be reached by horizontal thrust: rank ${lc_c3_rank} of 6.
  - The tutorial (p.46) says it holds as long as ω × ê_z ≠ 0, that is, only if the planet rotates about an axis other than the local vertical.
  - With the tutorial's Mars rotation (its eq. 108c) it holds in exact arithmetic. But the smallest eigenvalue of the controllability Gramian over ${lc_T} s is at round-off level (${lc_c3_eig_tut}); Condition 1's is ${lc_c1_eig}.
- **The tutorial's glide-slope test (106)**, which shows the glide slope can only touch instantaneously:
  - holds for a 45° cone (worst margin ${lc_fb45} N);
  - **fails for the paper's 90° cone** ((106b) is violated by ${lc_fb90} N).
  - In practice no solution in any of these batteries touched the glide slope.

**So the 2007 Section VI setting lies outside the 2013 guarantee.** The loose slack measured above does not contradict it.

**Does meeting Condition 3 remove the loose slack?** Same ${pt_n} instances, same N = ${N}. Rotation was added to the dynamics.

| Rotation | Cone | Material loose slack | of which not a reversal | Commanded thrust < 0.99 ρ1 | Median fuel (kg) |
|:--|:--|--:|--:|--:|--:|
| none (2007 model; Condition 3 fails) | none | ${lc_none_2007_off_mat} | ${lc_none_2007_off_nrev} | ${lc_none_2007_off_thr} | ${lc_none_2007_off_fuel} |
| none | 90° | ${lc_none_2007_90deg_mat} | ${lc_none_2007_90deg_nrev} | ${lc_none_2007_90deg_thr} | ${lc_none_2007_90deg_fuel} |
| none | 45° | ${lc_none_2007_45deg_mat} | ${lc_none_2007_45deg_nrev} | ${lc_none_2007_45deg_thr} | ${lc_none_2007_45deg_fuel} |
| Mars, tutorial (108c) (Condition 3 holds, barely) | none | ${lc_mars_tutorial_off_mat} | ${lc_mars_tutorial_off_nrev} | ${lc_mars_tutorial_off_thr} | ${lc_mars_tutorial_off_fuel} |
| Mars, tutorial | 90° | ${lc_mars_tutorial_90deg_mat} | ${lc_mars_tutorial_90deg_nrev} | ${lc_mars_tutorial_90deg_thr} | ${lc_mars_tutorial_90deg_fuel} |
| Mars, tutorial | 45° | ${lc_mars_tutorial_45deg_mat} | ${lc_mars_tutorial_45deg_nrev} | ${lc_mars_tutorial_45deg_thr} | ${lc_mars_tutorial_45deg_fuel} |
| 100× tutorial (diagnostic, NOT IN PAPER; Gramian eigenvalue ${lc_c3_eig_x100}) | none | ${lc_diagnostic_x100_off_mat} | ${lc_diagnostic_x100_off_nrev} | ${lc_diagnostic_x100_off_thr} | ${lc_diagnostic_x100_off_fuel} |
| 100× tutorial | 90° | ${lc_diagnostic_x100_90deg_mat} | ${lc_diagnostic_x100_90deg_nrev} | ${lc_diagnostic_x100_90deg_thr} | ${lc_diagnostic_x100_90deg_fuel} |
| 100× tutorial | 45° | ${lc_diagnostic_x100_45deg_mat} | ${lc_diagnostic_x100_45deg_nrev} | ${lc_diagnostic_x100_45deg_thr} | ${lc_diagnostic_x100_45deg_fuel} |

The 45° rows meet every hypothesis we can check once rotation is present: Conditions 1–3, and (106).

**What the measurements show:**

- **With Mars's real rotation, satisfying Condition 3 changes nothing measurable.** The coupling it relies on is |ω|·tf = ${lc_wt} rad over a median descent.
- **Exaggerated 100×, the coupling cuts the loose slack but does not remove it.**
  - At 90°, every remaining material case is a reversal, the zero-order-hold artefact that also appears with no pointing at all.
  - At 45°, ${lc_diagnostic_x100_45deg_mat} of ${pt_n} instances are still loose by the Section VI mechanism, even though every hypothesis listed above holds.
- **The theorem is a continuous-time statement, and the tutorial gives no guarantee after discretization.** This is the gap these numbers fall into.
- **Reading of the result (mine, not the sources'):** at N = ${N}, Condition 3 matters only through a coupling strong enough to act within one descent. Real Mars rotation is not strong enough.
- **The 2013 hypotheses are therefore stated, from a secondary source, and checked.** The primary paper's exact wording remains unverified.

## 4. Solve-time tail

N = ${N}, Clarabel ${clarabel} via CVXPY ${cvxpy}, ${hw}; ${tm_n} solves at the Algorithm 1 optimum for random initial states (`figures/solve_time_tail.png`).

| per solve (ms) | p50 | p95 | p99 | max |
|:--|--:|--:|--:|--:|
| wall, whole `solve()` call incl. parameter mapping | ${tm_p50} | ${tm_p95} | **${tm_p99}** | ${tm_max} |
| Clarabel-reported | ${ts_p50} | | ${ts_p99} | ${ts_max} |

- Compile, reported separately: building the problem ${build_ms} ms, first-solve canonicalization ${compile_ms} ms (median of 5 cold processes). Paid once.
- What DPP buys: ${dpp_p50} ms p50 (${dpp_p99} p99) re-solving through the cached mapping vs ${rc_p50} ms p50 (${rc_p99} p99) recompiling every call. Modest at this size: the interior-point solve itself dominates.
- Infeasible solves (${ti_n} sampled): p50 ${ti_p50} ms, p99 ${ti_p99} ms.
- A full Algorithm 1 search (14-point grid plus golden section) costs ${ta_p50} ms p50, ${ta_p99} ms p99. That is a once-per-flight cost here; the closed loop holds the landing time fixed and only re-times when the fixed-time problem is infeasible.

**p99 per solve is ${tm_p99} ms against the 50 ms target: met.**

## 5. Baselines at fixed mass error

Monte Carlo state, every method on the same plant (`results/baselines.json`). Landing error (m) / touchdown speed (m/s) / fuel (kg):

| ε | Convex MPC | Open loop | Polynomial (saturated) | Polynomial (ideal engine) | Constant T/W |
|:--|:--|:--|:--|:--|:--|
| −5 % | ${bl_closed_m005_err} / ${bl_closed_m005_spd} / ${bl_closed_m005_fuel} | ${bl_open_loop_m005_err} / ${bl_open_loop_m005_spd} / ${bl_open_loop_m005_fuel} | ${bl_polynomial_m005_err} / ${bl_polynomial_m005_spd} / ${bl_polynomial_m005_fuel} | ${bl_polynomial_ideal_m005_err} / ${bl_polynomial_ideal_m005_spd} / ${bl_polynomial_ideal_m005_fuel} | ${bl_constant_tw_m005_err} / ${bl_constant_tw_m005_spd} / ${bl_constant_tw_m005_fuel} |
| 0 | ${bl_closed_p000_err} / ${bl_closed_p000_spd} / ${bl_closed_p000_fuel} | ${bl_open_loop_p000_err} / ${bl_open_loop_p000_spd} / ${bl_open_loop_p000_fuel} | ${bl_polynomial_p000_err} / ${bl_polynomial_p000_spd} / ${bl_polynomial_p000_fuel} | ${bl_polynomial_ideal_p000_err} / ${bl_polynomial_ideal_p000_spd} / ${bl_polynomial_ideal_p000_fuel} | ${bl_constant_tw_p000_err} / ${bl_constant_tw_p000_spd} / ${bl_constant_tw_p000_fuel} |
| +5 % | ${bl_closed_p005_err} / ${bl_closed_p005_spd} / ${bl_closed_p005_fuel} | ${bl_open_loop_p005_err} / ${bl_open_loop_p005_spd} / ${bl_open_loop_p005_fuel} | ${bl_polynomial_p005_err} / ${bl_polynomial_p005_spd} / ${bl_polynomial_p005_fuel} | ${bl_polynomial_ideal_p005_err} / ${bl_polynomial_ideal_p005_spd} / ${bl_polynomial_ideal_p005_fuel} | ${bl_constant_tw_p005_err} / ${bl_constant_tw_p005_spd} / ${bl_constant_tw_p005_fuel} |

- **Constant T/W** (gravity turn, T/W = ${mc_tw} calibrated once at ε = 0 so the vehicle stops at zero altitude): the fuel scale for "slow down, ignore where": ${constant_tw_fuel_mean} kg on average and a median miss of ${constant_tw_landing_error_p50} m. Under mass error it either stops in mid-air or hits the ground.
- **Polynomial** (a = −6r/tgo² − 4ṙ/tgo − g, given the convex optimum's tgo = ${mc_tgo} s): its raw command lies outside [ρ1, ρ2] in ${poly_outside_runs} of runs, for ${poly_outside} of the flight on average. With the engine saturating that command it hits the ground every run. With an ideal engine it lands, cheaper than the convex solution, because it uses thrust the engine does not have and cuts through the glide slope. That frequency is the argument for the convex approach.
- **Open loop**: the same optimizer and the same mass handling as the closed loop, solved once. It lands only when the effective mass error is zero (ε = −5 % with the planning margin); otherwise it misses by hundreds of metres. The difference to the closed loop is the value of re-solving.

## 6. Closed loop under ±5 % mass error

Guidance: Problem 4 at N = ${N}, re-solved every 1 s from the current state with the landing time held fixed (shrinking horizon, Δt = time-to-go / N), stopping re-solves 0.25 s before touchdown. Plant: true point-mass dynamics (1)–(2), RK4, thrust saturated into [ρ1, ρ2], fuel exhaustion modelled. Mass error: the guidance believes m̂ = (1 + ε) m at all times and commands Tc = u m̂, so the vehicle's acceleration is (1 + ε) times what was planned.

**Three design choices came out of failures in development; each is reported, and each is switched off in an ablation on the same mass-error samples (`results/ablation.json`).**

1. **Plan with the heaviest plausible mass.** With the plain estimate, ε < 0 crashes: the guidance thinks the vehicle is lighter, every acceleration falls short by |ε|, and fuel-optimal plans end in a full-thrust burn with no headroom to absorb that (without the margin, on the Monte Carlo state: ε = −5 % ends in ${nm_m005_status} at ${nm_m005_spd} m/s, ${nm_m005_err} m from the target; ε = +5 % lands at ${nm_p005_spd} m/s; `results/baselines.json`). The guidance therefore plans with m̂(1 + δ), δ = 5.26 % (`config.yaml: plan_mass_margin`), mapping ε ∈ [−5 %, +5 %] to an effective error in [0, +10.5 %], where the vehicle is only ever stronger than planned. Ablation without it, on the first ${ab_n} samples: ${ab_nm_success} success overall, ${ab_nm_neg_success} of the ${ab_nm_neg} runs with ε < 0.
2. **Tighten the upper thrust bound in the nominal re-solve.** The fixed-time re-solve has zero slack during the final full-thrust burn: a deviation of a few percent makes it infeasible at the planned landing time, the hard problem is re-timed a little later, and repeated re-times stretch the flight until the fuel runs out. The nominal re-solve therefore plans with 0.97 ρ2; the first fallback re-solves at the same landing time with the true ρ2, using the headroom (constraint tightening, as in robust MPC). ${tightening_sentence}
3. **Fallback order.** When the tightened fixed-time re-solve is infeasible: the true thrust bounds at the same time; then the hard problem re-timed (Algorithm 1 from the current state); only if no tf works, a recovery problem with the glide slope and speed bound as penalized soft constraints; only then the old plan is held. Recursive feasibility counts a re-solve as feasible if any of the first three, all hard-constraint problems, has a solution.

Results, ${mc_runs} runs (`results/montecarlo.json`, `figures/montecarlo.png`), mass error ε ~ U[−5 %, +5 %]:

| | Value | Criterion |
|:--|--:|:--|
| Landing error p50 / p95 / max (m) | ${closed_landing_error_p50} / ${closed_landing_error_p95} / ${closed_landing_error_max} | |
| Within 5 m | ${closed_within_radius} | ≥ 95 %: **${crit_radius}** |
| Touchdown speed p50 / p95 / max (m/s) | ${closed_touchdown_speed_p50} / ${closed_touchdown_speed_p95} / ${closed_touchdown_speed_max} | |
| Below 1 m/s | ${closed_below_speed} | ≥ 95 %: **${crit_speed}** |
| Fuel mean / p95 (kg) | ${closed_fuel_mean} / ${closed_fuel_p95} | 400 on board; 5th-pct remaining ${fuel_rem_p05} kg; ${closed_exhausted} runs exhausted |
| Recursive feasibility, hard problem at some tf | ${rf_res} of ${rf_total} re-solves; ${rf_runs} of runs | > 99 %: **${crit_rf}** |
| Strict: feasible at the original landing time | ${strict_res} of re-solves; ${strict_runs} of runs | |
| Fallbacks used | ${rf_physical} true-ρ2 re-solves, ${rf_retimed_hard} hard re-times, ${rf_recovery} soft recoveries, ${rf_retimed} soft re-times, ${rf_held} holds | |

${mc_narrative}

Every failure is the same mechanism: a run of hard re-times, each pushing the landing a little later, stretches the flight until the tank is empty. Problem 4 does not constrain fuel (Remark 8: fuel is checked after the solve), and under this error model the guidance's own fuel estimate is off by the same ε, so it cannot see the tank running dry. Adding z_N ≥ ln m_dry to the guidance problem would not fix that without a separate fuel measurement; it is listed under limitations.

**The paper's own Section V state, flown the same way** (same ${mc_runs} ε samples): within 5 m ${section_v_closed_within_radius}, below 1 m/s ${section_v_closed_below_speed}, ${section_v_closed_exhausted} runs ran out of fuel. With 400 kg on board and a fuel-optimal need of about 400 kg before any margin, this state is not landable under mass error at all; that is a property of the scenario, not of the guidance.

## 7. The landable envelope

Initial downrange position × initial horizontal velocity, ${env_grid} grid, altitude 1500 m and vertical velocity −75 m/s from the Section V state (`figures/envelope.png`, `results/envelope.json`). Left: perfect mass knowledge (no margin). Right: landable only if the closed loop succeeds at both corners ε = −5 % and ε = +5 %, with the planning margin.

- Perfect mass knowledge: ${env_n0} landable cells; ${env_thrust} insufficient thrust, ${env_fuel0} insufficient fuel, ${env_gs} start outside the glide slope, ${env_fail0} feasible but failed in flight.
- With ±5 % mass error: ${env_n1} landable cells. ${env_lost} cells are lost: ${env_lost_fuel} to fuel, ${env_lost_flight} to in-flight failure.

**${env_sentence}**

## 8. Success criteria

| Criterion | Result |
|:--|:--|
| Discretization matches continuous propagation to 1e-9 | met |
| Slack tight at every node on 500/500 instances, or every violation characterized | met with one characterized exception (instance #${bat_loose_i}, solver residual) |
| Every solution satisfies exact (32) | met at every node; between nodes the lower bound dips by up to ${bat_e32_end_abs} N below ρ1 (characterized) |
| Convex ≥ best of 5 nonconvex starts on 20 instances | met on ${nc_t_ok}/${nc_n} against the Taylor-bounded Problem 1; **missed** on ${nc_e_beaten}/${nc_n} against exact (32), by ≤ ${nc_e_gap} % (Lemma 2 conservatism) |
| Fuel converges with N; gap stated | met; ${cv_mc_gap} kg (${cv_mc_gap_pct} %) at N = ${N} vs N = ${cv_ref} |
| p99 per-solve < 50 ms, compile separate | met: ${tm_p99} ms; compile ${compile_ms} ms |
| 1000 runs, ±5 %: ≥ 95 % within 5 m | ${crit_radius} (${closed_within_radius}) |
| ≥ 95 % below 1 m/s | ${crit_speed} (${closed_below_speed}) |
| Recursive feasibility > 99 % | ${crit_rf} (${rf_res} of re-solves; strict fixed-time ${strict_res}) |
| Envelope with and without mass error | produced |
| Pointing: loose fraction, Section VI heuristic, 2013 hypotheses | done; 2013 hypotheses stated from a **secondary source** (Malyuta et al. tutorial) and checked: Condition 3 fails in the 2007 model; the 2013 paper itself is still not in refs/ |

## 9. Choices not taken from the papers

All marked `NOT IN PAPER` in config.yaml: the speed bound V̂ = 150 m/s (never active in these scenarios); N = ${N} and Δt = tf/N (the paper gives no Δt); the solver; the instance distributions; the Monte Carlo state (Section V position, horizontal velocity halved, because the Section V state has no fuel margin); the 1 s re-solve period and the 0.25 s cutoff (ablation with a 1 s cutoff: touchdown speed p95 ${ab_ts_p95} m/s and ${ab_ts_below} below 1 m/s, against ${closed_touchdown_speed_p95} m/s and ${closed_below_speed} with 0.25 s); the planning mass margin; the 0.97 ρ2 tightening; the fallback chain; the envelope grid. The Section VI initial velocity is taken as zero (text and Fig. 10) rather than the Fig. 11 caption's 20 m/s, which contradicts that figure's own velocity panel.

## 10. Limitations

- 3-DoF point mass: no attitude dynamics, so thrust direction changes instantly (including the full reversals in section 3); no engine lag or throttle-rate limit.
- Constant gravity, flat surface, no aerodynamics, no winds.
- Perfect navigation except mass; mass error is a single persistent multiplicative bias, not an estimator with dynamics or noise.
- The thrust bound (32) is enforced at nodes; inside an interval the commanded thrust can dip below ρ1 by a fraction of a percent, which the plant then saturates.
- One vehicle, one altitude for the envelope; envelope corners at ±5 % stand in for the whole interval.
- Timings are on a laptop CPU with a general-purpose solver through CVXPY; a flight computer with a custom solver would differ in both directions.
- Fuel is not a constraint of the guidance problem (Remark 8), and the guidance has no fuel measurement independent of its mass belief; all closed-loop failures are fuel exhaustion.
- The 2013 theorem's hypotheses come from the Malyuta et al. tutorial, not from the 2013 paper itself, and the tutorial gives no proof. Rotation enters the guidance model only in the Theorem 2 experiment; the Monte Carlo and envelope use the non-rotating 2007 model.
