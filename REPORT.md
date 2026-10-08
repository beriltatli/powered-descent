**Question: can a fuel-optimal powered-descent guidance law built from lossless convexification (Açıkmeşe & Ploen 2007) be verified against its own theory, solved within a flight-software latency budget, and fly a soft pinpoint landing in closed loop when the vehicle mass is wrong by up to 5 %?**

Every number below is rendered from `results/*.json` by `scripts/report.py`; `tests/test_report.py` fails if this file and the results disagree. Hardware for every timing: Apple M3, 8 GB, macOS, Python 3.12.6, CVXPY 1.9.3, Clarabel 0.11.1.

## Headline

Monte Carlo state [1500.0, 0.0, 2000.0] m, [-75.0, 0.0, 50.0] m/s; 1000 runs, mass error ε ~ U[−5 %, +5 %] (the guidance believes m̂ = (1 + ε) m); success means landing error < 5 m **and** touchdown speed < 1 m/s. Solve time: N = 50, Clarabel, p99 over 600 solves.

| Method (ε ~ U[−5 %, +5 %]) | Mean fuel (kg) | Within 5 m | Below 1 m/s | p99 solve | Recursive feasibility |
|:--|--:|--:|--:|--:|--:|
| Constant T/W gravity turn | 181.54 | 0.0 % | 53.7 % | none (closed form) | n/a |
| Polynomial (Apollo-style), engine-saturated | 204.08 | 0.0 % | 0.0 % | none (closed form) | n/a |
| Open-loop convex optimum, flown blind | 357.90 | 0.2 % | 2.7 % | 20.1 ms (once) | n/a (one solve) |
| **Convex MPC, re-solved every 1 s (this repo)** | **334.61** | **100.0 %** | **99.3 %** | **20.1 ms** | **99.99 %** of re-solves, 99.3 % of runs |

Not flyable, shown for reference: the polynomial law with an *ideal* engine (no saturation) lands every run (100.0 % within 5 m) on 313.43 kg, but spends 20.1 % of its flight commanding thrust outside [ρ1, ρ2].

Recursive feasibility is reported two ways (section 6): above is "a hard-constraint solution existed at every re-solve, at some tf"; the stricter "the re-solve at the originally planned landing time was feasible" is 95.48 % of re-solves and 26.0 % of runs.

## 1. Theory in brief

Full derivation, with equation numbers, in [THEORY.md](THEORY.md). In one paragraph: the thrust-magnitude annulus ρ1 ≤ ‖Tc‖ ≤ ρ2 (3) is nonconvex; Problem 2 lifts it to a cone with a slack Γ, (14)–(15), and Lemma 1 proves the slack is tight at the optimum, so the relaxation is lossless. The change of variables σ = Γ/m, u = Tc/m, z = ln m, (25)–(31), makes the dynamics linear; the remaining nonconvex bound (32) is replaced by Taylor bounds (34) that Lemma 2 proves conservative. Zero-order-hold discretization gives Problem 4, an SOCP, solved for the time of flight by the line search of Algorithm 1.

Vehicle and scenario are the paper's Section V example (config.yaml cites page and equation for each value): ρ1 = 4,971.8 N, ρ2 = 13,258.2 N, α = 5.0863e-04 s/m.

**One deviation from the paper, argued here.** The paper fixes Δt and searches the integer N. This repo fixes N at compile time and makes Δt = tf/N a cvxpy Parameter, so one compiled problem serves every tf and every initial state. It is the same line search over tf; it is what makes the per-solve latency numbers meaningful for a guidance loop, and it lets closed-loop guidance refine Δt as time-to-go shrinks.

## 2. Losslessness verification (no pointing constraint)

**Reproduction of the paper.** At N = 50 the nominal Section V case (glide slope on) gives tf = 79.2 s and 403.6 kg; the paper's Fig. 7 reports 81 s and 399.5 kg with an unstated Δt. At N = 320 this repo gets 398.95 kg (`tests/test_problem.py` asserts the N = 200 case is within 1 kg).

**The paper's example has no fuel margin.** It carries 400 kg. Algorithm 1 at N = 50 needs 403.63 kg and classifies the case as **insufficient fuel**; it first becomes valid at N = 160, where the p99 solve time is 57.1 ms, over the 50 ms budget. This is why the Monte Carlo uses a nearby state with margin (section 6) and also flies the paper's state as given.

| N | Section V fuel (kg) | Section V class | MC-state fuel (kg) | p50 ms | p99 ms |
|--:|--:|:--|--:|--:|--:|
| 10 | 429.92 | insufficient fuel | 337.42 | 2.3 | 2.8 |
| 15 | 417.59 | insufficient fuel | 329.35 | 3.4 | 4.2 |
| 20 | 412.78 | insufficient fuel | 325.79 | 4.7 | 5.5 |
| 30 | 407.41 | insufficient fuel | 322.28 | 7.7 | 8.2 |
| 40 | 404.98 | insufficient fuel | 320.60 | 10.5 | 12.2 |
| 50 | 403.63 | insufficient fuel | 319.62 | 12.9 | 14.8 |
| 60 | 402.68 | insufficient fuel | 318.97 | 15.7 | 19.0 |
| 80 | 401.52 | insufficient fuel | 318.17 | 21.8 | 27.3 |
| 120 | 400.36 | insufficient fuel | 317.38 | 34.9 | 40.4 |
| 160 | 399.80 | valid | 316.99 | 50.4 | 57.1 |
| 240 | 399.24 | valid | 316.60 | 80.6 | 86.1 |
| 320 | 398.95 | valid | 316.41 | 98.0 | 111.0 |

Fuel converges monotonically with N. At the chosen N = 50 the gap to N = 320 is 3.22 kg (1.02 %) on the Monte Carlo state and 4.68 kg on the Section V state. N = 50 was chosen over N = 80 (p99 27.3 ms) and N = 120 (p99 40.4 ms) for headroom: a closed-loop flight makes 67 re-solves on average.

**Battery on 500 random feasible instances** (543 draws: 498 valid, 2 insufficient fuel, 43 insufficient thrust; none dropped). `results/lemma1_battery.json`, `figures/lemma1_battery.png`.

| Theory | Test | Result |
|:--|:--|:--|
| Exact ZOH | discrete vs `solve_ivp`, 12 random intervals | < 1e-9 relative (`tests/test_dynamics.py`); ballistic case to 1e-9 m |
| Lemma 1, slack tight | σ_k − ‖u_k‖ at every node, tolerance 1e-4 m/s² | median -1.3e-07, 99th pct 2.3e-06, 99.9th 1.3e-05, max 2.9e-04 m/s². **1 instance with a loose node** (#288, node 0, gap 2.9e-04); re-solved at 1e-10 tolerances it is tight to 5.6e-07, i.e. an interior-point residual, not a losslessness violation |
| Lemma 1, bang-bang | Γ_k on the (34) bounds | 774 nodes at a switch inside an interval, 8 on an arc shorter than one or two intervals, 198 pinned by the final thrust direction (37), 0 on an active state constraint, **0 unexplained** |
| Lemma 2 | exact (32) at every node | holds at all nodes; worst upper margin -0.0044 N (solver tolerance) |
| (32) between nodes | lower bound at the interval end | worst 14.4 N below ρ1: Γ = σm falls inside a ZOH interval, and (51) is imposed at the interval start only |
| (13) tight | fuel booked through σ vs burned through ‖u‖ | max difference 3.7e-04 kg |
| Inter-node | continuous re-propagation, 40 samples per interval | glide slope: max 10.11 m outside the cone, 95th pct 1.63 m; speed bound 0.000 m/s; subsurface 9.6e-14 m |

The paper's (13) is an equality, ṁ = −αΓ; the relaxation is (14). In terms of the actual thrust the mass law is ṁ ≤ −α‖Tc‖, tight exactly when the slack is, so "(13) tight" is the same fact as "slack tight", measured here in kilograms.

**Does Lemma 1 cover active state constraints?** No. The proof's Hamiltonian (16) and costate equations (19)–(20) have no multiplier terms for (7) or (8); with a state constraint active the costate need not have λ1 constant and λ2 affine, which the proof relies on. The classifier above found no intermediate thrust attributable to an active constraint at N = 50, but that is an observation, not a theorem. The citing list in the 2007 PDF includes Harris & Açıkmeşe (2014) on state-constrained lossless convexification; that paper is not in `refs/` and nothing here relies on it.

**Remark 10, unimodality** (`figures/tof_sweep.png`). The nominal fuel-vs-tf curve is unimodal; 39 of 39 feasible random instances are unimodal, and on 39 of 39 Algorithm 1's golden-section result is at least as good as the 80-point grid. The search strategy is valid on everything tested. The remark's second claim, that the optimum sits very close to the minimum feasible tf, does not hold here: feasible from 68.6 s, optimal at 79.5 s.

**Nonconvex comparison** (`results/nonconvex.json`, `figures/nonconvex_compare.png`). 20 instances, N = 15, SLSQP from 5 random starts each on Problem 1 at the convex tf.

- With the same Taylor bounds as Problem 4 (isolates Lemma 1): the convex optimum is at least as good as the best start on **20 of 20**; median relative gap of the best start 4.6e-08; only 60 of 100 starts reached a feasible point. The convex optimum, plugged into Problem 1 with σ = ‖u‖, violates its constraints by at most 1.6e-05. Because Problem 4 is a relaxation, "convex ≤ best start" is guaranteed whenever the convex solve is global; the content of the test is that the convex point is Problem-1 feasible and that no start beats it.
- With the exact bound (32), which is the literal original Problem 1: SLSQP beats the convex optimum on **every one of the 19 instances where a start reached a feasible point** (1 had none), by up to 0.059 % in β. **This criterion is missed as literally stated.** The cause is Lemma 2's conservatism (the Taylor interval is strictly inside the exact one), not a losslessness failure: warm-started from the convex optimum, SLSQP on the exact problem improves β by at most 0.059 %.

**Infeasibility classification** is Algorithm 1's (fuel vs thrust) plus a separate class for initial states that already violate a state constraint and one for solver failure (any non-certificate status). `tests/test_tof.py` exercises each.

## 3. The pointing-constraint gap, and what the 2013 paper adds

Section VI's claim reproduced (`results/pointing.json`, `figures/pointing_example.png`). On the paper's 5 km hover-start example: without pointing, tf = 70.4 s and 293.5 kg (paper: 69 s, 293.6 kg); with (61) exact, tf = 76.6 s and 303.3 kg (paper, with its heuristic: 78 s, 303 kg), and **15 nodes have u ≈ 0 while σ > 0**: the vehicle is told to thrust below ρ1, exactly the mechanism Section VI describes.

Over 200 random high, slow starts (thrust cone about +altitude):

| Cone | Formulation | Instances with material loose slack (gap > 1e-3) | Commanded thrust < 0.99 ρ1 | Pointing violated |
|:--|:--|--:|--:|--:|
| 90° (γ = 0, paper) | no pointing | 5 | 1 | 162 |
| 90° | (61) exact | 68 | 14 | 0 |
| 90° | heuristic (62) | 68 | 14 | 0 |
| 45° | no pointing | 5 | 1 | 200 |
| 45° | (61) exact | 105 | 49 | 0 |
| 45° | heuristic (62) | 103 | 46 | 50 (up to 42.5°) |

Why it happens, in my words: Lemma 1 works because, with no limit on direction, the Hamiltonian-maximizing thrust always points along R2 = λ2/m and uses the whole slack. A pointing cone can exclude every direction with a positive component along R2; then the best the relaxed problem can do is u = 0 and pay σ ≥ ρ1 e^{−z} anyway, which the original problem forbids. The relaxed problem's optimum is also non-unique in these arcs (sideways thrust at ρ1 on the cone edge costs the same σ); the paper's Fig. 11 shows horizontal thrust at about 5 kN for the first seconds, one such tight point, while the interior-point solver here returns the centre of the optimal face, u ≈ 0.

Three findings beyond the paper:

1. **Loose slack without any pointing constraint** (5 of 200; 2 nodes on the paper's example, thrust down to 0.79 ρ1). These are direction reversals: R2 passes through zero at one instant (the one point Lemma 1's proof allows), the optimal thrust flips from down to up, and zero-order hold averages the flip over a whole interval. Lossless in continuous time, not at the nodes.
2. **The heuristic as printed is a no-op at γ = 0**, the paper's own example: κ2 is a norm, so κ2 ≥ γ = 0 always holds and case 1 (exact) is always taken (`tests/test_pointing_nonconvex.py`). The paper's Fig. 11 cannot have come from case 2 of the rule as printed.
3. At 45°, where the heuristic does relax, it does not restore tightness here (material loose 103 vs 105) and it violates the pointing cone, at a median fuel cost of 17.22 kg.

**The 2013 paper (Açıkmeşe, Carson & Blackmore, IEEE TCST) was not available.** `refs/acikmese_carson_blackmore_2013.pdf` is absent and an author-hosted copy could not be fetched. Its theorem and hypotheses are **not stated** in THEORY.md, by the rule that the papers are the specification. That success criterion is missed. What is shown is the gap it was written to close.

## 4. Solve-time tail

N = 50, Clarabel 0.11.1 via CVXPY 1.9.3, Apple M3, 8 GB, macOS; 600 solves at the Algorithm 1 optimum for random initial states (`figures/solve_time_tail.png`).

| per solve (ms) | p50 | p95 | p99 | max |
|:--|--:|--:|--:|--:|
| wall, whole `solve()` call incl. parameter mapping | 11.9 | 18.0 | **20.1** | 25.7 |
| Clarabel-reported | 10.8 | | 19.1 | 24.6 |

- Compile, reported separately: building the problem 1.7 ms, first-solve canonicalization 16.4 ms (median of 5 cold processes). Paid once.
- What DPP buys: 12.7 ms p50 (14.0 p99) re-solving through the cached mapping vs 15.3 ms p50 (23.0 p99) recompiling every call. Modest at this size: the interior-point solve itself dominates.
- Infeasible solves (139 sampled): p50 6.4 ms, p99 22.2 ms.
- A full Algorithm 1 search (14-point grid plus golden section) costs 342 ms p50, 463 ms p99. That is a once-per-flight cost here; the closed loop holds the landing time fixed and only re-times when the fixed-time problem is infeasible.

**p99 per solve is 20.1 ms against the 50 ms target: met.**

## 5. Baselines at fixed mass error

Monte Carlo state, every method on the same plant (`results/baselines.json`). Landing error (m) / touchdown speed (m/s) / fuel (kg):

| ε | Convex MPC | Open loop | Polynomial (saturated) | Polynomial (ideal engine) | Constant T/W |
|:--|:--|:--|:--|:--|:--|
| −5 % | 0.00 / 0.00 / 321.9 | 1.06 / 0.05 / 324.8 | 1,378.23 / 70.99 / 197.8 | 0.01 / 0.15 / 315.6 | 2,554.06 / 30.03 / 145.6 |
| 0 | 0.02 / 0.13 / 334.7 | 687.39 / 17.58 / 354.6 | 1,306.26 / 71.70 / 203.8 | 0.00 / 0.01 / 313.4 | 2,524.50 / 0.05 / 203.6 |
| +5 % | 0.20 / 0.56 / 351.8 | 1,658.96 / 38.38 / 395.6 | 1,225.71 / 72.20 / 210.0 | 0.01 / 0.16 / 311.8 | 2,495.92 / 0.05 / 190.0 |

- **Constant T/W** (gravity turn, T/W = 1.611 calibrated once at ε = 0 so the vehicle stops at zero altitude): the fuel scale for "slow down, ignore where": 181.54 kg on average and a median miss of 2,522.16 m. Under mass error it either stops in mid-air or hits the ground.
- **Polynomial** (a = −6r/tgo² − 4ṙ/tgo − g, given the convex optimum's tgo = 57.9 s): its raw command lies outside [ρ1, ρ2] in 100.0 % of runs, for 36.5 % of the flight on average. With the engine saturating that command it hits the ground every run. With an ideal engine it lands, cheaper than the convex solution, because it uses thrust the engine does not have and cuts through the glide slope. That frequency is the argument for the convex approach.
- **Open loop**: the same optimizer and the same mass handling as the closed loop, solved once. It lands only when the effective mass error is zero (ε = −5 % with the planning margin); otherwise it misses by hundreds of metres. The difference to the closed loop is the value of re-solving.

## 6. Closed loop under ±5 % mass error

Guidance: Problem 4 at N = 50, re-solved every 1 s from the current state with the landing time held fixed (shrinking horizon, Δt = time-to-go / N), stopping re-solves 0.25 s before touchdown. Plant: true point-mass dynamics (1)–(2), RK4, thrust saturated into [ρ1, ρ2], fuel exhaustion modelled. Mass error: the guidance believes m̂ = (1 + ε) m at all times and commands Tc = u m̂, so the vehicle's acceleration is (1 + ε) times what was planned.

**Three design choices came out of failures in development; each is reported, and each is switched off in an ablation on the same mass-error samples (`results/ablation.json`).**

1. **Plan with the heaviest plausible mass.** With the plain estimate, ε < 0 crashes: the guidance thinks the vehicle is lighter, every acceleration falls short by |ε|, and fuel-optimal plans end in a full-thrust burn with no headroom to absorb that (without the margin, on the Monte Carlo state: ε = −5 % ends in ground contact at 6.79 m/s, 1.86 m from the target; ε = +5 % lands at 0.13 m/s; `results/baselines.json`). The guidance therefore plans with m̂(1 + δ), δ = 5.26 % (`config.yaml: plan_mass_margin`), mapping ε ∈ [−5 %, +5 %] to an effective error in [0, +10.5 %], where the vehicle is only ever stronger than planned. Ablation without it, on the first 300 samples: 65.3 % success overall, 29.7 % of the 148 runs with ε < 0.
2. **Tighten the upper thrust bound in the nominal re-solve.** The fixed-time re-solve has zero slack during the final full-thrust burn: a deviation of a few percent makes it infeasible at the planned landing time, the hard problem is re-timed a little later, and repeated re-times stretch the flight until the fuel runs out. The nominal re-solve therefore plans with 0.97 ρ2; the first fallback re-solves at the same landing time with the true ρ2, using the headroom (constraint tightening, as in robust MPC). On the same 1000 mass-error samples: 7 failures with the tightening, 15 without it; the tightening costs 7.6 kg of fuel on average, so it removes 8 of the 15 failures, at that fuel cost.
3. **Fallback order.** When the tightened fixed-time re-solve is infeasible: the true thrust bounds at the same time; then the hard problem re-timed (Algorithm 1 from the current state); only if no tf works, a recovery problem with the glide slope and speed bound as penalized soft constraints; only then the old plan is held. Recursive feasibility counts a re-solve as feasible if any of the first three, all hard-constraint problems, has a solution.

Results, 1000 runs (`results/montecarlo.json`, `figures/montecarlo.png`), mass error ε ~ U[−5 %, +5 %]:

| | Value | Criterion |
|:--|--:|:--|
| Landing error p50 / p95 / max (m) | 0.09 / 0.33 / 0.61 | |
| Within 5 m | 100.0 % | ≥ 95 %: **met** |
| Touchdown speed p50 / p95 / max (m/s) | 0.26 / 0.67 / 13.86 | |
| Below 1 m/s | 99.3 % | ≥ 95 %: **met** |
| Fuel mean / p95 (kg) | 334.61 / 351.08 | 400 on board; 5th-pct remaining 48.9 kg; 7 runs exhausted |
| Recursive feasibility, hard problem at some tf | 99.99 % of 66802 re-solves; 99.3 % of runs | > 99 %: **met** |
| Strict: feasible at the original landing time | 95.48 % of re-solves; 26.0 % of runs | |
| Fallbacks used | 2109 true-ρ2 re-solves, 904 hard re-times, 6 soft recoveries, 1 soft re-times, 0 holds | |

7 of 1000 runs failed a landing criterion, all with ε between -0.52 % and +4.63 %: ε = -0.52 %: 0.61 m, 11.35 m/s, fuel exhausted, flight 93.4 s, 13 hard re-times; ε = +0.18 %: 0.01 m, 12.11 m/s, fuel exhausted, flight 93.6 s, 5 hard re-times; ε = +3.59 %: 0.00 m, 8.67 m/s, fuel exhausted, flight 93.3 s, 8 hard re-times; ε = +3.61 %: 0.02 m, 13.86 m/s, fuel exhausted, flight 95.3 s, 11 hard re-times; ε = +4.42 %: 0.07 m, 12.21 m/s, fuel exhausted, flight 94.5 s, 10 hard re-times; ε = +4.61 %: 0.03 m, 13.76 m/s, fuel exhausted, flight 95.1 s, 5 hard re-times; ε = +4.63 %: 0.32 m, 12.75 m/s, fuel exhausted, flight 94.4 s, 8 hard re-times.

Every failure is the same mechanism: a run of hard re-times, each pushing the landing a little later, stretches the flight until the tank is empty. Problem 4 does not constrain fuel (Remark 8: fuel is checked after the solve), and under this error model the guidance's own fuel estimate is off by the same ε, so it cannot see the tank running dry. Adding z_N ≥ ln m_dry to the guidance problem would not fix that without a separate fuel measurement; it is listed under limitations.

**The paper's own Section V state, flown the same way** (same 1000 ε samples): within 5 m 45.5 %, below 1 m/s 0.0 %, 1000 runs ran out of fuel. With 400 kg on board and a fuel-optimal need of about 400 kg before any margin, this state is not landable under mass error at all; that is a property of the scenario, not of the guidance.

## 7. The landable envelope

Initial downrange position × initial horizontal velocity, 17 × 25 grid, altitude 1500 m and vertical velocity −75 m/s from the Section V state (`figures/envelope.png`, `results/envelope.json`). Left: perfect mass knowledge (no margin). Right: landable only if the closed loop succeeds at both corners ε = −5 % and ε = +5 %, with the planning margin.

- Perfect mass knowledge: 353 landable cells; 0 insufficient thrust, 72 insufficient fuel, 0 start outside the glide slope, 0 feasible but failed in flight.
- With ±5 % mass error: 306 landable cells. 47 cells are lost: 44 to fuel, 3 to in-flight failure.

**A ±5 % mass error shrinks the landable region from 353 to 306 grid cells, a 13.3 % loss.**

## 8. Success criteria

| Criterion | Result |
|:--|:--|
| Discretization matches continuous propagation to 1e-9 | met |
| Slack tight at every node on 500/500 instances, or every violation characterized | met with one characterized exception (instance #288, solver residual) |
| Every solution satisfies exact (32) | met at every node; between nodes the lower bound dips by up to 14.4 N below ρ1 (characterized) |
| Convex ≥ best of 5 nonconvex starts on 20 instances | met on 20/20 against the Taylor-bounded Problem 1; **missed** on 19/20 against exact (32), by ≤ 0.059 % (Lemma 2 conservatism) |
| Fuel converges with N; gap stated | met; 3.22 kg (1.02 %) at N = 50 vs N = 320 |
| p99 per-solve < 50 ms, compile separate | met: 20.1 ms; compile 16.4 ms |
| 1000 runs, ±5 %: ≥ 95 % within 5 m | met (100.0 %) |
| ≥ 95 % below 1 m/s | met (99.3 %) |
| Recursive feasibility > 99 % | met (99.99 % of re-solves; strict fixed-time 95.48 %) |
| Envelope with and without mass error | produced |
| Pointing: loose fraction, Section VI heuristic, 2013 hypotheses | first two done; **2013 hypotheses missed** (paper not available) |

## 9. Choices not taken from the papers

All marked `NOT IN PAPER` in config.yaml: the speed bound V̂ = 150 m/s (never active in these scenarios); N = 50 and Δt = tf/N (the paper gives no Δt); the solver; the instance distributions; the Monte Carlo state (Section V position, horizontal velocity halved, because the Section V state has no fuel margin); the 1 s re-solve period and the 0.25 s cutoff (ablation with a 1 s cutoff: touchdown speed p95 1.17 m/s and 85.0 % below 1 m/s, against 0.67 m/s and 99.3 % with 0.25 s); the planning mass margin; the 0.97 ρ2 tightening; the fallback chain; the envelope grid. The Section VI initial velocity is taken as zero (text and Fig. 10) rather than the Fig. 11 caption's 20 m/s, which contradicts that figure's own velocity panel.

## 10. Limitations

- 3-DoF point mass: no attitude dynamics, so thrust direction changes instantly (including the full reversals in section 3); no engine lag or throttle-rate limit.
- Constant gravity, flat surface, no aerodynamics, no winds.
- Perfect navigation except mass; mass error is a single persistent multiplicative bias, not an estimator with dynamics or noise.
- The thrust bound (32) is enforced at nodes; inside an interval the commanded thrust can dip below ρ1 by a fraction of a percent, which the plant then saturates.
- One vehicle, one altitude for the envelope; envelope corners at ±5 % stand in for the whole interval.
- Timings are on a laptop CPU with a general-purpose solver through CVXPY; a flight computer with a custom solver would differ in both directions.
- Fuel is not a constraint of the guidance problem (Remark 8), and the guidance has no fuel measurement independent of its mass belief; all closed-loop failures are fuel exhaustion.
- The 2013 theorem is not implemented or checked.
