# Theory

Equation numbers are from Açıkmeşe & Ploen, *Convex Programming Approach to Powered Descent Guidance
for Mars Landing*, JGCD 30(5), 2007 (`refs/acikmese_ploen_2007.pdf`). Page numbers are the journal's.
The PDF's text layer drops every Greek letter; all equations below were read from rendered page images.

## 1. Frame and model

Surface-fixed frame with its origin at the target (Fig. 1). The **first coordinate is altitude**,
opposite to gravity, so `g = [-3.7114, 0, 0]` m/s² (59). The vehicle is a point mass:

- `r̈ = g + Tc/m` (1), `ṁ = -α‖Tc‖` (2), with `α = 1/(Isp ge cos φ)`.
- Thrust magnitude must stay in an **annulus** `ρ1 ≤ ‖Tc‖ ≤ ρ2` (3), with `ρi = n Ti cos φ`. The engines
  cannot shut off, so the zero-thrust hole makes this set nonconvex.
- Boundary conditions (4)–(5): fixed initial position, velocity and mass; `r(tf) = ṙ(tf) = 0`.
- Assumption (6): `r0₁ > 0`, i.e. the vehicle starts above the target's horizontal plane.
- State constraints: no subsurface flight (7), and the general second-order cone form (8), which covers
  the speed bound (10) `‖ṙ‖ ≤ V̂` and the glide slope (11) `‖(r₂, r₃)‖ ≤ tan θ̃_alt · r₁`.

## 2. Problem 1 → Problem 2: the slack Γ (lossless relaxation)

**Problem 1** (p.1356) maximizes final mass, i.e. minimizes `∫‖Tc‖dt`, subject to all of the above.

**Problem 2** introduces a scalar slack `Γ(t)` and replaces `‖Tc‖` with `Γ` in the cost and the mass
equation, so `ṁ = -αΓ` (13), and swaps the annulus for

- `‖Tc‖ ≤ Γ` (14) — a second-order cone, convex;
- `ρ1 ≤ Γ ≤ ρ2` (15) — an interval, convex.

Geometrically (Fig. 3) the annulus in `Tc` space is lifted to a cone in `(Tc, Γ)` space. Problem 2's feasible set
strictly contains Problem 1's, because a point inside the hole is allowed if Γ is large enough.

Note: (13) is an **equality** in the paper. What makes this a relaxation is (14). In terms of the actual thrust,
the mass equation becomes `ṁ = -αΓ ≤ -α‖Tc‖`, which holds with equality exactly when the slack is tight.
"(13) is tight" and "the slack is tight" are therefore the same check (`verify/slack.py`).

### Lemma 1 (p.1356–1357): what it guarantees

If `(tf*, Tc*, Γ*)` solves Problem 2, then:

1. `‖Tc*(t)‖ = Γ*(t)` almost everywhere (22). The relaxed optimum is feasible for Problem 1 and therefore
   optimal for it.
2. `‖Tc*(t)‖ ∈ {ρ1, ρ2}` almost everywhere: the optimal thrust magnitude is **bang-bang**, switching on the sign of
   `R12 = R1 + ‖R2‖`.

Proof structure, which matters for the tests:

- Pontryagin with Hamiltonian (16). The costates give `λ̇1 = 0` (19) and `λ̇2 = -λ1` (20), so `λ2` is affine
  in time. The transversality conditions give `λ3(tf) = 0` and `H(tf) = 0`.
- If `λ2 ≡ 0`, then every multiplier is zero, which contradicts nontriviality. So `λ2` vanishes at most at one
  instant. H is linear in `Tc`, so the maximizer points `Tc` along `R2 = λ2/m` and drives `‖Tc‖` up to `Γ`.
  That gives (22).
- For bang-bang, `R12` must not vanish on an interval. When `λ1 ≠ 0` this is shown directly. The
  degenerate case `λ1 = 0` would make the thrust direction constant and perpendicular to `g`, and
  integrating the vertical channel then gives `r0₁ = -ṙ0₁²/(2ĝ) ≤ 0`, which **contradicts (6)**.
  **The initial-altitude hypothesis (6) is used only here**, for the bang-bang claim. Claim 1 (tightness) does not
  use it.

**What the proof does not cover: active state constraints.** Hamiltonian (16) and costate equations
(19)–(20) contain no multiplier terms for (7) or (8). With a state constraint active on an arc, the costate
picks up measure or multiplier terms, `λ1` need not be constant, and `λ2` need not be affine. As written, the
proof covers only arcs where the state constraints are inactive. The paper does not comment on this. The
article's citing list includes Harris & Açıkmeşe (2014), *Lossless convexification of non-convex optimal
control problems for state constrained linear systems*, Automatica 50(9), but that paper is not in `refs/` and
nothing here relies on it. Consequence for the tests: intermediate thrust values on an arc where the glide
slope is active are **not** a contradiction of Lemma 1 (`verify/bangbang.py` classifies them).

### Existence, Theorem 1 (p.1357–1358)

This is Filippov-type existence. If Problem 2 is feasible and `tf ≤ tmax` (bounded by fuel), then the
compactness conditions hold, and the set `Q⁺` (24) is convex because (13)–(15) are linear or SOC. So an optimal
solution **exists whenever a feasible one does**. Infeasibility is a statement about physics (not enough thrust
or fuel), not a numerical accident. This is what lets Algorithm 1 classify failures.

## 3. Change of variables, (25)–(31): remove the 1/m

The dynamics `Tc/m` are nonlinear in (Tc, m). Define

- `σ = Γ/m`, `u = Tc/m` (25). Then `r̈ = u + g` (26) is **linear**.
- `z = ln m` (30). Then `ṁ/m = -ασ` (27) becomes `ż = -ασ` (31), also linear.
- The cost `∫Γ dt` is replaced by `∫σ dt`. This is equivalent because `m(tf) = m0 exp(-α∫σ)` is monotone in
  `∫σ` (p.1358).
- The control constraints become `‖u‖ ≤ σ` (28), still SOC, and `ρ1 e^{-z} ≤ σ ≤ ρ2 e^{-z}` (32).

In (32), the left half is convex, since `e^{-z}` is convex and it sits on the `≤` side. The right half is not
convex.

## 4. Taylor bounds, (33)–(36): convexify (32) conservatively

Expand around `z0(t) = ln(m_wet − αρ2 t)` (35), the lightest the vehicle can possibly be at time t. Then:

- Lower bound: second-order expansion, `μ1[1 − (z−z0) + (z−z0)²/2] ≤ σ`. This is a convex quadratic `≤`
  affine, so it is SOC-representable.
- Upper bound: first-order expansion, `σ ≤ μ2[1 − (z−z0)]`. This is affine.
- Here `μi = ρi e^{-z0}` (33). Together these are (34).
- The bounds `ln(m_wet − αρ2 t) ≤ z ≤ ln(m_wet − αρ1 t)` (36) keep `z ≥ z0`, which the next lemma needs.

**Lemma 2 (p.1359): the approximations are conservative.** For `z ≥ z0`, the cubic remainder of `e^{-z}`
around `z0` is `−e^{-ẑ}(z−z0)³/6 ≤ 0`. So the quadratic expansion **over-estimates** `e^{-z}`, and the
lower bound in (34) is *larger* than the true one. The quadratic remainder of the linear expansion is
`+e^{-ẑ}(z−z0)²/2 ≥ 0`. So the linear expansion **under-estimates** `e^{-z}`, and the upper bound in (34) is
*smaller* than the true one. The approximated interval sits inside the exact one, so a solution of the
approximated problem always satisfies exact (32). Flip either sign and this inclusion breaks. That is why
`verify/exact32.py` checks every solution against (32), never against (34).

Error bounds (38)–(39) show the approximation is under about 2.5 % over the example's 70 s (Fig. 4).

**Problem 3** (p.1359) is Problem 2 in `(u, σ, z)` with (34) and (36).

## 5. Discretization, (40)–(53): Problem 4

Piecewise-constant basis functions (45) mean `u`, `σ` are held constant on `[t_{k}, t_{k+1})`. This is
exactly a zero-order hold on the LTI system (26), (31). With `Ac, Bc` from p.1360:

- `A = expm(Ac Δt)`, `B = ∫₀^Δt expm(Ac s) ds · Bc`. Computed in `dynamics/model.py` from one exponential of
  the augmented generator `[[Ac, Bc], [0, 0]]Δt`. **Euler (`B ≈ Bc Δt`) is wrong here.** It sets the
  position gain of an acceleration to 0 instead of `Δt²/2`. Over 50 intervals that is a position error of order
  `u Δt² N/2`, hundreds of metres, which the optimizer would treat as real and exploit.
- Gravity enters through `B` as a known input (`Λk [g; 0]` in (43)).
- Cost (47)–(48): `Δt Σ σ_k`.
- Constraints at the nodes: (50) `‖u_k‖ ≤ σ_k`, (51) the Taylor bounds, (52) the z box, (53) the state
  constraints. **Constraints hold only at nodes** (p.1360, "the constraints are only satisfied at the temporal
  nodes"). `verify/internode.py` measures what happens between them.

Problem 4 is an SOCP for each N.

## 6. Time of flight, (54)–(58), Algorithm 1, Remark 10

- Conservative bounds (55): `t_min = m_dry‖ṙ0‖/ρ2`, `t_max = m_fuel/(αρ1)`.
- Algorithm 1 (p.1361): line search on `tf` for the minimum of `β(tf)`, with `β = ∞` where Problem 4 is
  infeasible (58). Then classify:
  - fuel used `(1 − e^{−αβ*}) m_wet ≤ m_fuel` → valid;
  - otherwise, if `β* < ∞` → **insufficient fuel**;
  - if `β* = ∞` → **insufficient thrust** ("insufficient thrusters").
- Remark 10: in the authors' simulations `β(tf)` is **unimodal**, and the optimum is near the minimum feasible
  tf. This is an observation, not a theorem, so `tof/` records the sweep and checks it.
- Remark 8: fuel is *not* a constraint inside Problem 4. It is checked afterward, as above.

**Deviation from the paper (argued in REPORT.md).** The paper fixes Δt and searches the integer N. This repo
fixes N at compile time and makes `Δt = tf/N` a cvxpy `Parameter`. `A` and `B` are parameter matrices
(DPP-compliant: parameter × variable), and so are `z0(t_k)`, `μ1`, `μ2`. The line search over `tf` is the same
search the paper does. The difference is that one compiled problem serves every `tf` and every initial
condition. That is what makes the latency numbers honest, and it lets closed-loop guidance shrink Δt as
time-to-go shrinks.

## 7. Section VI: the pointing constraint is **not** lossless (p.1362–1365)

Constraint (60): `vᵀTc/‖Tc‖ ≥ γ = cos θ`. In the new variables this is `vᵀu ≥ γσ` (61), a convex SOC
constraint. The paper states plainly that the convexification **does not carry over**:

- Without pointing, Lemma 1 gives `u* = σ R2/‖R2‖`.
- With (61), if at some t the cone `{ξ : vᵀξ ≥ γ, ‖ξ‖ = 1}` lies entirely in the half-space `ξᵀR2 ≤ 0`, the
  Hamiltonian maximizer is `u = 0`. Since σ is still held at or above `ρ1 e^{-z}` by (32), this is a **loose slack**
  `‖u‖ = 0 < σ`. That violates `ρ1 ≤ ‖Tc‖`. Quoting p.1363: "a convex equivalent of Problem 1 is not given by
  Problem 2 when such thrust direction constraint exists."

**Heuristic (p.1363–1364).** Solve Problem 4 without pointing to get `r̂_k = u_k/‖u_k‖`, an estimate of
`R2/‖R2‖`. Let `κ1,k = vᵀr̂_k`, `κ2,k = ‖v − (vᵀr̂_k) r̂_k‖`. Then:

1. If `κ1,k ≥ 0` or `κ2,k ≥ γ`, enforce (61) exactly at node k.
2. Otherwise, enforce the relaxed `vᵀu_k ≥ κ2,k σ_k` (62), which guarantees that a cone direction with
   `r̂_kᵀξ ≥ 0` exists.

Paper's result: the pointing violation drops from 90° to under 5°, at about 10 kg more fuel (p.1365).
Implemented in `pointing/`, with the loose-slack fraction measured with and without the heuristic.

## 8. Açıkmeşe, Carson & Blackmore (2013): stated from a secondary source

**The 2013 paper itself (IEEE TCST 21(6), 2104–2113, doi 10.1109/tcst.2012.2237346) is still not in
`refs/`.** Two sources about it are:

- `refs/carson_2013_nasa_techbrief.pdf` (NASA Tech Briefs, Feb. 2013, p.25, NPO-47853). It asserts that "a
  relaxation of the thrust-pointing constraint also provides a lossless convexification". It gives no
  theorem and no hypotheses.
- `refs/malyuta_2022_tutorial.pdf` (Malyuta, Reynolds, Szmuk, Lew, Bonalli, Pavone, Açıkmeşe, *Convex
  Optimization for Trajectory Generation*, arXiv:2106.09125v1). It states the pointing result as its
  Theorem 2 (p.8) and cites **[87, 96] together**: [87] is Carson, Açıkmeşe & Blackmore, ACC 2011, and [96]
  is the 2013 paper. It does not say which part comes from which.

Everything below is the tutorial's statement, read from rendered pages. Equation and page numbers in this
section are the tutorial's. **It is not a transcription of [96].**

**The relaxation (p.8, sidebar p.9).** The setting is restricted to ζ = 1 (a running cost) and g0 = g1. The
pointing constraint (9d) is `n̂uᵀu ≥ n̂g g0(u)`. It is relaxed to (10d) `g0(u) ≤ σ` together with (10e)
`n̂uᵀu ≥ σ n̂g`. With `g0 = ‖·‖₂` and `n̂g = cos θmax`, (10e) is exactly this repo's (61), with `v = n̂u` and
`γ = cos θmax`. It is a half-space in (u, σ), so it stays convex even for θmax > 90°.

**So the 2013 relaxation is the same constraint the 2007 paper writes as (61).** What 2013 adds is a set of
hypotheses under which it is lossless. Section 7 shows that the 2007 paper says it is not lossless in
general.

**Theorem 2 (p.8).** "The solution of Problem 10 is globally optimal for Problem 9 if Conditions 1, 2, and 3
hold." Its hypotheses are:

- **Assumption 1 (p.7).** If any part of the terminal state is constrained, ∇ₓb[tf] has full row rank.
- **Assumption 2 (p.7).** The running cost is positive definite: ℓ(z) > 0 for z ≠ 0.
- **Condition 1 (p.7).** "The pair {A(·), B(·)} must be totally controllable." For LTI dynamics this is the
  Kalman rank test.
- **Condition 2 (eq. 8).** `m_LCvx = [∇ₓm[tf]; ∇ₜm[tf] + ζℓ(σ(tf))]` and the columns of
  `B_LCvx = [∇ₓb[tf]ᵀ; ∇ₜb[tf]ᵀ]` must be linearly independent. This is a transversality condition.
- **Condition 3 (p.8).** "Let N ∈ ℝ^{m×(m−1)} be a matrix whose columns span the nullspace of n̂u in (9d).
  The pair {A(·), B(·)N} must be totally controllable."

**What the tutorial adds around Theorem 2. These are not hypotheses of Theorem 2, and none of them is
attributed to [96]:**

- Problem 9 has **no state constraints**. For the glide slope, the tutorial's own rocket example (p.46–47)
  argues that it is only instantaneously active. The argument is a force balance plus bang-bang thrust [45].
  It is the conservative sufficient test (106), for all θ ∈ [π/2 − γp − γgs, π/2 + γp − γgs]:
  - (106a) `ρmin cos θ < m_dry‖g‖ sin γgs`;
  - (106b) `ρmax cos θ > m_wet‖g‖ sin γgs`.
- For general convex state constraints, Theorem 5 (p.14–15) needs them to be "activated at isolated time
  instances".
- For the rotating-planet model (104c)/(105), p.46 says: "Condition 3 holds as long as ω×ê_z ≠ 0, which
  means that the planet does not rotate about the local vertical."
- Every result is for continuous time. The tutorial gives no losslessness guarantee after discretization;
  its evidence is numerical (p.22, p.47).
- p.21 mentions that, for pointing, "the dynamical system can be perturbed by an arbitrarily small amount to
  induce normality". It does not say whether Theorem 2 needs such a perturbation.

**Not available from either source:**

- the proof;
- any bound on θmax beyond the geometry;
- whether tf is free or fixed in [96];
- any hypothesis of [96] that the tutorial does not restate.

### Hypotheses checked for this repo's model

Code: `verify/lcvx_conditions.py`. Tests: `tests/test_lcvx_conditions.py`. Results:
`results/lcvx_conditions.json`. Following p.46, z is treated as a fictitious state, so x = (r, ṙ) ∈ ℝ⁶,
n̂u = v = +altitude, and N spans the two horizontal axes.

- **Condition 1 holds.** A is a double integrator and B = [0; I], so the rank is 6.
- **Condition 2 holds on every solution.** Here m = 0, ℓ(σ) = σ (cost (47)) and b = x(tf), so
  `m_LCvx = [0; σ(tf)]` and `B_LCvx = [I₆; 0]`. These are independent exactly when σ(tf) ≠ 0. The left side
  of (51) is `μ1(1 − δz + δz²/2) ≥ μ1/2 > 0`, so σ(tf) > 0 is guaranteed. The tutorial reaches the same
  test on p.47 (107).
- **Condition 3 fails in the 2007 model.** Equation (1) has no planet rotation. With B N = [0; N], only the
  horizontal channels are reachable, and the vertical double integrator is not: the rank is 4 of 6. The same
  holds for any rotation about the vertical, as p.46 says. With the tutorial's Mars rotation (108c), rank 6
  holds in exact arithmetic. But the smallest eigenvalue of the controllability Gramian over 75 s is at
  round-off level: Condition 3 holds only by a coupling of order |ω| ≈ 1e-4 rad/s.
- **(106) holds for a 45° cone** (both inequalities). **For the paper's 90° cone, (106b) fails**; (106a)
  holds.

**Consequence.** The 2007 Section VI example (γ = 0, no rotation) is outside Theorem 2 on two counts:
Condition 3 and (106b). So the loose slack measured in Phase 4 does not contradict the 2013 result; the
2013 result does not apply to it.

**Does Condition 3 actually help?** `scripts/lcvx_conditions.py` adds rotation to the dynamics, using
(104c) through `dynamics.model.continuous_matrices`. It then re-runs the same 200 pointing instances under
three settings:

1. no rotation (the 2007 model);
2. the tutorial's ω;
3. that ω scaled 100× (NOT IN PAPER, a diagnostic).

The counts are in REPORT.md §3.
