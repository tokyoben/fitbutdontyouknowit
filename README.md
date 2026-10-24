# quadsmooth

Rates curve and volatility fitting in Rust, with no dependencies:

* a quadratic **smooth-forward** interest rate curve fitter for a dual-curve
  setup (separate projection and discount curves), solved jointly;
* a **shifted-SABR swaption volatility cube** fitter on top of it, with
  forwards taken from the fitted curve, a smoothness penalty across
  (expiry × tenor) nodes and a butterfly-arbitrage check.

Both use a damped Newton method with the **exact analytic Hessian** — from the
linear-in-parameters forward curve in one case, and from second-order
forward-mode automatic differentiation of the SABR formula in the other — so
convergence takes a handful of iterations and market-quote sensitivities fall
out of the Hessian without bump-and-refit.

```
cargo test
cargo run --example dual_curve     # synthetic OIS + 3M market
cargo run --example jpy_2y2y       # JPY 2y2y from TONA OIS benchmarks
cargo run --example swaption_cube  # SABR cube on the TONA curve, 2y2y swaptions
```

## Background (the non-quant version)

### What a curve is for

An interest rate curve answers one question: *what is a yen (or dollar) paid
on some future date worth today?* That number is the **discount factor**
`P(t)`. Everything else — zero rates, par swap rates, forward rates — is just
a different way of writing the same information. We cannot observe `P(t)` for
every date; we observe a few dozen traded prices (deposits, FRAs, swaps) and
have to *infer* a whole continuum of discount factors that is consistent with
them. That inference is what a curve fitter does.

### Projection rates vs. discount rates

A floating-rate swap leg involves two distinct uses of a curve:

* **Projection**: *what will the floating rate fix at?* A 3M IBOR or 6M TIBOR
  coupon is set by a panel-bank benchmark, so the expected coupon depends on
  where that benchmark is expected to be — the **projection curve**.
* **Discounting**: *what is that coupon worth today?* Swaps are collateralised
  and the collateral earns the overnight rate, so cash flows are discounted at
  the overnight (OIS: SOFR, €STR, TONA…) rate — the **discount curve**.

Before 2008 these were treated as one curve: banks funded at IBOR, so IBOR
was both the fixing and the discount rate. The crisis blew the IBOR–OIS
spread out from a couple of basis points to over 300, and the market moved
to **dual-curve** (multi-curve) pricing. The projection curve is now
typically *above* the discount curve by a tenor basis that reflects bank
credit and liquidity premia in the term benchmark.

The two curves cannot be built independently, because an IRS quote depends
on *both*: its floating coupons come from the projection curve and are
discounted on the OIS curve. This library therefore fits them **jointly** —
one optimisation over the stacked parameter vector — so the IRS repricing
condition is satisfied exactly with the correct discounting, rather than
bootstrapping the projection curve against a discount curve that was
built earlier and assumed fixed.

The two curves only differ because the floating leg and the collateral
referenced *different* indices: the coupon fixed off an unsecured term rate
(LIBOR, TIBOR) while the collateral earned the overnight rate. Where a market
has migrated fully to OIS (USD SOFR, GBP SONIA, JPY TONA for most new swaps)
the floating leg pays compounded overnight *and* the collateral earns that
same overnight rate, so there is no basis instrument left to separate the two
curves — they coincide, and the fitter falls back to a single curve
automatically. Markets where a term benchmark survived (EUR with EURIBOR,
legacy JPY TIBOR books, anything on Term SOFR / TORF) remain genuinely
multi-curve.

### Why smooth forwards matter

The **instantaneous forward rate** `f(t)` is the market's implied overnight
rate for a single future day. It is the derivative of the (log) discount
curve, so it magnifies every kink in it. Common interpolation choices
(linear on zero rates, log-linear on discount factors) reprice the
benchmarks perfectly but produce a *sawtooth* forward curve: flat steps with
jumps at each knot, or spikes where the zero curve bends.

That matters for derivatives in three ways:

1. **Forward-starting trades are priced off the forwards, not the par
   rates.** A 2y2y swap, a forward rate agreement, a caplet or a swaption
   strike all depend on `P(t₁)/P(t₂)` for dates *between* benchmark
   maturities. A sawtooth forward curve gives economically absurd answers —
   e.g. the 4y6m–5y forward 50bp above the 4y–4y6m forward purely because of
   where the knots sit — and traders will arbitrage anyone who quotes off it.
2. **Risk is cleaner.** With a smooth curve, a 1bp bump to one benchmark
   moves nearby forwards in a localised, well-behaved way. With a jagged
   curve, bumping the 5y quote can throw wild, oscillating deltas onto the 4y
   and 7y buckets, and P&L explain becomes noise.
3. **Options need the forward's distribution.** Cap/floor and swaption
   models are calibrated to forward rates; a discontinuous forward curve
   feeds them discontinuous inputs and produces implausible implied
   volatility surfaces.

The quadratic smooth-forward approach here makes `f(t)` piecewise quadratic
with continuous level *and* slope, then picks the curve with minimum total
curvature (`∫ f''²`) among those that reprice the market. You get a forward
curve that looks like something a central bank path could actually produce,
while still hitting every benchmark to a tiny fraction of a basis point.

### From curves to volatility

A curve tells you the *expected* path of rates. Options on rates — swaptions,
caps, floors — need the *uncertainty* around that path, which the market
expresses as an implied volatility for each option. Swaption quotes come on a
grid of option expiry × underlying swap tenor, each with a few strikes, and
the resulting **volatility cube** is as central to a rates desk as the curve
itself: it prices every option book, drives hedging, and is the calibration
target for the term-structure models used on exotics.

The difficulty is the same as for curves, only worse: a few hundred noisy,
sparse quotes have to be turned into a smooth, arbitrage-free surface that
gives a sensible volatility for *any* expiry, tenor and strike — including
nodes the brokers didn't quote that day. The market's standard answer is the
SABR model: each quoted smile is summarised by three parameters, and those
parameters are then smoothed across the grid. The cube fitter here does
exactly that in one global optimisation, with the forward swap rate at every
node taken from the smooth-forward curve so that curve and cube are mutually
consistent.

### Why bother with an exact Hessian

Fitting is an optimisation: find the parameters that minimise the repricing
error plus a smoothness penalty. Most curve builders use a bootstrap or a
first-order (Jacobian-only) solver. Because the forward curve here is linear
in its parameters, every instrument price is a sum of exponentials, and the
*second* derivatives come almost for free. That buys:

* **Fast, stable convergence** — true Newton steps, no line-search thrash
  near the solution, no Gauss–Newton approximation.
* **Analytic risk** — the sensitivities of every curve parameter to every
  market quote fall out of `H⁻¹` via the implicit function theorem, so
  bucketed deltas of any trade are available without bumping and refitting.
  The JPY example shows a 2y2y swap's risk resolving to −1.05 × 2y and
  +2.05 × 4y, as theory says it should.
* **The same solver for the cube** — the SABR objective supplies its exact
  Hessian through automatic differentiation, so a 24-node, 72-parameter cube
  converges in ~13 iterations rather than hundreds of Levenberg–Marquardt
  steps.

## Model

Each curve's instantaneous forward is piecewise quadratic between knots and
C¹ (continuous level and slope), flat beyond the last knot:

```
f(t) = a_i + b_i (t − t_i) + c_i (t − t_i)²       t_i ≤ t < t_{i+1}
a_{i+1} = a_i + b_i h_i + c_i h_i²,   b_{i+1} = b_i + 2 c_i h_i
```

The free parameters are `θ = (a_0, b_0, c_0, …, c_{n−1})`, and every
quantity — `f(t)`, `f'(t)`, `∫₀ᵗ f` — is linear in `θ`. Hence a discount
factor is `P(t) = exp(−g(t)·θ)` and a forward ratio `P(t₁)/P(t₂)` is also an
exponential of a linear form. Every instrument's residual therefore has the
shape

```
r(θ) = Σ_k coeff_k · exp(v_k · θ)
∇r   = Σ_k coeff_k e^{v_k·θ} v_k
∇²r  = Σ_k coeff_k e^{v_k·θ} v_k v_kᵀ
```

over the stacked parameter vector `[θ_discount ‖ θ_projection]`, which gives
the exact gradient and Hessian of the whole objective for free.

## Fitting

`DualCurveFitter` minimises

```
F(θ) = ½ Σ_j w_j r_j(θ)²  +  ½ Σ_curves [ λ_slope ∫ f'² + λ_curv ∫ f''² ]
```

with `r_j` normalised to rate units (PV divided by accrual / fixed-leg
annuity). The smoothness penalties are exact quadratic forms in `θ`
(`∫ f''² = Σ 4 h_i c_i²` for piecewise quadratics), so the Hessian

```
∇²F = Σ_j w_j (∇r_j ∇r_jᵀ + r_j ∇²r_j) + Q
```

is assembled analytically each iteration. Newton steps use Cholesky with
Levenberg–Marquardt damping as a fallback and an Armijo line search; typical
markets converge in a handful of iterations to ~1e-4 bp repricing error with
the default `λ_curv = 1e-8` (smaller λ → tighter repricing, e.g. 1e-12 gives
< 1e-6 bp).

Knots are placed at the instrument maturities of each curve: OIS define the
discount curve, deposits/FRAs/IRS on the projection curve define the
projection curve. If no projection instruments are given the problem is
solved as a single curve.

## Instruments

| Constructor | Pricing |
|---|---|
| `Instrument::deposit(curve, T, r)` / `fra(curve, t₁, t₂, r)` | `P(t₁)/P(t₂) − 1 = r τ` on `curve` |
| `Instrument::ois(T, K, freq)` | fixed vs. floating, both legs on the discount curve |
| `Instrument::irs(T, K, fixed_freq, float_freq)` | fixed vs. floating projected on the projection curve, discounted on the discount curve |

Schedules are rolled back from maturity (front stub); arbitrary schedules
can be supplied through `Instrument::Swap` and `Schedule::from_times`.

## Example

```rust
use quadsmooth::{CurveId, DualCurveFitter, Instrument};

let fitter = DualCurveFitter::new()
    .add(Instrument::ois(1.0, 0.030, 1)?)
    .add(Instrument::ois(5.0, 0.034, 1)?)
    .add(Instrument::ois(10.0, 0.036, 1)?)
    .add(Instrument::deposit(CurveId::Projection, 0.25, 0.033)?)
    .add(Instrument::irs(2.0, 0.0345, 1, 4)?)
    .add(Instrument::irs(5.0, 0.0365, 1, 4)?)
    .add(Instrument::irs(10.0, 0.0385, 1, 4)?);

let fit = fitter.fit()?;
println!("{} iterations, max |residual| = {:e}",
         fit.report.iterations, fit.report.max_abs_residual());

let df   = fit.curves.discount_factor(CurveId::Discount, 7.0)?;
let f3m  = fit.curves.simple_forward(CurveId::Projection, 4.0, 4.25)?;
let fwd  = fit.curves.forward(CurveId::Projection, 6.3)?;   // instantaneous
let proj = fit.curves.projection();                         // standalone QuadSmoothCurve

// Hessian-based risk: ∂θ/∂quote_j = −H⁻¹ w_j (∂r_j/∂q_j) ∇r_j
let weights = vec![1.0; fitter.instruments().len()];
let dtheta_dq = fit.quote_sensitivities(fitter.instruments(), &weights)?;
```

`fit.report` also exposes the residuals, the Jacobian `∇r_j` and the full
Hessian at the solution.

## Swaption volatility cube

The same machinery — exact Hessian, penalised global fit, shared Newton
solver — is applied to a **shifted-SABR swaption cube**: one smile per
(expiry, tenor) node, `β` and shift fixed as market conventions, forwards
sourced from the fitted curve.

### Smile model

Per node the normal (Bachelier) volatility is the Hagan et al. (2002)
expansion with Obloj's refinement: the integral `∫_K^F dx/x^β` that drives
`ζ` and the prefactor is kept exact rather than approximated by
`(FK)^{(1−β)/2} ln(F/K)`, which improves the wings and reduces exactly to the
standard normal-SABR formula at `β = 0`. Rates are shifted (`F + s`, `K + s`)
so strikes below zero are admissible, as JPY/EUR/CHF conventions require.

### Fitting across the cube

Parameters are fitted in unconstrained coordinates
`α = e^a, ρ = tanh r, ν = e^n` and the objective is

```
F = ½ Σ_quotes w (σ_model − σ_mkt)² [bp²]
  + ½ Σ_{p ∈ {a,r,n}} λ_p Σ_{adjacent nodes} (p_i − p_j)²
  + ½ ridge · |θ − θ₀|²
```

The smile formula is evaluated on a three-variable **hyper-dual number**
(`src/dual.rs`: value + gradient + Hessian propagated through every
arithmetic op), so each quote contributes its exact 3×3 Hessian block with no
hand-derived derivatives and no finite differences. A 24-node, 154-quote cube
converges in ~13 Newton iterations.

The cross-node penalty is what makes this a cube fitter rather than 24
independent smile fits: nodes missing from the broker run are filled in from
their neighbours, nodes with only an ATM quote inherit skew (`ρ, ν`) from
neighbours, and noisy quotes are smoothed rather than chased. Set the `λ`s
to zero to recover independent per-node calibration (the tests check that
this reproduces a known cube's parameters exactly).

### What you get back

* `SabrCube::node(i, j)` — fitted `SabrParams` at each grid node
* `params_at(expiry, tenor)` / `vol_at(...)` — bilinear interpolation of the
  transformed parameters in `(ln expiry, ln tenor)` for off-grid swaptions
* `price(...)` — Bachelier swaption value given forward, strike and annuity
* `arbitrage_report(range, step)` — butterfly check: implied density
  `∂²C/∂K²` scanned across strikes at every node
* `CubeReport` — residuals in bp, RMSE, iterations, and the exact Hessian over
  all node parameters (the basis for parameter-to-quote sensitivities, exactly
  as in the curve fitter)

```rust
use quadsmooth::{CubeConfig, CurveId, SabrCube, Strike, SwaptionGrid, VolQuote, OptionType};

// forwards from the fitted TONA curve: annual/annual forward-starting swaps
let grid = SwaptionGrid::from_curve(vec![1.0, 2.0, 5.0, 10.0], vec![2.0, 5.0, 10.0],
                                    &fit.curves, 1, 1, CurveId::Discount)?;
let quotes = vec![
    VolQuote::atm(2.0, 2.0, 0.0070),             // 70bp normal vol
    VolQuote::offset_bp(2.0, 2.0, -50.0, 0.0071),
    VolQuote::offset_bp(2.0, 2.0,  50.0, 0.0072),
    // …
];
let cfg = CubeConfig { beta: 0.5, shift: 0.02, ..CubeConfig::default() };
let cube = SabrCube::fit(grid, &quotes, &cfg)?.cube;

let vol  = cube.vol_at(2.0, 2.0, fwd, fwd + 0.0050)?;                 // +50bp 2y2y
let pv   = cube.price(OptionType::Call, 2.0, 2.0, fwd, fwd, annuity)?; // ATM payer
assert!(cube.arbitrage_report(0.03, 0.0005).is_free());
```

## Layout

- `src/curve.rs` — `QuadForwardBasis` (linear basis of the C¹ quadratic
  forward), `QuadSmoothCurve`, `DualBasis` (stacked two-curve parameter space)
- `src/instruments.rs` — `Instrument`, `Schedule`, expansion into exponential terms
- `src/fitter.rs` — objective, exact Hessian, damped Newton, `DualCurve`, quote sensitivities
- `src/linalg.rs` — small dense matrix type with Cholesky
- `src/newton.rs` — damped Newton minimiser shared by the curve and cube fitters
- `src/dual.rs` — second-order forward-mode AD (`Dual2`): value, gradient, Hessian in 3 variables
- `src/sabr.rs` — shifted SABR normal-vol smile over `Dual2`, Bachelier pricing, normal CDF
- `src/cube.rs` — `SwaptionGrid`, `VolQuote`, penalised cube fit, interpolation, arbitrage check
- `examples/dual_curve.rs` — full OIS + 3M market fit with printed forwards
- `examples/jpy_2y2y.rs` — JPY 2y2y forward swap from TONA OIS benchmarks, with Hessian-based risk
- `examples/swaption_cube.rs` — SABR cube on the TONA curve with missing nodes and noisy quotes, 2y2y swaption prices
- `tests/dual_curve.rs` — repricing, smoothness, single-curve mode, Hessian risk vs. refit
