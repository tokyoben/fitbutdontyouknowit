# quadsmooth

A quadratic **smooth-forward** interest rate curve fitter in Rust for a
dual-curve setup (separate projection and discount curves), solved jointly
with a damped Newton method and the **exact analytic Hessian**. No
dependencies.

```
cargo test
cargo run --example dual_curve
```

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

## Layout

- `src/curve.rs` — `QuadForwardBasis` (linear basis of the C¹ quadratic
  forward), `QuadSmoothCurve`, `DualBasis` (stacked two-curve parameter space)
- `src/instruments.rs` — `Instrument`, `Schedule`, expansion into exponential terms
- `src/fitter.rs` — objective, exact Hessian, damped Newton, `DualCurve`, quote sensitivities
- `src/linalg.rs` — small dense matrix type with Cholesky
- `examples/dual_curve.rs` — full OIS + 3M market fit with printed forwards
- `tests/dual_curve.rs` — repricing, smoothness, single-curve mode, Hessian risk vs. refit
