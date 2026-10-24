//! Joint dual-curve fitting with a damped Newton method and exact Hessian.
//!
//! The objective minimised is
//!
//! ```text
//! F(θ) = ½ Σ_j w_j r_j(θ)²  +  ½ Σ_curves [ λ_slope ∫ f'(t)² dt + λ_curv ∫ f''(t)² dt ]
//! ```
//!
//! where `r_j` is the normalised pricing residual of instrument `j` (see
//! [`Instrument::terms`]) and `θ = [θ_discount ‖ θ_projection]`. The
//! residuals are sums of exponentials of linear forms in `θ`, so `∇r_j` and
//! `∇²r_j` are exact, and the penalty is a fixed quadratic form, so the full
//! Hessian
//!
//! ```text
//! ∇²F = Σ_j w_j (∇r_j ∇r_jᵀ + r_j ∇²r_j) + Q
//! ```
//!
//! is assembled analytically on every iteration (no finite differences, no
//! Gauss–Newton truncation).

use crate::curve::{CurveId, DualBasis, QuadForwardBasis, QuadSmoothCurve};
use crate::instruments::{ExpTerm, Instrument};
use crate::linalg::{axpy, dot, norm_inf, Matrix};
use crate::Error;

/// Solver and regularisation settings.
#[derive(Clone, Debug, PartialEq)]
pub struct FitConfig {
    /// Weight `λ_curv` on `∫ f''(t)² dt` for each curve. The minimiser of
    /// this penalty alone is a linear forward, so a small positive value
    /// selects the smoothest curve among those that reprice the market. The
    /// repricing error scales roughly linearly with it.
    pub lambda_curvature: f64,
    /// Weight `λ_slope` on `∫ f'(t)² dt` for each curve (pulls the forward
    /// towards flat). Zero by default.
    pub lambda_slope: f64,
    /// Maximum number of Newton iterations.
    pub max_iterations: usize,
    /// Convergence when the infinity norm of the gradient falls below this.
    pub gradient_tol: f64,
    /// Convergence when the infinity norm of the accepted step falls below this.
    pub step_tol: f64,
    /// Starting instantaneous forward for both curves. When `None`, the mean
    /// quote of each curve's instruments is used.
    pub initial_forward: Option<f64>,
}

impl Default for FitConfig {
    fn default() -> Self {
        FitConfig {
            lambda_curvature: 1e-8,
            lambda_slope: 0.0,
            max_iterations: 100,
            gradient_tol: 1e-13,
            step_tol: 1e-13,
            initial_forward: None,
        }
    }
}

/// Collects instruments and weights, then fits both curves jointly.
#[derive(Clone, Debug, Default)]
pub struct DualCurveFitter {
    instruments: Vec<Instrument>,
    weights: Vec<f64>,
    config: FitConfig,
}

impl DualCurveFitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(mut self, config: FitConfig) -> Self {
        self.config = config;
        self
    }

    /// Adds an instrument with unit weight.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, instrument: Instrument) -> Self {
        self.add_weighted(instrument, 1.0)
    }

    /// Adds an instrument with least-squares weight `weight`.
    pub fn add_weighted(mut self, instrument: Instrument, weight: f64) -> Self {
        self.instruments.push(instrument);
        self.weights.push(weight);
        self
    }

    pub fn instruments(&self) -> &[Instrument] {
        &self.instruments
    }

    pub fn config(&self) -> &FitConfig {
        &self.config
    }

    /// Knots for `curve`: the sorted, de-duplicated maturities of the
    /// instruments that build it.
    fn knots_for(&self, curve: CurveId) -> Vec<f64> {
        let mut knots: Vec<f64> = self
            .instruments
            .iter()
            .filter(|i| i.knot_curve() == curve)
            .map(|i| i.maturity())
            .collect();
        knots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        knots.dedup_by(|a, b| (*a - *b).abs() <= 1e-10);
        knots
    }

    fn mean_quote(&self, curve: CurveId) -> Option<f64> {
        let quotes: Vec<f64> = self
            .instruments
            .iter()
            .filter(|i| i.knot_curve() == curve)
            .map(|i| i.quote())
            .collect();
        if quotes.is_empty() {
            None
        } else {
            Some(quotes.iter().sum::<f64>() / quotes.len() as f64)
        }
    }

    /// Runs the fit.
    ///
    /// If no instrument builds the projection curve, the problem is solved as
    /// a single curve and the projection curve is set equal to the discount
    /// curve.
    pub fn fit(&self) -> Result<FitResult, Error> {
        if self.instruments.is_empty() {
            return Err(Error::NoInstruments(CurveId::Discount));
        }
        for &w in &self.weights {
            if !(w > 0.0) || !w.is_finite() {
                return Err(Error::InvalidInstrument(format!(
                    "weights must be positive and finite, got {w}"
                )));
            }
        }

        let disc_knots = self.knots_for(CurveId::Discount);
        if disc_knots.is_empty() {
            return Err(Error::NoInstruments(CurveId::Discount));
        }
        let proj_knots = self.knots_for(CurveId::Projection);
        let single_curve = proj_knots.is_empty();

        let disc_basis = QuadForwardBasis::new(&disc_knots)?;
        let proj_basis = if single_curve {
            disc_basis.clone()
        } else {
            QuadForwardBasis::new(&proj_knots)?
        };
        let basis = DualBasis::new(disc_basis, proj_basis);
        let dim = basis.dim();
        let n_active = if single_curve {
            basis.basis(CurveId::Discount).dim()
        } else {
            dim
        };

        let cfg = &self.config;
        let penalty = assemble_penalty(&basis, cfg.lambda_slope, cfg.lambda_curvature);

        let terms = self
            .instruments
            .iter()
            .map(|inst| inst.terms(&basis))
            .collect::<Result<Vec<_>, _>>()?;

        let objective = Objective {
            terms,
            weights: self.weights.clone(),
            penalty,
            dim,
        };

        // Starting point: flat forwards at the mean quote of each curve.
        let mut theta = vec![0.0; dim];
        let disc_init = cfg
            .initial_forward
            .or_else(|| self.mean_quote(CurveId::Discount))
            .unwrap_or(0.0);
        let proj_init = cfg
            .initial_forward
            .or_else(|| self.mean_quote(CurveId::Projection))
            .unwrap_or(disc_init);
        theta[basis.offset(CurveId::Discount)] = disc_init;
        theta[basis.offset(CurveId::Projection)] = proj_init;

        let (theta, mut eval, iterations, converged) = newton(&objective, theta, n_active, cfg)?;

        let mut theta = theta;
        if single_curve {
            let n_d = basis.basis(CurveId::Discount).dim();
            let (d, p) = theta.split_at_mut(n_d);
            p.copy_from_slice(d);
            eval = objective.evaluate(&theta);
        }

        let curves = DualCurve { basis, theta };
        let report = FitReport {
            iterations,
            converged,
            objective: eval.value,
            gradient_norm: norm_inf(&eval.gradient[..n_active]),
            residuals: eval.residuals,
            jacobian: eval.jacobian,
            hessian: eval.hessian,
            single_curve,
        };
        Ok(FitResult { curves, report })
    }
}

/// Block-diagonal smoothness penalty over the stacked parameter space.
fn assemble_penalty(basis: &DualBasis, lambda_slope: f64, lambda_curvature: f64) -> Matrix {
    let mut penalty = Matrix::zeros(basis.dim());
    for curve in [CurveId::Discount, CurveId::Projection] {
        let q = basis
            .basis(curve)
            .penalty_matrix(lambda_slope, lambda_curvature);
        let off = basis.offset(curve);
        for i in 0..q.dim() {
            for j in 0..q.dim() {
                penalty.add_to(off + i, off + j, q.get(i, j));
            }
        }
    }
    penalty
}

/// The penalised least-squares objective in exponential-term form.
struct Objective {
    terms: Vec<Vec<ExpTerm>>,
    weights: Vec<f64>,
    penalty: Matrix,
    dim: usize,
}

struct Evaluation {
    value: f64,
    gradient: Vec<f64>,
    hessian: Matrix,
    residuals: Vec<f64>,
    /// `jacobian[j] = ∇r_j`
    jacobian: Vec<Vec<f64>>,
}

impl Objective {
    fn residual(&self, j: usize, theta: &[f64]) -> f64 {
        self.terms[j]
            .iter()
            .map(|t| t.coeff * dot(&t.v, theta).exp())
            .sum()
    }

    fn value(&self, theta: &[f64]) -> f64 {
        let mut f = 0.0;
        for j in 0..self.terms.len() {
            let r = self.residual(j, theta);
            f += 0.5 * self.weights[j] * r * r;
        }
        f + 0.5 * self.penalty.quadratic_form(theta)
    }

    fn evaluate(&self, theta: &[f64]) -> Evaluation {
        let dim = self.dim;
        let mut value = 0.0;
        let mut gradient = vec![0.0; dim];
        let mut hessian = Matrix::zeros(dim);
        let mut residuals = Vec::with_capacity(self.terms.len());
        let mut jacobian = Vec::with_capacity(self.terms.len());

        for (terms, &w) in self.terms.iter().zip(&self.weights) {
            // Evaluate each term once; r = Σ e_k, ∇r = Σ e_k v_k, ∇²r = Σ e_k v_k v_kᵀ.
            let es: Vec<f64> = terms
                .iter()
                .map(|t| t.coeff * dot(&t.v, theta).exp())
                .collect();
            let r: f64 = es.iter().sum();
            let mut grad_r = vec![0.0; dim];
            for (t, &e) in terms.iter().zip(&es) {
                axpy(e, &t.v, &mut grad_r);
            }

            value += 0.5 * w * r * r;
            axpy(w * r, &grad_r, &mut gradient);
            // Gauss–Newton part …
            hessian.add_scaled_outer(&grad_r, w);
            // … plus the exact second-order term r_j ∇²r_j.
            for (t, &e) in terms.iter().zip(&es) {
                hessian.add_scaled_outer(&t.v, w * r * e);
            }

            residuals.push(r);
            jacobian.push(grad_r);
        }

        value += 0.5 * self.penalty.quadratic_form(theta);
        axpy(1.0, &self.penalty.mul_vec(theta), &mut gradient);
        hessian.add_scaled(&self.penalty, 1.0);

        Evaluation {
            value,
            gradient,
            hessian,
            residuals,
            jacobian,
        }
    }
}

fn reduce(m: &Matrix, n: usize) -> Matrix {
    let mut r = Matrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            r.set(i, j, m.get(i, j));
        }
    }
    r
}

/// Damped Newton iteration over the first `n_active` parameters.
/// Returns `(θ, evaluation at θ, iterations, converged)`.
fn newton(
    objective: &Objective,
    mut theta: Vec<f64>,
    n_active: usize,
    cfg: &FitConfig,
) -> Result<(Vec<f64>, Evaluation, usize, bool), Error> {
    let mut eval = objective.evaluate(&theta);
    if !eval.value.is_finite() {
        return Err(Error::SolverFailed(
            "objective is not finite at the starting point".into(),
        ));
    }
    let mut mu = 0.0_f64; // Levenberg–Marquardt style damping, normally zero.
    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..cfg.max_iterations {
        let g = &eval.gradient[..n_active];
        if norm_inf(g) < cfg.gradient_tol {
            converged = true;
            break;
        }
        iterations += 1;

        let h = reduce(&eval.hessian, n_active);
        let max_diag = (0..n_active).fold(0.0_f64, |m, i| m.max(h.get(i, i).abs()));
        let neg_g: Vec<f64> = g.iter().map(|x| -x).collect();

        // Solve (H + μI) δ = −∇F, increasing μ until H + μI is positive
        // definite and δ is a descent direction.
        let delta = loop {
            let mut hm = h.clone();
            if mu > 0.0 {
                hm.add_diagonal(mu);
            }
            match hm.cholesky_solve(&neg_g) {
                Some(d) if dot(&d, g) < 0.0 && d.iter().all(|x| x.is_finite()) => break d,
                _ => {
                    mu = if mu == 0.0 {
                        1e-12 * (1.0 + max_diag)
                    } else {
                        mu * 10.0
                    };
                    if mu > 1e30 * (1.0 + max_diag) {
                        return Err(Error::SolverFailed(
                            "could not find a positive-definite damped Hessian".into(),
                        ));
                    }
                }
            }
        };
        let slope = dot(&delta, g);

        // Backtracking (Armijo) line search.
        let mut alpha = 1.0;
        let mut accepted = None;
        loop {
            let mut trial = theta.clone();
            axpy(alpha, &delta, &mut trial[..n_active]);
            let f_trial = objective.value(&trial);
            if f_trial.is_finite() && f_trial <= eval.value + 1e-4 * alpha * slope {
                accepted = Some(trial);
                break;
            }
            alpha *= 0.5;
            if alpha < 1e-12 {
                break;
            }
        }

        match accepted {
            Some(trial) => {
                let step = alpha * norm_inf(&delta);
                theta = trial;
                eval = objective.evaluate(&theta);
                mu *= 0.1;
                if mu < 1e-300 {
                    mu = 0.0;
                }
                if step < cfg.step_tol {
                    converged = true;
                    break;
                }
            }
            None => {
                // No decrease along the Newton direction: damp harder and retry.
                mu = if mu == 0.0 {
                    1e-6 * (1.0 + max_diag)
                } else {
                    mu * 100.0
                };
                if mu > 1e30 * (1.0 + max_diag) {
                    break;
                }
            }
        }
    }

    Ok((theta, eval, iterations, converged))
}

/// A jointly fitted discount and projection curve.
#[derive(Clone, Debug, PartialEq)]
pub struct DualCurve {
    basis: DualBasis,
    theta: Vec<f64>,
}

impl DualCurve {
    pub fn basis(&self) -> &DualBasis {
        &self.basis
    }

    /// Stacked parameters `[θ_discount ‖ θ_projection]`.
    pub fn parameters(&self) -> &[f64] {
        &self.theta
    }

    /// Extracts one curve as a standalone [`QuadSmoothCurve`].
    pub fn curve(&self, curve: CurveId) -> QuadSmoothCurve {
        self.basis
            .curve(curve, &self.theta)
            .expect("stacked parameters always match the basis")
    }

    pub fn discount(&self) -> QuadSmoothCurve {
        self.curve(CurveId::Discount)
    }

    pub fn projection(&self) -> QuadSmoothCurve {
        self.curve(CurveId::Projection)
    }

    fn integral(&self, curve: CurveId, t: f64) -> Result<f64, Error> {
        Ok(dot(
            &self.basis.basis(curve).integral_basis(t)?,
            self.basis.block(curve, &self.theta),
        ))
    }

    /// Discount factor on `curve`.
    pub fn discount_factor(&self, curve: CurveId, t: f64) -> Result<f64, Error> {
        Ok((-self.integral(curve, t)?).exp())
    }

    /// Instantaneous forward on `curve`.
    pub fn forward(&self, curve: CurveId, t: f64) -> Result<f64, Error> {
        Ok(dot(
            &self.basis.basis(curve).forward_basis(t)?,
            self.basis.block(curve, &self.theta),
        ))
    }

    /// Continuously compounded zero rate on `curve`.
    pub fn zero_rate(&self, curve: CurveId, t: f64) -> Result<f64, Error> {
        if t == 0.0 {
            return self.forward(curve, 0.0);
        }
        Ok(self.integral(curve, t)? / t)
    }

    /// Simply compounded forward over `[t1, t2]` on `curve`.
    pub fn simple_forward(&self, curve: CurveId, t1: f64, t2: f64) -> Result<f64, Error> {
        if t2 <= t1 {
            return Err(Error::InvalidInstrument(format!(
                "simple forward requires t2 > t1 (got {t1}, {t2})"
            )));
        }
        let i1 = self.integral(curve, t1)?;
        let i2 = self.integral(curve, t2)?;
        Ok(((i2 - i1).exp() - 1.0) / (t2 - t1))
    }

    /// Normalised pricing residual (in rate units) of `instrument` on these curves.
    pub fn residual(&self, instrument: &Instrument) -> Result<f64, Error> {
        Ok(instrument
            .terms(&self.basis)?
            .iter()
            .map(|t| t.coeff * dot(&t.v, &self.theta).exp())
            .sum())
    }

    /// Fixed-leg annuity `Σ τ_i P_d(t_i)` of a swap (`None` for a simple rate).
    fn annuity(&self, instrument: &Instrument) -> Result<Option<f64>, Error> {
        match instrument {
            Instrument::SimpleRate { .. } => Ok(None),
            Instrument::Swap {
                fixed,
                discount_curve,
                ..
            } => {
                let mut a = 0.0;
                for (_, t, tau) in fixed.periods() {
                    a += tau * self.discount_factor(*discount_curve, t)?;
                }
                Ok(Some(a))
            }
        }
    }

    /// The rate that would make `instrument` price to zero on these curves:
    /// the simple forward for a deposit/FRA, the par swap rate for a swap.
    pub fn par_rate(&self, instrument: &Instrument) -> Result<f64, Error> {
        match instrument {
            Instrument::SimpleRate {
                curve, start, end, ..
            } => self.simple_forward(*curve, *start, *end),
            Instrument::Swap {
                float,
                float_curve,
                discount_curve,
                ..
            } => {
                let annuity = self.annuity(instrument)?.unwrap();
                let mut float_pv = 0.0;
                for (s0, s1, tau) in float.periods() {
                    float_pv += tau
                        * self.simple_forward(*float_curve, s0, s1)?
                        * self.discount_factor(*discount_curve, s1)?;
                }
                Ok(float_pv / annuity)
            }
        }
    }

    /// `∂r_j/∂q_j`: how the normalised residual moves with the instrument's
    /// own quote (holding the curves fixed).
    fn residual_quote_derivative(&self, instrument: &Instrument) -> Result<f64, Error> {
        match self.annuity(instrument)? {
            None => Ok(-1.0),
            Some(a) => Ok(a / instrument.scale()),
        }
    }
}

/// Diagnostics from a fit.
#[derive(Clone, Debug, PartialEq)]
pub struct FitReport {
    /// Newton iterations taken.
    pub iterations: usize,
    /// Whether a convergence criterion was met.
    pub converged: bool,
    /// Final objective value `F(θ)`.
    pub objective: f64,
    /// Infinity norm of the final gradient over the active parameters.
    pub gradient_norm: f64,
    /// Final normalised residual per instrument, in rate units.
    pub residuals: Vec<f64>,
    /// `jacobian[j] = ∇r_j` over the stacked parameters.
    pub jacobian: Vec<Vec<f64>>,
    /// Exact Hessian `∇²F` at the solution over the stacked parameters.
    pub hessian: Matrix,
    /// `true` when no projection instruments were given and the projection
    /// curve was set equal to the discount curve.
    pub single_curve: bool,
}

impl FitReport {
    /// Largest absolute repricing error, in rate units.
    pub fn max_abs_residual(&self) -> f64 {
        norm_inf(&self.residuals)
    }
}

/// Fitted curves plus diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct FitResult {
    pub curves: DualCurve,
    pub report: FitReport,
}

impl FitResult {
    /// Sensitivities of the fitted parameters to each instrument quote,
    /// `result[j] = ∂θ/∂q_j`, obtained from the implicit function theorem on
    /// the first-order condition `∇F(θ, q) = 0`:
    ///
    /// ```text
    /// ∂θ/∂q_j = −H⁻¹ · w_j · (∂r_j/∂q_j) · ∇r_j
    /// ```
    ///
    /// (the term `w_j r_j ∂∇r_j/∂q_j` is dropped as `r_j ≈ 0` at the
    /// solution). Chain with `∂P(t)/∂θ = −P(t) g(t)` to get discount-factor or
    /// forward-rate sensitivities to market quotes.
    pub fn quote_sensitivities(
        &self,
        instruments: &[Instrument],
        weights: &[f64],
    ) -> Result<Vec<Vec<f64>>, Error> {
        let dim = self.report.hessian.dim();
        if instruments.len() != self.report.jacobian.len() || weights.len() != instruments.len() {
            return Err(Error::DimensionMismatch {
                expected: self.report.jacobian.len(),
                actual: instruments.len(),
            });
        }
        let mut out = Vec::with_capacity(instruments.len());
        for (j, inst) in instruments.iter().enumerate() {
            let dr_dq = self.curves.residual_quote_derivative(inst)?;
            let rhs: Vec<f64> = self.report.jacobian[j]
                .iter()
                .map(|x| -weights[j] * dr_dq * x)
                .collect();
            debug_assert_eq!(rhs.len(), dim);
            let sol = if self.report.single_curve {
                // Hessian is singular in the (inactive) projection block;
                // solve on the discount block and mirror.
                let n_d = self.curves.basis.basis(CurveId::Discount).dim();
                let h = reduce(&self.report.hessian, n_d);
                let d = h
                    .cholesky_solve(&rhs[..n_d])
                    .ok_or_else(|| Error::SolverFailed("Hessian not positive definite".into()))?;
                let mut full = d.clone();
                full.extend_from_slice(&d);
                full
            } else {
                self.report
                    .hessian
                    .cholesky_solve(&rhs)
                    .ok_or_else(|| Error::SolverFailed("Hessian not positive definite".into()))?
            };
            out.push(sol);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small two-curve objective evaluated away from the optimum, so the
    /// second-order terms `r_j ∇²r_j` are non-negligible.
    fn objective() -> (Objective, Vec<f64>) {
        let basis = DualBasis::new(
            QuadForwardBasis::new(&[1.0, 2.0, 5.0]).unwrap(),
            QuadForwardBasis::new(&[0.5, 2.0, 5.0]).unwrap(),
        );
        let instruments = [
            Instrument::ois(1.0, 0.03, 1).unwrap(),
            Instrument::ois(2.0, 0.031, 2).unwrap(),
            Instrument::ois(5.0, 0.034, 1).unwrap(),
            Instrument::deposit(CurveId::Projection, 0.5, 0.033).unwrap(),
            Instrument::irs(2.0, 0.035, 1, 4).unwrap(),
            Instrument::irs(5.0, 0.038, 1, 2).unwrap(),
        ];
        let terms = instruments
            .iter()
            .map(|i| i.terms(&basis).unwrap())
            .collect();
        let penalty = assemble_penalty(&basis, 1e-3, 1e-2);
        let dim = basis.dim();
        let objective = Objective {
            terms,
            weights: vec![1.0, 2.0, 0.5, 1.0, 1.5, 1.0],
            penalty,
            dim,
        };
        // θ_disc = (a0, b0, c0, c1, c2), θ_proj = (a0, b0, c0, c1, c2)
        let theta = vec![
            0.02, 0.004, 0.003, -0.001, 0.0005, //
            0.045, -0.002, -0.004, 0.002, -0.0003,
        ];
        (objective, theta)
    }

    #[test]
    fn gradient_matches_central_differences() {
        let (obj, theta) = objective();
        let eval = obj.evaluate(&theta);
        let h = 1e-6;
        for k in 0..theta.len() {
            let mut tp = theta.clone();
            let mut tm = theta.clone();
            tp[k] += h;
            tm[k] -= h;
            let fd = (obj.value(&tp) - obj.value(&tm)) / (2.0 * h);
            let an = eval.gradient[k];
            assert!(
                (fd - an).abs() < 1e-8 * (1.0 + an.abs()),
                "k={k}: fd {fd} vs analytic {an}"
            );
        }
    }

    #[test]
    fn hessian_matches_central_differences_of_gradient() {
        let (obj, theta) = objective();
        let eval = obj.evaluate(&theta);
        let h = 1e-6;
        for k in 0..theta.len() {
            let mut tp = theta.clone();
            let mut tm = theta.clone();
            tp[k] += h;
            tm[k] -= h;
            let gp = obj.evaluate(&tp).gradient;
            let gm = obj.evaluate(&tm).gradient;
            for i in 0..theta.len() {
                let fd = (gp[i] - gm[i]) / (2.0 * h);
                let an = eval.hessian.get(i, k);
                assert!(
                    (fd - an).abs() < 1e-6 * (1.0 + an.abs()),
                    "H[{i},{k}]: fd {fd} vs analytic {an}"
                );
                // Symmetry.
                assert!((eval.hessian.get(i, k) - eval.hessian.get(k, i)).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn hessian_is_not_just_gauss_newton() {
        // Away from the optimum the exact Hessian must differ from JᵀWJ + Q.
        let (obj, theta) = objective();
        let eval = obj.evaluate(&theta);
        let mut gn = obj.penalty.clone();
        for (j, w) in obj.weights.iter().enumerate() {
            gn.add_scaled_outer(&eval.jacobian[j], *w);
        }
        let mut diff = 0.0_f64;
        for i in 0..obj.dim {
            for k in 0..obj.dim {
                diff = diff.max((gn.get(i, k) - eval.hessian.get(i, k)).abs());
            }
        }
        assert!(diff > 1e-6, "second-order term vanished: {diff}");
    }
}
