//! Swaption volatility cube: a shifted-SABR smile at every (expiry, tenor)
//! node, fitted globally with a smoothness penalty across nodes.
//!
//! Each node carries `(α, ρ, ν)`; `β` and the shift are conventions fixed for
//! the whole cube. To keep the parameters admissible without constraints the
//! optimiser works in transformed coordinates
//!
//! ```text
//! α = exp(a),   ρ = tanh(r),   ν = exp(n)
//! ```
//!
//! and the objective is
//!
//! ```text
//! F = ½ Σ_quotes w (σ_SABR − σ_mkt)²·(1e4)²              residuals in bp
//!   + ½ Σ_params λ_p Σ_{adjacent nodes} (p_i − p_j)²        smoothness
//!   + ½ ridge · |θ − θ₀|²                                   identifiability
//! ```
//!
//! The SABR formula is evaluated on [`Dual2`] numbers, so every quote
//! contributes its exact 3×3 Hessian block; the penalties are fixed quadratic
//! forms. The resulting exact Hessian drives the shared Newton solver.
//! Nodes with no quotes at all (gaps in the broker grid) are filled in by the
//! smoothness term; nodes with only an ATM quote inherit skew (`ρ`, `ν`) from
//! their neighbours.

use crate::curve::CurveId;
use crate::dual::Dual2;
use crate::fitter::DualCurve;
use crate::instruments::{Instrument, Schedule};
use crate::linalg::{dot, norm_inf, Matrix};
use crate::newton::{self, NewtonConfig, TwiceDifferentiable};
use crate::sabr::{bachelier, hagan_normal_vol, OptionType, SabrParams};
use crate::Error;

/// How a quote's strike is specified.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Strike {
    /// At the forward swap rate.
    Atm,
    /// Forward plus an absolute rate offset (`0.0050` = +50bp).
    Offset(f64),
    /// An absolute strike rate.
    Absolute(f64),
}

impl Strike {
    /// Resolves to an absolute strike for forward `f`.
    pub fn resolve(self, f: f64) -> f64 {
        match self {
            Strike::Atm => f,
            Strike::Offset(d) => f + d,
            Strike::Absolute(k) => k,
        }
    }
}

/// A normal-volatility swaption quote.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolQuote {
    pub expiry: f64,
    pub tenor: f64,
    pub strike: Strike,
    /// Normal volatility in rate units per √year (`0.0060` = 60bp).
    pub normal_vol: f64,
    pub weight: f64,
}

impl VolQuote {
    pub fn new(expiry: f64, tenor: f64, strike: Strike, normal_vol: f64) -> Self {
        VolQuote {
            expiry,
            tenor,
            strike,
            normal_vol,
            weight: 1.0,
        }
    }

    pub fn atm(expiry: f64, tenor: f64, normal_vol: f64) -> Self {
        Self::new(expiry, tenor, Strike::Atm, normal_vol)
    }

    /// Quote at `offset_bp` basis points away from the forward.
    pub fn offset_bp(expiry: f64, tenor: f64, offset_bp: f64, normal_vol: f64) -> Self {
        Self::new(expiry, tenor, Strike::Offset(offset_bp * 1e-4), normal_vol)
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

/// The (expiry × tenor) grid and the forward swap rate at each node.
#[derive(Clone, Debug, PartialEq)]
pub struct SwaptionGrid {
    expiries: Vec<f64>,
    tenors: Vec<f64>,
    /// Row-major `[expiry][tenor]`.
    forwards: Vec<f64>,
}

fn check_axis(name: &str, v: &[f64]) -> Result<(), Error> {
    if v.is_empty() {
        return Err(Error::InvalidInstrument(format!("{name} axis is empty")));
    }
    for w in v.windows(2) {
        if !(w[1] > w[0]) {
            return Err(Error::InvalidInstrument(format!(
                "{name} axis must be strictly increasing"
            )));
        }
    }
    if !(v[0] > 0.0) {
        return Err(Error::InvalidInstrument(format!(
            "{name} axis must be positive"
        )));
    }
    Ok(())
}

impl SwaptionGrid {
    /// Grid with explicitly supplied forwards (row-major `[expiry][tenor]`).
    pub fn new(expiries: Vec<f64>, tenors: Vec<f64>, forwards: Vec<f64>) -> Result<Self, Error> {
        check_axis("expiry", &expiries)?;
        check_axis("tenor", &tenors)?;
        if forwards.len() != expiries.len() * tenors.len() {
            return Err(Error::DimensionMismatch {
                expected: expiries.len() * tenors.len(),
                actual: forwards.len(),
            });
        }
        Ok(SwaptionGrid {
            expiries,
            tenors,
            forwards,
        })
    }

    /// Grid whose forwards are the par rates of forward-starting swaps on a
    /// fitted [`DualCurve`]: floating leg on `float_curve`, discounting on
    /// the discount curve.
    pub fn from_curve(
        expiries: Vec<f64>,
        tenors: Vec<f64>,
        curve: &DualCurve,
        fixed_frequency: u32,
        float_frequency: u32,
        float_curve: CurveId,
    ) -> Result<Self, Error> {
        check_axis("expiry", &expiries)?;
        check_axis("tenor", &tenors)?;
        let mut forwards = Vec::with_capacity(expiries.len() * tenors.len());
        for &e in &expiries {
            for &t in &tenors {
                let swap = forward_swap(e, e + t, fixed_frequency, float_frequency, float_curve)?;
                forwards.push(curve.par_rate(&swap)?);
            }
        }
        Ok(SwaptionGrid {
            expiries,
            tenors,
            forwards,
        })
    }

    pub fn expiries(&self) -> &[f64] {
        &self.expiries
    }

    pub fn tenors(&self) -> &[f64] {
        &self.tenors
    }

    pub fn nodes(&self) -> usize {
        self.expiries.len() * self.tenors.len()
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> usize {
        i * self.tenors.len() + j
    }

    /// Forward swap rate at expiry index `i`, tenor index `j`.
    pub fn forward(&self, i: usize, j: usize) -> f64 {
        self.forwards[self.index(i, j)]
    }

    /// Node indices for an exact (expiry, tenor) match.
    pub fn node_of(&self, expiry: f64, tenor: f64) -> Option<(usize, usize)> {
        let i = self
            .expiries
            .iter()
            .position(|&e| (e - expiry).abs() < 1e-9)?;
        let j = self.tenors.iter().position(|&t| (t - tenor).abs() < 1e-9)?;
        Some((i, j))
    }
}

/// Forward-starting fixed-vs-floating swap from `start` to `end`.
pub fn forward_swap(
    start: f64,
    end: f64,
    fixed_frequency: u32,
    float_frequency: u32,
    float_curve: CurveId,
) -> Result<Instrument, Error> {
    if !(end > start) || start < 0.0 {
        return Err(Error::InvalidInstrument(format!(
            "forward swap needs 0 <= start < end (got {start}, {end})"
        )));
    }
    let sched = |freq: u32| -> Result<Schedule, Error> {
        if freq == 0 {
            return Err(Error::InvalidInstrument("frequency must be > 0".into()));
        }
        let n = (((end - start) * freq as f64) - 1e-9).ceil().max(1.0) as usize;
        let period = 1.0 / freq as f64;
        let mut times: Vec<f64> = (0..=n)
            .map(|k| (end - (n - k) as f64 * period).max(start))
            .collect();
        times[0] = start;
        if times.len() > 2 && times[1] - times[0] <= 1e-12 {
            times.remove(1);
        }
        Schedule::from_times(times)
    };
    Ok(Instrument::Swap {
        fixed_rate: 0.0,
        fixed: sched(fixed_frequency)?,
        float: sched(float_frequency)?,
        float_curve,
        discount_curve: CurveId::Discount,
    })
}

/// Conventions, regularisation and solver settings for a cube fit.
#[derive(Clone, Debug, PartialEq)]
pub struct CubeConfig {
    /// SABR `β`, fixed for the whole cube.
    pub beta: f64,
    /// Displacement applied to forwards and strikes.
    pub shift: f64,
    /// Smoothness weights on `a = ln α`, `r = atanh ρ`, `n = ln ν` between
    /// adjacent nodes, in bp² per unit².
    pub lambda_alpha: f64,
    pub lambda_rho: f64,
    pub lambda_nu: f64,
    /// Tiny pull of every parameter towards its starting value, so nodes with
    /// too few strikes to identify `ρ` and `ν` still have a unique solution.
    pub ridge: f64,
    pub initial_rho: f64,
    pub initial_nu: f64,
    pub newton: NewtonConfig,
}

impl Default for CubeConfig {
    fn default() -> Self {
        CubeConfig {
            beta: 0.5,
            shift: 0.0,
            lambda_alpha: 1e-2,
            lambda_rho: 1.0,
            lambda_nu: 1.0,
            ridge: 1e-4,
            initial_rho: 0.0,
            initial_nu: 0.3,
            newton: NewtonConfig {
                max_iterations: 200,
                gradient_tol: 1e-9,
                step_tol: 1e-10,
            },
        }
    }
}

/// Transformed → natural parameters.
fn to_params(x: [f64; 3], beta: f64, shift: f64) -> SabrParams {
    SabrParams {
        alpha: x[0].exp(),
        beta,
        rho: x[1].tanh(),
        nu: x[2].exp(),
        shift,
    }
}

/// Natural → transformed parameters.
fn to_transformed(p: &SabrParams) -> Result<[f64; 3], Error> {
    if !(p.alpha > 0.0) || !(p.nu > 0.0) || !(p.rho.abs() < 1.0) {
        return Err(Error::InvalidInstrument(format!(
            "need alpha > 0, nu > 0, |rho| < 1 (got {}, {}, {})",
            p.alpha, p.nu, p.rho
        )));
    }
    Ok([p.alpha.ln(), p.rho.atanh(), p.nu.ln()])
}

/// A fitted (or user-specified) SABR cube.
#[derive(Clone, Debug, PartialEq)]
pub struct SabrCube {
    grid: SwaptionGrid,
    beta: f64,
    shift: f64,
    /// Transformed parameters per node, row-major `[expiry][tenor]`.
    x: Vec<[f64; 3]>,
}

impl SabrCube {
    /// Cube from explicit node parameters (row-major `[expiry][tenor]`); the
    /// `beta`/`shift` of the supplied params are ignored in favour of the
    /// cube-wide conventions.
    pub fn new(
        grid: SwaptionGrid,
        beta: f64,
        shift: f64,
        params: &[SabrParams],
    ) -> Result<Self, Error> {
        if params.len() != grid.nodes() {
            return Err(Error::DimensionMismatch {
                expected: grid.nodes(),
                actual: params.len(),
            });
        }
        let x = params
            .iter()
            .map(to_transformed)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SabrCube {
            grid,
            beta,
            shift,
            x,
        })
    }

    /// Fits the cube to `quotes`. Every quote must sit on a grid node.
    pub fn fit(
        grid: SwaptionGrid,
        quotes: &[VolQuote],
        config: &CubeConfig,
    ) -> Result<CubeFit, Error> {
        if quotes.is_empty() {
            return Err(Error::InvalidInstrument("no volatility quotes".into()));
        }
        let n_nodes = grid.nodes();
        let dim = 3 * n_nodes;

        // Resolve quotes to (node, strike, vol, weight).
        let mut resolved = Vec::with_capacity(quotes.len());
        for q in quotes {
            let (i, j) = grid.node_of(q.expiry, q.tenor).ok_or_else(|| {
                Error::InvalidInstrument(format!(
                    "quote at expiry {} tenor {} is not on the grid",
                    q.expiry, q.tenor
                ))
            })?;
            if !(q.normal_vol > 0.0) || !(q.weight > 0.0) {
                return Err(Error::InvalidInstrument(
                    "quote vol and weight must be positive".into(),
                ));
            }
            let f = grid.forward(i, j);
            resolved.push(ResolvedQuote {
                node: grid.index(i, j),
                expiry: q.expiry,
                forward: f,
                strike: q.strike.resolve(f),
                vol: q.normal_vol,
                weight: q.weight,
            });
        }

        // Starting point: α from the quote closest to ATM at each node,
        // ρ and ν from the config; nodes without quotes take the mean a.
        let mut a0 = vec![None; n_nodes];
        let mut best_dist = vec![f64::INFINITY; n_nodes];
        for q in &resolved {
            let d = (q.strike - q.forward).abs();
            if d < best_dist[q.node] {
                best_dist[q.node] = d;
                let fs = q.forward + config.shift;
                if !(fs > 0.0) {
                    return Err(Error::InvalidInstrument(format!(
                        "shifted forward {fs} must be positive"
                    )));
                }
                a0[q.node] = Some((q.vol / fs.powf(config.beta)).ln());
            }
        }
        let known: Vec<f64> = a0.iter().flatten().copied().collect();
        let mean_a = known.iter().sum::<f64>() / known.len() as f64;
        let r0 = config.initial_rho.clamp(-0.999, 0.999).atanh();
        let n0 = config.initial_nu.ln();
        let mut theta0 = Vec::with_capacity(dim);
        for a in &a0 {
            theta0.extend_from_slice(&[a.unwrap_or(mean_a), r0, n0]);
        }

        let penalty = smoothness_matrix(&grid, config);
        let objective = CubeObjective {
            quotes: resolved,
            beta: config.beta,
            shift: config.shift,
            penalty,
            ridge: config.ridge,
            theta0: theta0.clone(),
            dim,
        };

        let min = newton::minimize(&objective, theta0, dim, &config.newton)?;
        let (value, residuals) = objective.residuals(&min.theta);
        let rmse = (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();
        let x = min
            .theta
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let cube = SabrCube {
            grid,
            beta: config.beta,
            shift: config.shift,
            x,
        };
        let report = CubeReport {
            iterations: min.iterations,
            converged: min.converged,
            objective: value,
            gradient_norm: norm_inf(&min.gradient),
            residuals_bp: residuals,
            rmse_bp: rmse,
            hessian: min.hessian,
        };
        Ok(CubeFit { cube, report })
    }

    pub fn grid(&self) -> &SwaptionGrid {
        &self.grid
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }

    pub fn shift(&self) -> f64 {
        self.shift
    }

    /// SABR parameters at expiry index `i`, tenor index `j`.
    pub fn node(&self, i: usize, j: usize) -> SabrParams {
        to_params(self.x[self.grid.index(i, j)], self.beta, self.shift)
    }

    /// Normal vol at a grid node for the given strike.
    pub fn node_vol(&self, i: usize, j: usize, strike: Strike) -> Result<f64, Error> {
        let f = self.grid.forward(i, j);
        self.node(i, j)
            .normal_vol(f, strike.resolve(f), self.grid.expiries[i])
    }

    /// SABR parameters for an arbitrary (expiry, tenor): bilinear
    /// interpolation of the transformed parameters in `(ln expiry, ln tenor)`,
    /// flat beyond the grid.
    pub fn params_at(&self, expiry: f64, tenor: f64) -> Result<SabrParams, Error> {
        if !(expiry > 0.0) || !(tenor > 0.0) {
            return Err(Error::InvalidInstrument(format!(
                "expiry and tenor must be positive (got {expiry}, {tenor})"
            )));
        }
        let (i0, i1, wi) = bracket(&self.grid.expiries, expiry);
        let (j0, j1, wj) = bracket(&self.grid.tenors, tenor);
        let mut x = [0.0; 3];
        for (k, xk) in x.iter_mut().enumerate() {
            let x00 = self.x[self.grid.index(i0, j0)][k];
            let x01 = self.x[self.grid.index(i0, j1)][k];
            let x10 = self.x[self.grid.index(i1, j0)][k];
            let x11 = self.x[self.grid.index(i1, j1)][k];
            *xk = (1.0 - wi) * ((1.0 - wj) * x00 + wj * x01) + wi * ((1.0 - wj) * x10 + wj * x11);
        }
        Ok(to_params(x, self.beta, self.shift))
    }

    /// Normal vol for an arbitrary swaption given its forward and strike.
    pub fn vol_at(&self, expiry: f64, tenor: f64, forward: f64, strike: f64) -> Result<f64, Error> {
        self.params_at(expiry, tenor)?
            .normal_vol(forward, strike, expiry)
    }

    /// Swaption value: Bachelier price on the cube vol times `annuity`.
    pub fn price(
        &self,
        option: OptionType,
        expiry: f64,
        tenor: f64,
        forward: f64,
        strike: f64,
        annuity: f64,
    ) -> Result<f64, Error> {
        let vol = self.vol_at(expiry, tenor, forward, strike)?;
        Ok(annuity * bachelier(option, forward, strike, expiry, vol))
    }

    /// Butterfly-arbitrage check: the implied density `∂²C/∂K²` must be
    /// non-negative. Strikes are scanned from `F − range` to `F + range` in
    /// steps of `step` (both in rate units), truncated at the shift boundary.
    pub fn arbitrage_report(&self, range: f64, step: f64) -> ArbitrageReport {
        let mut min_density = Vec::with_capacity(self.grid.nodes());
        let mut violations = Vec::new();
        for i in 0..self.grid.expiries.len() {
            let t = self.grid.expiries[i];
            for j in 0..self.grid.tenors.len() {
                let f = self.grid.forward(i, j);
                let p = self.node(i, j);
                let lo = (f - range).max(-self.shift + 2.0 * step);
                let hi = f + range;
                let n = ((hi - lo) / step).round().max(2.0) as usize;
                let price = |k: f64| -> Option<f64> {
                    p.normal_vol(f, k, t)
                        .ok()
                        .map(|v| bachelier(OptionType::Call, f, k, t, v))
                };
                let mut node_min = f64::INFINITY;
                for m in 1..n {
                    let k = lo + m as f64 * step;
                    if let (Some(cm), Some(c0), Some(cp)) =
                        (price(k - step), price(k), price(k + step))
                    {
                        let density = (cm - 2.0 * c0 + cp) / (step * step);
                        node_min = node_min.min(density);
                        if density < -1e-6 {
                            violations.push((i, j, k));
                        }
                    }
                }
                min_density.push(node_min);
            }
        }
        ArbitrageReport {
            min_density,
            violations,
        }
    }
}

/// Bracketing indices and interpolation weight in log-space, clamped to the axis.
fn bracket(axis: &[f64], v: f64) -> (usize, usize, f64) {
    if v <= axis[0] {
        return (0, 0, 0.0);
    }
    let last = axis.len() - 1;
    if v >= axis[last] {
        return (last, last, 0.0);
    }
    let i1 = axis.partition_point(|&a| a <= v);
    let i0 = i1 - 1;
    let w = (v.ln() - axis[i0].ln()) / (axis[i1].ln() - axis[i0].ln());
    (i0, i1, w)
}

/// First-difference smoothness penalty between grid neighbours, per parameter.
fn smoothness_matrix(grid: &SwaptionGrid, cfg: &CubeConfig) -> Matrix {
    let dim = 3 * grid.nodes();
    let mut q = Matrix::zeros(dim);
    let lambdas = [cfg.lambda_alpha, cfg.lambda_rho, cfg.lambda_nu];
    let mut couple = |m1: usize, m2: usize| {
        for (k, &lam) in lambdas.iter().enumerate() {
            if lam == 0.0 {
                continue;
            }
            let (p1, p2) = (3 * m1 + k, 3 * m2 + k);
            q.add_to(p1, p1, lam);
            q.add_to(p2, p2, lam);
            q.add_to(p1, p2, -lam);
            q.add_to(p2, p1, -lam);
        }
    };
    let (ne, nt) = (grid.expiries.len(), grid.tenors.len());
    for i in 0..ne {
        for j in 0..nt {
            if i + 1 < ne {
                couple(grid.index(i, j), grid.index(i + 1, j));
            }
            if j + 1 < nt {
                couple(grid.index(i, j), grid.index(i, j + 1));
            }
        }
    }
    q
}

struct ResolvedQuote {
    node: usize,
    expiry: f64,
    forward: f64,
    strike: f64,
    vol: f64,
    weight: f64,
}

struct CubeObjective {
    quotes: Vec<ResolvedQuote>,
    beta: f64,
    shift: f64,
    penalty: Matrix,
    ridge: f64,
    theta0: Vec<f64>,
    dim: usize,
}

const BP: f64 = 1e4;

impl CubeObjective {
    fn node_vars(theta: &[f64], node: usize) -> (Dual2, Dual2, Dual2) {
        let a = Dual2::variable(theta[3 * node], 0);
        let r = Dual2::variable(theta[3 * node + 1], 1);
        let n = Dual2::variable(theta[3 * node + 2], 2);
        (a.exp(), r.tanh(), n.exp())
    }

    /// Model vol (with derivatives w.r.t. the node's transformed parameters).
    fn model_vol(&self, theta: &[f64], q: &ResolvedQuote) -> Result<Dual2, Error> {
        let (alpha, rho, nu) = Self::node_vars(theta, q.node);
        hagan_normal_vol(
            alpha, rho, nu, self.beta, self.shift, q.forward, q.strike, q.expiry,
        )
    }

    fn penalty_terms(&self, theta: &[f64]) -> (f64, Vec<f64>) {
        let qt = self.penalty.mul_vec(theta);
        let mut value = 0.5 * dot(&qt, theta);
        let mut grad = qt;
        for i in 0..self.dim {
            let d = theta[i] - self.theta0[i];
            value += 0.5 * self.ridge * d * d;
            grad[i] += self.ridge * d;
        }
        (value, grad)
    }

    /// Objective value and per-quote residuals in bp.
    fn residuals(&self, theta: &[f64]) -> (f64, Vec<f64>) {
        let mut value = self.penalty_terms(theta).0;
        let mut res = Vec::with_capacity(self.quotes.len());
        for q in &self.quotes {
            let r = match self.model_vol(theta, q) {
                Ok(v) => (v.v - q.vol) * BP,
                Err(_) => f64::NAN,
            };
            value += 0.5 * q.weight * r * r;
            res.push(r);
        }
        (value, res)
    }
}

impl TwiceDifferentiable for CubeObjective {
    fn value(&self, theta: &[f64]) -> f64 {
        self.residuals(theta).0
    }

    fn value_gradient_hessian(&self, theta: &[f64]) -> (f64, Vec<f64>, Matrix) {
        let (mut value, mut gradient) = self.penalty_terms(theta);
        let mut hessian = self.penalty.clone();
        hessian.add_diagonal(self.ridge);

        for q in &self.quotes {
            let v = match self.model_vol(theta, q) {
                Ok(v) => v,
                Err(_) => return (f64::NAN, gradient, hessian),
            };
            let r = (v.v - q.vol) * BP;
            let g: [f64; 3] = v.g.map(|x| x * BP);
            value += 0.5 * q.weight * r * r;
            let base = 3 * q.node;
            for i in 0..3 {
                gradient[base + i] += q.weight * r * g[i];
                for j in 0..3 {
                    // w (∇σ ∇σᵀ + r ∇²σ)
                    hessian.add_to(
                        base + i,
                        base + j,
                        q.weight * (g[i] * g[j] + r * v.h[i][j] * BP),
                    );
                }
            }
        }
        (value, gradient, hessian)
    }
}

/// Diagnostics from a cube fit.
#[derive(Clone, Debug, PartialEq)]
pub struct CubeReport {
    pub iterations: usize,
    pub converged: bool,
    pub objective: f64,
    pub gradient_norm: f64,
    /// Model minus market, in bp of normal vol, one per input quote.
    pub residuals_bp: Vec<f64>,
    pub rmse_bp: f64,
    /// Exact Hessian over the transformed parameters `(a, r, n)` per node.
    pub hessian: Matrix,
}

impl CubeReport {
    pub fn max_abs_residual_bp(&self) -> f64 {
        norm_inf(&self.residuals_bp)
    }
}

/// Fitted cube plus diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct CubeFit {
    pub cube: SabrCube,
    pub report: CubeReport,
}

/// Result of [`SabrCube::arbitrage_report`].
#[derive(Clone, Debug, PartialEq)]
pub struct ArbitrageReport {
    /// Minimum implied density found at each node (row-major).
    pub min_density: Vec<f64>,
    /// `(expiry index, tenor index, strike)` where the density is negative.
    pub violations: Vec<(usize, usize, f64)>,
}

impl ArbitrageReport {
    pub fn is_free(&self) -> bool {
        self.violations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> SwaptionGrid {
        let expiries = vec![1.0, 2.0, 5.0, 10.0];
        let tenors = vec![2.0, 5.0, 10.0];
        let mut fwds = Vec::new();
        for (i, _) in expiries.iter().enumerate() {
            for (j, _) in tenors.iter().enumerate() {
                fwds.push(0.015 + 0.002 * i as f64 + 0.0015 * j as f64);
            }
        }
        SwaptionGrid::new(expiries, tenors, fwds).unwrap()
    }

    fn true_params(i: usize, j: usize) -> SabrParams {
        SabrParams {
            alpha: 0.025 + 0.002 * i as f64 - 0.001 * j as f64,
            beta: 0.5,
            rho: -0.25 + 0.05 * i as f64 + 0.03 * j as f64,
            nu: 0.45 - 0.04 * i as f64 + 0.02 * j as f64,
            shift: 0.02,
        }
    }

    fn true_cube() -> SabrCube {
        let g = grid();
        let mut ps = Vec::new();
        for i in 0..g.expiries().len() {
            for j in 0..g.tenors().len() {
                ps.push(true_params(i, j));
            }
        }
        SabrCube::new(g, 0.5, 0.02, &ps).unwrap()
    }

    fn quotes_from(cube: &SabrCube, offsets_bp: &[f64], skip: &[(usize, usize)]) -> Vec<VolQuote> {
        let g = cube.grid();
        let mut q = Vec::new();
        for i in 0..g.expiries().len() {
            for j in 0..g.tenors().len() {
                if skip.contains(&(i, j)) {
                    continue;
                }
                for &o in offsets_bp {
                    let v = cube.node_vol(i, j, Strike::Offset(o * 1e-4)).unwrap();
                    q.push(VolQuote::offset_bp(g.expiries()[i], g.tenors()[j], o, v));
                }
            }
        }
        q
    }

    #[test]
    fn recovers_exact_parameters_without_regularisation() {
        let truth = true_cube();
        let quotes = quotes_from(&truth, &[-150.0, -50.0, 0.0, 50.0, 150.0], &[]);
        let cfg = CubeConfig {
            shift: 0.02,
            lambda_alpha: 0.0,
            lambda_rho: 0.0,
            lambda_nu: 0.0,
            ridge: 0.0,
            ..CubeConfig::default()
        };
        let fit = SabrCube::fit(grid(), &quotes, &cfg).unwrap();
        assert!(fit.report.converged, "{:?}", fit.report);
        assert!(fit.report.max_abs_residual_bp() < 1e-6);
        for i in 0..4 {
            for j in 0..3 {
                let (p, t) = (fit.cube.node(i, j), true_params(i, j));
                assert!((p.alpha - t.alpha).abs() < 1e-7, "alpha {i},{j}");
                assert!(
                    (p.rho - t.rho).abs() < 1e-5,
                    "rho {i},{j}: {} vs {}",
                    p.rho,
                    t.rho
                );
                assert!(
                    (p.nu - t.nu).abs() < 1e-5,
                    "nu {i},{j}: {} vs {}",
                    p.nu,
                    t.nu
                );
            }
        }
    }

    #[test]
    fn fills_missing_node_from_neighbours() {
        let truth = true_cube();
        let quotes = quotes_from(&truth, &[-100.0, 0.0, 100.0], &[(1, 1)]);
        let cfg = CubeConfig {
            shift: 0.02,
            ..CubeConfig::default()
        };
        let fit = SabrCube::fit(grid(), &quotes, &cfg).unwrap();
        assert!(fit.report.converged);
        assert!(fit.report.rmse_bp < 0.05, "rmse {}", fit.report.rmse_bp);
        // The unquoted node should land close to the smooth truth.
        let (p, t) = (fit.cube.node(1, 1), true_params(1, 1));
        let v_fit = fit.cube.node_vol(1, 1, Strike::Atm).unwrap();
        let v_true = truth.node_vol(1, 1, Strike::Atm).unwrap();
        assert!(
            (v_fit - v_true).abs() * 1e4 < 3.0,
            "ATM gap {} bp",
            (v_fit - v_true) * 1e4
        );
        assert!((p.rho - t.rho).abs() < 0.1);
        assert!(fit.cube.arbitrage_report(0.03, 0.0005).is_free());
    }

    #[test]
    fn interpolation_hits_nodes_and_is_monotone_between() {
        let cube = true_cube();
        let p = cube.params_at(2.0, 5.0).unwrap();
        let t = true_params(1, 1);
        assert!((p.alpha - t.alpha).abs() < 1e-12 && (p.rho - t.rho).abs() < 1e-12);
        let mid = cube.params_at(3.0, 5.0).unwrap();
        let (lo, hi) = (true_params(1, 1), true_params(2, 1));
        assert!(mid.alpha > lo.alpha.min(hi.alpha) && mid.alpha < lo.alpha.max(hi.alpha));
        // Flat extrapolation.
        let far = cube.params_at(50.0, 0.1).unwrap();
        assert!((far.alpha - true_params(3, 0).alpha).abs() < 1e-12);
    }

    #[test]
    fn forward_swap_schedule_and_grid_from_curve() {
        use crate::fitter::DualCurveFitter;
        let fit = DualCurveFitter::new()
            .add(Instrument::ois(1.0, 0.02, 1).unwrap())
            .add(Instrument::ois(5.0, 0.025, 1).unwrap())
            .add(Instrument::ois(10.0, 0.03, 1).unwrap())
            .fit()
            .unwrap();
        let swap = forward_swap(2.0, 4.0, 1, 1, CurveId::Discount).unwrap();
        if let Instrument::Swap { fixed, .. } = &swap {
            assert_eq!(fixed.times, vec![2.0, 3.0, 4.0]);
        }
        let g = SwaptionGrid::from_curve(
            vec![1.0, 2.0],
            vec![2.0, 5.0],
            &fit.curves,
            1,
            1,
            CurveId::Discount,
        )
        .unwrap();
        assert_eq!(g.nodes(), 4);
        // 2y into 2y forward equals the par rate of the same swap.
        assert!((g.forward(1, 0) - fit.curves.par_rate(&swap).unwrap()).abs() < 1e-15);
        assert!(g.forward(1, 1) > g.forward(0, 0));
    }

    #[test]
    fn rejects_off_grid_quote() {
        let q = [VolQuote::atm(3.0, 5.0, 0.006)];
        assert!(SabrCube::fit(grid(), &q, &CubeConfig::default()).is_err());
    }
}
