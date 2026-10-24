//! Piecewise-quadratic instantaneous forward curves with C¹ continuity.
//!
//! On segment `i` between knots `t_i ≤ t < t_{i+1}` (with `d = t − t_i`,
//! `h_i = t_{i+1} − t_i`) the instantaneous forward is
//!
//! ```text
//! f(t) = a_i + b_i·d + c_i·d²
//! ```
//!
//! Continuity of `f` and `f'` across knots gives the recurrences
//!
//! ```text
//! a_{i+1} = a_i + b_i·h_i + c_i·h_i²
//! b_{i+1} = b_i + 2·c_i·h_i
//! ```
//!
//! so the free parameters are `θ = (a_0, b_0, c_0, …, c_{n−1})`. Every
//! quantity of interest — `f(t)`, `f'(t)` and the integrated forward
//! `I(t) = ∫₀ᵗ f(s) ds` — is **linear** in `θ`, and the discount factor is
//! `P(t) = exp(−I(t)) = exp(−g(t)·θ)` for a basis vector `g(t)` that depends
//! only on the knots. That linearity is what makes the exact Hessian of
//! instrument prices cheap to compute in the fitter.
//!
//! Beyond the last knot the forward is extrapolated flat.

use crate::linalg::{axpy, dot, Matrix};
use crate::Error;

/// Identifies one of the two curves in a dual-curve setup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CurveId {
    /// The discounting curve (e.g. OIS / RFR).
    Discount,
    /// The projection curve for floating-rate fixings (e.g. 3M IBOR).
    Projection,
}

impl std::fmt::Display for CurveId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CurveId::Discount => write!(f, "discount"),
            CurveId::Projection => write!(f, "projection"),
        }
    }
}

/// The linear basis of a quadratic-forward curve on a fixed set of knots.
///
/// All basis vectors have length [`dim()`](Self::dim) `= n + 2` where `n`
/// is the number of segments; index `0` is `a_0`, index `1` is `b_0` and
/// index `2 + i` is `c_i`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuadForwardBasis {
    /// `t_0 = 0 < t_1 < … < t_n`
    knots: Vec<f64>,
    /// `a_i` as a linear functional of θ, for `i = 0..=n`.
    a_lin: Vec<Vec<f64>>,
    /// `b_i` as a linear functional of θ, for `i = 0..=n`.
    b_lin: Vec<Vec<f64>>,
    /// `I(t_i)` as a linear functional of θ, for `i = 0..=n`.
    i_lin: Vec<Vec<f64>>,
}

impl QuadForwardBasis {
    /// Builds the basis from strictly increasing, positive knot times
    /// (year fractions). A knot at `t = 0` is added automatically.
    pub fn new(maturities: &[f64]) -> Result<Self, Error> {
        let mut knots = Vec::with_capacity(maturities.len() + 1);
        knots.push(0.0);
        for &m in maturities {
            if !m.is_finite() || m <= 0.0 {
                return Err(Error::InvalidKnots(format!(
                    "knot {m} must be finite and > 0"
                )));
            }
            if m <= *knots.last().unwrap() {
                return Err(Error::InvalidKnots(
                    "knots must be strictly increasing".to_string(),
                ));
            }
            knots.push(m);
        }
        let n = knots.len() - 1;
        if n == 0 {
            return Err(Error::NoKnots);
        }
        let dim = n + 2;

        let unit = |k: usize| {
            let mut e = vec![0.0; dim];
            e[k] = 1.0;
            e
        };

        let mut a_lin = Vec::with_capacity(n + 1);
        let mut b_lin = Vec::with_capacity(n + 1);
        let mut i_lin = Vec::with_capacity(n + 1);
        a_lin.push(unit(0));
        b_lin.push(unit(1));
        i_lin.push(vec![0.0; dim]);

        for i in 0..n {
            let h = knots[i + 1] - knots[i];
            let c = 2 + i;
            let (a, b, int) = (&a_lin[i], &b_lin[i], &i_lin[i]);

            // a_{i+1} = a_i + b_i h + c_i h²
            let mut a_next = a.clone();
            axpy(h, b, &mut a_next);
            a_next[c] += h * h;

            // b_{i+1} = b_i + 2 c_i h
            let mut b_next = b.clone();
            b_next[c] += 2.0 * h;

            // I(t_{i+1}) = I(t_i) + a_i h + b_i h²/2 + c_i h³/3
            let mut i_next = int.clone();
            axpy(h, a, &mut i_next);
            axpy(h * h / 2.0, b, &mut i_next);
            i_next[c] += h * h * h / 3.0;

            a_lin.push(a_next);
            b_lin.push(b_next);
            i_lin.push(i_next);
        }

        Ok(QuadForwardBasis {
            knots,
            a_lin,
            b_lin,
            i_lin,
        })
    }

    /// Number of free parameters (`segments + 2`).
    pub fn dim(&self) -> usize {
        self.knots.len() + 1
    }

    /// Number of quadratic segments.
    pub fn segments(&self) -> usize {
        self.knots.len() - 1
    }

    /// Knot times including the leading zero.
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Index of the segment containing `t` (`segments()` for the flat
    /// extrapolation region past the last knot).
    fn segment_of(&self, t: f64) -> usize {
        match self.knots.binary_search_by(|k| k.partial_cmp(&t).unwrap()) {
            Ok(i) => i,
            Err(i) => i - 1,
        }
    }

    fn check_time(&self, t: f64) -> Result<(), Error> {
        if !t.is_finite() || t < 0.0 {
            Err(Error::NegativeTime(t))
        } else {
            Ok(())
        }
    }

    /// Basis vector `g(t)` with `∫₀ᵗ f(s) ds = g(t)·θ`.
    pub fn integral_basis(&self, t: f64) -> Result<Vec<f64>, Error> {
        self.check_time(t)?;
        let i = self.segment_of(t);
        let d = t - self.knots[i];
        let mut g = self.i_lin[i].clone();
        axpy(d, &self.a_lin[i], &mut g);
        if i < self.segments() {
            axpy(d * d / 2.0, &self.b_lin[i], &mut g);
            g[2 + i] += d * d * d / 3.0;
        }
        Ok(g)
    }

    /// Basis vector `φ(t)` with `f(t) = φ(t)·θ`.
    pub fn forward_basis(&self, t: f64) -> Result<Vec<f64>, Error> {
        self.check_time(t)?;
        let i = self.segment_of(t);
        let d = t - self.knots[i];
        let mut phi = self.a_lin[i].clone();
        if i < self.segments() {
            axpy(d, &self.b_lin[i], &mut phi);
            phi[2 + i] += d * d;
        }
        Ok(phi)
    }

    /// Basis vector `ψ(t)` with `f'(t) = ψ(t)·θ` (zero past the last knot).
    pub fn slope_basis(&self, t: f64) -> Result<Vec<f64>, Error> {
        self.check_time(t)?;
        let i = self.segment_of(t);
        if i >= self.segments() {
            return Ok(vec![0.0; self.dim()]);
        }
        let d = t - self.knots[i];
        let mut psi = self.b_lin[i].clone();
        psi[2 + i] += 2.0 * d;
        Ok(psi)
    }

    /// Smoothness penalty matrix `Q` such that
    ///
    /// ```text
    /// θᵀ Q θ = λ_slope · ∫ f'(t)² dt + λ_curvature · ∫ f''(t)² dt
    /// ```
    ///
    /// integrated over `[0, t_n]`. Both integrals are exact for piecewise
    /// quadratics: on a segment of width `h`,
    /// `∫ f'² = h b² + 2h² b c + (4/3) h³ c²` and `∫ f''² = 4 h c²`.
    pub fn penalty_matrix(&self, lambda_slope: f64, lambda_curvature: f64) -> Matrix {
        let dim = self.dim();
        let mut q = Matrix::zeros(dim);
        for i in 0..self.segments() {
            let h = self.knots[i + 1] - self.knots[i];
            let c = 2 + i;
            if lambda_curvature != 0.0 {
                q.add_to(c, c, lambda_curvature * 4.0 * h);
            }
            if lambda_slope != 0.0 {
                let b = &self.b_lin[i];
                // h · b bᵀ
                q.add_scaled_outer(b, lambda_slope * h);
                // h² (b e_cᵀ + e_c bᵀ)
                for k in 0..dim {
                    let v = lambda_slope * h * h * b[k];
                    q.add_to(k, c, v);
                    q.add_to(c, k, v);
                }
                // (4/3) h³ e_c e_cᵀ
                q.add_to(c, c, lambda_slope * 4.0 / 3.0 * h * h * h);
            }
        }
        q
    }
}

/// A fitted quadratic smooth-forward curve: a basis plus its parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct QuadSmoothCurve {
    basis: QuadForwardBasis,
    theta: Vec<f64>,
}

impl QuadSmoothCurve {
    /// Creates a curve from a basis and parameter vector.
    pub fn new(basis: QuadForwardBasis, theta: Vec<f64>) -> Result<Self, Error> {
        if theta.len() != basis.dim() {
            return Err(Error::DimensionMismatch {
                expected: basis.dim(),
                actual: theta.len(),
            });
        }
        Ok(QuadSmoothCurve { basis, theta })
    }

    /// A flat curve with constant instantaneous forward `rate` on the given knots.
    pub fn flat(maturities: &[f64], rate: f64) -> Result<Self, Error> {
        let basis = QuadForwardBasis::new(maturities)?;
        let mut theta = vec![0.0; basis.dim()];
        theta[0] = rate;
        Ok(QuadSmoothCurve { basis, theta })
    }

    pub fn basis(&self) -> &QuadForwardBasis {
        &self.basis
    }

    /// Parameter vector `θ = (a_0, b_0, c_0, …, c_{n−1})`.
    pub fn parameters(&self) -> &[f64] {
        &self.theta
    }

    /// Knot times including the leading zero.
    pub fn knots(&self) -> &[f64] {
        self.basis.knots()
    }

    /// `(a_i, b_i, c_i)` for segment `i`, so that
    /// `f(t) = a_i + b_i (t − t_i) + c_i (t − t_i)²` on that segment.
    pub fn segment_coefficients(&self, i: usize) -> Option<(f64, f64, f64)> {
        if i >= self.basis.segments() {
            return None;
        }
        Some((
            dot(&self.basis.a_lin[i], &self.theta),
            dot(&self.basis.b_lin[i], &self.theta),
            self.theta[2 + i],
        ))
    }

    /// Instantaneous forward rate `f(t)`.
    pub fn forward(&self, t: f64) -> Result<f64, Error> {
        Ok(dot(&self.basis.forward_basis(t)?, &self.theta))
    }

    /// Slope of the instantaneous forward `f'(t)`.
    pub fn forward_slope(&self, t: f64) -> Result<f64, Error> {
        Ok(dot(&self.basis.slope_basis(t)?, &self.theta))
    }

    /// Integrated forward `∫₀ᵗ f(s) ds = −ln P(t)`.
    pub fn integrated_forward(&self, t: f64) -> Result<f64, Error> {
        Ok(dot(&self.basis.integral_basis(t)?, &self.theta))
    }

    /// Discount factor `P(t) = exp(−∫₀ᵗ f)`.
    pub fn discount_factor(&self, t: f64) -> Result<f64, Error> {
        Ok((-self.integrated_forward(t)?).exp())
    }

    /// Continuously compounded zero rate `−ln P(t) / t` (equals `f(0)` at `t = 0`).
    pub fn zero_rate(&self, t: f64) -> Result<f64, Error> {
        if t == 0.0 {
            return self.forward(0.0);
        }
        Ok(self.integrated_forward(t)? / t)
    }

    /// Simply compounded forward rate over `[t1, t2]`: `(P(t1)/P(t2) − 1) / (t2 − t1)`.
    pub fn simple_forward(&self, t1: f64, t2: f64) -> Result<f64, Error> {
        if t2 <= t1 {
            return Err(Error::InvalidInstrument(format!(
                "simple forward requires t2 > t1 (got {t1}, {t2})"
            )));
        }
        let i1 = self.integrated_forward(t1)?;
        let i2 = self.integrated_forward(t2)?;
        Ok(((i2 - i1).exp() - 1.0) / (t2 - t1))
    }
}

/// The stacked parameter space of a discount and a projection curve.
///
/// Parameters are laid out as `[θ_discount ‖ θ_projection]`; the
/// `*_full` methods return basis vectors over this full space with zeros in
/// the other curve's block, so instrument prices can be differentiated
/// jointly with respect to both curves.
#[derive(Clone, Debug, PartialEq)]
pub struct DualBasis {
    discount: QuadForwardBasis,
    projection: QuadForwardBasis,
}

impl DualBasis {
    pub fn new(discount: QuadForwardBasis, projection: QuadForwardBasis) -> Self {
        DualBasis {
            discount,
            projection,
        }
    }

    /// Total number of parameters across both curves.
    pub fn dim(&self) -> usize {
        self.discount.dim() + self.projection.dim()
    }

    /// Offset of a curve's parameter block in the stacked vector.
    pub fn offset(&self, curve: CurveId) -> usize {
        match curve {
            CurveId::Discount => 0,
            CurveId::Projection => self.discount.dim(),
        }
    }

    pub fn basis(&self, curve: CurveId) -> &QuadForwardBasis {
        match curve {
            CurveId::Discount => &self.discount,
            CurveId::Projection => &self.projection,
        }
    }

    /// Slice of the stacked parameter vector belonging to `curve`.
    pub fn block<'a>(&self, curve: CurveId, theta: &'a [f64]) -> &'a [f64] {
        let off = self.offset(curve);
        &theta[off..off + self.basis(curve).dim()]
    }

    /// `g(t)` for `curve`, embedded in the full parameter space, with
    /// `−ln P_curve(t) = g(t)·θ`.
    pub fn integral_basis_full(&self, curve: CurveId, t: f64) -> Result<Vec<f64>, Error> {
        let local = self.basis(curve).integral_basis(t)?;
        let mut full = vec![0.0; self.dim()];
        let off = self.offset(curve);
        full[off..off + local.len()].copy_from_slice(&local);
        Ok(full)
    }

    /// Extracts the fitted curve for `curve` from a stacked parameter vector.
    pub fn curve(&self, curve: CurveId, theta: &[f64]) -> Result<QuadSmoothCurve, Error> {
        if theta.len() != self.dim() {
            return Err(Error::DimensionMismatch {
                expected: self.dim(),
                actual: theta.len(),
            });
        }
        QuadSmoothCurve::new(self.basis(curve).clone(), self.block(curve, theta).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_curve() -> QuadSmoothCurve {
        let basis = QuadForwardBasis::new(&[0.5, 1.0, 2.0, 5.0, 10.0]).unwrap();
        // a_0, b_0, c_0..c_4
        let theta = vec![0.03, 0.004, 0.002, -0.0015, 0.0007, -0.0002, 0.00005];
        QuadSmoothCurve::new(basis, theta).unwrap()
    }

    #[test]
    fn forward_is_c1_at_knots() {
        let c = sample_curve();
        for &k in &c.knots()[1..c.knots().len() - 1] {
            let eps = 1e-7;
            let f_minus = c.forward(k - eps).unwrap();
            let f_plus = c.forward(k + eps).unwrap();
            assert!((f_minus - f_plus).abs() < 1e-8, "f jump at {k}");
            let s_minus = c.forward_slope(k - eps).unwrap();
            let s_plus = c.forward_slope(k + eps).unwrap();
            assert!((s_minus - s_plus).abs() < 1e-8, "f' jump at {k}");
        }
    }

    #[test]
    fn integral_matches_quadrature() {
        let c = sample_curve();
        for &t in &[0.3, 1.0, 1.7, 4.2, 9.9, 10.0, 13.0] {
            // Simpson's rule on a fine grid.
            let n = 20_000;
            let h = t / n as f64;
            let mut s = c.forward(0.0).unwrap() + c.forward(t).unwrap();
            for k in 1..n {
                let w = if k % 2 == 1 { 4.0 } else { 2.0 };
                s += w * c.forward(k as f64 * h).unwrap();
            }
            let num = s * h / 3.0;
            let ana = c.integrated_forward(t).unwrap();
            assert!((num - ana).abs() < 1e-10, "t={t}: {num} vs {ana}");
        }
    }

    #[test]
    fn flat_extrapolation_past_last_knot() {
        let c = sample_curve();
        let f_end = c.forward(10.0).unwrap();
        assert!((c.forward(15.0).unwrap() - f_end).abs() < 1e-15);
        assert_eq!(c.forward_slope(12.0).unwrap(), 0.0);
        let df10 = c.discount_factor(10.0).unwrap();
        let df12 = c.discount_factor(12.0).unwrap();
        assert!((df12 / df10 - (-2.0 * f_end).exp()).abs() < 1e-14);
    }

    #[test]
    fn flat_curve_has_constant_forward() {
        let c = QuadSmoothCurve::flat(&[1.0, 2.0], 0.05).unwrap();
        for &t in &[0.0, 0.3, 1.0, 1.5, 2.0, 3.0] {
            assert!((c.forward(t).unwrap() - 0.05).abs() < 1e-15);
            assert!((c.zero_rate(t).unwrap() - 0.05).abs() < 1e-15);
        }
    }

    #[test]
    fn penalty_matrix_matches_quadrature() {
        let c = sample_curve();
        let q_slope = c.basis().penalty_matrix(1.0, 0.0);
        let q_curv = c.basis().penalty_matrix(0.0, 1.0);
        let t_end = *c.knots().last().unwrap();
        let n = 200_000;
        let h = t_end / n as f64;
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for k in 0..n {
            let t = (k as f64 + 0.5) * h;
            let seg = c.basis().segment_of(t);
            let (_, _, ci) = c.segment_coefficients(seg).unwrap();
            s1 += c.forward_slope(t).unwrap().powi(2) * h;
            s2 += (2.0 * ci).powi(2) * h;
        }
        assert!((q_slope.quadratic_form(c.parameters()) - s1).abs() < 1e-9);
        assert!((q_curv.quadratic_form(c.parameters()) - s2).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_knots() {
        assert!(QuadForwardBasis::new(&[]).is_err());
        assert!(QuadForwardBasis::new(&[1.0, 1.0]).is_err());
        assert!(QuadForwardBasis::new(&[2.0, 1.0]).is_err());
        assert!(QuadForwardBasis::new(&[0.0, 1.0]).is_err());
    }
}
