//! Calibration instruments and their representation as sums of exponential
//! terms in the stacked curve parameters.
//!
//! Every instrument's (normalised) pricing residual can be written as
//!
//! ```text
//! r(θ) = Σ_k  coeff_k · exp(v_k · θ)
//! ```
//!
//! because each cash flow is a discount factor `exp(−g_d(t)·θ)` or a
//! forward-rate ratio `P(t₁)/P(t₂) = exp((g_p(t₂) − g_p(t₁))·θ)`, and
//! products of exponentials are exponentials. The gradient and Hessian then
//! follow exactly:
//!
//! ```text
//! ∇r  = Σ_k coeff_k e^{v_k·θ} v_k
//! ∇²r = Σ_k coeff_k e^{v_k·θ} v_k v_kᵀ
//! ```

use crate::curve::{CurveId, DualBasis};
use crate::Error;

/// One term `coeff · exp(v · θ)` of an instrument's residual.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpTerm {
    pub coeff: f64,
    /// Exponent vector over the full (stacked) parameter space.
    pub v: Vec<f64>,
}

/// A payment schedule given as period boundaries `t_0 < t_1 < … < t_n`
/// (year fractions). Period `i` accrues over `[t_{i−1}, t_i]` and pays at `t_i`.
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub times: Vec<f64>,
}

impl Schedule {
    /// Regular schedule with `frequency` payments per year ending at
    /// `maturity`, rolled backwards from maturity so any stub is at the front.
    pub fn regular(maturity: f64, frequency: u32) -> Result<Self, Error> {
        if !(maturity > 0.0) || !maturity.is_finite() {
            return Err(Error::InvalidInstrument(format!(
                "maturity must be positive, got {maturity}"
            )));
        }
        if frequency == 0 {
            return Err(Error::InvalidInstrument("frequency must be > 0".into()));
        }
        let period = 1.0 / frequency as f64;
        let n = ((maturity / period) - 1e-9).ceil().max(1.0) as usize;
        let mut times: Vec<f64> = (0..=n)
            .map(|k| (maturity - (n - k) as f64 * period).max(0.0))
            .collect();
        times[0] = 0.0;
        // Rounding may have produced a zero-length first period.
        if times.len() > 1 && times[1] <= 1e-12 {
            times.remove(1);
        }
        Schedule::from_times(times)
    }

    /// Schedule from explicit, strictly increasing boundaries starting at `≥ 0`.
    pub fn from_times(times: Vec<f64>) -> Result<Self, Error> {
        if times.len() < 2 {
            return Err(Error::InvalidInstrument(
                "schedule needs at least one period".into(),
            ));
        }
        if times[0] < 0.0 {
            return Err(Error::NegativeTime(times[0]));
        }
        for w in times.windows(2) {
            if !(w[1] > w[0]) {
                return Err(Error::InvalidInstrument(
                    "schedule times must be strictly increasing".into(),
                ));
            }
        }
        Ok(Schedule { times })
    }

    /// Final payment time.
    pub fn maturity(&self) -> f64 {
        *self.times.last().unwrap()
    }

    /// Iterator over `(start, end, accrual)` for each period.
    pub fn periods(&self) -> impl Iterator<Item = (f64, f64, f64)> + '_ {
        self.times.windows(2).map(|w| (w[0], w[1], w[1] - w[0]))
    }
}

/// A calibration instrument.
#[derive(Clone, Debug, PartialEq)]
pub enum Instrument {
    /// A simply-compounded rate `rate` fixed on `curve` over `[start, end]`:
    /// a deposit when `start = 0`, a FRA (or a futures rate, convexity
    /// adjusted by the caller) otherwise.
    SimpleRate {
        curve: CurveId,
        start: f64,
        end: f64,
        rate: f64,
    },
    /// A fixed-for-floating swap. Fixed coupons `fixed_rate · τ` are paid on
    /// `fixed`; floating coupons are the simple forwards of `float_curve`
    /// over each `float` period. Both legs are discounted on `discount_curve`.
    Swap {
        fixed_rate: f64,
        fixed: Schedule,
        float: Schedule,
        float_curve: CurveId,
        discount_curve: CurveId,
    },
}

impl Instrument {
    /// Deposit from `0` to `maturity` fixed on `curve`.
    pub fn deposit(curve: CurveId, maturity: f64, rate: f64) -> Result<Self, Error> {
        Self::fra(curve, 0.0, maturity, rate)
    }

    /// Forward rate agreement over `[start, end]` fixed on `curve`.
    pub fn fra(curve: CurveId, start: f64, end: f64, rate: f64) -> Result<Self, Error> {
        if start < 0.0 {
            return Err(Error::NegativeTime(start));
        }
        if !(end > start) || !end.is_finite() {
            return Err(Error::InvalidInstrument(format!(
                "FRA requires end > start (got {start}, {end})"
            )));
        }
        Ok(Instrument::SimpleRate {
            curve,
            start,
            end,
            rate,
        })
    }

    /// Overnight-indexed swap: fixed vs. compounded overnight, both legs on
    /// the discount curve, `frequency` payments per year on both legs.
    pub fn ois(maturity: f64, rate: f64, frequency: u32) -> Result<Self, Error> {
        let sched = Schedule::regular(maturity, frequency)?;
        Ok(Instrument::Swap {
            fixed_rate: rate,
            fixed: sched.clone(),
            float: sched,
            float_curve: CurveId::Discount,
            discount_curve: CurveId::Discount,
        })
    }

    /// Vanilla interest rate swap: fixed leg with `fixed_frequency` payments
    /// per year vs. floating leg projected on the projection curve with
    /// `float_frequency` payments per year; discounted on the discount curve.
    pub fn irs(
        maturity: f64,
        rate: f64,
        fixed_frequency: u32,
        float_frequency: u32,
    ) -> Result<Self, Error> {
        Ok(Instrument::Swap {
            fixed_rate: rate,
            fixed: Schedule::regular(maturity, fixed_frequency)?,
            float: Schedule::regular(maturity, float_frequency)?,
            float_curve: CurveId::Projection,
            discount_curve: CurveId::Discount,
        })
    }

    /// Final cash-flow time.
    pub fn maturity(&self) -> f64 {
        match self {
            Instrument::SimpleRate { end, .. } => *end,
            Instrument::Swap { fixed, float, .. } => fixed.maturity().max(float.maturity()),
        }
    }

    /// The quoted rate.
    pub fn quote(&self) -> f64 {
        match self {
            Instrument::SimpleRate { rate, .. } => *rate,
            Instrument::Swap { fixed_rate, .. } => *fixed_rate,
        }
    }

    /// The curve whose knots this instrument's maturity defines: the fixing
    /// curve of a simple rate, or the floating-leg curve of a swap (so an OIS
    /// builds the discount curve and an IRS builds the projection curve).
    pub fn knot_curve(&self) -> CurveId {
        match self {
            Instrument::SimpleRate { curve, .. } => *curve,
            Instrument::Swap { float_curve, .. } => *float_curve,
        }
    }

    /// Whether pricing this instrument touches `curve` at all.
    pub fn depends_on(&self, curve: CurveId) -> bool {
        match self {
            Instrument::SimpleRate { curve: c, .. } => *c == curve,
            Instrument::Swap {
                float_curve,
                discount_curve,
                ..
            } => *float_curve == curve || *discount_curve == curve,
        }
    }

    /// Normalisation applied to the PV so the residual is in rate units:
    /// the accrual of a simple rate, or the fixed-leg year-fraction sum (the
    /// zero-rate annuity) of a swap.
    pub fn scale(&self) -> f64 {
        match self {
            Instrument::SimpleRate { start, end, .. } => end - start,
            Instrument::Swap { fixed, .. } => fixed.periods().map(|(_, _, tau)| tau).sum(),
        }
    }

    /// Expands the residual `PV / scale` into exponential terms over the
    /// stacked parameter space of `basis`.
    ///
    /// * Simple rate: `r = [P(start)/P(end) − 1 − rate·τ] / τ`
    /// * Swap:        `r = [Σ fixed_rate·τ_i·P_d(t_i) − Σ (P_f(s_{i−1})/P_f(s_i) − 1)·P_d(s_i)] / scale`
    pub fn terms(&self, basis: &DualBasis) -> Result<Vec<ExpTerm>, Error> {
        let dim = basis.dim();
        let scale = self.scale();
        let mut terms = Vec::new();
        match self {
            Instrument::SimpleRate {
                curve,
                start,
                end,
                rate,
            } => {
                let g1 = basis.integral_basis_full(*curve, *start)?;
                let g2 = basis.integral_basis_full(*curve, *end)?;
                // P(start)/P(end) = exp((g2 − g1)·θ)
                let v: Vec<f64> = g2.iter().zip(&g1).map(|(a, b)| a - b).collect();
                terms.push(ExpTerm {
                    coeff: 1.0 / scale,
                    v,
                });
                terms.push(ExpTerm {
                    coeff: -(1.0 + rate * scale) / scale,
                    v: vec![0.0; dim],
                });
            }
            Instrument::Swap {
                fixed_rate,
                fixed,
                float,
                float_curve,
                discount_curve,
            } => {
                for (_, t, tau) in fixed.periods() {
                    let gd = basis.integral_basis_full(*discount_curve, t)?;
                    terms.push(ExpTerm {
                        coeff: fixed_rate * tau / scale,
                        v: gd.iter().map(|x| -x).collect(),
                    });
                }
                for (s0, s1, _) in float.periods() {
                    let gd = basis.integral_basis_full(*discount_curve, s1)?;
                    let gf0 = basis.integral_basis_full(*float_curve, s0)?;
                    let gf1 = basis.integral_basis_full(*float_curve, s1)?;
                    // −(P_f(s0)/P_f(s1)) · P_d(s1) = −exp((gf1 − gf0 − gd)·θ)
                    let v: Vec<f64> = (0..dim).map(|k| gf1[k] - gf0[k] - gd[k]).collect();
                    terms.push(ExpTerm {
                        coeff: -1.0 / scale,
                        v,
                    });
                    // +P_d(s1)
                    terms.push(ExpTerm {
                        coeff: 1.0 / scale,
                        v: gd.iter().map(|x| -x).collect(),
                    });
                }
            }
        }
        Ok(terms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_schedule_rolls_back_from_maturity() {
        let s = Schedule::regular(2.0, 2).unwrap();
        assert_eq!(s.times, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        let s = Schedule::regular(1.25, 1).unwrap();
        assert_eq!(s.times.len(), 3);
        assert!((s.times[1] - 0.25).abs() < 1e-12);
        assert_eq!(s.times[2], 1.25);
        let s = Schedule::regular(0.25, 4).unwrap();
        assert_eq!(s.times, vec![0.0, 0.25]);
    }

    #[test]
    fn instrument_metadata() {
        let ois = Instrument::ois(5.0, 0.03, 1).unwrap();
        assert_eq!(ois.knot_curve(), CurveId::Discount);
        assert!(!ois.depends_on(CurveId::Projection));
        assert_eq!(ois.scale(), 5.0);
        let irs = Instrument::irs(2.0, 0.04, 1, 4).unwrap();
        assert_eq!(irs.knot_curve(), CurveId::Projection);
        assert!(irs.depends_on(CurveId::Discount));
        assert_eq!(irs.maturity(), 2.0);
        let dep = Instrument::deposit(CurveId::Projection, 0.5, 0.02).unwrap();
        assert!((dep.scale() - 0.5).abs() < 1e-15);
        assert!(Instrument::fra(CurveId::Projection, 1.0, 0.5, 0.0).is_err());
    }
}
