//! # smoothrates
//!
//! A quadratic smooth-forward interest rate curve fitter for a dual-curve
//! (projection + discount) framework, solved jointly with a damped Newton
//! method using the **exact analytic Hessian**.
//!
//! ## Model
//!
//! Each curve's instantaneous forward `f(t)` is piecewise quadratic between
//! knots with continuous level and slope (C¹), and flat beyond the last
//! knot. The free parameters `θ = (a_0, b_0, c_0, …, c_{n−1})` enter every
//! quantity linearly, so a discount factor is `exp(−g(t)·θ)` and every
//! instrument price is a sum of exponentials of linear forms — which is why
//! gradients and Hessians are exact and cheap. See [`curve`].
//!
//! ## Instruments
//!
//! * Deposits and FRAs fixed on either curve,
//! * OIS (fixed vs. compounded overnight, both legs on the discount curve),
//! * IRS (fixed vs. floating projected on the projection curve, discounted
//!   on the discount curve).
//!
//! See [`instruments`].
//!
//! ## Fitting
//!
//! [`DualCurveFitter`] minimises the weighted sum of squared repricing
//! residuals (in rate units) plus a quadratic smoothness penalty on
//! `∫ f'²` and/or `∫ f''²` for each curve, and returns both curves together
//! with the residuals, Jacobian and Hessian at the solution. See [`fitter`].
//!
//! ```
//! use smoothrates::{CurveId, DualCurveFitter, FitConfig, Instrument};
//!
//! let fit = DualCurveFitter::new()
//!     .with_config(FitConfig::default())
//!     // Discount curve from OIS
//!     .add(Instrument::ois(1.0, 0.030, 1).unwrap())
//!     .add(Instrument::ois(2.0, 0.031, 1).unwrap())
//!     .add(Instrument::ois(5.0, 0.034, 1).unwrap())
//!     .add(Instrument::ois(10.0, 0.036, 1).unwrap())
//!     // Projection curve from a 3M deposit and IRS (annual fixed vs 3M float)
//!     .add(Instrument::deposit(CurveId::Projection, 0.25, 0.033).unwrap())
//!     .add(Instrument::irs(1.0, 0.034, 1, 4).unwrap())
//!     .add(Instrument::irs(2.0, 0.0345, 1, 4).unwrap())
//!     .add(Instrument::irs(5.0, 0.0365, 1, 4).unwrap())
//!     .add(Instrument::irs(10.0, 0.0385, 1, 4).unwrap())
//!     .fit()
//!     .unwrap();
//!
//! assert!(fit.report.converged);
//! assert!(fit.report.max_abs_residual() < 1e-7);
//! let df = fit.curves.discount_factor(CurveId::Discount, 5.0).unwrap();
//! let fwd_3m = fit.curves.simple_forward(CurveId::Projection, 2.0, 2.25).unwrap();
//! assert!(df > 0.0 && df < 1.0 && fwd_3m > 0.0);
//! ```

// `!(x > 0.0)` is used deliberately as a NaN-safe guard throughout.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod cube;
pub mod curve;
pub mod dual;
pub mod fitter;
pub mod instruments;
pub mod linalg;
pub mod newton;
pub mod sabr;

pub use cube::{
    ArbitrageReport, CubeConfig, CubeFit, CubeReport, SabrCube, Strike, SwaptionGrid, VolQuote,
};
pub use curve::{CurveId, DualBasis, QuadForwardBasis, QuadSmoothCurve};
pub use fitter::{DualCurve, DualCurveFitter, FitConfig, FitReport, FitResult};
pub use instruments::{ExpTerm, Instrument, Schedule};
pub use linalg::Matrix;
pub use newton::{NewtonConfig, TwiceDifferentiable};
pub use sabr::{bachelier, norm_cdf, OptionType, SabrParams};

/// Errors produced by curve construction, instrument definition or fitting.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// A curve needs at least one knot beyond `t = 0`.
    NoKnots,
    /// Knots must be finite, positive and strictly increasing.
    InvalidKnots(String),
    /// A time argument was negative or not finite.
    NegativeTime(f64),
    /// A parameter vector did not match its basis.
    DimensionMismatch { expected: usize, actual: usize },
    /// An instrument was ill-defined.
    InvalidInstrument(String),
    /// No instrument builds the given curve.
    NoInstruments(CurveId),
    /// The Newton solver could not make progress.
    SolverFailed(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NoKnots => write!(f, "a curve needs at least one knot beyond t = 0"),
            Error::InvalidKnots(msg) => write!(f, "invalid knots: {msg}"),
            Error::NegativeTime(t) => write!(f, "time must be finite and >= 0, got {t}"),
            Error::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Error::InvalidInstrument(msg) => write!(f, "invalid instrument: {msg}"),
            Error::NoInstruments(c) => write!(f, "no instruments build the {c} curve"),
            Error::SolverFailed(msg) => write!(f, "solver failed: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
