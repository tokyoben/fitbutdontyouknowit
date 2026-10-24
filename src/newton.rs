//! Damped Newton minimisation for objectives that supply their exact Hessian.
//!
//! Each step solves `(H + μI) δ = −∇F` by Cholesky, increasing the
//! Levenberg–Marquardt damping `μ` until the system is positive definite and
//! `δ` is a descent direction, then backtracks along `δ` until the Armijo
//! condition holds. `μ` decays after every accepted step, so near a well
//! conditioned optimum the iteration is pure Newton.

use crate::linalg::{axpy, dot, norm_inf, Matrix};
use crate::Error;

/// An objective with exact gradient and Hessian.
pub trait TwiceDifferentiable {
    /// Objective value only (used in the line search).
    fn value(&self, theta: &[f64]) -> f64;
    /// Objective value, gradient and Hessian at `theta`.
    fn value_gradient_hessian(&self, theta: &[f64]) -> (f64, Vec<f64>, Matrix);
}

/// Stopping rules for [`minimize`].
#[derive(Clone, Debug, PartialEq)]
pub struct NewtonConfig {
    pub max_iterations: usize,
    /// Converged when the infinity norm of the gradient falls below this.
    pub gradient_tol: f64,
    /// Converged when the infinity norm of the accepted step falls below this.
    pub step_tol: f64,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        NewtonConfig {
            max_iterations: 100,
            gradient_tol: 1e-13,
            step_tol: 1e-13,
        }
    }
}

/// Outcome of [`minimize`].
#[derive(Clone, Debug, PartialEq)]
pub struct Minimum {
    pub theta: Vec<f64>,
    pub value: f64,
    pub gradient: Vec<f64>,
    pub hessian: Matrix,
    pub iterations: usize,
    pub converged: bool,
}

fn leading_block(m: &Matrix, n: usize) -> Matrix {
    let mut r = Matrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            r.set(i, j, m.get(i, j));
        }
    }
    r
}

/// Minimises `objective` over the first `n_active` coordinates of `theta`
/// (the remaining coordinates are held fixed).
pub fn minimize<O: TwiceDifferentiable>(
    objective: &O,
    mut theta: Vec<f64>,
    n_active: usize,
    cfg: &NewtonConfig,
) -> Result<Minimum, Error> {
    let (mut value, mut gradient, mut hessian) = objective.value_gradient_hessian(&theta);
    if !value.is_finite() {
        return Err(Error::SolverFailed(
            "objective is not finite at the starting point".into(),
        ));
    }
    let mut mu = 0.0_f64;
    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..cfg.max_iterations {
        let g = &gradient[..n_active];
        if norm_inf(g) < cfg.gradient_tol {
            converged = true;
            break;
        }
        iterations += 1;

        let h = leading_block(&hessian, n_active);
        let max_diag = (0..n_active).fold(0.0_f64, |m, i| m.max(h.get(i, i).abs()));
        let neg_g: Vec<f64> = g.iter().map(|x| -x).collect();

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

        let mut alpha = 1.0;
        let mut accepted = None;
        loop {
            let mut trial = theta.clone();
            axpy(alpha, &delta, &mut trial[..n_active]);
            let f_trial = objective.value(&trial);
            if f_trial.is_finite() && f_trial <= value + 1e-4 * alpha * slope {
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
                let (v, gr, hs) = objective.value_gradient_hessian(&theta);
                value = v;
                gradient = gr;
                hessian = hs;
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

    Ok(Minimum {
        theta,
        value,
        gradient,
        hessian,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rosenbrock in 2D plus a quadratic in a third, fixed coordinate.
    struct Rosenbrock;

    impl TwiceDifferentiable for Rosenbrock {
        fn value(&self, t: &[f64]) -> f64 {
            let (x, y) = (t[0], t[1]);
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2) + t[2] * t[2]
        }
        fn value_gradient_hessian(&self, t: &[f64]) -> (f64, Vec<f64>, Matrix) {
            let (x, y) = (t[0], t[1]);
            let g = vec![
                -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
                200.0 * (y - x * x),
                2.0 * t[2],
            ];
            let mut h = Matrix::zeros(3);
            h.set(0, 0, 2.0 - 400.0 * (y - x * x) + 800.0 * x * x);
            h.set(0, 1, -400.0 * x);
            h.set(1, 0, -400.0 * x);
            h.set(1, 1, 200.0);
            h.set(2, 2, 2.0);
            (self.value(t), g, h)
        }
    }

    #[test]
    fn minimises_rosenbrock_with_fixed_coordinate() {
        let cfg = NewtonConfig {
            max_iterations: 200,
            gradient_tol: 1e-10,
            step_tol: 1e-14,
        };
        let m = minimize(&Rosenbrock, vec![-1.2, 1.0, 0.7], 2, &cfg).unwrap();
        assert!(m.converged);
        assert!((m.theta[0] - 1.0).abs() < 1e-8);
        assert!((m.theta[1] - 1.0).abs() < 1e-8);
        assert_eq!(m.theta[2], 0.7, "inactive coordinate must not move");
    }
}
