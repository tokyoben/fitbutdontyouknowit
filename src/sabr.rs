//! Shifted SABR smile in normal (Bachelier) volatility, and Bachelier pricing.
//!
//! The smile is the Hagan, Kumar, Lesniewski & Woodward (2002) normal-vol
//! expansion for general `β` (their eq. B.69a) with one refinement due to
//! Obloj (2008): the integral `∫_K^F dx / x^β` that enters both the
//! prefactor and `ζ` is kept exact, `(F^{1−β} − K^{1−β}) / (1−β)`, instead
//! of its leading-order approximation `(FK)^{(1−β)/2} ln(F/K)`. This is more
//! accurate in the wings and reduces *exactly* to the standard normal SABR
//! formula (`ζ = ν(F−K)/α`) at `β = 0`. Rates are shifted, `F + s`, `K + s`,
//! so negative strikes and forwards down to `−s` are admissible.
//!
//! The formula is written once over [`Dual2`] numbers, so the exact gradient
//! and Hessian with respect to `(α, ρ, ν)` fall out of the same code path
//! that produces the value.
//!
//! All volatilities are *normal* (absolute, in rate units per √year): a
//! 60bp normal vol is `0.0060`.

use crate::dual::Dual2;
use crate::Error;

/// SABR parameters for one smile. `beta` and `shift` are market conventions
/// held fixed; `alpha`, `rho`, `nu` are fitted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SabrParams {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub nu: f64,
    /// Displacement `s`: the model is driven on `F + s` (0 for USD/GBP
    /// conventions, typically 1–3% in EUR/JPY/CHF).
    pub shift: f64,
}

impl SabrParams {
    /// Normal volatility at strike `k` for forward `f` and expiry `t`.
    pub fn normal_vol(&self, f: f64, k: f64, t: f64) -> Result<f64, Error> {
        hagan_normal_vol(
            Dual2::constant(self.alpha),
            Dual2::constant(self.rho),
            Dual2::constant(self.nu),
            self.beta,
            self.shift,
            f,
            k,
            t,
        )
        .map(|d| d.v)
    }
}

/// `ζ / x̂(ζ)` with `x̂(ζ) = ln[(√(1 − 2ρζ + ζ²) − ρ + ζ) / (1 − ρ)]`,
/// using its Taylor series near `ζ = 0` where the ratio is `0/0`.
fn zeta_over_x(zeta: Dual2, rho: Dual2) -> Dual2 {
    if zeta.v.abs() < 1e-4 {
        // 1 − ρζ/2 + (2 − 3ρ²)ζ²/12 + O(ζ³)
        return 1.0 - rho * zeta * 0.5 + (2.0 - rho * rho * 3.0) * zeta * zeta / 12.0;
    }
    let root = (1.0 - 2.0 * rho * zeta + zeta * zeta).sqrt();
    let x = ((root - rho + zeta) / (1.0 - rho)).ln();
    zeta / x
}

/// `2 sinh(m·u/2) / m`, continued to `u` as `m → 0`, with a series for small
/// arguments to avoid cancellation.
fn two_sinh_half_over(m: f64, u: f64) -> f64 {
    let x = m * u / 2.0;
    if x.abs() < 1e-3 {
        // u · (1 + x²/6 + x⁴/120)
        u * (1.0 + x * x / 6.0 + x.powi(4) / 120.0)
    } else {
        2.0 * x.sinh() / m
    }
}

/// `2 sinh(u/2) / [2 sinh(m·u/2)/m]`, i.e. `(F − K) / ∫_K^F x^{−β} dx` up to the
/// factor `(FK)^{β/2}`; equals 1 at `u = 0`.
fn sinh_ratio(m: f64, u: f64) -> f64 {
    if u.abs() < 1e-3 {
        // Hagan's expansion of the same ratio, accurate to O(u⁶).
        let u2 = u * u;
        let m2 = m * m;
        (1.0 + u2 / 24.0 + u2 * u2 / 1920.0) / (1.0 + m2 * u2 / 24.0 + m2 * m2 * u2 * u2 / 1920.0)
    } else {
        two_sinh_half_over(1.0, u) / two_sinh_half_over(m, u)
    }
}

/// Hagan et al. (2002) normal-volatility expansion on shifted rates with the
/// exact `∫ dx / x^β` (see module docs),
/// evaluated over [`Dual2`] so that derivatives with respect to whichever of
/// `alpha`, `rho`, `nu` are marked as variables are propagated exactly.
#[allow(clippy::too_many_arguments)]
pub fn hagan_normal_vol(
    alpha: Dual2,
    rho: Dual2,
    nu: Dual2,
    beta: f64,
    shift: f64,
    f: f64,
    k: f64,
    t: f64,
) -> Result<Dual2, Error> {
    let fs = f + shift;
    let ks = k + shift;
    if !(fs > 0.0) || !(ks > 0.0) {
        return Err(Error::InvalidInstrument(format!(
            "shifted forward {fs} and strike {ks} must be positive"
        )));
    }
    if !(t > 0.0) {
        return Err(Error::InvalidInstrument(format!(
            "expiry must be positive, got {t}"
        )));
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err(Error::InvalidInstrument(format!(
            "beta must lie in [0, 1], got {beta}"
        )));
    }

    let fk = fs * ks;
    let u = (fs / ks).ln();
    let omb = 1.0 - beta;

    // I = ∫_K^F x^{−β} dx = (F^{1−β} − K^{1−β}) / (1−β) = (FK)^{(1−β)/2} · 2 sinh((1−β)u/2) / (1−β),
    // and F − K = (FK)^{1/2} · 2 sinh(u/2). Both are written through sinh so the
    // K → F limit is smooth; series are used where the arguments are tiny.
    let i_over_root = two_sinh_half_over(omb, u); // 2 sinh((1−β)u/2)/(1−β)

    // ζ = (ν/α) · I
    let zeta = nu / alpha * (fk.powf(omb / 2.0) * i_over_root);
    let zx = zeta_over_x(zeta, rho);

    // α (F − K) / I  =  α (FK)^{β/2} · 2 sinh(u/2) / [2 sinh((1−β)u/2)/(1−β)]
    let pre = alpha * (fk.powf(beta / 2.0) * sinh_ratio(omb, u));

    // 1 + [ −β(2−β)α² / (24 (FK)^{1−β}) + ρανβ / (4 (FK)^{(1−β)/2}) + (2−3ρ²)ν²/24 ] T
    let term1 = alpha * alpha * (-beta * (2.0 - beta) / (24.0 * fk.powf(omb)));
    let term2 = rho * alpha * nu * (beta / (4.0 * fk.powf(omb / 2.0)));
    let term3 = (2.0 - rho * rho * 3.0) * nu * nu / 24.0;
    let corr = 1.0 + (term1 + term2 + term3) * t;

    let vol = pre * zx * corr;
    if !vol.is_finite() {
        return Err(Error::SolverFailed(
            "SABR expansion produced a non-finite volatility".into(),
        ));
    }
    Ok(vol)
}

/// Standard normal density.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF to ~1e-15, via the convergent series
/// `Φ(x) = ½ + φ(x)·Σ x^{2n+1}/(1·3·…·(2n+1))` for moderate `|x|` and the
/// Mills-ratio continued fraction in the tails.
pub fn norm_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 1.0 - norm_cdf(-x);
    }
    if x > 38.0 {
        return 1.0;
    }
    if x < 3.0 {
        let mut term = x;
        let mut sum = x;
        let x2 = x * x;
        let mut n = 1.0;
        while term.abs() > 1e-17 * sum.abs() {
            term *= x2 / (2.0 * n + 1.0);
            sum += term;
            n += 1.0;
            if n > 500.0 {
                break;
            }
        }
        0.5 + norm_pdf(x) * sum
    } else {
        // Q(x) = φ(x) / (x + 1/(x + 2/(x + 3/(x + …)))), evaluated bottom-up.
        let mut cf = x;
        for k in (1..=80).rev() {
            cf = x + k as f64 / cf;
        }
        1.0 - norm_pdf(x) / cf
    }
}

/// Call or put.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionType {
    /// Payer swaption / caplet (call on the rate).
    Call,
    /// Receiver swaption / floorlet (put on the rate).
    Put,
}

/// Bachelier (normal) option value per unit annuity:
/// `call = (F − K) Φ(d) + σ√T φ(d)`, `d = (F − K)/(σ√T)`.
pub fn bachelier(option: OptionType, f: f64, k: f64, t: f64, normal_vol: f64) -> f64 {
    let sd = normal_vol * t.sqrt();
    let intrinsic = match option {
        OptionType::Call => (f - k).max(0.0),
        OptionType::Put => (k - f).max(0.0),
    };
    if sd <= 0.0 {
        return intrinsic;
    }
    let d = (f - k) / sd;
    match option {
        OptionType::Call => (f - k) * norm_cdf(d) + sd * norm_pdf(d),
        OptionType::Put => (k - f) * norm_cdf(-d) + sd * norm_pdf(d),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_cdf_reference_values() {
        let cases = [
            (0.0, 0.5),
            (1.0, 0.841_344_746_068_543),
            (-1.0, 0.158_655_253_931_457),
            (1.959_963_984_540_054, 0.975),
            (3.0, 0.998_650_101_968_370),
            (4.0, 0.999_968_328_758_167),
            (-5.0, 2.866_515_718_791_94e-7),
            (8.0, 1.0 - 6.220_960_574_271_78e-16),
        ];
        for (x, p) in cases {
            assert!(
                (norm_cdf(x) - p).abs() < 2e-15,
                "Φ({x}) = {} vs {p}",
                norm_cdf(x)
            );
        }
    }

    #[test]
    fn bachelier_put_call_parity_and_atm() {
        let (f, k, t, v) = (0.025, 0.021, 2.0, 0.0065);
        let c = bachelier(OptionType::Call, f, k, t, v);
        let p = bachelier(OptionType::Put, f, k, t, v);
        assert!((c - p - (f - k)).abs() < 1e-16);
        // ATM: σ√T / √(2π)
        let atm = bachelier(OptionType::Call, f, f, t, v);
        assert!((atm - v * t.sqrt() / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-16);
    }

    #[test]
    fn beta_zero_matches_normal_sabr_closed_form() {
        // For β = 0, B.69a reduces to α · ζ/x̂(ζ) · (1 + (2−3ρ²)ν²T/24) with ζ = ν(F−K)/α.
        let p = SabrParams {
            alpha: 0.0060,
            beta: 0.0,
            rho: -0.2,
            nu: 0.35,
            shift: 0.0,
        };
        let (f, t) = (0.02, 3.0);
        for k in [0.005, 0.015, 0.02, 0.03, 0.045] {
            let zeta = p.nu * (f - k) / p.alpha;
            let zx = if zeta.abs() < 1e-12 {
                1.0
            } else {
                let x = (((1.0 - 2.0 * p.rho * zeta + zeta * zeta).sqrt() - p.rho + zeta)
                    / (1.0 - p.rho))
                    .ln();
                zeta / x
            };
            let expected =
                p.alpha * zx * (1.0 + (2.0 - 3.0 * p.rho * p.rho) * p.nu * p.nu * t / 24.0);
            let got = p.normal_vol(f, k, t).unwrap();
            assert!((got - expected).abs() < 1e-14, "K={k}: {got} vs {expected}");
        }
    }

    #[test]
    fn smile_is_continuous_through_atm() {
        let p = SabrParams {
            alpha: 0.03,
            beta: 0.5,
            rho: 0.1,
            nu: 0.4,
            shift: 0.02,
        };
        let (f, t) = (0.015, 5.0);
        let atm = p.normal_vol(f, f, t).unwrap();
        let near = p.normal_vol(f, f + 1e-7, t).unwrap();
        let far = p.normal_vol(f, f + 1e-3, t).unwrap();
        // The smile has a finite ATM slope, so a 1e-7 strike move changes
        // the vol by O(1e-8); continuity means no jump beyond that.
        assert!((atm - near).abs() < 1e-7);
        assert!((atm - far).abs() < 1e-4 && (atm - far).abs() > 0.0);
    }

    #[test]
    fn dual_derivatives_match_finite_differences() {
        let (beta, shift, f, k, t) = (0.5, 0.02, 0.012, 0.02, 4.0);
        let x = [0.025, -0.3, 0.45];
        let eval = |a: f64, r: f64, n: f64| {
            hagan_normal_vol(
                Dual2::constant(a),
                Dual2::constant(r),
                Dual2::constant(n),
                beta,
                shift,
                f,
                k,
                t,
            )
            .unwrap()
            .v
        };
        let d = hagan_normal_vol(
            Dual2::variable(x[0], 0),
            Dual2::variable(x[1], 1),
            Dual2::variable(x[2], 2),
            beta,
            shift,
            f,
            k,
            t,
        )
        .unwrap();
        let h = 1e-5;
        let at = |dx: [f64; 3]| eval(x[0] + dx[0], x[1] + dx[1], x[2] + dx[2]);
        for i in 0..3 {
            let mut e = [0.0; 3];
            e[i] = h;
            let m = e.map(|v| -v);
            let fd = (at(e) - at(m)) / (2.0 * h);
            assert!(
                (fd - d.g[i]).abs() < 1e-7 * (1.0 + fd.abs()),
                "g[{i}] {fd} vs {}",
                d.g[i]
            );
            for j in 0..3 {
                let mut ej = [0.0; 3];
                ej[j] = h;
                let pp = [e[0] + ej[0], e[1] + ej[1], e[2] + ej[2]];
                let pm = [e[0] - ej[0], e[1] - ej[1], e[2] - ej[2]];
                let mp = [m[0] + ej[0], m[1] + ej[1], m[2] + ej[2]];
                let mm = [m[0] - ej[0], m[1] - ej[1], m[2] - ej[2]];
                let fd = (at(pp) - at(pm) - at(mp) + at(mm)) / (4.0 * h * h);
                assert!(
                    (fd - d.h[i][j]).abs() < 1e-5 * (1.0 + fd.abs()),
                    "h[{i}][{j}] {fd} vs {}",
                    d.h[i][j]
                );
            }
        }
    }
}
