//! Second-order forward-mode automatic differentiation in three variables.
//!
//! A [`Dual2`] carries a value, its gradient and its full Hessian with
//! respect to three independent variables. Arithmetic propagates all three
//! exactly (no truncation), so any closed-form expression evaluated on
//! `Dual2` inputs yields its exact gradient and Hessian — this is how the
//! SABR smile formula is differentiated with respect to `(α, ρ, ν)` without
//! deriving the derivatives by hand.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Number of independent variables.
pub const N: usize = 3;

/// Value, gradient and Hessian with respect to `N` variables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual2 {
    pub v: f64,
    pub g: [f64; N],
    pub h: [[f64; N]; N],
}

impl Dual2 {
    /// A constant (zero derivatives).
    pub const fn constant(v: f64) -> Self {
        Dual2 {
            v,
            g: [0.0; N],
            h: [[0.0; N]; N],
        }
    }

    /// The `i`-th independent variable with value `v`.
    pub fn variable(v: f64, i: usize) -> Self {
        let mut d = Dual2::constant(v);
        d.g[i] = 1.0;
        d
    }

    /// Chain rule for a scalar function given `f(v)`, `f'(v)`, `f''(v)`.
    fn apply(self, f: f64, d1: f64, d2: f64) -> Self {
        let mut out = Dual2::constant(f);
        for i in 0..N {
            out.g[i] = d1 * self.g[i];
            for j in 0..N {
                out.h[i][j] = d1 * self.h[i][j] + d2 * self.g[i] * self.g[j];
            }
        }
        out
    }

    pub fn exp(self) -> Self {
        let e = self.v.exp();
        self.apply(e, e, e)
    }

    pub fn ln(self) -> Self {
        self.apply(self.v.ln(), 1.0 / self.v, -1.0 / (self.v * self.v))
    }

    pub fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.apply(s, 0.5 / s, -0.25 / (s * self.v))
    }

    pub fn recip(self) -> Self {
        let r = 1.0 / self.v;
        self.apply(r, -r * r, 2.0 * r * r * r)
    }

    /// `self^p` for a constant exponent.
    pub fn powf(self, p: f64) -> Self {
        if p == 0.0 {
            return Dual2::constant(1.0);
        }
        if p == 1.0 {
            return self;
        }
        let f = self.v.powf(p);
        self.apply(f, p * f / self.v, p * (p - 1.0) * f / (self.v * self.v))
    }

    pub fn powi(self, n: i32) -> Self {
        self.powf(n as f64)
    }

    pub fn tanh(self) -> Self {
        let t = self.v.tanh();
        let sech2 = 1.0 - t * t;
        self.apply(t, sech2, -2.0 * t * sech2)
    }

    pub fn is_finite(&self) -> bool {
        self.v.is_finite()
            && self.g.iter().all(|x| x.is_finite())
            && self.h.iter().flatten().all(|x| x.is_finite())
    }
}

impl From<f64> for Dual2 {
    fn from(v: f64) -> Self {
        Dual2::constant(v)
    }
}

impl Neg for Dual2 {
    type Output = Dual2;
    fn neg(self) -> Dual2 {
        self.apply(-self.v, -1.0, 0.0)
    }
}

impl Add for Dual2 {
    type Output = Dual2;
    fn add(self, o: Dual2) -> Dual2 {
        let mut out = Dual2::constant(self.v + o.v);
        for i in 0..N {
            out.g[i] = self.g[i] + o.g[i];
            for j in 0..N {
                out.h[i][j] = self.h[i][j] + o.h[i][j];
            }
        }
        out
    }
}

impl Sub for Dual2 {
    type Output = Dual2;
    fn sub(self, o: Dual2) -> Dual2 {
        self + (-o)
    }
}

impl Mul for Dual2 {
    type Output = Dual2;
    fn mul(self, o: Dual2) -> Dual2 {
        let mut out = Dual2::constant(self.v * o.v);
        for i in 0..N {
            out.g[i] = self.v * o.g[i] + o.v * self.g[i];
            for j in 0..N {
                out.h[i][j] = self.v * o.h[i][j]
                    + o.v * self.h[i][j]
                    + self.g[i] * o.g[j]
                    + o.g[i] * self.g[j];
            }
        }
        out
    }
}

impl Div for Dual2 {
    type Output = Dual2;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, o: Dual2) -> Dual2 {
        self * o.recip()
    }
}

macro_rules! scalar_ops {
    ($($t:ty),*) => {$(
        impl Add<$t> for Dual2 { type Output = Dual2; fn add(self, o: $t) -> Dual2 { self + Dual2::constant(o as f64) } }
        impl Sub<$t> for Dual2 { type Output = Dual2; fn sub(self, o: $t) -> Dual2 { self - Dual2::constant(o as f64) } }
        impl Mul<$t> for Dual2 { type Output = Dual2; fn mul(self, o: $t) -> Dual2 { self * Dual2::constant(o as f64) } }
        impl Div<$t> for Dual2 { type Output = Dual2; #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, o: $t) -> Dual2 { self * Dual2::constant(1.0 / o as f64) } }
        impl Add<Dual2> for $t { type Output = Dual2; fn add(self, o: Dual2) -> Dual2 { Dual2::constant(self as f64) + o } }
        impl Sub<Dual2> for $t { type Output = Dual2; fn sub(self, o: Dual2) -> Dual2 { Dual2::constant(self as f64) - o } }
        impl Mul<Dual2> for $t { type Output = Dual2; fn mul(self, o: Dual2) -> Dual2 { Dual2::constant(self as f64) * o } }
        impl Div<Dual2> for $t { type Output = Dual2; #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, o: Dual2) -> Dual2 { Dual2::constant(self as f64) * o.recip() } }
    )*};
}
scalar_ops!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    /// A test function mixing every supported operation.
    fn f(x: [Dual2; 3]) -> Dual2 {
        let [a, b, c] = x;
        (a * b).ln() * c.sqrt() + (a / c).exp() * 0.5 - b.tanh() * a.powf(1.5)
            + (c - a) / (b + 2.0)
            + 3.0 / c
    }

    fn f_val(x: [f64; 3]) -> f64 {
        f([
            Dual2::constant(x[0]),
            Dual2::constant(x[1]),
            Dual2::constant(x[2]),
        ])
        .v
    }

    #[test]
    fn gradient_and_hessian_match_finite_differences() {
        let x = [1.3, 0.7, 2.1];
        let d = f([
            Dual2::variable(x[0], 0),
            Dual2::variable(x[1], 1),
            Dual2::variable(x[2], 2),
        ]);
        assert!((d.v - f_val(x)).abs() < 1e-14);
        let h = 1e-5;
        for i in 0..3 {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (f_val(xp) - f_val(xm)) / (2.0 * h);
            assert!((fd - d.g[i]).abs() < 1e-8, "g[{i}] {fd} vs {}", d.g[i]);
            for j in 0..3 {
                let mut xpp = x;
                let mut xpm = x;
                let mut xmp = x;
                let mut xmm = x;
                xpp[i] += h;
                xpp[j] += h;
                xpm[i] += h;
                xpm[j] -= h;
                xmp[i] -= h;
                xmp[j] += h;
                xmm[i] -= h;
                xmm[j] -= h;
                let fd = (f_val(xpp) - f_val(xpm) - f_val(xmp) + f_val(xmm)) / (4.0 * h * h);
                assert!(
                    (fd - d.h[i][j]).abs() < 1e-5,
                    "h[{i}][{j}] {fd} vs {}",
                    d.h[i][j]
                );
                assert!((d.h[i][j] - d.h[j][i]).abs() < 1e-14);
            }
        }
    }
}
