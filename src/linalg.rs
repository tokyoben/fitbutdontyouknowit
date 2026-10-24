//! Minimal dense linear algebra: a square symmetric matrix type with a
//! Cholesky solver. The fitting problems here have a few dozen unknowns at
//! most, so a straightforward O(n³) implementation is more than adequate.

/// Dense square matrix stored row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    n: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// `n × n` matrix of zeros.
    pub fn zeros(n: usize) -> Self {
        Matrix {
            n,
            data: vec![0.0; n * n],
        }
    }

    /// `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Dimension of the (square) matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }

    #[inline]
    pub fn add_to(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] += v;
    }

    /// Row `i` as a slice.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.n..(i + 1) * self.n]
    }

    /// `self += scale · v vᵀ`
    pub fn add_scaled_outer(&mut self, v: &[f64], scale: f64) {
        debug_assert_eq!(v.len(), self.n);
        if scale == 0.0 {
            return;
        }
        for i in 0..self.n {
            let vi = scale * v[i];
            if vi == 0.0 {
                continue;
            }
            let row = &mut self.data[i * self.n..(i + 1) * self.n];
            for (r, &vj) in row.iter_mut().zip(v.iter()) {
                *r += vi * vj;
            }
        }
    }

    /// `self += scale · other`
    pub fn add_scaled(&mut self, other: &Matrix, scale: f64) {
        debug_assert_eq!(other.n, self.n);
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += scale * b;
        }
    }

    /// `self += scale · I`
    pub fn add_diagonal(&mut self, scale: f64) {
        for i in 0..self.n {
            self.add_to(i, i, scale);
        }
    }

    /// Matrix–vector product `self · x`.
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);
        (0..self.n)
            .map(|i| self.row(i).iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Quadratic form `xᵀ · self · x`.
    pub fn quadratic_form(&self, x: &[f64]) -> f64 {
        self.mul_vec(x).iter().zip(x).map(|(a, b)| a * b).sum()
    }

    /// Solves `self · x = rhs` for a symmetric positive-definite `self` via
    /// Cholesky factorisation. Returns `None` if the matrix is not
    /// (numerically) positive definite.
    pub fn cholesky_solve(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        let n = self.n;
        debug_assert_eq!(rhs.len(), n);
        // Lower-triangular factor L with A = L Lᵀ, stored row-major.
        let mut l = vec![0.0; n * n];
        for j in 0..n {
            let mut s = self.get(j, j);
            for k in 0..j {
                s -= l[j * n + k] * l[j * n + k];
            }
            if !(s > 0.0) || !s.is_finite() {
                return None;
            }
            let ljj = s.sqrt();
            l[j * n + j] = ljj;
            for i in (j + 1)..n {
                let mut s = self.get(i, j);
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = s / ljj;
            }
        }
        // Forward substitution: L y = rhs
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = rhs[i];
            for k in 0..i {
                s -= l[i * n + k] * y[k];
            }
            y[i] = s / l[i * n + i];
        }
        // Back substitution: Lᵀ x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= l[k * n + i] * x[k];
            }
            x[i] = s / l[i * n + i];
        }
        Some(x)
    }
}

/// `y += alpha · x`
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    if alpha == 0.0 {
        return;
    }
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// Dot product.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Infinity norm.
pub fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solves_spd_system() {
        let mut a = Matrix::zeros(3);
        let vals = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]];
        for i in 0..3 {
            for j in 0..3 {
                a.set(i, j, vals[i][j]);
            }
        }
        let x_true = vec![1.0, -2.0, 0.5];
        let rhs = a.mul_vec(&x_true);
        let x = a.cholesky_solve(&rhs).unwrap();
        for (a, b) in x.iter().zip(&x_true) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn cholesky_rejects_indefinite() {
        let mut a = Matrix::identity(2);
        a.set(1, 1, -1.0);
        assert!(a.cholesky_solve(&[1.0, 1.0]).is_none());
    }

    #[test]
    fn outer_product_update() {
        let mut m = Matrix::zeros(2);
        m.add_scaled_outer(&[1.0, 2.0], 3.0);
        assert_eq!(m.get(0, 0), 3.0);
        assert_eq!(m.get(0, 1), 6.0);
        assert_eq!(m.get(1, 0), 6.0);
        assert_eq!(m.get(1, 1), 12.0);
    }
}
