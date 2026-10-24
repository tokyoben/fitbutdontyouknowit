//! Prices a JPY 2y2y forward-starting swap from spot TONA OIS benchmarks.
//!
//! Discounting: TONA OIS curve. Projection: (a) TONA itself — the JPY
//! market's standard swap is now the TONA OIS, so 2y2y is simply the forward
//! OIS rate; (b) an illustrative 6M TIBOR projection curve built from IRS
//! quotes = TONA OIS + a TIBOR/TONA basis, to exercise the dual-curve path.
//!
//! Run with `cargo run --example jpy_2y2y`.

use quadsmooth::{CurveId, DualCurveFitter, FitConfig, Instrument, Schedule};

/// Market snapshot (see README of this example / chat for sources).
const AS_OF: &str = "2026-10-21 (TONA OIS mids, London 09:00)";

/// TONA OIS par rates: (maturity years, rate). Annual fixed vs annual
/// compounded TONA, ACT/365F — treated here in year-fraction terms.
const TONA_OIS: &[(f64, f64)] = &[
    (1.0, 0.0165),
    (2.0, 0.0190),
    (3.0, 0.0205),
    (4.0, 0.0217),
    (5.0, 0.0228),
    (7.0, 0.0249),
    (10.0, 0.0277),
    (20.0, 0.0334),
    (30.0, 0.0350),
];

/// TONA fixing (1D OIS), used as a short-end anchor.
const TONA_FIXING: f64 = 0.01227;

/// Indicative 6M TIBOR vs TONA basis (bp) applied to build the illustrative
/// TIBOR projection curve. The only quote found was an undated 6M-tenor mid of
/// ~14.4bp; it is used flat across the curve purely for illustration.
const TIBOR_TONA_BASIS_BP: f64 = 14.4;

/// 2y2y forward swap: start 2y, end 4y.
const FWD_START: f64 = 2.0;
const FWD_END: f64 = 4.0;

fn forward_swap(
    start: f64,
    end: f64,
    fixed_freq: u32,
    float_freq: u32,
    float_curve: CurveId,
) -> Result<Instrument, quadsmooth::Error> {
    let sched = |freq: u32| {
        let n = ((end - start) * freq as f64).round() as usize;
        Schedule::from_times((0..=n).map(|k| start + k as f64 / freq as f64).collect())
    };
    Ok(Instrument::Swap {
        fixed_rate: 0.0, // irrelevant for par_rate()
        fixed: sched(fixed_freq)?,
        float: sched(float_freq)?,
        float_curve,
        discount_curve: CurveId::Discount,
    })
}

fn main() -> Result<(), quadsmooth::Error> {
    println!("JPY 2y2y pricing — market as of {AS_OF}\n");

    // ---------------------------------------------------------------
    // (a) TONA-only: discount = projection = TONA OIS curve.
    // ---------------------------------------------------------------
    let mut tona = DualCurveFitter::new().with_config(FitConfig {
        lambda_curvature: 1e-10,
        ..FitConfig::default()
    });
    // 1-day deposit at the TONA fixing anchors the very short end.
    tona = tona.add(Instrument::deposit(
        CurveId::Discount,
        1.0 / 365.0,
        TONA_FIXING,
    )?);
    for &(t, r) in TONA_OIS {
        tona = tona.add(Instrument::ois(t, r, 1)?);
    }
    let fit_a = tona.fit()?;
    println!(
        "TONA OIS fit: converged={} iters={} max|resid|={:.2e} (rate units)",
        fit_a.report.converged,
        fit_a.report.iterations,
        fit_a.report.max_abs_residual()
    );
    println!("{:>5} {:>9} {:>9} {:>10}", "t", "fwd", "zero", "DF");
    for &t in &[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 30.0] {
        println!(
            "{:>5.1} {:>8.4}% {:>8.4}% {:>10.6}",
            t,
            fit_a.curves.forward(CurveId::Discount, t)? * 100.0,
            fit_a.curves.zero_rate(CurveId::Discount, t)? * 100.0,
            fit_a.curves.discount_factor(CurveId::Discount, t)?
        );
    }

    let fwd_ois = forward_swap(FWD_START, FWD_END, 1, 1, CurveId::Discount)?;
    let r_2y2y_ois = fit_a.curves.par_rate(&fwd_ois)?;
    let df2 = fit_a.curves.discount_factor(CurveId::Discount, FWD_START)?;
    let df4 = fit_a.curves.discount_factor(CurveId::Discount, FWD_END)?;
    let cc_2y2y = (df2 / df4).ln() / (FWD_END - FWD_START);
    println!(
        "\n2y2y TONA OIS par rate (annual/annual):   {:.4}%",
        r_2y2y_ois * 100.0
    );
    println!(
        "2y2y continuously compounded forward:     {:.4}%   (P(2)/P(4) = {:.6})",
        cc_2y2y * 100.0,
        df2 / df4
    );

    // ---------------------------------------------------------------
    // (b) Dual curve: TONA discounting, illustrative 6M TIBOR projection.
    //     IRS quotes = TONA OIS + flat basis; semi-annual fixed vs 6M float.
    // ---------------------------------------------------------------
    let basis = TIBOR_TONA_BASIS_BP * 1e-4;
    let mut dual = DualCurveFitter::new().with_config(FitConfig {
        lambda_curvature: 1e-10,
        ..FitConfig::default()
    });
    dual = dual.add(Instrument::deposit(
        CurveId::Discount,
        1.0 / 365.0,
        TONA_FIXING,
    )?);
    for &(t, r) in TONA_OIS {
        dual = dual.add(Instrument::ois(t, r, 1)?);
    }
    // 6M TIBOR deposit ≈ 6M TONA OIS + basis (TONA OIS 6M interpolated from the fit).
    let tona_6m = fit_a.curves.simple_forward(CurveId::Discount, 0.0, 0.5)?;
    dual = dual.add(Instrument::deposit(
        CurveId::Projection,
        0.5,
        tona_6m + basis,
    )?);
    for &(t, r) in TONA_OIS {
        dual = dual.add(Instrument::irs(t, r + basis, 2, 2)?);
    }
    let fit_b = dual.fit()?;
    println!(
        "\nDual-curve fit: converged={} iters={} max|resid|={:.2e}",
        fit_b.report.converged,
        fit_b.report.iterations,
        fit_b.report.max_abs_residual()
    );
    println!(
        "{:>5} {:>10} {:>10} {:>10}",
        "t", "f_TONA", "f_TIBOR", "basis bp"
    );
    for &t in &[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0] {
        let fd = fit_b.curves.forward(CurveId::Discount, t)?;
        let fp = fit_b.curves.forward(CurveId::Projection, t)?;
        println!(
            "{:>5.1} {:>9.4}% {:>9.4}% {:>10.2}",
            t,
            fd * 100.0,
            fp * 100.0,
            (fp - fd) * 1e4
        );
    }
    let fwd_tibor = forward_swap(FWD_START, FWD_END, 2, 2, CurveId::Projection)?;
    let r_2y2y_tibor = fit_b.curves.par_rate(&fwd_tibor)?;
    println!(
        "\n2y2y vs 6M TIBOR, TONA-discounted (s.a./s.a.): {:.4}%  (basis to OIS 2y2y: {:.1}bp)",
        r_2y2y_tibor * 100.0,
        (r_2y2y_tibor - r_2y2y_ois) * 1e4
    );

    // Hessian-based risk of the 2y2y OIS rate to each TONA benchmark (per 1bp).
    let insts = tona.instruments();
    let weights = vec![1.0; insts.len()];
    let sens = fit_a.quote_sensitivities(insts, &weights)?;
    println!("\n2y2y OIS rate sensitivity to benchmark quotes (bp per 1bp bump):");
    for (inst, dtheta) in insts.iter().zip(&sens) {
        // Finite-difference-free: shift θ by dθ/dq·1bp and reprice.
        let mut theta = fit_a.curves.parameters().to_vec();
        for (t, d) in theta.iter_mut().zip(dtheta) {
            *t += d * 1e-4;
        }
        let disc = fit_a.curves.basis().curve(CurveId::Discount, &theta)?;
        // Reprice the forward swap on the bumped curve.
        let (mut annuity, mut float_pv) = (0.0, 0.0);
        if let Instrument::Swap { fixed, float, .. } = &fwd_ois {
            for (_, t, tau) in fixed.periods() {
                annuity += tau * disc.discount_factor(t)?;
            }
            for (s0, s1, tau) in float.periods() {
                float_pv += tau * disc.simple_forward(s0, s1)? * disc.discount_factor(s1)?;
            }
        }
        let label = match inst {
            Instrument::SimpleRate { .. } => "TONA 1D".to_string(),
            _ => format!("OIS {:>2.0}y", inst.maturity()),
        };
        println!(
            "  {label:<9} {:>7.3}",
            (float_pv / annuity - r_2y2y_ois) * 1e4
        );
    }
    Ok(())
}
