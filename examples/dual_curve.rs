//! Fits an OIS discount curve and a 3M projection curve jointly and prints
//! the resulting smooth forwards, repricing errors and solver diagnostics.
//!
//! Run with `cargo run --example dual_curve`.

use quadsmooth::{CurveId, DualCurveFitter, FitConfig, Instrument};

fn main() -> Result<(), quadsmooth::Error> {
    let fitter = DualCurveFitter::new()
        .with_config(FitConfig {
            lambda_curvature: 1e-8,
            ..FitConfig::default()
        })
        // ---- OIS quotes (annual fixed vs compounded overnight) → discount curve
        .add(Instrument::ois(1.0, 0.0300, 1)?)
        .add(Instrument::ois(2.0, 0.0310, 1)?)
        .add(Instrument::ois(3.0, 0.0320, 1)?)
        .add(Instrument::ois(5.0, 0.0340, 1)?)
        .add(Instrument::ois(7.0, 0.0350, 1)?)
        .add(Instrument::ois(10.0, 0.0360, 1)?)
        .add(Instrument::ois(15.0, 0.0370, 1)?)
        .add(Instrument::ois(20.0, 0.0370, 1)?)
        .add(Instrument::ois(30.0, 0.0360, 1)?)
        // ---- 3M deposit, FRAs and IRS (annual fixed vs 3M float) → projection curve
        .add(Instrument::deposit(CurveId::Projection, 0.25, 0.0330)?)
        .add(Instrument::fra(CurveId::Projection, 0.25, 0.50, 0.0335)?)
        .add(Instrument::fra(CurveId::Projection, 0.50, 0.75, 0.0340)?)
        .add(Instrument::irs(1.0, 0.0340, 1, 4)?)
        .add(Instrument::irs(2.0, 0.0345, 1, 4)?)
        .add(Instrument::irs(3.0, 0.0350, 1, 4)?)
        .add(Instrument::irs(5.0, 0.0365, 1, 4)?)
        .add(Instrument::irs(7.0, 0.0375, 1, 4)?)
        .add(Instrument::irs(10.0, 0.0385, 1, 4)?)
        .add(Instrument::irs(15.0, 0.0395, 1, 4)?)
        .add(Instrument::irs(20.0, 0.0395, 1, 4)?)
        .add(Instrument::irs(30.0, 0.0385, 1, 4)?);

    let fit = fitter.fit()?;
    let r = &fit.report;
    println!(
        "converged: {}  iterations: {}  objective: {:.3e}  |grad|∞: {:.3e}  max |residual|: {:.3e}",
        r.converged,
        r.iterations,
        r.objective,
        r.gradient_norm,
        r.max_abs_residual()
    );
    println!(
        "parameters: {} (Hessian {}×{})\n",
        fit.curves.parameters().len(),
        r.hessian.dim(),
        r.hessian.dim()
    );

    println!(
        "{:<34} {:>9} {:>11} {:>12}",
        "instrument", "quote", "par", "resid (bp)"
    );
    for (inst, res) in fitter.instruments().iter().zip(&r.residuals) {
        let label = match inst {
            Instrument::SimpleRate { start, end, .. } if *start == 0.0 => {
                format!("deposit {:.2}y", end)
            }
            Instrument::SimpleRate { start, end, .. } => format!("FRA {:.2}y x {:.2}y", start, end),
            Instrument::Swap {
                float_curve: CurveId::Discount,
                ..
            } => {
                format!("OIS {:.0}y", inst.maturity())
            }
            Instrument::Swap { .. } => format!("IRS {:.0}y", inst.maturity()),
        };
        println!(
            "{:<34} {:>8.4}% {:>10.6}% {:>12.3e}",
            label,
            inst.quote() * 100.0,
            fit.curves.par_rate(inst)? * 100.0,
            res * 1e4
        );
    }

    println!(
        "\n{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "t", "f_disc", "f_proj", "basis(bp)", "z_disc", "z_proj"
    );
    let mut t = 0.0;
    while t <= 31.0 + 1e-9 {
        let fd = fit.curves.forward(CurveId::Discount, t)?;
        let fp = fit.curves.forward(CurveId::Projection, t)?;
        let zd = fit.curves.zero_rate(CurveId::Discount, t)?;
        let zp = fit.curves.zero_rate(CurveId::Projection, t)?;
        println!(
            "{:>6.2} {:>11.4}% {:>11.4}% {:>12.2} {:>11.4}% {:>11.4}%",
            t,
            fd * 100.0,
            fp * 100.0,
            (fp - fd) * 1e4,
            zd * 100.0,
            zp * 100.0
        );
        t += 1.0;
    }

    // Quote sensitivities via the Hessian: dθ/dq for the 10y IRS, pushed
    // through to the 10y projection zero rate.
    let weights = vec![1.0; fitter.instruments().len()];
    let sens = fit.quote_sensitivities(fitter.instruments(), &weights)?;
    let idx = fitter
        .instruments()
        .iter()
        .position(|i| i.knot_curve() == CurveId::Projection && i.maturity() == 10.0)
        .unwrap();
    let g = fit
        .curves
        .basis()
        .integral_basis_full(CurveId::Projection, 10.0)?;
    let dz_dq: f64 = g.iter().zip(&sens[idx]).map(|(a, b)| a * b).sum::<f64>() / 10.0;
    println!("\nd(10y projection zero)/d(10y IRS quote) = {dz_dq:.4}");

    Ok(())
}
