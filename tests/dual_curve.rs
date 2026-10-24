use quadsmooth::{CurveId, DualCurveFitter, FitConfig, Instrument};

fn market() -> DualCurveFitter {
    DualCurveFitter::new()
        // OIS → discount curve
        .add(Instrument::ois(1.0, 0.0300, 1).unwrap())
        .add(Instrument::ois(2.0, 0.0310, 1).unwrap())
        .add(Instrument::ois(3.0, 0.0320, 1).unwrap())
        .add(Instrument::ois(5.0, 0.0340, 1).unwrap())
        .add(Instrument::ois(7.0, 0.0350, 1).unwrap())
        .add(Instrument::ois(10.0, 0.0360, 1).unwrap())
        .add(Instrument::ois(15.0, 0.0370, 1).unwrap())
        .add(Instrument::ois(20.0, 0.0370, 1).unwrap())
        .add(Instrument::ois(30.0, 0.0360, 1).unwrap())
        // 3M deposit, FRAs, IRS (annual fixed vs 3M float) → projection curve
        .add(Instrument::deposit(CurveId::Projection, 0.25, 0.0330).unwrap())
        .add(Instrument::fra(CurveId::Projection, 0.25, 0.50, 0.0335).unwrap())
        .add(Instrument::fra(CurveId::Projection, 0.50, 0.75, 0.0340).unwrap())
        .add(Instrument::irs(1.0, 0.0340, 1, 4).unwrap())
        .add(Instrument::irs(2.0, 0.0345, 1, 4).unwrap())
        .add(Instrument::irs(3.0, 0.0350, 1, 4).unwrap())
        .add(Instrument::irs(5.0, 0.0365, 1, 4).unwrap())
        .add(Instrument::irs(7.0, 0.0375, 1, 4).unwrap())
        .add(Instrument::irs(10.0, 0.0385, 1, 4).unwrap())
        .add(Instrument::irs(15.0, 0.0395, 1, 4).unwrap())
        .add(Instrument::irs(20.0, 0.0395, 1, 4).unwrap())
        .add(Instrument::irs(30.0, 0.0385, 1, 4).unwrap())
}

#[test]
fn dual_curve_reprices_all_instruments() {
    let fitter = market();
    let fit = fitter.fit().unwrap();
    assert!(fit.report.converged, "did not converge: {:?}", fit.report);
    assert!(
        fit.report.iterations <= 20,
        "too many iterations: {}",
        fit.report.iterations
    );
    assert!(
        fit.report.max_abs_residual() < 1e-7,
        "max residual {}",
        fit.report.max_abs_residual()
    );
    // Par rates computed directly from the curves match the quotes.
    for inst in fitter.instruments() {
        let par = fit.curves.par_rate(inst).unwrap();
        assert!(
            (par - inst.quote()).abs() < 1e-7,
            "par {par} vs quote {} for {inst:?}",
            inst.quote()
        );
    }
}

#[test]
fn tighter_penalty_gives_exact_repricing() {
    let cfg = FitConfig {
        lambda_curvature: 1e-12,
        ..FitConfig::default()
    };
    let fit = market().with_config(cfg).fit().unwrap();
    assert!(fit.report.converged);
    assert!(fit.report.max_abs_residual() < 1e-10);
}

#[test]
fn forwards_are_smooth_and_curves_differ() {
    let fit = market().fit().unwrap();
    for curve in [CurveId::Discount, CurveId::Projection] {
        let c = fit.curves.curve(curve);
        for &k in &c.knots()[1..c.knots().len() - 1] {
            let eps = 1e-7;
            let jump = c.forward(k + eps).unwrap() - c.forward(k - eps).unwrap();
            let slope_jump = c.forward_slope(k + eps).unwrap() - c.forward_slope(k - eps).unwrap();
            assert!(jump.abs() < 1e-8, "forward jump {jump} at {k} on {curve}");
            assert!(
                slope_jump.abs() < 1e-8,
                "slope jump {slope_jump} at {k} on {curve}"
            );
        }
        // Reasonable forward levels everywhere.
        let mut t = 0.0;
        while t <= 30.0 {
            let f = c.forward(t).unwrap();
            assert!(f > 0.0 && f < 0.08, "forward {f} at {t} on {curve}");
            t += 0.05;
        }
    }
    // The IBOR curve sits above OIS (positive basis in the quotes).
    for &t in &[1.0, 5.0, 10.0, 20.0] {
        let d = fit.curves.zero_rate(CurveId::Discount, t).unwrap();
        let p = fit.curves.zero_rate(CurveId::Projection, t).unwrap();
        assert!(p > d, "projection {p} not above discount {d} at {t}");
    }
}

#[test]
fn hessian_quote_sensitivities_predict_bumped_refit() {
    // At the optimum ∇F ≈ 0, and the Hessian-based quote sensitivities
    // (implicit function theorem) must predict the parameter moves of a
    // refit under a small quote bump to second order.
    let fitter = market();
    let fit = fitter.fit().unwrap();
    assert!(fit.report.gradient_norm < 1e-10);

    let weights = vec![1.0; fitter.instruments().len()];
    let sens = fit
        .quote_sensitivities(fitter.instruments(), &weights)
        .unwrap();

    // Bump the 5y IRS quote by 1bp and refit.
    let bump = 1e-4;
    let idx = fitter
        .instruments()
        .iter()
        .position(|i| {
            matches!(
                i,
                Instrument::Swap {
                    float_curve: CurveId::Projection,
                    ..
                }
            ) && i.maturity() == 5.0
        })
        .unwrap();
    let mut bumped = DualCurveFitter::new();
    for (j, inst) in fitter.instruments().iter().enumerate() {
        let inst = if j == idx {
            match inst {
                Instrument::Swap {
                    fixed_rate,
                    fixed,
                    float,
                    float_curve,
                    discount_curve,
                } => Instrument::Swap {
                    fixed_rate: fixed_rate + bump,
                    fixed: fixed.clone(),
                    float: float.clone(),
                    float_curve: *float_curve,
                    discount_curve: *discount_curve,
                },
                other => other.clone(),
            }
        } else {
            inst.clone()
        };
        bumped = bumped.add(inst);
    }
    let fit_b = bumped.fit().unwrap();
    let predicted: Vec<f64> = fit
        .curves
        .parameters()
        .iter()
        .zip(&sens[idx])
        .map(|(t, s)| t + s * bump)
        .collect();
    for (p, a) in predicted.iter().zip(fit_b.curves.parameters()) {
        // Second-order effects of a 1bp bump are ~1e-8 at most.
        assert!((p - a).abs() < 1e-7, "predicted {p} vs actual {a}");
    }
    // The bump should move the projection 5y zero rate by roughly 1bp.
    let z0 = fit.curves.zero_rate(CurveId::Projection, 5.0).unwrap();
    let z1 = fit_b.curves.zero_rate(CurveId::Projection, 5.0).unwrap();
    assert!((z1 - z0 - bump).abs() < 0.3 * bump, "zero move {}", z1 - z0);
    // …and leave the discount curve essentially untouched (the joint
    // penalised fit couples the curves only at O(λ), here ~1e-10).
    let d0 = fit.curves.zero_rate(CurveId::Discount, 5.0).unwrap();
    let d1 = fit_b.curves.zero_rate(CurveId::Discount, 5.0).unwrap();
    assert!((d1 - d0).abs() < 1e-8, "discount moved by {}", d1 - d0);
}

#[test]
fn single_curve_mode_copies_discount_to_projection() {
    let fit = DualCurveFitter::new()
        .add(Instrument::ois(1.0, 0.03, 1).unwrap())
        .add(Instrument::ois(2.0, 0.032, 1).unwrap())
        .add(Instrument::ois(5.0, 0.035, 1).unwrap())
        .fit()
        .unwrap();
    assert!(fit.report.single_curve);
    assert!(fit.report.converged);
    assert!(fit.report.max_abs_residual() < 1e-7);
    for &t in &[0.5, 1.0, 3.3, 5.0, 7.0] {
        let d = fit.curves.forward(CurveId::Discount, t).unwrap();
        let p = fit.curves.forward(CurveId::Projection, t).unwrap();
        assert_eq!(d, p);
    }
}

#[test]
fn flat_market_recovers_flat_forward() {
    // A flat 4% continuously compounded curve: OIS par rates are then
    // (exp(0.04) − 1) for annual payments regardless of maturity.
    let r = 0.04_f64;
    let par = r.exp() - 1.0;
    let fit = DualCurveFitter::new()
        .add(Instrument::ois(1.0, par, 1).unwrap())
        .add(Instrument::ois(3.0, par, 1).unwrap())
        .add(Instrument::ois(10.0, par, 1).unwrap())
        .add(Instrument::irs(2.0, par, 1, 1).unwrap())
        .add(Instrument::irs(7.0, par, 1, 1).unwrap())
        .fit()
        .unwrap();
    assert!(fit.report.converged);
    for curve in [CurveId::Discount, CurveId::Projection] {
        for &t in &[0.0, 0.7, 2.5, 6.0, 10.0] {
            let f = fit.curves.forward(curve, t).unwrap();
            assert!((f - r).abs() < 1e-7, "{curve} forward {f} at {t}");
        }
    }
}

#[test]
fn errors_are_reported() {
    assert!(DualCurveFitter::new().fit().is_err());
    // Only projection instruments: discount curve is undefined.
    let res = DualCurveFitter::new()
        .add(Instrument::irs(2.0, 0.03, 1, 4).unwrap())
        .fit();
    assert_eq!(
        res.unwrap_err(),
        quadsmooth::Error::NoInstruments(CurveId::Discount)
    );
}
