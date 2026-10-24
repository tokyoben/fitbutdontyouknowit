//! Fits a JPY-style swaption cube on top of the TONA OIS curve and prices
//! 2y-into-2y swaptions off it.
//!
//! Forwards come from the quad-smooth TONA curve (same benchmarks as the
//! `jpy_2y2y` example). Market vols are *synthetic*: generated from a smooth
//! "true" SABR cube, perturbed with ±0.3bp noise, with two nodes dropped to
//! show the smoothness term filling gaps.
//!
//! Run with `cargo run --example swaption_cube`.

use quadsmooth::{
    CubeConfig, CurveId, DualCurveFitter, FitConfig, Instrument, OptionType, SabrCube, SabrParams,
    Strike, SwaptionGrid, VolQuote,
};

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
const TONA_FIXING: f64 = 0.01227;

const BETA: f64 = 0.5;
const SHIFT: f64 = 0.02;

/// Deterministic noise in [−1, 1).
fn noise(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn main() -> Result<(), quadsmooth::Error> {
    // ---- Discount/projection curve (TONA OIS) -------------------------------
    let mut fitter = DualCurveFitter::new().with_config(FitConfig {
        lambda_curvature: 1e-10,
        ..FitConfig::default()
    });
    fitter = fitter.add(Instrument::deposit(
        CurveId::Discount,
        1.0 / 365.0,
        TONA_FIXING,
    )?);
    for &(t, r) in TONA_OIS {
        fitter = fitter.add(Instrument::ois(t, r, 1)?);
    }
    let curve = fitter.fit()?.curves;

    // ---- Grid with forwards from the curve ----------------------------------
    let expiries = vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
    let tenors = vec![1.0, 2.0, 5.0, 10.0];
    let grid = SwaptionGrid::from_curve(
        expiries.clone(),
        tenors.clone(),
        &curve,
        1,
        1,
        CurveId::Discount,
    )?;

    println!("Forward swap rates (%), expiry × tenor:");
    print!("{:>6}", "");
    for t in &tenors {
        print!("{:>9.0}y", t);
    }
    println!();
    for (i, e) in expiries.iter().enumerate() {
        print!("{:>5.0}y", e);
        for j in 0..tenors.len() {
            print!("{:>10.4}", grid.forward(i, j) * 100.0);
        }
        println!();
    }

    // ---- Synthetic "true" cube and noisy quotes -----------------------------
    let mut truth = Vec::new();
    for (i, &e) in expiries.iter().enumerate() {
        for (j, &t) in tenors.iter().enumerate() {
            let atm = 0.0055 + 0.0010 * (1.0 + e).ln() + 0.0004 * t.ln();
            truth.push(SabrParams {
                alpha: atm / (grid.forward(i, j) + SHIFT).powf(BETA),
                beta: BETA,
                rho: -0.15 - 0.03 * (1.0 + e).ln(),
                nu: 0.55 / (1.0 + e).powf(0.3),
                shift: SHIFT,
            });
        }
    }
    let true_cube = SabrCube::new(grid.clone(), BETA, SHIFT, &truth)?;

    let offsets_bp = [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0];
    let dropped = [(2usize, 0usize), (4, 1)]; // 3y×1y and 7y×2y missing from the broker run
    let mut seed = 42u64;
    let mut quotes = Vec::new();
    for (i, &e) in expiries.iter().enumerate() {
        for (j, &t) in tenors.iter().enumerate() {
            if dropped.contains(&(i, j)) {
                continue;
            }
            for &o in &offsets_bp {
                let v = true_cube.node_vol(i, j, Strike::Offset(o * 1e-4))?;
                let v = v + 0.3e-4 * noise(&mut seed);
                quotes.push(VolQuote::offset_bp(e, t, o, v));
            }
        }
    }

    // ---- Fit ----------------------------------------------------------------
    let cfg = CubeConfig {
        beta: BETA,
        shift: SHIFT,
        ..CubeConfig::default()
    };
    let fit = SabrCube::fit(grid.clone(), &quotes, &cfg)?;
    let r = &fit.report;
    println!(
        "\nCube fit: {} quotes, {} nodes ({} unquoted), converged={} iters={} rmse={:.3}bp max|res|={:.3}bp",
        quotes.len(),
        grid.nodes(),
        dropped.len(),
        r.converged,
        r.iterations,
        r.rmse_bp,
        r.max_abs_residual_bp()
    );

    println!(
        "\n{:>6} {:>6} {:>9} {:>8} {:>7} {:>7} {:>10}",
        "expiry", "tenor", "ATM(bp)", "alpha", "rho", "nu", "true ATM"
    );
    for (i, &e) in expiries.iter().enumerate() {
        for (j, &t) in tenors.iter().enumerate() {
            let p = fit.cube.node(i, j);
            let atm = fit.cube.node_vol(i, j, Strike::Atm)? * 1e4;
            let atm_true = true_cube.node_vol(i, j, Strike::Atm)? * 1e4;
            let flag = if dropped.contains(&(i, j)) { " *" } else { "" };
            println!(
                "{:>5.0}y {:>5.0}y {:>9.2} {:>8.4} {:>7.3} {:>7.3} {:>10.2}{}",
                e, t, atm, p.alpha, p.rho, p.nu, atm_true, flag
            );
        }
    }
    println!("  * node had no quotes; filled by the smoothness penalty");

    let arb = fit.cube.arbitrage_report(0.0300, 0.0005);
    println!(
        "\nButterfly arbitrage check (F±300bp): {} — min implied density {:.3e}",
        if arb.is_free() { "free" } else { "VIOLATIONS" },
        arb.min_density
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    );

    // ---- 2y into 2y swaptions ----------------------------------------------
    let (e, t) = (2.0, 2.0);
    let fwd_swap = quadsmooth::cube::forward_swap(e, e + t, 1, 1, CurveId::Discount)?;
    let f = curve.par_rate(&fwd_swap)?;
    let annuity = if let Instrument::Swap { fixed, .. } = &fwd_swap {
        fixed
            .periods()
            .map(|(_, ti, tau)| tau * curve.discount_factor(CurveId::Discount, ti).unwrap())
            .sum::<f64>()
    } else {
        unreachable!()
    };
    println!(
        "\n2y2y: forward {:.4}%  annuity {:.6}  (per unit notional)",
        f * 100.0,
        annuity
    );
    println!(
        "{:>10} {:>9} {:>11} {:>12}",
        "strike", "vol(bp)", "payer", "receiver"
    );
    for off_bp in [-100.0, -50.0, 0.0, 50.0, 100.0] {
        let k = f + off_bp * 1e-4;
        let vol = fit.cube.vol_at(e, t, f, k)?;
        let payer = fit.cube.price(OptionType::Call, e, t, f, k, annuity)?;
        let recv = fit.cube.price(OptionType::Put, e, t, f, k, annuity)?;
        println!(
            "{:>+7.0}bp {:>9.2} {:>11.6} {:>12.6}",
            off_bp,
            vol * 1e4,
            payer,
            recv
        );
    }

    // Off-grid interpolation: 4y into 3y.
    let p = fit.cube.params_at(4.0, 3.0)?;
    println!(
        "\n4y3y interpolated SABR: alpha {:.4} rho {:.3} nu {:.3} (between 3y/5y × 2y/5y nodes)",
        p.alpha, p.rho, p.nu
    );
    Ok(())
}
