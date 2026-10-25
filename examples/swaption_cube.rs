//! Fits a JPY-style swaption cube on top of the TONA OIS curve and prices
//! 2y-into-2y swaptions off it.
//!
//! Forwards come from the quad-smooth TONA curve (same benchmarks as the
//! `jpy_2y2y` example). Market vols are *synthetic*: generated from a smooth
//! "true" SABR cube, perturbed with ±0.3bp noise, with two nodes dropped to
//! show the smoothness term filling gaps.
//!
//! Also writes three charts and a table of the fitted cube to `docs/`:
//! `vol_smiles.svg`, `vol_atm_surface.svg`, `sabr_params.svg` and
//! `vol_cube.csv`.
//!
//! Run with `cargo run --example swaption_cube`.

use std::fmt::Write as _;
use std::fs;

use quadsmooth::{
    CubeConfig, CurveId, DualCurveFitter, FitConfig, Instrument, OptionType, SabrCube, SabrParams,
    Strike, SwaptionGrid, VolQuote,
};

#[path = "common/svg.rs"]
mod svg;
use svg::{Key, LegendItem, LinePanel, Series, INK, INK_2, MUTED, SERIES};

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

    // ---- Charts -------------------------------------------------------------
    fs::create_dir_all("docs").expect("create docs/");
    let ctx = ChartData {
        expiries: &expiries,
        tenors: &tenors,
        quotes: &quotes,
        dropped: &dropped,
        fit: &fit.cube,
        truth: &true_cube,
        offsets_bp: &offsets_bp,
    };
    fs::write("docs/vol_smiles.svg", smiles_chart(&ctx)?).expect("write svg");
    fs::write("docs/vol_atm_surface.svg", atm_heatmap(&ctx)?).expect("write svg");
    fs::write("docs/sabr_params.svg", params_chart(&ctx)).expect("write svg");
    fs::write("docs/vol_cube.csv", cube_table(&ctx)?).expect("write csv");
    println!("\nwrote docs/vol_smiles.svg, docs/vol_atm_surface.svg, docs/sabr_params.svg, docs/vol_cube.csv");
    Ok(())
}

struct ChartData<'a> {
    expiries: &'a [f64],
    tenors: &'a [f64],
    quotes: &'a [VolQuote],
    dropped: &'a [(usize, usize)],
    fit: &'a SabrCube,
    truth: &'a SabrCube,
    offsets_bp: &'a [f64],
}

const W: f64 = 960.0;
const BP: f64 = 1e4;

fn bp_label(x: f64) -> String {
    if x == 0.0 {
        "ATM".to_string()
    } else {
        format!("{x:+.0}")
    }
}

/// Fitted smile across strike offsets at node (i, j), in bp.
fn smile(cube: &SabrCube, i: usize, j: usize) -> Result<Vec<(f64, f64)>, quadsmooth::Error> {
    (-150..=150)
        .step_by(2)
        .map(|o| {
            let o = o as f64;
            Ok((o, cube.node_vol(i, j, Strike::Offset(o / BP))? * BP))
        })
        .collect()
}

/// Smiles at four expiries for the 5y tenor with market quotes, and the
/// two unquoted nodes against the vols the true cube would have shown.
fn smiles_chart(c: &ChartData) -> Result<String, quadsmooth::Error> {
    let panel_h = 430.0;
    let h = 2.0 * panel_h + 24.0;
    let j = c.tenors.iter().position(|&t| t == 5.0).unwrap();
    let shown = [1.0, 2.0, 5.0, 10.0];
    let names: Vec<String> = shown.iter().map(|e| format!("{e:.0}y expiry")).collect();
    let mut top = Vec::new();
    for (k, &e) in shown.iter().enumerate() {
        let i = c.expiries.iter().position(|&x| x == e).unwrap();
        let mut s = Series::line(&names[k], SERIES[k], smile(c.fit, i, j)?);
        s.dots = c
            .quotes
            .iter()
            .filter(|q| q.expiry == e && q.tenor == 5.0)
            .map(|q| match q.strike {
                Strike::Offset(o) => (o * BP, q.normal_vol * BP),
                _ => unreachable!(),
            })
            .collect();
        top.push(s);
    }

    let gap_names: Vec<String> = c
        .dropped
        .iter()
        .map(|&(i, j)| format!("{:.0}y × {:.0}y", c.expiries[i], c.tenors[j]))
        .collect();
    let mut bottom = Vec::new();
    for (k, &(i, j)) in c.dropped.iter().enumerate() {
        let mut s = Series::line(&gap_names[k], SERIES[k], smile(c.fit, i, j)?);
        s.open_dots = c
            .offsets_bp
            .iter()
            .map(|&o| Ok((o, c.truth.node_vol(i, j, Strike::Offset(o / BP))? * BP)))
            .collect::<Result<_, quadsmooth::Error>>()?;
        bottom.push(s);
    }

    let x_ticks: Vec<(f64, String)> = (-3..=3)
        .map(|k| (50.0 * k as f64, bp_label(50.0 * k as f64)))
        .collect();
    let y_fmt = |y: f64| format!("{y:.0}bp");
    let mut out = String::new();
    svg::open(
        &mut out,
        W,
        h,
        "Swaption smiles from the fitted SABR cube",
        "Normal volatility against strike offset from the forward. Top: 1y, 2y, 5y and 10y expiries into 5y swaps, fitted smiles with the market quotes. Bottom: the two grid nodes with no quotes, fitted smiles against the vols the generating cube would have quoted. Values in docs/vol_cube.csv.",
    );
    svg::line_panel(
        &mut out,
        &LinePanel {
            x: 0.0,
            y: 0.0,
            w: W,
            h: panel_h,
            title: "Fitted smiles, expiries into 5y swaps",
            subtitle: "Normal volatility by strike offset from the forward swap rate. Dots are the market quotes the cube was fitted to.",
            series: &top,
            extra_legend: &[LegendItem {
                label: "Market quote",
                color: INK_2,
                key: Key::Dot,
            }],
            x_range: (-150.0, 150.0),
            x_ticks: &x_ticks,
            y_step: 5.0,
            y_fmt: &y_fmt,
            x_marks: &[],
            note: Some("Strike offset from the forward, bp"),
        },
    );
    svg::rule(&mut out, svg::ML, W - svg::MR, panel_h + 12.0);
    svg::line_panel(
        &mut out,
        &LinePanel {
            x: 0.0,
            y: panel_h + 24.0,
            w: W,
            h: panel_h,
            title: "Nodes with no quotes, filled from their neighbours",
            subtitle: "These smiles come only from the smoothness penalty. Open circles are what the market would have quoted (not used in the fit).",
            series: &bottom,
            extra_legend: &[LegendItem {
                label: "Would-be quote, not fitted",
                color: INK_2,
                key: Key::OpenDot,
            }],
            x_range: (-150.0, 150.0),
            x_ticks: &x_ticks,
            y_step: 5.0,
            y_fmt: &y_fmt,
            x_marks: &[],
            note: Some("Strike offset from the forward, bp"),
        },
    );
    svg::close(&mut out);
    Ok(out)
}

/// Heatmap of ATM normal vol over expiry × tenor.
fn atm_heatmap(c: &ChartData) -> Result<String, quadsmooth::Error> {
    let (ne, nt) = (c.expiries.len(), c.tenors.len());
    let mut atm = vec![vec![0.0; nt]; ne];
    for (i, row) in atm.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = c.fit.node_vol(i, j, Strike::Atm)? * BP;
        }
    }
    let lo = atm.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
    let hi = atm
        .iter()
        .flatten()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let (left, top, cell_h, gap) = (svg::ML + 24.0, 136.0, 58.0, 2.0);
    let cell_w = (W - left - svg::MR) / nt as f64;
    let h = top + ne as f64 * cell_h + 96.0;
    let mut out = String::new();
    svg::open(
        &mut out,
        W,
        h,
        "ATM swaption volatility surface",
        "At-the-money normal volatility in basis points for each expiry and swap tenor of the fitted SABR cube. Two nodes had no market quotes and were filled by the smoothness penalty. Values in docs/vol_cube.csv.",
    );
    svg::header(
        &mut out,
        svg::ML,
        0.0,
        "ATM normal volatility (bp), expiry × tenor",
        "Fitted SABR cube on the TONA curve. Darker is higher volatility. Cells marked “no quotes” were filled by the smoothness penalty.",
    );
    svg::text(
        &mut out,
        left + nt as f64 * cell_w / 2.0,
        top - 34.0,
        12.5,
        INK_2,
        "middle",
        "Swap tenor",
    );
    for (j, t) in c.tenors.iter().enumerate() {
        svg::text(
            &mut out,
            left + (j as f64 + 0.5) * cell_w,
            top - 12.0,
            12.5,
            MUTED,
            "middle",
            &format!("{t:.0}y"),
        );
    }
    let _ = writeln!(
        out,
        r#"<text x="{0}" y="{1}" font-size="12.5" fill="{INK_2}" text-anchor="middle" transform="rotate(-90 {0} {1})">Option expiry</text>"#,
        svg::ML - 40.0,
        top + ne as f64 * cell_h / 2.0
    );
    for (i, e) in c.expiries.iter().enumerate() {
        let y = top + i as f64 * cell_h;
        svg::text(
            &mut out,
            left - 10.0,
            y + cell_h / 2.0 + 4.0,
            12.5,
            MUTED,
            "end",
            &format!("{e:.0}y"),
        );
        for j in 0..nt {
            let x = left + j as f64 * cell_w;
            let (fill, white) = svg::ramp((atm[i][j] - lo) / (hi - lo));
            let ink = if white { "#ffffff" } else { INK };
            let _ = writeln!(
                out,
                r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" rx="3" fill="{fill}"/>"#,
                x + gap / 2.0,
                y + gap / 2.0,
                cell_w - gap,
                cell_h - gap
            );
            let unquoted = c.dropped.contains(&(i, j));
            let cy = y + cell_h / 2.0 + if unquoted { -1.0 } else { 5.0 };
            let _ = writeln!(
                out,
                r#"<text x="{:.1}" y="{cy:.1}" font-size="14" font-weight="600" fill="{ink}" text-anchor="middle" style="font-variant-numeric: tabular-nums">{:.1}</text>"#,
                x + cell_w / 2.0,
                atm[i][j]
            );
            if unquoted {
                svg::text(
                    &mut out,
                    x + cell_w / 2.0,
                    y + cell_h / 2.0 + 15.0,
                    11.5,
                    ink,
                    "middle",
                    "no quotes",
                );
            }
        }
    }

    // Colour scale.
    let sy = top + ne as f64 * cell_h + 34.0;
    let (sx0, sw) = (left, 240.0);
    let _ = writeln!(out, r#"<defs><linearGradient id="ramp">"#);
    for k in 0..=12 {
        let _ = writeln!(
            out,
            r#"<stop offset="{:.3}" stop-color="{}"/>"#,
            k as f64 / 12.0,
            svg::ramp(k as f64 / 12.0).0
        );
    }
    let _ = writeln!(out, r#"</linearGradient></defs>"#);
    let _ = writeln!(
        out,
        r#"<rect x="{sx0}" y="{sy}" width="{sw}" height="10" rx="2" fill="url(#ramp)"/>"#
    );
    svg::text(
        &mut out,
        sx0,
        sy + 28.0,
        12.0,
        MUTED,
        "start",
        &format!("{lo:.1}bp"),
    );
    svg::text(
        &mut out,
        sx0 + sw,
        sy + 28.0,
        12.0,
        MUTED,
        "end",
        &format!("{hi:.1}bp"),
    );
    svg::close(&mut out);
    Ok(out)
}

/// ρ and ν term structures by tenor.
fn params_chart(c: &ChartData) -> String {
    let panel_h = 400.0;
    let h = 2.0 * panel_h + 24.0;
    let names: Vec<String> = c.tenors.iter().map(|t| format!("{t:.0}y tenor")).collect();
    let build = |param: fn(&SabrParams) -> f64| -> Vec<Series> {
        c.tenors
            .iter()
            .enumerate()
            .map(|(j, _)| {
                let pts: Vec<(f64, f64)> = c
                    .expiries
                    .iter()
                    .enumerate()
                    .map(|(i, &e)| (e, param(&c.fit.node(i, j))))
                    .collect();
                let mut s = Series::line(&names[j], SERIES[j], pts.clone());
                for (i, p) in pts.into_iter().enumerate() {
                    if c.dropped.contains(&(i, j)) {
                        s.open_dots.push(p);
                    } else {
                        s.dots.push(p);
                    }
                }
                s
            })
            .collect()
    };
    // The generating cube's ρ and ν depend only on expiry, so one reference
    // line per panel shows what the fit is recovering.
    let reference = |param: fn(&SabrParams) -> f64| -> Series {
        let pts = c
            .expiries
            .iter()
            .enumerate()
            .map(|(i, &e)| (e, param(&c.truth.node(i, 0))))
            .collect();
        Series::line("Generating cube", MUTED, pts)
    };
    let mut rho = build(|p| p.rho);
    rho.push(reference(|p| p.rho));
    let mut nu = build(|p| p.nu);
    nu.push(reference(|p| p.nu));
    let x_ticks: Vec<(f64, String)> = c
        .expiries
        .iter()
        .map(|&e| (e, format!("{e:.0}y")))
        .collect();
    let fmt2 = |y: f64| format!("{y:.2}");
    let open_key = [LegendItem {
        label: "Node with no quotes",
        color: INK_2,
        key: Key::OpenDot,
    }];
    let mut out = String::new();
    svg::open(
        &mut out,
        W,
        h,
        "Fitted SABR parameters by expiry and tenor",
        "Correlation rho and vol-of-vol nu of the fitted SABR cube at each grid node, by option expiry, one line per swap tenor, with the generating cube's values as a grey reference line. Open markers are nodes with no market quotes. Values in docs/vol_cube.csv.",
    );
    svg::line_panel(
        &mut out,
        &LinePanel {
            x: 0.0,
            y: 0.0,
            w: W,
            h: panel_h,
            title: "Correlation ρ (rho) by option expiry",
            subtitle: "Sets the skew: negative ρ means vol rises as strikes fall. The least well-pinned parameter, so the ±0.3bp quote noise shows up here.",
            series: &rho,
            extra_legend: &open_key,
            x_range: (0.5, 10.5),
            x_ticks: &x_ticks,
            y_step: 0.02,
            y_fmt: &fmt2,
            x_marks: &[],
            note: Some("Option expiry"),
        },
    );
    svg::rule(&mut out, svg::ML, W - svg::MR, panel_h + 12.0);
    svg::line_panel(
        &mut out,
        &LinePanel {
            x: 0.0,
            y: panel_h + 24.0,
            w: W,
            h: panel_h,
            title: "Vol-of-vol ν (nu) by option expiry",
            subtitle: "Sets the smile's curvature. It falls with expiry, so long-dated smiles are flatter. Recovered closely at every tenor.",
            series: &nu,
            extra_legend: &open_key,
            x_range: (0.5, 10.5),
            x_ticks: &x_ticks,
            y_step: 0.05,
            y_fmt: &fmt2,
            x_marks: &[],
            note: Some("Option expiry"),
        },
    );
    svg::close(&mut out);
    out
}

/// Table view of the fitted cube: one row per node.
fn cube_table(c: &ChartData) -> Result<String, quadsmooth::Error> {
    let mut csv = String::from("expiry_y,tenor_y,forward_pct,quoted,alpha,rho,nu");
    for o in c.offsets_bp {
        let _ = write!(
            csv,
            ",vol_bp_{}",
            bp_label(*o).replace('+', "p").replace('-', "m")
        );
    }
    csv.push('\n');
    for (i, &e) in c.expiries.iter().enumerate() {
        for (j, &t) in c.tenors.iter().enumerate() {
            let p = c.fit.node(i, j);
            let _ = write!(
                csv,
                "{e},{t},{:.5},{},{:.6},{:.5},{:.5}",
                c.fit.grid().forward(i, j) * 100.0,
                !c.dropped.contains(&(i, j)),
                p.alpha,
                p.rho,
                p.nu
            );
            for &o in c.offsets_bp {
                let _ = write!(
                    csv,
                    ",{:.3}",
                    c.fit.node_vol(i, j, Strike::Offset(o / BP))? * BP
                );
            }
            csv.push('\n');
        }
    }
    Ok(csv)
}
