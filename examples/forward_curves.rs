//! Plots forward curves to show what "smooth forwards" buys.
//!
//! The TONA OIS benchmarks from the `jpy_2y2y` example are fitted three ways:
//!
//! * the quad-smooth curve from this crate;
//! * a bootstrap with flat instantaneous forwards between knots (log-linear
//!   discount factors);
//! * a bootstrap with zero rates linear between knots.
//!
//! All three reprice every benchmark exactly; they differ only in what they
//! say *between* benchmarks. The example writes `docs/forward_curves.svg`
//! (two panels: the three curves' instantaneous forwards, and the quad-smooth
//! curve's forwards by tenor) and `docs/forward_curves.csv` with the plotted
//! values.
//!
//! Run with `cargo run --example forward_curves`.

use std::fmt::Write as _;
use std::fs;

use quadsmooth::{CurveId, DualCurve, DualCurveFitter, FitConfig, Instrument};

#[path = "common/svg.rs"]
mod svg;
use svg::{LinePanel, Series, SERIES};

const AS_OF: &str = "21 Oct 2026";
const TONA_FIXING: f64 = 0.01227;
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

// ---------------------------------------------------------------------------
// Bootstrapped comparison curves
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Interp {
    /// Flat instantaneous forward between knots (log-linear discount factors).
    FlatForward,
    /// Zero rate linear between knots.
    LinearZero,
}

/// A curve defined by zero rates at knots, flat zero before the first knot,
/// and flat forward (FlatForward) or flat zero (LinearZero) after the last.
struct Bootstrapped {
    interp: Interp,
    knots: Vec<f64>,
    zeros: Vec<f64>,
}

impl Bootstrapped {
    /// `−ln P(t) = z(t)·t`
    fn log_df(&self, t: f64) -> f64 {
        let (k, z) = (&self.knots, &self.zeros);
        let n = k.len();
        if t <= k[0] {
            return z[0] * t;
        }
        if t >= k[n - 1] {
            return match self.interp {
                Interp::LinearZero => z[n - 1] * t,
                Interp::FlatForward => {
                    z[n - 1] * k[n - 1] + self.forward(k[n - 1] - 1e-12) * (t - k[n - 1])
                }
            };
        }
        let i = k.partition_point(|&x| x <= t) - 1;
        let w = (t - k[i]) / (k[i + 1] - k[i]);
        match self.interp {
            Interp::LinearZero => ((1.0 - w) * z[i] + w * z[i + 1]) * t,
            Interp::FlatForward => (1.0 - w) * z[i] * k[i] + w * z[i + 1] * k[i + 1],
        }
    }

    fn df(&self, t: f64) -> f64 {
        (-self.log_df(t)).exp()
    }

    /// Instantaneous forward `d/dt (z t)` (right derivative at knots).
    fn forward(&self, t: f64) -> f64 {
        let (k, z) = (&self.knots, &self.zeros);
        let n = k.len();
        if t < k[0] {
            return z[0];
        }
        let i = if t >= k[n - 1] {
            n - 2
        } else {
            k.partition_point(|&x| x <= t) - 1
        };
        let (t0, t1) = (k[i], k[i + 1]);
        match self.interp {
            Interp::FlatForward => (z[i + 1] * t1 - z[i] * t0) / (t1 - t0),
            Interp::LinearZero => {
                if t >= k[n - 1] {
                    return z[n - 1];
                }
                let slope = (z[i + 1] - z[i]) / (t1 - t0);
                z[i] + slope * (t - t0) + slope * t
            }
        }
    }
}

/// Par rate of an instrument on a curve given only by its discount function.
fn par_rate(inst: &Instrument, df: &dyn Fn(f64) -> f64) -> f64 {
    match inst {
        Instrument::SimpleRate { start, end, .. } => (df(*start) / df(*end) - 1.0) / (end - start),
        Instrument::Swap { fixed, float, .. } => {
            let annuity: f64 = fixed.periods().map(|(_, t, tau)| tau * df(t)).sum();
            let float_pv: f64 = float
                .periods()
                .map(|(s0, s1, _)| (df(s0) / df(s1) - 1.0) * df(s1))
                .sum();
            float_pv / annuity
        }
    }
}

/// Sequential bootstrap: solve each knot's zero rate so its instrument
/// reprices, holding earlier knots fixed.
fn bootstrap(interp: Interp, instruments: &[Instrument]) -> Bootstrapped {
    let mut curve = Bootstrapped {
        interp,
        knots: Vec::new(),
        zeros: Vec::new(),
    };
    for inst in instruments {
        curve.knots.push(inst.maturity());
        curve.zeros.push(inst.quote());
        let target = inst.quote();
        let residual = |c: &Bootstrapped| par_rate(inst, &|t| c.df(t)) - target;
        // Secant iteration on the new knot's zero rate.
        let n = curve.zeros.len() - 1;
        let (mut z0, mut z1) = (target, target + 1e-4);
        curve.zeros[n] = z0;
        let mut r0 = residual(&curve);
        for _ in 0..50 {
            curve.zeros[n] = z1;
            let r1 = residual(&curve);
            if r1.abs() < 1e-15 || (r1 - r0).abs() < 1e-300 {
                break;
            }
            let z2 = z1 - r1 * (z1 - z0) / (r1 - r0);
            (z0, r0, z1) = (z1, r1, z2);
        }
        curve.zeros[n] = z1;
    }
    curve
}

const W: f64 = 960.0;
const PANEL_H: f64 = 430.0;

fn main() -> Result<(), quadsmooth::Error> {
    let mut instruments = vec![Instrument::deposit(
        CurveId::Discount,
        1.0 / 365.0,
        TONA_FIXING,
    )?];
    for &(t, r) in TONA_OIS {
        instruments.push(Instrument::ois(t, r, 1)?);
    }

    let mut fitter = DualCurveFitter::new().with_config(FitConfig {
        lambda_curvature: 1e-10,
        ..FitConfig::default()
    });
    for inst in &instruments {
        fitter = fitter.add(inst.clone());
    }
    let smooth: DualCurve = fitter.fit()?.curves;
    let flat = bootstrap(Interp::FlatForward, &instruments);
    let linz = bootstrap(Interp::LinearZero, &instruments);

    // Every curve must reprice every benchmark.
    println!(
        "{:<10} {:>12} {:>14} {:>14}",
        "benchmark", "quad-smooth", "flat forward", "linear zero"
    );
    for inst in &instruments {
        let q = inst.quote();
        let err = |df: &dyn Fn(f64) -> f64| (par_rate(inst, df) - q) * 1e4;
        println!(
            "{:<10} {:>10.2e}bp {:>12.2e}bp {:>12.2e}bp",
            if inst.maturity() < 0.5 {
                "TONA 1D".to_string()
            } else {
                format!("OIS {:.0}y", inst.maturity())
            },
            err(&|t| smooth.discount_factor(CurveId::Discount, t).unwrap()),
            err(&|t| flat.df(t)),
            err(&|t| linz.df(t)),
        );
    }

    // Sample.
    let dt: f64 = 0.025;
    let x_max: f64 = 30.0;
    let n = (x_max / dt).round() as usize;
    // Stop one step short of 30y: at the last knot the bootstraps switch to
    // their extrapolation rule, which is not what the chart is about.
    let grid: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let pct = 100.0;
    let inst_smooth: Vec<(f64, f64)> = grid
        .iter()
        .map(|&t| (t, smooth.forward(CurveId::Discount, t).unwrap() * pct))
        .collect();
    let inst_flat: Vec<(f64, f64)> = grid.iter().map(|&t| (t, flat.forward(t) * pct)).collect();
    let inst_linz: Vec<(f64, f64)> = grid.iter().map(|&t| (t, linz.forward(t) * pct)).collect();
    let fwd = |tau: f64| -> Vec<(f64, f64)> {
        grid.iter()
            .filter(|&&t| t + tau <= x_max + 1e-9)
            .map(|&t| {
                (
                    t,
                    smooth
                        .simple_forward(CurveId::Discount, t, t + tau)
                        .unwrap()
                        * pct,
                )
            })
            .collect()
    };
    let fwd_3m = fwd(0.25);
    let fwd_1y = fwd(1.0);
    let zero: Vec<(f64, f64)> = grid
        .iter()
        .map(|&t| (t, smooth.zero_rate(CurveId::Discount, t).unwrap() * pct))
        .collect();

    // Discontinuities: the largest move in the instantaneous forward between
    // adjacent samples. A continuous curve moves by f'·dt; a jump shows up whole.
    let max_jump = |p: &[(f64, f64)]| {
        p.windows(2)
            .map(|w| (w[1].1 - w[0].1).abs())
            .fold(0.0_f64, f64::max)
            * 100.0
    };
    println!("\nLargest move in the instantaneous forward between {dt}y samples:");
    println!("  quad-smooth   {:>6.2}bp", max_jump(&inst_smooth));
    println!("  flat forward  {:>6.2}bp", max_jump(&inst_flat));
    println!("  linear zero   {:>6.2}bp", max_jump(&inst_linz));

    // SVG.
    let knots: Vec<f64> = TONA_OIS.iter().map(|p| p.0).collect();
    let height = 2.0 * PANEL_H + 24.0;
    let x_ticks: Vec<(f64, String)> = (0..=6)
        .map(|i| (5.0 * i as f64, format!("{}y", 5 * i)))
        .collect();
    let pct_fmt = |y: f64| format!("{y:.1}%");
    let mut out = String::new();
    svg::open(
        &mut out,
        W,
        height,
        &format!("TONA OIS forward curves, {AS_OF}"),
        "Instantaneous forward rates of the TONA curve under quad-smooth fitting, flat-forward bootstrap and linear-zero bootstrap, and the quad-smooth curve's instantaneous, 3-month and 1-year forwards and zero rates, 0 to 30 years. Values in docs/forward_curves.csv.",
    );
    let top_title = format!("One set of TONA OIS quotes ({AS_OF}), three curves");
    svg::line_panel(
        &mut out,
        &LinePanel {
            x: 0.0,
            y: 0.0,
            w: W,
            h: PANEL_H,
            title: &top_title,
            subtitle: "Instantaneous forward rate. All three reprice every benchmark exactly; only the quad-smooth fit keeps the forward continuous.",
            series: &[
                Series::line("Quad-smooth (this crate)", SERIES[0], inst_smooth.clone()),
                Series::line("Bootstrap, flat forwards", SERIES[1], inst_flat.clone()),
                Series::line("Bootstrap, linear zero rates", SERIES[2], inst_linz.clone()),
            ],
            extra_legend: &[],
            x_range: (0.0, x_max),
            x_ticks: &x_ticks,
            y_step: 0.5,
            y_fmt: &pct_fmt,
            x_marks: &knots,
            note: Some("Ticks on the axis mark the OIS benchmark maturities"),
        },
    );
    svg::rule(&mut out, svg::ML, W - svg::MR, PANEL_H + 12.0);
    svg::line_panel(
        &mut out,
        &LinePanel {
            x: 0.0,
            y: PANEL_H + 24.0,
            w: W,
            h: PANEL_H,
            title: "Quad-smooth curve: forward rates by tenor",
            subtitle: "Instantaneous, 3-month and 1-year simple forwards starting at each date, with the zero curve for reference.",
            series: &[
                Series::line("Instantaneous forward", SERIES[0], inst_smooth.clone()),
                Series::line("3M forward", SERIES[1], fwd_3m.clone()),
                Series::line("1Y forward", SERIES[2], fwd_1y.clone()),
                Series::line("Zero rate", SERIES[3], zero.clone()),
            ],
            extra_legend: &[],
            x_range: (0.0, x_max),
            x_ticks: &x_ticks,
            y_step: 0.5,
            y_fmt: &pct_fmt,
            x_marks: &[],
            note: None,
        },
    );
    svg::close(&mut out);

    // CSV table view (every 0.25y).
    let mut csv = String::from(
        "t_years,inst_fwd_quad_smooth_pct,inst_fwd_flat_forward_pct,inst_fwd_linear_zero_pct,fwd_3m_pct,fwd_1y_pct,zero_pct\n",
    );
    let step = (0.25 / dt).round() as usize;
    for i in (0..n).step_by(step) {
        let get = |p: &[(f64, f64)]| p.get(i).map(|v| format!("{:.5}", v.1)).unwrap_or_default();
        let _ = writeln!(
            csv,
            "{:.2},{},{},{},{},{},{}",
            grid[i],
            get(&inst_smooth),
            get(&inst_flat),
            get(&inst_linz),
            get(&fwd_3m),
            get(&fwd_1y),
            get(&zero)
        );
    }

    fs::create_dir_all("docs").expect("create docs/");
    fs::write("docs/forward_curves.svg", out).expect("write svg");
    fs::write("docs/forward_curves.csv", csv).expect("write csv");
    println!("\nwrote docs/forward_curves.svg and docs/forward_curves.csv");
    Ok(())
}
