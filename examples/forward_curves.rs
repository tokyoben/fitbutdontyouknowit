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

// ---------------------------------------------------------------------------
// SVG
// ---------------------------------------------------------------------------

const W: f64 = 960.0;
const PANEL_H: f64 = 430.0;
const ML: f64 = 64.0;
const MR: f64 = 28.0;
const MT: f64 = 104.0;
const MB: f64 = 58.0;

const SURFACE: &str = "#fcfcfb";
const INK: &str = "#0b0b0b";
const INK_2: &str = "#52514e";
const MUTED: &str = "#898781";
const GRID: &str = "#e1e0d9";
const AXIS: &str = "#c3c2b7";
const FONT: &str = "system-ui, -apple-system, 'Segoe UI', sans-serif";
/// Categorical slots 1–4, in fixed order.
const SERIES: [&str; 4] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"];

struct Series<'a> {
    name: &'a str,
    color: &'a str,
    points: Vec<(f64, f64)>,
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn nice_bounds(lo: f64, hi: f64, step: f64) -> (f64, f64) {
    ((lo / step).floor() * step, (hi / step).ceil() * step)
}

/// One panel at vertical offset `y0`. Values are in percent.
#[allow(clippy::too_many_arguments)]
fn panel(
    svg: &mut String,
    y0: f64,
    title: &str,
    subtitle: &str,
    series: &[Series],
    x_max: f64,
    knot_marks: &[f64],
) {
    let (lo, hi) = series
        .iter()
        .flat_map(|s| s.points.iter().map(|p| p.1))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
            (a.min(v), b.max(v))
        });
    let y_step = 0.5;
    let (y_lo, y_hi) = nice_bounds(lo, hi, y_step);
    let (px0, px1) = (ML, W - MR);
    let (py0, py1) = (y0 + PANEL_H - MB, y0 + MT);
    let sx = |x: f64| px0 + (x / x_max) * (px1 - px0);
    let sy = |y: f64| py0 - (y - y_lo) / (y_hi - y_lo) * (py0 - py1);

    let _ = writeln!(
        svg,
        r#"<text x="{ML}" y="{}" font-size="16" font-weight="600" fill="{INK}">{}</text>"#,
        y0 + 30.0,
        esc(title)
    );
    let _ = writeln!(
        svg,
        r#"<text x="{ML}" y="{}" font-size="13" fill="{INK_2}">{}</text>"#,
        y0 + 51.0,
        esc(subtitle)
    );

    // Legend: one row under the subtitle, short line keys, text in ink.
    let mut lx = ML;
    for s in series {
        let ly = y0 + 76.0;
        let _ = writeln!(
            svg,
            r#"<line x1="{lx}" y1="{ly}" x2="{}" y2="{ly}" stroke="{}" stroke-width="2.5" stroke-linecap="round"/>"#,
            lx + 18.0,
            s.color
        );
        let _ = writeln!(
            svg,
            r#"<text x="{}" y="{}" font-size="12.5" fill="{INK_2}">{}</text>"#,
            lx + 24.0,
            ly + 4.0,
            esc(s.name)
        );
        lx += 24.0 + 7.4 * s.name.chars().count() as f64 + 26.0;
    }

    // Gridlines and y ticks.
    let mut y = y_lo;
    while y <= y_hi + 1e-9 {
        let py = sy(y);
        let _ = writeln!(
            svg,
            r#"<line x1="{px0}" y1="{py:.1}" x2="{px1}" y2="{py:.1}" stroke="{}" stroke-width="1"/>"#,
            if (y - y_lo).abs() < 1e-9 { AXIS } else { GRID }
        );
        let _ = writeln!(
            svg,
            r#"<text x="{}" y="{:.1}" font-size="12" fill="{MUTED}" text-anchor="end" style="font-variant-numeric: tabular-nums">{y:.1}%</text>"#,
            px0 - 8.0,
            py + 4.0
        );
        y += y_step;
    }
    // X ticks every 5 years.
    let mut x = 0.0;
    while x <= x_max + 1e-9 {
        let _ = writeln!(
            svg,
            r#"<text x="{:.1}" y="{}" font-size="12" fill="{MUTED}" text-anchor="middle" style="font-variant-numeric: tabular-nums">{x:.0}y</text>"#,
            sx(x),
            py0 + 20.0
        );
        x += 5.0;
    }
    // Benchmark maturities as short ticks on the baseline.
    if !knot_marks.is_empty() {
        for &k in knot_marks.iter().filter(|&&k| k <= x_max) {
            let _ = writeln!(
                svg,
                r#"<line x1="{0:.1}" y1="{1:.1}" x2="{0:.1}" y2="{2:.1}" stroke="{INK_2}" stroke-width="1.5"/>"#,
                sx(k),
                py0 - 6.0,
                py0
            );
        }
        let _ = writeln!(
            svg,
            r#"<text x="{px1}" y="{}" font-size="12" fill="{MUTED}" text-anchor="end">Ticks on the axis mark the OIS benchmark maturities</text>"#,
            py0 + 42.0
        );
    }

    // Series: 2px lines, round joins. Drawn in reverse so slot 1 sits on top.
    for s in series.iter().rev() {
        let mut d = String::new();
        for (i, &(x, y)) in s.points.iter().enumerate() {
            let _ = write!(
                d,
                "{}{:.2},{:.2}",
                if i == 0 { "M" } else { " L" },
                sx(x),
                sy(y)
            );
        }
        let _ = writeln!(
            svg,
            r#"<path d="{d}" fill="none" stroke="{}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>"#,
            s.color
        );
    }
}

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
    let mut svg = String::new();
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" font-family="{FONT}" role="img" aria-labelledby="t d">"#
    );
    let _ = writeln!(
        svg,
        r#"<title id="t">TONA OIS forward curves, {AS_OF}</title><desc id="d">Instantaneous forward rates of the TONA curve under quad-smooth fitting, flat-forward bootstrap and linear-zero bootstrap, and the quad-smooth curve's instantaneous, 3-month and 1-year forwards and zero rates, 0 to 30 years. Values in docs/forward_curves.csv.</desc>"#
    );
    let _ = writeln!(
        svg,
        r#"<rect width="{W}" height="{height}" fill="{SURFACE}"/>"#
    );
    panel(
        &mut svg,
        0.0,
        &format!("One set of TONA OIS quotes ({AS_OF}), three curves"),
        "Instantaneous forward rate. All three reprice every benchmark exactly; only the quad-smooth fit keeps the forward continuous.",
        &[
            Series { name: "Quad-smooth (this crate)", color: SERIES[0], points: inst_smooth.clone() },
            Series { name: "Bootstrap, flat forwards", color: SERIES[1], points: inst_flat.clone() },
            Series { name: "Bootstrap, linear zero rates", color: SERIES[2], points: inst_linz.clone() },
        ],
        x_max,
        &knots,
    );
    let _ = writeln!(
        svg,
        r#"<line x1="{ML}" y1="{0}" x2="{1}" y2="{0}" stroke="{GRID}" stroke-width="1"/>"#,
        PANEL_H + 12.0,
        W - MR
    );
    panel(
        &mut svg,
        PANEL_H + 24.0,
        "Quad-smooth curve: forward rates by tenor",
        "Instantaneous, 3-month and 1-year simple forwards starting at each date, with the zero curve for reference.",
        &[
            Series { name: "Instantaneous forward", color: SERIES[0], points: inst_smooth.clone() },
            Series { name: "3M forward", color: SERIES[1], points: fwd_3m.clone() },
            Series { name: "1Y forward", color: SERIES[2], points: fwd_1y.clone() },
            Series { name: "Zero rate", color: SERIES[3], points: zero.clone() },
        ],
        x_max,
        &[],
    );
    svg.push_str("</svg>\n");

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
    fs::write("docs/forward_curves.svg", svg).expect("write svg");
    fs::write("docs/forward_curves.csv", csv).expect("write csv");
    println!("\nwrote docs/forward_curves.svg and docs/forward_curves.csv");
    Ok(())
}
