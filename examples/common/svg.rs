//! Minimal SVG chart primitives shared by the plotting examples.
//!
//! Charts are written as static SVG so they render in a GitHub README:
//! a light surface, hairline gridlines, 2px series lines, >= 8px markers with
//! a surface ring, a legend row under the subtitle, and text in ink tokens
//! (never in series colours).
#![allow(dead_code)]

use std::fmt::Write as _;

pub const SURFACE: &str = "#fcfcfb";
pub const INK: &str = "#0b0b0b";
pub const INK_2: &str = "#52514e";
pub const MUTED: &str = "#898781";
pub const GRID: &str = "#e1e0d9";
pub const AXIS: &str = "#c3c2b7";
pub const FONT: &str = "system-ui, -apple-system, 'Segoe UI', sans-serif";
/// Categorical slots 1–4, in fixed order.
pub const SERIES: [&str; 4] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"];
/// Sequential blue ramp, steps 100–700.
pub const BLUES: [&str; 13] = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
];

/// Panel padding: left (y labels), right, top (title, subtitle, legend), bottom (x labels, note).
pub const ML: f64 = 64.0;
pub const MR: f64 = 28.0;
pub const MT: f64 = 104.0;
pub const MB: f64 = 58.0;

pub fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub fn nice_bounds(lo: f64, hi: f64, step: f64) -> (f64, f64) {
    ((lo / step).floor() * step, (hi / step).ceil() * step)
}

/// Opens an SVG document of the given size with an accessible title and
/// description, on the chart surface.
pub fn open(svg: &mut String, w: f64, h: f64, title: &str, desc: &str) {
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}" role="img" aria-labelledby="t d">"#
    );
    let _ = writeln!(
        svg,
        r#"<title id="t">{}</title><desc id="d">{}</desc>"#,
        esc(title),
        esc(desc)
    );
    let _ = writeln!(svg, r#"<rect width="{w}" height="{h}" fill="{SURFACE}"/>"#);
}

pub fn close(svg: &mut String) {
    svg.push_str("</svg>\n");
}

/// Horizontal hairline separating stacked panels.
pub fn rule(svg: &mut String, x0: f64, x1: f64, y: f64) {
    let _ = writeln!(
        svg,
        r#"<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{GRID}" stroke-width="1"/>"#
    );
}

pub fn text(svg: &mut String, x: f64, y: f64, size: f64, fill: &str, anchor: &str, s: &str) {
    let _ = writeln!(
        svg,
        r#"<text x="{x:.1}" y="{y:.1}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" style="font-variant-numeric: tabular-nums">{}</text>"#,
        esc(s)
    );
}

/// Panel title (semibold) and subtitle (secondary ink).
pub fn header(svg: &mut String, x: f64, y: f64, title: &str, subtitle: &str) {
    let _ = writeln!(
        svg,
        r#"<text x="{x}" y="{}" font-size="16" font-weight="600" fill="{INK}">{}</text>"#,
        y + 30.0,
        esc(title)
    );
    text(svg, x, y + 51.0, 13.0, INK_2, "start", subtitle);
}

/// Marker drawn with a 2px surface ring; hollow markers are filled with the
/// surface and ringed in the series colour.
pub fn dot(svg: &mut String, x: f64, y: f64, color: &str, hollow: bool) {
    if hollow {
        let _ = writeln!(
            svg,
            r#"<circle cx="{x:.2}" cy="{y:.2}" r="6" fill="{SURFACE}"/><circle cx="{x:.2}" cy="{y:.2}" r="4" fill="{SURFACE}" stroke="{color}" stroke-width="2"/>"#
        );
    } else {
        let _ = writeln!(
            svg,
            r#"<circle cx="{x:.2}" cy="{y:.2}" r="6" fill="{SURFACE}"/><circle cx="{x:.2}" cy="{y:.2}" r="4" fill="{color}"/>"#
        );
    }
}

/// How a legend entry's key is drawn.
#[derive(Clone, Copy)]
pub enum Key {
    Line,
    Dot,
    OpenDot,
}

pub struct LegendItem<'a> {
    pub label: &'a str,
    pub color: &'a str,
    pub key: Key,
}

/// One legend row starting at (x, y), keys in colour, labels in ink.
pub fn legend(svg: &mut String, x: f64, y: f64, items: &[LegendItem]) {
    let mut lx = x;
    for it in items {
        let text_x = match it.key {
            Key::Line => {
                let _ = writeln!(
                    svg,
                    r#"<line x1="{lx}" y1="{y}" x2="{}" y2="{y}" stroke="{}" stroke-width="2.5" stroke-linecap="round"/>"#,
                    lx + 18.0,
                    it.color
                );
                lx + 24.0
            }
            Key::Dot | Key::OpenDot => {
                dot(svg, lx + 5.0, y, it.color, matches!(it.key, Key::OpenDot));
                lx + 15.0
            }
        };
        text(svg, text_x, y + 4.0, 12.5, INK_2, "start", it.label);
        lx = text_x + 7.2 * it.label.chars().count() as f64 + 26.0;
    }
}

/// A plotted series: a 2px line plus optional solid and hollow markers.
pub struct Series<'a> {
    pub name: &'a str,
    pub color: &'a str,
    pub points: Vec<(f64, f64)>,
    pub dots: Vec<(f64, f64)>,
    pub open_dots: Vec<(f64, f64)>,
}

impl<'a> Series<'a> {
    pub fn line(name: &'a str, color: &'a str, points: Vec<(f64, f64)>) -> Self {
        Series {
            name,
            color,
            points,
            dots: Vec::new(),
            open_dots: Vec::new(),
        }
    }
}

/// A line chart panel. `x`, `y` is its top-left corner.
pub struct LinePanel<'a> {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub title: &'a str,
    pub subtitle: &'a str,
    pub series: &'a [Series<'a>],
    /// Legend entries appended after the series (e.g. what a marker means).
    pub extra_legend: &'a [LegendItem<'a>],
    pub x_range: (f64, f64),
    pub x_ticks: &'a [(f64, String)],
    pub y_step: f64,
    pub y_fmt: &'a dyn Fn(f64) -> String,
    /// Short ticks on the baseline (e.g. benchmark maturities).
    pub x_marks: &'a [f64],
    /// Footnote under the x axis, right-aligned.
    pub note: Option<&'a str>,
}

pub fn line_panel(svg: &mut String, p: &LinePanel) {
    let values = p.series.iter().flat_map(|s| {
        s.points
            .iter()
            .chain(s.dots.iter())
            .chain(s.open_dots.iter())
            .map(|v| v.1)
    });
    let (lo, hi) = values.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
        (a.min(v), b.max(v))
    });
    let (y_lo, y_hi) = nice_bounds(lo, hi, p.y_step);
    let (px0, px1) = (p.x + ML, p.x + p.w - MR);
    let (py0, py1) = (p.y + p.h - MB, p.y + MT);
    let (x_lo, x_hi) = p.x_range;
    let sx = |x: f64| px0 + (x - x_lo) / (x_hi - x_lo) * (px1 - px0);
    let sy = |y: f64| py0 - (y - y_lo) / (y_hi - y_lo) * (py0 - py1);

    header(svg, px0, p.y, p.title, p.subtitle);
    let mut items: Vec<LegendItem> = p
        .series
        .iter()
        .map(|s| LegendItem {
            label: s.name,
            color: s.color,
            key: Key::Line,
        })
        .collect();
    items.extend(p.extra_legend.iter().map(|it| LegendItem {
        label: it.label,
        color: it.color,
        key: it.key,
    }));
    legend(svg, px0, p.y + 76.0, &items);

    let mut y = y_lo;
    while y <= y_hi + 1e-9 {
        let py = sy(y);
        let _ = writeln!(
            svg,
            r#"<line x1="{px0}" y1="{py:.1}" x2="{px1}" y2="{py:.1}" stroke="{}" stroke-width="1"/>"#,
            if (y - y_lo).abs() < 1e-9 { AXIS } else { GRID }
        );
        text(svg, px0 - 8.0, py + 4.0, 12.0, MUTED, "end", &(p.y_fmt)(y));
        y += p.y_step;
    }
    for (x, label) in p.x_ticks {
        text(svg, sx(*x), py0 + 20.0, 12.0, MUTED, "middle", label);
    }
    for &k in p.x_marks.iter().filter(|&&k| k >= x_lo && k <= x_hi) {
        let _ = writeln!(
            svg,
            r#"<line x1="{0:.1}" y1="{1:.1}" x2="{0:.1}" y2="{2:.1}" stroke="{INK_2}" stroke-width="1.5"/>"#,
            sx(k),
            py0 - 6.0,
            py0
        );
    }
    if let Some(note) = p.note {
        text(svg, px1, py0 + 42.0, 12.0, MUTED, "end", note);
    }

    // Lines in reverse so slot 1 sits on top, then all markers above lines.
    for s in p.series.iter().rev() {
        if s.points.is_empty() {
            continue;
        }
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
    for s in p.series.iter().rev() {
        for &(x, y) in &s.dots {
            dot(svg, sx(x), sy(y), s.color, false);
        }
        for &(x, y) in &s.open_dots {
            dot(svg, sx(x), sy(y), s.color, true);
        }
    }
}

/// Parses `#rrggbb`.
fn rgb(hex: &str) -> (f64, f64, f64) {
    let p = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap() as f64;
    (p(1), p(3), p(5))
}

/// Colour on the sequential ramp for `u` in [0, 1] (interpolated between
/// steps), and whether text on it should be white.
pub fn ramp(u: f64) -> (String, bool) {
    let u = u.clamp(0.0, 1.0) * (BLUES.len() - 1) as f64;
    let i = (u.floor() as usize).min(BLUES.len() - 2);
    let w = u - i as f64;
    let (a, b) = (rgb(BLUES[i]), rgb(BLUES[i + 1]));
    let mix = |x: f64, y: f64| (x + (y - x) * w).round() as u8;
    let hex = format!(
        "#{:02x}{:02x}{:02x}",
        mix(a.0, b.0),
        mix(a.1, b.1),
        mix(a.2, b.2)
    );
    // Steps from 400 (#3987e5) darker carry white text.
    (hex, u >= 6.0)
}
