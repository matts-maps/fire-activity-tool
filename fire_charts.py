#!/usr/bin/env python3
"""
fire_charts.py - PNG charts for the fire activity tool (Phase 4).

Three charts, written next to the summary CSV:

  {geoextent}_fire_activity_chart_detections.png   fire detections per period
  {geoextent}_fire_activity_chart_burned.png       burned area per period (km2)
  {geoextent}_fire_activity_chart_admin.png        fire detections by admin area, whole run

Each period is a column; with a year-earlier comparison the same period a year
earlier is a thin black line across the column. Values are written at the tip
of each column. Periods not yet available have no column and a dash instead of
a value; periods with a quality flag get an asterisk and a footnote.

Styling: Noto Sans, white
background, no gridlines, rust red columns, plain black comparison
line, months on one row with the year beneath, source footer.

Charts can also be redrawn from an existing summary CSV without re-running the
tool (handy after changing a title):

    python fire_charts.py out\\sct_fire_activity_summary.csv --area-name Scotland

Only matplotlib is needed (plus pandas for the CSV route).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Style (a warm "ember" palette)
# ---------------------------------------------------------------------------

EMBER_RED_DEEP = "#7A1F1A"
EMBER_RED_MID = "#B23A26"
EMBER_RED_BRIGHT = "#E2603F"
EMBER_BLUE = "#4D6B7A"
EMBER_GREY = "#8A7F76"
EMBER_BLACK = "#1A1512"

DETECTION_COLOUR = EMBER_RED_MID
BURNED_COLOUR = EMBER_RED_DEEP
ADMIN_COLOUR = EMBER_RED_MID
TICK_COLOUR = "#000000"
TICK_WIDTH_PT = 0.5
TEXT_COLOUR = EMBER_BLACK
AXIS_COLOUR = EMBER_GREY

FONT_PREFERENCE = ("Noto Sans", "Segoe UI", "Arial", "DejaVu Sans")
DPI = 200
MAX_ADMIN_BARS = 15

MONTHS_FULL = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December")
MONTHS_ABBR = tuple(m[:3] for m in MONTHS_FULL)
MONTHS_LETTER = tuple(m[0] for m in MONTHS_FULL)

FREQ_WORD = {"monthly": "month", "weekly": "week", "annual": "year"}

# Plain-English footnotes for quality flags (detections and burned area).
FLAG_NOTES = {
    "partial_period": "data ends part-way through the period; the year-earlier line covers the same days only",
    "missing_days": "some days in the period have no satellite imagery",
    "no_imagery": "no satellite imagery in the period",
    "low_coverage": "MODIS could map less than 90 per cent of the land (usually cloud), so burned area may be "
                    "under-counted",
    "missing_month": "a month is missing from the Earth Engine archive; no year-earlier comparison",
    "coverage_unknown": "no mapped-coverage figure for the period",
}
NOT_YET = "not_yet_available"

# Below this mapped share a year-earlier burned area is marked (same rule as fire_burned.LOW_COVERAGE_PCT;
# fire_charts stays importable on its own, so the value is checked by a test rather than imported).
LOW_COVERAGE_PCT = 90.0

# Footnotes for flags on the year-earlier side of a comparison, marked with a dagger.
PRIOR_MARK = "\u2020"
PRIOR_FLAG_NOTES = {
    "missing_days": "the year-earlier period has days with no satellite imagery, so the year-earlier line is "
                    "likely too low",
    "no_imagery": "no satellite imagery in the year-earlier period, so no year-earlier line",
    "low_coverage": "MODIS could map less than 90 per cent of the land a year earlier, so the year-earlier line "
                    "may be too low",
}

_font_family: Optional[str] = None


def chart_font() -> str:
    """First installed font from FONT_PREFERENCE (Noto Sans when it is there)."""
    global _font_family
    if _font_family is None:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        _font_family = next((f for f in FONT_PREFERENCE if f in installed), "sans-serif")
    return _font_family


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": chart_font(),
        "font.size": 9,
        "text.color": TEXT_COLOUR,
        "axes.edgecolor": AXIS_COLOUR,
        "axes.labelcolor": TEXT_COLOUR,
        "xtick.color": AXIS_COLOUR,
        "ytick.color": AXIS_COLOUR,
        "axes.grid": False,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return plt


# ---------------------------------------------------------------------------
# Pure helpers (tested offline)
# ---------------------------------------------------------------------------

def num(v) -> Optional[float]:
    """A summary value as a float, or None for blanks ('' / None / NaN)."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        try:
            v = float(v)
        except ValueError:
            return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def infer_freq(labels: list[str]) -> str:
    if labels and all(re.fullmatch(r"\d{4}-W\d{2}", s) for s in labels):
        return "weekly"
    if labels and all(re.fullmatch(r"\d{4}-\d{2}", s) for s in labels):
        return "monthly"
    if labels and all(re.fullmatch(r"\d{4}", s) for s in labels):
        return "annual"
    raise ValueError("Could not tell the period length from the period labels "
                     f"(e.g. {labels[:3]}); expected 2026-03, 2026-W14 or 2026.")


def format_value(v: float, decimals: int) -> str:
    if decimals == 0:
        return f"{int(round(v)):,}"
    return f"{v:,.{decimals}f}"


def burned_decimals(values: list[Optional[float]]) -> int:
    """Decimals for burned area labels: 1 normally, 0 once values are large."""
    vals = [v for v in values if v is not None]
    return 0 if vals and max(vals) >= 1000 else 1


def flag_parts(flag: str) -> list[str]:
    return [f for f in (flag or "").split("+") if f and f != "ok"]


def label_month(period_start: date, freq: str) -> date:
    """The month a period is shown under: its own month, or for a week the
    month holding its Thursday (the ISO rule, so a week sits in the month with
    most of its days)."""
    if freq == "weekly":
        return period_start + timedelta(days=3)
    return period_start


def month_groups(starts: list[date], freq: str) -> list[tuple[int, int, date]]:
    """Runs of consecutive bars in the same month: (first index, last index, month)."""
    groups: list[tuple[int, int, date]] = []
    for i, s in enumerate(starts):
        m = label_month(s, freq)
        key = (m.year, m.month)
        if groups and (groups[-1][2].year, groups[-1][2].month) == key and groups[-1][1] == i - 1:
            groups[-1] = (groups[-1][0], i, groups[-1][2])
        else:
            groups.append((i, i, date(m.year, m.month, 1)))
    return groups


def year_groups(starts: list[date], freq: str) -> list[tuple[int, int, int]]:
    groups: list[tuple[int, int, int]] = []
    for i, s in enumerate(starts):
        y = label_month(s, freq).year
        if groups and groups[-1][2] == y:
            groups[-1] = (groups[-1][0], i, y)
        else:
            groups.append((i, i, y))
    return groups


def pick_month_style(slot_widths_px: list[float], measure: Callable[[str], float], pad_px: float = 6.0) -> str:
    """Densest month format that fits every slot: 'full', 'abbr' or 'letter'.
    `slot_widths_px` are the widths available to each month label; `measure`
    returns a string's width in pixels."""
    widest = {
        "full": max(measure(m) for m in MONTHS_FULL),
        "abbr": max(measure(m) for m in MONTHS_ABBR),
    }
    usable = min(slot_widths_px) if slot_widths_px else 0
    for style in ("full", "abbr"):
        if widest[style] + pad_px <= usable:
            return style
    return "letter"


def month_text(d: date, style: str) -> str:
    return {"full": MONTHS_FULL, "abbr": MONTHS_ABBR, "letter": MONTHS_LETTER}[style][d.month - 1]


def label_anchors(heights: list[float], priors: list[Optional[float]], px_per_value: float, gap_px: float,
                  label_h_px) -> list[float]:
    """Data value each column's label sits on: the column top, or the
    year-earlier line when that line would cross the label (between the column
    top and the top of the label, with a little room either side)."""
    sizes = label_h_px if isinstance(label_h_px, (list, tuple)) else [label_h_px] * len(heights)
    out = []
    for h, p, size in zip(heights, priors, sizes):
        if p is not None and p >= h:
            dy = (p - h) * px_per_value
            if dy <= gap_px + size + 2:
                out.append(p)
                continue
        out.append(h)
    return out


def wrap_to_width(text: str, measure: Callable[[str], float], max_px: float) -> str:
    """Wrap text onto as many lines as it needs to fit `max_px`."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if cur and measure(trial) > max_px:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def geo_token(geoextent: str) -> str:
    """Same rule as fire_tool.sanitize_token, so chart names match the CSV and
    GeoPackage names exactly."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(geoextent).lower()).strip("_") or "na"


def chart_file_name(geoextent: str, kind: str) -> str:
    return f"{geo_token(geoextent)}_fire_activity_chart_{kind}.png"


def clear_charts(out_dir: Path, geoextent: str, locked: Optional[list] = None) -> int:
    """Remove charts left by an earlier run of the same geoextent, so a chart
    that no longer applies (no admin areas, burned area off) does not linger.
    Files that cannot be removed (open in a viewer on Windows) are skipped and
    their names added to `locked`."""
    locked = [] if locked is None else locked
    n = 0
    for kind in ("detections", "burned", "admin"):
        for p in (Path(out_dir) / chart_file_name(geoextent, kind),
                  (Path(out_dir) / chart_file_name(geoextent, kind)).with_suffix(".tmp.png")):
            try:
                if p.exists():
                    p.unlink()
                    n += 1
            except OSError:
                locked.append(p.name)
    return n


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _save(fig, plt, path: Path) -> Path:
    """Write via a temporary file so a half-written PNG never replaces a good
    one; the temporary file is removed whatever happens. A PNG open in a viewer
    on Windows cannot be replaced: that raises a plain OSError naming it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    try:
        fig.savefig(tmp, dpi=DPI)
        try:
            tmp.replace(path)
        except PermissionError:
            raise OSError(f"Could not replace {path.name}; close it if it is open in another program and "
                          "run again.") from None
    finally:
        plt.close(fig)
        try:
            tmp.unlink()
        except OSError:
            pass
    return path


class PeriodSeries:
    """Values for one period column chart."""

    def __init__(self, labels, starts, values, priors, flags, decimals: int, nya=None, prior_flags=None):
        self.labels = labels            # period labels
        self.starts = starts            # period start dates
        self.values = values            # float or None (None = no value: not yet available or no data)
        self.priors = priors            # float or None
        self.flags = flags              # list of flag lists (not_yet_available left out)
        self.decimals = decimals
        # Periods not yet published; other blank values are 'no data' and carry a flag.
        self.nya = list(nya) if nya is not None else [v is None for v in values]
        # Flags on the year-earlier side (keys of PRIOR_FLAG_NOTES), one list per period.
        self.prior_flags = list(prior_flags) if prior_flags is not None else [[] for _ in values]


def _text_width_px(fig, renderer, text: str, size: float) -> float:
    t = fig.text(0, 0, text, fontsize=size)
    w = t.get_window_extent(renderer=renderer).width
    t.remove()
    return w


def _draw_period_axis(ax, fig, starts: list[date], freq: str, size: float = 8.5) -> None:
    """Month labels on one row with the year on its own row beneath (annual:
    years only). Drawn as text below the axis, not as tick labels."""
    import matplotlib.transforms as mtransforms

    ax.set_xticks([])
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    renderer = fig.canvas.get_renderer()
    ax_px = ax.get_window_extent(renderer=renderer).width
    lo, hi = ax.get_xlim()
    px_per_unit = ax_px / (hi - lo)
    # Offsets below the axis in points, converted to axes fraction.
    ax_h_pt = ax.get_window_extent(renderer=renderer).height * 72 / fig.dpi
    row1 = -6 / ax_h_pt
    row2 = -20 / ax_h_pt

    if freq == "annual":
        for i, s in enumerate(starts):
            ax.text(i, row1, str(s.year), transform=trans, ha="center", va="top", fontsize=size)
        return

    groups = month_groups(starts, freq)
    widths = [(b - a + 1) * px_per_unit for a, b, _ in groups]
    # Judge the style on whole months; a part month at either end of a weekly
    # chart gets a letter or nothing rather than forcing every label down.
    full_widths = [w for (a, b, m), w in zip(groups, widths)
                   if freq == "monthly" or (b - a + 1) >= 4] or widths
    measure = lambda s: _text_width_px(fig, renderer, s, size)  # noqa: E731
    style = pick_month_style(full_widths, measure)
    for (a, b, m), w in zip(groups, widths):
        text = month_text(m, style)
        if measure(text) + 4 > w:
            text = month_text(m, "letter")
            if measure(text) + 2 > w:
                continue
        ax.text((a + b) / 2, row1, text, transform=trans, ha="center", va="top", fontsize=size)
    # Small ticks between months for weekly charts, so the month spans read clearly.
    if freq == "weekly":
        for a, _, _ in groups[1:]:
            ax.plot([a - 0.5, a - 0.5], [0, -4 / ax_h_pt], transform=trans, color=AXIS_COLOUR,
                    linewidth=0.6, clip_on=False)
    years = year_groups(starts, freq)
    for a, b, y in years:
        ax.text((a + b) / 2, row2, str(y), transform=trans, ha="center", va="top", fontsize=size,
                fontweight="bold")
    for a, _, _ in years[1:]:
        ax.plot([a - 0.5, a - 0.5], [0, row2 - 10 / ax_h_pt], transform=trans, color=AXIS_COLOUR,
                linewidth=0.6, clip_on=False)


TITLE_PT, SUBTITLE_PT, FOOTER_PT = 13, 9, 6.8


def _frame(fig, title: str, subtitle: str, foot_lines: list[str], plot_h_in: float, below_axis_in: float,
           left: Optional[float] = None, right: float = 0.985) -> None:
    """Title, wrapped subtitle and wrapped footer, with the figure height set
    so none of them can overlap the legend, the plot or each other. Text
    widths depend only on the figure width, so they are measured first and
    the height is fitted around them."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w_in = fig.get_figwidth()
    max_px = w_in * fig.dpi * 0.97
    sub = wrap_to_width(subtitle, lambda t: _text_width_px(fig, renderer, t, SUBTITLE_PT), max_px)
    foot = "\n".join(wrap_to_width(x, lambda t: _text_width_px(fig, renderer, t, FOOTER_PT), max_px)
                     for x in foot_lines if x)
    n_sub = sub.count("\n") + 1
    n_foot = foot.count("\n") + 1 if foot else 0
    line_sub = SUBTITLE_PT * 1.3 / 72
    line_foot = FOOTER_PT * 1.35 / 72
    top_in = 0.40 + n_sub * line_sub + 0.48            # title, subtitle, then room for the legend row
    bottom_in = 0.10 + n_foot * line_foot + 0.16 + below_axis_in
    h_in = top_in + plot_h_in + bottom_in
    fig.set_size_inches(w_in, h_in)
    adjust = {"top": 1 - top_in / h_in, "bottom": bottom_in / h_in, "right": right}
    if left is not None:
        adjust["left"] = left
    fig.subplots_adjust(**adjust)
    fig.text(0.012, 1 - 0.12 / h_in, title, ha="left", va="top", fontsize=TITLE_PT, fontweight="bold")
    fig.text(0.012, 1 - 0.40 / h_in, sub, ha="left", va="top", fontsize=SUBTITLE_PT, color=TEXT_COLOUR,
             linespacing=1.3)
    if foot:
        fig.text(0.012, 0.10 / h_in, foot, ha="left", va="bottom", fontsize=FOOTER_PT, color=TEXT_COLOUR,
                 linespacing=1.35)
    fig.canvas.draw()


def _legend(fig, ax, colour: str, current_text: str, prior_text: Optional[str]) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colour, edgecolor="none", label=current_text)]
    if prior_text:
        handles.append(Line2D([0], [0], color=TICK_COLOUR, linewidth=1.0, label=prior_text))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles), frameon=False,
              fontsize=8, handlelength=1.4, borderaxespad=0.3, columnspacing=1.6)


def period_column_chart(series: PeriodSeries, freq: str, path: Path, *, title: str, subtitle: str,
                        colour: str, compare: bool, footer: list[str], legend_current: str,
                        legend_prior: str) -> Path:
    plt = _plt()
    n = len(series.labels)
    width_in = min(16.0, max(8.0, 3.0 + 0.42 * n)) if freq != "weekly" else min(16.0, max(8.0, 3.0 + 0.24 * n))
    flagged_kinds = sorted({f for fl in series.flags for f in fl if f in FLAG_NOTES})
    has_nya = any(series.nya)
    has_no_data = any(v is None and not ny for v, ny in zip(series.values, series.nya))
    foot_lines = list(footer)
    if flagged_kinds:
        foot_lines = ["* " + "; ".join(FLAG_NOTES[k] for k in flagged_kinds) + "."] + foot_lines
    pflags = ([[f for f in fl if f in PRIOR_FLAG_NOTES] if not ny else []
               for fl, ny in zip(series.prior_flags, series.nya)] if compare else [[] for _ in range(n)])
    prior_kinds = sorted({f for fl in pflags for f in fl})
    if prior_kinds:
        foot_lines = [PRIOR_MARK + " " + "; ".join(PRIOR_FLAG_NOTES[k] for k in prior_kinds) + "."] + foot_lines
    if has_nya or has_no_data:
        dash = []
        if has_nya:
            dash.append("not yet available (data not yet published for this period)")
        if has_no_data:
            dash.append("no data (see * note)")
        foot_lines = ["\u2014 " + "; ".join(dash) + "."] + foot_lines
    fig, ax = plt.subplots(figsize=(width_in, 4.6), dpi=DPI)
    fig.subplots_adjust(left=0.03, right=0.985)

    x = list(range(n))
    vals = [v if v is not None else 0.0 for v in series.values]
    ax.bar(x, vals, width=0.72, color=colour, edgecolor="none", zorder=2)
    tick_half = 0.36
    # No year-earlier line where the current period has no data yet.
    priors = ([p if v is not None else None for v, p in zip(series.values, series.priors)] if compare
              else [None] * n)
    for i, p in enumerate(priors):
        if p is not None:
            ax.plot([i - tick_half, i + tick_half], [p, p], color=TICK_COLOUR, linewidth=TICK_WIDTH_PT,
                    solid_capstyle="butt", zorder=3)
    heights = [v if v is not None else 0.0 for v in series.values]
    tops = [max(h, p if p is not None else 0.0) for h, p in zip(heights, priors)]
    ymax = max(tops) if tops and max(tops) > 0 else 1.0
    ax.set_ylim(0, ymax * 1.16)
    ax.set_xlim(-0.6, n - 0.4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOUR)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_yticks([])

    # Value labels sit just above the column; only when the year-earlier line
    # would run through the label are they pushed above the line instead.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer=renderer)
    pitch_px = ax_box.width / (n + 0.2)
    texts = []
    for i, v in enumerate(series.values):
        if v is None:
            texts.append("\u2014" + ("*" if series.flags[i] and not series.nya[i] else ""))
        else:
            texts.append(format_value(v, series.decimals) + ("*" if series.flags[i] else "")
                         + (PRIOR_MARK if pflags[i] else ""))
    size = 8.0
    while size > 6.0 and max(_text_width_px(fig, renderer, t, size) for t in texts) > pitch_px * 0.96:
        size -= 0.5
    show_values = max(_text_width_px(fig, renderer, t, size) for t in texts) <= pitch_px * 0.98
    if not show_values:
        foot_lines.insert(0, "Too many periods to label each column; the values are in the summary CSV.")
    _frame(fig, title, subtitle, foot_lines, plot_h_in=2.9, below_axis_in=0.68)
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer=renderer)
    px_per_val = ax_box.height / ax.get_ylim()[1]
    pt_px = fig.dpi / 72
    label_h_px = size * 1.25 * pt_px
    gap_px = 3 * pt_px
    anchors = label_anchors(heights, priors, px_per_val, gap_px, label_h_px)
    if show_values:
        for i, t in enumerate(texts):
            ax.annotate(t, (i, anchors[i]), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=size, color=TEXT_COLOUR)
    else:
        for i, v in enumerate(series.values):
            if v is None:
                ax.annotate("\u2014", (i, 0), xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=7, color=TEXT_COLOUR)

    _draw_period_axis(ax, fig, series.starts, freq)
    _legend(fig, ax, colour, legend_current, legend_prior if compare and any(p is not None for p in priors)
            else None)
    return _save(fig, plt, path)


def admin_bar_chart(rows: list[dict], path: Path, *, title: str, subtitle: str, compare: bool,
                    footer: list[str], max_bars: int = MAX_ADMIN_BARS) -> Optional[Path]:
    """Horizontal bars: whole-run fire detections per admin area, largest first,
    with the year-earlier count as a black line."""
    totals = [r for r in rows if str(r.get("period")) == "all"]
    if len(totals) < 2:
        return None
    totals.sort(key=lambda r: (-(num(r.get("fire_count")) or 0), str(r.get("unit_name"))))
    shown = totals[:max_bars]
    if all((num(r.get("fire_count")) or 0) == 0 and (num(r.get("prior_fire_count")) or 0) == 0 for r in shown):
        return None
    plt = _plt()
    names = [str(r.get("unit_name")) for r in shown]
    # Repeated names get their ID added so they stay apart.
    dup = {n for n in names if names.count(n) > 1}
    names = [f"{n} ({r.get('unit_id')})" if n in dup else n for n, r in zip(names, shown)]
    vals = [num(r.get("fire_count")) or 0.0 for r in shown]
    priors = [num(r.get("prior_fire_count")) if compare else None for r in shown]
    k = len(shown)
    foot_lines = list(footer)
    if len(totals) > k:
        foot_lines.insert(0, f"Showing the {k} areas with most detections, of {len(totals)}; all areas are in "
                             "the admin summary CSV.")
    fig, ax = plt.subplots(figsize=(9.0, 4.0), dpi=DPI)
    y = list(range(k))
    ax.barh(y, vals, height=0.68, color=ADMIN_COLOUR, edgecolor="none", zorder=2)
    for i, p in enumerate(priors):
        if p is not None:
            ax.plot([p, p], [i - 0.34, i + 0.34], color=TICK_COLOUR, linewidth=TICK_WIDTH_PT,
                    solid_capstyle="butt", zorder=3)
    ends = [max(v, p if p is not None else 0.0) for v, p in zip(vals, priors)]
    xmax = max(ends) if max(ends) > 0 else 1.0
    ax.set_xlim(0, xmax * 1.14)
    ax.set_ylim(k - 0.5, -0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5, color=TEXT_COLOUR)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOUR)
    ax.spines["left"].set_linewidth(0.8)
    fig.canvas.draw()
    # Left margin to fit the longest name, then value labels at the bar ends
    # (pushed past the year-earlier line only where it would cross the label).
    renderer = fig.canvas.get_renderer()
    widest = max(t.get_window_extent(renderer=renderer).width for t in ax.get_yticklabels())
    left = min(0.5, (widest + 14) / (fig.get_figwidth() * fig.dpi))
    _frame(fig, title, subtitle, foot_lines, plot_h_in=0.32 * k + 0.1, below_axis_in=0.22, left=left, right=0.97)
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer=renderer)
    labels = [format_value(v, 0) for v in vals]
    widths = [_text_width_px(fig, renderer, t, 8) for t in labels]
    anchors = label_anchors(vals, priors, ax_box.width / ax.get_xlim()[1], 4 * fig.dpi / 72, widths)
    for i, t in enumerate(labels):
        ax.annotate(t, (anchors[i], i), xytext=(4, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=8, color=TEXT_COLOUR)
    _legend(fig, ax, ADMIN_COLOUR, "This run",
            "Same periods a year earlier" if compare and any(p is not None for p in priors) else None)
    return _save(fig, plt, path)


# ---------------------------------------------------------------------------
# Building the charts from the tool's outputs
# ---------------------------------------------------------------------------

def _range_text(starts: list[date], ends: list[date], freq: str = "monthly") -> str:
    # Weeks are named by the month holding their Thursday, as on the axis.
    a, b = (label_month(starts[0], freq), label_month(starts[-1], freq)) if freq == "weekly" else (starts[0], ends[-1])
    if a.year == b.year:
        return f"{a:%B} to {b:%B %Y}" if a.month != b.month else f"{a:%B %Y}"
    return f"{a:%B %Y} to {b:%B %Y}"


def _years_text(starts: list[date], freq: str) -> str:
    ys = sorted({label_month(s, freq).year for s in starts})
    return str(ys[0]) if len(ys) == 1 else f"{ys[0]} to {ys[-1]}"


def write_charts(summary_rows: list[dict], out_dir: Path, geoextent: str, *, freq: str, compare: bool,
                 area_name: str, sensor_label: str, sensor_source: str, burned_on: bool,
                 admin_rows: Optional[list[dict]] = None, log: Callable[[str], None] = print) -> list[Path]:
    """Write every chart that applies and return their paths. Charts left from
    an earlier run of the same geoextent are cleared first."""
    out_dir = Path(out_dir)
    locked: list = []
    clear_charts(out_dir, geoextent, locked)
    if locked:
        log("Warning: could not remove old chart(s) " + ", ".join(locked) + " (open in another program?)")
    if not summary_rows:
        return []
    labels = [str(r["period"]) for r in summary_rows]
    starts = [date.fromisoformat(str(r["period_start"])) for r in summary_rows]
    ends = [date.fromisoformat(str(r["period_end"])) for r in summary_rows]
    unit = FREQ_WORD[freq]
    when = _range_text(starts, ends, freq)
    written: list[Path] = []
    common_foot = [f"Chart: fire activity tool. Area: {area_name}."]

    # Fire detections
    flags = []
    values = []
    for r in summary_rows:
        f = str(r.get("qa_flag") or "ok")
        values.append(None if NOT_YET in flag_parts(f) else num(r.get("fire_count")))
        flags.append([x for x in flag_parts(f) if x != NOT_YET])
    priors = [num(r.get("prior_fire_count")) for r in summary_rows]
    det_prior_flags = [[f for f in flag_parts(str(r.get("prior_qa_flag") or "")) if f in PRIOR_FLAG_NOTES]
                       for r in summary_rows]
    det = PeriodSeries(labels, starts, values, priors, flags, 0,
                       [NOT_YET in flag_parts(str(r.get("qa_flag") or "")) for r in summary_rows],
                       det_prior_flags)
    det_path = period_column_chart(
        det, freq, out_dir / chart_file_name(geoextent, "detections"),
        title=f"Active fire detections, {area_name}",
        subtitle=f"Fire detections per {unit}, {when}. One detection is one fire pixel on one day; {sensor_label}.",
        colour=DETECTION_COLOUR, compare=compare,
        footer=[f"Source: {sensor_source}, via Google Earth Engine. Near-real-time data, described by NASA as "
                "not science quality.", *common_foot],
        legend_current=_years_text(starts, freq), legend_prior="Same period a year earlier")
    written.append(det_path)

    # Burned area
    if burned_on and any("burned_area_km2" in r for r in summary_rows):
        b_vals, b_flags, b_nya = [], [], []
        for r in summary_rows:
            f = str(r.get("burned_qa_flag") or "ok")
            v = num(r.get("burned_area_km2"))
            b_vals.append(None if NOT_YET in flag_parts(f) else v)
            b_flags.append([x for x in flag_parts(f) if x != NOT_YET])
            b_nya.append(NOT_YET in flag_parts(f))
        if any(v is not None for v in b_vals):
            b_priors = [num(r.get("prior_burned_area_km2")) for r in summary_rows]
            dec = burned_decimals(b_vals + b_priors)
            weekly_note = (["Weekly burned area is worked out from each pixel's approximate burn day, so treat "
                            "it as approximate."] if freq == "weekly" else [])
            b_prior_flags = []
            for r in summary_rows:
                cov = num(r.get("prior_burned_mapped_pct"))
                b_prior_flags.append(["low_coverage"] if cov is not None and cov < LOW_COVERAGE_PCT else [])
            bs = PeriodSeries(labels, starts, b_vals, b_priors, b_flags, dec, b_nya, b_prior_flags)
            written.append(period_column_chart(
                bs, freq, out_dir / chart_file_name(geoextent, "burned"),
                title=f"Burned area, {area_name}",
                subtitle=f"Square kilometres burned per {unit}, {when}. MODIS burned area, 500 m, published "
                         "monthly about three months behind.",
                colour=BURNED_COLOUR, compare=compare,
                footer=[*weekly_note, "Source: NASA MODIS burned area (MCD64A1 collection 6.1), via Google "
                        "Earth Engine.", *common_foot],
                legend_current=_years_text(starts, freq), legend_prior="Same period a year earlier"))
        else:
            log("Charts: no burned area chart (no published burned area for these periods yet)")

    # Admin areas
    if admin_rows:
        p = admin_bar_chart(
            admin_rows, out_dir / chart_file_name(geoextent, "admin"),
            title=f"Active fire detections by area, {area_name}",
            subtitle=f"Fire detections, {when}, whole run; {sensor_label}.",
            compare=compare,
            footer=["The year-earlier line counts only periods that have a year-earlier match."
                    if compare and any(num(r.get("prior_fire_count")) is not None for r in admin_rows
                                       if str(r.get("period")) == "all") else "",
                    f"Source: {sensor_source}, via Google Earth Engine.", *common_foot])
        if p:
            written.append(p)
    return written


def area_name_from_file(path: str, fallback: str) -> str:
    """A readable area name for chart titles: the most detailed name field whose
    value is the same for every feature (Scotland for a single-feature Scotland file, or
    for a file of Scotland's council areas), otherwise `fallback`."""
    try:
        import pyogrio
        df = pyogrio.read_dataframe(path, read_geometry=False)
    except Exception:  # noqa: BLE001 - a title is never worth failing a run for
        return fallback
    candidates = ("ADM4_EN", "ADM3_EN", "ADM2_EN", "ADM1_EN", "ADM0_EN", "adm2_name", "adm1_name", "adm0_name",
                  "NAME_4", "NAME_3", "NAME_2", "NAME_1", "NAME_0", "ADM2_NAME", "ADM1_NAME", "ADM0_NAME",
                  "shapeName", "name_en", "NAME_EN", "name", "NAME", "Name", "COUNTRY", "country")
    for c in candidates:
        if c in df.columns:
            vals = df[c].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            if len(vals) == len(df) and vals.nunique() == 1:
                return vals.iloc[0]
    return fallback


# ---------------------------------------------------------------------------
# Command line: redraw charts from an existing summary CSV
# ---------------------------------------------------------------------------

SENSOR_TEXT = {
    "viirs": ("VIIRS 375 m (Suomi NPP)", "NASA LANCE VIIRS active fire (Suomi NPP, collection 2)"),
    "modis": ("MODIS 1 km", "NASA FIRMS MODIS active fire"),
}


def charts_from_csv(csv_path: str, area_name: Optional[str] = None, out_dir: Optional[str] = None,
                    log: Callable[[str], None] = print) -> list[Path]:
    import pandas as pd

    csv = Path(csv_path)
    m = re.fullmatch(r"(.+)_fire_activity_summary\.csv", csv.name)
    if not m:
        raise ValueError(f"{csv.name} does not look like a fire activity summary CSV "
                         "({geoextent}_fire_activity_summary.csv).")
    geoextent = m.group(1)
    df = pd.read_csv(csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rows = df.to_dict("records")
    if not rows:
        raise ValueError(f"{csv.name} has no rows.")
    freq = infer_freq([r["period"] for r in rows])
    compare = "prior_fire_count" in df.columns
    sensor_key = rows[0].get("sensor", "viirs")
    label, source = SENSOR_TEXT.get(sensor_key, SENSOR_TEXT["viirs"])
    admin_csv = csv.with_name(f"{geoextent}_fire_activity_admin_summary.csv")
    admin_rows = (pd.read_csv(admin_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
                  .to_dict("records") if admin_csv.exists() else None)
    if not area_name:
        info = csv.with_name(f"{geoextent}_fire_activity_run_info.txt")
        if info.exists():
            m2 = re.search(r"^Charts: .*area name '(.+)'$", info.read_text(encoding="utf-8", errors="replace"),
                           re.MULTILINE)
            area_name = m2.group(1) if m2 else None
    written = write_charts(rows, Path(out_dir) if out_dir else csv.parent, geoextent, freq=freq, compare=compare,
                           area_name=area_name or geoextent.upper(), sensor_label=label, sensor_source=source,
                           burned_on="burned_area_km2" in df.columns, admin_rows=admin_rows, log=log)
    for p in written:
        log(f"  {p}")
    return written


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Redraw the fire activity charts from an existing summary CSV.")
    p.add_argument("summary_csv", help="a {geoextent}_fire_activity_summary.csv written by fire_tool.py")
    p.add_argument("--area-name", help="area name for the chart titles, e.g. Scotland (default: the name used "
                                       "by the run that wrote the CSV, otherwise the geoextent code)")
    p.add_argument("--out-dir", help="folder for the PNGs (default: next to the CSV)")
    args = p.parse_args(argv)
    try:
        charts_from_csv(args.summary_csv, args.area_name, args.out_dir)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
