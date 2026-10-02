#!/usr/bin/env python3
"""
fire_tool.py - active fire detections for an area of interest, by period, with
an optional same-period-last-year comparison, summarised by hexagon, admin area
and kernel density (spatial summaries are in fire_spatial.py).

Given an area-of-interest boundary file (shapefile, GeoPackage, GeoJSON), a date
range and a frequency (weekly, monthly or annual), this pulls daily active fire
detections from Google Earth Engine, turns them into points, and writes:

  {geoextent}_fire_activity.gpkg            fire points, hexagon and admin area layers
  {geoextent}_fire_activity_shp/            the same layers as shapefiles (--vector-format shp or both)
  {geoextent}_fire_activity_summary.csv     one row per period
  {geoextent}_fire_activity_admin_summary.csv   one row per admin area per period (if admin areas)
  {geoextent}_fire_activity_density/        kernel density rasters (fires per km2)
  {geoextent}_fire_activity_burned/         burned area rasters (share of each cell burned)
  {geoextent}_fire_activity_run_info.txt    which sensor was used and why, settings

Burned area (fire_burned.py) comes from MODIS MCD64A1 (500 m, monthly, about
three months behind) and adds burned km2 columns to the summary, hexagon and
admin outputs. Switch it off with --no-burned-area.

Quick start (from ArcGIS Pro's Python Command Prompt, with earth-engine-gis active):

    python fire_tool.py --aoi-file aoi\\my_area.shp --geoextent abc ^
        --start 2025-01-01 --end 2026-01-01 --freq monthly ^
        --compare-prior-year --out-dir out

Earth Engine uses the project saved once with 'earthengine set_project
<your-project-id>'; --ee-project overrides it for one run.

Add --dry-run to see the sensor choice, periods and download chunks without
contacting Earth Engine. Run --diagnose once to check the Earth Engine datasets
look the way this script expects.

Design notes
------------
The core of the fire activity tool. Pure logic
(periods, sensor rule, chunking, summaries, file naming) has no Earth Engine
dependency so it is unit-tested offline in tests/. Only the EarthEngineFetcher
class and run_diagnose() touch the `ee` module.

Counting rule: one detection = one fire pixel on one day. If a dataset holds more
than one image for a day, duplicate pixels on that day are collapsed to one.

Sensor rule: Earth Engine's VIIRS 375 m archive starts 3 September 2023. A run
uses VIIRS only if every date it needs (including year-earlier comparison
periods) falls on or after that date; otherwise the whole run uses MODIS 1 km,
so a comparison never mixes sensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

import fire_burned
import fire_charts
import fire_spatial
from shapely.errors import ShapelyError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_FREQS = ("weekly", "monthly", "annual")
SENSOR_CHOICES = ("auto", "viirs", "modis")

VIIRS_ARCHIVE_START = date(2023, 9, 3)
MODIS_ARCHIVE_START = date(2000, 11, 1)

# Earth Engine requests get uncomfortable above a few MB of geometry. If the
# area of interest serialises larger than this, it is simplified automatically.
MAX_AOI_GEOJSON_BYTES = 1_500_000
AUTO_SIMPLIFY_STEPS_DEG = (0.0005, 0.001, 0.002, 0.005, 0.01)

# Download cache: bump CACHE_VERSION whenever bands or sampling change, so old
# cached chunks are never reused. Chunks ending within CACHE_SETTLE_DAYS of the
# latest data are not cached, because near-real-time data can still be revised.
CACHE_VERSION = "v2"
CACHE_SETTLE_DAYS = 7
# Days after which missing imagery is treated as a permanent gap (no backfill coming).
GAP_SETTLE_DAYS = 90
# Above this many detections a run warns that it will be slow and memory-hungry.
LARGE_RUN_DETECTIONS = 3_000_000

# Retries for rate limits and network drops (seconds to wait before each retry).
RETRY_WAITS_S = (10, 30, 60)


@dataclass(frozen=True)
class SensorConfig:
    key: str
    label: str
    collection_id: str
    archive_start: date
    resolution_m: int
    default_min_confidence: int
    high_confidence: int
    confidence_note: str
    satellite: str
    # Band that is above zero only on fire pixels. Needed because the VIIRS rasters
    # are not masked: non-fire pixels hold 0 in every band, and confidence 0 (low)
    # is indistinguishable from background (found with --diagnose, 1 October 2026).
    fire_band: str
    # Earth Engine band name -> output column name
    bands: tuple[tuple[str, str], ...]


SENSORS: dict[str, SensorConfig] = {
    "viirs": SensorConfig(
        key="viirs",
        label="VIIRS (Suomi NPP), 375 m, near real time",
        collection_id="NASA/LANCE/SNPP_VIIRS/C2",
        archive_start=VIIRS_ARCHIVE_START,
        resolution_m=375,
        default_min_confidence=1,
        high_confidence=2,
        confidence_note="0 = low, 1 = nominal, 2 = high",
        satellite="Suomi NPP",
        fire_band="Bright_ti4",
        bands=(
            ("confidence", "confidence"),
            ("frp", "frp_mw"),
            ("Bright_ti4", "brightness_k"),
            ("DayNight", "day_night_code"),
            ("acq_time", "acq_time_s"),
        ),
    ),
    "modis": SensorConfig(
        key="modis",
        label="MODIS (Terra and Aqua, via FIRMS), 1 km",
        collection_id="FIRMS",
        archive_start=MODIS_ARCHIVE_START,
        resolution_m=1000,
        default_min_confidence=30,
        high_confidence=80,
        confidence_note="0 to 100 per cent",
        satellite="Terra/Aqua",
        fire_band="T21",
        bands=(
            ("confidence", "confidence"),
            ("T21", "brightness_k"),
        ),
    ),
}

# Columns every fetched detection table is normalised to.
DETECTION_COLUMNS = [
    "date", "longitude", "latitude", "confidence", "frp_mw", "brightness_k",
    "day_night_code", "acq_time_s",
]

SUMMARY_COLUMNS = [
    "period", "period_start", "period_end", "sensor", "fire_count",
    "high_confidence_count", "day_count", "night_count", "frp_sum_mw",
    "frp_mean_mw", "days_with_fire", "days_in_period", "days_with_imagery",
    "qa_flag",
]
COMPARISON_COLUMNS = [
    "prior_period", "prior_fire_count", "prior_days_with_imagery",
    "prior_qa_flag", "change_abs", "change_pct", "comparison_note",
]


# ---------------------------------------------------------------------------
# Periods (adapted from the nightlight tool)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Period:
    label: str      # "2025-03" (monthly), "2025" (annual), "2025-W05" (weekly)
    start: date     # inclusive
    end: date       # exclusive

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    @property
    def last_day(self) -> date:
        return self.end - timedelta(days=1)


def fmt_date(d: date) -> str:
    """Prose date, e.g. 3 September 2023."""
    return f"{d.day} {d:%B %Y}"


def _month_add(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    return date(d.year + month_index // 12, month_index % 12 + 1, 1)


def build_periods(start: str, end: str, freq: str) -> list[Period]:
    """Split [start, end) into Periods. `end` is exclusive. Monthly snaps to the
    1st, annual to 1 January, weekly to the Monday of the start date's ISO week,
    so the first and last periods are always whole periods."""
    if freq not in VALID_FREQS:
        raise ValueError(f"--freq must be one of {', '.join(VALID_FREQS)}, got {freq!r}")
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    if end_d <= start_d:
        raise ValueError(f"--end ({end}) must be after --start ({start})")

    periods: list[Period] = []
    if freq == "monthly":
        cur = date(start_d.year, start_d.month, 1)
        while cur < end_d:
            nxt = _month_add(cur, 1)
            periods.append(Period(f"{cur.year:04d}-{cur.month:02d}", cur, nxt))
            cur = nxt
    elif freq == "weekly":
        cur = start_d - timedelta(days=start_d.weekday())
        while cur < end_d:
            nxt = cur + timedelta(days=7)
            iso_year, iso_week, _ = cur.isocalendar()
            periods.append(Period(f"{iso_year:04d}-W{iso_week:02d}", cur, nxt))
            cur = nxt
    else:  # annual
        cur = date(start_d.year, 1, 1)
        while cur < end_d:
            nxt = date(cur.year + 1, 1, 1)
            periods.append(Period(f"{cur.year:04d}", cur, nxt))
            cur = nxt
    return periods


def year_ago_period(period: Period, freq: str) -> Optional[Period]:
    """The same period one year earlier: same month, same year, or same ISO week
    number. Returns None for ISO week 53 when the previous year has no week 53."""
    if freq == "monthly":
        start_d = date(period.start.year - 1, period.start.month, 1)
        return Period(f"{start_d.year:04d}-{start_d.month:02d}", start_d, _month_add(start_d, 1))
    if freq == "annual":
        y = period.start.year - 1
        return Period(f"{y:04d}", date(y, 1, 1), date(y + 1, 1, 1))
    if freq == "weekly":
        iso_year, iso_week, _ = period.start.isocalendar()
        try:
            start_d = date.fromisocalendar(iso_year - 1, iso_week, 1)
        except ValueError:
            return None
        return Period(f"{iso_year - 1:04d}-W{iso_week:02d}", start_d, start_d + timedelta(days=7))
    raise ValueError(f"freq must be one of {VALID_FREQS}, got {freq!r}")


def build_prior_map(periods: list[Period], freq: str) -> dict[str, Optional[Period]]:
    return {p.label: year_ago_period(p, freq) for p in periods}


# ---------------------------------------------------------------------------
# Sensor rule
# ---------------------------------------------------------------------------

def required_date_range(periods: list[Period], prior_map: Optional[dict]) -> tuple[date, date]:
    """Earliest start and latest (exclusive) end of every period the run needs,
    including year-earlier comparison periods."""
    starts = [p.start for p in periods]
    ends = [p.end for p in periods]
    if prior_map:
        for prior in prior_map.values():
            if prior is not None:
                starts.append(prior.start)
                ends.append(prior.end)
    return min(starts), max(ends)


def choose_sensor(requested: str, periods: list[Period], prior_map: Optional[dict]) -> tuple[str, str]:
    """Return (sensor_key, plain-English reason). Raises ValueError when the
    requested sensor cannot cover the dates."""
    if requested not in SENSOR_CHOICES:
        raise ValueError(f"--sensor must be one of {SENSOR_CHOICES}, got {requested!r}")
    earliest, _ = required_date_range(periods, prior_map)
    needs = "including year-earlier comparison periods" if prior_map else "for the requested periods"
    viirs_ok = earliest >= VIIRS_ARCHIVE_START

    if earliest < MODIS_ARCHIVE_START:
        raise ValueError(
            f"The run needs data from {fmt_date(earliest)} ({needs}), but the earliest "
            f"fire data available is MODIS from {fmt_date(MODIS_ARCHIVE_START)}."
        )
    if requested == "viirs":
        if not viirs_ok:
            raise ValueError(
                f"--sensor viirs: the run needs data from {fmt_date(earliest)} ({needs}), but "
                f"Earth Engine's VIIRS archive starts {fmt_date(VIIRS_ARCHIVE_START)}. Use a later "
                "--start, drop --compare-prior-year, or use --sensor auto / modis."
            )
        return "viirs", "VIIRS requested with --sensor viirs."
    if requested == "modis":
        return "modis", "MODIS requested with --sensor modis."
    if viirs_ok:
        return "viirs", (
            f"Every date needed ({needs}) falls on or after {fmt_date(VIIRS_ARCHIVE_START)}, "
            "so the finer 375 m VIIRS data was used."
        )
    return "modis", (
        f"The run needs data from {fmt_date(earliest)} ({needs}), before Earth Engine's VIIRS "
        f"archive starts on {fmt_date(VIIRS_ARCHIVE_START)}, so MODIS (1 km) was used for every "
        "period to keep comparisons on one sensor."
    )


# ---------------------------------------------------------------------------
# Download windows and chunks
# ---------------------------------------------------------------------------

def merge_windows(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Merge overlapping or touching [start, end) intervals."""
    merged: list[tuple[date, date]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def fetch_windows(periods: list[Period], prior_map: Optional[dict]) -> list[tuple[date, date]]:
    intervals = [(p.start, p.end) for p in periods]
    if prior_map:
        intervals += [(p.start, p.end) for p in prior_map.values() if p is not None]
    return merge_windows(intervals)


def split_into_month_chunks(windows: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Cut each window at calendar month boundaries, so chunks line up between
    runs and the download cache can be reused."""
    chunks = []
    for s, e in windows:
        cur = s
        while cur < e:
            nxt = min(_month_add(date(cur.year, cur.month, 1), 1), e)
            chunks.append((cur, nxt))
            cur = nxt
    return chunks


class SplittableFetchError(Exception):
    """A chunk failed for size reasons (memory, time-out) and may succeed if split."""


class TransientFetchError(Exception):
    """A chunk failed for a temporary reason (rate limit, network) and may succeed
    if the same request is retried after a pause."""


def _fetch_with_retry(fetch, start, end, log, waits=None, sleep=time.sleep):
    waits = RETRY_WAITS_S if waits is None else waits
    for attempt, wait in enumerate(list(waits) + [None]):
        try:
            return fetch(start, end)
        except TransientFetchError as exc:
            if wait is None:
                raise RuntimeError(f"Earth Engine kept refusing the request for {start} to {end}: {exc}. "
                                   "Wait a few minutes and run again; finished months are cached.") from exc
            log(f"    temporary problem ({exc}); retrying in {wait} s")
            sleep(wait)


def fetch_with_split(fetch: Callable[[date, date], pd.DataFrame], start: date, end: date,
                     log: Callable[[str], None] = print) -> pd.DataFrame:
    """Fetch [start, end); on a size failure, halve the window and retry, down to
    single days."""
    try:
        return _fetch_with_retry(fetch, start, end, log)
    except SplittableFetchError as exc:
        days = (end - start).days
        if days <= 1:
            raise RuntimeError(
                f"Earth Engine could not process a single day ({start}) for this area: {exc}. "
                "Try a coarser --simplify-tolerance or a smaller area."
            ) from exc
        mid = start + timedelta(days=days // 2)
        log(f"    too large for one request, splitting {start} to {end} at {mid}")
        parts = [fetch_with_split(fetch, start, mid, log), fetch_with_split(fetch, mid, end, log)]
        return pd.concat([p for p in parts if not p.empty] or [empty_detections()], ignore_index=True)


# ---------------------------------------------------------------------------
# Detection tables
# ---------------------------------------------------------------------------

def empty_detections() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in DETECTION_COLUMNS}).astype({"date": "object"})


def normalise_detections(raw: pd.DataFrame, sensor: SensorConfig) -> pd.DataFrame:
    """Rename Earth Engine band columns to output names, keep only known columns,
    and coerce types. Missing columns (e.g. frp for MODIS) become empty."""
    if raw is None or raw.empty:
        return empty_detections()
    df = raw.rename(columns=dict(sensor.bands))
    for col in DETECTION_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[DETECTION_COLUMNS].copy()
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    for col in DETECTION_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def dedupe_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeat detections of the same pixel on the same day to one,
    keeping the most confident (then most intense) record."""
    if df.empty:
        return df
    work = df.copy()
    work["_lon"] = work["longitude"].round(6)
    work["_lat"] = work["latitude"].round(6)
    work = work.sort_values(["confidence", "frp_mw"], ascending=False, na_position="last")
    work = work.drop_duplicates(subset=["date", "_lon", "_lat"], keep="first")
    work = work.drop(columns=["_lon", "_lat"]).sort_values(["date", "latitude", "longitude"])
    return work.reset_index(drop=True)


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Detections in date order, so detections_in() can find a period by binary
    search instead of scanning every row for every period (large areas)."""
    if df.empty or df["date"].is_monotonic_increasing:
        return df
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def detections_in(df: pd.DataFrame, period: Period, assume_sorted: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    if assume_sorted:
        dates = df["date"].to_numpy()
        lo = dates.searchsorted(period.start.isoformat(), side="left")
        hi = dates.searchsorted(period.end.isoformat(), side="left")
        return df.iloc[lo:hi]
    mask = (df["date"] >= period.start.isoformat()) & (df["date"] < period.end.isoformat())
    return df[mask]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def period_qa_flag(period: Period, days_with_imagery: int, latest_data_date: Optional[date]) -> str:
    if latest_data_date is not None and period.start > latest_data_date:
        return "not_yet_available"
    if latest_data_date is not None and period.last_day > latest_data_date:
        return "partial_period"
    if days_with_imagery == 0:
        return "no_imagery"
    if days_with_imagery < period.days:
        return "missing_days"
    return "ok"


def period_stats(df: pd.DataFrame, period: Period, sensor: SensorConfig,
                 imagery_dates: set[str], latest_data_date: Optional[date], assume_sorted: bool = False) -> dict:
    sub = detections_in(df, period, assume_sorted)
    days_with_imagery = sum(
        1 for i in range(period.days) if (period.start + timedelta(days=i)).isoformat() in imagery_dates
    )
    has_daynight = sensor.key == "viirs"
    has_frp = sensor.key == "viirs"
    frp = sub["frp_mw"].dropna() if has_frp else pd.Series(dtype=float)
    return {
        "period": period.label,
        "period_start": period.start.isoformat(),
        "period_end": period.last_day.isoformat(),
        "sensor": sensor.key,
        "fire_count": int(len(sub)),
        "high_confidence_count": int((sub["confidence"] >= sensor.high_confidence).sum()),
        "day_count": int((sub["day_night_code"] == 1).sum()) if has_daynight else "",
        "night_count": int((sub["day_night_code"] == 0).sum()) if has_daynight else "",
        "frp_sum_mw": round(float(frp.sum()), 1) if has_frp else "",
        "frp_mean_mw": round(float(frp.mean()), 1) if has_frp and len(frp) else "",
        "days_with_fire": int(sub["date"].nunique()),
        "days_in_period": period.days,
        "days_with_imagery": days_with_imagery,
        "qa_flag": period_qa_flag(period, days_with_imagery, latest_data_date),
    }


def change_values(current: int, prior: Optional[int]) -> tuple:
    if prior is None:
        return "", ""
    abs_change = current - prior
    pct = round(100.0 * abs_change / prior, 1) if prior > 0 else ""
    return abs_change, pct


def imagery_notes(cur_days: int, cur_window: int, prior_days: int, prior_window: int) -> list[str]:
    """Plain-English warnings when either side of a comparison has days without
    satellite imagery. Counts are left as they are (fire is too episodic to scale
    by days), so the note is the warning."""
    notes = []
    if cur_days == 0:
        return ["no satellite imagery in this period, so no comparison"]
    if prior_days == 0:
        notes.append("no satellite imagery in the year-earlier period, so no comparison")
    elif prior_days < prior_window:
        notes.append(f"year-earlier period has imagery on only {prior_days} of {prior_window} days, "
                     "so the change is likely overstated")
    if cur_days < cur_window:
        notes.append(f"this period has imagery on only {cur_days} of {cur_window} days, "
                     "so the change is likely understated")
    return notes


def uneven_comparisons(rows: list[dict]) -> list[str]:
    """Periods whose year-earlier comparison is affected by missing imagery on either side."""
    return [str(r["period"]) for r in rows if "imagery" in str(r.get("comparison_note") or "")]


def build_summary(df: pd.DataFrame, periods: list[Period], prior_map: Optional[dict],
                  sensor: SensorConfig, imagery_dates: set[str],
                  latest_data_date: Optional[date]) -> list[dict]:
    rows = []
    df = sort_by_date(df)
    for p in periods:
        row = period_stats(df, p, sensor, imagery_dates, latest_data_date, assume_sorted=True)
        if prior_map is not None:
            prior = prior_map.get(p.label)
            if prior is None:
                row.update({"prior_period": "", "prior_fire_count": "", "prior_days_with_imagery": "",
                            "prior_qa_flag": "", "change_abs": "", "change_pct": "",
                            "comparison_note": "no matching ISO week 53 in the previous year"})
            else:
                note = ""
                compare_to = prior
                if row["qa_flag"] == "partial_period" and latest_data_date is not None:
                    # Like-for-like: compare only the same number of days of the prior period.
                    days_available = (latest_data_date - p.start).days + 1
                    compare_to = Period(prior.label, prior.start, prior.start + timedelta(days=days_available))
                    note = f"first {days_available} days of both periods only"
                elif row["qa_flag"] == "not_yet_available":
                    compare_to = None
                    note = "no current data yet"
                if compare_to is None:
                    row.update({"prior_period": prior.label, "prior_fire_count": "", "prior_days_with_imagery": "",
                                "prior_qa_flag": "", "change_abs": "", "change_pct": ""})
                else:
                    ps = period_stats(df, compare_to, sensor, imagery_dates, latest_data_date, assume_sorted=True)
                    ch_abs, ch_pct = change_values(row["fire_count"], ps["fire_count"])
                    row.update({"prior_period": prior.label, "prior_fire_count": ps["fire_count"],
                                "prior_days_with_imagery": ps["days_with_imagery"],
                                "prior_qa_flag": ps["qa_flag"], "change_abs": ch_abs, "change_pct": ch_pct})
                    if ps["qa_flag"] == "no_imagery" or row["qa_flag"] == "no_imagery":
                        # Nothing to compare: a count of 0 there means "not observed", not "no fires".
                        # A blank prior count also keeps the period out of the hexagon, admin and
                        # density "compared" totals.
                        row.update({"prior_fire_count": "", "change_abs": "", "change_pct": ""})
                    # Each side is measured against its own length (February, leap years); a
                    # partial current period only against the days compared.
                    cur_window = compare_to.days if row["qa_flag"] == "partial_period" else p.days
                    note = "; ".join(x for x in [note, *imagery_notes(row["days_with_imagery"], cur_window,
                                                                      ps["days_with_imagery"], compare_to.days)]
                                     if x)
                row["comparison_note"] = note
        rows.append(row)
    return rows


def assign_period_labels(dates: pd.Series, periods: list[Period]) -> pd.Series:
    """Label each ISO date string with the period containing it ('' if none).
    Periods must be sorted and non-overlapping."""
    if not len(dates):
        return pd.Series(dtype="object")
    import numpy as np
    starts = np.array([p.start.isoformat() for p in periods], dtype=object)
    ends = np.array([p.end.isoformat() for p in periods], dtype=object)
    labels = np.array([p.label for p in periods] + [""], dtype=object)    # last slot = no period
    d = dates.fillna("").astype(str).to_numpy(dtype=object)
    i = np.searchsorted(starts, d, side="right") - 1
    ok = i >= 0
    ok[ok] = d[ok] < ends[i[ok]]
    i = np.where(ok, i, len(periods))
    return pd.Series(labels[i], index=dates.index, dtype="object")


def build_point_tables(df: pd.DataFrame, periods: list[Period], prior_map: Optional[dict],
                       sensor: SensorConfig, latest_data_date: Optional[date] = None
                       ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Split detections into the current-period table and (if comparing) the
    year-earlier table, with readable columns."""
    def tidy(sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.copy()
        out["sensor"] = sensor.key
        out["satellite"] = sensor.satellite
        if sensor.key == "viirs":
            out["day_night"] = out["day_night_code"].map({1: "day", 0: "night"}).fillna("")
            secs = pd.to_numeric(out["acq_time_s"], errors="coerce")
            hh = (secs // 3600).astype("Int64").astype(str).str.zfill(2)
            mm = ((secs % 3600) // 60).astype("Int64").astype(str).str.zfill(2)
            out["acq_time_utc"] = (hh + ":" + mm).where(secs.notna(), "")
        else:
            out["day_night"] = ""
            out["acq_time_utc"] = ""
            out["frp_mw"] = float("nan")
        return out.drop(columns=["day_night_code", "acq_time_s"])

    df = sort_by_date(df)
    current = df.copy()
    current["period"] = assign_period_labels(current["date"], periods) if len(current) else ""
    current = current[current["period"] != ""] if len(current) else current
    current = tidy(current)

    prior_table = None
    if prior_map is not None:
        pairs = [(lbl, pr) for lbl, pr in prior_map.items() if pr is not None]
        by_label = {p.label: p for p in periods}
        parts = []
        for lbl, pr in pairs:
            sub = detections_in(df, pr, assume_sorted=True).copy()
            if sub.empty:
                continue
            sub["period"] = pr.label
            sub["compares_with"] = lbl
            # Mirrors build_summary: a partial current period is compared with the
            # same number of days of the prior period only.
            cur = by_label[lbl]
            in_window = "yes"
            if latest_data_date is not None and cur.start > latest_data_date:
                sub["in_comparison"] = "no"
            elif latest_data_date is not None and latest_data_date < cur.last_day:
                cutoff = (pr.start + timedelta(days=(latest_data_date - cur.start).days + 1)).isoformat()
                sub["in_comparison"] = (sub["date"] < cutoff).map({True: "yes", False: "no"})
            else:
                sub["in_comparison"] = in_window
            parts.append(sub)
        prior_df = (pd.concat(parts, ignore_index=True) if parts
                    else empty_detections().assign(period="", compares_with="", in_comparison=""))
        prior_table = tidy(prior_df)
    return current.reset_index(drop=True), prior_table


# ---------------------------------------------------------------------------
# Naming and writing outputs
# ---------------------------------------------------------------------------

def sanitize_token(s: str) -> str:
    import re
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(s).lower()).strip("_")
    return token or "na"


def output_filename(geoextent: str, suffix: str) -> str:
    """e.g. abc_fire_activity.gpkg, abc_fire_activity_summary.csv"""
    return f"{sanitize_token(geoextent)}_fire_activity{suffix}"


def points_layer_name(geoextent: str, sensor_key: str, prior: bool = False) -> str:
    """Layer names inside the GeoPackage, following the GIS data naming
    convention: geoextent_category_theme_geometry_scale_source_permission_freetext.
    Category evnt, theme fir (fire location), point, s0 (as the nightlight tool),
    source nasa, permission pp."""
    free = f"{sensor_key}_prior_year" if prior else sensor_key
    return f"{sanitize_token(geoextent)}_evnt_fir_pt_s0_nasa_pp_{sanitize_token(free)}"


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV via a temporary file, so a CSV open in Excel is never left half
    written, and say plainly what to do if it is locked."""
    tmp = path.with_name(path.stem + "_writing.csv")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")   # BOM so Excel reads accents
    try:
        os.replace(tmp, path)
    except PermissionError as exc:
        raise RuntimeError(f"Could not replace {path.name}; it is probably open in Excel. Close it and run "
                           f"again. The new version is in {tmp.name} meanwhile.") from exc


def write_summary_csv(rows: list[dict], path: Path, compare: bool, burned: bool = False) -> None:
    cols = SUMMARY_COLUMNS + (COMPARISON_COLUMNS if compare else [])
    if burned:
        cols += (fire_burned.SUMMARY_COLUMNS + (fire_burned.COMPARISON_COLUMNS if compare else [])
                 + [fire_burned.NOTE_COLUMN])
    write_csv_atomic(pd.DataFrame(rows, columns=cols), path)


def run_output_files(out_dir: Path, geoextent: str) -> list[Path]:
    """Every file a run of this geoextent writes or replaces in the output folder."""
    out_dir = Path(out_dir)
    names = ["_summary.csv", ".gpkg", "_writing.gpkg", "_admin_summary.csv", "_run_info.txt"]
    files = [out_dir / output_filename(geoextent, n) for n in names]
    # Charts are left out on purpose: a chart open in a viewer must not stop a run
    # (write_charts skips a locked chart and says so).
    files += fire_spatial.density_raster_files(out_dir, geoextent)
    files += fire_burned.burned_raster_files(out_dir, geoextent)
    files += shapefile_files(out_dir, geoextent)
    return [f for f in files if f.exists()]


LOCK_PROBE_SUFFIX = ".lockcheck"


def restore_lock_probes(folders: list[Path]) -> None:
    """Put back any output a crashed or interrupted lock check left renamed
    (X.<pid>.lockcheck while X is missing); remove probes whose file was
    written again since."""
    for folder in folders:
        if not Path(folder).is_dir():
            continue
        for probe in Path(folder).glob("*" + LOCK_PROBE_SUFFIX):
            original = probe.with_name(probe.name.rsplit(".", 2)[0])
            try:
                if original.exists():
                    probe.unlink()
                else:
                    os.rename(probe, original)
            except OSError:
                pass   # still in use: left for a later run


def locked_files(paths: list[Path]) -> list[Path]:
    """Files another program holds open. On Windows a file open in ArcGIS Pro or
    Excel cannot be renamed, so each file is renamed to a scratch name and back."""
    locked = []
    for p in paths:
        probe = p.with_name(f"{p.name}.{os.getpid()}{LOCK_PROBE_SUFFIX}")
        try:
            os.rename(p, probe)
        except FileNotFoundError:
            continue        # gone since it was listed
        except PermissionError:
            locked.append(p)
            continue
        try:
            os.rename(probe, p)
        except OSError as exc:
            raise RuntimeError(f"Could not put {p.name} back after checking it (it is now {probe.name}); "
                               "rename it back by hand.") from exc
    return locked


def check_outputs_unlocked(out_dir: Path, geoextent: str) -> None:
    """Stop before anything is written if an earlier run's outputs are open in
    another program, so a run never leaves a mix of new and old files."""
    out_dir = Path(out_dir)
    restore_lock_probes([out_dir, out_dir / shp_dir_name(geoextent), out_dir / fire_spatial.density_dir_name(geoextent),
                         out_dir / fire_burned.burned_dir_name(geoextent)])
    locked = locked_files(run_output_files(out_dir, geoextent))
    if locked:
        shown = ", ".join(p.name for p in locked[:8]) + (f" and {len(locked) - 8} more" if len(locked) > 8 else "")
        raise RuntimeError(
            f"These outputs from an earlier run are open in another program: {shown}. Close them (in ArcGIS Pro, "
            "remove the layers from your map; in Excel, close the file) and run again. Nothing has been changed.")


def _layer_gdf(table: pd.DataFrame):
    """One output layer as a GeoDataFrame and its geometry type. Plain tables with
    longitude/latitude become points; date columns become real dates so ArcGIS
    Pro's time slider can use them."""
    import geopandas as gpd

    t = table.copy()
    for col in ("date", "period_start", "period_end"):
        if col in t.columns:
            t[col] = pd.to_datetime(t[col], errors="coerce")
    if isinstance(t, gpd.GeoDataFrame):
        gdf = t
    else:
        gdf = gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t["longitude"], t["latitude"]), crs="EPSG:4326")
    geom_type = "Point" if not isinstance(table, gpd.GeoDataFrame) else (
        "MultiPolygon" if (gdf.geom_type == "MultiPolygon").any() else "Polygon")
    return gdf, geom_type


# ---------------------------------------------------------------------------
# Shapefiles (--vector-format shp or both)
# ---------------------------------------------------------------------------

VECTOR_CHOICES = ("gpkg", "shp", "both")

# Shapefile field names can be at most 10 characters. Fixed short names for the
# tool's own longer fields, so they read sensibly and stay the same between runs;
# anything else is shortened automatically. Every run writes the full list to
# {geoextent}_fire_activity_shp_fields.csv in the shapefile folder.
SHP_FIELD_NAMES = {
    "brightness_k": "bright_k", "acq_time_utc": "acq_time", "compares_with": "compare_to",
    "in_comparison": "in_compare", "area_in_aoi_km2": "aoi_km2", "days_with_fire": "fire_days",
    "prior_fire_count": "prior_fire", "compared_fire_count": "cmp_fire", "burned_area_km2": "burn_km2",
    "burned_pixels": "burn_px", "burned_share_pct": "burn_shr", "prior_burned_area_km2": "pr_burn_km",
    "compared_burned_area_km2": "cmp_burn", "burned_change_km2": "burn_chkm2",
    "burned_change_pct": "burn_chpct", "period_start": "per_start", "prior_period": "pr_period",
    "comparison_note": "cmp_note", "burned_qa_flag": "burn_flag", "burned_note": "burn_note",
    "share_in_aoi_pct": "aoi_shr", "fires_per_1000_km2": "fire_dens",
    "prior_fires_per_1000_km2": "pr_fire_dn", "burned_mapped_pct": "burn_map",
    "prior_burned_mapped_pct": "pr_burn_mp", "high_confidence_count": "high_conf",
    "days_with_imagery": "img_days", "prior_days_with_imagery": "pr_img_day",
}


def shapefile_field_names(columns) -> dict:
    """Map each column to a unique shapefile field name of at most 10 characters."""
    out, used = {}, set()
    for col in columns:
        if col == "geometry":
            continue
        name = col if len(col) <= 10 else SHP_FIELD_NAMES.get(col, col[:10])
        base, n = name, 1
        while name.lower() in used:          # shapefile field names ignore case
            n += 1
            name = f"{base[:10 - len(str(n)) - 1]}_{n}"
        used.add(name.lower())
        out[col] = name
    return out


def shp_dir_name(geoextent: str) -> str:
    return output_filename(geoextent, "_shp")


def shp_fields_csv_name(geoextent: str) -> str:
    return output_filename(geoextent, "_shp_fields.csv")


def shapefile_files(out_dir: Path, geoextent: str) -> list[Path]:
    """This tool's shapefile files (and their field list) in the output folder."""
    folder = Path(out_dir) / shp_dir_name(geoextent)
    if not folder.exists():
        return []
    prefix = f"{sanitize_token(geoextent)}_evnt_fir_"
    return [f for f in folder.iterdir()
            if (f.name.startswith(prefix) and f.suffix.lower() in
                (".shp", ".shx", ".dbf", ".prj", ".cpg", ".xml", ".sbn", ".sbx", ".qix"))
            or f.name == shp_fields_csv_name(geoextent)]


def clear_shapefiles(out_dir: Path, geoextent: str) -> int:
    """Remove shapefiles left by an earlier run. Returns how many files were removed."""
    n = 0
    for old in shapefile_files(out_dir, geoextent):
        try:
            old.unlink()
            n += 1
        except PermissionError as exc:
            raise RuntimeError(f"Could not replace {old.name}; it is probably open in ArcGIS Pro. Remove the "
                               "shapefiles from your map and run again.") from exc
    return n


def _write_shapefile(gdf, path: Path, geom_type: str) -> None:
    """Write one shapefile. Shapefiles cannot hold date-and-time fields (the
    GeoPackage writer's choice), so dates are written as plain dates, which keeps
    them usable by ArcGIS Pro's time slider. UTF-8 with a .cpg file, so accented
    names show correctly. Field arrays are built the way pyogrio's own writer
    builds them, with a mask for blank values."""
    import numpy as np
    import pyogrio.raw
    import shapely

    fields = [c for c in gdf.columns if c != gdf.geometry.name]
    data, mask = [], []
    for c in fields:
        col = gdf[c]
        if pd.api.types.is_datetime64_any_dtype(col):
            data.append(col.to_numpy().astype("datetime64[D]"))
            mask.append(None)
        elif isinstance(col.array, pd.api.extensions.ExtensionArray) and not isinstance(
                col.array, pd.arrays.NumpyExtensionArray if hasattr(pd.arrays, "NumpyExtensionArray")
                else pd.arrays.PandasArray):
            na = col.isna().to_numpy()
            if pd.api.types.is_integer_dtype(col) or pd.api.types.is_bool_dtype(col):
                data.append(col.fillna(0).to_numpy(dtype="int64" if pd.api.types.is_integer_dtype(col) else bool))
            elif pd.api.types.is_float_dtype(col):
                data.append(col.to_numpy(dtype="float64", na_value=np.nan))
            else:
                data.append(col.astype(object).where(~col.isna(), None).to_numpy(dtype=object))
            mask.append(na)
        else:
            data.append(col.to_numpy())
            mask.append(None)
    pyogrio.raw.write(str(path), geometry=shapely.to_wkb(gdf.geometry.values), field_data=data, fields=fields,
                      field_mask=mask, geometry_type=geom_type, crs=gdf.crs.to_wkt() if gdf.crs else None,
                      driver="ESRI Shapefile", encoding="UTF-8")


def write_shapefiles(out_dir: Path, geoextent: str, layers: list[tuple[str, pd.DataFrame]]) -> list[Path]:
    """One shapefile per layer in {geoextent}_fire_activity_shp, with field names
    shortened to 10 characters and the full names listed in a CSV beside them.
    Returns the .shp paths."""
    folder = Path(out_dir) / shp_dir_name(geoextent)
    folder.mkdir(parents=True, exist_ok=True)
    clear_shapefiles(out_dir, geoextent)
    written, key_rows = [], {}
    for name, table in layers:
        gdf, geom_type = _layer_gdf(table)
        names = shapefile_field_names(gdf.columns)
        for full, short in names.items():
            key_rows.setdefault((short, full), []).append(name)
        gdf = gdf.rename(columns=names)
        path = folder / f"{name}.shp"
        _write_shapefile(gdf, path, geom_type)
        written.append(path)
    key = pd.DataFrame([{"shapefile_field": s, "full_name": f, "layers": "; ".join(ls)}
                        for (s, f), ls in key_rows.items()])
    write_csv_atomic(key, folder / shp_fields_csv_name(geoextent))
    return written


def write_gpkg(path: Path, layers: list[tuple[str, pd.DataFrame]]) -> None:
    """Write every layer to one GeoPackage (see _layer_gdf)."""
    tmp = path.with_name(path.stem + "_writing.gpkg")
    if tmp.exists():
        try:
            tmp.unlink()
        except PermissionError as exc:
            raise RuntimeError(f"Could not remove {tmp.name} left from an earlier run; it is probably open in "
                               "ArcGIS Pro. Remove it from your map and run again.") from exc
    for name, table in layers:
        gdf, geom_type = _layer_gdf(table)
        gdf.to_file(tmp, layer=name, driver="GPKG", geometry_type=geom_type if gdf.empty else None)
    try:
        os.replace(tmp, path)
    except PermissionError as exc:
        raise RuntimeError(
            f"Could not replace {path.name}; it is probably open in ArcGIS Pro. Remove it from your map "
            f"(or close the project) and run again. The new results are in {tmp.name} meanwhile."
        ) from exc


# ---------------------------------------------------------------------------
# Area of interest
# ---------------------------------------------------------------------------

@dataclass
class AOI:
    geometry: object          # shapely geometry, EPSG:4326
    geojson: dict
    feature_count: int
    source_crs: str
    area_km2: float
    simplified_deg: Optional[float]
    note: str

    @property
    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self.geojson, sort_keys=True).encode()).hexdigest()[:12]


def load_aoi(path: str, simplify_tolerance: Optional[float] = None) -> AOI:
    import geopandas as gpd
    from shapely.geometry import mapping

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"The area-of-interest file {path} has no features.")
    note = ""
    if gdf.crs is None:
        note = "File has no coordinate system set; assumed WGS84 longitude/latitude. "
        gdf = gdf.set_crs(epsg=4326)
    src_crs = gdf.crs.to_string()
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    # Repair each feature before merging: a self-intersecting feature would
    # otherwise break the merge, and buffer(0) can silently drop part of it.
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = [fire_spatial.polygon_parts(g) for g in gdf.geometry]
    geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
    if geom is None or geom.is_empty:
        raise ValueError(f"The area-of-interest file {path} has no usable polygon geometry.")
    if not geom.is_valid:
        geom = fire_spatial.polygon_parts(geom)
    area_km2 = float(gpd.GeoSeries([geom], crs=4326).to_crs(epsg=6933).area.iloc[0] / 1e6)

    used_tol = None
    if simplify_tolerance:
        geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        used_tol = simplify_tolerance
    gj = mapping(geom)
    if len(json.dumps(gj)) > MAX_AOI_GEOJSON_BYTES:
        for tol in AUTO_SIMPLIFY_STEPS_DEG:
            if used_tol and tol <= used_tol:
                continue
            cand = geom.simplify(tol, preserve_topology=True)
            gj = mapping(cand)
            used_tol = tol
            if len(json.dumps(gj)) <= MAX_AOI_GEOJSON_BYTES:
                geom = cand
                note += (f"Boundary simplified automatically to {tol} degrees (about {round(tol * 111000)} m) "
                         "to keep the Earth Engine request small. ")
                break
        else:
            raise ValueError("The boundary is too detailed to send to Earth Engine even after simplifying; "
                             "pass a larger --simplify-tolerance.")
    return AOI(geom, json.loads(json.dumps(gj)), len(gdf), src_crs, area_km2, used_tol, note.strip())


# ---------------------------------------------------------------------------
# Earth Engine glue - the only part that touches `ee`
# ---------------------------------------------------------------------------

_SPLITTABLE_MESSAGES = ("memory limit", "computation timed out", "too large", "too many pixels",
                        "capacity exceeded")
_TRANSIENT_MESSAGES = ("too many concurrent", "rate limit", "quota exceeded", "429", "503",
                       "service unavailable", "deadline exceeded", "internal error")


def init_ee(project: Optional[str]) -> None:
    import ee
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not start Earth Engine ({exc}). Check you are in the earth-engine-gis environment and "
            "have run 'earthengine authenticate' once. If it says no project was found, save yours once with "
            "'earthengine set_project <your-project-id>', or give it with --ee-project (or the window's "
            "Earth Engine project box)."
        ) from exc


def _ms_to_date(ms) -> date:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()


class EarthEngineFetcher:
    """Pulls fire pixel detections for one sensor and area of interest."""

    def __init__(self, sensor: SensorConfig, aoi_geojson: dict, min_confidence: int):
        import ee
        self.ee = ee
        self.sensor = sensor
        self.min_confidence = min_confidence
        self.region = ee.Geometry(aoi_geojson, None, False)  # planar edges, as in the source file
        self.collection = ee.ImageCollection(sensor.collection_id)

    def latest_data_date(self) -> Optional[date]:
        ms = self.collection.aggregate_max("system:time_start").getInfo()
        return _ms_to_date(ms) if ms is not None else None

    def imagery_dates(self, start: date, end: date) -> set[str]:
        ms_list = (self.collection.filterDate(start.isoformat(), end.isoformat())
                   .aggregate_array("system:time_start").getInfo()) or []
        return {_ms_to_date(ms).isoformat() for ms in ms_list}

    def _detections_fc(self, start: date, end: date):
        ee = self.ee
        band_names = [b for b, _ in self.sensor.bands]
        region = self.region
        min_conf = self.min_confidence

        fire_band = self.sensor.fire_band

        def per_image(img):
            img = ee.Image(img)
            conf = img.select("confidence")
            is_fire = img.select(fire_band).gt(0).And(conf.gte(min_conf))
            stacked = img.select(band_names).addBands(ee.Image.pixelLonLat()).updateMask(is_fire)
            samples = stacked.sample(
                region=region,
                projection=conf.projection(),   # native pixel grid, so one sample per fire pixel
                dropNulls=True,
                tileScale=4,
                geometries=False,
            )
            day = img.date().format("YYYY-MM-dd")
            return samples.map(lambda f: f.set("date", day))

        col = self.collection.filterDate(start.isoformat(), end.isoformat())
        return ee.FeatureCollection(col.map(per_image)).flatten()

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        ee = self.ee
        try:
            raw = ee.data.computeFeatures({
                "expression": self._detections_fc(start, end),
                "fileFormat": "PANDAS_DATAFRAME",
            })
        except ee.EEException as exc:
            msg = str(exc).lower()
            if any(m in msg for m in _SPLITTABLE_MESSAGES):
                raise SplittableFetchError(str(exc)) from exc
            if any(m in msg for m in _TRANSIENT_MESSAGES):
                raise TransientFetchError(str(exc)) from exc
            raise RuntimeError(f"Earth Engine error for {start} to {end}: {exc}") from exc
        except (ConnectionError, TimeoutError, OSError) as exc:
            raise TransientFetchError(f"network problem: {exc}") from exc
        return normalise_detections(raw, self.sensor)


def run_diagnose(project: Optional[str], aoi: Optional[AOI]) -> int:
    """Print what the Earth Engine datasets look like right now, so assumptions
    in this script (band names, native pixel size, latest date) can be checked."""
    import ee
    init_ee(project)
    print("Earth Engine diagnostics\n")
    for sensor in SENSORS.values():
        col = ee.ImageCollection(sensor.collection_id)
        print(f"{sensor.label}  [{sensor.collection_id}]")
        try:
            latest_ms = col.aggregate_max("system:time_start").getInfo()
            latest = _ms_to_date(latest_ms)
            print(f"  latest image date: {fmt_date(latest)}")
            recent = col.filterDate((latest - timedelta(days=30)).isoformat(), (latest + timedelta(days=1)).isoformat())
            n_img = recent.size().getInfo()
            n_days = len({_ms_to_date(m) for m in recent.aggregate_array("system:time_start").getInfo()})
            print(f"  images in the last 30 days: {n_img} across {n_days} distinct days")
            img = ee.Image(recent.sort("system:time_start", False).first())
            print(f"  bands: {img.bandNames().getInfo()}")
            proj = img.select("confidence").projection()
            print(f"  confidence band CRS: {proj.crs().getInfo()}, nominal scale {proj.nominalScale().getInfo():.1f} m")
            if aoi is not None:
                region = ee.Geometry(aoi.geojson, None, False)
                n_unmasked = img.select("confidence").reduceRegion(
                    ee.Reducer.count(), region, crs=proj, maxPixels=1e10, tileScale=4).getInfo().get("confidence")
                n_fire = img.select(sensor.fire_band).gt(0).selfMask().reduceRegion(
                    ee.Reducer.count(), region, crs=proj, maxPixels=1e10, tileScale=4).getInfo().get(sensor.fire_band)
                n_pixels_area = aoi.area_km2 * 1e6 / (sensor.resolution_m ** 2)
                print(f"  latest image, your area (about {n_pixels_area:,.0f} pixels): {n_unmasked} unmasked, "
                      f"{n_fire} with {sensor.fire_band} above zero (should be a handful of fire pixels)")
                fetcher = EarthEngineFetcher(sensor, aoi.geojson, sensor.default_min_confidence)
                df = fetcher.fetch(latest - timedelta(days=6), latest + timedelta(days=1))
                print(f"  detections in your area over the last 7 days of data: {len(df)}")
                if len(df):
                    print(f"  example detection: {df.iloc[0].to_dict()}")
                    if sensor.key == "viirs":
                        print(f"  acq_time range: {df['acq_time_s'].min()} to {df['acq_time_s'].max()} "
                              "(seconds since midnight should be 0 to 86399); "
                              f"DayNight values: {sorted(df['day_night_code'].dropna().unique().tolist())}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
        print()
    try:
        for line in fire_burned.diagnose_lines(aoi.geojson if aoi is not None else None):
            print(line)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR (burned area): {exc}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Cached download loop
# ---------------------------------------------------------------------------

def chunk_is_cacheable(start: date, end: date, latest_data_date: Optional[date], imagery_dates: set[str]) -> bool:
    """A chunk is safe to cache only if it is well behind the latest data (so
    near-real-time revisions have settled) and every day in it has imagery (so
    later backfills are not frozen out). Gaps still there GAP_SETTLE_DAYS after
    the chunk ends are treated as permanent, and the chunk is cached anyway, so
    an old outage does not mean downloading that month on every run."""
    if latest_data_date is None:
        return False
    last = end - timedelta(days=1)
    if last > latest_data_date - timedelta(days=CACHE_SETTLE_DAYS):
        return False
    if last <= latest_data_date - timedelta(days=GAP_SETTLE_DAYS):
        return True
    d = start
    while d < end:
        if d.isoformat() not in imagery_dates:
            return False
        d += timedelta(days=1)
    return True


def collect_detections(fetch: Callable[[date, date], pd.DataFrame], chunks: list[tuple[date, date]],
                       cache_dir: Optional[Path], latest_data_date: Optional[date],
                       sensor: SensorConfig, log: Callable[[str], None] = print,
                       imagery_dates: Optional[set[str]] = None) -> pd.DataFrame:
    """Fetch every chunk (from cache where possible), then dedupe. See
    chunk_is_cacheable() for which chunks are cached."""
    imagery_dates = imagery_dates or set()
    frames = []
    for i, (s, e) in enumerate(chunks, 1):
        cache_file = cache_dir / f"{s.isoformat()}_{e.isoformat()}.csv" if cache_dir else None
        if cache_file is not None and cache_file.exists():
            df = pd.read_csv(cache_file, dtype={"date": str})
            log(f"  [{i}/{len(chunks)}] {s} to {e - timedelta(days=1)}: {len(df)} detections (cached)")
        else:
            if latest_data_date is not None and s > latest_data_date:
                log(f"  [{i}/{len(chunks)}] {s} to {e - timedelta(days=1)}: after latest data, skipped")
                continue
            df = fetch_with_split(fetch, s, e, log)
            log(f"  [{i}/{len(chunks)}] {s} to {e - timedelta(days=1)}: {len(df)} detections")
            if cache_file is not None and chunk_is_cacheable(s, e, latest_data_date, imagery_dates):
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_suffix(".tmp")
                df.to_csv(tmp, index=False)
                os.replace(tmp, cache_file)
        frames.append(df)
    frames = [f for f in frames if not f.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else empty_detections()
    combined = normalise_detections(combined, sensor) if not combined.empty else combined
    return dedupe_detections(combined)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _finite_float(text: str) -> float:
    """argparse type: an ordinary number (not nan or inf)."""
    import math
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"{text!r} is not an ordinary number")
    return v


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Active fire detections for an area of interest, by period, from Google Earth Engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--aoi-file", help="boundary file (shapefile, GeoPackage, GeoJSON); all features are merged into one area")
    p.add_argument("--geoextent", help="location code used in every output name, e.g. ukr or kherson")
    p.add_argument("--start", help="first date, YYYY-MM-DD")
    p.add_argument("--end", help="end date, YYYY-MM-DD, exclusive (2026-01-01 means up to 31 December 2025)")
    p.add_argument("--freq", choices=VALID_FREQS, help="period length")
    p.add_argument("--compare-prior-year", action="store_true",
                   help="also count the same period one year earlier and add change columns")
    p.add_argument("--sensor", choices=SENSOR_CHOICES, default="auto",
                   help="auto uses VIIRS 375 m when every needed date is on or after 3 September 2023, otherwise MODIS 1 km")
    p.add_argument("--min-confidence", type=int, default=None,
                   help="lowest detection confidence kept. Default: VIIRS 1 (nominal), MODIS 30")
    p.add_argument("--out-dir", help="folder for outputs (created if missing)")
    p.add_argument("--ee-project", help="your Earth Engine Cloud project ID; default: the project saved with "
                                        "'earthengine set_project'")
    p.add_argument("--simplify-tolerance", type=_finite_float, default=None,
                   help="simplify the boundary by this many degrees before sending it to Earth Engine")
    p.add_argument("--no-cache", action="store_true", help="ignore and do not write the download cache")
    p.add_argument("--dry-run", action="store_true",
                   help="show the sensor choice, periods and download chunks without contacting Earth Engine")
    p.add_argument("--vector-format", choices=VECTOR_CHOICES, default="gpkg",
                   help="vector outputs (points, hexagons, admin areas): gpkg = one GeoPackage, shp = shapefiles "
                        "in a folder (field names shortened to 10 characters), both")
    g = p.add_argument_group("spatial summaries")
    g.add_argument("--hex-size-km", type=_finite_float, default=None,
                   help="hexagon size, km across the flats. Default: picked from the area's size "
                        "(about 2 km for a district, 25 km for a large country)")
    g.add_argument("--admin-file", help="admin boundaries to count fires by (any boundary file). Default: the "
                                        "area-of-interest file's own features, if it has more than one")
    g.add_argument("--unit-name-field", help="field holding admin area names, e.g. ADM2_EN (default: found automatically)")
    g.add_argument("--unit-id-field", help="field holding admin area codes, e.g. ADM2_PCODE (default: numbered 1, 2, 3...)")
    g.add_argument("--density", choices=fire_spatial.DENSITY_CHOICES, default="all",
                   help="density rasters: all = whole run and every period, totals = whole run only, none")
    g.add_argument("--density-radius-km", type=_finite_float, default=None,
                   help="kernel density search radius in km. Default: the hexagon size")
    b = p.add_argument_group("burned area (MODIS MCD64A1, 500 m, monthly, about three months behind)")
    b.add_argument("--no-burned-area", action="store_true",
                   help="skip burned area (faster; fire detections only)")
    b.add_argument("--burned-rasters", choices=fire_burned.RASTER_CHOICES, default="all",
                   help="burned area rasters: all = whole run and every period, totals = whole run only, none")
    c = p.add_argument_group("charts")
    c.add_argument("--no-charts", action="store_true", help="skip the PNG charts")
    c.add_argument("--area-name", help="area name used in chart titles, e.g. Scotland. Default: read from the "
                                      "area-of-interest file's name fields, otherwise the geoextent code")
    p.add_argument("--diagnose", action="store_true",
                   help="print what the Earth Engine fire datasets look like now and exit")
    p.add_argument("--wizard", action="store_true",
                   help="answer questions instead of typing options (also starts when no options are given)")
    return p


def _require(args, names: list[str], parser: argparse.ArgumentParser) -> None:
    missing = [f"--{n.replace('_', '-')}" for n in names if not getattr(args, n)]
    if missing:
        parser.error("missing required argument(s): " + ", ".join(missing))


def load_admin_for_run(args, aoi: AOI, log: Callable[[str], None]):
    """Admin units for the per-area counts: --admin-file if given, otherwise the
    area-of-interest file's own features when it has more than one, otherwise
    none."""
    admin_file = getattr(args, "admin_file", None)
    name_field = getattr(args, "unit_name_field", None)
    id_field = getattr(args, "unit_id_field", None)
    if admin_file:
        admin = fire_spatial.load_admin_units(admin_file, aoi.geometry, name_field, id_field, "the --admin-file")
    elif aoi.feature_count > 1:
        admin = fire_spatial.load_admin_units(args.aoi_file, aoi.geometry, name_field, id_field,
                                              "the area-of-interest file", allow_unnamed=True)
    else:
        if name_field or id_field:
            raise ValueError("--unit-name-field and --unit-id-field need --admin-file, or an area-of-interest "
                             "file with more than one feature.")
        log("Admin areas: none (area of interest is a single feature and no --admin-file given)")
        return None
    log(f"Admin areas: {len(admin.gdf)} from {admin.source}")
    return admin


def run(args, fetcher_factory=None, log: Callable[[str], None] = print, burned_fetcher_factory=None) -> int:
    """Main pipeline. `fetcher_factory(sensor, aoi_geojson, min_conf)` returns an
    object with latest_data_date(), imagery_dates(start, end) and fetch(start, end);
    tests pass a fake one, normal runs use EarthEngineFetcher.
    `burned_fetcher_factory(aoi_geojson)` likewise returns an object with
    latest_image_date(), fetch(start, end) and coverage(months); normal runs use
    fire_burned.BurnedAreaFetcher."""
    burned_on = not getattr(args, "no_burned_area", False)
    burned_raster_mode = getattr(args, "burned_rasters", "all") or "all"
    if burned_raster_mode not in fire_burned.RASTER_CHOICES:
        raise ValueError(f"--burned-rasters must be one of {fire_burned.RASTER_CHOICES}")
    periods = build_periods(args.start, args.end, args.freq)
    prior_map = build_prior_map(periods, args.freq) if args.compare_prior_year else None
    sensor_key, reason = choose_sensor(args.sensor, periods, prior_map)
    sensor = SENSORS[sensor_key]
    min_conf = args.min_confidence if args.min_confidence is not None else sensor.default_min_confidence
    windows = fetch_windows(periods, prior_map)
    chunks = split_into_month_chunks(windows)

    log(f"Sensor: {sensor.label}")
    log(f"  {reason}")
    log(f"Minimum confidence: {min_conf} ({sensor.confidence_note})")
    log(f"Periods: {len(periods)} {args.freq}, {periods[0].label} to {periods[-1].label}"
        + (" plus year-earlier comparisons" if prior_map else ""))
    log(f"Download: {len(chunks)} monthly chunk(s) covering "
        + ", ".join(f"{s:%d/%m/%Y} to {(e - timedelta(days=1)):%d/%m/%Y}" for s, e in windows))

    aoi = load_aoi(args.aoi_file, args.simplify_tolerance)
    log(f"Area of interest: {aoi.feature_count} feature(s), {aoi.area_km2:,.0f} km2, source CRS {aoi.source_crs}")
    if aoi.note:
        log(f"  {aoi.note}")

    admin = load_admin_for_run(args, aoi, log)
    # Hexagon size and density radius are settled and checked here, before any
    # download, and the same size is passed on to the spatial summaries.
    ea_crs = fire_spatial.equal_area_crs(aoi.geometry)
    aoi_ea = fire_spatial.to_equal_area(aoi.geometry, ea_crs)
    hex_auto = getattr(args, "hex_size_km", None) is None
    hex_size = fire_spatial.auto_hex_size_km(aoi_ea.area / 1e6) if hex_auto else float(args.hex_size_km)
    fire_spatial.check_hex_size(aoi_ea, hex_size)
    radius = getattr(args, "density_radius_km", None)
    if radius is not None and radius <= 0:
        raise ValueError("--density-radius-km must be greater than zero")
    log(f"Hexagons: {hex_size:g} km across" + (" (automatic; set --hex-size-km to change)" if hex_auto else ""))
    burned_months = fire_burned.months_needed(windows) if burned_on else []
    if burned_on:
        log(f"Burned area: {fire_burned.LABEL}; up to {len(burned_months)} monthly image(s) to read"
            + ("; weekly totals use approximate burn dates" if args.freq == "weekly" else "")
            + ". Months not yet published are reported as not yet available.")
    else:
        log("Burned area: off (--no-burned-area)")
    charts_on = not getattr(args, "no_charts", False)
    log("Charts: " + ("detections" + (", burned area" if burned_on else "") + (", admin areas" if admin else "")
                      + " (PNG)" if charts_on else "off (--no-charts)"))
    log("Vector outputs: " + {"gpkg": "GeoPackage", "shp": "shapefiles (field names shortened to 10 characters)",
                              "both": "GeoPackage and shapefiles"}[getattr(args, "vector_format", None) or "gpkg"])

    if args.dry_run:
        for p in periods:
            pr = prior_map.get(p.label) if prior_map else None
            log(f"  {p.label}: {p.start:%d/%m/%Y} to {p.last_day:%d/%m/%Y}"
                + (f"  vs {pr.label}" if pr else ("  vs (no matching week)" if prior_map else "")))
        log("Dry run only, nothing downloaded.")
        return 0

    # Before any download: stop now if earlier outputs are open in ArcGIS Pro or Excel.
    check_outputs_unlocked(Path(args.out_dir), args.geoextent)

    ee_ready = False
    if fetcher_factory is None:
        init_ee(args.ee_project)
        ee_ready = True
        fetcher_factory = EarthEngineFetcher
    if burned_on and burned_fetcher_factory is None:
        if not ee_ready:
            init_ee(args.ee_project)

        def burned_fetcher_factory(geojson):
            return fire_burned.BurnedAreaFetcher(geojson, SplittableFetchError, TransientFetchError)
    fetcher = fetcher_factory(sensor, aoi.geojson, min_conf)

    latest = fetcher.latest_data_date()
    log(f"Latest {sensor.key.upper()} data in Earth Engine: {fmt_date(latest)}" if latest else "Latest data date unknown")
    imagery: set[str] = set()
    for s, e in windows:
        imagery |= fetcher.imagery_dates(s, e)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else out_dir / ".cache" / f"{CACHE_VERSION}_{sensor.key}_c{min_conf}_{aoi.fingerprint}"
    log("Fetching detections:")
    detections = collect_detections(fetcher.fetch, chunks, cache_dir, latest, sensor, log, imagery)
    if len(detections) > LARGE_RUN_DETECTIONS:
        log(f"Warning: {len(detections):,} detections is a very large run. Expect it to take a while, use several "
            "GB of memory and write a GeoPackage of a GB or more. For a whole continent, consider "
            "--min-confidence 2, --density totals, or splitting the area into countries.")

    summary = build_summary(detections, periods, prior_map, sensor, imagery, latest)
    current_pts, prior_pts = build_point_tables(detections, periods, prior_map, sensor, latest)

    # Burned area (Phase 3)
    burned_cur = burned_pri = None
    burned_latest = None
    burned_total_pixels = 0
    coverage: dict = {}
    if burned_on:
        bfetcher = burned_fetcher_factory(aoi.geojson)
        latest_img = bfetcher.latest_image_date()
        burned_latest = fire_burned.latest_covered_day(latest_img)
        log("Burned area: latest published month "
            + (f"{latest_img:%B %Y} (data up to {fmt_date(burned_latest)})" if burned_latest else "unknown"))
        b_cache = (None if args.no_cache else
                   out_dir / ".cache" / f"{fire_burned.BURNED_CACHE_VERSION}_{aoi.fingerprint}")
        published = None
        if burned_months and burned_latest is not None and hasattr(bfetcher, "published_months"):
            published = bfetcher.published_months(burned_months[0], fire_burned.next_month(burned_months[-1]))
        log("Fetching burned area:")
        burned = fire_burned.collect_burned(bfetcher.fetch, burned_months, b_cache, burned_latest, log,
                                            fetch_with_split, published)
        burned_total_pixels = len(burned)
        coverage = fire_burned.collect_coverage(bfetcher, burned_months, b_cache, burned_latest, log, published)
        missing = frozenset() if published is None else frozenset(
            fire_burned.month_key(m) for m in burned_months
            if m <= burned_latest and fire_burned.month_key(m) not in published)
        fire_burned.add_burned_to_summary(summary, burned, periods, prior_map, burned_latest, coverage, args.freq,
                                          missing)
        burned_cur, burned_pri = fire_burned.burned_point_tables(burned, periods, prior_map, burned_latest,
                                                                 args.freq, summary)

    csv_path = out_dir / output_filename(args.geoextent, "_summary.csv")
    gpkg_path = out_dir / output_filename(args.geoextent, ".gpkg")
    info_path = out_dir / output_filename(args.geoextent, "_run_info.txt")

    # Checked again just before writing, in case something was opened during the download.
    check_outputs_unlocked(out_dir, args.geoextent)
    write_summary_csv(summary, csv_path, prior_map is not None, burned_on)
    layers = [(points_layer_name(args.geoextent, sensor.key), current_pts)]
    if prior_pts is not None:
        layers.append((points_layer_name(args.geoextent, sensor.key, prior=True), prior_pts))
    log("Building spatial summaries:")
    spatial = fire_spatial.spatial_summaries(
        aoi.geometry, current_pts, prior_pts, periods, summary, sensor, args.geoextent, out_dir,
        hex_size_km=hex_size, hex_size_auto=hex_auto, density=getattr(args, "density", "all") or "all",
        density_radius_km=getattr(args, "density_radius_km", None), admin=admin,
        burned_current=burned_cur, burned_prior=burned_pri, burned_pixel_km2=fire_burned.PIXEL_KM2,
        burned_pixel_m=fire_burned.PIXEL_M, log=log)
    layers += spatial.layers
    burned_rasters, burned_info = [], []
    if not burned_on:
        removed = fire_burned.clear_burned_rasters(out_dir, args.geoextent)
        if removed:
            log(f"Removed {removed} burned area raster file(s) left from an earlier run")
    if burned_on:
        burned_rasters, burned_info = fire_burned.write_burned_rasters(
            aoi.geometry, burned_cur, burned_pri, periods, summary, args.geoextent, out_dir,
            burned_raster_mode, log)
    vector_format = getattr(args, "vector_format", None) or "gpkg"
    shapefiles: list[Path] = []
    if vector_format in ("gpkg", "both"):
        write_gpkg(gpkg_path, layers)
    elif gpkg_path.exists():
        gpkg_path.unlink()          # from an earlier run; the lock check has made sure it is not open
        log(f"Removed {gpkg_path.name} left from an earlier run (shapefiles only this time)")
    if vector_format in ("shp", "both"):
        shapefiles = write_shapefiles(out_dir, args.geoextent, layers)
    else:
        removed = clear_shapefiles(out_dir, args.geoextent)
        if removed:
            log(f"Removed {removed} shapefile file(s) left from an earlier run (GeoPackage only this time)")
    admin_csv = None
    admin_csv_path = out_dir / output_filename(args.geoextent, "_admin_summary.csv")
    if spatial.admin_rows:
        admin_csv = admin_csv_path
        write_csv_atomic(pd.DataFrame(spatial.admin_rows), admin_csv)
    elif admin_csv_path.exists():
        # Left from an earlier run with admin areas; it no longer matches this run.
        try:
            admin_csv_path.unlink()
            log(f"Removed {admin_csv_path.name} left from an earlier run (no admin areas this time)")
        except OSError:
            log(f"Warning: could not remove {admin_csv_path.name} from an earlier run (open in another program?)")

    charts: list[Path] = []
    chart_note = "off (--no-charts)"
    if charts_on:
        area_name = getattr(args, "area_name", None) or fire_charts.area_name_from_file(
            args.aoi_file, args.geoextent.upper())
        label, source = fire_charts.SENSOR_TEXT[sensor.key]
        try:
            charts = fire_charts.write_charts(
                summary, out_dir, args.geoextent, freq=args.freq, compare=prior_map is not None,
                area_name=area_name, sensor_label=label, sensor_source=source, burned_on=burned_on,
                admin_rows=spatial.admin_rows or None, log=log)
            chart_note = f"{len(charts)} PNG file(s), area name '{area_name}'"
        except Exception as exc:  # noqa: BLE001 - a chart problem must not lose a finished run
            chart_note = f"failed ({type(exc).__name__}: {exc}); every other output was written"
            log(f"Warning: charts {chart_note}")
            fire_charts.clear_charts(out_dir, args.geoextent)   # no half set of charts left behind
    else:
        locked: list = []
        removed = fire_charts.clear_charts(out_dir, args.geoextent, locked)
        if removed:
            log(f"Removed {removed} chart file(s) left from an earlier run")
        if locked:
            log("Warning: could not remove old chart(s) " + ", ".join(locked) + " (open in another program?)")

    flagged = [r for r in summary if r["qa_flag"] != "ok"]
    uneven = uneven_comparisons(summary)
    info = [
        "Fire activity tool - run information",
        f"Run at: {datetime.now():%d/%m/%Y %H:%M}",
        f"Area of interest: {Path(args.aoi_file).name} ({aoi.feature_count} feature(s), {aoi.area_km2:,.0f} km2)",
        f"Boundary note: {aoi.note or 'none'}",
        f"Periods: {args.freq}, {periods[0].label} to {periods[-1].label}",
        f"Year-earlier comparison: {'yes' if prior_map else 'no'}",
        f"Sensor: {sensor.label} [{sensor.collection_id}]",
        f"Why this sensor: {reason}",
        f"Minimum confidence kept: {min_conf} ({sensor.confidence_note})",
        f"Latest data date in Earth Engine: {fmt_date(latest)}" if latest else "Latest data date: unknown",
        "Counting rule: one detection = one fire pixel on one day.",
        "Data quality: near-real-time data, described by NASA as not science quality.",
        f"Detections fetched: {len(detections)} (current periods: {len(current_pts)}"
        + (f", year-earlier periods: {len(prior_pts)}"
           + ("; a detection can count in both when the run spans more than a year)"
              if len(current_pts) + len(prior_pts) > len(detections) else ")")
           if prior_pts is not None else ")"),
        f"Periods with a quality flag: {len(flagged)}"
        + ("" if not flagged else " (" + ", ".join(f"{r['period']} {r['qa_flag']}" for r in flagged[:12])
                                   + (" ..." if len(flagged) > 12 else "") + ")"),
        *([f"Comparisons affected by missing imagery in either year: {len(uneven)} ("
           + ", ".join(uneven[:12]) + (" ..." if len(uneven) > 12 else "")
           + "); see comparison_note in the summary CSV. Counts are not scaled up for missing days."]
          if uneven else []),
        *spatial.info,
        *_burned_info_lines(burned_on, burned_latest, burned_total_pixels, summary, args.freq, burned_info),
        f"Charts: {chart_note}",
        "Files:",
        f"  {csv_path.name}",
        *([f"  {admin_csv.name}"] if admin_csv else []),
        *([f"  {gpkg_path.name} - layers: " + ", ".join(n for n, _ in layers)]
          if vector_format in ("gpkg", "both") else []),
        *([f"  {shp_dir_name(args.geoextent)}/ - {len(shapefiles)} shapefile(s), one per layer; shortened field "
           f"names are listed in {shp_fields_csv_name(args.geoextent)}"] if shapefiles else []),
        *([f"  {spatial.rasters[0].parent.name}/ - {len(spatial.rasters)} density raster(s)"] if spatial.rasters else []),
        *([f"  {burned_rasters[0].parent.name}/ - {len(burned_rasters)} burned area raster(s)"]
          if burned_rasters else []),
        *[f"  {c.name}" for c in charts],
    ]
    info_path.write_text("\n".join(info) + "\n", encoding="utf-8")

    log(f"\nDone. {len(current_pts)} detections in the requested periods.")
    log(f"  {csv_path}\n  {info_path}")
    if vector_format in ("gpkg", "both"):
        log(f"  {gpkg_path}")
    if shapefiles:
        log(f"  {shapefiles[0].parent}  ({len(shapefiles)} shapefiles)")
    if admin_csv:
        log(f"  {admin_csv}")
    if spatial.rasters:
        log(f"  {spatial.rasters[0].parent}  ({len(spatial.rasters)} density rasters)")
    if burned_rasters:
        log(f"  {burned_rasters[0].parent}  ({len(burned_rasters)} burned area rasters)")
    for c in charts:
        log(f"  {c}")
    return 0


def _burned_info_lines(on: bool, latest: Optional[date], n_pixels: int, summary: list[dict], freq: str,
                       raster_info: list[str]) -> list[str]:
    if not on:
        return ["Burned area: off (--no-burned-area)"]
    lines = [
        f"Burned area: {fire_burned.LABEL} [{fire_burned.COLLECTION_ID}]",
        "  Latest published month: "
        + (f"{latest:%B %Y} (data up to {fmt_date(latest)}); later periods are 'not_yet_available', not zero"
           if latest else "unknown"),
        f"  Burned area = burned pixels x {fire_burned.PIXEL_KM2:.4f} km2 (the MODIS sinusoidal grid is "
        "equal-area). Pixels are placed in hexagons and admin areas by their centre.",
        f"  Burned pixels fetched: {n_pixels:,} (current and year-earlier periods together)",
        "  Mapped coverage (burned_mapped_pct) is the share of land pixels MODIS could map; below "
        f"{fire_burned.LOW_COVERAGE_PCT:g} per cent the period is flagged low_coverage and burned area may be "
        "under-counted (cloud, missing data). Other flags: partial_period (only part published, compared with the "
        "same dates a year earlier), missing_month (a month absent from the Earth Engine archive; no comparison), "
        "coverage_unknown (no coverage figure). Several flags are joined with '+'.",
    ]
    if freq == "weekly":
        lines.append("  Weekly burned area is derived from each pixel's approximate burn day, so treat weekly "
                     "figures as approximate.")
    flagged = [r for r in summary if r.get("burned_qa_flag", "ok") != "ok"]
    if flagged:
        lines.append("  Periods with a burned area flag: " + ", ".join(
            f"{r['period']} {r['burned_qa_flag']}" for r in flagged[:12]) + (" ..." if len(flagged) > 12 else ""))
    return lines + raster_info


def main(argv: Optional[list[str]] = None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    if not raw or raw == ["--wizard"]:
        import fire_wizard
        return fire_wizard.wizard_main(main)
    parser = build_arg_parser()
    args = parser.parse_args(raw)
    if args.wizard:
        parser.error("--wizard asks for every option itself; run it on its own: python fire_tool.py --wizard")
    try:
        if args.diagnose:
            aoi = load_aoi(args.aoi_file, args.simplify_tolerance) if args.aoi_file else None
            return run_diagnose(args.ee_project, aoi)
        _require(args, ["aoi_file", "start", "end", "freq"] + ([] if args.dry_run else ["geoextent", "out_dir"]),
                 parser)
        return run(args)
    except (ValueError, RuntimeError, OSError, ShapelyError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ != "EEException":
            raise
        print(f"\nEarth Engine error: {exc}\nWait a few minutes and run again; finished months are cached.",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
