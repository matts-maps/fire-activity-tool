"""
fire_burned.py - burned area for the fire activity tool (Phase 3).

Source: MODIS burned area, MCD64A1 collection 6.1 (Earth Engine ID
MODIS/061/MCD64A1). One image per calendar month at 500 m (463 m sinusoidal
pixels), where each burned pixel carries the approximate day of year it burned
(BurnDate). The product runs about three months behind.

How it is used here:

  * Burned pixels inside the area of interest are fetched from Earth Engine as
    points (pixel centres) with their burn date, one month at a time, and cached.
    Everything else runs locally, as in Phase 2.
  * Burned area = number of burned pixels x pixel area. The MODIS sinusoidal grid
    is an equal-area projection, so every pixel covers the same 0.2147 km2.
  * Monthly and annual totals come straight from the data. Weekly totals are
    derived from the burn day of each pixel and flagged as approximate, because
    the burn day can be a few days out (see the Uncertainty band).
  * Periods after the latest published month are "not yet available" (blank, not
    zero). A period that is only partly covered is compared like-for-like with the
    same number of days a year earlier, as the fire counts are.
  * Mapped coverage: MODIS cannot map burns where cloud or missing data got in the
    way (QA bit 1). The share of land pixels that could be mapped is reported per
    period, so a low value warns that burned area may be under-counted.

Rasters show the share of each cell that burned (0 to 1) on the native MODIS
sinusoidal grid. At native resolution the cells are single pixels, so values are
simply 0 (not burned) or 1 (burned). Large areas are written at a coarser cell
made of whole pixels, so values become fractions.

Only BurnedAreaFetcher touches the `ee` module; the rest is tested offline.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

COLLECTION_ID = "MODIS/061/MCD64A1"
ARCHIVE_START = date(2000, 11, 1)
LABEL = "MODIS burned area (MCD64A1 collection 6.1), 500 m, monthly"

# MODIS sinusoidal grid (the product's native grid). Equal-area, so every pixel
# has the same area.
SINU_PROJ4 = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
GRID_X0 = -20015109.354           # left edge of the global grid, metres
GRID_Y0 = 10007554.677            # top edge of the global grid, metres
PIXEL_M = 463.31271652791656      # 1111950.5197 m tile / 2400 pixels
PIXEL_KM2 = PIXEL_M ** 2 / 1e6    # 0.21466 km2

BURNED_CACHE_VERSION = "ba_v1"
# Rasters: cells are whole multiples of the native pixel, enlarged until the
# grid is no wider or taller than this many cells.
MAX_RASTER_CELLS = 3000
NODATA = -9999.0
LOW_COVERAGE_PCT = 90.0
RASTER_CHOICES = ("all", "totals", "none")

BURNED_COLUMNS = ["date", "longitude", "latitude", "burn_date_uncertainty_days"]

SUMMARY_COLUMNS = ["burned_area_km2", "burned_pixels", "burned_mapped_pct", "burned_qa_flag"]
COMPARISON_COLUMNS = ["prior_burned_area_km2", "prior_burned_mapped_pct", "burned_change_km2",
                      "burned_change_pct"]
NOTE_COLUMN = "burned_note"


def fmt_date(d: date) -> str:
    return f"{d.day} {d:%B %Y}"


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def next_month(d: date) -> date:
    return date(d.year + (d.month // 12), d.month % 12 + 1, 1)


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def latest_covered_day(latest_image_date: Optional[date]) -> Optional[date]:
    """Each image covers a whole calendar month, so the latest image (dated the 1st
    of its month) covers data up to the last day of that month."""
    if latest_image_date is None:
        return None
    return next_month(month_start(latest_image_date)) - timedelta(days=1)


# ---------------------------------------------------------------------------
# Burned pixel tables
# ---------------------------------------------------------------------------

def empty_burned() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BURNED_COLUMNS}).astype({"date": "object"})


def normalise_burned(raw: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Earth Engine samples (BurnDate, Uncertainty, longitude, latitude, year) to
    one row per burned pixel with its real burn date."""
    if raw is None or raw.empty:
        return empty_burned()
    df = raw.copy()
    if "date" not in df.columns:
        doy = pd.to_numeric(df["BurnDate"], errors="coerce")
        year = pd.to_numeric(df["year"], errors="coerce")
        ok = doy.ge(1) & doy.le(366) & year.notna()
        df = df[ok].copy()
        df["date"] = [(date(int(y), 1, 1) + timedelta(days=int(d) - 1)).isoformat()
                      for y, d in zip(year[ok], doy[ok])]
    if "Uncertainty" in df.columns:
        df = df.rename(columns={"Uncertainty": "burn_date_uncertainty_days"})
    for col in BURNED_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[BURNED_COLUMNS].copy()
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    for col in BURNED_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["burn_date_uncertainty_days"] < 0, "burn_date_uncertainty_days"] = float("nan")
    return df.reset_index(drop=True)


def pixel_index(lon, lat) -> tuple[np.ndarray, np.ndarray]:
    """Column and row of the native MODIS 463 m pixel holding each point."""
    from pyproj import CRS, Transformer
    t = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(SINU_PROJ4), always_xy=True)
    x, y = t.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    col = np.floor((np.asarray(x) - GRID_X0) / PIXEL_M).astype("int64")
    row = np.floor((GRID_Y0 - np.asarray(y)) / PIXEL_M).astype("int64")
    return col, row


def add_pixel_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["pix_col"] = pd.Series(dtype="int64")
        out["pix_row"] = pd.Series(dtype="int64")
        return out
    out["pix_col"], out["pix_row"] = pixel_index(out["longitude"], out["latitude"])
    return out


def dedupe_burned(df: pd.DataFrame) -> pd.DataFrame:
    """One row per pixel per burn date (protects against a pixel being returned
    twice, e.g. by overlapping requests)."""
    if df.empty:
        return add_pixel_keys(df)
    work = add_pixel_keys(df)
    work = work.drop_duplicates(subset=["date", "pix_col", "pix_row"], keep="first")
    return work.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)


def burned_in(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["date"] >= start.isoformat()) & (df["date"] < end.isoformat())]


def unique_pixels(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df[["pix_col", "pix_row"]].drop_duplicates().shape[0])


def area_km2(n_pixels: int) -> float:
    return round(n_pixels * PIXEL_KM2, 2)


# ---------------------------------------------------------------------------
# Availability and coverage per period
# ---------------------------------------------------------------------------

def months_with_data(start: date, end: date, latest_day: Optional[date]) -> list[str]:
    """Month keys of the published days of [start, end)."""
    if latest_day is None:
        return []
    stop = min(end, latest_day + timedelta(days=1))
    out, d = [], month_start(start)
    while d < stop:
        out.append(month_key(d))
        d = next_month(d)
    return out


def burned_qa_flag(start: date, end: date, latest_day: Optional[date], mapped_pct,
                   missing: frozenset = frozenset()) -> str:
    """not_yet_available, or any of partial_period / missing_month /
    low_coverage / coverage_unknown joined with '+', or ok."""
    last = end - timedelta(days=1)
    if latest_day is None or start > latest_day:
        return "not_yet_available"
    flags = []
    if last > latest_day:
        flags.append("partial_period")
    if any(m in missing for m in months_with_data(start, end, latest_day)):
        flags.append("missing_month")
    if _blank(mapped_pct):
        flags.append("coverage_unknown")
    elif mapped_pct < LOW_COVERAGE_PCT:
        flags.append("low_coverage")
    return "+".join(flags) or "ok"


def is_partial(flag: str) -> bool:
    return "partial_period" in flag


def prior_cutoff(period, prior, latest_day: date, freq: str) -> date:
    """Exclusive end of the year-earlier window matching a partly published
    period. Burned area is published in calendar months, so annual and monthly
    periods use the same calendar date a year earlier (28 February for 29
    February); weekly periods use the same number of days into the ISO week."""
    nxt = latest_day + timedelta(days=1)
    if freq == "weekly":
        return prior.start + timedelta(days=(nxt - period.start).days)
    try:
        cut = nxt.replace(year=nxt.year - (period.start.year - prior.start.year))
    except ValueError:                       # 29 February -> 1 March
        cut = date(nxt.year - (period.start.year - prior.start.year), 3, 1)
    return min(max(cut, prior.start), prior.end)


def mapped_pct_for(start: date, end: date, coverage: dict, latest_day: Optional[date] = None):
    """Share of land pixels that could be mapped, over the days of [start, end)
    that have data, weighting each month by how many of its days fall in the
    window. Blank when no month in the window has coverage figures."""
    stop = end if latest_day is None else min(end, latest_day + timedelta(days=1))
    land = valid = 0.0
    d = start
    seen = False
    while d < stop:
        m_end = min(next_month(d), stop)
        days = (m_end - d).days
        month_days = (next_month(d) - month_start(d)).days
        cov = coverage.get(month_key(d))
        if cov is not None:
            seen = True
            w = days / month_days
            land += w * cov[0]
            valid += w * cov[1]
        d = m_end
    if not seen or land <= 0:
        return ""
    return round(100.0 * valid / land, 1)


def _blank(v) -> bool:
    return v == "" or v is None or (isinstance(v, float) and math.isnan(v))


def add_burned_to_summary(rows: list[dict], burned: pd.DataFrame, periods, prior_map: Optional[dict],
                          latest_day: Optional[date], coverage: dict, freq: str,
                          missing: frozenset = frozenset()) -> None:
    """Add burned area columns to each summary row (in place). `missing` holds
    month keys ('YYYY-MM') that are published but have no image in the archive:
    a period made up only of such months is blank, and any period touching one
    gets the missing_month flag and no year-earlier comparison."""
    missing = frozenset(missing)
    weekly_note = "weekly totals use approximate burn dates" if freq == "weekly" else ""
    for row, p in zip(rows, periods):
        cov = mapped_pct_for(p.start, p.end, coverage, latest_day)
        flag = burned_qa_flag(p.start, p.end, latest_day, cov, missing)
        data_months = months_with_data(p.start, p.end, latest_day)
        all_missing = bool(data_months) and all(m in missing for m in data_months)
        avail = flag != "not_yet_available" and not all_missing
        n_cur = unique_pixels(burned_in(burned, p.start, p.end)) if avail else None
        row["burned_area_km2"] = area_km2(n_cur) if avail else ""
        row["burned_pixels"] = n_cur if avail else ""
        row["burned_mapped_pct"] = cov if avail else ""
        row["burned_qa_flag"] = flag
        notes = [weekly_note] if weekly_note and avail else []
        if "missing_month" in flag:
            notes.append("month(s) missing from the Earth Engine archive: "
                         + ", ".join(m for m in data_months if m in missing))
        if prior_map is None:
            if flag == "not_yet_available":
                notes.append("burned area not yet published for this period")
            row[NOTE_COLUMN] = "; ".join(notes)
            continue
        prior = prior_map.get(p.label)
        values = {"prior_burned_area_km2": "", "prior_burned_mapped_pct": "", "burned_change_km2": "",
                  "burned_change_pct": ""}
        if prior is None:
            notes.append("no matching ISO week 53 in the previous year")
        elif flag == "not_yet_available":
            notes.append("burned area not yet published for this period")
        elif "missing_month" in flag:
            notes.append("no year-earlier comparison")
        else:
            cmp_end = prior.end
            if is_partial(flag):
                cmp_end = prior_cutoff(p, prior, latest_day, freq)
                notes.append(f"burned area: {fmt_date(p.start)} to {fmt_date(latest_day)} compared with "
                             f"{fmt_date(prior.start)} to {fmt_date(cmp_end - timedelta(days=1))} only")
            prior_missing = [m for m in months_with_data(prior.start, cmp_end, latest_day) if m in missing]
            if prior_missing:
                notes.append("no year-earlier comparison: month(s) missing from the Earth Engine archive in the "
                             "year before: " + ", ".join(prior_missing))
                row.update(values)
                row[NOTE_COLUMN] = "; ".join(n for n in notes if n)
                continue
            n_pri = unique_pixels(burned_in(burned, prior.start, cmp_end))
            prior_cov = mapped_pct_for(prior.start, cmp_end, coverage, latest_day)
            if not _blank(prior_cov) and prior_cov < LOW_COVERAGE_PCT:
                notes.append(f"year-earlier mapped coverage only {prior_cov:g} per cent, so its burned area may be "
                             "under-counted")
            values.update({
                "prior_burned_area_km2": area_km2(n_pri),
                "prior_burned_mapped_pct": prior_cov,
                "burned_change_km2": area_km2(n_cur - n_pri),
                "burned_change_pct": round(100.0 * (n_cur - n_pri) / n_pri, 1) if n_pri > 0 else "",
            })
        row.update(values)
        row[NOTE_COLUMN] = "; ".join(n for n in notes if n)


def compared_burned_labels(rows) -> set:
    return {r["period"] for r in rows if not _blank(r.get("prior_burned_area_km2", ""))}


# ---------------------------------------------------------------------------
# Per-period pixel tables (for hexagons, admin areas and rasters)
# ---------------------------------------------------------------------------

def burned_point_tables(burned: pd.DataFrame, periods, prior_map: Optional[dict],
                        latest_day: Optional[date], freq: str,
                        summary_rows: list[dict]) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Burned pixels labelled with their period (one row per pixel per period),
    and, when comparing, the year-earlier pixels with compares_with and
    in_comparison (like-for-like days for a partly covered period). Follows the
    summary rows from add_burned_to_summary: periods with a blank burned area
    are left out, and year-earlier pixels count only where the summary has a
    year-earlier figure."""
    by_label = {r["period"]: r for r in summary_rows}
    compared = compared_burned_labels(summary_rows)
    def per_period(sub, label):
        sub = sub.drop_duplicates(subset=["pix_col", "pix_row"]).copy()
        sub["period"] = label
        return sub

    cols = list(burned.columns) + ["period"]
    parts = []
    for p in periods:
        if _blank(by_label.get(p.label, {}).get("burned_area_km2", "")):
            continue
        sub = burned_in(burned, p.start, p.end)
        if len(sub):
            parts.append(per_period(sub, p.label))
    current = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)

    prior_table = None
    if prior_map is not None:
        parts = []
        for p in periods:
            pr = prior_map.get(p.label)
            if pr is None:
                continue
            sub = burned_in(burned, pr.start, pr.end)
            if sub.empty:
                continue
            sub = per_period(sub, pr.label)
            sub["compares_with"] = p.label
            if p.label not in compared:
                sub["in_comparison"] = "no"
            elif latest_day < p.last_day:
                cutoff = prior_cutoff(p, pr, latest_day, freq).isoformat()
                sub["in_comparison"] = np.where(sub["date"] < cutoff, "yes", "no")
            else:
                sub["in_comparison"] = "yes"
            parts.append(sub)
        prior_table = (pd.concat(parts, ignore_index=True) if parts
                       else pd.DataFrame(columns=cols + ["compares_with", "in_comparison"]))
    return current.reset_index(drop=True), prior_table


# ---------------------------------------------------------------------------
# Rasters (share of each cell burned, native sinusoidal grid)
# ---------------------------------------------------------------------------

@dataclass
class BurnedGrid:
    col0: int                    # global pixel column of the left edge
    row0: int                    # global pixel row of the top edge
    factor: int                  # native pixels per cell side
    nx: int
    ny: int
    inside: np.ndarray = field(repr=False)     # bool (ny, nx), row 0 = top

    @property
    def cell_m(self) -> float:
        return PIXEL_M * self.factor

    @property
    def left(self) -> float:
        return GRID_X0 + self.col0 * PIXEL_M

    @property
    def top(self) -> float:
        return GRID_Y0 - self.row0 * PIXEL_M


def sinusoidal_crs():
    from pyproj import CRS
    return CRS.from_proj4(SINU_PROJ4)


def make_burned_grid(aoi_geom, max_cells: int = MAX_RASTER_CELLS) -> BurnedGrid:
    """Grid aligned to the native MODIS pixels, covering the area of interest."""
    import geopandas as gpd
    import shapely
    aoi_s = gpd.GeoSeries([shapely.segmentize(aoi_geom, 0.01)], crs=4326).to_crs(sinusoidal_crs()).iloc[0]
    minx, miny, maxx, maxy = aoi_s.bounds
    c_min = int(math.floor((minx - GRID_X0) / PIXEL_M))
    c_max = int(math.ceil((maxx - GRID_X0) / PIXEL_M))
    r_min = int(math.floor((GRID_Y0 - maxy) / PIXEL_M))
    r_max = int(math.ceil((GRID_Y0 - miny) / PIXEL_M))
    span = max(c_max - c_min, r_max - r_min, 1)
    factor = max(1, int(math.ceil(span / max_cells)))
    c_min = (c_min // factor) * factor
    r_min = (r_min // factor) * factor
    nx = int(math.ceil((c_max - c_min) / factor))
    ny = int(math.ceil((r_max - r_min) / factor))
    cell = PIXEL_M * factor
    xs = GRID_X0 + c_min * PIXEL_M + (np.arange(nx) + 0.5) * cell
    ys = GRID_Y0 - r_min * PIXEL_M - (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)
    shapely.prepare(aoi_s)
    inside = shapely.contains_xy(aoi_s, gx.ravel(), gy.ravel()).reshape(ny, nx)
    return BurnedGrid(c_min, r_min, factor, nx, ny, inside)


def burned_share(pixels: pd.DataFrame, grid: BurnedGrid) -> np.ndarray:
    """Share of each cell burned (0 to 1); each pixel counted once however many
    times it burned. NODATA outside the area of interest."""
    counts = np.zeros((grid.ny, grid.nx))
    if len(pixels):
        px = pixels[["pix_col", "pix_row"]].drop_duplicates()
        c = (px["pix_col"].to_numpy() - grid.col0) // grid.factor
        r = (px["pix_row"].to_numpy() - grid.row0) // grid.factor
        ok = (c >= 0) & (c < grid.nx) & (r >= 0) & (r < grid.ny)
        np.add.at(counts, (r[ok], c[ok]), 1.0)
    out = (counts / grid.factor ** 2).astype("float32")
    # A burned pixel whose cell centre falls just outside the boundary is kept
    # rather than hidden, so no fetched burn disappears from the map.
    out[~grid.inside & (counts == 0)] = NODATA
    return out


def write_burned_raster(path: Path, data: np.ndarray, grid: BurnedGrid) -> Path:
    """GeoTIFF via rasterio when available, otherwise an Esri ASCII grid."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        rasterio = None
    crs = sinusoidal_crs()
    if rasterio is not None:
        tmp = path.with_name(path.stem + "_writing.tif")
        with rasterio.open(tmp, "w", driver="GTiff", width=grid.nx, height=grid.ny, count=1, dtype="float32",
                           crs=crs.to_wkt(), transform=from_origin(grid.left, grid.top, grid.cell_m, grid.cell_m),
                           nodata=NODATA, compress="deflate") as dst:
            dst.write(data, 1)
        _replace(tmp, path)
        return path
    from pyproj.enums import WktVersion
    asc = path.with_suffix(".asc")
    tmp = asc.with_name(asc.stem + "_writing.asc")
    bottom = grid.top - grid.ny * grid.cell_m
    header = (f"ncols {grid.nx}\nnrows {grid.ny}\nxllcorner {grid.left:.3f}\nyllcorner {bottom:.3f}\n"
              f"cellsize {grid.cell_m:.6f}\nNODATA_value {NODATA:g}\n")
    with open(tmp, "w", encoding="ascii") as fh:
        fh.write(header)
        np.savetxt(fh, data, fmt="%.4g")
    _replace(tmp, asc)
    prj_tmp = asc.with_name(asc.stem + "_writing.prj")
    prj_tmp.write_text(crs.to_wkt(WktVersion.WKT1_ESRI), encoding="ascii")
    _replace(prj_tmp, asc.with_suffix(".prj"))
    return asc


def _replace(tmp: Path, final: Path) -> None:
    try:
        os.replace(tmp, final)
    except PermissionError as exc:
        raise RuntimeError(f"Could not replace {final.name}; it is probably open in ArcGIS Pro. Remove it from "
                           f"your map and run again. The new version is in {tmp.name} meanwhile.") from exc


def _tok(s: str) -> str:
    import re
    return re.sub(r"[^0-9a-z]+", "_", str(s).lower()).strip("_") or "na"


def burned_file_name(geoextent: str, label: str, variant: str = "") -> str:
    """e.g. sct_evnt_fir_ras_s0_nasa_pp_mcd64a1_burned_2026_03_prior_year.tif"""
    tail = f"_{variant}" if variant else ""
    return f"{_tok(geoextent)}_evnt_fir_ras_s0_nasa_pp_mcd64a1_burned_{_tok(label)}{tail}.tif"


def burned_dir_name(geoextent: str) -> str:
    return f"{_tok(geoextent)}_fire_activity_burned"


def burned_raster_files(out_dir: Path, geoextent: str) -> list:
    """This tool's burned area raster files in the output folder (other files are ignored)."""
    folder = Path(out_dir) / burned_dir_name(geoextent)
    if not folder.exists():
        return []
    prefix = f"{_tok(geoextent)}_evnt_fir_ras_s0_nasa_pp_mcd64a1_burned_"
    return [f for f in folder.iterdir() if f.name.startswith(prefix)
            and f.suffix.lower() in (".tif", ".asc", ".prj", ".xml", ".ovr")]


def clear_burned_rasters(out_dir: Path, geoextent: str) -> int:
    """Remove this tool's burned area rasters from an earlier run (other files in
    the folder are left alone), so the folder never holds rasters that do not
    match the current outputs. Returns how many files were removed."""
    n = 0
    for old in burned_raster_files(out_dir, geoextent):
        try:
            old.unlink()
            n += 1
        except PermissionError as exc:
            raise RuntimeError(f"Could not replace {old.name}; it is probably open in ArcGIS Pro. Remove the "
                               "burned area rasters from your map and run again.") from exc
    return n


def write_burned_rasters(aoi_geom, current: pd.DataFrame, prior: Optional[pd.DataFrame], periods, summary_rows,
                         geoextent: str, out_dir: Path, mode: str = "all",
                         log: Callable[[str], None] = print) -> tuple[list[Path], list[str]]:
    """Share-burned rasters for the whole run and (mode 'all') each period, with
    year-earlier and change rasters when comparing. Returns (paths, info lines)."""
    if mode not in RASTER_CHOICES:
        raise ValueError(f"--burned-rasters must be one of {RASTER_CHOICES}, got {mode!r}")
    clear_burned_rasters(out_dir, geoextent)
    if mode == "none":
        return [], []
    grid = make_burned_grid(aoi_geom)
    folder = out_dir / burned_dir_name(geoextent)
    folder.mkdir(parents=True, exist_ok=True)
    compare = prior is not None
    paths: list[Path] = []
    in_cmp = prior[prior["in_comparison"] == "yes"] if compare and len(prior) else prior

    def write(label, cur, pri, with_prior, cur_compared=None):
        cs = burned_share(cur, grid)
        paths.append(write_burned_raster(folder / burned_file_name(geoextent, label), cs, grid))
        if with_prior:
            ps = burned_share(pri, grid)
            paths.append(write_burned_raster(folder / burned_file_name(geoextent, label, "prior_year"), ps, grid))
            base = cs if cur_compared is None else burned_share(cur_compared, grid)
            valid = (cs != NODATA) | (ps != NODATA)
            diff = np.where(valid, np.where(base == NODATA, 0, base) - np.where(ps == NODATA, 0, ps), NODATA)
            paths.append(write_burned_raster(folder / burned_file_name(geoextent, label, "change"),
                                             diff.astype("float32"), grid))

    compared = compared_burned_labels(summary_rows) if compare else set()
    cur_cmp = current[current["period"].isin(compared)] if compare and len(current) else current
    write("total", current, in_cmp, compare, cur_cmp if compare else None)
    if mode == "all":
        by_label = {r["period"]: r for r in summary_rows}
        for p in periods:
            row = by_label.get(p.label, {})
            if _blank(row.get("burned_area_km2", "")):
                continue
            cp = current[current["period"] == p.label] if len(current) else current
            has_prior = compare and p.label in compared
            pp = in_cmp[in_cmp["compares_with"] == p.label] if has_prior and len(in_cmp) else in_cmp
            write(p.label, cp, pp, has_prior)
    fmt = "GeoTIFF" if paths and paths[0].suffix == ".tif" else "Esri ASCII grid (rasterio not installed)"
    cell = (f"native {PIXEL_M:.0f} m pixels, values 0 (not burned) or 1 (burned)" if grid.factor == 1 else
            f"{grid.cell_m / 1000:.2f} km cells ({grid.factor} x {grid.factor} native pixels), values = share "
            "of the cell burned, 0 to 1")
    log(f"Burned area rasters: {len(paths)} on the MODIS sinusoidal grid, {cell}")
    info = [f"Burned area rasters: {len(paths)} {fmt} file(s) in {folder.name}, MODIS sinusoidal projection, {cell}.",
            "  'total' rasters show land burned at least once in the run; change rasters are current minus year "
            "earlier (the total change raster uses only periods with a year-earlier match)."]
    return paths, info


# ---------------------------------------------------------------------------
# Download loop with cache
# ---------------------------------------------------------------------------

def months_needed(windows: list[tuple[date, date]]) -> list[date]:
    """First day of every calendar month any [start, end) window touches (one
    MCD64A1 image each). Whole months are always read, so the cache can be
    shared between weekly, monthly and annual runs."""
    months = set()
    for s, e in windows:
        cur = month_start(s)
        while cur < e:
            months.add(cur)
            cur = next_month(cur)
    return sorted(months)


def collect_burned(fetch: Callable[[date, date], pd.DataFrame], months: list[date],
                   cache_dir: Optional[Path], latest_day: Optional[date], log: Callable[[str], None] = print,
                   fetch_with_split: Optional[Callable] = None,
                   published: Optional[set] = None) -> pd.DataFrame:
    """Fetch every whole month (from cache where possible). MCD64A1 is a final
    product, so every published month is cached. `published` (month keys
    'YYYY-MM' that have an image) lets gaps in the archive be skipped with a
    message instead of failing."""
    frames = []
    for i, m in enumerate(months, 1):
        s, e = m, next_month(m)
        tag = f"  [{i}/{len(months)}] {m:%B %Y}"
        if latest_day is None or s > latest_day:
            log(f"{tag}: not yet published, skipped")
            continue
        if published is not None and month_key(m) not in published:
            log(f"{tag}: no image in the Earth Engine archive for this month, skipped (burned area for it is "
                "flagged coverage_unknown)")
            continue
        cache_file = cache_dir / f"{month_key(m)}.csv" if cache_dir else None
        if cache_file is not None and cache_file.exists():
            df = pd.read_csv(cache_file, dtype={"date": str})
            log(f"{tag}: {len(df)} burned pixels (cached)")
        else:
            df = fetch_with_split(fetch, s, e, log) if fetch_with_split else fetch(s, e)
            df = normalise_burned(df)
            log(f"{tag}: {len(df)} burned pixels")
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_suffix(".tmp")
                df.to_csv(tmp, index=False)
                os.replace(tmp, cache_file)
        frames.append(df)
    frames = [normalise_burned(f) for f in frames if not f.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else empty_burned()
    return dedupe_burned(combined)


def collect_coverage(fetcher, months: list[date], cache_dir: Optional[Path], latest_day: Optional[date],
                     log: Callable[[str], None] = print, published: Optional[set] = None) -> dict:
    """{'YYYY-MM': (land pixels, mapped land pixels)} for each published month."""
    months = [m for m in months if latest_day is not None and m <= latest_day
              and (published is None or month_key(m) in published)]
    cov: dict = {}
    todo = []
    for m in months:
        f = cache_dir / f"coverage_{month_key(m)}.json" if cache_dir else None
        if f is not None and f.exists():
            v = json.loads(f.read_text(encoding="utf-8"))
            cov[month_key(m)] = (float(v[0]), float(v[1]))
        else:
            todo.append(m)
    if todo:
        fresh = fetcher.coverage(todo)
        for m in todo:
            v = fresh.get(month_key(m))
            if v is None:
                continue
            cov[month_key(m)] = v
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                f = cache_dir / f"coverage_{month_key(m)}.json"
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(list(v)), encoding="utf-8")
                os.replace(tmp, f)
    return cov


# ---------------------------------------------------------------------------
# Earth Engine glue - the only part that touches `ee`
# ---------------------------------------------------------------------------

def _ms_to_date(ms) -> date:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()


class BurnedAreaFetcher:
    """Burned pixels and mapped coverage from MCD64A1 for one area of interest."""

    def __init__(self, aoi_geojson: dict, splittable=None, transient=None):
        import ee
        self.ee = ee
        self.region = ee.Geometry(aoi_geojson, None, False)
        self.collection = ee.ImageCollection(COLLECTION_ID)
        self._splittable = splittable
        self._transient = transient

    def latest_image_date(self) -> Optional[date]:
        try:
            ms = self.collection.aggregate_max("system:time_start").getInfo()
        except self.ee.EEException as exc:
            raise RuntimeError(f"Earth Engine error reading the latest burned area month: {exc}") from exc
        return _ms_to_date(ms) if ms is not None else None

    def published_months(self, start: date, end: date) -> set:
        """Month keys ('YYYY-MM') that have an image between start and end."""
        try:
            ms_list = (self.collection.filterDate(start.isoformat(), end.isoformat())
                       .aggregate_array("system:time_start").getInfo()) or []
        except self.ee.EEException as exc:
            raise RuntimeError(f"Earth Engine error listing burned area months: {exc}") from exc
        return {month_key(_ms_to_date(ms)) for ms in ms_list}

    def _image_for(self, d: date):
        ee = self.ee
        m0 = month_start(d)
        return ee.Image(self.collection.filterDate(m0.isoformat(), next_month(m0).isoformat()).first())

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        """Burned pixels with a burn date in [start, end). The window must sit
        inside one calendar month (one image)."""
        if month_start(start) != month_start(end - timedelta(days=1)):
            raise ValueError(f"burned area request {start} to {end} crosses a month boundary")
        ee = self.ee
        img = self._image_for(start)
        lo = start.timetuple().tm_yday
        hi = (end - timedelta(days=1)).timetuple().tm_yday
        bd = img.select("BurnDate")
        burned = bd.gte(lo).And(bd.lte(hi))
        # Uncertainty is unmasked to -1 so a burned pixel without it is not
        # dropped by dropNulls (normalise_burned turns -1 back into blank).
        stacked = (bd.addBands(img.select("Uncertainty").unmask(-1)).addBands(ee.Image.pixelLonLat())
                   .updateMask(burned))
        fc = stacked.sample(region=self.region, projection=bd.projection(), dropNulls=True, tileScale=4,
                            geometries=False).map(lambda f: f.set("year", start.year))
        try:
            raw = ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
        except ee.EEException as exc:
            self._raise(exc, start, end)
        except (ConnectionError, TimeoutError, OSError) as exc:
            if self._transient:
                raise self._transient(f"network problem: {exc}") from exc
            raise
        return normalise_burned(raw)

    def _raise(self, exc, start, end):
        msg = str(exc).lower()
        if self._splittable and any(m in msg for m in ("memory limit", "computation timed out", "too large",
                                                         "too many pixels", "capacity exceeded")):
            raise self._splittable(str(exc)) from exc
        if self._transient and any(m in msg for m in ("too many concurrent", "rate limit", "quota exceeded", "429",
                                                        "503", "service unavailable", "deadline exceeded",
                                                        "internal error")):
            raise self._transient(str(exc)) from exc
        raise RuntimeError(f"Earth Engine error (burned area) for {start} to {end}: {exc}") from exc

    def coverage(self, months: list[date]) -> dict:
        """Land pixels and mapped land pixels (QA bits 0 and 1 set) per month."""
        ee = self.ee
        region = self.region
        out = {}
        for m in months:
            img = self._image_for(m)
            qa = img.select("QA").unmask(0)
            land = qa.bitwiseAnd(1).neq(0)
            mapped = land.And(qa.bitwiseAnd(2).neq(0))
            stats = (ee.Image.cat([land.rename("land"), mapped.rename("mapped")])
                     .reduceRegion(ee.Reducer.sum(), region, crs=qa.projection(), maxPixels=1e13, tileScale=4))
            v = None
            for wait in (10, 30, 60, None):
                try:
                    v = stats.getInfo()
                    break
                except ee.EEException as exc:
                    msg = str(exc).lower()
                    transient = any(t in msg for t in ("too many concurrent", "rate limit", "quota exceeded", "429",
                                                       "503", "service unavailable", "deadline exceeded",
                                                       "internal error"))
                    if not transient or wait is None:
                        raise RuntimeError(f"Earth Engine error checking burned area coverage for "
                                           f"{month_key(m)}: {exc}") from exc
                    time.sleep(wait)
            v = v or {}
            out[month_key(m)] = (float(v.get("land") or 0), float(v.get("mapped") or 0))
        return out


def diagnose_lines(aoi_geojson: Optional[dict]) -> list[str]:
    """What the burned area dataset looks like now (for fire_tool --diagnose)."""
    import ee
    col = ee.ImageCollection(COLLECTION_ID)
    lines = [f"{LABEL}  [{COLLECTION_ID}]"]
    latest = _ms_to_date(col.aggregate_max("system:time_start").getInfo())
    lines.append(f"  latest image: {fmt_date(latest)} (covers up to {fmt_date(latest_covered_day(latest))})")
    img = ee.Image(col.sort("system:time_start", False).first())
    lines.append(f"  bands: {img.bandNames().getInfo()}")
    proj = img.select("BurnDate").projection()
    lines.append(f"  BurnDate CRS: {proj.crs().getInfo()}, nominal scale {proj.nominalScale().getInfo():.2f} m "
                 f"(expected {PIXEL_M:.2f})")
    if aoi_geojson is not None:
        region = ee.Geometry(aoi_geojson, None, False)
        bd = img.select("BurnDate")
        stats = bd.updateMask(bd.gt(0)).reduceRegion(ee.Reducer.minMax().combine(ee.Reducer.count(), "", True),
                                                     region, crs=proj, maxPixels=1e13, tileScale=4).getInfo()
        m0 = month_start(latest)
        lo, hi = m0.timetuple().tm_yday, (next_month(m0) - timedelta(days=1)).timetuple().tm_yday
        lines.append(f"  latest image, your area: {stats.get('BurnDate_count')} burned pixels, BurnDate "
                     f"{stats.get('BurnDate_min')} to {stats.get('BurnDate_max')} (that month is days {lo} to {hi})")
        masked = bd.mask().Not().reduceRegion(ee.Reducer.sum(), region, crs=proj, maxPixels=1e13,
                                               tileScale=4).getInfo()
        lines.append(f"  masked BurnDate pixels in your area: {masked.get('BurnDate')} (unburned land, water or no "
                     "data - Earth Engine masks unburned pixels)")
        # The mapped-coverage figure needs the QA band to cover unburned land too.
        qa = img.select("QA")
        q = ee.Image.cat([qa.mask().rename("qa_unmasked"),
                          qa.unmask(0).bitwiseAnd(1).neq(0).rename("land"),
                          qa.unmask(0).bitwiseAnd(3).eq(3).rename("mapped")]) \
            .reduceRegion(ee.Reducer.sum(), region, crs=proj, maxPixels=1e13, tileScale=4).getInfo()
        land, mapped = q.get("land") or 0, q.get("mapped") or 0
        lines.append(f"  QA band, your area: {q.get('qa_unmasked'):,.0f} unmasked pixels, {land:,.0f} land, "
                     f"{mapped:,.0f} land with valid data"
                     + (f" ({100 * mapped / land:.1f} per cent mapped)" if land else ""))
        lines.append("  (land should be close to your area's pixel count; if it is only about the number of burned "
                     "pixels, the QA band is masked like BurnDate and the mapped-coverage figure is not usable)")
    return lines
