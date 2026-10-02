"""Offline tests for fire_burned.py (Phase 3: burned area).

Run from the tool folder:  python -m pytest tests -q
"""
import sys
from argparse import Namespace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fire_burned as fb  # noqa: E402
import fire_tool as ft  # noqa: E402

try:   # rasterio is optional: without it rasters are Esri ASCII grids
    import rasterio as _rasterio  # noqa: F401
    RASTER_EXT = ".tif"
except ImportError:
    RASTER_EXT = ".asc"

SAMPLE_AOI = Path(__file__).resolve().parents[1] / "sample_aoi" / "toy_bbox.geojson"   # -8.3..-7.7, 39.8..40.2


def pixel_centre(col, row):
    """Longitude/latitude of the centre of a native MODIS 463 m pixel."""
    from pyproj import CRS, Transformer
    t = Transformer.from_crs(CRS.from_proj4(fb.SINU_PROJ4), CRS.from_epsg(4326), always_xy=True)
    x = fb.GRID_X0 + (col + 0.5) * fb.PIXEL_M
    y = fb.GRID_Y0 - (row + 0.5) * fb.PIXEL_M
    return t.transform(x, y)


def pixel_at(lon, lat):
    c, r = fb.pixel_index([lon], [lat])
    return int(c[0]), int(r[0])


def burned_rows(lon, lat, n, day, dx=1):
    """n burned pixels in a row starting at the pixel holding (lon, lat)."""
    c0, r0 = pixel_at(lon, lat)
    out = []
    for i in range(n):
        x, y = pixel_centre(c0 + i * dx, r0)
        out.append({"date": day.isoformat(), "longitude": x, "latitude": y, "burn_date_uncertainty_days": 2})
    return out


# ---------------------------------------------------------------------------
# Grid, dates, normalising
# ---------------------------------------------------------------------------

def test_pixel_area_and_index_round_trip():
    assert fb.PIXEL_KM2 == pytest.approx(0.2147, abs=1e-4)
    for col, row in [(17000, 10500), (40000, 3000), (60000, 20000)]:
        lon, lat = pixel_centre(col, row)
        assert pixel_at(lon, lat) == (col, row)


def test_normalise_converts_burn_day_to_date_and_drops_unburned():
    raw = pd.DataFrame({"BurnDate": [32, 0, 60, -1], "Uncertainty": [1, 0, 3, 0], "longitude": [1, 2, 3, 4],
                        "latitude": [5, 6, 7, 8], "year": [2025, 2025, 2024, 2025]})
    df = fb.normalise_burned(raw)
    assert list(df["date"]) == ["2025-02-01", "2024-02-29"]     # leap year: day 60 = 29 February
    assert list(df["burn_date_uncertainty_days"]) == [1, 3]
    assert fb.normalise_burned(None).empty


def test_latest_covered_day_is_end_of_image_month():
    assert fb.latest_covered_day(date(2026, 7, 1)) == date(2026, 7, 31)
    assert fb.latest_covered_day(date(2026, 2, 1)) == date(2026, 2, 28)
    assert fb.latest_covered_day(None) is None


def test_months_needed_are_whole_months_without_repeats():
    assert fb.months_needed([(date(2025, 7, 28), date(2025, 8, 11)), (date(2025, 8, 25), date(2025, 9, 1))]) == [
        date(2025, 7, 1), date(2025, 8, 1)]


def test_qa_flags():
    latest = date(2025, 8, 31)
    assert fb.burned_qa_flag(date(2025, 9, 1), date(2025, 10, 1), latest, "") == "not_yet_available"
    assert fb.burned_qa_flag(date(2025, 1, 1), date(2026, 1, 1), latest, 99.0) == "partial_period"
    assert fb.burned_qa_flag(date(2025, 1, 1), date(2026, 1, 1), latest, 50.0) == "partial_period+low_coverage"
    assert fb.burned_qa_flag(date(2025, 8, 1), date(2025, 9, 1), latest, 80.0) == "low_coverage"
    assert fb.burned_qa_flag(date(2025, 8, 1), date(2025, 9, 1), latest, 95.0) == "ok"
    assert fb.burned_qa_flag(date(2025, 8, 1), date(2025, 9, 1), latest, "") == "coverage_unknown"
    assert fb.burned_qa_flag(date(2025, 8, 1), date(2025, 9, 1), None, "") == "not_yet_available"


def test_mapped_pct_weights_months_by_days():
    cov = {"2025-07": (1000.0, 1000.0), "2025-08": (1000.0, 500.0)}
    # Week 28 July to 3 August: 4 days of July (100%), 3 days of August (50%).
    pct = fb.mapped_pct_for(date(2025, 7, 28), date(2025, 8, 4), cov)
    w_jul, w_aug = 4 / 31, 3 / 31
    assert pct == round(100 * (w_jul * 1000 + w_aug * 500) / (w_jul * 1000 + w_aug * 1000), 1)
    assert fb.mapped_pct_for(date(2025, 9, 1), date(2025, 10, 1), cov) == ""
    # Days after the latest published day are left out.
    assert fb.mapped_pct_for(date(2025, 7, 1), date(2025, 9, 30), cov, date(2025, 7, 31)) == 100.0


# ---------------------------------------------------------------------------
# Summary rows
# ---------------------------------------------------------------------------

def _tables(df, periods, freq, latest, compare=True, cov=None, missing=frozenset()):
    prior = ft.build_prior_map(periods, freq) if compare else None
    rows = [{"period": p.label} for p in periods]
    fb.add_burned_to_summary(rows, df, periods, prior, latest, cov or {}, freq, missing)
    return fb.burned_point_tables(df, periods, prior, latest, freq, rows), rows


def _burned_table():
    rows = (burned_rows(-8.2, 40.0, 5, date(2025, 8, 10))
            + burned_rows(-8.0, 40.0, 2, date(2025, 8, 25))
            + burned_rows(-7.8, 40.1, 3, date(2024, 8, 5))
            + burned_rows(-7.75, 40.15, 1, date(2024, 8, 28))
            + burned_rows(-7.9, 39.9, 2, date(2024, 9, 15)))
    return fb.dedupe_burned(fb.normalise_burned(pd.DataFrame(rows)))


def test_summary_monthly_with_comparison_and_not_yet_available():
    df = _burned_table()
    periods = ft.build_periods("2025-08-01", "2025-10-01", "monthly")
    prior = ft.build_prior_map(periods, "monthly")
    rows = [{"period": p.label} for p in periods]
    cov = {"2025-08": (1000.0, 990.0), "2024-08": (1000.0, 800.0), "2024-09": (1000.0, 1000.0)}
    fb.add_burned_to_summary(rows, df, periods, prior, date(2025, 8, 31), cov, "monthly")
    aug, sep = rows
    assert aug["burned_area_km2"] == round(7 * fb.PIXEL_KM2, 2)
    assert aug["prior_burned_area_km2"] == round(4 * fb.PIXEL_KM2, 2)
    assert aug["burned_change_km2"] == round(aug["burned_area_km2"] - aug["prior_burned_area_km2"], 2)
    assert aug["burned_change_pct"] == 75.0
    assert aug["burned_mapped_pct"] == 99.0 and aug["prior_burned_mapped_pct"] == 80.0
    assert aug["burned_qa_flag"] == "ok"
    # September 2025 is not published yet: blank, not zero, and no comparison.
    assert sep["burned_qa_flag"] == "not_yet_available"
    assert sep["burned_area_km2"] == "" and sep["prior_burned_area_km2"] == "" and sep["burned_change_km2"] == ""
    assert "not yet published" in sep["burned_note"]


def test_summary_partial_period_is_like_for_like():
    df = _burned_table()
    periods = ft.build_periods("2025-01-01", "2026-01-01", "annual")
    prior = ft.build_prior_map(periods, "annual")
    rows = [{"period": p.label} for p in periods]
    fb.add_burned_to_summary(rows, df, periods, prior, date(2025, 8, 31), {"2025-01": (10.0, 10.0)}, "annual")
    r = rows[0]
    assert r["burned_qa_flag"] == "partial_period"
    # Year earlier counts only 1 January to 31 August 2024: September 2024 pixels left out.
    assert r["prior_burned_area_km2"] == round(4 * fb.PIXEL_KM2, 2)
    assert "1 January 2024 to 31 August 2024" in r["burned_note"]


def test_partial_annual_period_uses_same_calendar_date_across_leap_years():
    from datetime import date as D
    p25 = ft.build_periods("2025-01-01", "2026-01-01", "annual")[0]
    p24 = ft.build_periods("2024-01-01", "2025-01-01", "annual")[0]
    # 2025 to 31 August -> 2024 to 31 August (not 30 August, although 2024 is a leap year).
    assert fb.prior_cutoff(p25, ft.year_ago_period(p25, "annual"), D(2025, 8, 31), "annual") == D(2024, 9, 1)
    # 2024 to 31 August -> 2023 to 31 August (not into September).
    assert fb.prior_cutoff(p24, ft.year_ago_period(p24, "annual"), D(2024, 8, 31), "annual") == D(2023, 9, 1)
    # 2024 to 29 February -> 2023 to 28 February.
    assert fb.prior_cutoff(p24, ft.year_ago_period(p24, "annual"), D(2024, 2, 29), "annual") == D(2023, 3, 1)
    # A burn on 31 August 2024 counts against 2025 data to 31 August.
    df = fb.dedupe_burned(fb.normalise_burned(pd.DataFrame(burned_rows(-8.2, 40.0, 2, D(2024, 8, 31)))))
    rows = [{"period": "2025"}]
    fb.add_burned_to_summary(rows, df, [p25], ft.build_prior_map([p25], "annual"), D(2025, 8, 31), {}, "annual")
    assert rows[0]["prior_burned_area_km2"] == round(2 * fb.PIXEL_KM2, 2)
    (cur, pri), _ = _tables(df, [p25], "annual", D(2025, 8, 31))
    assert list(pri["in_comparison"]) == ["yes", "yes"]


def test_weekly_partial_period_uses_same_days_into_the_iso_week():
    periods = ft.build_periods("2025-07-28", "2025-08-04", "weekly")          # 28 July to 3 August 2025
    pr = ft.year_ago_period(periods[0], "weekly")                              # 29 July to 4 August 2024
    assert fb.prior_cutoff(periods[0], pr, date(2025, 7, 31), "weekly") == date(2024, 8, 2)


def test_low_year_earlier_coverage_is_noted():
    df = _burned_table()
    periods = ft.build_periods("2025-08-01", "2025-09-01", "monthly")
    rows = [{"period": "2025-08"}]
    cov = {"2025-08": (1000.0, 990.0), "2024-08": (1000.0, 500.0)}
    fb.add_burned_to_summary(rows, df, periods, ft.build_prior_map(periods, "monthly"), date(2025, 8, 31), cov,
                             "monthly")
    assert rows[0]["burned_qa_flag"] == "ok"
    assert "year-earlier mapped coverage only 50 per cent" in rows[0]["burned_note"]


def test_summary_weekly_has_approximate_note_and_no_prior_without_comparison():
    df = _burned_table()
    periods = ft.build_periods("2025-08-04", "2025-08-18", "weekly")
    rows = [{"period": p.label} for p in periods]
    fb.add_burned_to_summary(rows, df, periods, None, date(2025, 8, 31), {}, "weekly")
    assert rows[0]["burned_area_km2"] == round(5 * fb.PIXEL_KM2, 2)     # 10 August in week 32
    assert "prior_burned_area_km2" not in rows[0]
    assert "approximate burn dates" in rows[0]["burned_note"]          # note kept without a comparison
    rows = [{"period": p.label} for p in periods]
    fb.add_burned_to_summary(rows, df, periods, ft.build_prior_map(periods, "weekly"), date(2025, 8, 31), {},
                             "weekly")
    assert "approximate burn dates" in rows[0]["burned_note"]


def test_point_tables_mark_like_for_like_prior_pixels():
    df = _burned_table()
    periods = ft.build_periods("2025-01-01", "2026-01-01", "annual")
    (cur, pri), _ = _tables(df, periods, "annual", date(2025, 8, 31))
    assert len(cur) == 7 and set(cur["period"]) == {"2025"}
    assert len(pri) == 6
    assert (pri["in_comparison"] == "yes").sum() == 4      # September 2024 pixels marked "no"
    assert set(pri["compares_with"]) == {"2025"}


def test_point_tables_skip_unpublished_periods():
    df = _burned_table()
    periods = ft.build_periods("2025-08-01", "2025-10-01", "monthly")
    (cur, pri), _ = _tables(df, periods, "monthly", date(2025, 8, 31))
    assert set(cur["period"]) == {"2025-08"}
    assert set(pri.loc[pri["compares_with"] == "2025-09", "in_comparison"]) == {"no"}


def test_same_pixel_burning_twice_in_one_period_counts_once():
    rows = burned_rows(-8.2, 40.0, 1, date(2025, 3, 5)) + burned_rows(-8.2, 40.0, 1, date(2025, 4, 20))
    df = fb.dedupe_burned(fb.normalise_burned(pd.DataFrame(rows)))
    assert len(df) == 2                                  # different dates kept
    periods = ft.build_periods("2025-01-01", "2026-01-01", "annual")
    out = [{"period": "2025"}]
    fb.add_burned_to_summary(out, df, periods, None, date(2025, 12, 31), {}, "annual")
    assert out[0]["burned_area_km2"] == round(fb.PIXEL_KM2, 2)


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------

def test_grid_native_for_small_area_and_share_values():
    from shapely.geometry import box
    aoi = box(-8.3, 39.8, -7.7, 40.2)
    grid = fb.make_burned_grid(aoi)
    assert grid.factor == 1
    df = _burned_table()
    share = fb.burned_share(df[df["date"] >= "2025-01-01"], grid)
    vals = share[share != fb.NODATA]
    assert set(np.unique(vals)) <= {0.0, 1.0}
    assert vals.sum() == 7
    assert grid.inside.sum() > 0 and (share == fb.NODATA).sum() == (~grid.inside).sum()


def test_grid_coarsens_large_areas_to_fractions():
    from shapely.geometry import box
    grid = fb.make_burned_grid(box(-8.3, 39.8, -7.7, 40.2), max_cells=20)
    assert grid.factor > 1 and max(grid.nx, grid.ny) <= 21
    df = _burned_table()
    share = fb.burned_share(df[df["date"] >= "2025-01-01"], grid)
    vals = share[share != fb.NODATA]
    assert vals.sum() * grid.factor ** 2 == pytest.approx(7)
    assert vals.max() < 1


# ---------------------------------------------------------------------------
# Download loop and cache
# ---------------------------------------------------------------------------

class FakeBurnedFetcher:
    """Burned pixels in the toy box, latest image August 2025."""
    calls = []
    coverage_calls = []
    latest = date(2025, 8, 1)

    def __init__(self, aoi_geojson):
        self.df = _burned_table()

    def latest_image_date(self):
        return self.latest

    def fetch(self, start, end):
        FakeBurnedFetcher.calls.append((start, end))
        sub = self.df[(self.df["date"] >= start.isoformat()) & (self.df["date"] < end.isoformat())]
        out = pd.DataFrame({
            "BurnDate": [date.fromisoformat(d).timetuple().tm_yday for d in sub["date"]],
            "Uncertainty": sub["burn_date_uncertainty_days"].values,
            "longitude": sub["longitude"].values, "latitude": sub["latitude"].values, "year": start.year})
        return fb.normalise_burned(out)

    def coverage(self, months):
        FakeBurnedFetcher.coverage_calls.append(list(months))
        return {fb.month_key(m): (1000.0, 950.0) for m in months}


def test_collect_burned_caches_published_months_and_skips_later(tmp_path):
    FakeBurnedFetcher.calls = []
    f = FakeBurnedFetcher(None)
    months = fb.months_needed([(date(2025, 7, 1), date(2025, 10, 1))])
    logs = []
    df = fb.collect_burned(f.fetch, months, tmp_path, date(2025, 8, 31), logs.append)
    assert len(df) == 7 and len(FakeBurnedFetcher.calls) == 2      # July and August; September skipped
    assert any("not yet published" in m for m in logs)
    df2 = fb.collect_burned(f.fetch, months, tmp_path, date(2025, 8, 31), logs.append)
    assert len(FakeBurnedFetcher.calls) == 2                         # both from cache now
    assert df2[["date", "pix_col", "pix_row"]].equals(df[["date", "pix_col", "pix_row"]])


def test_weekly_run_cache_is_reused_by_monthly_run(tmp_path):
    FakeBurnedFetcher.calls = []
    f = FakeBurnedFetcher(None)
    weekly = fb.months_needed([(date(2025, 7, 28), date(2025, 8, 11))])
    fb.collect_burned(f.fetch, weekly, tmp_path, date(2025, 8, 31), lambda m: None)
    assert FakeBurnedFetcher.calls == [(date(2025, 7, 1), date(2025, 8, 1)), (date(2025, 8, 1), date(2025, 9, 1))]
    fb.collect_burned(f.fetch, fb.months_needed([(date(2025, 8, 1), date(2025, 9, 1))]), tmp_path,
                      date(2025, 8, 31), lambda m: None)
    assert len(FakeBurnedFetcher.calls) == 2


def test_gap_in_archive_is_skipped_and_flagged(tmp_path):
    FakeBurnedFetcher.calls = []
    f = FakeBurnedFetcher(None)
    logs = []
    months = [date(2025, 7, 1), date(2025, 8, 1)]
    fb.collect_burned(f.fetch, months, None, date(2025, 8, 31), logs.append, published={"2025-08"})
    assert FakeBurnedFetcher.calls == [(date(2025, 8, 1), date(2025, 9, 1))]
    assert any("no image in the Earth Engine archive" in m for m in logs)
    cov = fb.collect_coverage(f, months, None, date(2025, 8, 31), published={"2025-08"})
    assert set(cov) == {"2025-08"}
    rows = [{"period": "2025-07"}]
    periods = ft.build_periods("2025-07-01", "2025-08-01", "monthly")
    fb.add_burned_to_summary(rows, fb.empty_burned().pipe(fb.dedupe_burned), periods, None, date(2025, 8, 31), cov,
                             "monthly")
    assert rows[0]["burned_qa_flag"] == "coverage_unknown"


def test_missing_uncertainty_is_kept_as_blank():
    raw = pd.DataFrame({"BurnDate": [200], "Uncertainty": [-1], "longitude": [1.0], "latitude": [2.0],
                        "year": [2025]})
    df = fb.normalise_burned(raw)
    assert len(df) == 1 and df["burn_date_uncertainty_days"].isna().all()


def test_collect_coverage_uses_cache(tmp_path):
    FakeBurnedFetcher.coverage_calls = []
    f = FakeBurnedFetcher(None)
    months = [date(2025, 7, 1), date(2025, 8, 1), date(2025, 9, 1)]
    cov = fb.collect_coverage(f, months, tmp_path, date(2025, 8, 31))
    assert set(cov) == {"2025-07", "2025-08"}
    fb.collect_coverage(f, months, tmp_path, date(2025, 8, 31))
    assert FakeBurnedFetcher.coverage_calls == [[date(2025, 7, 1), date(2025, 8, 1)]]


def test_fetch_split_works_for_burned_area():
    f = FakeBurnedFetcher(None)

    def fetch(s, e):
        if (e - s).days > 8:
            raise ft.SplittableFetchError("User memory limit exceeded")
        return f.fetch(s, e)
    df = fb.collect_burned(fetch, [date(2025, 8, 1)], None, date(2025, 8, 31), lambda m: None, ft.fetch_with_split)
    assert len(df) == 7


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class FakeFireFetcher:
    def __init__(self, sensor, aoi_geojson, min_conf):
        self.sensor = sensor
        self.latest = date(2025, 9, 30)

    def latest_data_date(self):
        return self.latest

    def imagery_dates(self, start, end):
        d, out = start, set()
        while d < end and d <= self.latest:
            out.add(d.isoformat())
            d += timedelta(days=1)
        return out

    def fetch(self, start, end):
        rows, d = [], start
        while d < end and d <= self.latest:
            if d.day % 3 == 0:
                rows.append({"date": d.isoformat(), "longitude": -8.2, "latitude": 40.0, "confidence": 1,
                             "frp": 3.0, "Bright_ti4": 330.0, "DayNight": 1, "acq_time": 45000})
            d += timedelta(days=1)
        return ft.normalise_detections(pd.DataFrame(rows), self.sensor)


def _admin_file(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    gdf = gpd.GeoDataFrame({"ADM2_EN": ["West", "East"], "ADM2_PCODE": ["PT01", "PT02"]},
                           geometry=[box(-8.3, 39.8, -8.0, 40.2), box(-8.0, 39.8, -7.7, 40.2)], crs=4326)
    path = tmp_path / "admin.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def _args(tmp_path, **kw):
    base = dict(aoi_file=str(SAMPLE_AOI), geoextent="prt", start="2025-08-01", end="2025-10-01", freq="monthly",
                compare_prior_year=True, sensor="auto", min_confidence=None, out_dir=str(tmp_path), ee_project=None,
                simplify_tolerance=None, no_cache=True, dry_run=False, hex_size_km=None, admin_file=None,
                unit_name_field=None, unit_id_field=None, density="none", density_radius_km=None,
                no_burned_area=False, burned_rasters="all")
    base.update(kw)
    return Namespace(**base)


def _run(tmp_path, **kw):
    return ft.run(_args(tmp_path, **kw), fetcher_factory=FakeFireFetcher, log=lambda m: None,
                  burned_fetcher_factory=FakeBurnedFetcher)


def test_end_to_end_burned_outputs(tmp_path):
    import geopandas as gpd
    rasterio = pytest.importorskip("rasterio")
    assert _run(tmp_path, admin_file=str(_admin_file(tmp_path)), unit_id_field="ADM2_PCODE") == 0

    summary = pd.read_csv(tmp_path / "prt_fire_activity_summary.csv", keep_default_na=False)
    for col in fb.SUMMARY_COLUMNS + fb.COMPARISON_COLUMNS:
        assert col in summary.columns
    aug = summary[summary["period"] == "2025-08"].iloc[0]
    sep = summary[summary["period"] == "2025-09"].iloc[0]
    assert float(aug["burned_area_km2"]) == round(7 * fb.PIXEL_KM2, 2)
    assert float(aug["prior_burned_area_km2"]) == round(4 * fb.PIXEL_KM2, 2)
    assert float(aug["burned_mapped_pct"]) == 95.0
    assert sep["burned_qa_flag"] == "not_yet_available" and sep["burned_area_km2"] == ""

    gpkg = tmp_path / "prt_fire_activity.gpkg"
    hexes = gpd.read_file(gpkg, layer="prt_evnt_fir_py_s0_nasa_pp_viirs_hex2km_total")
    assert hexes["burned_area_km2"].sum() == pytest.approx(7 * fb.PIXEL_KM2, abs=0.05)
    assert hexes["burned_pixels"].sum() == 7 == int(aug["burned_pixels"])
    assert hexes["prior_burned_area_km2"].sum() == pytest.approx(4 * fb.PIXEL_KM2, abs=0.05)
    assert hexes["burned_share_pct"].max() > 0
    by_p = gpd.read_file(gpkg, layer="prt_evnt_fir_py_s0_nasa_pp_viirs_hex2km_by_period")
    # Hexagons with burned area but no fires still get a row.
    burned_only = by_p[(by_p["period"] == "2025-08") & (by_p["fire_count"] == 0) & (by_p["burned_area_km2"] > 0)]
    assert len(burned_only) > 0
    assert by_p.loc[by_p["period"] == "2025-09", "burned_area_km2"].isna().all()
    assert by_p.loc[by_p["period"] == "2025-08", "burned_area_km2"].sum() == pytest.approx(7 * fb.PIXEL_KM2,
                                                                                          abs=0.05)

    admin = pd.read_csv(tmp_path / "prt_fire_activity_admin_summary.csv")
    tot = admin[admin["period"] == "all"].set_index("unit_id")
    assert tot["burned_area_km2"].sum() == pytest.approx(7 * fb.PIXEL_KM2, abs=0.05)
    assert tot.loc["PT02", "prior_burned_area_km2"] == pytest.approx(4 * fb.PIXEL_KM2, abs=0.05)

    folder = tmp_path / "prt_fire_activity_burned"
    names = sorted(p.name for p in folder.iterdir())
    assert "prt_evnt_fir_ras_s0_nasa_pp_mcd64a1_burned_total.tif" in names
    assert "prt_evnt_fir_ras_s0_nasa_pp_mcd64a1_burned_2025_08_change.tif" in names
    assert not any("2025_09" in n for n in names)         # September 2025 not published
    with rasterio.open(folder / "prt_evnt_fir_ras_s0_nasa_pp_mcd64a1_burned_2025_08.tif") as src:
        a = src.read(1)
        assert src.crs is not None and src.res[0] == pytest.approx(fb.PIXEL_M)
    assert (a == 1).sum() == 7
    with rasterio.open(folder / "prt_evnt_fir_ras_s0_nasa_pp_mcd64a1_burned_2025_08_change.tif") as src:
        c = src.read(1)
    assert (c == 1).sum() == 7 and (c == -1).sum() == 4

    info = (tmp_path / "prt_fire_activity_run_info.txt").read_text(encoding="utf-8")
    assert "MCD64A1" in info and "August 2025" in info and "not_yet_available" in info


def test_burned_area_can_be_switched_off(tmp_path):
    import geopandas as gpd

    class Boom:
        def __init__(self, *_):
            raise AssertionError("burned fetcher should not be created")
    assert ft.run(_args(tmp_path, no_burned_area=True), fetcher_factory=FakeFireFetcher, log=lambda m: None,
                  burned_fetcher_factory=Boom) == 0
    summary = pd.read_csv(tmp_path / "prt_fire_activity_summary.csv")
    assert "burned_area_km2" not in summary.columns
    hexes = gpd.read_file(tmp_path / "prt_fire_activity.gpkg", layer="prt_evnt_fir_py_s0_nasa_pp_viirs_hex2km_total")
    assert "burned_area_km2" not in hexes.columns
    assert not (tmp_path / "prt_fire_activity_burned").exists()


def test_old_burned_rasters_are_cleared_when_rasters_or_burned_area_are_off(tmp_path):
    _run(tmp_path)
    folder = tmp_path / "prt_fire_activity_burned"
    (folder / "my_notes.txt").write_text("keep me")
    assert any(p.suffix in (".tif", ".asc") for p in folder.iterdir())
    _run(tmp_path, burned_rasters="none")
    assert [p.name for p in folder.iterdir()] == ["my_notes.txt"]
    _run(tmp_path)
    ft.run(_args(tmp_path, no_burned_area=True), fetcher_factory=FakeFireFetcher, log=lambda m: None)
    assert [p.name for p in folder.iterdir()] == ["my_notes.txt"]


def test_earth_engine_errors_give_a_plain_message(tmp_path, monkeypatch, capsys):
    class EEException(Exception):
        pass

    def boom(args, **kw):
        raise EEException("Too many concurrent aggregations.")
    monkeypatch.setattr(ft, "run", boom)
    code = ft.main(["--aoi-file", str(SAMPLE_AOI), "--geoextent", "x", "--start", "2025-08-01", "--end",
                    "2025-09-01", "--freq", "monthly", "--out-dir", str(tmp_path)])
    assert code == 2 and "Earth Engine error" in capsys.readouterr().err


def test_burned_rasters_totals_and_none(tmp_path):
    _run(tmp_path / "t", burned_rasters="totals")
    names = sorted(p.name for p in (tmp_path / "t" / "prt_fire_activity_burned").iterdir())
    rasters = [n for n in names if not n.endswith(".prj")]   # an ASCII grid comes with a .prj
    assert rasters == [f"prt_evnt_fir_ras_s0_nasa_pp_mcd64a1_burned_total{RASTER_EXT}",
                       f"prt_evnt_fir_ras_s0_nasa_pp_mcd64a1_burned_total_change{RASTER_EXT}",
                       f"prt_evnt_fir_ras_s0_nasa_pp_mcd64a1_burned_total_prior_year{RASTER_EXT}"]
    _run(tmp_path / "n", burned_rasters="none", compare_prior_year=False)
    assert not (tmp_path / "n" / "prt_fire_activity_burned").exists()
    summary = pd.read_csv(tmp_path / "n" / "prt_fire_activity_summary.csv")
    assert "burned_area_km2" in summary.columns and "prior_burned_area_km2" not in summary.columns


def test_nothing_published_yet_gives_blanks_not_zeros(tmp_path):
    import geopandas as gpd

    class Late(FakeBurnedFetcher):
        latest = date(2025, 6, 1)
    ft.run(_args(tmp_path, compare_prior_year=False), fetcher_factory=FakeFireFetcher, log=lambda m: None,
           burned_fetcher_factory=Late)
    summary = pd.read_csv(tmp_path / "prt_fire_activity_summary.csv", keep_default_na=False)
    assert set(summary["burned_qa_flag"]) == {"not_yet_available"}
    assert set(summary["burned_area_km2"]) == {""}
    hexes = gpd.read_file(tmp_path / "prt_fire_activity.gpkg", layer="prt_evnt_fir_py_s0_nasa_pp_viirs_hex2km_total")
    assert hexes["burned_area_km2"].isna().all() and hexes["burned_pixels"].isna().all()


def test_dry_run_mentions_burned_area(tmp_path):
    msgs = []
    assert ft.run(_args(tmp_path, dry_run=True), fetcher_factory=None, log=msgs.append) == 0
    assert any("MCD64A1" in m and "up to 4 monthly image(s)" in m for m in msgs)
    msgs = []
    ft.run(_args(tmp_path, dry_run=True, no_burned_area=True), fetcher_factory=None, log=msgs.append)
    assert any("Burned area: off" in m for m in msgs)


def test_cli_has_burned_options():
    args = ft.build_arg_parser().parse_args(["--no-burned-area", "--burned-rasters", "totals"])
    assert args.no_burned_area and args.burned_rasters == "totals"


def test_month_missing_from_archive_blanks_or_flags_and_drops_comparison():
    df = _burned_table()
    latest = date(2025, 8, 31)
    missing = frozenset({"2025-08"})
    # Monthly: the missing month itself is blank, flagged, not compared.
    periods = ft.build_periods("2025-07-01", "2025-09-01", "monthly")
    (cur, pri), rows = _tables(df, periods, "monthly", latest, cov={"2025-07": (10.0, 10.0)}, missing=missing)
    jul, aug = rows
    assert jul["burned_qa_flag"] == "ok" and jul["burned_area_km2"] == 0.0
    assert "missing_month" in aug["burned_qa_flag"] and aug["burned_area_km2"] == ""
    assert aug["prior_burned_area_km2"] == "" and "missing from the Earth Engine archive" in aug["burned_note"]
    assert set(cur["period"]) <= {"2025-07"}
    assert (pri.loc[pri["compares_with"] == "2025-08", "in_comparison"] == "no").all()
    # Annual: a period touching the missing month keeps its figure but is flagged and not compared.
    periods = ft.build_periods("2025-01-01", "2026-01-01", "annual")
    (cur, pri), rows = _tables(df, periods, "annual", latest, missing=missing)
    assert "missing_month" in rows[0]["burned_qa_flag"] and "partial_period" in rows[0]["burned_qa_flag"]
    assert rows[0]["burned_area_km2"] != "" and rows[0]["prior_burned_area_km2"] == ""
    assert (pri["in_comparison"] == "no").all()


def test_month_missing_in_the_year_before_drops_comparison():
    df = _burned_table()
    periods = ft.build_periods("2025-08-01", "2025-09-01", "monthly")
    (cur, pri), rows = _tables(df, periods, "monthly", date(2025, 8, 31), cov={"2025-08": (10.0, 10.0)},
                               missing=frozenset({"2024-08"}))
    assert rows[0]["burned_qa_flag"] == "ok" and rows[0]["burned_area_km2"] == round(7 * fb.PIXEL_KM2, 2)
    assert rows[0]["prior_burned_area_km2"] == "" and "year before: 2024-08" in rows[0]["burned_note"]
    assert (pri["in_comparison"] == "no").all()


def test_end_to_end_with_gap_in_archive(tmp_path):
    import geopandas as gpd

    class Gappy(FakeBurnedFetcher):
        def published_months(self, start, end):
            return {"2024-08", "2024-09"}                    # August 2025 missing
    ft.run(_args(tmp_path), fetcher_factory=FakeFireFetcher, log=lambda m: None, burned_fetcher_factory=Gappy)
    s = pd.read_csv(tmp_path / "prt_fire_activity_summary.csv", keep_default_na=False)
    aug = s[s["period"] == "2025-08"].iloc[0]
    assert "missing_month" in aug["burned_qa_flag"] and aug["burned_area_km2"] == ""
    hexes = gpd.read_file(tmp_path / "prt_fire_activity.gpkg", layer="prt_evnt_fir_py_s0_nasa_pp_viirs_hex2km_total")
    assert hexes["burned_area_km2"].isna().all()
    names = [p.name for p in (tmp_path / "prt_fire_activity_burned").iterdir()]
    assert not any("2025_08" in n for n in names)
