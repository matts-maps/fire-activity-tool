"""Offline tests for fire_tool.py - no Earth Engine needed.

Run from the tool folder:  python -m pytest tests -q
"""
import sys
from argparse import Namespace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fire_tool as ft  # noqa: E402

SAMPLE_AOI = Path(__file__).resolve().parents[1] / "sample_aoi" / "toy_bbox.geojson"


# --- periods ---------------------------------------------------------------

def test_monthly_periods_snap_and_cover():
    ps = ft.build_periods("2025-01-15", "2025-04-01", "monthly")
    assert [p.label for p in ps] == ["2025-01", "2025-02", "2025-03"]
    assert ps[0].start == date(2025, 1, 1) and ps[-1].end == date(2025, 4, 1)
    assert ps[1].days == 28


def test_weekly_periods_start_on_monday():
    ps = ft.build_periods("2026-01-01", "2026-01-15", "weekly")
    assert ps[0].start.weekday() == 0
    assert ps[0].label == "2026-W01"
    assert all(p.days == 7 for p in ps)


def test_annual_periods():
    ps = ft.build_periods("2024-06-01", "2026-01-01", "annual")
    assert [p.label for p in ps] == ["2024", "2025"]


def test_bad_inputs():
    with pytest.raises(ValueError):
        ft.build_periods("2025-01-01", "2024-01-01", "monthly")
    with pytest.raises(ValueError):
        ft.build_periods("2025-01-01", "2025-02-01", "daily")


def test_year_ago_periods():
    m = ft.Period("2025-03", date(2025, 3, 1), date(2025, 4, 1))
    assert ft.year_ago_period(m, "monthly").label == "2024-03"
    w = ft.build_periods("2026-03-02", "2026-03-03", "weekly")[0]
    prior = ft.year_ago_period(w, "weekly")
    assert prior.label == "2025-W10" and prior.start.weekday() == 0 and prior.days == 7
    # ISO week 53 of 2020 has no 2019 equivalent
    w53 = ft.Period("2020-W53", date(2020, 12, 28), date(2021, 1, 4))
    assert ft.year_ago_period(w53, "weekly") is None


def test_week_53_row_has_note():
    ps = [ft.Period("2020-W53", date(2020, 12, 28), date(2021, 1, 4))]
    rows = ft.build_summary(ft.empty_detections(), ps, ft.build_prior_map(ps, "weekly"), ft.SENSORS["modis"],
                            set(), date(2026, 1, 1))
    assert rows[0]["prior_period"] == "" and "week 53" in rows[0]["comparison_note"]


def test_prose_dates_not_zero_padded():
    assert ft.fmt_date(date(2023, 9, 3)) == "3 September 2023"


# --- sensor rule -----------------------------------------------------------

def _plan(start, end, freq, compare):
    ps = ft.build_periods(start, end, freq)
    return ps, (ft.build_prior_map(ps, freq) if compare else None)


def test_auto_uses_viirs_when_all_dates_covered():
    ps, pm = _plan("2025-01-01", "2026-01-01", "monthly", True)
    assert ft.choose_sensor("auto", ps, pm)[0] == "viirs"


def test_auto_falls_back_to_modis_when_comparison_predates_viirs():
    ps, pm = _plan("2024-01-01", "2025-01-01", "monthly", True)  # needs 2023-01
    key, reason = ft.choose_sensor("auto", ps, pm)
    assert key == "modis" and "MODIS" in reason


def test_auto_viirs_without_comparison_from_archive_start():
    ps, pm = _plan("2023-10-01", "2024-01-01", "monthly", False)
    assert ft.choose_sensor("auto", ps, pm)[0] == "viirs"


def test_september_2023_month_is_modis_because_archive_starts_on_the_3rd():
    ps, pm = _plan("2023-09-01", "2023-10-01", "monthly", False)
    assert ft.choose_sensor("auto", ps, pm)[0] == "modis"


def test_forced_viirs_too_early_errors():
    ps, pm = _plan("2024-01-01", "2024-02-01", "monthly", True)
    with pytest.raises(ValueError, match="VIIRS archive starts"):
        ft.choose_sensor("viirs", ps, pm)


def test_before_modis_errors():
    ps, pm = _plan("2000-01-01", "2000-02-01", "monthly", False)
    with pytest.raises(ValueError):
        ft.choose_sensor("auto", ps, pm)


# --- windows and chunks ----------------------------------------------------

def test_windows_merge_current_and_prior():
    ps, pm = _plan("2025-01-01", "2026-01-01", "annual", True)
    assert ft.fetch_windows(ps, pm) == [(date(2024, 1, 1), date(2026, 1, 1))]


def test_windows_stay_separate_when_gap():
    ps, pm = _plan("2025-06-01", "2025-07-01", "monthly", True)
    assert ft.fetch_windows(ps, pm) == [(date(2024, 6, 1), date(2024, 7, 1)), (date(2025, 6, 1), date(2025, 7, 1))]


def test_month_chunks_cut_weekly_window_at_month_boundary():
    chunks = ft.split_into_month_chunks([(date(2025, 1, 27), date(2025, 2, 10))])
    assert chunks == [(date(2025, 1, 27), date(2025, 2, 1)), (date(2025, 2, 1), date(2025, 2, 10))]


def test_fetch_with_split_halves_on_size_errors():
    calls = []

    def fetch(s, e):
        calls.append((s, e))
        if (e - s).days > 8:
            raise ft.SplittableFetchError("User memory limit exceeded")
        return pd.DataFrame({"date": [s.isoformat()], "longitude": [1.0], "latitude": [1.0]})

    out = ft.fetch_with_split(fetch, date(2025, 1, 1), date(2025, 2, 1), log=lambda m: None)
    assert len(out) == 4  # 31 days -> 15+16 -> 7+8+8+8
    assert sorted(out["date"]) == sorted({s.isoformat() for s, e in calls if (e - s).days <= 8})


def test_fetch_with_split_gives_up_at_one_day():
    def fetch(s, e):
        raise ft.SplittableFetchError("timed out")

    with pytest.raises(RuntimeError, match="single day"):
        ft.fetch_with_split(fetch, date(2025, 1, 1), date(2025, 1, 3), log=lambda m: None)


# --- detections and summaries -----------------------------------------------

def _det(rows):
    return ft.normalise_detections(pd.DataFrame(rows), ft.SENSORS["viirs"])


def test_normalise_renames_and_fills():
    df = _det([{"date": "2025-01-02", "longitude": 1, "latitude": 2, "confidence": 2, "frp": 5.5,
                "Bright_ti4": 330, "DayNight": 1, "acq_time": 3600, "geo": None}])
    assert list(df.columns) == ft.DETECTION_COLUMNS
    assert df.loc[0, "frp_mw"] == 5.5 and df.loc[0, "brightness_k"] == 330


def test_dedupe_keeps_most_confident():
    df = _det([
        {"date": "2025-01-02", "longitude": 1.0, "latitude": 2.0, "confidence": 1, "frp": 3},
        {"date": "2025-01-02", "longitude": 1.0, "latitude": 2.0, "confidence": 2, "frp": 1},
        {"date": "2025-01-03", "longitude": 1.0, "latitude": 2.0, "confidence": 1, "frp": 1},
    ])
    out = ft.dedupe_detections(df)
    assert len(out) == 2
    assert out[out["date"] == "2025-01-02"]["confidence"].iloc[0] == 2


def test_qa_flags():
    p = ft.Period("2025-03", date(2025, 3, 1), date(2025, 4, 1))
    assert ft.period_qa_flag(p, 31, date(2025, 6, 1)) == "ok"
    assert ft.period_qa_flag(p, 30, date(2025, 6, 1)) == "missing_days"
    assert ft.period_qa_flag(p, 0, date(2025, 6, 1)) == "no_imagery"
    assert ft.period_qa_flag(p, 20, date(2025, 3, 20)) == "partial_period"
    assert ft.period_qa_flag(p, 0, date(2025, 2, 1)) == "not_yet_available"


def test_change_values():
    assert ft.change_values(15, 10) == (5, 50.0)
    assert ft.change_values(3, 0) == (3, "")
    assert ft.change_values(3, None) == ("", "")


def test_summary_with_comparison():
    ps, pm = _plan("2025-03-01", "2025-05-01", "monthly", True)
    df = _det([
        {"date": "2025-03-05", "longitude": 1, "latitude": 1, "confidence": 2, "frp": 10, "DayNight": 1},
        {"date": "2025-03-06", "longitude": 1, "latitude": 1, "confidence": 1, "frp": 20, "DayNight": 0},
        {"date": "2024-03-10", "longitude": 1, "latitude": 1, "confidence": 1, "frp": 5, "DayNight": 1},
    ])
    imagery = {(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(800)}
    rows = ft.build_summary(df, ps, pm, ft.SENSORS["viirs"], imagery, date(2026, 1, 1))
    mar, apr = rows
    assert mar["fire_count"] == 2 and mar["prior_fire_count"] == 1
    assert mar["change_abs"] == 1 and mar["change_pct"] == 100.0
    assert mar["high_confidence_count"] == 1 and mar["day_count"] == 1 and mar["night_count"] == 1
    assert mar["frp_sum_mw"] == 30.0 and mar["frp_mean_mw"] == 15.0
    assert mar["qa_flag"] == "ok" and mar["period_end"] == "2025-03-31"
    assert apr["fire_count"] == 0 and apr["change_pct"] == ""


def test_point_tables_split_current_and_prior():
    ps, pm = _plan("2025-03-01", "2025-04-01", "monthly", True)
    df = _det([
        {"date": "2025-03-05", "longitude": 1, "latitude": 1, "confidence": 2, "frp": 10, "DayNight": 1, "acq_time": 4500},
        {"date": "2024-03-10", "longitude": 2, "latitude": 2, "confidence": 1, "frp": 5, "DayNight": 0, "acq_time": 0},
    ])
    cur, prior = ft.build_point_tables(df, ps, pm, ft.SENSORS["viirs"])
    assert len(cur) == 1 and cur.loc[0, "period"] == "2025-03" and cur.loc[0, "acq_time_utc"] == "01:15"
    assert cur.loc[0, "day_night"] == "day"
    assert len(prior) == 1 and prior.loc[0, "compares_with"] == "2025-03" and prior.loc[0, "period"] == "2024-03"


def test_layer_names_follow_convention():
    assert ft.points_layer_name("UKR", "viirs") == "ukr_evnt_fir_pt_s0_nasa_pp_viirs"
    assert ft.points_layer_name("ukr", "modis", prior=True) == "ukr_evnt_fir_pt_s0_nasa_pp_modis_prior_year"
    assert ft.output_filename("Kherson Oblast", "_summary.csv") == "kherson_oblast_fire_activity_summary.csv"


# --- end to end with a fake Earth Engine -----------------------------------

class FakeFetcher:
    """Two detections per day inside the toy box, one repeated (to test dedupe),
    data ending on a fixed latest date."""
    calls = []

    def __init__(self, sensor, aoi_geojson, min_conf):
        self.sensor = sensor
        self.latest = date(2025, 9, 20)

    def latest_data_date(self):
        return self.latest

    def imagery_dates(self, start, end):
        d, out = start, set()
        while d < end and d <= self.latest:
            out.add(d.isoformat())
            d += timedelta(days=1)
        return out

    def fetch(self, start, end):
        FakeFetcher.calls.append((start, end))
        rows = []
        d = start
        while d < end and d <= self.latest:
            for k, lon in enumerate((0.1, 0.2, 0.2)):
                rows.append({"date": d.isoformat(), "longitude": lon, "latitude": 0.1, "confidence": 1 + (k == 0),
                             "frp": 2.0, "Bright_ti4": 320, "DayNight": k % 2, "acq_time": 7200})
            d += timedelta(days=1)
        return ft.normalise_detections(pd.DataFrame(rows), self.sensor)


def _args(tmp_path, **kw):
    base = dict(aoi_file=str(SAMPLE_AOI), geoextent="tst", start="2025-08-01", end="2025-10-01", freq="monthly",
                compare_prior_year=True, sensor="auto", min_confidence=None, out_dir=str(tmp_path),
                ee_project=None, simplify_tolerance=None, no_cache=False, dry_run=False, no_burned_area=True)
    base.update(kw)
    return Namespace(**base)


def test_end_to_end_outputs(tmp_path):
    import geopandas as gpd
    FakeFetcher.calls = []
    assert ft.run(_args(tmp_path), fetcher_factory=FakeFetcher, log=lambda m: None) == 0

    summary = pd.read_csv(tmp_path / "tst_fire_activity_summary.csv")
    aug, sep = summary.to_dict("records")
    assert aug["sensor"] == "viirs" and aug["fire_count"] == 31 * 2 and aug["prior_fire_count"] == 31 * 2
    assert aug["qa_flag"] == "ok" and aug["change_pct"] == 0.0
    assert sep["qa_flag"] == "partial_period" and sep["fire_count"] == 20 * 2
    # partial month compared like-for-like with the same 20 days a year earlier
    assert sep["prior_fire_count"] == 20 * 2 and sep["change_abs"] == 0
    assert sep["comparison_note"] == "first 20 days of both periods only"

    layers = gpd.list_layers(tmp_path / "tst_fire_activity.gpkg")["name"].tolist()
    assert layers[:2] == ["tst_evnt_fir_pt_s0_nasa_pp_viirs", "tst_evnt_fir_pt_s0_nasa_pp_viirs_prior_year"]
    assert [n for n in layers if "_hex" in n] == ["tst_evnt_fir_py_s0_nasa_pp_viirs_hex2km_total",
                                                   "tst_evnt_fir_py_s0_nasa_pp_viirs_hex2km_by_period"]
    pts = gpd.read_file(tmp_path / "tst_fire_activity.gpkg", layer=layers[0])
    assert len(pts) == (31 + 20) * 2
    assert pts.crs.to_epsg() == 4326 and set(pts["period"]) == {"2025-08", "2025-09"}
    info = (tmp_path / "tst_fire_activity_run_info.txt").read_text()
    assert "VIIRS" in info and "September 2025 partial_period" not in info and "2025-09 partial_period" in info

    # Second run uses the cache for complete chunks and only re-fetches September 2025
    FakeFetcher.calls = []
    ft.run(_args(tmp_path), fetcher_factory=FakeFetcher, log=lambda m: None)
    assert FakeFetcher.calls == [(date(2025, 9, 1), date(2025, 10, 1))]


def test_end_to_end_no_fires_writes_empty_layers(tmp_path):
    import geopandas as gpd

    class Empty(FakeFetcher):
        def fetch(self, start, end):
            return ft.empty_detections()

    ft.run(_args(tmp_path, compare_prior_year=False, no_cache=True), fetcher_factory=Empty, log=lambda m: None)
    s = pd.read_csv(tmp_path / "tst_fire_activity_summary.csv")
    assert s["fire_count"].tolist() == [0, 0] and "prior_fire_count" not in s.columns
    pts = gpd.read_file(tmp_path / "tst_fire_activity.gpkg")
    assert len(pts) == 0


def test_modis_run_has_blank_viirs_only_columns(tmp_path):
    class Modis(FakeFetcher):
        def fetch(self, start, end):
            df = super().fetch(start, end)
            df["confidence"] = 50
            return df

    ft.run(_args(tmp_path, start="2024-01-01", end="2024-02-01"), fetcher_factory=Modis, log=lambda m: None)
    s = pd.read_csv(tmp_path / "tst_fire_activity_summary.csv")
    assert s.loc[0, "sensor"] == "modis" and pd.isna(s.loc[0, "frp_sum_mw"]) and pd.isna(s.loc[0, "day_count"])


def test_dry_run_needs_no_fetcher(tmp_path):
    msgs = []
    assert ft.run(_args(tmp_path, dry_run=True), fetcher_factory=None, log=msgs.append) == 0
    assert any("Dry run" in m for m in msgs) and any("vs 2024-08" in m for m in msgs)


def test_cli_reports_missing_args():
    with pytest.raises(SystemExit):
        ft.main(["--start", "2025-01-01"])


def test_projected_shapefile_is_reprojected(tmp_path):
    import geopandas as gpd
    shp = tmp_path / "aoi_utm.shp"
    g = gpd.read_file(SAMPLE_AOI).to_crs(epsg=32629)
    g = pd.concat([g, g.translate(1000, 0).to_frame("geometry").set_crs(epsg=32629)], ignore_index=True)
    gpd.GeoDataFrame(g, geometry="geometry", crs=32629).to_file(shp)
    aoi = ft.load_aoi(str(shp))
    minx, miny, maxx, maxy = aoi.geometry.bounds
    assert -8.31 < minx < -8.29 and 39.79 < miny < 39.81 and aoi.feature_count == 2
    assert 2000 < aoi.area_km2 < 3000 and "32629" in aoi.source_crs


def test_detailed_boundary_is_simplified_automatically(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point
    big = Point(-8, 40).buffer(1.0, quad_segs=40000)  # ~160,000 vertices
    path = tmp_path / "big.geojson"
    gpd.GeoDataFrame(geometry=[big], crs=4326).to_file(path)
    aoi = ft.load_aoi(str(path))
    assert aoi.simplified_deg is not None and "simplified automatically" in aoi.note


# --- review fixes ------------------------------------------------------------

def test_transient_errors_are_retried_then_succeed():
    attempts = []

    def fetch(s, e):
        attempts.append(1)
        if len(attempts) < 3:
            raise ft.TransientFetchError("Too many concurrent aggregations")
        return ft.empty_detections()

    out = ft._fetch_with_retry(fetch, date(2025, 1, 1), date(2025, 2, 1), log=lambda m: None,
                               waits=(0, 0, 0), sleep=lambda s: None)
    assert out.empty and len(attempts) == 3


def test_transient_errors_give_up_with_clear_message():
    def fetch(s, e):
        raise ft.TransientFetchError("429")

    with pytest.raises(RuntimeError, match="kept refusing"):
        ft._fetch_with_retry(fetch, date(2025, 1, 1), date(2025, 2, 1), log=lambda m: None,
                             waits=(0,), sleep=lambda s: None)


def test_cache_rule():
    latest = date(2025, 9, 20)
    full = {(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(400)}
    assert ft.chunk_is_cacheable(date(2025, 8, 1), date(2025, 9, 1), latest, full)
    assert not ft.chunk_is_cacheable(date(2025, 9, 1), date(2025, 9, 15), latest, full)  # too recent
    gap = full - {"2025-08-15"}
    assert not ft.chunk_is_cacheable(date(2025, 8, 1), date(2025, 9, 1), latest, gap)  # missing day
    assert not ft.chunk_is_cacheable(date(2025, 8, 1), date(2025, 9, 1), None, full)


def test_prior_layer_marks_like_for_like_window():
    ps, pm = _plan("2025-09-01", "2025-10-01", "monthly", True)
    df = _det([
        {"date": "2024-09-05", "longitude": 1, "latitude": 1, "confidence": 1},
        {"date": "2024-09-25", "longitude": 1, "latitude": 1, "confidence": 1},
    ])
    _, prior = ft.build_point_tables(df, ps, pm, ft.SENSORS["viirs"], latest_data_date=date(2025, 9, 20))
    assert prior.set_index("date")["in_comparison"].to_dict() == {"2024-09-05": "yes", "2024-09-25": "no"}


def test_locked_geopackage_gives_plain_message(tmp_path, monkeypatch):
    def locked(src, dst):
        raise PermissionError("in use")
    monkeypatch.setattr(ft.os, "replace", locked)
    with pytest.raises(RuntimeError, match="open in ArcGIS Pro"):
        ft.write_gpkg(tmp_path / "x.gpkg", [("layer", _det([{"date": "2025-01-01", "longitude": 1,
                                                                       "latitude": 1, "confidence": 1}]))])


def test_scotland_sample_boundary_if_present():
    # Natural Earth 1:10m map unit (public domain); small enough to need no simplifying.
    shp = Path(__file__).resolve().parents[1] / "sample_aoi" / "scotland_admn_ad0_py_s0_naturalearthdata_pp.shp"
    if not shp.exists():
        pytest.skip("Scotland sample not present")
    aoi = ft.load_aoi(str(shp))
    assert 70000 < aoi.area_km2 < 90000
    assert len(__import__("json").dumps(aoi.geojson)) <= ft.MAX_AOI_GEOJSON_BYTES
