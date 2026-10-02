"""Regression tests for the Phase 5 whole-tool review (2 October 2026).

Run from the tool folder:  python -m pytest tests -q
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fire_charts as fc  # noqa: E402
import fire_tool as ft  # noqa: E402
import fire_wizard as fw  # noqa: E402
from test_fire_tool import FakeFetcher, _args, _det, _plan  # noqa: E402


def _days(start: date, n: int) -> set[str]:
    return {(start + timedelta(days=i)).isoformat() for i in range(n)}


# --- 1. comparisons with missing imagery are flagged -----------------------------

def _july(imagery, latest=date(2026, 1, 1)):
    ps, pm = _plan("2025-07-01", "2025-08-01", "monthly", True)
    df = _det([{"date": "2025-07-05", "longitude": 1, "latitude": 1, "confidence": 1},
               {"date": "2024-07-05", "longitude": 1, "latitude": 1, "confidence": 1}])
    return ft.build_summary(df, ps, pm, ft.SENSORS["viirs"], imagery, latest)[0]


def test_year_earlier_gap_is_noted():
    imagery = _days(date(2024, 7, 1), 22) | _days(date(2025, 7, 1), 31)
    row = _july(imagery)
    assert row["prior_qa_flag"] == "missing_days" and row["prior_days_with_imagery"] == 22
    assert "year-earlier period has imagery on only 22 of 31 days" in row["comparison_note"]
    assert row["change_abs"] == 0          # counts are warned about, never scaled


def test_current_gap_is_noted():
    imagery = _days(date(2024, 7, 1), 31) | _days(date(2025, 7, 1), 26)
    row = _july(imagery)
    assert row["qa_flag"] == "missing_days"
    assert "this period has imagery on only 26 of 31 days" in row["comparison_note"]


def test_no_year_earlier_imagery_means_no_comparison():
    row = _july(_days(date(2025, 7, 1), 31))
    assert row["prior_qa_flag"] == "no_imagery"
    assert row["prior_fire_count"] == "" and row["change_abs"] == "" and row["change_pct"] == ""
    assert "no satellite imagery in the year-earlier period" in row["comparison_note"]


def test_full_imagery_leaves_note_blank():
    row = _july(_days(date(2024, 7, 1), 31) | _days(date(2025, 7, 1), 31))
    assert row["comparison_note"] == ""


def test_partial_period_keeps_its_note_and_adds_the_gap():
    imagery = _days(date(2024, 7, 1), 31) - {"2024-07-03"} | _days(date(2025, 7, 1), 10)
    row = _july(imagery, latest=date(2025, 7, 10))
    assert row["qa_flag"] == "partial_period"
    assert row["comparison_note"].startswith("first 10 days of both periods only; year-earlier period has "
                                             "imagery on only 9 of 10 days")


class GappyFetcher(FakeFetcher):
    """No imagery on 3 to 7 August 2024 (and so no fires those days)."""
    gap = _days(date(2024, 8, 3), 5)

    def imagery_dates(self, start, end):
        return super().imagery_dates(start, end) - self.gap

    def fetch(self, start, end):
        df = super().fetch(start, end)
        return df[~df["date"].isin(self.gap)].reset_index(drop=True)


def test_run_info_lists_uneven_comparisons(tmp_path):
    assert ft.run(_args(tmp_path), fetcher_factory=GappyFetcher, log=lambda m: None) == 0
    summary = pd.read_csv(tmp_path / "tst_fire_activity_summary.csv", keep_default_na=False)
    aug = summary.set_index("period").loc["2025-08"]
    assert "26 of 31 days" in aug["comparison_note"]
    info = (tmp_path / "tst_fire_activity_run_info.txt").read_text(encoding="utf-8")
    assert "Comparisons affected by missing imagery in either year: 1 (2025-08)" in info


def test_chart_marks_uneven_year_earlier_periods(tmp_path, monkeypatch):
    feet = []
    real_frame = fc._frame

    def spy(fig, title, subtitle, foot_lines, *a, **k):
        feet.append(list(foot_lines))
        return real_frame(fig, title, subtitle, foot_lines, *a, **k)
    monkeypatch.setattr(fc, "_frame", spy)
    rows = [{"period": f"2025-0{m}", "period_start": f"2025-0{m}-01", "period_end": f"2025-0{m}-28",
             "fire_count": 10, "prior_fire_count": 5, "qa_flag": "ok", "prior_qa_flag": "ok"} for m in (7, 8)]
    rows[0]["prior_qa_flag"] = "missing_days"
    fc.write_charts(rows, tmp_path, "tst", freq="monthly", compare=True, area_name="Test",
                    sensor_label="VIIRS", sensor_source="NASA", burned_on=False)
    assert any(line.startswith(fc.PRIOR_MARK + " the year-earlier period has days with no satellite imagery")
               for line in feet[0])
    # Without a comparison there is no year-earlier line, so no dagger either.
    feet.clear()
    fc.write_charts(rows, tmp_path, "tst", freq="monthly", compare=False, area_name="Test",
                    sensor_label="VIIRS", sensor_source="NASA", burned_on=False)
    assert not any(line.startswith(fc.PRIOR_MARK) for line in feet[0])


# --- 2 and 3. outputs open in another program --------------------------------

class NoDownload(FakeFetcher):
    def __init__(self, *a):
        raise AssertionError("nothing should be downloaded while outputs are locked")


def test_locked_output_stops_the_run_before_download_and_changes_nothing(tmp_path, monkeypatch):
    assert ft.run(_args(tmp_path), fetcher_factory=FakeFetcher, log=lambda m: None) == 0
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir() if p.is_file()}
    real_rename = ft.os.rename

    def excel_has_it(src, dst):
        if str(src).endswith("_summary.csv"):
            raise PermissionError("used by another process")
        return real_rename(src, dst)
    monkeypatch.setattr(ft.os, "rename", excel_has_it)
    with pytest.raises(RuntimeError, match="open in another program: tst_fire_activity_summary.csv.*Nothing has"):
        ft.run(_args(tmp_path), fetcher_factory=NoDownload, log=lambda m: None)
    after = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir() if p.is_file()}
    assert after == before


def test_lock_check_leaves_files_where_they_were(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x")
    assert ft.locked_files([f]) == [] and f.read_text() == "x"
    assert [p.name for p in tmp_path.iterdir()] == ["a.csv"]


def test_csv_open_in_excel_gives_a_plain_message(tmp_path, monkeypatch):
    def locked(src, dst):
        raise PermissionError("in use")
    monkeypatch.setattr(ft.os, "replace", locked)
    with pytest.raises(RuntimeError, match="open in Excel"):
        ft.write_csv_atomic(pd.DataFrame({"a": [1]}), tmp_path / "x.csv")


# --- 6. permanent imagery gaps do not stop caching --------------------------------

def test_old_months_with_gaps_are_cached():
    latest = date(2026, 10, 1)
    gap = _days(date(2024, 1, 1), 1000) - {"2024-07-10"}
    assert ft.chunk_is_cacheable(date(2024, 7, 1), date(2024, 8, 1), latest, gap)       # long settled
    recent_gap = _days(date(2026, 1, 1), 300) - {"2026-08-10"}
    assert not ft.chunk_is_cacheable(date(2026, 8, 1), date(2026, 9, 1), latest, recent_gap)


# --- 7. paths pasted with quotes ---------------------------------------------------

def test_quoted_paths_are_cleaned():
    assert fw.clean_path('"H:\\data\\esp.shp"') == "H:\\data\\esp.shp"
    assert fw.clean_path("  'out folder'  ") == "out folder"
    argv = fw.build_argv_from_form({"aoi_file": '"H:\\data\\esp.shp"', "geoextent": "esp", "start": "2026-01-01",
                                    "end": "2026-03-01", "freq": "monthly", "compare": True,
                                    "out_dir": '"H:\\out"', "ee_project": "my-ee-project"})
    assert argv[argv.index("--aoi-file") + 1] == "H:\\data\\esp.shp"
    assert argv[argv.index("--out-dir") + 1] == "H:\\out"


def test_window_accepts_a_quoted_pasted_path():
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:  # noqa: BLE001
        pytest.skip("no display for Tkinter")
    root.withdraw()
    try:
        import fire_gui
        aoi = Path(__file__).resolve().parents[1] / "sample_aoi" / "toy_bbox.geojson"
        g = fire_gui.FireGUI(root, tool_main=lambda argv: 0)
        g.aoi_file.set(f'"{aoi}"')                  # as pasted from Explorer's Copy as path
        g.geoextent.set("tst")
        g.start.set("2025-08-01")
        g.end.set("2025-10-01")
        a = ft.build_arg_parser().parse_args(g.build_argv(dry_run=True))
        assert a.aoi_file == str(aoi) and g.aoi_file.get() == str(aoi)
        assert not g.aoi_info.get().startswith("Couldn't read")
    finally:
        root.destroy()


# --- re-review fixes -----------------------------------------------------------------

def test_february_and_leap_years_are_not_flagged_as_missing_imagery():
    imagery = _days(date(2023, 1, 1), 1200)          # every day has imagery
    for start, end, freq in (("2025-02-01", "2025-03-01", "monthly"), ("2024-02-01", "2024-03-01", "monthly"),
                             ("2025-01-01", "2026-01-01", "annual"), ("2024-01-01", "2025-01-01", "annual")):
        ps, pm = _plan(start, end, freq, True)
        row = ft.build_summary(ft.empty_detections(), ps, pm, ft.SENSORS["viirs"], imagery, date(2026, 6, 1))[0]
        assert row["comparison_note"] == "", (start, row["comparison_note"])


def test_current_period_without_imagery_gets_no_change():
    row = _july(_days(date(2024, 7, 1), 31))           # nothing in July 2025
    assert row["qa_flag"] == "no_imagery"
    assert row["prior_fire_count"] == "" and row["change_abs"] == "" and row["change_pct"] == ""
    assert row["comparison_note"] == "no satellite imagery in this period, so no comparison"
    assert ft.uneven_comparisons([row]) == ["2025-07"]


def test_left_over_lock_probe_is_put_back_and_not_reported(tmp_path):
    assert ft.run(_args(tmp_path), fetcher_factory=FakeFetcher, log=lambda m: None) == 0
    csv = tmp_path / "tst_fire_activity_summary.csv"
    text = csv.read_text(encoding="utf-8-sig")
    # An interrupted check left the summary renamed.
    csv.rename(tmp_path / f"{csv.name}.999{ft.LOCK_PROBE_SUFFIX}")
    ft.check_outputs_unlocked(tmp_path, "tst")
    assert csv.read_text(encoding="utf-8-sig") == text
    assert not list(tmp_path.glob("*" + ft.LOCK_PROBE_SUFFIX))
    # A probe whose file was written again since is just tidied away.
    (tmp_path / f"{csv.name}.998{ft.LOCK_PROBE_SUFFIX}").write_text("old")
    ft.check_outputs_unlocked(tmp_path, "tst")
    assert csv.read_text(encoding="utf-8-sig") == text and not list(tmp_path.glob("*" + ft.LOCK_PROBE_SUFFIX))


def test_chart_open_in_a_viewer_does_not_stop_the_run(tmp_path, monkeypatch):
    assert ft.run(_args(tmp_path), fetcher_factory=FakeFetcher, log=lambda m: None) == 0
    real_rename = ft.os.rename

    def viewer_has_it(src, dst):
        if str(src).endswith(".png"):
            raise PermissionError("used by another process")
        return real_rename(src, dst)
    monkeypatch.setattr(ft.os, "rename", viewer_has_it)
    assert ft.run(_args(tmp_path), fetcher_factory=FakeFetcher, log=lambda m: None) == 0


def test_period_labels_cope_with_a_missing_date():
    ps = ft.build_periods("2025-01-01", "2025-03-01", "monthly")
    s = pd.Series(["2025-01-05", None, "2025-02-10", "2025-04-01"], index=[7, 3, 9, 1], dtype=object)
    out = ft.assign_period_labels(s, ps)
    assert out.to_dict() == {7: "2025-01", 3: "", 9: "2025-02", 1: ""}


def test_chart_and_burned_area_use_the_same_low_coverage_rule():
    import fire_burned
    assert fc.LOW_COVERAGE_PCT == fire_burned.LOW_COVERAGE_PCT


# --- calendar picker (2 October 2026) -------------------------------------------------

def test_date_boxes_are_calendar_pickers_or_say_how_to_get_one(monkeypatch):
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:  # noqa: BLE001
        pytest.skip("no display for Tkinter")
    root.withdraw()
    try:
        import fire_gui
        g = fire_gui.FireGUI(root, tool_main=lambda argv: 0)
        log = g.log.get("1.0", "end")
        if fire_gui.DateEntry is not None:
            assert "calendar picker is not installed" not in log
            assert g.start.get() == "" and g.end.get() == ""     # empty until a date is chosen
        else:
            assert "pip install tkcalendar" in log
        # Without the package the window still opens, with typed boxes and the install hint.
        monkeypatch.setattr(fire_gui, "DateEntry", None)
        g2 = fire_gui.FireGUI(tk.Toplevel(root), tool_main=lambda argv: 0)
        assert "pip install tkcalendar" in g2.log.get("1.0", "end")
    finally:
        root.destroy()


def test_calendar_package_is_a_requirement():
    req = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
    assert any(line.startswith("tkcalendar") for line in req.splitlines())


# --- shareable: no personal Earth Engine project built in (2 October 2026) ------------

def test_no_earth_engine_project_is_built_in():
    assert fw.DEFAULT_EE_PROJECT == ""
    a = ft.build_arg_parser().parse_args(["--aoi-file", "a.shp", "--dry-run", "--start", "2026-01-01",
                                          "--end", "2026-02-01", "--freq", "monthly"])
    assert a.ee_project is None
    form = {"aoi_file": "a.shp", "geoextent": "x", "start": "2026-01-01", "end": "2026-02-01",
            "freq": "monthly", "compare": True, "out_dir": "out", "ee_project": ""}
    assert "--ee-project" not in fw.build_argv_from_form(form)


def test_chart_footer_names_only_the_tool(tmp_path, monkeypatch):
    feet = []
    real_frame = fc._frame

    def spy(fig, title, subtitle, foot_lines, *a, **k):
        feet.append(list(foot_lines))
        return real_frame(fig, title, subtitle, foot_lines, *a, **k)
    monkeypatch.setattr(fc, "_frame", spy)
    rows = [{"period": "2025-07", "period_start": "2025-07-01", "period_end": "2025-07-31", "fire_count": 3,
             "prior_fire_count": 2, "qa_flag": "ok", "prior_qa_flag": "ok"}]
    fc.write_charts(rows, tmp_path, "tst", freq="monthly", compare=True, area_name="Test",
                    sensor_label="VIIRS", sensor_source="NASA", burned_on=False)
    assert feet[0][-1] == "Chart: fire activity tool. Area: Test."


# --- shapefile output (--vector-format, 2 October 2026) --------------------------------

def test_shapefile_field_names_are_short_unique_and_stable():
    cols = ["date", "prior_burned_area_km2", "prior_burned_mapped_pct", "burned_change_km2", "burned_change_pct",
            "prior_fire_count", "prior_fires_per_1000_km2", "geometry", "some_brand_new_long_field",
            "some_brand_new_long_other"]
    m = ft.shapefile_field_names(cols)
    assert "geometry" not in m
    assert all(len(v) <= 10 for v in m.values())
    assert len({v.lower() for v in m.values()}) == len(m)          # unique, ignoring case
    assert m["prior_burned_area_km2"] == "pr_burn_km" and m["date"] == "date"
    assert m == ft.shapefile_field_names(cols)                      # same names every run
    assert all(len(v) <= 10 for v in ft.SHP_FIELD_NAMES.values())
    assert len(set(ft.SHP_FIELD_NAMES.values())) == len(ft.SHP_FIELD_NAMES)


def _burned_run(tmp_path, **kw):
    from test_fire_burned import _admin_file, _run
    tmp_path.mkdir(parents=True, exist_ok=True)
    admin = tmp_path / "admin.gpkg" if (tmp_path / "admin.gpkg").exists() else _admin_file(tmp_path)
    return _run(tmp_path, admin_file=str(admin), unit_id_field="ADM2_PCODE", **kw)


def test_shapefiles_hold_the_same_layers_as_the_geopackage(tmp_path):
    import geopandas as gpd
    import pyogrio
    assert _burned_run(tmp_path / "both", vector_format="both") == 0
    out = tmp_path / "both"
    gpkg = out / "prt_fire_activity.gpkg"
    shp_dir = out / "prt_fire_activity_shp"
    layers = [n for n, _ in pyogrio.list_layers(gpkg)]
    assert sorted(p.stem for p in shp_dir.glob("*.shp")) == sorted(layers)
    key = pd.read_csv(shp_dir / "prt_fire_activity_shp_fields.csv", keep_default_na=False)
    full = dict(zip(key["shapefile_field"], key["full_name"]))
    for name in layers:
        g = gpd.read_file(gpkg, layer=name)
        s = gpd.read_file(shp_dir / f"{name}.shp")
        assert len(g) == len(s) and s.crs is not None
        assert all(len(c) <= 10 for c in s.columns)
        assert sorted(full.get(c, c) for c in s.columns) == sorted(g.columns)
        for num in ("fire_count", "burned_pixels"):
            short = next((k for k, v in full.items() if v == num), num)
            if num in g.columns:
                assert g[num].fillna(0).sum() == s[short].fillna(0).sum()
    assert (shp_dir / "prt_evnt_fir_pt_s0_nasa_pp_viirs.cpg").read_text().strip().upper().replace("-", "") == "UTF8"
    # Dates are real date fields (dBASE type D), so ArcGIS Pro's time slider can use them.
    raw = (shp_dir / "prt_evnt_fir_py_s0_nasa_pp_viirs_hex2km_by_period.dbf").read_bytes()
    types, i = {}, 32
    while raw[i] != 0x0D:
        types[raw[i:i + 11].split(b"\0")[0].decode()] = chr(raw[i + 11])
        i += 32
    assert types["per_start"] == "D" and types["period_end"] == "D"
    pts = gpd.read_file(shp_dir / "prt_evnt_fir_pt_s0_nasa_pp_viirs.shp")
    pts_g = gpd.read_file(gpkg, layer="prt_evnt_fir_pt_s0_nasa_pp_viirs")
    assert sorted(pd.to_datetime(pts["date"]).dt.date) == sorted(pd.to_datetime(pts_g["date"]).dt.date)
    info = (out / "prt_fire_activity_run_info.txt").read_text(encoding="utf-8")
    assert "prt_fire_activity_shp/ - 6 shapefile(s)" in info and "prt_fire_activity.gpkg - layers" in info


def test_switching_vector_format_clears_the_other_format(tmp_path):
    out = tmp_path / "run"
    _burned_run(out, vector_format="both")
    shp_dir = out / "prt_fire_activity_shp"
    (shp_dir / "my_notes.txt").write_text("keep me")
    _burned_run(out, vector_format="gpkg")                         # shapefiles go, the user's file stays
    assert [p.name for p in shp_dir.iterdir()] == ["my_notes.txt"]
    assert (out / "prt_fire_activity.gpkg").exists()
    _burned_run(out, vector_format="shp")                          # GeoPackage goes
    assert not (out / "prt_fire_activity.gpkg").exists()
    assert len(list(shp_dir.glob("*.shp"))) == 6
    info = (out / "prt_fire_activity_run_info.txt").read_text(encoding="utf-8")
    assert ".gpkg - layers" not in info


def test_open_shapefile_stops_the_run_before_download(tmp_path, monkeypatch):
    out = tmp_path / "run"
    _burned_run(out, vector_format="shp")
    real_rename = ft.os.rename

    def arcgis_has_it(src, dst):
        if str(src).endswith(".dbf"):
            raise PermissionError("in use")
        return real_rename(src, dst)
    monkeypatch.setattr(ft.os, "rename", arcgis_has_it)
    with pytest.raises(RuntimeError, match=r"open in another program: .*\.dbf"):
        ft.run(_args(out, geoextent="prt", vector_format="shp"), fetcher_factory=NoDownload, log=lambda m: None)


def test_vector_format_reaches_the_options_from_form_and_window():
    base = {"aoi_file": "a.shp", "geoextent": "x", "start": "2026-01-01", "end": "2026-02-01",
            "freq": "monthly", "compare": True, "out_dir": "out"}
    assert "--vector-format" not in fw.build_argv_from_form(base)                   # GeoPackage is the default
    argv = fw.build_argv_from_form({**base, "vector_format": "both"})
    assert ft.build_arg_parser().parse_args(argv).vector_format == "both"
    with pytest.raises(ValueError, match="Vector output"):
        fw.build_argv_from_form({**base, "vector_format": "kml"})
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:  # noqa: BLE001
        pytest.skip("no display for Tkinter")
    root.withdraw()
    try:
        import fire_gui
        g = fire_gui.FireGUI(root, tool_main=lambda a: 0)
        assert g.vector_format.get() == "GeoPackage (one file)"
        g.vector_format.set("Shapefiles (one folder)")
        assert g.collect_fields()["vector_format"] == "shp"
    finally:
        root.destroy()
