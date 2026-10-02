"""Offline tests for Phase 4: charts (fire_charts.py), wizard and form builder
(fire_wizard.py) and the GUI's form handling (fire_gui.py, skipped without a
display).

Run from the tool folder:  python -m pytest tests -q
"""
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fire_charts as fc  # noqa: E402
import fire_tool as ft  # noqa: E402
import fire_wizard as fw  # noqa: E402
from test_fire_burned import FakeBurnedFetcher, FakeFireFetcher, _admin_file, _args  # noqa: E402

SCOTLAND = ROOT / "sample_aoi" / "scotland_admn_ad0_py_s0_naturalearthdata_pp.shp"
TOY = ROOT / "sample_aoi" / "toy_bbox.geojson"


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def test_num_handles_blanks_and_strings():
    assert fc.num("") is None and fc.num(None) is None and fc.num(float("nan")) is None
    assert fc.num("12.5") == 12.5 and fc.num(3) == 3.0 and fc.num("abc") is None


def test_infer_freq():
    assert fc.infer_freq(["2026-03", "2026-04"]) == "monthly"
    assert fc.infer_freq(["2026-W01", "2026-W02"]) == "weekly"
    assert fc.infer_freq(["2025", "2026"]) == "annual"
    with pytest.raises(ValueError):
        fc.infer_freq(["2026-03", "2026"])


def test_month_groups_put_weeks_in_the_month_holding_their_thursday():
    # ISO week 1 of 2026 starts Monday 29 December 2025; its Thursday is 1 January.
    starts = [date(2025, 12, 29), date(2026, 1, 5), date(2026, 1, 26), date(2026, 2, 2)]
    groups = fc.month_groups(starts, "weekly")
    assert groups == [(0, 2, date(2026, 1, 1)), (3, 3, date(2026, 2, 1))]
    assert fc.year_groups(starts, "weekly") == [(0, 3, 2026)]


def test_pick_month_style_prefers_full_names_then_abbreviations_then_letters():
    measure = lambda s: 7.0 * len(s)  # noqa: E731
    assert fc.pick_month_style([200, 200], measure) == "full"      # September = 63 px
    assert fc.pick_month_style([40, 40], measure) == "abbr"        # Sep = 21 px
    assert fc.pick_month_style([20, 200], measure) == "letter"     # narrowest slot decides


def test_label_anchor_moves_above_the_year_earlier_line_only_when_it_would_cross_the_label():
    # Line far above the column: label stays on the column.
    assert fc.label_anchors([10], [100], px_per_value=2, gap_px=4, label_h_px=12) == [10]
    # Line just above the column (inside the label): label goes above the line.
    assert fc.label_anchors([10], [14], px_per_value=2, gap_px=4, label_h_px=12) == [14]
    # Line below the column top: label stays on the column.
    assert fc.label_anchors([10], [5], px_per_value=2, gap_px=4, label_h_px=12) == [10]
    assert fc.label_anchors([10], [None], px_per_value=2, gap_px=4, label_h_px=12) == [10]


def test_area_name_from_file():
    assert fc.area_name_from_file(str(SCOTLAND), "SCT") == "Scotland"
    assert fc.area_name_from_file(str(TOY), "TOY") == "toy_bbox_central_portugal"   # its name field
    assert fc.area_name_from_file("no/such/file.shp", "X") == "X"


def _rows(n=6, nya_from=None, flag_at=None):
    rows = []
    for i in range(n):
        m = 3 + i
        rows.append({"period": f"2026-{m:02d}", "period_start": f"2026-{m:02d}-01", "period_end": f"2026-{m:02d}-28",
                     "fire_count": 10 * i, "prior_fire_count": 5 * i, "qa_flag": "ok",
                     "burned_area_km2": 1.5 * i, "prior_burned_area_km2": 2.0, "burned_qa_flag": "ok"})
    if nya_from is not None:
        for r in rows[nya_from:]:
            r["burned_area_km2"] = ""
            r["prior_burned_area_km2"] = ""
            r["burned_qa_flag"] = "not_yet_available"
    if flag_at is not None:
        rows[flag_at]["qa_flag"] = "missing_days"
    return rows


def _png_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def test_write_charts_writes_detections_burned_and_admin(tmp_path):
    admin = [{"period": "all", "unit_id": str(i), "unit_name": f"Area {i}", "fire_count": i * 3,
              "prior_fire_count": 2} for i in range(20)]
    paths = fc.write_charts(_rows(nya_from=4, flag_at=1), tmp_path, "sct", freq="monthly", compare=True,
                            area_name="Scotland", sensor_label="VIIRS 375 m", sensor_source="NASA LANCE VIIRS",
                            burned_on=True, admin_rows=admin, log=lambda m: None)
    names = sorted(p.name for p in paths)
    assert names == ["sct_fire_activity_chart_admin.png", "sct_fire_activity_chart_burned.png",
                     "sct_fire_activity_chart_detections.png"]
    for p in paths:
        w, h = _png_size(p)
        assert w >= 1600 and h >= 800
    assert not list(tmp_path.glob("*.tmp.png"))


def test_no_burned_chart_when_nothing_published_and_no_admin_chart_for_one_area(tmp_path):
    msgs = []
    paths = fc.write_charts(_rows(nya_from=0), tmp_path, "sct", freq="monthly", compare=False,
                            area_name="Scotland", sensor_label="VIIRS", sensor_source="NASA", burned_on=True,
                            admin_rows=[{"period": "all", "unit_id": "1", "unit_name": "Only", "fire_count": 5}],
                            log=msgs.append)
    assert [p.name for p in paths] == ["sct_fire_activity_chart_detections.png"]
    assert any("no burned area chart" in m for m in msgs)


def test_old_charts_are_cleared(tmp_path):
    for kind in ("detections", "burned", "admin"):
        (tmp_path / fc.chart_file_name("sct", kind)).write_bytes(b"old")
    fc.write_charts(_rows(), tmp_path, "sct", freq="monthly", compare=True, area_name="S", sensor_label="V",
                    sensor_source="N", burned_on=False, admin_rows=None, log=lambda m: None)
    assert not (tmp_path / fc.chart_file_name("sct", "burned")).exists()
    assert not (tmp_path / fc.chart_file_name("sct", "admin")).exists()
    assert (tmp_path / fc.chart_file_name("sct", "detections")).read_bytes() != b"old"


def test_many_weekly_periods_still_draw(tmp_path):
    from datetime import timedelta
    rows = []
    d = date(2024, 9, 2)
    for i in range(110):
        s = d + timedelta(weeks=i)
        iso = s.isocalendar()
        rows.append({"period": f"{iso[0]}-W{iso[1]:02d}", "period_start": s.isoformat(),
                     "period_end": (s + timedelta(days=6)).isoformat(), "fire_count": 1000 + i, "qa_flag": "ok"})
    paths = fc.write_charts(rows, tmp_path, "w", freq="weekly", compare=False, area_name="S", sensor_label="V",
                            sensor_source="N", burned_on=False, log=lambda m: None)
    assert paths[0].exists()


def test_charts_from_csv_redraws_from_a_summary(tmp_path):
    pd.DataFrame(_rows(nya_from=5)).assign(sensor="viirs").to_csv(
        tmp_path / "sct_fire_activity_summary.csv", index=False, encoding="utf-8-sig")
    paths = fc.charts_from_csv(str(tmp_path / "sct_fire_activity_summary.csv"), "Scotland", log=lambda m: None)
    assert {p.name for p in paths} == {"sct_fire_activity_chart_detections.png", "sct_fire_activity_chart_burned.png"}
    assert fc.main([str(tmp_path / "nope.csv")]) == 2


# ---------------------------------------------------------------------------
# Charts in a full run (fake Earth Engine)
# ---------------------------------------------------------------------------

def _run(tmp_path, **kw):
    return ft.run(_args(tmp_path, **kw), fetcher_factory=FakeFireFetcher, log=lambda m: None,
                  burned_fetcher_factory=FakeBurnedFetcher)


def test_full_run_writes_charts_and_lists_them(tmp_path):
    assert _run(tmp_path, admin_file=str(_admin_file(tmp_path)), unit_id_field="ADM2_PCODE",
                area_name="Toy area") == 0
    for kind in ("detections", "burned", "admin"):
        assert (tmp_path / f"prt_fire_activity_chart_{kind}.png").exists()
    info = (tmp_path / "prt_fire_activity_run_info.txt").read_text(encoding="utf-8")
    assert "Charts: 3 PNG file(s), area name 'Toy area'" in info
    assert "prt_fire_activity_chart_admin.png" in info


def test_no_charts_option_clears_old_charts(tmp_path):
    _run(tmp_path)
    assert (tmp_path / "prt_fire_activity_chart_detections.png").exists()
    assert _run(tmp_path, no_charts=True) == 0
    assert not list(tmp_path.glob("*_chart_*.png"))
    assert "Charts: off" in (tmp_path / "prt_fire_activity_run_info.txt").read_text(encoding="utf-8")


def test_chart_failure_does_not_lose_the_run(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no fonts")
    monkeypatch.setattr(fc, "write_charts", boom)
    assert _run(tmp_path) == 0
    assert (tmp_path / "prt_fire_activity_summary.csv").exists()
    assert "Charts: failed (RuntimeError: no fonts)" in (tmp_path / "prt_fire_activity_run_info.txt").read_text(
        encoding="utf-8")


def test_new_options_parse():
    a = ft.build_arg_parser().parse_args(["--no-charts", "--area-name", "Scotland"])
    assert a.no_charts and a.area_name == "Scotland"


def test_wizard_flag_with_other_options_is_refused():
    with pytest.raises(SystemExit):
        ft.main(["--wizard", "--freq", "monthly"])


# ---------------------------------------------------------------------------
# Form builder
# ---------------------------------------------------------------------------

def _form(**kw):
    base = {"aoi_file": "a.shp", "geoextent": "sct", "start": "2026-03-01", "end": "2026-10-01",
            "freq": "monthly", "compare": True, "out_dir": "out", "ee_project": "my-ee-project"}
    base.update(kw)
    return base


def test_form_builds_the_same_options_a_typed_command_would():
    argv = fw.build_argv_from_form(_form(admin_file="adm.shp", unit_name_field="ADM2_EN", hex_size_km="5",
                                         density="totals", density_radius_km="3", burned_rasters="none",
                                         area_name="Scotland", sensor="modis", min_confidence="50",
                                         no_cache=True))
    a = ft.build_arg_parser().parse_args(argv)
    assert (a.aoi_file, a.geoextent, a.start, a.end, a.freq) == ("a.shp", "sct", "2026-03-01", "2026-10-01",
                                                                 "monthly")
    assert a.compare_prior_year and a.sensor == "modis" and a.min_confidence == 50 and a.no_cache
    assert a.admin_file == "adm.shp" and a.unit_name_field == "ADM2_EN" and a.hex_size_km == 5.0
    assert a.density == "totals" and a.density_radius_km == 3.0 and a.burned_rasters == "none"
    assert a.area_name == "Scotland" and not a.no_charts and not a.no_burned_area and not a.dry_run


def test_form_defaults_add_nothing_extra_and_switches_turn_things_off():
    argv = fw.build_argv_from_form(_form(compare=False, burned_area=False, charts=False, area_name="X",
                                         burned_rasters="none"))
    assert "--compare-prior-year" not in argv and "--sensor" not in argv and "--density" not in argv
    assert "--no-burned-area" in argv and "--burned-rasters" not in argv
    assert "--no-charts" in argv and "--area-name" not in argv


@pytest.mark.parametrize("change, message", [
    ({"aoi_file": ""}, "area-of-interest"),
    ({"geoextent": ""}, "geoextent"),
    ({"start": ""}, "start date"),
    ({"start": "1/3/2026"}, "YYYY-MM-DD"),
    ({"end": "2026-03-01"}, "after the start"),
    ({"out_dir": " "}, "output folder"),
    ({"hex_size_km": "big"}, "number"),
    ({"hex_size_km": "0"}, "greater than zero"),
    ({"min_confidence": "1.5"}, "whole number"),
    ({"freq": "daily"}, "Frequency"),
])
def test_form_errors_are_plain(change, message):
    with pytest.raises(ValueError, match=message):
        fw.build_argv_from_form(_form(**change))


def test_dry_run_form_does_not_need_geoextent_or_output_folder():
    argv = fw.build_argv_from_form(_form(geoextent="", out_dir=""), dry_run=True)
    assert argv[-1] == "--dry-run" and "--geoextent" not in argv


def test_quote_argv_quotes_paths_with_spaces():
    assert fw.quote_argv(["--aoi-file", r"H:\My Data\a.shp", "--freq", "monthly"]) == \
        r'python fire_tool.py --aoi-file "H:\My Data\a.shp" --freq monthly'


# ---------------------------------------------------------------------------
# Wizard (scripted answers)
# ---------------------------------------------------------------------------

class Script:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts, self.said = [], []

    def ask(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"wizard asked more than expected: {prompt}")
        return self.answers.pop(0)

    def say(self, text):
        self.said.append(text)


def test_wizard_defaults_give_a_standard_run():
    s = Script([str(SCOTLAND), "sct", "2026-03-01", "2026-10-01"] + [""] * 16)
    argv, check = fw.run_wizard(fw.Prompter(s.ask, s.say))
    a = ft.build_arg_parser().parse_args(argv)
    assert a.aoi_file == str(SCOTLAND) and a.geoextent == "sct" and a.freq == "monthly"
    assert a.compare_prior_year and a.sensor == "auto" and a.density == "all" and a.burned_rasters == "all"
    assert a.area_name is None          # Scotland is read from the file anyway
    assert a.out_dir == "out" and a.ee_project is None and check is True   # blank: saved project
    assert not s.answers


def test_wizard_re_asks_bad_answers_and_records_choices(tmp_path):
    admin = _admin_file(tmp_path)
    s = Script([
        "missing.shp", str(TOY),            # not found, then found
        "prt",
        "2026-13-01", "2026-03-01",         # bad date, then good
        "2026-02-01",                       # end before start ...
        "2026-03-01", "2026-04-01",         # ... so both asked again
        "2",                                # weekly
        "n",                                # no comparison
        "3",                                # MODIS
        str(admin),                         # admin file
        "NOPE", "ADM2_EN",                  # bad field, then good
        "ADM2_PCODE",
        "x", "5",                           # hexagon size: bad, then 5
        "3",                                # no density (so no radius question)
        "n",                                # no burned area
        "y", "Toy area",                    # charts, area name
        "results",
        "x", "2",                           # vector outputs: bad, then shapefiles
        "my-project",
        "n",                                # no advanced settings
        "n"])                               # no dry run first
    argv, check = fw.run_wizard(fw.Prompter(s.ask, s.say))
    a = ft.build_arg_parser().parse_args(argv)
    assert a.aoi_file == str(TOY) and a.start == "2026-03-01" and a.end == "2026-04-01"
    assert a.freq == "weekly" and not a.compare_prior_year and a.sensor == "modis"
    assert a.admin_file == str(admin) and a.unit_name_field == "ADM2_EN" and a.unit_id_field == "ADM2_PCODE"
    assert a.hex_size_km == 5.0 and a.density == "none" and a.density_radius_km is None
    assert a.no_burned_area and a.area_name == "Toy area" and a.out_dir == "results"
    assert a.ee_project == "my-project" and a.vector_format == "shp" and check is False
    assert not s.answers


def test_wizard_main_runs_dry_run_then_asks_before_the_real_run():
    calls = []

    def tool(argv):
        calls.append(argv)
        return 0
    s = Script([str(SCOTLAND), "sct", "2026-03-01", "2026-10-01"] + [""] * 16 + ["n"])
    assert fw.wizard_main(tool, fw.Prompter(s.ask, s.say)) == 0
    assert len(calls) == 1 and calls[0][-1] == "--dry-run"
    assert any("python fire_tool.py --aoi-file" in t for t in s.said)


def test_wizard_main_cancel():
    def ask(prompt):
        raise KeyboardInterrupt
    assert fw.wizard_main(lambda a: 0, fw.Prompter(ask, lambda t: None)) == 1


def test_no_options_starts_the_wizard(monkeypatch):
    seen = {}
    monkeypatch.setattr(fw, "wizard_main", lambda tool_main: seen.setdefault("called", tool_main) and 0)
    assert ft.main([]) == 0 and seen["called"] is ft.main


# ---------------------------------------------------------------------------
# GUI form (needs a display; skipped otherwise)
# ---------------------------------------------------------------------------

def _tk_root():
    try:
        import tkinter as tk
        return tk.Tk()
    except Exception:  # noqa: BLE001
        pytest.skip("no display for Tkinter")


def test_gui_form_to_options(tmp_path):
    root = _tk_root()
    root.withdraw()
    try:
        import fire_gui
        g = fire_gui.FireGUI(root, tool_main=lambda argv: 0)
        g.set_aoi(str(SCOTLAND))
        assert g.area_name.get() == "Scotland"
        g.geoextent.set("sct")
        g.start.set("2026-03-01")
        g.end.set("2026-10-01")
        g.freq.set("Weekly (Monday to Sunday)")
        g.density.set("None")
        g.radius.set("4")          # ignored while density is off
        g.unit_name_field.set("NAME_1")   # ignored: one feature and no admin file
        a = ft.build_arg_parser().parse_args(g.build_argv(dry_run=True))
        assert a.freq == "weekly" and a.density == "none" and a.density_radius_km is None
        assert a.unit_name_field is None and a.dry_run and a.area_name == "Scotland"
        g.set_admin(str(_admin_file(tmp_path)))
        assert g.unit_name_field.get() == "ADM2_EN"
        a = ft.build_arg_parser().parse_args(g.build_argv(dry_run=False))
        assert a.admin_file and a.unit_name_field == "ADM2_EN"
        # A typed area name is kept when another file is chosen.
        g.area_name.set("My area")
        g.set_aoi(str(SCOTLAND))
        assert g.area_name.get() == "My area"
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# Regression tests for the Phase 4 review findings
# ---------------------------------------------------------------------------

def test_chart_names_follow_the_same_rule_as_other_outputs():
    assert fc.chart_file_name("sct-n", "detections") == "sct_n_fire_activity_chart_detections.png"
    assert ft.output_filename("sct-n", "_summary.csv").startswith(fc.geo_token("sct-n") + "_")


def test_clear_charts_skips_files_it_cannot_remove(tmp_path, monkeypatch):
    p = tmp_path / fc.chart_file_name("sct", "admin")
    p.write_bytes(b"x")
    real_unlink = Path.unlink

    def locked_unlink(self, *a, **k):
        if self.name == p.name:
            raise PermissionError("in use")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", locked_unlink)
    locked = []
    assert fc.clear_charts(tmp_path, "sct", locked) == 0
    assert locked == [p.name]


def test_locked_old_chart_with_charts_off_still_writes_run_info(tmp_path, monkeypatch):
    _run(tmp_path)
    real_unlink = Path.unlink

    def locked_unlink(self, *a, **k):
        if "_chart_" in self.name:
            raise PermissionError("in use")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", locked_unlink)
    msgs = []
    assert ft.run(_args(tmp_path, no_charts=True), fetcher_factory=FakeFireFetcher, log=msgs.append,
                  burned_fetcher_factory=FakeBurnedFetcher) == 0
    assert "Charts: off" in (tmp_path / "prt_fire_activity_run_info.txt").read_text(encoding="utf-8")
    assert any("could not remove old chart" in m for m in msgs)


def test_stale_admin_csv_is_removed_and_not_charted(tmp_path):
    _run(tmp_path, admin_file=str(_admin_file(tmp_path)))
    assert (tmp_path / "prt_fire_activity_admin_summary.csv").exists()
    _run(tmp_path)
    assert not (tmp_path / "prt_fire_activity_admin_summary.csv").exists()
    paths = fc.charts_from_csv(str(tmp_path / "prt_fire_activity_summary.csv"), log=lambda m: None)
    assert "prt_fire_activity_chart_admin.png" not in {p.name for p in paths}


def test_failed_charts_leave_no_half_set(tmp_path, monkeypatch):
    real = fc.admin_bar_chart

    def boom(*a, **k):
        raise RuntimeError("late failure")
    monkeypatch.setattr(fc, "admin_bar_chart", boom)
    assert _run(tmp_path, admin_file=str(_admin_file(tmp_path))) == 0
    assert not list(tmp_path.glob("*_chart_*.png")) and not list(tmp_path.glob("*.tmp.png"))
    monkeypatch.setattr(fc, "admin_bar_chart", real)


def test_missing_month_is_no_data_not_not_yet_available(tmp_path, monkeypatch):
    seen = {}
    real = fc.period_column_chart

    def spy(series, freq, path, **kw):
        seen[path.name] = (series, kw)
        return real(series, freq, path, **kw)
    monkeypatch.setattr(fc, "period_column_chart", spy)
    rows = _rows(n=4)
    rows[1].update(burned_area_km2="", prior_burned_area_km2="", burned_qa_flag="missing_month")
    rows[3].update(burned_area_km2="", prior_burned_area_km2="", burned_qa_flag="not_yet_available")
    rows[2].update(qa_flag="partial_period")
    fc.write_charts(rows, tmp_path, "sct", freq="monthly", compare=True, area_name="S", sensor_label="V",
                    sensor_source="N", burned_on=True, log=lambda m: None)
    s, _ = seen["sct_fire_activity_chart_burned.png"]
    assert s.values[1] is None and s.nya[1] is False and s.flags[1] == ["missing_month"]
    assert s.values[3] is None and s.nya[3] is True
    d, _ = seen["sct_fire_activity_chart_detections.png"]
    assert d.values[2] == 20 and d.flags[2] == ["partial_period"] and not any(d.nya)


def test_combined_not_yet_available_flag_is_never_drawn_as_zero(tmp_path, monkeypatch):
    seen = {}
    real = fc.period_column_chart
    monkeypatch.setattr(fc, "period_column_chart",
                        lambda series, freq, path, **kw: seen.setdefault(path.name, series) and real(
                            series, freq, path, **kw))
    rows = _rows(n=3)
    rows[2].update(burned_area_km2="0", burned_qa_flag="not_yet_available+low_coverage")
    fc.write_charts(rows, tmp_path, "sct", freq="monthly", compare=True, area_name="S", sensor_label="V",
                    sensor_source="N", burned_on=True, log=lambda m: None)
    s = seen["sct_fire_activity_chart_burned.png"]
    assert s.values[2] is None and s.nya[2]


def test_redraw_reuses_the_runs_area_name(tmp_path):
    _run(tmp_path, area_name="Toy area")
    import re as _re
    titles = []
    real = fc.period_column_chart
    fc.period_column_chart = lambda series, freq, path, **kw: (titles.append(kw["title"]),
                                                                real(series, freq, path, **kw))[1]
    try:
        fc.charts_from_csv(str(tmp_path / "prt_fire_activity_summary.csv"), log=lambda m: None)
    finally:
        fc.period_column_chart = real
    assert titles and all(t.endswith("Toy area") for t in titles)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_form_rejects_non_finite_numbers(value):
    with pytest.raises(ValueError, match="ordinary number"):
        fw.build_argv_from_form(_form(hex_size_km=value))


def test_form_rejects_unicode_digits_for_confidence():
    with pytest.raises(ValueError, match="whole number"):
        fw.build_argv_from_form(_form(min_confidence="\u00b2"))


def test_values_starting_with_a_dash_survive_argparse():
    argv = fw.build_argv_from_form(_form(geoextent="-sct", area_name="-East"))
    a = ft.build_arg_parser().parse_args(argv)
    assert a.geoextent == "-sct" and a.area_name == "-East"


def test_quote_arg_round_trips_through_windows_rules():
    import subprocess
    for a in [r"C:\My Data\out" + "\\", r"H:\a b\c.shp", "a&b", 'say "hi"', "plain", ""]:
        # subprocess.list2cmdline follows the same Microsoft rules for quoting.
        q = fw.quote_arg(a)
        if " " in a or '"' in a or not a:
            assert q == subprocess.list2cmdline([a])
    assert fw.quote_arg("a&b") == '"a&b"'


def test_wizard_advanced_settings():
    s = Script([str(SCOTLAND), "sct", "2026-03-01", "2026-10-01"] + [""] * 14
               + ["y", "1.5", "50", "0.002", "y", ""])
    argv, check = fw.run_wizard(fw.Prompter(s.ask, s.say))
    a = ft.build_arg_parser().parse_args(argv)
    assert a.min_confidence == 50 and a.simplify_tolerance == 0.002 and a.no_cache and check
    assert not s.answers


def test_gui_typed_paths_are_read_and_worker_always_finishes(tmp_path):
    root = _tk_root()
    root.withdraw()
    try:
        import geopandas as gpd
        from shapely.geometry import box
        import fire_gui
        multi = tmp_path / "multi.gpkg"
        gpd.GeoDataFrame({"ADM2_EN": ["West", "East"], "ADM2_PCODE": ["A", "B"]},
                         geometry=[box(-8.3, 39.8, -8.0, 40.2), box(-8.0, 39.8, -7.7, 40.2)],
                         crs=4326).to_file(multi)
        g = fire_gui.FireGUI(root, tool_main=lambda argv: (_ for _ in ()).throw(KeyboardInterrupt()))
        g.aoi_file.set(str(multi))          # typed, not browsed
        g.geoextent.set("prt")
        g.start.set("2026-03-01")
        g.end.set("2026-04-01")
        g.unit_id_field.set("ADM2_PCODE")
        a = ft.build_arg_parser().parse_args(g.build_argv(dry_run=True))
        assert a.unit_name_field == "ADM2_EN" and a.unit_id_field == "ADM2_PCODE"
        g.aoi_file.set(str(SCOTLAND))       # now a single-feature file, typed
        a = ft.build_arg_parser().parse_args(g.build_argv(dry_run=True))
        assert a.unit_name_field is None and a.unit_id_field is None
        g.simplify.set("0.001")
        assert ft.build_arg_parser().parse_args(g.build_argv(dry_run=True)).simplify_tolerance == 0.001
        # A KeyboardInterrupt inside the run still switches the buttons back on.
        g._start(dry_run=True)
        g._worker.join(10)
        import time
        for _ in range(20):
            root.update()
            time.sleep(0.05)
        assert str(g.run_button.cget("state")) == "normal"
        assert "KeyboardInterrupt" in g.log.get("1.0", "end")
        w = fire_gui._QueueWriter(g._log_queue)
        assert w.encoding == "utf-8" and w.isatty() is False
    finally:
        root.destroy()


@pytest.mark.parametrize("opt", ["--hex-size-km", "--density-radius-km", "--simplify-tolerance"])
def test_typed_command_rejects_nan_and_inf(opt):
    for bad in ("nan", "inf"):
        with pytest.raises(SystemExit):
            ft.build_arg_parser().parse_args([opt, bad])
    assert getattr(ft.build_arg_parser().parse_args([opt, "2.5"]), opt[2:].replace("-", "_")) == 2.5
