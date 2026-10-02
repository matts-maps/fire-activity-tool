"""Offline tests for fire_spatial.py (Phase 2: hexagons, admin areas, density).

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
import fire_spatial as fs  # noqa: E402
import fire_tool as ft  # noqa: E402


def raster_names(folder):
    """File names in a raster folder, leaving out the .prj that comes with an
    Esri ASCII grid when rasterio is not installed."""
    return sorted(p.name for p in folder.iterdir() if p.suffix != ".prj")

SAMPLE_AOI = Path(__file__).resolve().parents[1] / "sample_aoi" / "toy_bbox.geojson"   # -8.3..-7.7, 39.8..40.2


def _box(x0, y0, x1, y1):
    from shapely.geometry import box
    return box(x0, y0, x1, y1)


def _pts(rows, period="2026-08", prior=False):
    """Point table shaped like fire_tool.build_point_tables output."""
    df = pd.DataFrame([{"date": d, "longitude": x, "latitude": y, "confidence": 1, "frp_mw": 2.0,
                        "brightness_k": 330.0, "sensor": "viirs", "satellite": "Suomi NPP", "day_night": "day",
                        "acq_time_utc": "12:00", "period": period} for d, x, y in rows])
    if df.empty:
        df = ft.empty_detections().assign(period="", sensor="", satellite="", day_night="", acq_time_utc="")
    if prior:
        df["compares_with"] = "2026" + period[4:] if len(df) else ""
        df["in_comparison"] = "yes"
    return df


# --- hexagon size and grid ---------------------------------------------------

def test_auto_hex_size_follows_area():
    assert fs.auto_hex_size_km(2_000) == 2          # a district
    assert fs.auto_hex_size_km(79_646) == 10        # Scotland
    assert fs.auto_hex_size_km(600_000) == 25       # a large country, capped
    assert fs.auto_hex_size_km(50) == 1             # tiny areas floor at 1 km


def test_hex_grid_is_equal_area_and_covers_aoi():
    aoi = _box(-8.3, 39.8, -7.7, 40.2)
    crs = fs.equal_area_crs(aoi)
    aoi_ea = fs.to_equal_area(aoi, crs)
    hexes = fs.build_hex_grid(aoi_ea, 5)
    assert hexes.geometry.area.round(-3).nunique() == 1                     # every hexagon the same size
    assert abs(hexes.geometry.area.iloc[0] / 1e6 - fs.hex_area_km2(5)) < 0.01
    assert aoi_ea.difference(hexes.union_all()).area < 1.0                  # no gaps (under 1 m2 of slivers)
    assert abs(hexes["area_in_aoi_km2"].sum() - aoi_ea.area / 1e6) < 0.5     # area inside adds up
    assert hexes["hex_id"].is_unique and hexes["hex_id"].min() == 1


def test_hex_grid_refuses_absurd_size():
    aoi_ea = fs.to_equal_area(_box(-10, 35, 10, 55), fs.equal_area_crs(_box(-10, 35, 10, 55)))
    with pytest.raises(ValueError, match="larger --hex-size-km"):
        fs.build_hex_grid(aoi_ea, 0.5)


# --- counting by zone -----------------------------------------------------------

def _zone_setup():
    import geopandas as gpd
    aoi = _box(0, 0, 2, 1)
    crs = fs.equal_area_crs(aoi)
    zones = gpd.GeoDataFrame({"zid": ["a", "b"], "area_in_aoi_km2": [1.0, 1.0]},
                             geometry=[_box(0, 0, 1, 1), _box(1, 0, 2, 1)], crs=4326).to_crs(crs)
    periods = [ft.Period("2026-08", date(2026, 8, 1), date(2026, 9, 1)),
               ft.Period("2026-09", date(2026, 9, 1), date(2026, 10, 1))]
    return aoi, crs, zones, periods


def test_zone_counts_add_up_and_edge_points_count_once():
    _, crs, zones, periods = _zone_setup()
    cur = _pts([("2026-08-02", 0.5, 0.5), ("2026-08-03", 0.5, 0.5), ("2026-08-04", 1.0, 0.5),   # last on the shared edge
                ("2026-08-05", 1.5, 0.5)])
    pri = _pts([("2025-08-02", 1.5, 0.5)], period="2025-08", prior=True)
    rows = [{"period": "2026-08", "qa_flag": "ok", "prior_period": "2025-08", "prior_fire_count": 1,
             "comparison_note": ""},
            {"period": "2026-09", "qa_flag": "not_yet_available", "prior_period": "2025-09", "prior_fire_count": "",
             "comparison_note": "no current data yet"}]
    zt = fs.zone_tables(zones, "zid", fs.points_to_gdf(cur, crs), fs.points_to_gdf(pri, crs), periods, rows,
                        "viirs", compare=True)
    t = zt.totals.set_index("zid")
    assert t["fire_count"].sum() == 4 and zt.unassigned_current == 0
    assert t.loc["a", "days_with_fire"] >= 2 and t.loc["b", "prior_fire_count"] == 1
    assert pd.isna(t.loc["a", "change_pct"])           # no prior fires: percentage left blank, not infinite
    bp = zt.by_period
    assert set(bp["period"]) == {"2026-08"}            # zero rows and the not-yet-available period left out
    assert bp["fire_count"].sum() == 4


def test_partial_period_prior_uses_like_for_like_points_only():
    _, crs, zones, periods = _zone_setup()
    cur = _pts([("2026-08-02", 0.5, 0.5)])
    pri = _pts([("2025-08-02", 0.5, 0.5), ("2025-08-25", 0.5, 0.5)], period="2025-08", prior=True)
    pri.loc[1, "in_comparison"] = "no"                 # beyond the days available this year
    rows = [{"period": "2026-08", "qa_flag": "partial_period", "prior_period": "2025-08", "prior_fire_count": 1,
             "comparison_note": "first 10 days of both periods only"}]
    zt = fs.zone_tables(zones, "zid", fs.points_to_gdf(cur, crs), fs.points_to_gdf(pri, crs), periods[:1], rows,
                        "viirs", compare=True)
    assert zt.totals.set_index("zid").loc["a", "prior_fire_count"] == 1
    assert zt.by_period["prior_fire_count"].tolist() == [1]


def test_zone_tables_without_comparison_have_no_prior_columns():
    _, crs, zones, periods = _zone_setup()
    zt = fs.zone_tables(zones, "zid", fs.points_to_gdf(_pts([("2026-08-02", 0.5, 0.5)]), crs), None, periods[:1],
                        [{"period": "2026-08", "qa_flag": "ok"}], "modis", compare=False)
    assert "prior_fire_count" not in zt.totals and "prior_fire_count" not in zt.by_period
    assert "frp_sum_mw" not in zt.totals                # MODIS has no fire radiative power


def test_offshore_point_snaps_to_nearest_area_within_pixel_size():
    _, crs, zones, periods = _zone_setup()
    cur = _pts([("2026-08-02", 0.5, 1.002), ("2026-08-03", 0.5, 1.5)])   # ~220 m and ~55 km north of zone a
    zone = fs.assign_zone(fs.points_to_gdf(cur, crs), zones, "zid", snap_m=375)
    assert zone.tolist()[0] == "a" and pd.isna(zone.tolist()[1])


# --- density ---------------------------------------------------------------------

def test_density_integrates_to_point_count_and_masks_outside():
    aoi = _box(-8.3, 39.8, -7.7, 40.2)
    crs = fs.equal_area_crs(aoi)
    grid = fs.make_density_grid(fs.to_equal_area(aoi, crs), crs, 5)
    pts = fs.points_to_gdf(_pts([("2025-08-01", -8.0, 40.0)] * 3 + [("2025-08-01", -8.05, 40.02)]), crs)
    surf = fs.density_surface(pts, grid)
    inside = surf[surf != fs.NODATA]
    cell_km2 = (grid.cell_m / 1000) ** 2
    assert abs(inside.sum() * cell_km2 - 4) < 0.01      # four fires, all well inside: nothing lost
    assert (surf[~grid.inside] == fs.NODATA).all() and inside.min() >= 0
    assert fs.density_surface(pts.iloc[:0], grid)[grid.inside].max() == 0


def test_density_grid_is_capped_for_huge_areas():
    aoi = _box(-10, 35, 30, 60)
    crs = fs.equal_area_crs(aoi)
    grid = fs.make_density_grid(fs.to_equal_area(aoi, crs), crs, 1)
    assert max(grid.nx, grid.ny) <= fs.MAX_DENSITY_CELLS + 1


def test_ascii_grid_fallback_without_rasterio(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_rasterio(name, *a, **k):
        if name.startswith("rasterio"):
            raise ImportError("no rasterio")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_rasterio)
    aoi = _box(-8.3, 39.8, -7.7, 40.2)
    crs = fs.equal_area_crs(aoi)
    grid = fs.make_density_grid(fs.to_equal_area(aoi, crs), crs, 5)
    written = fs.write_raster(tmp_path / "x.tif", np.zeros((grid.ny, grid.nx), "float32"), grid)
    assert written.suffix == ".asc" and written.exists() and written.with_suffix(".prj").exists()
    assert written.read_text().splitlines()[0] == f"ncols {grid.nx}"


# --- admin areas -----------------------------------------------------------------

def _admin_file(tmp_path, with_names=True):
    import geopandas as gpd
    g = gpd.GeoDataFrame({"ADM2_EN": ["West", "East", "Far away"], "ADM2_PCODE": ["P1", "P2", "P3"]},
                         geometry=[_box(-8.3, 39.8, -8.0, 40.2), _box(-8.0, 39.8, -7.5, 40.2), _box(5, 5, 6, 6)],
                         crs=4326)
    if not with_names:
        g = g.drop(columns=["ADM2_EN"])
    path = tmp_path / ("admin.gpkg" if with_names else "admin_nonames.gpkg")
    g.to_file(path)
    return path


def test_admin_units_name_found_and_outside_units_dropped(tmp_path):
    units = fs.load_admin_units(str(_admin_file(tmp_path)), _box(-8.3, 39.8, -7.7, 40.2), None, "ADM2_PCODE", "test")
    assert units.gdf["unit_name"].tolist() == ["West", "East"] and units.dropped == 1
    east = units.gdf.set_index("unit_id").loc["P2"]
    assert 55 < east["share_in_aoi_pct"] < 65            # 0.3 of 0.5 degrees wide lies inside


def test_admin_bad_field_lists_what_is_there(tmp_path):
    with pytest.raises(ValueError, match="ADM2_PCODE"):
        fs.load_admin_units(str(_admin_file(tmp_path)), _box(-8.3, 39.8, -7.7, 40.2), "NAME_9", None, "test")
    with pytest.raises(ValueError, match="--unit-name-field"):
        fs.load_admin_units(str(_admin_file(tmp_path, with_names=False)), _box(-8.3, 39.8, -7.7, 40.2), None, None,
                            "test")
    u = fs.load_admin_units(str(_admin_file(tmp_path, with_names=False)), _box(-8.3, 39.8, -7.7, 40.2), None, None,
                            "test", allow_unnamed=True)
    assert u.gdf["unit_name"].tolist() == ["Area 1", "Area 2"]


# --- naming ----------------------------------------------------------------------

def test_names_follow_convention():
    assert fs.zone_layer_name("sct", "viirs", fs.hex_kind(10), True) == "sct_evnt_fir_py_s0_nasa_pp_viirs_hex10km_total"
    assert fs.hex_kind(2.5) == "hex2p5km"
    assert fs.density_file_name("sct", "viirs", "2026-W05", "change") == \
        "sct_evnt_fir_ras_s0_nasa_pp_viirs_density_2026_w05_change.tif"


# --- end to end through fire_tool.run --------------------------------------------

class PortugalFetcher:
    """Fires inside the toy box: one in the west half each day, two in the east
    half on even days. Data ends 20 September 2025."""

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
        rows, d = [], start
        while d < end and d <= self.latest:
            rows.append({"date": d.isoformat(), "longitude": -8.2, "latitude": 40.0, "confidence": 1})
            if d.day % 2 == 0:
                rows += [{"date": d.isoformat(), "longitude": lon, "latitude": 40.1, "confidence": 2}
                         for lon in (-7.9, -7.8)]
            d += timedelta(days=1)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.assign(frp=3.0, Bright_ti4=330.0, DayNight=1, acq_time=45000)
        return ft.normalise_detections(df, self.sensor)


def _args(tmp_path, **kw):
    base = dict(aoi_file=str(SAMPLE_AOI), geoextent="prt", start="2025-08-01", end="2025-10-01", freq="monthly",
                compare_prior_year=True, sensor="auto", min_confidence=None, out_dir=str(tmp_path), ee_project=None,
                simplify_tolerance=None, no_cache=True, dry_run=False, hex_size_km=None, admin_file=None,
                unit_name_field=None, unit_id_field=None, density="all", density_radius_km=None,
                no_burned_area=True)
    base.update(kw)
    return Namespace(**base)


def test_end_to_end_spatial_outputs(tmp_path):
    import geopandas as gpd
    admin = _admin_file(tmp_path)
    assert ft.run(_args(tmp_path, admin_file=str(admin), unit_id_field="ADM2_PCODE"),
                  fetcher_factory=PortugalFetcher, log=lambda m: None) == 0
    gpkg = tmp_path / "prt_fire_activity.gpkg"
    layers = gpd.list_layers(gpkg)["name"].tolist()
    assert "prt_evnt_fir_py_s0_nasa_pp_viirs_hex2km_total" in layers
    assert "prt_evnt_fir_py_s0_nasa_pp_viirs_admin_by_period" in layers

    summary = pd.read_csv(tmp_path / "prt_fire_activity_summary.csv")
    hexes = gpd.read_file(gpkg, layer="prt_evnt_fir_py_s0_nasa_pp_viirs_hex2km_total")
    assert hexes["fire_count"].sum() == summary["fire_count"].sum()
    assert hexes["prior_fire_count"].sum() == summary["prior_fire_count"].sum()   # like-for-like September too
    assert hexes.crs.to_epsg() == 4326

    by_period = gpd.read_file(gpkg, layer="prt_evnt_fir_py_s0_nasa_pp_viirs_admin_by_period")
    sums = by_period.groupby("period")["fire_count"].sum()
    assert sums.to_dict() == summary.set_index("period")["fire_count"].to_dict()
    assert str(by_period["period_start"].dtype).startswith("datetime")             # time slider ready

    admin_csv = pd.read_csv(tmp_path / "prt_fire_activity_admin_summary.csv")
    total_rows = admin_csv[admin_csv["period"] == "all"].set_index("unit_name")
    aug_sep_days = 31 + 20
    assert total_rows.loc["West", "fire_count"] == aug_sep_days
    assert total_rows.loc["East", "fire_count"] == summary["fire_count"].sum() - aug_sep_days
    assert "fires_per_1000_km2" in admin_csv.columns

    dens = raster_names(tmp_path / "prt_fire_activity_density")
    assert len(dens) == 9                       # total + 2 periods, each current, prior and change
    info = (tmp_path / "prt_fire_activity_run_info.txt").read_text()
    assert "Hexagons:" in info and "Admin areas: 2" in info and "outside the area of interest left out" in info


def test_density_options(tmp_path):
    ft.run(_args(tmp_path / "t", density="totals"), fetcher_factory=PortugalFetcher, log=lambda m: None)
    assert len(raster_names(tmp_path / "t" / "prt_fire_activity_density")) == 3
    ft.run(_args(tmp_path / "n", density="none", compare_prior_year=False), fetcher_factory=PortugalFetcher,
           log=lambda m: None)
    assert not (tmp_path / "n" / "prt_fire_activity_density").exists()


def test_multi_feature_aoi_becomes_admin_areas(tmp_path):
    admin = _admin_file(tmp_path)
    import geopandas as gpd
    aoi = tmp_path / "two_parts.gpkg"
    gpd.read_file(admin).iloc[:2].to_file(aoi)
    msgs = []
    ft.run(_args(tmp_path, aoi_file=str(aoi), compare_prior_year=False, density="none", hex_size_km=5),
           fetcher_factory=PortugalFetcher, log=msgs.append)
    assert any("Admin areas: 2" in m for m in msgs) and any("Hexagons: 5 km" in m for m in msgs)
    assert (tmp_path / "prt_fire_activity_admin_summary.csv").exists()


def test_unit_fields_without_admin_source_error(tmp_path):
    with pytest.raises(ValueError, match="--admin-file"):
        ft.run(_args(tmp_path, unit_name_field="NAME_1", dry_run=True), fetcher_factory=None, log=lambda m: None)


def test_no_fires_still_writes_spatial_layers(tmp_path):
    import geopandas as gpd

    class Empty(PortugalFetcher):
        def fetch(self, start, end):
            return ft.empty_detections()

    ft.run(_args(tmp_path, admin_file=str(_admin_file(tmp_path))), fetcher_factory=Empty, log=lambda m: None)
    gpkg = tmp_path / "prt_fire_activity.gpkg"
    hexes = gpd.read_file(gpkg, layer="prt_evnt_fir_py_s0_nasa_pp_viirs_hex2km_total")
    assert len(hexes) > 0 and hexes["fire_count"].sum() == 0
    assert len(gpd.read_file(gpkg, layer="prt_evnt_fir_py_s0_nasa_pp_viirs_hex2km_by_period")) == 0


def test_masking_band_set_for_each_sensor():
    assert ft.SENSORS["viirs"].fire_band == "Bright_ti4" and ft.SENSORS["modis"].fire_band == "T21"
    assert ft.CACHE_VERSION == "v2"


# --- regressions from the independent review (1 October 2026) --------------------

def test_fires_along_a_long_straight_edge_are_not_lost():
    """Long box edges must follow the projection's curve, not be cut as chords."""
    aoi = _box(-10, 35, 30, 60)
    crs = fs.equal_area_crs(aoi)
    aoi_ea = fs.to_equal_area(aoi, crs)
    hexes = fs.build_hex_grid(aoi_ea, 25).set_crs(crs)
    pts = fs.points_to_gdf(_pts([("2026-08-02", x, 35.02) for x in np.linspace(-9.9, 29.9, 50)]), crs)
    assert fs.assign_zone(pts, hexes, "hex_id").notna().all()
    assert abs(aoi_ea.area / 1e6 - 8.31e6) / 8.31e6 < 0.01


def _week53_rows():
    return [{"period": "2026-W52", "qa_flag": "ok", "prior_period": "2025-W52", "prior_fire_count": 1,
             "comparison_note": ""},
            {"period": "2026-W53", "qa_flag": "ok", "prior_period": "", "prior_fire_count": "",
             "comparison_note": "no matching ISO week 53 in the previous year"}]


def test_total_change_only_counts_periods_with_a_match():
    _, crs, zones, _ = _zone_setup()
    periods = [ft.Period("2026-W52", date(2026, 12, 21), date(2026, 12, 28)),
               ft.Period("2026-W53", date(2026, 12, 28), date(2027, 1, 4))]
    cur = pd.concat([_pts([("2026-12-22", 0.5, 0.5)], period="2026-W52"),
                     _pts([("2026-12-29", 0.5, 0.5)] * 5, period="2026-W53")], ignore_index=True)
    pri = _pts([("2025-12-23", 0.5, 0.5)], period="2025-W52", prior=True).assign(compares_with="2026-W52")
    zt = fs.zone_tables(zones, "zid", fs.points_to_gdf(cur, crs), fs.points_to_gdf(pri, crs), periods,
                        _week53_rows(), "viirs", compare=True)
    a = zt.totals.set_index("zid").loc["a"]
    assert a["fire_count"] == 6 and a["compared_fire_count"] == 1 and a["change_abs"] == 0


def test_no_year_earlier_fires_gives_zero_not_blank():
    _, crs, zones, periods = _zone_setup()
    cur = _pts([("2026-08-02", 0.5, 0.5)])
    pri = _pts([], prior=True).assign(compares_with="", in_comparison="")
    rows = [{"period": "2026-08", "qa_flag": "ok", "prior_period": "2025-08", "prior_fire_count": 0,
             "comparison_note": ""}]
    zt = fs.zone_tables(zones, "zid", fs.points_to_gdf(cur, crs), fs.points_to_gdf(pri, crs), periods[:1], rows,
                        "viirs", compare=True)
    assert zt.by_period["prior_fire_count"].tolist() == [0] and zt.by_period["change_abs"].tolist() == [1]


def test_self_intersecting_boundary_keeps_its_full_area(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Polygon
    bowtie = Polygon([(-8.3, 39.8), (-7.7, 40.2), (-7.7, 39.8), (-8.3, 40.2)])
    path = tmp_path / "bowtie.geojson"
    gpd.GeoDataFrame({"n": [1, 2]}, geometry=[bowtie, _box(-7.6, 39.8, -7.5, 39.9)], crs=4326).to_file(path)
    aoi = ft.load_aoi(str(path))
    assert aoi.geometry.is_valid and abs(aoi.geometry.area - (0.12 + 0.01)) < 1e-6


def test_duplicate_or_blank_unit_ids_are_rejected_clearly(tmp_path):
    import geopandas as gpd
    path = tmp_path / "dup.gpkg"
    gpd.GeoDataFrame({"ADM2_EN": ["A", "B"], "code": ["X", "X"]},
                     geometry=[_box(-8.3, 39.8, -8.0, 40.2), _box(-8.0, 39.8, -7.7, 40.2)], crs=4326).to_file(path)
    with pytest.raises(ValueError, match="not unique"):
        fs.load_admin_units(str(path), _box(-8.3, 39.8, -7.7, 40.2), None, "code", "test")


def test_bad_sizes_fail_before_download(tmp_path):
    with pytest.raises(ValueError, match="greater than zero"):
        ft.run(_args(tmp_path, hex_size_km=-1, dry_run=True), fetcher_factory=None, log=lambda m: None)
    with pytest.raises(ValueError, match="greater than zero"):
        ft.run(_args(tmp_path, density_radius_km=0, dry_run=True), fetcher_factory=None, log=lambda m: None)
    with pytest.raises(ValueError, match="larger --hex-size-km"):
        ft.run(_args(tmp_path, hex_size_km=0.01, dry_run=True), fetcher_factory=None, log=lambda m: None)


def test_dry_run_and_real_run_agree_on_hexagon_size(tmp_path):
    msgs = []
    ft.run(_args(tmp_path, density="none"), fetcher_factory=PortugalFetcher, log=msgs.append)
    assert [m for m in msgs if m.startswith("Hexagons:")] == ["Hexagons: 2 km across (automatic; set --hex-size-km to change)"]


def test_rerun_clears_old_density_rasters(tmp_path):
    ft.run(_args(tmp_path), fetcher_factory=PortugalFetcher, log=lambda m: None)
    keep = tmp_path / "prt_fire_activity_density" / "my_notes.txt"
    keep.write_text("mine")
    ft.run(_args(tmp_path, density="totals"), fetcher_factory=PortugalFetcher, log=lambda m: None)
    names = raster_names(tmp_path / "prt_fire_activity_density")
    assert len(names) == 4 and "my_notes.txt" in names
    # Density switched off: the earlier rasters go too, the user's own file stays.
    ft.run(_args(tmp_path, density="none"), fetcher_factory=PortugalFetcher, log=lambda m: None)
    assert [p.name for p in (tmp_path / "prt_fire_activity_density").iterdir()] == ["my_notes.txt"]
