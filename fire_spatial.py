"""
fire_spatial.py - spatial summaries for the fire activity tool (Phase 2).

Takes the fire point tables built by fire_tool.py and produces:

  hexagon grid layers   counts per hexagon, for the whole run and per period
  admin area layers     counts per admin unit (optional), whole run and per period
  density rasters       kernel density (fires per km2) per period and for the
                        whole run, with year-earlier and change rasters

All geometry work is done in a Lambert azimuthal equal-area projection centred
on the area of interest, so hexagons are the same size everywhere and densities
are true per-km2 values. Vector layers are written back in WGS84 (EPSG:4326) to
match the point layers; rasters stay in the equal-area projection.

No Earth Engine here: everything runs locally with geopandas, shapely, pyproj
and numpy. rasterio is used for GeoTIFFs when installed; otherwise rasters are
written as Esri ASCII grids (.asc with a .prj), which ArcGIS Pro opens directly.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

NODATA = -9999.0

# Hexagon sizes the automatic rule snaps to (km, distance across flats).
NICE_HEX_SIZES_KM = (1, 2, 5, 10, 15, 20, 25)
AUTO_HEX_TARGET_COUNT = 1000
MAX_HEX_CANDIDATES = 2_000_000

# Density grid: cell size is the search radius divided by this, enlarged if the
# grid would be wider or taller than MAX_DENSITY_CELLS cells.
DENSITY_CELLS_PER_RADIUS = 5
MAX_DENSITY_CELLS = 2000

DENSITY_CHOICES = ("all", "totals", "none")

# Admin unit name fields tried, in order, when --unit-name-field is not given.
COMMON_NAME_FIELDS = ("ADM2_EN", "ADM1_EN", "ADM3_EN", "adm2_name", "adm1_name", "NAME_2", "NAME_1",
                      "NAME_3", "name", "NAME", "Name")


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def equal_area_crs(aoi_geom):
    """Lambert azimuthal equal-area projection centred on the area of interest."""
    from pyproj import CRS
    c = aoi_geom.centroid
    return CRS.from_proj4(f"+proj=laea +lat_0={c.y:.4f} +lon_0={c.x:.4f} +x_0=0 +y_0=0 "
                          "+datum=WGS84 +units=m +no_defs")


# Longitude/latitude edges are densified to this spacing (degrees) before
# projecting, so a long straight boundary edge (a bounding box, a border along a
# parallel) follows its true curved path in the equal-area projection instead of
# being cut across as a chord, which would drop fires near the edge.
DENSIFY_DEG = 0.01


def project_gdf(gdf, crs):
    """Reproject a WGS84 GeoDataFrame to `crs`, densifying edges first."""
    import shapely
    out = gdf.copy()
    out["geometry"] = shapely.segmentize(out.geometry.values, DENSIFY_DEG)
    return out.to_crs(crs)


def to_equal_area(geom, crs):
    import geopandas as gpd
    import shapely
    return gpd.GeoSeries([shapely.segmentize(geom, DENSIFY_DEG)], crs=4326).to_crs(crs).iloc[0]


def polygon_parts(geom):
    """Keep only the polygon parts of a geometry (make_valid can return lines and
    points alongside polygons)."""
    import shapely
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = []
        for g in geom.geoms:
            g = polygon_parts(g)
            if isinstance(g, Polygon):
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend(g.geoms)
        return MultiPolygon(polys) if polys else Polygon()
    return Polygon()


# ---------------------------------------------------------------------------
# Hexagons
# ---------------------------------------------------------------------------

def hex_area_km2(size_km: float) -> float:
    """Area of a hexagon `size_km` across the flats."""
    return math.sqrt(3) / 2 * size_km ** 2


def auto_hex_size_km(area_km2: float) -> float:
    """Pick a hexagon size giving roughly AUTO_HEX_TARGET_COUNT hexagons, snapped
    to a round number: about 2 km for a district, 10 km for Scotland, 25 km
    (the cap) for a large country."""
    ideal = math.sqrt(max(area_km2, 1e-9) / AUTO_HEX_TARGET_COUNT / (math.sqrt(3) / 2))
    return float(min(NICE_HEX_SIZES_KM, key=lambda s: abs(math.log(s) - math.log(ideal))))


def build_hex_grid(aoi_ea, size_km: float):
    """Pointy-topped hexagons `size_km` across the flats, covering the
    equal-area area-of-interest geometry. Returns a GeoDataFrame (equal-area
    CRS not set) with hex_id, area_km2 and area_in_aoi_km2, keeping only
    hexagons that touch the area of interest."""
    import geopandas as gpd
    import shapely

    w = size_km * 1000.0               # flat-to-flat width = column spacing
    r = w / math.sqrt(3)               # centre-to-corner radius
    dy = 1.5 * r                       # row spacing
    minx, miny, maxx, maxy = aoi_ea.bounds
    n_rows = int(math.ceil((maxy - miny) / dy)) + 2
    n_cols = int(math.ceil((maxx - minx) / w)) + 2
    if n_rows * n_cols > MAX_HEX_CANDIDATES:
        raise ValueError(f"A {size_km:g} km hexagon grid would need about {n_rows * n_cols:,} hexagons for this "
                         "area. Use a larger --hex-size-km.")
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    cx = minx - w + cols * w + (rows % 2) * (w / 2)
    cy = miny - dy + rows * dy
    cx, cy = cx.ravel(), cy.ravel()
    angles = np.deg2rad(np.arange(30, 390, 60))          # corners at 30, 90, ... degrees
    xs = cx[:, None] + r * np.cos(angles)[None, :]
    ys = cy[:, None] + r * np.sin(angles)[None, :]
    rings = np.stack([xs, ys], axis=-1)
    rings = np.concatenate([rings, rings[:, :1, :]], axis=1)   # close each ring
    polys = shapely.polygons(rings)

    shapely.prepare(aoi_ea)
    keep = shapely.intersects(aoi_ea, polys)
    polys = polys[keep]
    inside = shapely.contains_properly(aoi_ea, polys)
    full = hex_area_km2(size_km)
    area_in = np.full(len(polys), full)
    edge = ~inside
    if edge.any():
        area_in[edge] = shapely.area(shapely.intersection(polys[edge], aoi_ea)) / 1e6
    overlap = area_in > 0                      # drop hexagons that only touch the boundary
    polys, area_in = polys[overlap], area_in[overlap]
    return gpd.GeoDataFrame({
        "hex_id": np.arange(1, len(polys) + 1, dtype="int64"),
        "area_km2": np.round(np.full(len(polys), full), 3),
        "area_in_aoi_km2": np.round(area_in, 3),
    }, geometry=polys)


# ---------------------------------------------------------------------------
# Admin units
# ---------------------------------------------------------------------------

@dataclass
class AdminUnits:
    gdf: object                 # GeoDataFrame in EPSG:4326: unit_id, unit_name, area_in_aoi_km2, share_in_aoi_pct
    source: str                 # plain-English description of where the units came from
    dropped: int = 0            # units that do not touch the area of interest


def pick_name_field(columns, requested: Optional[str], what: str) -> str:
    cols = [c for c in columns if c != "geometry"]
    if requested:
        if requested not in cols:
            raise ValueError(f"{what} has no field called {requested!r}. Fields available: {', '.join(cols)}")
        return requested
    for c in COMMON_NAME_FIELDS:
        if c in cols:
            return c
    raise ValueError(f"Could not tell which field in {what} holds the area names. Pass --unit-name-field "
                     f"(fields available: {', '.join(cols)}).")


def load_admin_units(path: str, aoi_geom, name_field: Optional[str], id_field: Optional[str],
                     what: str, allow_unnamed: bool = False) -> AdminUnits:
    """Read admin units, keep those touching the area of interest, and record
    how much of each lies inside it (in km2, on the equal-area projection)."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"{what} ({path}) has no features.")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    try:
        name_col = pick_name_field(gdf.columns, name_field, what)
    except ValueError:
        if name_field or not allow_unnamed:
            raise
        name_col = None
    if id_field and id_field not in gdf.columns:
        raise ValueError(f"{what} has no field called {id_field!r}. Fields available: "
                         f"{', '.join(c for c in gdf.columns if c != 'geometry')}")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = [polygon_parts(g) for g in gdf.geometry]
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if id_field:
        ids = gdf[id_field]
        blank = ids.isna() | (ids.astype(str).str.strip() == "")
        if blank.any():
            raise ValueError(f"{what}: {int(blank.sum())} area(s) have no value in {id_field}. Fill them in, or "
                             "leave out --unit-id-field to number the areas instead.")
        dup = ids.astype(str)[ids.astype(str).duplicated()].unique()
        if len(dup):
            raise ValueError(f"{what}: {id_field} is not unique (repeated: {', '.join(map(str, dup[:5]))}). Use a "
                             "field with one value per area, or leave out --unit-id-field.")
    out = gpd.GeoDataFrame({
        "unit_id": (gdf[id_field].astype(str) if id_field else pd.Series(range(1, len(gdf) + 1), index=gdf.index)
                    .astype(str)),
        "unit_name": (gdf[name_col].astype(str) if name_col else
                      pd.Series([f"Area {i}" for i in range(1, len(gdf) + 1)], index=gdf.index)),
    }, geometry=gdf.geometry, crs=4326)

    ea = equal_area_crs(aoi_geom)
    aoi_ea = to_equal_area(aoi_geom, ea)
    units_ea = project_gdf(out, ea)
    inter = units_ea.geometry.intersection(aoi_ea)
    total = units_ea.geometry.area
    out["area_in_aoi_km2"] = (inter.area / 1e6).round(3).values
    out["share_in_aoi_pct"] = np.where(total > 0, (inter.area / total * 100).round(1), 0.0)
    touching = out["area_in_aoi_km2"] > 0
    dropped = int((~touching).sum())
    out = out[touching].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"None of the areas in {what} overlap the area of interest.")
    source = f"{Path(path).name}, " + (f"names from field {name_col}" if name_col else "no name field found, areas numbered") + (f", IDs from {id_field}" if id_field else "")
    return AdminUnits(out, source, dropped)


# ---------------------------------------------------------------------------
# Assigning points to zones and counting
# ---------------------------------------------------------------------------

def points_to_gdf(table: pd.DataFrame, crs=None):
    import geopandas as gpd
    g = gpd.GeoDataFrame(table.copy(), geometry=gpd.points_from_xy(table["longitude"], table["latitude"]), crs=4326)
    return g.to_crs(crs) if crs is not None else g


def assign_zone(points_gdf, zones_gdf, id_col: str, snap_m: float = 0.0) -> pd.Series:
    """Zone ID for each point (NaN where none). Points on a shared edge go to one
    zone only. Points outside every zone but within `snap_m` metres of one (a
    fire pixel centre just offshore of a detailed coastline) go to the nearest."""
    import geopandas as gpd
    result = pd.Series(np.nan, index=points_gdf.index, dtype="object")
    if points_gdf.empty or zones_gdf.empty:
        return result
    hits = gpd.sjoin(points_gdf[["geometry"]], zones_gdf[[id_col, "geometry"]], how="inner", predicate="intersects")
    hits = hits[~hits.index.duplicated(keep="first")]
    result.loc[hits.index] = hits[id_col].values
    missing = result.isna()
    if snap_m > 0 and missing.any():
        near = gpd.sjoin_nearest(points_gdf.loc[missing, ["geometry"]], zones_gdf[[id_col, "geometry"]],
                                 how="inner", max_distance=snap_m)
        near = near[~near.index.duplicated(keep="first")]
        result.loc[near.index] = near[id_col].values
    return result


def _count_by(zone: pd.Series, mask=None, extra: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-zone fire count (and FRP sum / days with fire when `extra` given)."""
    df = pd.DataFrame({"zone": zone})
    if extra is not None:
        df = df.join(extra)
    if mask is not None:
        df = df[mask]
    df = df[df["zone"].notna()]
    agg = {"fire_count": ("zone", "size")}
    if extra is not None and "frp_mw" in df:
        agg["frp_sum_mw"] = ("frp_mw", "sum")
    if extra is not None and "date" in df:
        agg["days_with_fire"] = ("date", "nunique")
    return df.groupby("zone").agg(**agg) if len(df) else pd.DataFrame(columns=list(agg)).rename_axis("zone")


def compared_period_labels(summary_rows) -> set:
    """Periods whose summary row has a year-earlier count to compare with."""
    return {r["period"] for r in summary_rows if r.get("prior_fire_count", "") != ""
            and not (isinstance(r.get("prior_fire_count"), float) and math.isnan(r["prior_fire_count"]))}


def _change(cur: pd.Series, prior: pd.Series) -> tuple[pd.Series, pd.Series]:
    abs_change = cur - prior
    pct = pd.Series(np.where(prior > 0, (100.0 * abs_change / prior.where(prior > 0, 1)).round(1), np.nan),
                    index=cur.index)
    return abs_change, pct


@dataclass
class ZoneTables:
    totals: object              # GeoDataFrame, one row per zone
    by_period: object           # GeoDataFrame, one row per zone per period (zero rows left out)
    unassigned_current: int = 0
    unassigned_prior: int = 0
    unassigned_burned: int = 0


@dataclass
class BurnedInput:
    """Burned pixel centres for the zone tables (Phase 3). `current` and `prior`
    are GeoDataFrames in the zones' CRS, one row per pixel per period, from
    fire_burned.burned_point_tables; `pixel_km2` is the area of one pixel."""
    current: object
    prior: object               # None without a comparison
    pixel_km2: float
    snap_m: float = 0.0


def _burned_zone_columns(zones, id_col, burned: BurnedInput, periods, summary_rows, compare: bool):
    """Per-zone burned area: totals (dict of Series by zone) and per-period
    (dict label -> dict of Series). Values in km2."""
    rows = {r["period"]: r for r in summary_rows}
    available = {p.label for p in periods if rows.get(p.label, {}).get("burned_area_km2", "") not in ("", None)
                 and not (isinstance(rows[p.label]["burned_area_km2"], float)
                          and math.isnan(rows[p.label]["burned_area_km2"]))}
    compared = {lbl for lbl in available if rows[lbl].get("prior_burned_area_km2", "") != ""}
    cur, pri = burned.current, burned.prior
    cz = assign_zone(cur, zones, id_col, burned.snap_m)
    count = lambda z, m=None: _count_by(z, mask=m)["fire_count"]   # noqa: E731
    k = burned.pixel_km2
    totals = {"burned_area_km2": (count(cz) * k) if available else None,
              "burned_pixels": count(cz) if available else None}
    if compare and pri is not None:
        pz = assign_zone(pri, zones, id_col, burned.snap_m)
        pm = (pri["in_comparison"] == "yes") if len(pri) else pd.Series(dtype=bool)
        cm = cur["period"].isin(compared) if len(cur) else pd.Series(dtype=bool)
        totals["prior_burned_area_km2"] = (count(pz, pm) * k) if compared else None
        totals["compared_burned_area_km2"] = (count(cz, cm) * k) if compared else None
    else:
        pz = None
    per = {}
    for p in periods:
        if p.label not in available:
            per[p.label] = None
            continue
        m = (cur["period"] == p.label) if len(cur) else pd.Series(dtype=bool)
        d = {"burned_area_km2": (count(cz, m) * k), "burned_pixels": count(cz, m)}
        if compare and pz is not None and p.label in compared:
            pm = ((pri["compares_with"] == p.label) & (pri["in_comparison"] == "yes")) if len(pri) else \
                pd.Series(dtype=bool)
            d["prior_burned_area_km2"] = (count(pz, pm) * k)
        per[p.label] = d
    unassigned = int(cz.isna().sum())
    return totals, per, unassigned


def _km2_change(cur: pd.Series, prior: pd.Series) -> tuple[pd.Series, pd.Series]:
    abs_change = (cur - prior).round(2)
    pct = pd.Series(np.where(prior > 0, (100.0 * abs_change / prior.where(prior > 0, 1)).round(1), np.nan),
                    index=cur.index)
    return abs_change, pct


def zone_tables(zones, id_col: str, current, prior, periods, summary_rows, sensor_key: str,
                compare: bool, snap_m: float = 0.0, rate_per_km2: Optional[float] = None,
                burned: Optional[BurnedInput] = None) -> ZoneTables:
    """Count current and year-earlier points per zone, for the whole run and per
    period. `zones` and the point GeoDataFrames must share a projected CRS.
    Year-earlier counts use only points marked in_comparison == 'yes', so a
    partial current period is compared like-for-like, as in the summary CSV.
    With `burned`, burned area columns (km2) are added as well."""
    viirs = sensor_key == "viirs"
    b_tot, b_per, b_unassigned = (_burned_zone_columns(zones, id_col, burned, periods, summary_rows, compare)
                                  if burned is not None else (None, None, 0))
    extra_cols = ["date"] + (["frp_mw"] if viirs else [])
    cur_zone = assign_zone(current, zones, id_col, snap_m)
    zones_idx = zones.set_index(id_col)

    totals = zones.copy()
    c = _count_by(cur_zone, extra=current[extra_cols])
    totals["fire_count"] = totals[id_col].map(c["fire_count"]).fillna(0).astype("int64")
    totals["days_with_fire"] = totals[id_col].map(c["days_with_fire"]).fillna(0).astype("int64")
    if viirs:
        totals["frp_sum_mw"] = totals[id_col].map(c["frp_sum_mw"]).fillna(0.0).round(1)
    unassigned_prior = 0
    prior_zone = None
    if compare and prior is not None:
        prior_zone = assign_zone(prior, zones, id_col, snap_m)
        in_cmp = (prior["in_comparison"] == "yes") if len(prior) else pd.Series(dtype=bool)
        p = _count_by(prior_zone, mask=in_cmp)
        totals["prior_fire_count"] = totals[id_col].map(p["fire_count"]).fillna(0).astype("int64")
        # Change is measured only over periods that have a year-earlier match
        # (not ISO week 53 without one, not periods with no data yet).
        compared = compared_period_labels(summary_rows)
        cm = current["period"].isin(compared) if len(current) else pd.Series(dtype=bool)
        cc = _count_by(cur_zone, mask=cm)
        totals["compared_fire_count"] = totals[id_col].map(cc["fire_count"]).fillna(0).astype("int64")
        totals["change_abs"], totals["change_pct"] = _change(totals["compared_fire_count"],
                                                             totals["prior_fire_count"])
        unassigned_prior = int(prior_zone[in_cmp].isna().sum()) if len(prior) else 0
    if rate_per_km2:
        area = totals["area_in_aoi_km2"].where(totals["area_in_aoi_km2"] > 0)
        totals["fires_per_1000_km2"] = (totals["fire_count"] / area * rate_per_km2).round(2)
        if "prior_fire_count" in totals:
            totals["prior_fires_per_1000_km2"] = (totals["prior_fire_count"] / area * rate_per_km2).round(2)
    if b_tot is not None:
        area = totals["area_in_aoi_km2"].where(totals["area_in_aoi_km2"] > 0)

        def zmap(s):
            return np.nan if s is None else totals[id_col].map(s).fillna(0.0)

        totals["burned_area_km2"] = zmap(b_tot["burned_area_km2"])
        totals["burned_pixels"] = (pd.array([pd.NA] * len(totals), dtype="Int64") if b_tot["burned_pixels"] is None
                                   else totals[id_col].map(b_tot["burned_pixels"]).fillna(0).astype("int64"))
        totals["burned_share_pct"] = (totals["burned_area_km2"] / area * 100).round(2)
        if "prior_burned_area_km2" in b_tot:
            totals["prior_burned_area_km2"] = zmap(b_tot["prior_burned_area_km2"])
            totals["compared_burned_area_km2"] = zmap(b_tot["compared_burned_area_km2"])
            if b_tot["prior_burned_area_km2"] is None:
                totals["burned_change_km2"] = np.nan
                totals["burned_change_pct"] = np.nan
            else:
                totals["burned_change_km2"], totals["burned_change_pct"] = _km2_change(
                    totals["compared_burned_area_km2"], totals["prior_burned_area_km2"])
        for col in ("burned_area_km2", "prior_burned_area_km2", "compared_burned_area_km2"):
            if col in totals:
                totals[col] = totals[col].round(2)

    # Per period, long format.
    summary_by_label = {r["period"]: r for r in summary_rows}
    parts = []
    for per in periods:
        row = summary_by_label.get(per.label, {})
        cur_mask = (current["period"] == per.label) if len(current) else pd.Series(dtype=bool)
        cp = _count_by(cur_zone, mask=cur_mask, extra=current[extra_cols])
        frame = pd.DataFrame({id_col: zones[id_col].values})
        frame["period"] = per.label
        frame["period_start"] = per.start.isoformat()
        frame["period_end"] = per.last_day.isoformat()
        frame["fire_count"] = frame[id_col].map(cp["fire_count"]).fillna(0).astype("int64")
        if viirs:
            frame["frp_sum_mw"] = frame[id_col].map(cp["frp_sum_mw"]).fillna(0.0).round(1)
        frame["qa_flag"] = row.get("qa_flag", "")
        keep = frame["fire_count"] > 0
        if compare:
            frame["prior_period"] = row.get("prior_period", "")
            prior_counted = prior is not None and row.get("prior_fire_count", "") != ""
            if prior_counted:
                pm = (prior["compares_with"] == per.label) & (prior["in_comparison"] == "yes")
                pp = _count_by(prior_zone, mask=pm)
                frame["prior_fire_count"] = frame[id_col].map(pp["fire_count"]).fillna(0).astype("float64")
                frame["change_abs"], frame["change_pct"] = _change(frame["fire_count"], frame["prior_fire_count"])
                keep |= frame["prior_fire_count"] > 0
            else:
                frame["prior_fire_count"] = np.nan
                frame["change_abs"] = np.nan
                frame["change_pct"] = np.nan
            frame["comparison_note"] = row.get("comparison_note", "")
        if b_per is not None:
            bp = b_per.get(per.label)
            frame["burned_qa_flag"] = row.get("burned_qa_flag", "")
            if bp is None:
                frame["burned_area_km2"] = np.nan
                frame["burned_pixels"] = pd.array([pd.NA] * len(frame), dtype="Int64")
            else:
                frame["burned_area_km2"] = frame[id_col].map(bp["burned_area_km2"]).fillna(0.0)
                frame["burned_pixels"] = frame[id_col].map(bp["burned_pixels"]).fillna(0).astype("Int64")
                keep |= frame["burned_area_km2"] > 0
            if compare:
                if bp is not None and "prior_burned_area_km2" in bp:
                    frame["prior_burned_area_km2"] = frame[id_col].map(bp["prior_burned_area_km2"]).fillna(0.0)
                    frame["burned_change_km2"], frame["burned_change_pct"] = _km2_change(
                        frame["burned_area_km2"], frame["prior_burned_area_km2"])
                    keep |= frame["prior_burned_area_km2"] > 0
                else:
                    frame["prior_burned_area_km2"] = np.nan
                    frame["burned_change_km2"] = np.nan
                    frame["burned_change_pct"] = np.nan
            frame["burned_note"] = row.get("burned_note", "")
            for col in ("burned_area_km2", "prior_burned_area_km2"):
                if col in frame:
                    frame[col] = frame[col].round(2)
        parts.append(frame[keep])
    by_period = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[id_col, "period"])
    geom = zones_idx.geometry
    import geopandas as gpd
    attrs = zones.drop(columns="geometry").set_index(id_col)
    by_period = by_period.join(attrs, on=id_col)
    lead = [id_col] + [c for c in attrs.columns]
    by_period = by_period[lead + [c for c in by_period.columns if c not in lead]]
    by_period = gpd.GeoDataFrame(by_period, geometry=geom.reindex(by_period[id_col]).values, crs=zones.crs)
    return ZoneTables(totals, by_period, int(cur_zone.isna().sum()), unassigned_prior, b_unassigned)


# ---------------------------------------------------------------------------
# Kernel density
# ---------------------------------------------------------------------------

@dataclass
class DensityGrid:
    crs: object                 # pyproj CRS (equal-area)
    x0: float                   # left edge, metres
    y0: float                   # bottom edge, metres
    cell_m: float
    nx: int
    ny: int
    radius_m: float
    inside: np.ndarray = field(repr=False)   # bool (ny, nx), row 0 = top

    @property
    def top(self) -> float:
        return self.y0 + self.ny * self.cell_m


def check_hex_size(aoi_ea, size_km: float) -> None:
    """Raise a plain error for a hexagon size that is not positive or would
    need too many hexagons, before anything is downloaded."""
    if size_km <= 0:
        raise ValueError("--hex-size-km must be greater than zero")
    w = size_km * 1000.0
    minx, miny, maxx, maxy = aoi_ea.bounds
    n = (int(math.ceil((maxy - miny) / (1.5 * w / math.sqrt(3)))) + 2) * (int(math.ceil((maxx - minx) / w)) + 2)
    if n > MAX_HEX_CANDIDATES:
        raise ValueError(f"A {size_km:g} km hexagon grid would need about {n:,} hexagons for this area. "
                         "Use a larger --hex-size-km.")


def make_density_grid(aoi_ea, crs, radius_km: float) -> DensityGrid:
    import shapely
    radius_m = radius_km * 1000.0
    cell = radius_m / DENSITY_CELLS_PER_RADIUS
    minx, miny, maxx, maxy = aoi_ea.bounds
    span = max(maxx - minx, maxy - miny)
    if span / cell > MAX_DENSITY_CELLS:
        cell = span / MAX_DENSITY_CELLS
    # Snap the grid to whole cells so reruns line up.
    x0 = math.floor(minx / cell) * cell
    y0 = math.floor(miny / cell) * cell
    nx = int(math.ceil((maxx - x0) / cell))
    ny = int(math.ceil((maxy - y0) / cell))
    xs = x0 + (np.arange(nx) + 0.5) * cell
    ys = y0 + ny * cell - (np.arange(ny) + 0.5) * cell      # top row first
    gx, gy = np.meshgrid(xs, ys)
    shapely.prepare(aoi_ea)
    inside = shapely.contains_xy(aoi_ea, gx.ravel(), gy.ravel()).reshape(ny, nx)
    return DensityGrid(crs, x0, y0, cell, nx, ny, radius_m, inside)


def _quartic_kernel(grid: DensityGrid) -> np.ndarray:
    """Quartic (biweight) kernel, as ArcGIS Kernel Density uses, on the grid's
    cells, scaled so one point adds up to exactly 1 fire across the surface
    when values are in fires per km2."""
    k = int(math.ceil(grid.radius_m / grid.cell_m))
    off = np.arange(-k, k + 1) * grid.cell_m
    dx, dy = np.meshgrid(off, off)
    d2 = (dx ** 2 + dy ** 2) / grid.radius_m ** 2
    kern = np.where(d2 < 1, (1 - d2) ** 2, 0.0)
    total = kern.sum()
    if total == 0:                      # radius smaller than a cell
        kern = np.zeros_like(kern)
        kern[k, k] = 1.0
        total = 1.0
    cell_km2 = (grid.cell_m / 1000.0) ** 2
    return kern / (total * cell_km2)


def density_surface(points_ea, grid: DensityGrid) -> np.ndarray:
    """Kernel density of `points_ea` (GeoDataFrame in the grid's CRS) in fires
    per km2, NODATA outside the area of interest. Each fire adds up to 1 across
    the full surface; the part spreading beyond the boundary is masked off."""
    counts = np.zeros((grid.ny, grid.nx))
    if len(points_ea):
        x = points_ea.geometry.x.to_numpy()
        y = points_ea.geometry.y.to_numpy()
        col = np.floor((x - grid.x0) / grid.cell_m).astype(int)
        row = np.floor((grid.top - y) / grid.cell_m).astype(int)
        ok = (col >= 0) & (col < grid.nx) & (row >= 0) & (row < grid.ny)
        np.add.at(counts, (row[ok], col[ok]), 1.0)
    kern = _quartic_kernel(grid)
    if counts.any():
        k = kern.shape[0] // 2
        shape = (grid.ny + 2 * k, grid.nx + 2 * k)
        conv = np.fft.irfft2(np.fft.rfft2(counts, shape) * np.fft.rfft2(kern, shape), shape)
        surf = conv[k:k + grid.ny, k:k + grid.nx]
        surf[np.abs(surf) < 1e-12] = 0.0             # FFT round-off
    else:
        surf = counts
    out = surf.astype("float32")
    out[~grid.inside] = NODATA
    return out


def write_raster(path: Path, data: np.ndarray, grid: DensityGrid) -> Path:
    """GeoTIFF via rasterio when available, otherwise an Esri ASCII grid. Returns
    the path actually written."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        rasterio = None
    if rasterio is not None:
        tmp = path.with_name(path.stem + "_writing.tif")
        with rasterio.open(tmp, "w", driver="GTiff", width=grid.nx, height=grid.ny, count=1, dtype="float32",
                           crs=grid.crs.to_wkt(), transform=from_origin(grid.x0, grid.top, grid.cell_m, grid.cell_m),
                           nodata=NODATA, compress="deflate") as dst:
            dst.write(data, 1)
        _replace(tmp, path)
        return path
    from pyproj.enums import WktVersion
    asc = path.with_suffix(".asc")
    tmp = asc.with_name(asc.stem + "_writing.asc")
    header = (f"ncols {grid.nx}\nnrows {grid.ny}\nxllcorner {grid.x0:.3f}\nyllcorner {grid.y0:.3f}\n"
              f"cellsize {grid.cell_m:.3f}\nNODATA_value {NODATA:g}\n")
    with open(tmp, "w", encoding="ascii") as fh:
        fh.write(header)
        np.savetxt(fh, data, fmt="%.6g")
    _replace(tmp, asc)
    prj_tmp = asc.with_name(asc.stem + "_writing.prj")
    prj_tmp.write_text(grid.crs.to_wkt(WktVersion.WKT1_ESRI), encoding="ascii")
    _replace(prj_tmp, asc.with_suffix(".prj"))
    return asc


def density_dir_name(geoextent: str) -> str:
    return f"{_tok(geoextent)}_fire_activity_density"


def density_raster_files(out_dir: Path, geoextent: str) -> list[Path]:
    """This tool's density raster files in the output folder (other files are ignored)."""
    folder = Path(out_dir) / density_dir_name(geoextent)
    if not folder.exists():
        return []
    prefix = f"{_tok(geoextent)}_evnt_fir_ras_s0_nasa_pp_"
    return [f for f in folder.iterdir() if f.name.startswith(prefix)
            and f.suffix.lower() in (".tif", ".asc", ".prj", ".xml", ".ovr")]


def clear_density_rasters(out_dir: Path, geoextent: str) -> int:
    """Remove density rasters left by an earlier run. Returns how many files were removed."""
    n = 0
    for old in density_raster_files(out_dir, geoextent):
        try:
            old.unlink()
            n += 1
        except PermissionError as exc:
            raise RuntimeError(f"Could not replace {old.name}; it is probably open in ArcGIS Pro. Remove "
                               "the density rasters from your map and run again.") from exc
    return n


def _replace(tmp: Path, final: Path) -> None:
    try:
        os.replace(tmp, final)
    except PermissionError as exc:
        raise RuntimeError(f"Could not replace {final.name}; it is probably open in ArcGIS Pro. Remove it from "
                           f"your map and run again. The new version is in {tmp.name} meanwhile.") from exc


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def _tok(s: str) -> str:
    import re
    return re.sub(r"[^0-9a-z]+", "_", str(s).lower()).strip("_") or "na"


def zone_layer_name(geoextent: str, sensor_key: str, kind: str, totals: bool) -> str:
    """e.g. sct_evnt_fir_py_s0_nasa_pp_viirs_hex10km_total. Theme fir (fire
    location), polygon geometry; free text says which summary it is."""
    return f"{_tok(geoextent)}_evnt_fir_py_s0_nasa_pp_{_tok(sensor_key)}_{_tok(kind)}_{'total' if totals else 'by_period'}"


def density_file_name(geoextent: str, sensor_key: str, label: str, variant: str = "") -> str:
    """e.g. sct_evnt_fir_ras_s0_nasa_pp_viirs_density_2026_03_change.tif"""
    tail = f"_{variant}" if variant else ""
    return f"{_tok(geoextent)}_evnt_fir_ras_s0_nasa_pp_{_tok(sensor_key)}_density_{_tok(label)}{tail}.tif"


def hex_kind(size_km: float) -> str:
    return f"hex{size_km:g}km".replace(".", "p")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class SpatialResult:
    layers: list                # (layer name, GeoDataFrame in EPSG:4326)
    rasters: list               # paths written
    admin_rows: list            # rows for the admin summary CSV
    info: list                  # lines for the run information file


def spatial_summaries(aoi_geom, current: pd.DataFrame, prior: Optional[pd.DataFrame], periods, summary_rows,
                      sensor, geoextent: str, out_dir: Path, *, hex_size_km: Optional[float] = None,
                      hex_size_auto: bool = False,
                      density: str = "all", density_radius_km: Optional[float] = None,
                      admin: Optional[AdminUnits] = None,
                      burned_current: Optional[pd.DataFrame] = None,
                      burned_prior: Optional[pd.DataFrame] = None,
                      burned_pixel_km2: float = 0.0,
                      burned_pixel_m: float = 0.0,
                      log: Callable[[str], None] = print) -> SpatialResult:
    """Build every Phase 2 output. `current` and `prior` are the point tables
    from fire_tool.build_point_tables (prior is None without a comparison).
    `burned_current` / `burned_prior` (Phase 3, optional) are burned pixel
    tables from fire_burned.burned_point_tables; when given, hexagon and admin
    layers gain burned area columns."""
    import geopandas as gpd

    if density not in DENSITY_CHOICES:
        raise ValueError(f"--density must be one of {DENSITY_CHOICES}, got {density!r}")
    compare = prior is not None
    crs = equal_area_crs(aoi_geom)
    aoi_ea = to_equal_area(aoi_geom, crs)
    area_km2 = aoi_ea.area / 1e6
    cur_ea = points_to_gdf(current, crs)
    pri_ea = points_to_gdf(prior, crs) if compare else None
    layers, rasters, admin_rows = [], [], []
    c = aoi_geom.centroid
    lat = f"{abs(c.y):.2f} {'N' if c.y >= 0 else 'S'}"
    lon = f"{abs(c.x):.2f} {'E' if c.x >= 0 else 'W'}"
    info = [f"Equal-area projection for hexagons and density: Lambert azimuthal equal area centred on {lat}, {lon}"]

    # Hexagons
    auto = hex_size_km is None or hex_size_auto
    size = auto_hex_size_km(area_km2) if hex_size_km is None else float(hex_size_km)
    check_hex_size(aoi_ea, size)
    hexes = build_hex_grid(aoi_ea, size).set_crs(crs)
    b_hex = b_admin = None
    if burned_current is not None:
        b_cur = points_to_gdf(burned_current, crs)
        b_pri = points_to_gdf(burned_prior, crs) if compare and burned_prior is not None else None
        b_hex = BurnedInput(b_cur, b_pri, burned_pixel_km2)
        b_admin = BurnedInput(b_cur, b_pri, burned_pixel_km2, snap_m=burned_pixel_m)
    hz = zone_tables(hexes, "hex_id", cur_ea, pri_ea, periods, summary_rows, sensor.key, compare, burned=b_hex)
    kind = hex_kind(size)
    layers.append((zone_layer_name(geoextent, sensor.key, kind, True), hz.totals.to_crs(4326)))
    layers.append((zone_layer_name(geoextent, sensor.key, kind, False), hz.by_period.to_crs(4326)))
    info.append(f"Hexagons: {len(hexes):,}, {size:g} km across ({hex_area_km2(size):.1f} km2 each)"
                + (", size chosen automatically for the area" if auto else ""))
    if hz.unassigned_current or hz.unassigned_prior:
        info.append(f"  Points outside every hexagon (should be none): {hz.unassigned_current} current, "
                    f"{hz.unassigned_prior} year earlier")
    if hz.unassigned_burned:
        info.append(f"  Burned pixels outside every hexagon (should be none): {hz.unassigned_burned}")

    # Admin units
    if admin is not None:
        units_ea = project_gdf(admin.gdf, crs)
        az = zone_tables(units_ea, "unit_id", cur_ea, pri_ea, periods, summary_rows, sensor.key, compare,
                         snap_m=sensor.resolution_m, rate_per_km2=1000.0, burned=b_admin)
        layers.append((zone_layer_name(geoextent, sensor.key, "admin", True), az.totals.to_crs(4326)))
        layers.append((zone_layer_name(geoextent, sensor.key, "admin", False), az.by_period.to_crs(4326)))
        tot = pd.DataFrame(az.totals.drop(columns="geometry"))
        tot.insert(2, "period", "all")
        per = pd.DataFrame(az.by_period.drop(columns="geometry"))
        admin_rows = pd.concat([per, tot], ignore_index=True).to_dict("records")
        info.append(f"Admin areas: {len(units_ea)} from {admin.source}"
                    + (f" ({admin.dropped} outside the area of interest left out)" if admin.dropped else ""))
        if az.unassigned_current or az.unassigned_prior:
            info.append(f"  Fire points not inside any admin area, even allowing {sensor.resolution_m} m for "
                        f"coastline detail: {az.unassigned_current} current, {az.unassigned_prior} year earlier. "
                        "They are counted in the hexagons and totals but not in any admin area.")
        if az.unassigned_burned:
            info.append(f"  Burned pixels not inside any admin area (allowing {burned_pixel_m:.0f} m): "
                        f"{az.unassigned_burned} current. They are in the hexagons and totals but not in any "
                        "admin area.")

    # Density
    if density == "none":
        removed = clear_density_rasters(out_dir, geoextent)
        if removed:
            log(f"Removed {removed} density raster file(s) left from an earlier run (density off)")
    else:
        radius = float(density_radius_km) if density_radius_km is not None else size
        if radius <= 0:
            raise ValueError("--density-radius-km must be greater than zero")
        grid = make_density_grid(aoi_ea, crs, radius)
        dens_dir = out_dir / density_dir_name(geoextent)
        dens_dir.mkdir(parents=True, exist_ok=True)
        # Clear this tool's rasters from an earlier run, so the folder only ever
        # holds one run's set (other files in the folder are left alone).
        clear_density_rasters(out_dir, geoextent)
        log(f"Density: {radius:g} km search radius, {grid.cell_m / 1000:g} km cells ({grid.nx} x {grid.ny})")

        def write(label, cur_pts, pri_pts, write_prior, cur_compared=None):
            cs = density_surface(cur_pts, grid)
            rasters.append(write_raster(dens_dir / density_file_name(geoextent, sensor.key, label), cs, grid))
            if write_prior:
                ps = density_surface(pri_pts, grid)
                rasters.append(write_raster(dens_dir / density_file_name(geoextent, sensor.key, label, "prior_year"),
                                            ps, grid))
                base = cs if cur_compared is None else density_surface(cur_compared, grid)
                diff = np.where(grid.inside, base - ps, NODATA).astype("float32")
                rasters.append(write_raster(dens_dir / density_file_name(geoextent, sensor.key, label, "change"),
                                            diff, grid))

        in_cmp = pri_ea[pri_ea["in_comparison"] == "yes"] if compare and len(pri_ea) else pri_ea
        compared = compared_period_labels(summary_rows) if compare else set()
        cur_cmp = cur_ea[cur_ea["period"].isin(compared)] if compare and len(cur_ea) else cur_ea
        write("total", cur_ea, in_cmp, compare, cur_cmp if compare else None)
        if density == "all":
            by_label = {r["period"]: r for r in summary_rows}
            for p in periods:
                row = by_label.get(p.label, {})
                if row.get("qa_flag") == "not_yet_available":
                    continue
                cp = cur_ea[cur_ea["period"] == p.label] if len(cur_ea) else cur_ea
                has_prior = compare and row.get("prior_fire_count", "") != ""
                pp = in_cmp[in_cmp["compares_with"] == p.label] if has_prior and len(in_cmp) else in_cmp
                write(p.label, cp, pp, has_prior)
        fmt = "GeoTIFF" if rasters and rasters[0].suffix == ".tif" else "Esri ASCII grid (rasterio not installed)"
        info.append(f"Density: quartic kernel, {radius:g} km search radius, {grid.cell_m / 1000:g} km cells, "
                    f"values in fires per km2; {len(rasters)} {fmt} file(s) in {dens_dir.name}")
        info.append("  Change rasters are current minus year earlier. The 'total' rasters cover every period "
                    "in the run combined; the total change raster uses only periods that have a year-earlier "
                    "match. Kernel spread beyond the boundary is cut off, so fires near the edge add up to "
                    "slightly less than one each.")
    return SpatialResult(layers, rasters, admin_rows, info)
