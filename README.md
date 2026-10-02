---
tags: [readme, fire-activity-tool, tools]
---

# Fire activity tool

Counts and maps active fire detections and burned area inside an area of interest, by week, month or year, and compares each period with the same period a year earlier. Results come as fire points, hexagon counts, counts by admin area, kernel density rasters and burned area rasters. Data comes from Google Earth Engine: active fires from VIIRS 375 m (from 3 September 2023) or MODIS 1 km (from November 2000); burned area from MODIS MCD64A1 500 m (monthly, from November 2000, about three months behind). No arcpy needed; outputs open directly in ArcGIS Pro.

Status: complete (2 October 2026). Version 1.0.

Three ways to run it, all doing exactly the same work: the **window** (`python fire_gui.py`), the **wizard** (`python fire_tool.py` with nothing after it, which asks questions), or a **typed command** with options. The wizard prints the typed command for the run it set up, and the window shows it at the top of its log, so a run can be repeated or tweaked by typing.

## Set-up (once)

You need ArcGIS Pro (for its Python) and a Google account registered for Earth Engine with a Google Cloud project: sign up at https://code.earthengine.google.com/register (free for non-commercial use) and note the project ID you register, often something like `ee-yourname`.

1. Open **ArcGIS Pro > Python Command Prompt** from the Start Menu.
2. Make a conda environment for Earth Engine tools, a copy of ArcGIS Pro's own (never install packages into `arcgispro-py3` itself). Skip this if you already have one:
   ```
   conda create --name earth-engine-gis --clone arcgispro-py3
   ```
3. Switch to it and go to the folder where you saved the tool (`cd /d` also changes drive):
   ```
   activate earth-engine-gis
   cd /d "D:\tools\fire-activity-tool"
   ```
   using your own folder in place of `D:\tools\fire-activity-tool`.
4. Install the packages, then sign in to Earth Engine and save your project ID, both once per computer (replace `your-project-id` with yours):
   ```
   pip install -r requirements.txt
   earthengine authenticate
   earthengine set_project your-project-id
   ```
   The tool then uses that project without being told. To use a different project for one run, give `--ee-project`, or fill in the window's Earth Engine project box.
5. Check the packages are there:
   ```
   python -c "import ee, geopandas, pyogrio, shapely, pyproj, numpy, matplotlib, tkinter, tkcalendar; print('ok', geopandas.__version__)"
   ```
   If that errors, run `pip install -r requirements.txt` again in the same environment.
   Optional, for GeoTIFF density and burned area rasters (without it the tool writes Esri ASCII grids, `.asc`, which ArcGIS Pro also opens):
   ```
   pip install rasterio
   ```
   Charts use the **Noto Sans** font when Windows has it, otherwise Segoe UI. To get Noto Sans: download it from Google Fonts, right-click the font files > **Install for all users**, then delete matplotlib's font list so it looks again: `del %USERPROFILE%\.matplotlib\fontlist-*.json`
6. Run the offline tests (no Earth Engine needed, about 40 seconds; the window tests are skipped if no screen is available):
   ```
   python -m pytest tests -q
   ```

## Running it

### The window

```
python fire_gui.py
```

The start and end dates are calendar pickers (click the arrow), or type them as YYYY-MM-DD. If the calendar package is missing, the boxes are plain typed boxes and the log panel says how to install it.

Choose the boundary file with **Browse...** (the window then shows its features and fields, fills in the area name for chart titles, and, for a file with several features, the admin name field). Fill in the geoextent code and the dates, then:

- **Check settings (no download)** - a dry run: sensor choice, periods and download chunks, in a few seconds.
- **Run** - the real run. Progress appears in the log panel; the window stays usable. A finished run ends with **Finished.**, a failed one with **Stopped with an error** and the reason above it.
- **Check Earth Engine datasets** - the same as `--diagnose`.
- **Open output folder** - opens it in File Explorer.

The output folder defaults to `out` inside the tool folder, and the Earth Engine project box is blank, meaning the project saved with `earthengine set_project`. The year-earlier comparison is ticked by default (a typed command needs `--compare-prior-year`). Paths pasted with quotes (File Explorer's **Copy as path**) are fine.

### The wizard

```
python fire_tool.py
```

It asks for each setting in turn (press Enter to take the default in [brackets]), prints the typed command for the run, offers a dry run first, and asks before downloading anything. Its default output folder is `out` in whichever folder the prompt is in, which is the tool folder if you used the `cd /d` step above.

### Typed commands

The examples use the sample boundary in `sample_aoi`: Scotland from Natural Earth (1:10m map units, public domain, naturalearthdata.com). Swap in your own boundary file for real work.

Check what a run will do without downloading anything:
```
python fire_tool.py --aoi-file sample_aoi\scotland_admn_ad0_py_s0_naturalearthdata_pp.shp --start 2026-03-01 --end 2026-10-01 --freq monthly --compare-prior-year --dry-run
```

Real run:
```
python fire_tool.py --aoi-file sample_aoi\scotland_admn_ad0_py_s0_naturalearthdata_pp.shp --geoextent sct --start 2026-03-01 --end 2026-10-01 --freq monthly --compare-prior-year --out-dir out
```

Check the Earth Engine datasets look as the tool expects (run once, and whenever results look odd):
```
python fire_tool.py --diagnose --aoi-file sample_aoi\scotland_admn_ad0_py_s0_naturalearthdata_pp.shp
```

### Options

| Option | Meaning |
|---|---|
| `--aoi-file` | Boundary file (shapefile, GeoPackage, GeoJSON). All features are merged into one area. Very detailed boundaries are simplified automatically and the run says so |
| `--geoextent` | Location code used in every output name, e.g. `sct` |
| `--start`, `--end` | Dates as YYYY-MM-DD. `--end` is exclusive: `2026-01-01` means up to 31 December 2025. Periods snap to whole months, years or Monday-to-Sunday weeks |
| `--freq` | `weekly`, `monthly` or `annual` |
| `--compare-prior-year` | Also count the same period one year earlier (same month, year or ISO week number) and add change columns |
| `--sensor` | `auto` (default), `viirs` or `modis`. Auto uses VIIRS only when every date needed, including the comparison year, is on or after 3 September 2023; otherwise MODIS for the whole run, so comparisons never mix sensors |
| `--min-confidence` | Lowest confidence kept. Default VIIRS 1 (nominal and high), MODIS 30 |
| `--out-dir` | Output folder |
| `--vector-format` | Vector outputs (fire points, hexagons, admin areas): `gpkg` (default, one GeoPackage), `shp` (shapefiles in a folder) or `both` |
| `--ee-project` | Earth Engine Cloud project ID. Default: the project saved with `earthengine set_project` |
| `--simplify-tolerance` | Simplify the boundary by this many degrees before sending it |
| `--no-cache` | Ignore the download cache |
| `--dry-run` | Show sensor choice, periods and download chunks only |
| `--diagnose` | Print what the Earth Engine datasets look like now |
| `--hex-size-km` | Hexagon size, km across the flats. Default: picked from the area's size, aiming for about 1,000 hexagons (2 km for a district, 10 km for Scotland, 25 km at most) |
| `--admin-file` | Admin boundaries to count fires by, e.g. a council-area or ADM2 file. Default: the area-of-interest file's own features when it has more than one; otherwise no admin layers |
| `--unit-name-field` | Field with the admin area names. Default: found automatically (`ADM2_EN`, `ADM1_EN`, `NAME_2`, `NAME_1`, `name` and similar) |
| `--unit-id-field` | Field with admin codes, e.g. `ADM2_PCODE`. Must be unique. Default: areas numbered 1, 2, 3 |
| `--density` | `all` (default: whole run plus every period), `totals` (whole run only) or `none` |
| `--density-radius-km` | Kernel density search radius. Default: the hexagon size |
| `--no-burned-area` | Skip burned area (fire detections only) |
| `--burned-rasters` | `all` (default: whole run plus every period), `totals` (whole run only) or `none` |
| `--no-charts` | Skip the PNG charts |
| `--area-name` | Area name used in chart titles, e.g. `Scotland`. Default: read from the boundary file's name fields (the most detailed one that is the same for every feature: `Scotland` for the sample Scotland file or for a file of Scotland's council areas), otherwise the geoextent code |
| `--wizard` | Start the wizard (the same as giving no options at all) |

## Outputs (in `--out-dir`)

| File | Content |
|---|---|
| `{geoextent}_fire_activity_summary.csv` | One row per period: fire count, high-confidence count, day and night counts and fire radiative power (VIIRS only), days with fire, days with imagery, quality flag. With `--compare-prior-year`: prior period, prior count, the year-earlier period's days with imagery (`prior_days_with_imagery`) and quality flag (`prior_qa_flag`), change, percentage change and `comparison_note` (see quality flags below) |
| `{geoextent}_fire_activity.gpkg` | Layer `{geoextent}_evnt_fir_pt_s0_nasa_pp_{sensor}`: one point per fire pixel per day, with date, period, confidence, fire radiative power, brightness, day/night, time (UTC). With a comparison, a second layer `..._prior_year` with `compares_with` and `in_comparison` columns |
| `{geoextent}_fire_activity.gpkg` (Phase 2 layers) | `..._py_s0_nasa_pp_{sensor}_hex{size}km_total`: every hexagon, with the whole run's fire count, days with fire, fire radiative power (VIIRS) and, with a comparison, year-earlier count, `compared_fire_count` (current fires in periods that have a year-earlier match), change and percentage change. `..._hex{size}km_by_period`: one row per hexagon per period, only where either year had a fire, with real date fields for ArcGIS Pro's time slider. `..._admin_total` and `..._admin_by_period`: the same by admin area, plus area inside the area of interest and fires per 1,000 km² |
| `{geoextent}_fire_activity_shp\` | With `--vector-format shp` or `both`: the same layers as shapefiles, one per layer, named like the GeoPackage layers. Shapefile field names can be at most 10 characters, so longer names are shortened (for example `prior_burned_area_km2` becomes `pr_burn_km`) and `{geoextent}_fire_activity_shp_fields.csv` in the same folder lists every short name with its full name. Dates are real date fields, so ArcGIS Pro's time slider works. A run with only one format removes the other format's files from an earlier run |
| `{geoextent}_fire_activity_admin_summary.csv` | The admin area counts as a table: one row per area per period where either year had a fire or burned area (areas with nothing in a period are left out of that period), plus a `period = all` row for every area |
| `{geoextent}_fire_activity_density\` | Kernel density rasters in fires per km², named `..._ras_s0_nasa_pp_{sensor}_density_{period}.tif`, with `_prior_year` and `_change` (current minus year earlier) versions. `total` covers the whole run. Rasters from an earlier run of the same geoextent are cleared first, including when density is `none` |
| `{geoextent}_fire_activity_burned\` | Burned area rasters on the MODIS sinusoidal grid, named `..._ras_s0_nasa_pp_mcd64a1_burned_{period}.tif`, with `_prior_year` and `_change` versions. Values are the share of each cell that burned: at the native 463 m pixel (any area up to about 1,400 km across) that is simply 0 or 1; larger areas use cells of several whole pixels, so values are fractions. `total` shows land burned at least once in the run. No raster is written for periods not yet published. Burned area rasters from an earlier run of the same geoextent are cleared first, including when rasters or burned area are switched off |
| `{geoextent}_fire_activity_chart_detections.png` | Fire detections per period, one column each, with the same period a year earlier as a thin black line across the column. Values are written on the columns; `*` marks a period with a quality flag and `†` a period whose year-earlier period had days without imagery (both explained in the chart's footnote); a dash marks a period with no data yet |
| `{geoextent}_fire_activity_chart_burned.png` | Burned area (km²) per period, drawn the same way; `†` marks low mapped coverage a year earlier. Periods not yet published show a dash, never zero. Not drawn if no period in the run is published yet |
| `{geoextent}_fire_activity_chart_admin.png` | Fire detections for the whole run by admin area, largest first (the top 15; all areas are in the admin CSV), with the year-earlier count as a black line. Only with admin areas |
| `{geoextent}_fire_activity_run_info.txt` | Sensor used and why, settings, latest data date, flagged periods, comparisons affected by missing imagery, hexagon size, admin source, density settings, burned area month and settings. Written last, so if it matches the other files' time the run finished |

Burned area columns (summary CSV, hexagon and admin layers, admin CSV): `burned_area_km2`, `burned_pixels` (exact count; area = pixels x 0.2147 km², because the MODIS grid is equal-area), `burned_mapped_pct` (share of land MODIS could map that month; summary only), `burned_qa_flag`, `burned_note`, and with a comparison `prior_burned_area_km2`, `prior_burned_mapped_pct`, `burned_change_km2` and `burned_change_pct`. Hexagon and admin totals add `burned_share_pct` (burned area as a share of the area inside the area of interest; over a long run where the same land burns repeatedly it can pass 100) and `compared_burned_area_km2` (current burned area in periods that have a year-earlier match). Periods after the latest published month are blank and flagged `not_yet_available`, never zero. Weekly burned area is worked out from each pixel's approximate burn day, so treat it as approximate. Burned pixels are placed in hexagons and admin areas by their centre; totals add up period by period, so land that burned twice in the run counts twice there (the `total` raster counts it once).

Burned area quality flags: `ok`; `not_yet_available`; otherwise one or more of these joined with `+`: `partial_period` (only part of the period published; compared with the same calendar dates a year earlier, or the same days into the week for weekly runs); `low_coverage` (MODIS could map less than 90 per cent of the land, usually because of cloud, so burned area may be under-counted); `missing_month` (a published month has no image in the Earth Engine archive: the period gets no year-earlier comparison, and is blank if every month in it is missing); `coverage_unknown` (no coverage figure). Low coverage in the year-earlier period is written in `burned_note`. A partly published annual period is compared with the same calendar dates a year earlier, whereas fire detections use the same number of days; in a leap year the two cut-offs can be a day apart.

Quality flags (fire detections): `ok`; `partial_period` (data ends part-way through, so the comparison uses the same number of days from the year before, noted in `comparison_note`; this takes the place of `missing_days` for that period); `not_yet_available`; `missing_days`; `no_imagery`. The year-earlier period has its own flag in `prior_qa_flag`.

Missing imagery in either year makes a comparison uneven, so `comparison_note` says so: for example "year-earlier period has imagery on only 22 of 31 days, so the change is likely overstated", or the same for the current period ("likely understated"). Counts are not scaled up for missing days, because fire comes in bursts and scaling would invent fires. If either period has no imagery at all, the year-earlier count and the change are left blank (and the period is left out of the hexagon, admin and density "compared" totals). The run info lists every affected comparison. Missing satellite days are common in VIIRS near-real-time data: in the Spain test run, July 2024 had imagery on only 22 of its 31 days.

Hexagons and density are built in a Lambert azimuthal equal-area projection centred on the area of interest, so every hexagon covers the same area and densities are true per-km² values. Vector layers are saved in WGS84 like the points; rasters stay in the equal-area projection (ArcGIS Pro reprojects them on the fly). Hexagons are kept whole at the boundary; `area_in_aoi_km2` says how much of each lies inside. Percentage change is left blank where the year before had no fires. Fire pixel centres just offshore of a detailed coastline (within one pixel) are counted in the nearest admin area; any still outside every admin area are listed in the run info.

Persistent heat sources show up as fires on many days: in the March to September Scotland test run, a cluster at the Grangemouth industrial site has detections on 26 days in 2025 and 28 days in 2026, at low fire radiative power. They are in both years, so they mostly cancel out in the change figures, but they inflate the raw counts.

Counting rule: one detection is one fire pixel on one day. Data is near real time and described by NASA as not science quality.

Downloads are cached in `out\.cache\` by month (burned area in `ba_v1_...` folders, one file per whole month, shared by weekly, monthly and annual runs; every published month is cached, as the product is final), so re-running with different periods, hexagon sizes or admin files is quicker. The cache moved to version 2 on 1 October 2026 (fire pixels are now picked out by brightness temperature), so the first run after updating downloads everything again; the old `v1_...` cache folders can be deleted. Months within 7 days of the latest data are always downloaded again. Months with missing days are downloaded again in case the gap is filled, until they are 90 days old; after that the gap is treated as permanent and the month is cached.

Large areas: a run keeps every detection in memory. The Spain test (half a million km², 21 months, about 210,000 detections) took about a quarter of an hour, mostly downloading; a whole continent of VIIRS data can be millions of detections, several GB of memory and a GeoPackage of a GB or more, and the run warns above three million. For areas that size, raise `--min-confidence` to 2, use `--density totals`, or run country by country.

### Charts

Charts are 200 dots per inch PNGs with rust red columns and black year-earlier lines, Noto Sans, white background, no gridlines, months along the bottom with the year beneath, and a source footnote. Weekly charts group the weeks under the month that holds each week's Thursday (the ISO rule). With very many periods (roughly more than 60) the columns are too narrow to label; the footnote then points to the summary CSV. Charts from an earlier run of the same geoextent are removed first, so a chart that no longer applies never lingers. If drawing a chart fails, every other output is still written and the run info says why.

Redraw the charts from a finished run without downloading anything (for example after changing the title):
```
python fire_charts.py out\sct_fire_activity_summary.csv --area-name Scotland
```
Without `--area-name` it reuses the name the run used.

## If something goes wrong

Check in this order: the window is the ArcGIS Pro Python Command Prompt; the prompt shows the tool's folder, not `C:\Users\...` (use `cd /d`, which also changes drive); `earth-engine-gis` is active; a project is saved (`earthengine set_project`) or given, and it is your real project ID, not `your-project-id` copied from an example. An error saying no project was found means none is saved yet.

Files open in another program: before downloading anything, and again just before writing, the tool checks whether any output from an earlier run of the same geoextent is open (the GeoPackage, shapefiles or rasters in ArcGIS Pro, a CSV in Excel). If one is, it stops, names the files and changes nothing: remove the layers from your map or close the file, then run again. A chart PNG open in an image viewer is the exception: the run finishes, but says which chart it could not replace.

If the window does not open, check `python -c "import tkinter"` in the same prompt; tkinter comes with ArcGIS Pro's Python, so an error there usually means a different Python is running.
