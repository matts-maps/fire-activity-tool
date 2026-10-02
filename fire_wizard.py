#!/usr/bin/env python3
"""
fire_wizard.py - question-and-answer set-up for fire_tool.py, and the form-to-
options builder the GUI uses (Phase 4).

Both only ever build the same option list a typed command would use and hand
it to fire_tool.main(), so there is one code path that does the work however
the run was set up. The wizard starts on its own when fire_tool.py is run with
no options:

    python fire_tool.py

and prints the equivalent command at the end, so a run set up once in the
wizard can be repeated or tweaked by typing.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Callable, Optional

# No project is built in, so the tool can be shared: blank means the project saved
# on this computer with 'earthengine set_project <project-id>'.
DEFAULT_EE_PROJECT = ""
DEFAULT_OUT_DIR = "out"
FREQS = ("monthly", "weekly", "annual")
SENSORS = ("auto", "viirs", "modis")
RASTER_MODES = ("all", "totals", "none")
VECTOR_MODES = ("gpkg", "shp", "both")
VECTOR_OPTIONS = ["GeoPackage (one file)", "Shapefiles (one folder; field names shortened to 10 characters)",
                  "Both"]


# ---------------------------------------------------------------------------
# File helpers (local file reads only, no Earth Engine)
# ---------------------------------------------------------------------------

def file_fields(path: str) -> list[str]:
    """Non-geometry field names in a boundary file."""
    import pyogrio
    info = pyogrio.read_info(path)
    return [str(f) for f in info["fields"]]


def feature_count(path: str) -> int:
    import pyogrio
    try:
        n = int(pyogrio.read_info(path, force_feature_count=True)["features"])
    except TypeError:  # older pyogrio without force_feature_count
        n = int(pyogrio.read_info(path)["features"])
    if n < 0:  # some drivers cannot report a count without reading
        n = len(pyogrio.read_dataframe(path, read_geometry=False))
    return n


def suggest_name_field(fields: list[str]) -> Optional[str]:
    import fire_spatial
    for c in fire_spatial.COMMON_NAME_FIELDS:
        if c in fields:
            return c
    return None


def check_date(text: str, what: str) -> str:
    try:
        date.fromisoformat(text.strip())
    except ValueError:
        raise ValueError(f"{what} must be a date written YYYY-MM-DD, e.g. 2026-03-01 (got {text!r}).") from None
    return text.strip()


_CMD_SPECIAL = set(' \t&^|<>()"')


def quote_arg(a: str) -> str:
    """One argument quoted for a Windows command prompt (Python's own rules for
    backslashes before a quote, so a folder path ending in a backslash still
    works)."""
    if a and not (set(a) & _CMD_SPECIAL):
        return a
    out, bs = ['"'], 0
    for ch in a:
        if ch == "\\":
            bs += 1
            continue
        if ch == '"':
            out.append("\\" * (bs * 2 + 1) + '"')
        else:
            out.append("\\" * bs + ch)
        bs = 0
    out.append("\\" * (bs * 2) + '"')
    return "".join(out)


def quote_argv(argv: list[str]) -> str:
    """The option list as one line for a Windows command prompt."""
    return "python fire_tool.py " + " ".join(quote_arg(a) for a in argv)


def add_opt(argv: list[str], name: str, value: str) -> None:
    """Append an option and its value; a value starting with '-' is joined with
    '=' so it is not mistaken for another option."""
    if value.startswith("-"):
        argv.append(f"{name}={value}")
    else:
        argv += [name, value]


# ---------------------------------------------------------------------------
# Form values -> option list (used by the GUI; pure, tested offline)
# ---------------------------------------------------------------------------

def _clean(v) -> str:
    return "" if v is None else str(v).strip()


def clean_path(v) -> str:
    """A typed or pasted path, without the quotes Windows Explorer's 'Copy as path' adds."""
    return _clean(v).strip('"').strip("'").strip()


def _positive_number(v: str, what: str) -> str:
    try:
        f = float(v)
    except ValueError:
        raise ValueError(f"{what} must be a number (got {v!r}).") from None
    if not math.isfinite(f):
        raise ValueError(f"{what} must be an ordinary number (got {v!r}).")
    if f <= 0:
        raise ValueError(f"{what} must be greater than zero.")
    return v


def build_argv_from_form(fields: dict, dry_run: bool = False) -> list[str]:
    """Turn a dict of form values into fire_tool.py options. Raises ValueError
    with a plain message for anything missing or malformed, so the GUI can
    show it before starting a run.

    Keys (blank or missing = not set): aoi_file (required), geoextent
    (required unless dry_run), start, end (required, YYYY-MM-DD), freq
    (required), compare (bool), sensor, min_confidence, out_dir (required
    unless dry_run), ee_project, admin_file, unit_name_field, unit_id_field,
    hex_size_km, density, density_radius_km, burned_area (bool, default
    True), burned_rasters, charts (bool, default True), area_name,
    simplify_tolerance, no_cache (bool), vector_format (gpkg, shp or both)."""
    f = {k: v for k, v in fields.items()}
    argv: list[str] = []
    aoi = clean_path(f.get("aoi_file"))
    if not aoi:
        raise ValueError("Choose an area-of-interest boundary file.")
    add_opt(argv, "--aoi-file", aoi)
    geo = _clean(f.get("geoextent"))
    if geo:
        add_opt(argv, "--geoextent", geo)
    elif not dry_run:
        raise ValueError("Enter a geoextent code for the output names, e.g. sct.")
    start = _clean(f.get("start"))
    end = _clean(f.get("end"))
    if not start or not end:
        raise ValueError("Enter both a start date and an end date (YYYY-MM-DD).")
    start, end = check_date(start, "Start date"), check_date(end, "End date")
    if date.fromisoformat(end) <= date.fromisoformat(start):
        raise ValueError("The end date must be after the start date (the end date itself is not included).")
    argv += ["--start", start, "--end", end]
    freq = _clean(f.get("freq")) or "monthly"
    if freq not in FREQS:
        raise ValueError(f"Frequency must be one of {', '.join(FREQS)}.")
    argv += ["--freq", freq]
    if f.get("compare"):
        argv.append("--compare-prior-year")
    sensor = _clean(f.get("sensor")) or "auto"
    if sensor not in SENSORS:
        raise ValueError(f"Sensor must be one of {', '.join(SENSORS)}.")
    if sensor != "auto":
        argv += ["--sensor", sensor]
    mc = _clean(f.get("min_confidence"))
    if mc:
        if not (mc.isascii() and mc.isdigit()):
            raise ValueError("Minimum confidence must be a whole number, 0 or more (VIIRS 0 to 2, MODIS 0 to 100).")
        argv += ["--min-confidence", mc]
    out_dir = clean_path(f.get("out_dir"))
    if out_dir:
        add_opt(argv, "--out-dir", out_dir)
    elif not dry_run:
        raise ValueError("Choose an output folder.")
    ee = _clean(f.get("ee_project"))
    if ee:
        add_opt(argv, "--ee-project", ee)
    vector = _clean(f.get("vector_format")) or "gpkg"
    if vector not in VECTOR_MODES:
        raise ValueError(f"Vector output must be one of {', '.join(VECTOR_MODES)} (got {vector!r}).")
    if vector != "gpkg":
        argv += ["--vector-format", vector]
    tol = _clean(f.get("simplify_tolerance"))
    if tol:
        argv += ["--simplify-tolerance", _positive_number(tol, "Simplify tolerance")]
    admin = clean_path(f.get("admin_file"))
    if admin:
        add_opt(argv, "--admin-file", admin)
    name_f, id_f = _clean(f.get("unit_name_field")), _clean(f.get("unit_id_field"))
    if name_f:
        add_opt(argv, "--unit-name-field", name_f)
    if id_f:
        add_opt(argv, "--unit-id-field", id_f)
    hexs = _clean(f.get("hex_size_km"))
    if hexs:
        argv += ["--hex-size-km", _positive_number(hexs, "Hexagon size")]
    density = _clean(f.get("density")) or "all"
    if density not in RASTER_MODES:
        raise ValueError(f"Density rasters must be one of {', '.join(RASTER_MODES)}.")
    if density != "all":
        argv += ["--density", density]
    radius = _clean(f.get("density_radius_km"))
    if radius:
        argv += ["--density-radius-km", _positive_number(radius, "Density search radius")]
    if f.get("burned_area", True):
        br = _clean(f.get("burned_rasters")) or "all"
        if br not in RASTER_MODES:
            raise ValueError(f"Burned area rasters must be one of {', '.join(RASTER_MODES)}.")
        if br != "all":
            argv += ["--burned-rasters", br]
    else:
        argv.append("--no-burned-area")
    if f.get("charts", True):
        area = _clean(f.get("area_name"))
        if area:
            add_opt(argv, "--area-name", area)
    else:
        argv.append("--no-charts")
    if f.get("no_cache"):
        argv.append("--no-cache")
    if dry_run:
        argv.append("--dry-run")
    return argv


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------

class Prompter:
    """Question helpers. `ask` is input() normally; tests pass a fake."""

    def __init__(self, ask: Callable[[str], str] = input, say: Callable[[str], None] = print):
        self.ask = ask
        self.say = say

    def text(self, label: str, default: Optional[str] = None, required: bool = True) -> str:
        suffix = f" [{default}]" if default else ""
        while True:
            val = self.ask(f"{label}{suffix}: ").strip().strip('"')
            if val:
                return val
            if default is not None:
                return default
            if not required:
                return ""
            self.say("  (needed - please enter a value)")

    def choice(self, label: str, options: list[str], default: int = 0) -> int:
        self.say(f"\n{label}:")
        for i, opt in enumerate(options, 1):
            self.say(f"  {i}. {opt}" + ("  (default)" if i - 1 == default else ""))
        while True:
            val = self.ask(f"Choose 1-{len(options)} [{default + 1}]: ").strip()
            if not val:
                return default
            if val.isdigit() and 1 <= int(val) <= len(options):
                return int(val) - 1
            self.say(f"  Please enter a number from 1 to {len(options)}.")

    def yes_no(self, label: str, default: bool = True) -> bool:
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            val = self.ask(f"{label} {suffix}: ").strip().lower()
            if not val:
                return default
            if val in ("y", "yes"):
                return True
            if val in ("n", "no"):
                return False
            self.say("  Please answer y or n.")

    def date(self, label: str, default: Optional[str] = None) -> str:
        while True:
            val = self.text(label, default)
            try:
                return check_date(val, label)
            except ValueError as e:
                self.say(f"  {e}")

    def number(self, label: str, blank_ok: bool = True) -> str:
        while True:
            val = self.text(label, default="" if blank_ok else None, required=not blank_ok)
            if not val:
                return ""
            try:
                return _positive_number(val, label)
            except ValueError as e:
                self.say(f"  {e}")

    def existing_file(self, label: str, required: bool = True) -> str:
        while True:
            val = self.text(label, default=None if required else "", required=required)
            if not val:
                return ""
            if Path(val).exists():
                return val
            self.say(f"  Can't find {val} - check the path (relative paths start from the tool folder).")

    def field(self, label: str, fields: list[str], default: Optional[str] = None, blank_ok: bool = False) -> str:
        while True:
            val = self.text(label, default=default if default else ("" if blank_ok else None),
                            required=not blank_ok)
            if not val or not fields or val in fields:
                return val
            self.say(f"  '{val}' is not one of the fields listed above - type one exactly"
                     + (", or leave blank." if blank_ok else "."))


def run_wizard(p: Optional[Prompter] = None) -> tuple[list[str], bool]:
    """Ask for every setting and return (option list, check first with a dry run)."""
    p = p or Prompter()
    say = p.say
    say("Fire activity tool - set-up questions")
    say("Press Enter to take the default shown in [brackets]. Ctrl+C cancels.\n")
    argv: list[str] = []

    aoi = p.existing_file("Area-of-interest boundary file (shapefile, GeoPackage or GeoJSON)")
    add_opt(argv, "--aoi-file", aoi)
    try:
        n_feat = feature_count(aoi)
        aoi_fields = file_fields(aoi)
    except Exception as e:  # noqa: BLE001 - the run itself reports a bad file properly
        say(f"  (Couldn't read {aoi} to look inside it: {e})")
        n_feat, aoi_fields = 1, []
    say(f"  {n_feat} feature(s)" + (f"; fields: {', '.join(aoi_fields)}" if aoi_fields else ""))
    add_opt(argv, "--geoextent", p.text("Geoextent code for the output names, e.g. sct"))

    say("\nDates. The end date is not included: 2026-10-01 means up to 30 September 2026.")
    while True:
        start = p.date("Start date (YYYY-MM-DD)")
        end = p.date("End date (YYYY-MM-DD)")
        if date.fromisoformat(end) > date.fromisoformat(start):
            break
        say("  The end date must be after the start date.")
    argv += ["--start", start, "--end", end]
    freq = FREQS[p.choice("Period length", ["Monthly", "Weekly (Monday to Sunday)", "Annual"], 0)]
    argv += ["--freq", freq]
    if p.yes_no("Compare each period with the same period a year earlier?", True):
        argv.append("--compare-prior-year")
    s = p.choice("Fire sensor", [
        "Automatic: VIIRS 375 m when every date needed is from 3 September 2023, otherwise MODIS 1 km",
        "VIIRS 375 m only",
        "MODIS 1 km only (back to November 2000)"], 0)
    if s:
        argv += ["--sensor", SENSORS[s]]

    say("\nAdmin areas (fire counts per area, as a layer and a CSV).")
    if n_feat > 1:
        say(f"  The area-of-interest file has {n_feat} features, so they are used as the admin areas "
            "unless you give a separate file.")
    admin = p.existing_file("Separate admin boundary file (blank to skip)", required=False)
    names_from = admin or (aoi if n_feat > 1 else "")
    if admin:
        add_opt(argv, "--admin-file", admin)
    if names_from:
        try:
            fields = file_fields(names_from)
        except Exception as e:  # noqa: BLE001
            say(f"  (Couldn't list the fields in {names_from}: {e})")
            fields = []
        if fields:
            say(f"  Fields in {Path(names_from).name}: {', '.join(fields)}")
        guess = suggest_name_field(fields)
        name_f = p.field("Field holding the area names", fields, default=guess, blank_ok=bool(not admin))
        if name_f:
            add_opt(argv, "--unit-name-field", name_f)
        id_f = p.field("Field holding a unique code for each area, e.g. ADM2_PCODE (blank to number them)",
                       fields, blank_ok=True)
        if id_f:
            add_opt(argv, "--unit-id-field", id_f)

    say("\nSpatial summaries.")
    hexs = p.number("Hexagon size in km across (blank = automatic, about 1,000 hexagons)")
    if hexs:
        argv += ["--hex-size-km", hexs]
    d = p.choice("Kernel density rasters", ["Whole run and every period", "Whole run only", "None"], 0)
    if d:
        argv += ["--density", RASTER_MODES[d]]
    if d != 2:  # a radius only matters if some density raster is made
        radius = p.number("Density search radius in km (blank = the hexagon size)")
        if radius:
            argv += ["--density-radius-km", radius]

    say("\nBurned area (MODIS, 500 m, monthly, about three months behind).")
    if p.yes_no("Include burned area?", True):
        b = p.choice("Burned area rasters", ["Whole run and every period", "Whole run only", "None"], 0)
        if b:
            argv += ["--burned-rasters", RASTER_MODES[b]]
    else:
        argv.append("--no-burned-area")

    if p.yes_no("\nDraw PNG charts?", True):
        import fire_charts
        guess = fire_charts.area_name_from_file(aoi, "")
        area = p.text("Area name for the chart titles", default=guess or None)
        if area and area != guess:
            add_opt(argv, "--area-name", area)
    else:
        argv.append("--no-charts")

    add_opt(argv, "--out-dir", p.text("\nOutput folder", DEFAULT_OUT_DIR))
    v = p.choice("Vector outputs (fire points, hexagons, admin areas)", VECTOR_OPTIONS, 0)
    if v:
        argv += ["--vector-format", VECTOR_MODES[v]]
    ee = p.text("Earth Engine project ID (blank = the one saved with 'earthengine set_project')",
                DEFAULT_EE_PROJECT, required=False)
    if ee:
        add_opt(argv, "--ee-project", ee)
    if p.yes_no("Change any advanced settings (minimum confidence, boundary simplifying, download cache)?",
                False):
        while True:
            mc = p.text("Minimum confidence (blank = default: VIIRS 1 = nominal and high, MODIS 30)", default="",
                        required=False)
            if not mc or (mc.isascii() and mc.isdigit()):
                break
            say("  Please enter a whole number, 0 or more, or leave blank.")
        if mc:
            argv += ["--min-confidence", mc]
        tol = p.number("Simplify the boundary by this many degrees before sending it, e.g. 0.001 "
                       "(blank = automatic, only when the boundary is very detailed)")
        if tol:
            argv += ["--simplify-tolerance", tol]
        if p.yes_no("Ignore the download cache and download everything again?", False):
            argv.append("--no-cache")
    check_first = p.yes_no("Check the settings first without downloading anything (dry run)?", True)
    return argv, check_first


def wizard_main(tool_main: Callable[[list[str]], int], p: Optional[Prompter] = None) -> int:
    """Run the wizard, then the tool, through fire_tool.main."""
    p = p or Prompter()
    try:
        argv, check_first = run_wizard(p)
        p.say("\nThe same run as a typed command (copy it to repeat or adjust the run):")
        p.say("  " + quote_argv(argv) + "\n")
        if check_first:
            code = tool_main(argv + ["--dry-run"])
            if code != 0:
                return code
            if not p.yes_no("\nGo ahead with the real run?", True):
                p.say("Stopped before downloading anything.")
                return 0
        return tool_main(argv)
    except (KeyboardInterrupt, EOFError):
        p.say("\nCancelled.")
        return 1
