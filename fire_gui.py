#!/usr/bin/env python3
"""
fire_gui.py - window front end for the fire activity tool (Phase 4).

Same one-code-path rule as the wizard: the form is turned into the option list
a typed command would use (fire_wizard.build_argv_from_form) and passed to
fire_tool.main(), so the GUI cannot drift from the command line. This file is
only window glue.

Run it from ArcGIS Pro's Python Command Prompt with earth-engine-gis active:

    python fire_gui.py

The date boxes are calendar pickers (tkcalendar, in requirements.txt). If it is
missing, the window still opens with typed date boxes and says how to install it.
"""

from __future__ import annotations

import io
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import fire_wizard
from fire_wizard import DEFAULT_EE_PROJECT, build_argv_from_form, file_fields, feature_count, quote_argv

try:
    from tkcalendar import DateEntry
except ImportError:  # pragma: no cover - the window says how to install it
    DateEntry = None

CALENDAR_MISSING = ("The calendar picker is not installed, so the date boxes are typed (YYYY-MM-DD). To add it, "
                    "close this window and run:  pip install tkcalendar  in the same prompt, then reopen.")

HERE = Path(__file__).resolve().parent

FREQ_LABELS = {"Monthly": "monthly", "Weekly (Monday to Sunday)": "weekly", "Annual": "annual"}
SENSOR_LABELS = {
    "Automatic (VIIRS from 3 September 2023, else MODIS)": "auto",
    "VIIRS 375 m only": "viirs",
    "MODIS 1 km only": "modis",
}
RASTER_LABELS = {"Whole run and every period": "all", "Whole run only": "totals", "None": "none"}
VECTOR_LABELS = {"GeoPackage (one file)": "gpkg", "Shapefiles (one folder)": "shp", "Both": "both"}


class _QueueWriter(io.TextIOBase):
    """Text stream that sends print() output from the worker thread to the log
    panel. A real TextIOBase (with encoding and isatty) because some libraries
    look for those on sys.stdout."""

    def __init__(self, q: "queue.Queue[str]") -> None:
        super().__init__()
        self._q = q

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return "utf-8"

    @property
    def errors(self) -> str:  # type: ignore[override]
        return "replace"

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if text:
            self._q.put(str(text))
        return len(text)

    def flush(self) -> None:  # pragma: no cover
        pass


class FireGUI:
    """The window. A plain class taking a Tk root so a test can build it, set
    values and call its methods directly."""

    def __init__(self, root: tk.Tk, tool_main=None) -> None:
        self.root = root
        self.root.title("Fire activity tool")
        self.root.geometry("940x760")
        self.root.minsize(640, 420)
        self._tool_main = tool_main
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._aoi_features = 0
        self._loaded_aoi = None      # path whose details are currently shown
        self._loaded_admin = None
        self._area_name_auto = ""
        self._build()
        self._refresh_states()
        self._poll()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build(self) -> None:
        canvas = tk.Canvas(self.root, highlightthickness=0)
        bar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self._canvas = canvas
        frm = ttk.Frame(canvas, padding=10)
        win = canvas.create_window((0, 0), window=frm, anchor="nw")
        frm.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(seq, self._on_wheel)
        frm.columnconfigure(1, weight=1)
        self._frm = frm
        self._row = 0

        def heading(text):
            if self._row:
                ttk.Separator(frm).grid(row=self._row, column=0, columnspan=3, sticky="ew", pady=(8, 2))
                self._row += 1
            ttk.Label(frm, text=text, font=("", 10, "bold")).grid(row=self._row, column=0, columnspan=3,
                                                                  sticky="w", padx=6, pady=3)
            self._row += 1

        def row(label, widget, button=None, hint=None):
            ttk.Label(frm, text=label).grid(row=self._row, column=0, sticky="w", padx=6, pady=3)
            widget.grid(row=self._row, column=1, sticky="w", padx=6, pady=3)
            if button is not None:
                button.grid(row=self._row, column=2, sticky="w", padx=6, pady=3)
            self._row += 1
            if hint:
                ttk.Label(frm, text=hint, foreground="#666").grid(row=self._row, column=1, columnspan=2,
                                                                  sticky="w", padx=6)
                self._row += 1
            return widget

        heading("Area of interest")
        self.aoi_file = tk.StringVar()
        aoi_entry = row("Boundary file", ttk.Entry(frm, textvariable=self.aoi_file, width=58),
                        ttk.Button(frm, text="Browse...", command=self._browse_aoi))
        aoi_entry.bind("<FocusOut>", lambda e: self.sync_files())
        self.aoi_info = tk.StringVar(value="Shapefile, GeoPackage or GeoJSON. All features are merged into one area.")
        ttk.Label(frm, textvariable=self.aoi_info, foreground="#666", wraplength=560).grid(
            row=self._row, column=1, columnspan=2, sticky="w", padx=6)
        self._row += 1
        self.geoextent = tk.StringVar()
        row("Geoextent code for output names", ttk.Entry(frm, textvariable=self.geoextent, width=12),
            hint="e.g. sct; every output file name starts with it")

        heading("Dates")
        self.start = tk.StringVar()
        self.end = tk.StringVar()
        row("Start date (YYYY-MM-DD)", self._date_widget(self.start))
        row("End date (YYYY-MM-DD, not included)", self._date_widget(self.end),
            hint="2026-10-01 means up to 30 September 2026")
        self.freq = tk.StringVar(value="Monthly")
        row("Period length", ttk.Combobox(frm, textvariable=self.freq, values=list(FREQ_LABELS),
                                          state="readonly", width=28))
        self.compare = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Compare each period with the same period a year earlier",
                        variable=self.compare).grid(row=self._row, column=1, columnspan=2, sticky="w", padx=6)
        self._row += 1
        self.sensor = tk.StringVar(value=list(SENSOR_LABELS)[0])
        row("Fire sensor", ttk.Combobox(frm, textvariable=self.sensor, values=list(SENSOR_LABELS),
                                        state="readonly", width=48))
        self.min_confidence = tk.StringVar()
        row("Minimum confidence", ttk.Entry(frm, textvariable=self.min_confidence, width=8),
            hint="blank = default (VIIRS 1 = nominal and high; MODIS 30)")

        heading("Admin areas")
        self.admin_file = tk.StringVar()
        admin_entry = row("Admin boundary file (optional)", ttk.Entry(frm, textvariable=self.admin_file, width=58),
                          ttk.Button(frm, text="Browse...", command=self._browse_admin))
        admin_entry.bind("<FocusOut>", lambda e: self.sync_files())
        self.admin_info = tk.StringVar(value="Leave blank for none, or to use the area-of-interest file's own "
                                             "features when it has more than one.")
        ttk.Label(frm, textvariable=self.admin_info, foreground="#666", wraplength=560).grid(
            row=self._row, column=1, columnspan=2, sticky="w", padx=6)
        self._row += 1
        self.unit_name_field = tk.StringVar()
        self.unit_name_combo = row("Area name field", ttk.Combobox(frm, textvariable=self.unit_name_field,
                                                                   width=28))
        self.unit_id_field = tk.StringVar()
        self.unit_id_combo = row("Unique code field (optional)",
                                 ttk.Combobox(frm, textvariable=self.unit_id_field, width=28),
                                 hint="e.g. ADM2_PCODE; blank numbers the areas 1, 2, 3")

        heading("Hexagons and density")
        self.hex_size = tk.StringVar()
        row("Hexagon size, km across", ttk.Entry(frm, textvariable=self.hex_size, width=8),
            hint="blank = automatic (about 1,000 hexagons; Scotland 10 km)")
        self.density = tk.StringVar(value=list(RASTER_LABELS)[0])
        self.density_combo = row("Density rasters", ttk.Combobox(frm, textvariable=self.density,
                                                                 values=list(RASTER_LABELS), state="readonly",
                                                                 width=28))
        self.density_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_states())
        self.radius = tk.StringVar()
        self.radius_entry = row("Density search radius, km", ttk.Entry(frm, textvariable=self.radius, width=8),
                                hint="blank = the hexagon size")

        heading("Burned area (MODIS, 500 m, monthly, about three months behind)")
        self.burned = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Include burned area", variable=self.burned,
                        command=self._refresh_states).grid(row=self._row, column=1, sticky="w", padx=6)
        self._row += 1
        self.burned_rasters = tk.StringVar(value=list(RASTER_LABELS)[0])
        self.burned_combo = row("Burned area rasters", ttk.Combobox(frm, textvariable=self.burned_rasters,
                                                                    values=list(RASTER_LABELS),
                                                                    state="readonly", width=28))

        heading("Charts")
        self.charts = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Draw PNG charts (detections, burned area, admin areas)", variable=self.charts,
                        command=self._refresh_states).grid(row=self._row, column=1, columnspan=2, sticky="w",
                                                           padx=6)
        self._row += 1
        self.area_name = tk.StringVar()
        self.area_name_entry = row("Area name for chart titles",
                                   ttk.Entry(frm, textvariable=self.area_name, width=30),
                                   hint="filled in from the boundary file when it has a name field")

        heading("Output and Earth Engine")
        self.out_dir = tk.StringVar(value=str(HERE / "out"))
        row("Output folder", ttk.Entry(frm, textvariable=self.out_dir, width=58),
            ttk.Button(frm, text="Browse...", command=self._browse_out))
        self.vector_format = tk.StringVar(value=list(VECTOR_LABELS)[0])
        row("Vector outputs", ttk.Combobox(frm, textvariable=self.vector_format, values=list(VECTOR_LABELS),
                                           state="readonly", width=26),
            hint="shapefiles: field names shortened to 10 characters; full names in a CSV beside them")
        self.ee_project = tk.StringVar(value=DEFAULT_EE_PROJECT)
        row("Earth Engine project ID", ttk.Entry(frm, textvariable=self.ee_project, width=20),
            hint="blank = the project saved on this computer with 'earthengine set_project'")
        self.simplify = tk.StringVar()
        row("Simplify boundary, degrees", ttk.Entry(frm, textvariable=self.simplify, width=10),
            hint="blank = automatic (only very detailed boundaries are simplified); e.g. 0.001 is about 100 m")
        self.no_cache = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Ignore the download cache (download everything again)",
                        variable=self.no_cache).grid(row=self._row, column=1, columnspan=2, sticky="w", padx=6)
        self._row += 1

        buttons = ttk.Frame(frm)
        buttons.grid(row=self._row, column=0, columnspan=3, sticky="w", padx=6, pady=(10, 4))
        self._row += 1
        self.check_button = ttk.Button(buttons, text="Check settings (no download)",
                                       command=lambda: self._start(dry_run=True))
        self.run_button = ttk.Button(buttons, text="Run", command=lambda: self._start(dry_run=False))
        self.diagnose_button = ttk.Button(buttons, text="Check Earth Engine datasets", command=self._diagnose)
        self.open_button = ttk.Button(buttons, text="Open output folder", command=self._open_out)
        for i, b in enumerate((self.check_button, self.run_button, self.diagnose_button, self.open_button)):
            b.grid(row=0, column=i, padx=(0, 8))
        self.progress = ttk.Progressbar(buttons, mode="indeterminate", length=140)
        self.progress.grid(row=0, column=4, padx=(8, 0))

        self.log = tk.Text(frm, height=16, width=110, state="disabled", wrap="word", font=("Consolas", 9))
        self.log.grid(row=self._row, column=0, columnspan=3, sticky="nsew", padx=6, pady=4)
        frm.rowconfigure(self._row, weight=1)
        if DateEntry is None:
            self._append(CALENDAR_MISSING + "\n\n")

    def _date_widget(self, var: tk.StringVar) -> tk.Widget:
        if DateEntry is not None:
            had = var.get()
            w = DateEntry(self._frm, textvariable=var, date_pattern="yyyy-mm-dd", width=12)
            if not had:
                var.set("")  # DateEntry fills in today; keep the box empty until chosen
            return w
        return ttk.Entry(self._frm, textvariable=var, width=14)

    def _on_wheel(self, event) -> None:
        if getattr(event, "num", None) == 4:
            d = -1
        elif getattr(event, "num", None) == 5:
            d = 1
        else:
            d = -1 if event.delta > 0 else 1
        if isinstance(event.widget, tk.Text):
            return  # let the log panel scroll itself
        self._canvas.yview_scroll(d, "units")

    # ------------------------------------------------------------------
    # Reacting to choices
    # ------------------------------------------------------------------
    def _refresh_states(self) -> None:
        self.burned_combo.configure(state="readonly" if self.burned.get() else "disabled")
        self.area_name_entry.configure(state="normal" if self.charts.get() else "disabled")
        dens_none = RASTER_LABELS.get(self.density.get()) == "none"
        self.radius_entry.configure(state="disabled" if dens_none else "normal")

    def set_aoi(self, path: str) -> None:
        """Load an area-of-interest file: count features, fill in the area name
        and, for a multi-feature file, the admin field lists."""
        import fire_charts
        self.aoi_file.set(path)
        self._loaded_aoi = path
        try:
            n = feature_count(path)
            fields = file_fields(path)
        except Exception as e:  # noqa: BLE001 - shown, not fatal
            self._aoi_features = 0
            self.aoi_info.set(f"Couldn't read the file: {e}")
            return
        self._aoi_features = n
        self.aoi_info.set(f"{n} feature(s). Fields: {', '.join(fields) or 'none'}"
                          + (". With more than one feature they are also used as the admin areas, unless an "
                             "admin file is given below." if n > 1 else ""))
        name = fire_charts.area_name_from_file(path, "")
        if name and (not self.area_name.get() or self.area_name.get() == self._area_name_auto):
            self.area_name.set(name)
            self._area_name_auto = name
        if not self.admin_file.get().strip():
            if n > 1:
                self._fill_admin_fields(fields)
            else:
                self._fill_admin_fields([])

    def set_admin(self, path: str) -> None:
        self.admin_file.set(path)
        self._loaded_admin = path
        if not path.strip():
            self.admin_info.set("No admin file: the area-of-interest file's own features are used when it has "
                                "more than one.")
            if self._loaded_aoi:
                self.set_aoi(self._loaded_aoi)
            return
        try:
            fields = file_fields(path)
            n = feature_count(path)
        except Exception as e:  # noqa: BLE001
            self.admin_info.set(f"Couldn't read the file: {e}")
            return
        self.admin_info.set(f"{n} area(s). Fields: {', '.join(fields)}")
        self._fill_admin_fields(fields)

    def _fill_admin_fields(self, fields: list[str]) -> None:
        self.unit_name_combo["values"] = fields
        self.unit_id_combo["values"] = [""] + fields
        if self.unit_name_field.get() not in fields:
            self.unit_name_field.set(fire_wizard.suggest_name_field(fields) or "")
        if self.unit_id_field.get() and self.unit_id_field.get() not in fields:
            self.unit_id_field.set("")

    def sync_files(self) -> None:
        """Re-read the boundary files if their paths were typed or pasted
        rather than browsed, so the feature count and field lists match what
        will actually be run."""
        # Paths pasted from Explorer's 'Copy as path' come wrapped in quotes.
        for var in (self.aoi_file, self.admin_file, self.out_dir):
            cleaned = fire_wizard.clean_path(var.get())
            if cleaned != var.get():
                var.set(cleaned)
        aoi = self.aoi_file.get().strip()
        if aoi != (self._loaded_aoi or ""):
            if aoi:
                self.set_aoi(aoi)
            else:
                self._loaded_aoi, self._aoi_features = None, 0
        admin = self.admin_file.get().strip()
        if admin != (self._loaded_admin or ""):
            self.set_admin(admin)

    def _browse_aoi(self) -> None:
        p = filedialog.askopenfilename(title="Area-of-interest boundary file", filetypes=[
            ("Boundary files", "*.shp *.gpkg *.geojson *.json"), ("All files", "*.*")])
        if p:
            self.set_aoi(p)

    def _browse_admin(self) -> None:
        p = filedialog.askopenfilename(title="Admin boundary file", filetypes=[
            ("Boundary files", "*.shp *.gpkg *.geojson *.json"), ("All files", "*.*")])
        if p:
            self.set_admin(p)

    def _browse_out(self) -> None:
        p = filedialog.askdirectory(title="Output folder")
        if p:
            self.out_dir.set(p)

    def _open_out(self) -> None:
        p = Path(fire_wizard.clean_path(self.out_dir.get()) or ".")
        if not p.exists():
            messagebox.showinfo("Output folder", f"{p} doesn't exist yet - it is made on the first run.")
            return
        if hasattr(os, "startfile"):
            os.startfile(str(p))  # noqa: S606 - Windows Explorer
        else:
            messagebox.showinfo("Output folder", str(p))

    # ------------------------------------------------------------------
    # Form -> options
    # ------------------------------------------------------------------
    def collect_fields(self) -> dict:
        self.sync_files()
        admin = self.admin_file.get().strip()
        uses_admin = bool(admin) or self._aoi_features > 1
        return {
            "aoi_file": self.aoi_file.get(),
            "geoextent": self.geoextent.get(),
            "start": self.start.get(),
            "end": self.end.get(),
            "freq": FREQ_LABELS.get(self.freq.get(), "monthly"),
            "compare": self.compare.get(),
            "sensor": SENSOR_LABELS.get(self.sensor.get(), "auto"),
            "min_confidence": self.min_confidence.get(),
            "admin_file": admin,
            # Field choices only mean something when there are admin areas.
            "unit_name_field": self.unit_name_field.get() if uses_admin else "",
            "unit_id_field": self.unit_id_field.get() if uses_admin else "",
            "hex_size_km": self.hex_size.get(),
            "density": RASTER_LABELS.get(self.density.get(), "all"),
            "density_radius_km": "" if RASTER_LABELS.get(self.density.get()) == "none" else self.radius.get(),
            "burned_area": self.burned.get(),
            "burned_rasters": RASTER_LABELS.get(self.burned_rasters.get(), "all"),
            "charts": self.charts.get(),
            "area_name": self.area_name.get(),
            "out_dir": self.out_dir.get(),
            "ee_project": self.ee_project.get(),
            "vector_format": VECTOR_LABELS.get(self.vector_format.get(), "gpkg"),
            "simplify_tolerance": self.simplify.get(),
            "no_cache": self.no_cache.get(),
        }

    def build_argv(self, dry_run: bool) -> list[str]:
        return build_argv_from_form(self.collect_fields(), dry_run=dry_run)

    # ------------------------------------------------------------------
    # Running in a background thread
    # ------------------------------------------------------------------
    def _start(self, dry_run: bool) -> None:
        if self._worker and self._worker.is_alive():
            return
        try:
            argv = self.build_argv(dry_run)
        except ValueError as e:
            messagebox.showerror("Check your settings", str(e))
            return
        self._launch(argv)

    def _diagnose(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        argv = ["--diagnose"]
        self.sync_files()
        if self.aoi_file.get().strip():
            argv += ["--aoi-file", self.aoi_file.get().strip()]
        if self.ee_project.get().strip():
            argv += ["--ee-project", self.ee_project.get().strip()]
        self._launch(argv)

    def _launch(self, argv: list[str]) -> None:
        self._append(("\n" if self.log.index("end-1c") != "1.0" else "") + quote_argv(argv) + "\n\n")
        for b in (self.check_button, self.run_button, self.diagnose_button):
            b.configure(state="disabled")
        self.progress.start(10)
        self._worker = threading.Thread(target=self._work, args=(argv,), daemon=True)
        self._worker.start()

    def _work(self, argv: list[str]) -> None:
        tool_main = self._tool_main
        if tool_main is None:
            import fire_tool
            tool_main = fire_tool.main
        writer = _QueueWriter(self._log_queue)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = writer, writer
        code = 1
        try:
            code = tool_main(argv)
        except SystemExit as e:  # argparse errors
            code = e.code if isinstance(e.code, int) else 2
        except BaseException as e:  # noqa: BLE001 - shown in the log, never raised into Tk
            import traceback
            self._log_queue.put(f"\nFailed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            code = 1
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            # Always reached, so the buttons are switched back on whatever happened.
            self._log_queue.put(f"__DONE__{code if isinstance(code, int) else 1}")

    def _poll(self) -> None:
        try:
            while True:
                item = self._log_queue.get_nowait()
                if item.startswith("__DONE__"):
                    self._done(int(item[len("__DONE__"):] or 0))
                else:
                    self._append(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _done(self, code: int) -> None:
        self.progress.stop()
        self.progress.configure(value=0)
        for b in (self.check_button, self.run_button, self.diagnose_button):
            b.configure(state="normal")
        self._append("\nFinished.\n" if code == 0 else f"\nStopped with an error (code {code}) - see above.\n")

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    FireGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
