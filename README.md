# VTA — Visualization Tool for Astronomy

A standalone desktop FITS image viewer for Python, built on
PySide6 + pyqtgraph + astropy. VTA is a modern reimplementation, in spirit,
of Aaron Barth's IDL **ATV**, aimed at quick interactive inspection of
astronomical images — display, photometry, source separations, and
spectral extraction in one window.

![VTA photometry tab](docs/screenshot_photometry.png)

## Highlights

- Interactive FITS display with the usual stretches, scalings, and colormaps
- imexam-style aperture photometry with a live radial profile and FWHM
- Source-to-source separations with position angle and propagated errors
- Trace-and-extract spectral extraction
- Box statistics, row/column/vector cuts, blink/RGB, WCS-aware overlays

## Features

- **Display:** FITS loading (multi-extension, primary-header inheritance),
  3-D cube stepping with median/average combining, zoom / pan / recenter,
  an adjustable image/analysis splitter, magnifier, and an always-on N/E
  compass.
- **Stretches & scaling:** linear, log, sqrt, asinh (adjustable β),
  histogram equalization; AutoScale / ZScale / Full / manual limits.
- **Color:** red-orange (default), red-white, grey, blue-white, plus
  viridis/inferno/magma/cividis/turbo and rainbow, with inversion and
  mouse brightness/contrast.
- **Coordinates:** cursor readout in J2000 (sexagesimal or degrees),
  B1950, Galactic, Ecliptic, or pixels.
- **Analysis tabs:** Photometry, Statistics, Separations, and Spectrum
  (described below).
- **Cuts:** independent row / column / arbitrary-vector cut windows, with
  an optional angular (arcsec/arcmin/degree) x-axis.
- **Annotation & manipulation:** arrows, compass, scale bar, contours;
  WCS-preserving rotations/flips; blink buffers (with a click-through Blink
  mode) and RGB composites; FITS / spectrum / plot saving.

## The analysis tabs

### Photometry

ATV/DAOPHOT aperture photometry with centroiding. The radial profile and
FWHM can be shown in pixels or arcsec (when a celestial WCS is present),
the aperture/sky circles can be toggled on the image, and measurements can
be logged to a CSV file.

![VTA photometry tab](docs/screenshot_photometry.png)

### Statistics

Box statistics around the cursor (total, min/max, mean, median, σ), a
rendered subimage, and a pixel-value histogram.

![VTA statistics tab](docs/screenshot_statistics.png)

### Separations

imexam one source as a reference, then another as a target, to get the
pixel offsets, on-sky ΔRA/ΔDec, total separation, position angle (East of
North), and Δmag — each with propagated uncertainties from the centroid
S/N — accumulated into a table and exportable to CSV.

![VTA separations tab](docs/screenshot_separations.png)

### Spectrum

Click a trace in `spectrum` mode to trace and extract a 1-D spectrum (a
port of ATV's `atvextract`): iterative trace centroiding, polynomial trace
fit, partial-pixel aperture summation, and background subtraction. All
extraction parameters are editable in the tab and re-extract live; the
result can be saved as 1-D FITS or text.

![VTA spectrum tab](docs/screenshot_spectrum.png)

## Installation

VTA needs **Python 3.10+** and the packages in `requirements.txt`
(PySide6, pyqtgraph, astropy, numpy, scipy, photutils, matplotlib). All of
them ship as pre-built wheels on every major platform, so no compiler is
required. VTA itself is a single pure-Python file and is OS-agnostic — the
only differences below are how you get Python and the Qt system libraries.

First get the code (any platform):

```bash
git clone https://github.com/ciardi/vta.git
cd vta
```

A virtual environment is recommended but optional. The per-platform notes
below cover the Python install and any extra system libraries.

### macOS (recommended: Homebrew Python)

```bash
brew install python                       # if you don't already have it
python3 -m pip install -r requirements.txt
python3 vta.py image.fits
```

Qt's macOS (Cocoa) plugin is bundled in the PySide6 wheel, so there are no
extra system libraries to install. Works on both Apple-silicon and Intel
Macs. The system Python that ships with macOS also works, but a Homebrew
(or python.org) Python is cleaner and easier to keep current.

### macOS via a Unix/X11 setup (MacPorts, fink, or forcing X11)

If you are running a Unix-style stack on macOS (e.g. MacPorts Python) the
install is the same:

```bash
python3 -m pip install -r requirements.txt
python3 vta.py image.fits
```

VTA uses Qt's native Cocoa backend by default, which is what you want — you
do **not** need XQuartz. If you have a reason to force X11 (e.g. remote
display through an X server), install XQuartz and set
`QT_QPA_PLATFORM=xcb`; otherwise leave it unset and Cocoa is used
automatically.

### Linux (native)

```bash
python3 -m pip install -r requirements.txt
```

On a minimal/headless Linux you may also need the Qt xcb runtime
libraries. On Debian/Ubuntu:

```bash
sudo apt install libxcb-cursor0
```

(Other distributions: install the equivalent `xcb` / `libxcb-cursor`
package.) Then:

```bash
python3 vta.py image.fits
```

### WSL2 (Windows Subsystem for Linux)

Treat WSL2 like native Linux. With a recent Windows 11 + WSL2 (WSLg
provides the display automatically), install the same xcb libraries and
run it:

```bash
sudo apt install libxcb-cursor0
python3 -m pip install -r requirements.txt
python3 vta.py image.fits
```

VTA automatically selects the `xcb` Qt platform under WSL2, so no manual
`QT_QPA_PLATFORM` setting is needed. (Note: a WSL2 Python environment is
separate from any native-Windows Python — install the requirements in
whichever one you run VTA from.)

### Windows (native Python)

Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
or the Microsoft Store, then from PowerShell or Command Prompt:

```bat
python -m pip install -r requirements.txt
python vta.py image.fits
```

Qt's Windows plugin is self-contained in the PySide6 wheel, so **none** of
the Linux `xcb` steps apply. A native window opens directly. On a high-DPI
(4K) display, if scaling looks off you can set:

```bat
set QT_ENABLE_HIGHDPI_SCALING=1
```

## Usage

```bash
python vta.py                # open with no image
python vta.py image.fits     # open a FITS file directly
```

In the GUI, see **Help ▸ VTA help** (or press **F1**) for the full guide.
Pick an interaction **mode** from the toolbar pulldown (scan, color, zoom,
imexam, vector, row, col, spectrum, blink) and click on the image.

## Status

VTA is in active development and provided as-is. Bug reports and feature
requests are welcome via the Issues tab.

## Heritage & credits

Heritage: **ATV** by Aaron Barth. VTA is an independent Python
reimplementation and is not affiliated with or derived from the ATV source
code; it reimplements comparable functionality and ports several of ATV's
algorithms (centroiding, aperture photometry, spectral tracing).

Author: David R. Ciardi.

## License

Released under the MIT License — see [LICENSE](LICENSE).
