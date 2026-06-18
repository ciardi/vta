#!/usr/bin/env python3
"""
Visualization Tool for Astronomy (VTA) - an astronomical FITS image
viewer for Python.

A modern reimplementation of Aaron Barth's IDL ATV built on
PySide6 + pyqtgraph + astropy (+ scipy, photutils, matplotlib).

Features
--------
Display
    FITS loading (multi-extension, primary-header inheritance), 3-D cube
    viewing with plane stepping and median/average combining, zoom / pan /
    recenter, the ATV stretches (linear, log, sqrt, asinh with adjustable
    beta, histogram equalization) with AutoScale / ZScale / Full / manual
    limits, ATV and matplotlib colormaps with inversion and ATV-style
    mouse brightness/contrast, a magnifier panel, and a cursor readout in
    J2000 (sexagesimal or degrees), B1950, Galactic, Ecliptic, or pixel
    coordinates.
Analysis
    imexam-style click measurements: ATV/DAOPHOT aperture photometry
    (counts or magnitudes) with centroiding, radial profile + FWHM,
    box statistics with subimage and histogram; row / column / arbitrary
    vector cuts in independent plot windows (vector cuts can use an
    angular x axis from the WCS pixel scale); spectral extraction with
    iterative trace centroiding, polynomial trace fit, partial-pixel
    aperture summation, and background subtraction (port of atvextract).
Annotation & manipulation
    Dialog-driven labels (mouse-drawn arrows, compass, scale bar,
    contours), exact rotations/flips that transform the WCS with the
    pixels, arbitrary-angle rotation, blink buffers with auto-blink,
    RGB composites, and FITS / spectrum / plot saving.

The numerical core (stretches, WCS handling, photometry, statistics,
profiles, cuts, tracing/extraction) is kept as plain functions with no
Qt dependence so it can be unit-tested and reused; the GUI is built on
top by build_gui().

Usage
-----
    python vta.py [image.fits]

In the GUI, see Help > VTA help (or F1) for the full guide.

Author: David R. Ciardi.  Heritage: ATV by Aaron Barth.

Version: 2026-06-16
"""

from __future__ import annotations

import os
import sys
import re
import argparse
import numpy as np

# Update this date whenever VTA is changed.
__version__ = "2026-06-13"
__date__ = "2026-06-13"


def _configure_qt_platform():
    """Make Qt's platform plugin discoverable without any shell setup.

    PySide6 normally finds its own plugins, but in some environments
    (certain WSL2 / launcher / mixed-Qt setups) that auto-detection
    doesn't kick in and Qt aborts with 'could not find the Qt platform
    plugin'. We point QT_PLUGIN_PATH at *this* PySide6's bundled plugins,
    and on WSL2 default the platform to xcb (X11 via XWayland), which is
    the reliable path there. Both are only set when not already provided,
    so an explicit user choice is never overridden.
    """
    try:
        import PySide6
        plugins = os.path.join(os.path.dirname(PySide6.__file__),
                               "Qt", "plugins")
        if os.path.isdir(plugins) and not os.environ.get("QT_PLUGIN_PATH"):
            os.environ["QT_PLUGIN_PATH"] = plugins
    except Exception:
        pass

    if sys.platform.startswith("linux") and not os.environ.get("QT_QPA_PLATFORM"):
        try:
            with open("/proc/version") as fh:
                if "microsoft" in fh.read().lower():     # running under WSL
                    os.environ["QT_QPA_PLATFORM"] = "xcb"
        except Exception:
            pass


# ======================================================================
#  PURE LOGIC  (no Qt dependency -- safe to import and unit-test)
# ======================================================================

SCALINGS = ("linear", "log", "sqrt", "asinh", "histeq")


def robust_sky(image: np.ndarray):
    """Sigma-clipped median and stddev, a modern proxy for ATV's
    DAOPHOT-style sky() mode/sigma used in autoscale.

    Returns (skymode, skysig). Falls back to nan-aware estimates if
    astropy is unavailable.
    """
    try:
        from astropy.stats import sigma_clipped_stats
        _, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
        return float(median), float(std)
    except Exception:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.median(finite)), float(np.std(finite))


def autoscale_limits(image, scaling, image_min, image_max):
    """Mirror of atv_autoscale: min = skymode - 2*skysig (clipped to
    image_min); max depends on stretch mode. Returns a dict of the
    derived display parameters.
    """
    skymode, skysig = robust_sky(image)
    imstd = float(np.nanstd(image))

    min_value = max(skymode - 2.0 * skysig, image_min)
    if scaling == "linear":
        max_value = min(skymode + 2.0 * imstd, image_max)
    elif scaling == "log":
        max_value = min(skymode + 4.0 * imstd, image_max)
    else:  # asinh, histeq, sqrt -> full top
        max_value = image_max

    if not np.isfinite(min_value):
        min_value = image_min
    if not np.isfinite(max_value):
        max_value = image_max
    if min_value >= max_value:
        min_value -= 1.0
        max_value += 1.0

    asinh_beta = skysig if skysig > 0 else 1.0
    return dict(min_value=float(min_value), max_value=float(max_value),
                skymode=skymode, skysig=skysig, asinh_beta=float(asinh_beta))


def zscale_limits(image, contrast=0.25, n_samples=10000):
    """IRAF/DS9-style zscale, the de-facto modern default. Uses
    astropy's ZScaleInterval when available, else a simple fallback."""
    try:
        from astropy.visualization import ZScaleInterval
        lo, hi = ZScaleInterval(contrast=contrast,
                                nsamples=n_samples).get_limits(image)
        return float(lo), float(hi)
    except Exception:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def hist_equalize(image, min_value, max_value, nbins=256):
    """Histogram equalization over [min_value, max_value] -> [0, 1].
    Equivalent in spirit to IDL hist_equal used by atv_scaleimage."""
    span = max_value - min_value
    if span <= 0:
        span = 1.0
    finite = np.isfinite(image)
    clipped = np.clip(image, min_value, max_value)
    vals = clipped[finite]
    hist, _ = np.histogram(vals, bins=nbins, range=(min_value, max_value))
    cdf = np.cumsum(hist).astype(np.float64)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    idx = ((clipped - min_value) / span * (nbins - 1))
    idx = np.where(np.isfinite(idx), idx, 0)
    idx = np.clip(idx, 0, nbins - 1).astype(np.intp)
    out = cdf[idx]
    out[~finite] = 0.0
    return out


def transform_image(image, scaling, min_value, max_value, asinh_beta):
    """Apply the chosen ATV stretch, returning (display_array, lo, hi)
    where display values in [lo, hi] map linearly onto the colormap.
    Non-finite pixels are pushed to `lo` (ATV's bytscl /nan -> bottom).

    This is the direct analogue of atv_scaleimage, but instead of
    byte-scaling here we hand pyqtgraph the transformed array + levels
    so contrast can be adjusted live without recomputing.
    """
    img = np.asarray(image, dtype=np.float64)

    if scaling == "linear":
        disp = img.copy()
        lo, hi = float(min_value), float(max_value)

    elif scaling == "log":
        offset = min_value - (max_value - min_value) * 0.01
        with np.errstate(invalid="ignore", divide="ignore"):
            disp = np.log10(img - offset)
        lo = float(np.log10(min_value - offset))
        hi = float(np.log10(max_value - offset))

    elif scaling == "sqrt":
        with np.errstate(invalid="ignore"):
            disp = np.sqrt(np.clip(img - min_value, 0, None))
        lo = 0.0
        hi = float(np.sqrt(max(max_value - min_value, 1e-30)))

    elif scaling == "asinh":
        beta = asinh_beta if asinh_beta != 0 else 1.0
        disp = np.arcsinh((img - min_value) / beta)
        lo = 0.0
        hi = float(np.arcsinh((max_value - min_value) / beta))

    elif scaling == "histeq":
        disp = hist_equalize(img, min_value, max_value)
        lo, hi = 0.0, 1.0

    else:
        raise ValueError(f"unknown scaling: {scaling!r}")

    disp = np.where(np.isfinite(disp), disp, lo)
    if hi <= lo:
        hi = lo + 1e-6
    return disp, lo, hi


COORD_SYSTEMS = ["J2000", "J2000 deg", "B1950", "Galactic", "Ecliptic",
                 "Pixel"]


def format_coords(wcs, x, y, system="J2000"):
    """Return (coord_string, system_label) for image pixel (x, y) in the
    requested coordinate system. Empty string if no celestial WCS or if
    'Pixel' is selected (caller then shows x/y only)."""
    if system == "Pixel" or wcs is None:
        return "", ""
    try:
        if not getattr(wcs, "has_celestial", False):
            return "", ""
        sky = wcs.pixel_to_world(x, y)
        if isinstance(sky, (list, tuple)):
            sky = next(s for s in sky
                       if hasattr(s, "ra") or hasattr(s, "l"))
        if system == "Galactic":
            g = sky.galactic
            return f"l {g.l.deg:.5f}   b {g.b.deg:+.5f}", "Galactic"
        if system == "Ecliptic":
            from astropy.coordinates import BarycentricTrueEcliptic
            e = sky.transform_to(BarycentricTrueEcliptic())
            return f"\u03bb {e.lon.deg:.5f}   \u03b2 {e.lat.deg:+.5f}", "Ecliptic"
        if system == "B1950":
            from astropy.coordinates import FK4
            c = sky.transform_to(FK4(equinox="B1950"))
            ra = c.ra.to_string(unit="hour", sep=":", precision=2, pad=True)
            dec = c.dec.to_string(sep=":", precision=1, alwayssign=True,
                                  pad=True)
            return f"\u03b1 {ra}   \u03b4 {dec}", "B1950"
        c = sky.icrs
        if system == "J2000 deg":
            return (f"\u03b1 {c.ra.deg:.6f}   \u03b4 {c.dec.deg:+.6f}", "J2000")
        ra = c.ra.to_string(unit="hour", sep=":", precision=2, pad=True)
        dec = c.dec.to_string(sep=":", precision=1, alwayssign=True, pad=True)
        return f"\u03b1 {ra}   \u03b4 {dec}", "J2000"
    except Exception:
        return "", ""


def list_fits_extensions(path):
    """Inventory the HDUs in a FITS file. Returns a list of dicts with
    index, name, type, shape, and is_image, so a multi-extension file
    can offer the user a choice of which array to display.

    Robust to files with non-standard trailing bytes after the last
    valid HDU (e.g. some Keck NIRC2 frames): HDUs are walked one at a
    time and the scan stops at the last readable one instead of letting
    astropy's full-file scan raise."""
    import warnings
    from astropy.io import fits
    out = []
    with fits.open(path, lazy_load_hdus=True) as hdul:
        i = 0
        while True:
            try:
                h = hdul[i]
            except IndexError:
                break                      # clean end of file
            except Exception:
                # unparseable trailing data: keep the HDUs read so far
                warnings.warn(f"{os.path.basename(path)}: unreadable data "
                              f"after HDU {i - 1}; ignoring the remainder.")
                break
            try:
                data = h.data
            except Exception:
                data = None                # corrupt payload: not displayable
            is_image = (data is not None and getattr(data, "ndim", 0) >= 2
                        and "TableHDU" not in type(h).__name__)
            out.append(dict(index=i, name=(h.name or f"HDU{i}"),
                            type=type(h).__name__,
                            shape=(tuple(data.shape) if data is not None
                                   else None),
                            is_image=bool(is_image)))
            i += 1
    return out


_WCS_NUM_RE = re.compile(
    r"^(CD\d+_\d+|PC\d+_\d+|CDELT\d+|CRVAL\d+|CRPIX\d+|CROTA\d+|"
    r"PV\d+_\d+|LONPOLE|LATPOLE|EQUINOX|EPOCH|RESTFRQ|RESTWAV)$")


def _coerce_wcs_numeric(header):
    """Return a copy of the header with string-valued numeric WCS cards
    converted to floats.

    NIRC2 (and some other instruments) write numeric WCS keywords as
    quoted strings, e.g. ``CD1_1 = '-0.000002764418'``. IDL coerces these
    to floats automatically in numeric context, so ATV reads them; astropy
    instead treats them as strings and silently drops the CD matrix,
    defaulting to a 1 deg/pixel identity scale. That throws both the
    coordinate readout and the compass orientation far off. Converting the
    affected cards to floats before building the WCS fixes it without
    affecting headers that were already numeric."""
    h = header.copy()
    for card in list(h.keys()):
        if card and _WCS_NUM_RE.match(card) and isinstance(h[card], str):
            try:
                h[card] = float(h[card])
            except (ValueError, TypeError):
                pass
    return h


def read_fits_extension_full(path, index):
    """Read an HDU keeping its native dimensionality (may be 2-D or 3-D),
    inheriting primary-header keywords. Returns (data_nd, header, wcs_full)
    where wcs_full has the full axis set. Opens lazily so files with
    unreadable trailing bytes (e.g. some NIRC2 frames) still load."""
    import warnings
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(path, lazy_load_hdus=True) as hdul:
        hdu = hdul[index]
        if getattr(hdu, "data", None) is None:
            raise ValueError(f"HDU {index} has no image data")
        header = hdul[0].header.copy()
        if index != 0:
            header.update(hdu.header)
        data = np.asarray(hdu.data, dtype=np.float64)
        data = np.squeeze(data)            # drop length-1 axes
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wfull = WCS(_coerce_wcs_numeric(header))
        except Exception:
            wfull = None
    return data, header, wfull


def read_fits_extension(path, index):
    """Read a specific HDU as a 2-D image (port of ATV's fits_read /pdu).
    Returns (data2d, header, wcs)."""
    data, header, wfull = read_fits_extension_full(path, index)
    while data.ndim > 2:
        data = data[0]
    wcs = None
    if wfull is not None:
        try:
            wcs = wfull.celestial if wfull.has_celestial else (
                wfull if wfull.naxis == 2 else None)
        except Exception:
            wcs = None
    return data, header, wcs


def load_fits(path):
    """Load the first 2-D image HDU of a FITS file. Returns
    (data2d, header, wcs_or_None)."""
    exts = list_fits_extensions(path)
    img = next((e for e in exts if e["is_image"]), None)
    if img is None:
        raise ValueError("no image data found in FITS file")
    return read_fits_extension(path, img["index"])


def format_counts(v):
    """Format a counts value as fixed-point nnnnnn.n (one decimal) until it
    needs more than six integer digits, then fall back to scientific
    notation."""
    if not np.isfinite(v):
        return "nan"
    if abs(v) < 1.0e6:
        return f"{v:.1f}"
    return f"{v:.4e}"


def transform_wcs_affine(wcs, A, b):
    """Return a new WCS for data remapped by a pixel affine
    ``old = A @ new + b`` (0-indexed (x, y) pixel coordinates). Uses the
    linear CD/CRPIX relations (ignores SIP distortion). Returns None if
    there is no celestial WCS.

    Derivation: world_int(p) = M @ (p - crpix0), with M the pixel-scale
    matrix. Substituting p_old = A p_new + b gives
    CD_new = M @ A and crpix0_new = A^{-1}(crpix0_old - b)."""
    if wcs is None or not getattr(wcs, "has_celestial", False):
        return None
    from astropy.wcs import WCS
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    M = wcs.pixel_scale_matrix                     # 2x2, degrees
    crpix0 = np.asarray(wcs.wcs.crpix, float) - 1.0
    cd_new = M @ A
    crpix0_new = np.linalg.solve(A, crpix0 - b)
    w2 = WCS(naxis=2)
    w2.wcs.crpix = list(crpix0_new + 1.0)
    w2.wcs.cd = cd_new
    w2.wcs.crval = list(wcs.wcs.crval)
    w2.wcs.ctype = [str(c) for c in wcs.wcs.ctype]
    try:
        w2.wcs.cunit = [str(u) for u in wcs.wcs.cunit]
    except Exception:
        pass
    if getattr(wcs.wcs, "radesys", ""):
        w2.wcs.radesys = wcs.wcs.radesys
    if np.isfinite(getattr(wcs.wcs, "equinox", np.nan)):
        w2.wcs.equinox = wcs.wcs.equinox
    return w2


def compass_vectors(wcs, shape, x=None, y=None):
    """Compute unit pixel-space direction vectors for North (increasing
    Dec) and East (increasing RA) at pixel (x, y) (default: image
    center), for drawing a compass. Returns ((nx, ny), (ex, ey)) as unit
    vectors, or None if there is no usable celestial WCS."""
    if wcs is None or not getattr(wcs, "has_celestial", False):
        return None
    try:
        ny, nx = shape
        xc = nx / 2.0 if x is None else float(x)
        yc = ny / 2.0 if y is None else float(y)
        c0 = wcs.pixel_to_world(xc, yc)
        ra0, dec0 = c0.ra.deg, c0.dec.deg
        delta = 1.0 / 3600.0  # 1 arcsec step
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        north = SkyCoord(ra0 * u.deg, (dec0 + delta) * u.deg, frame=c0.frame)
        east = SkyCoord((ra0 + delta / np.cos(np.radians(dec0))) * u.deg,
                        dec0 * u.deg, frame=c0.frame)
        nxp, nyp = wcs.world_to_pixel(north)
        exp, eyp = wcs.world_to_pixel(east)
        nv = np.array([float(nxp) - xc, float(nyp) - yc])
        ev = np.array([float(exp) - xc, float(eyp) - yc])
        nv = nv / (np.hypot(*nv) or 1.0)
        ev = ev / (np.hypot(*ev) or 1.0)
        return (nv[0], nv[1]), (ev[0], ev[1])
    except Exception:
        return None


def vector_cut(image, x0, y0, x1, y1):
    """Sample image values along the line from (x0,y0) to (x1,y1) using
    bilinear interpolation (port of atv_vectorplot, but bilinear rather
    than 4-corner average). Returns (distance, values)."""
    d = np.hypot(x1 - x0, y1 - y0)
    n = int(d) + 1
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    dist = np.hypot(xs - x0, ys - y0)
    try:
        from scipy.ndimage import map_coordinates
        vals = map_coordinates(image, np.vstack([ys, xs]), order=1,
                               mode="nearest")
    except Exception:
        vals = image[np.clip(np.round(ys).astype(int), 0, image.shape[0] - 1),
                     np.clip(np.round(xs).astype(int), 0, image.shape[1] - 1)]
    return dist, vals


def row_values(image, y):
    """Pixel values along image row y (port of atv_rowplot)."""
    y = int(np.clip(round(y), 0, image.shape[0] - 1))
    return np.arange(image.shape[1]), image[y, :]


def col_values(image, x):
    """Pixel values down image column x (port of atv_colplot)."""
    x = int(np.clip(round(x), 0, image.shape[1] - 1))
    return np.arange(image.shape[0]), image[:, x]


def spec_tracepoint(yslice, traceguess, traceheight=7):
    """Port of atv_get_tracepoint: iterative background-subtracted
    centroid of a spatial slice within a window of traceheight pixels
    around the current guess. Converges when the centroid moves < 0.2 px
    (max 10 iterations)."""
    ysize = len(yslice)
    yvec = np.arange(ysize, dtype=float)
    ycen = float(traceguess)
    ylow, yhigh = 0, ysize - 1
    for _ in range(10):
        last = ycen
        ylow = max(int(round(last - traceheight / 2.0)), 0)
        yhigh = min(int(round(last + traceheight / 2.0)), ysize - 1)
        small = yslice[ylow:yhigh + 1]
        if small.size == 0:
            return float(traceguess)
        m = float(np.min(small))
        denom = float(np.sum(small - m))
        if denom <= 0 or not np.isfinite(denom):
            ycen = last
        else:
            ycen = float(np.sum((yvec * (yslice - m))[ylow:yhigh + 1]) / denom)
        if not np.isfinite(ycen):
            ycen = last
        if abs(ycen - last) < 0.2:
            break
    if ycen < ylow or ycen > yhigh or not np.isfinite(ycen):
        ycen = float(traceguess)
    return float(ycen)


def trace_spectrum(data, x0, y0, tracestep=21, traceheight=7,
                   traceorder=3, xregion=None):
    """Port of atv_trace: centroid the spectral trace every tracestep
    pixels along x, starting from a click at (x0, y0), tracing outward in
    both directions with a slope-aware guess, then fit a polynomial of
    order traceorder. Returns (tracecenters, tracepoints, xspec,
    fulltrace)."""
    ny, nx = data.shape
    if xregion is None:
        xregion = (0, nx - 1)
    x0 = min(max(float(x0), tracestep), nx - tracestep)
    guess = min(max(float(y0), traceheight / 2.0), ny - traceheight / 2.0)

    xsize = int(xregion[1]) - int(xregion[0]) + 1
    twidth = int(tracestep) // 2
    ntp = max(int(xsize // tracestep), 1)
    centers = (np.arange(ntp) * int(tracestep) + int(tracestep) // 2
               + int(xregion[0]))
    points = np.zeros(ntp, dtype=float)
    mid = int(np.argmin(np.abs(centers - x0)))

    def yprofile(xc):
        """Column-summed spatial profile in a strip of ±twidth pixels around
        column xc (NaNs zeroed).
        """
        lo = max(int(xc) - twidth, 0)
        hi = min(int(xc) + twidth, nx - 1)
        ysl = np.nansum(data[:, lo:hi + 1], axis=1)
        return np.where(np.isfinite(ysl), ysl, 0.0)

    # peak up on the starting point (move guess to the local maximum)
    ysl = yprofile(x0)
    ymin = max(1, int(guess - traceheight / 2.0))
    ymax = min(int(guess + traceheight / 2.0), ny - 2)
    sl = ysl[ymin:ymax + 1]
    if sl.size > 0 and np.any(sl != sl[0]):
        guess = guess - traceheight / 2.0 + int(np.argmax(sl))

    # trace from the start point to higher x
    g = guess
    for i in range(mid, ntp):
        ysl = yprofile(centers[i])
        if np.min(ysl) == np.max(ysl):
            points[i] = g
            ycen = g
        else:
            ycen = spec_tracepoint(ysl, g, traceheight)
            points[i] = ycen
        if i == mid:
            g = ycen
        else:
            g = min(max(ycen + (points[i] - points[i - 1]) / 2.0, 1.0),
                    ny - 1.0)
        if not np.isfinite(g):
            g = points[i - 1]

    # now trace from the start point to lower x
    g = points[mid]
    for i in range(mid - 1, -1, -1):
        ysl = yprofile(centers[i])
        if np.min(ysl) == np.max(ysl):
            points[i] = g
            ycen = g
        else:
            ycen = spec_tracepoint(ysl, g, traceheight)
            points[i] = ycen
        g = min(max(ycen - (points[i + 1] - points[i]) / 2.0, 1.0), ny - 1.0)
        if not np.isfinite(g):
            g = points[i + 1]

    order = min(int(traceorder), max(ntp - 1, 0))
    coeffs = np.polynomial.polynomial.polyfit(centers.astype(float),
                                              points, order)
    xspec = np.arange(xsize) + int(xregion[0])
    fulltrace = np.polynomial.polynomial.polyval(xspec.astype(float), coeffs)
    return centers, points, xspec, fulltrace


def extract_spectrum(data, xspec, fulltrace, lower=-5, upper=5,
                     backsub=True, back=(-25, -15, 15, 25)):
    """Port of atvextract's summation: for each column, sum the pixels in
    the aperture [trace+lower, trace+upper] with fractional pixels at the
    aperture edges, optionally subtracting a background level taken as the
    mean of the medians of two offset strips. Columns whose background
    region runs off the image are set to 0 (as in ATV)."""
    ny = data.shape[0]
    n = len(xspec)
    spec = np.zeros(n, dtype=float)
    nxpoints = float(upper - lower)
    for j in range(n):
        i = int(xspec[j])
        t = float(fulltrace[j])
        if (t + back[0] < 0) or (t + back[3] > ny):
            continue
        ytop = int(t + upper - 0.5)
        ybottom = int(t + lower + 0.5) + 1
        if ybottom - 1 < 0 or ytop + 1 > ny - 1 or ytop < ybottom:
            continue
        upperfraction = t + upper - 0.5 - ytop
        lowerfraction = 1.0 - upperfraction
        signal = float(np.nansum(data[ybottom:ytop + 1, i]))
        signal += upperfraction * float(data[ytop + 1, i])
        signal += lowerfraction * float(data[ybottom - 1, i])
        if backsub:
            l0, l1 = int(t + back[0]), int(t + back[1])
            u0, u1 = int(t + back[2]), int(t + back[3])
            lowerback = float(np.nanmedian(data[l0:l1 + 1, i]))
            upperback = float(np.nanmedian(data[u0:u1 + 1, i]))
            signal -= 0.5 * (lowerback + upperback) * nxpoints
        spec[j] = signal
    return spec


def region_box(image, x, y, half=20):
    """Sub-array of a (2*half+1) box around (x, y), clipped to the image
    (used for the region histogram, port of atv_histplot's default box)."""
    ny, nx = image.shape
    half_x = int(min(half, nx / 2.0))
    half_y = int(min(half, ny / 2.0))
    x = int(round(x))
    y = int(round(y))
    x1, x2 = max(x - half_x, 0), min(x + half_x, nx - 1)
    y1, y2 = max(y - half_y, 0), min(y + half_y, ny - 1)
    return image[y1:y2 + 1, x1:x2 + 1], (x1, x2, y1, y2)


# ----------------------------------------------------------------------
#  Analysis: centroid, statistics, radial profile, aperture photometry
#  (faithful ports of atv_imcenterf, atv_stats_refresh, atv_radplotf,
#   atv_splinefwhm, atv_apphot_refresh + IDLastro aper)
# ----------------------------------------------------------------------

def centroid_com(image, x, y, centerbox=5, outersky=20.0):
    """Iterative center-of-mass centroid around (x, y). Port of
    atv_imcenterf (M. Liu / AJB): find the max pixel in a ~1.5x box,
    then take the intensity-weighted centroid in the centering box after
    a quick min-subtraction. Returns (xcen, ycen, warning)."""
    ny, nx = image.shape
    x, y = int(round(x)), int(round(y))
    if centerbox <= 0:
        return float(x), float(y), ""
    if centerbox % 2 == 0:
        centerbox += 1
    dc = (centerbox - 1) // 2
    bigbox = int(round(1.5 * centerbox))
    if bigbox % 2 == 0:
        bigbox += 1
    db = (bigbox - 1) // 2

    # NaN guard over the outer-sky region
    minx, maxx = max(0, int(x - outersky)), min(nx - 1, int(x + outersky))
    miny, maxy = max(0, int(y - outersky)), min(ny - 1, int(y + outersky))
    sub = image[miny:maxy + 1, minx:maxx + 1]
    if sub.size == 0 or not np.isfinite(np.nanmean(sub)):
        return float(x), float(y), "Region contains NaN values."

    # 1) brightest pixel in the big box
    x0, x1 = max(x - db, 0), min(x + db, nx - 1)
    y0, y1 = max(y - db, 0), min(y + db, ny - 1)
    cut = image[y0:y1 + 1, x0:x1 + 1]
    my, mx = np.unravel_index(np.nanargmax(cut), cut.shape)
    xx, yy = mx + x0, my + y0

    # 2) centroid in the centering box about that pixel
    x0, x1 = int(round(max(xx - dc, 0))), int(round(min(xx + dc, nx - 1)))
    y0, y1 = int(round(max(yy - dc, 0))), int(round(min(yy + dc, ny - 1)))
    cut = image[y0:y1 + 1, x0:x1 + 1].astype(float)
    cut = cut - np.nanmin(cut)              # quick-and-dirty sky subtraction
    tot = np.nansum(cut)
    if tot <= 0 or not np.isfinite(tot):
        return float(x), float(y), "Unable to center."
    ii = np.arange(cut.shape[1])
    jj = np.arange(cut.shape[0])
    xcen = np.nansum(cut * ii[None, :]) / tot + x0
    ycen = np.nansum(cut * jj[:, None]) / tot + y0

    warning = ""
    if abs(xcen - x) > 3 or abs(ycen - y) > 3:
        warning = "Possible mis-centering?"
    if not (np.isfinite(xcen) and np.isfinite(ycen)):
        return float(x), float(y), "Unable to center."
    return float(xcen), float(ycen), warning


def box_statistics(image, x, y, boxsize=11):
    """Port of atv_stats_refresh: min/max/mean/median/stddev in an odd
    box centered on (x, y). Returns a dict including the cut array."""
    ny, nx = image.shape
    boxsize = max(int(boxsize), 3)
    if boxsize % 2 == 0:
        boxsize += 1
    b = round((boxsize - 1) / 2)
    xmin = int(np.clip(round(x - b), 0, nx - 1))
    xmax = int(np.clip(round(x + b), 0, nx - 1))
    ymin = int(np.clip(round(y - b), 0, ny - 1))
    ymax = int(np.clip(round(y + b), 0, ny - 1))
    cut = image[ymin:ymax + 1, xmin:xmax + 1].astype(float)
    return dict(npix=int(cut.size),
                total=float(np.nansum(cut)),
                min=float(np.nanmin(cut)), max=float(np.nanmax(cut)),
                mean=float(np.nanmean(cut)), median=float(np.nanmedian(cut)),
                std=float(np.nanstd(cut)),
                box=(xmin, xmax, ymin, ymax), cut=cut)


def spline_fwhm(rad, prof):
    """Port of atv_splinefwhm: spline the radial profile x50, then march
    from the peak (assumed at the minimum radius) to the half-maximum
    crossing. Returns (fwhm, warning); fwhm < 0 signals failure."""
    rad = np.asarray(rad, float)
    prof = np.asarray(prof, float)
    good = np.isfinite(rad) & np.isfinite(prof)
    rad, prof = rad[good], prof[good]
    if rad.size < 3:
        return -1.0, "Unable to measure FWHM!"
    order = np.argsort(rad)
    rad, prof = rad[order], prof[order]
    if rad[np.argmax(prof)] != rad.min():
        return -1.0, "Profile peak is off-center!"

    n = rad.size
    splrad = rad.min() + np.arange(n * 50 + 1) * (rad.max() - rad.min()) / (n * 50)
    try:
        from scipy.interpolate import CubicSpline
        splprof = CubicSpline(rad, prof)(splrad)
    except Exception:
        splprof = np.interp(splrad, rad, prof)

    half = 0.5 * splprof.max()
    below = np.where(splprof < half)[0]
    if below.size == 0 or below[0] < 2 or below[0] >= splrad.size:
        return -1.0, "Unable to measure FWHM!"
    i = below[0]
    return float(splrad[i] + splrad[i - 1]), ""


def radial_profile(image, x, y, outersky=20.0):
    """Port of atv_radplotf: differential annular profile with a
    median-annulus sky. Returns per-pixel scatter (r_pts, v_pts), the
    sky-subtracted mean profile per 1-pixel annulus (r_prof, prof),
    the sky level/sigma, and the spline FWHM."""
    ny, nx = image.shape
    inrad = 0.5 * np.sqrt(2)
    outrad = float(round(outersky * 1.2))
    drad = 1.0
    insky = outrad + drad
    outsky = insky + drad + 20.0

    x0 = int(max(np.floor(x - outsky), 0))
    x1 = int(min(np.ceil(x + outsky), nx - 1))
    y0 = int(max(np.floor(y - outsky), 0))
    y1 = int(min(np.ceil(y + outsky), ny - 1))
    img = image[y0:y1 + 1, x0:x1 + 1].astype(float)
    xcen, ycen = x - x0, y - y0
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    distsq = (xx - xcen) ** 2 + (yy - ycen) ** 2

    skymask = (distsq > insky ** 2) & (distsq <= outsky ** 2) & np.isfinite(img)
    skyann = img[skymask] if skymask.sum() > 0 else img[np.isfinite(img)]
    sky = float(np.median(skyann))
    skysig = float(np.std(skyann))

    nrad = int(np.ceil((outrad - inrad) / drad)) + 1
    r_prof, prof = [], []
    for i in range(nrad):
        if i == 0:
            rin, rout, rin2 = 0.0, inrad, -0.01
        else:
            rin = inrad + drad * (i - 1)
            rout = min(rin + drad, outrad)
            rin2 = rin * rin
        m = (distsq > rin2) & (distsq <= rout * rout) & np.isfinite(img)
        if m.sum() > 0:
            r_prof.append((rout + rin) / 2.0)
            prof.append(float(np.mean(img[m])) - sky)
    r_prof, prof = np.array(r_prof), np.array(prof)

    pm = (distsq >= 0) & (distsq <= outrad * outrad) & np.isfinite(img)
    r_pts, v_pts = np.sqrt(distsq[pm]), img[pm]

    fwhm, fwhm_warn = spline_fwhm(r_prof, prof)
    return dict(r_pts=r_pts, v_pts=v_pts, r_prof=r_prof, prof=prof,
                sky=sky, skysig=skysig, fwhm=fwhm, fwhm_warning=fwhm_warn,
                outrad=outrad)


def aperture_photometry_atv(image, x, y, aprad=5.0, innersky=10.0,
                            outersky=20.0, gain=1.0, readnoise=0.0,
                            skytype=0, magunits=False, zeropoint=25.0,
                            exptime=1.0):
    """DAOPHOT-style circular aperture photometry. Port of
    atv_apphot_refresh + IDLastro `aper`. skytype: 0=mode, 1=median,
    2=none. Returns flux (counts or magnitude), flux_err, sky, skysig,
    nsky, area, and a warning. Uses exact aperture overlap, which is
    more accurate than ATV's integer pixel masking."""
    from photutils.aperture import (CircularAperture, CircularAnnulus,
                                    aperture_photometry)
    from astropy.stats import sigma_clip

    warning = ""
    ap = CircularAperture((x, y), r=aprad)
    ann = CircularAnnulus((x, y), r_in=innersky, r_out=outersky)

    skyvals = ann.to_mask(method="center").get_values(image)
    skyvals = skyvals[np.isfinite(skyvals)]
    nsky = int(skyvals.size)

    if skytype == 2:                                   # no sky subtraction
        sky, skysig = 0.0, 0.0
    elif skytype == 1:                                 # median sky
        sky = float(np.median(skyvals)) if nsky else 0.0
        skysig = float(np.std(skyvals)) if nsky else 0.0
    else:                                              # DAOPHOT mode
        if nsky:
            cv = sigma_clip(skyvals, sigma=3, maxiters=5).compressed()
            if cv.size:
                sky = float(3 * np.median(cv) - 2 * np.mean(cv))
                skysig = float(np.std(cv))
            else:
                sky, skysig = float(np.median(skyvals)), float(np.std(skyvals))
        else:
            sky, skysig, warning = 0.0, 0.0, "No pixels in sky!"

    raw = float(aperture_photometry(image, ap, method="exact")["aperture_sum"][0])
    area = float(ap.area_overlap(image, method="exact"))
    net = raw - sky * area

    # aper three-term error (+ optional readnoise; rn=0 reproduces ATV default)
    skyvar = skysig ** 2
    sigsq = skyvar / nsky if nsky else 0.0
    var = (max(net, 0.0) / gain) + area * skyvar + sigsq * area ** 2 \
        + area * (readnoise / gain) ** 2
    flux_err = float(np.sqrt(var)) if var > 0 else 0.0
    counts_err = flux_err          # error in counts, before any mag conversion

    if net > 0:
        mag = zeropoint - 2.5 * np.log10(net / exptime)
        mag_err = 1.0857 * counts_err / net
    else:
        mag, mag_err = 99.999, 0.0

    if magunits:
        if net > 0:
            flux, flux_err = mag, mag_err
        else:
            flux, flux_err, warning = 99.999, 0.0, "Error in computing flux!"
    else:
        flux = net

    return dict(flux=float(flux), flux_err=float(flux_err), sky=float(sky),
                skysig=float(skysig), nsky=nsky, area=float(area),
                net=float(net), raw=float(raw), counts=float(net),
                counts_err=float(counts_err), mag=float(mag),
                mag_err=float(mag_err), magunits=bool(magunits),
                warning=warning)


# ======================================================================
#  Qt / pyqtgraph GUI
# ======================================================================

HELP_HTML = """
<h2>Visualization Tool for Astronomy (VTA) &mdash; help</h2>
<p>An astronomical FITS image viewer. Heritage: ATV by Aaron Barth.</p>

<h3>Opening images</h3>
<p><b>Open</b> (toolbar or File menu) loads a FITS file. Multi-extension
files get an <b>ext</b> selector in the toolbar; header keywords are
inherited from the primary HDU. 3-D files open the <b>cube</b> dock
(bottom): step planes with the slider/spinbox, or combine N planes with
median/average. From a terminal: <code>python vta.py image.fits</code>.</p>

<h3>Modes (toolbar pulldown)</h3>
<table border="0" cellpadding="3">
<tr><td><b>scan</b></td><td>pan/zoom with the mouse; readout follows the
pointer</td></tr>
<tr><td><b>color</b></td><td>drag to set brightness (left/right) and
contrast (up/down), ATV style</td></tr>
<tr><td><b>zoom</b></td><td>click to zoom in, right-click out, drag a box
to view a region</td></tr>
<tr><td><b>imexam</b></td><td>click a source to measure it (photometry,
radial profile, statistics)</td></tr>
<tr><td><b>vector</b></td><td>click-drag-release a line for an arbitrary
cut; x axis can be angular (arcsec/arcmin/degree) when a WCS pixel scale
exists</td></tr>
<tr><td><b>row / col</b></td><td>click to plot that row / column; the
spinbox in the plot window fine-tunes</td></tr>
<tr><td><b>spectrum</b></td><td>click on a spectral trace to extract it
(see below)</td></tr>
</table>

<h3>Keyboard</h3>
<table border="0" cellpadding="3">
<tr><td><b>arrows</b></td><td>move the cursor 1 px (Shift = 10 px)</td></tr>
<tr><td><b>p</b></td><td>photometry at the cursor</td></tr>
<tr><td><b>r</b> / <b>c</b></td><td>plot the row / column through the
cursor (works from the plot windows too)</td></tr>
<tr><td><b>1 2 3</b></td><td>show blink buffer 1/2/3 (when filled)</td></tr>
<tr><td><b>F1</b></td><td>this help</td></tr>
</table>

<h3>Display</h3>
<p><b>Stretch</b>: linear, log, sqrt, asinh (with editable &beta;),
histeq. <b>Range</b>: AutoScale (robust), ZScale (IRAF), Full, or type
min/max. <b>Colormaps</b>: ATV's grey / blue-white / red-orange plus
rainbow, viridis, inferno, magma, cividis, turbo, with <b>invert</b>;
all panels (image, magnifier, statistics subimage) stay in sync.
<b>Center</b> refits the frame.</p>

<h3>Coordinate readout</h3>
<p>The status-bar selector (bottom right) switches the readout between
J2000 (sexagesimal), J2000 degrees, B1950, Galactic, Ecliptic, and
Pixel. All but Pixel need a celestial WCS in the header.</p>

<h3>Analysis</h3>
<p><b>Photometry</b> (imexam click or <b>p</b>): ATV/DAOPHOT-style
aperture photometry with centroiding; set aperture/sky radii, gain, read
noise, zero point, exposure time; counts or magnitudes; sky from DAOPHOT
mode, median, or none. The radial profile shows binned points, a spline,
the FWHM, and the aperture/sky radii; save it with the button below.
<b>Statistics</b> tab: box statistics, a square subimage in the current
display settings, and the pixel histogram.</p>

<h3>Labels (toolbar pulldown)</h3>
<p>Dialog-driven annotations, each with OK / Apply / Cancel:
<b>Arrow</b> (drawn by mouse drag; style + restyle from the dialog),
<b>Compass</b> (N/E at a chosen position; needs a celestial WCS),
<b>Scale bar</b> (angular length from the WCS pixel scale),
<b>Contours</b> (color, line style, min/max levels, N levels). Erase
arrows or all annotations from the same menu. Annotations live at fixed
image coordinates and pan/zoom with the image.</p>

<h3>Rotate (toolbar pulldown)</h3>
<p>Exact 90&deg;/180&deg;/270&deg; rotations and X/Y/XY flips transform
the <i>data and the WCS together</i> &mdash; coordinates stay exact.
Arbitrary-angle rotation resamples and drops the celestial WCS.
<b>Reset</b> restores the originally loaded image. <b>File &rarr; Save
image as FITS</b> writes the current orientation with its WCS.</p>

<h3>Blink / RGB (toolbar pulldown)</h3>
<p>Store the current image to one of three buffers, recall with the menu
or keys 1/2/3 (zoom/pan is kept so frames stay registered), or
<b>Auto-blink</b> through the filled buffers. <b>Make RGB&hellip;</b>
assigns R/G/B from the buffers (zscale or min/max channel scaling);
in RGB mode the readout shows R/G/B values and photometry/stretch are
disabled; <b>Exit RGB</b> returns.</p>

<h3>Spectral extraction</h3>
<p>Pick <b>spectrum</b> mode and click on a spectral trace. VTA peaks
up at the click, traces the order by iterative centroiding every
<i>trace step</i> pixels (outward in both directions), fits a polynomial,
and extracts with partial-pixel aperture summation and optional
background subtraction from two offset strips &mdash; a port of ATV's
<code>atvextract</code>. Overlays show the trace points (+), the fitted
trace (blue), the aperture (yellow), and the background regions
(magenta). In the spectrum window, <b>Parameters&hellip;</b> opens the
non-modal dialog (trace step/height/order, x region, aperture, background
regions, hold-trace-fixed) &mdash; edits re-extract live &mdash; and
<b>Save spectrum&hellip;</b> writes 1-D FITS or two-column text.
Dispersion must run along x (rotate 90&deg; first if DISPAXIS=2).</p>

<h3>Plot windows</h3>
<p>Row, column, vector, and spectrum plots each have their own window,
so all four can be open at once. Each has the matplotlib navigation
toolbar (pan/zoom/save) plus its own controls in the same row.</p>

<h3>Cube viewing</h3>
<p>For 3-D FITS files the bottom dock steps through planes (slider or
spinbox), optionally combining N planes by median or average; the label
shows the plane number and the third-axis world coordinate. The stretch
and zoom are held fixed across planes &mdash; AutoScale rescales the
current plane.</p>
"""


def build_gui():
    """Import Qt lazily so the pure-logic half stays importable headless."""
    import pyqtgraph as pg
    from PySide6 import QtWidgets, QtCore, QtGui
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    pg.setConfigOptions(imageAxisOrder="row-major", background="k",
                        foreground="w", antialias=False)

    COLORMAPS = ["red-orange", "red-white", "grey", "blue-white", "rainbow",
                 "viridis", "inferno", "magma", "cividis", "turbo"]

    # ATV's named annotation colors
    LABEL_COLORS = {"red": "#ff3333", "black": "#000000", "green": "#37ff37",
                    "blue": "#4d9fff", "cyan": "#37ffff", "magenta": "#ff37ff",
                    "yellow": "#ffe24d", "white": "#ffffff"}
    LABEL_COLOR_NAMES = ["red", "black", "green", "blue", "cyan",
                         "magenta", "yellow", "white"]
    LINE_STYLES = {"solid": QtCore.Qt.SolidLine, "dotted": QtCore.Qt.DotLine,
                   "dashed": QtCore.Qt.DashLine, "dashdot": QtCore.Qt.DashDotLine}
    LINE_STYLE_NAMES = ["solid", "dotted", "dashed", "dashdot"]

    # custom colormap control points: position -> (r, g, b) 0-255
    CUSTOM_CMAPS = {
        "red-white": [(0.0, (0, 0, 0)), (0.5, (190, 20, 20)),
                      (0.82, (255, 130, 70)), (1.0, (255, 255, 255))],
        "blue-white": [(0.0, (0, 0, 0)), (0.45, (0, 70, 220)),
                       (1.0, (255, 255, 255))],
        "red-orange": [(0.0, (0, 0, 0)), (0.4, (170, 0, 0)),
                       (0.75, (255, 140, 0)), (1.0, (255, 255, 200))],
    }

    def get_colormap(name):
        """Resolve a colormap robustly across pyqtgraph sources.
        'grey' and the CUSTOM_CMAPS are built by hand; the rest try the
        built-in file maps first, then matplotlib."""
        if name == "grey":
            return pg.ColorMap(np.array([0.0, 1.0]),
                               np.array([[0, 0, 0, 255],
                                         [255, 255, 255, 255]], dtype=np.ubyte))
        if name in CUSTOM_CMAPS:
            pts = CUSTOM_CMAPS[name]
            pos = np.array([p for p, _ in pts])
            col = np.array([[r, g, b, 255] for _, (r, g, b) in pts],
                           dtype=np.ubyte)
            return pg.ColorMap(pos, col)
        try:
            return pg.colormap.get(name)
        except Exception:
            return pg.colormap.get(name, source="matplotlib")

    class StretchViewBox(pg.ViewBox):
        """ViewBox that hands left-drags to the Viewer in 'color' mode
        (brightness/contrast) and suppresses panning in 'vector' mode."""
        viewer = None

        def mouseDragEvent(self, ev, axis=None):
            """Route mouse drags by interaction mode: arrow drawing when armed,
            color (brightness/contrast), vector cut, else default pyqtgraph
            pan/zoom.
            """
            v = self.viewer
            if v is not None and getattr(v, "_arrow_armed", False) \
                    and ev.button() == QtCore.Qt.LeftButton:
                ev.accept()
                v._arrow_drag(ev)
                return
            if v is not None and v.mode == "color" \
                    and ev.button() == QtCore.Qt.LeftButton:
                ev.accept()
                v._color_drag(ev.scenePos())
                return
            if v is not None and v.mode == "vector" \
                    and ev.button() == QtCore.Qt.LeftButton:
                ev.accept()
                v._vector_drag(ev)
                return
            super().mouseDragEvent(ev, axis)

    class GripSplitterHandle(QtWidgets.QSplitterHandle):
        """Splitter handle that paints a row of grip dots, highlights on
        hover, and shows a horizontal-resize cursor, so it's obvious the
        bar between the image and the analysis panel can be dragged."""

        def __init__(self, orientation, parent):
            super().__init__(orientation, parent)
            self.setCursor(QtCore.Qt.SplitHCursor)
            self._hover = False

        def enterEvent(self, ev):
            self._hover = True
            self.update()

        def leaveEvent(self, ev):
            self._hover = False
            self.update()

        def paintEvent(self, ev):
            qp = QtGui.QPainter(self)
            r = self.rect()
            qp.fillRect(r, QtGui.QColor("#9ec2ff") if self._hover
                        else QtGui.QColor("#c4c4c4"))
            qp.setPen(QtGui.QPen(QtGui.QColor("#8a8a8a")))
            qp.drawLine(r.left(), r.top(), r.left(), r.bottom())
            qp.drawLine(r.right(), r.top(), r.right(), r.bottom())
            qp.setPen(QtCore.Qt.NoPen)
            qp.setBrush(QtGui.QColor("#3a6ea5") if self._hover
                        else QtGui.QColor("#5a5a5a"))
            cx, cy = r.center().x() + 1, r.center().y()
            for i in range(-4, 5):
                qp.drawEllipse(QtCore.QPointF(cx, cy + i * 7.0), 1.5, 1.5)

    class GripSplitter(QtWidgets.QSplitter):
        """QSplitter using GripSplitterHandle for a visible drag grip."""

        def createHandle(self):
            return GripSplitterHandle(self.orientation(), self)

    class CompassWidget(QtWidgets.QWidget):
        """Always-on N/E compass driven by the image WCS, shown beneath the
        magnifier. Independent of the annotation-compass dialog/overlay.
        Drawn in black; shows 'No WCS' when the image has no celestial WCS."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFixedHeight(118)
            self.setMinimumWidth(210)
            self._vecs = None     # ((nx, ny), (ex, ey)) unit vectors, or None

        def set_wcs(self, wcs, shape):
            self._vecs = (compass_vectors(wcs, shape)
                          if (wcs is not None and shape) else None)
            self.update()

        def paintEvent(self, ev):
            import math
            qp = QtGui.QPainter(self)
            qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
            r = self.rect()
            # blend with the surrounding dock panel (no odd inset box)
            qp.fillRect(r, self.palette().color(QtGui.QPalette.Window))
            fg = QtGui.QColor("#000000")              # compass in black
            if self._vecs is None:
                qp.setPen(fg)
                f = qp.font()
                f.setPointSize(11)
                qp.setFont(f)
                qp.drawText(r, QtCore.Qt.AlignCenter, "No WCS")
                return
            cx, cy = r.center().x(), r.center().y()
            L = 0.34 * min(r.width(), r.height())
            pen = QtGui.QPen(fg)
            pen.setWidth(2)
            qp.setPen(pen)

            def arrow(vx, vy, label):
                # pixel y increases upward; widget y increases downward
                tx, ty = cx + vx * L, cy - vy * L
                qp.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(tx, ty))
                ang = math.atan2(ty - cy, tx - cx)
                for da in (math.radians(150), math.radians(-150)):
                    qp.drawLine(QtCore.QPointF(tx, ty),
                                QtCore.QPointF(tx + 9 * math.cos(ang + da),
                                               ty + 9 * math.sin(ang + da)))
                lx, ly = cx + vx * (L + 13), cy - vy * (L + 13)
                qp.drawText(QtCore.QRectF(lx - 9, ly - 9, 18, 18),
                            QtCore.Qt.AlignCenter, label)

            (nxv, nyv), (exv, eyv) = self._vecs
            arrow(nxv, nyv, "N")
            arrow(exv, eyv, "E")

    class ImageModel:
        """Replaces ATV's monolithic global `state` struct. Owns the
        image data + derived display parameters; knows nothing about Qt."""

        def __init__(self):
            """Create an empty model; call set_data() to install an image."""
            self.data = None
            self.header = None
            self.wcs = None
            self.image_min = 0.0
            self.image_max = 1.0
            self.min_value = 0.0
            self.max_value = 1.0
            self.scaling = "asinh"        # ATV's default
            self.asinh_beta = 1.0

        def set_data(self, data, header=None, wcs=None):
            """Install a new image (with optional header and WCS), record its
            data min/max, and autoscale the display range.
            """
            self.data = data
            self.header = header
            self.wcs = wcs
            finite = data[np.isfinite(data)]
            self.image_min = float(finite.min()) if finite.size else 0.0
            self.image_max = float(finite.max()) if finite.size else 1.0
            self.autoscale()

        def autoscale(self):
            """ATV-style robust autoscale of the display range (median ±
            N*sigma via sigma-clipped statistics); also refreshes the
            default asinh beta.
            """
            p = autoscale_limits(self.data, self.scaling,
                                 self.image_min, self.image_max)
            self.min_value = p["min_value"]
            self.max_value = p["max_value"]
            self.asinh_beta = p["asinh_beta"]

        def zscale(self):
            """Set the display range with the IRAF zscale algorithm."""
            self.min_value, self.max_value = zscale_limits(self.data)

        def full_range(self):
            """Set the display range to the full data min/max."""
            self.min_value, self.max_value = self.image_min, self.image_max

        def display(self):
            """Return (stretched_image, lo, hi) applying the current scaling
            mode (linear/log/sqrt/asinh/histeq) to the display range.
            """
            return transform_image(self.data, self.scaling,
                                   self.min_value, self.max_value,
                                   self.asinh_beta)

    class Viewer(QtWidgets.QMainWindow):
        def __init__(self, model):
            """Build the full UI (widgets, toolbar, menus, docks), initialize
            blink/RGB/cube/extraction state, install the key-event filter,
            and show the model's image if present.
            """
            super().__init__()
            self.model = model
            self.setWindowTitle("VTA")
            self.resize(1460, 880)
            # blink / RGB / cube state
            self._blink = [None, None, None]   # 3 saved display states
            self._blink_timer = None
            self._blink_idx = 0
            # what is currently on screen: None = live (loaded) image,
            # 0/1/2 = the corresponding blink buffer
            self._current_buffer = None
            self._rgb_mode = False
            self._rgb_channels = None
            self._cube = None
            self._nslices = 0
            self._slice = 0
            # spectral extraction state (ATV defaults)
            self._xpar = dict(tracestep=21, traceheight=7, traceorder=3,
                              xstart=0, xend=0, lower=-5, upper=5,
                              backsub=True, back1=-25, back2=-15,
                              back3=15, back4=25, fixed=False)
            self._spec_init = None      # last click (x, y)
            self._spec_trace = None     # (centers, points, xspec, fulltrace)
            self._spec = None           # extracted spectrum
            self._spec_items = []       # image overlays
            self._build_widgets()
            self._build_toolbar()
            self._build_menu()
            self._build_view_dock()
            self._build_analysis_dock()
            self._build_cube_dock()
            self._add_panel_toggles()
            # default layout: narrow View dock (left); image | analysis split
            self.resizeDocks([self.view_dock], [240], QtCore.Qt.Horizontal)
            self._main_splitter.setSizes([900, 540])
            # arrow keys move the cursor when our window is active
            QtWidgets.QApplication.instance().installEventFilter(self)
            if self.model.data is not None:
                self._capture_original()
                self.refresh(reset_view=True)
                self._redraw_annotations()

        # ---- layout -------------------------------------------------
        def _build_widgets(self):
            """Create the central pyqtgraph image view with its overlays
            (apertures, cursor, vector line, annotations), the
            histogram/LUT, and the status bar with coordinate-system
            selector.
            """
            central = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(central)
            v.setContentsMargins(0, 0, 0, 0)

            self.glw = pg.GraphicsLayoutWidget()
            v.addWidget(self.glw)
            # adjustable divider: image on the left, analysis panel (added in
            # _build_analysis_dock) on the right; both scale as it's dragged
            self._main_splitter = GripSplitter(QtCore.Qt.Horizontal)
            self._main_splitter.setChildrenCollapsible(False)
            self._main_splitter.setHandleWidth(9)
            self._main_splitter.addWidget(central)
            self._main_splitter.setStretchFactor(0, 1)  # image gets extra width
            self.setCentralWidget(self._main_splitter)

            self.vb = StretchViewBox(lockAspect=True)
            self.vb.setMenuEnabled(False)     # free right-click for zoom-out
            self.glw.addItem(self.vb, row=0, col=0)
            self.vb.invertY(False)            # FITS origin lower-left
            self.img_item = pg.ImageItem()
            self.vb.addItem(self.img_item)

            # brightness/contrast state (mouse-stretch level window)
            self._bc = [0.5, 0.526]           # neutral -> full natural window
            self._nat_lohi = (0.0, 1.0)
            self._cursor_xy = None
            self._vec_start = None
            self._disp = None                 # cached transformed display array
            self._stats_xy = None             # last stats/subimage position

            # vector-cut endpoints overlay
            self._vec_line = pg.PlotDataItem(pen=pg.mkPen("#ffd24d", width=1.5))
            self._vec_line.setVisible(False)
            self.vb.addItem(self._vec_line)

            # ---- annotation parameters (edited via the Labels dialogs) ----
            self._compass = dict(on=False, x=None, y=None, atvertex=True,
                                 color="green", thick=1.5, size=1.0,
                                 arrowlen=100)
            self._scalebar = dict(on=False, x=None, y=None, length=10.0,
                                  units="arcsec", color="white", thick=3.0,
                                  size=1.0)
            self._contour = dict(on=False, color="green", linestyle="solid",
                                 thick=0.9, minval=None, maxval=None, nlevels=6)
            self._arrow_style = dict(color="yellow", thick=1.6, headfrac=0.16)
            self._arrow_armed = False
            self._arrows = []        # list of dicts {tail, head, color,...,item}
            self._compass_items = []
            self._scalebar_items = []
            self._contour_items = []
            self._arrow_preview = pg.PlotDataItem(
                pen=pg.mkPen("#ffe24d", width=1.4, style=QtCore.Qt.DashLine))
            self._arrow_preview.setVisible(False)
            self.vb.addItem(self._arrow_preview)

            self.hist = pg.HistogramLUTItem(image=self.img_item)
            self.glw.addItem(self.hist, row=0, col=1)

            # aperture / sky annulus overlays (aprad=green, insky=cyan,
            # outsky=magenta), drawn at the measured centroid in imexam mode
            self.mode = "scan"
            self._ap_items = {}
            for key, color in (("aprad", "#37ff37"), ("innersky", "#37ffff"),
                               ("outersky", "#ff37ff")):
                item = pg.PlotDataItem(pen=pg.mkPen(color, width=1.5))
                item.setVisible(False)
                self.vb.addItem(item)
                self._ap_items[key] = item
            self._cen_marker = pg.ScatterPlotItem(
                size=9, pen=pg.mkPen("#37ff37", width=1.5), brush=None, pxMode=True)
            self._cen_marker.setVisible(False)
            self.vb.addItem(self._cen_marker)

            # cursor-position marker (tracks mouse + arrow-key moves)
            self._cursor_marker = pg.ScatterPlotItem(
                size=16, symbol="+", pen=pg.mkPen("#ffd24d", width=1.3),
                brush=None, pxMode=True)
            self._cursor_marker.setVisible(False)
            self.vb.addItem(self._cursor_marker)

            # status readouts
            self.status = self.statusBar()
            self.buffer_label = QtWidgets.QLabel("")
            self.xy_label = QtWidgets.QLabel("x= --  y= --  value= --")
            self.wcs_label = QtWidgets.QLabel("")
            self.buffer_label.setStyleSheet(
                "font-family: monospace; font-weight: bold;")
            self.xy_label.setStyleSheet("font-family: monospace;")
            self.wcs_label.setStyleSheet("font-family: monospace;")
            # buffer indicator sits at the far left, before x/y/value
            self.status.addWidget(self.buffer_label)
            self.status.addWidget(self.xy_label)
            self.status.addPermanentWidget(self.wcs_label)
            self._update_buffer_label()
            # coordinate-system selector
            self._coordsys = "J2000"
            self.coordsys_combo = QtWidgets.QComboBox()
            self.coordsys_combo.addItems(COORD_SYSTEMS)
            self.coordsys_combo.setToolTip("Coordinate system for the readout")
            self.coordsys_combo.currentTextChanged.connect(self._on_coordsys)
            self.status.addPermanentWidget(self.coordsys_combo)

            self.glw.scene().sigMouseMoved.connect(self._on_mouse_move)
            self.glw.scene().sigMouseClicked.connect(self._on_click)
            self.vb.viewer = self

        def _build_toolbar(self):
            """Create the top toolbar: Open, mode selector, extension selector,
            Labels/Rotate/Blink-RGB menus, stretch + asinh beta, colormap +
            invert, and the view/scaling buttons.
            """
            tb = self.addToolBar("controls")
            tb.setMovable(False)
            self._toolbar = tb

            tb.addAction("Open", self.open_file)
            tb.addSeparator()

            tb.addWidget(QtWidgets.QLabel(" mode "))
            self.mode_combo = QtWidgets.QComboBox()
            self.mode_combo.addItems(["scan", "color", "zoom", "imexam",
                                      "vector", "row", "col", "spectrum",
                                      "blink"])
            self.mode_combo.setToolTip(
                "scan: pan/zoom + readout.   color: drag to set "
                "brightness (left/right) & contrast (up/down).   "
                "zoom: click in / right-click out / drag a box to view a "
                "region.   imexam: click to measure.   vector: click two "
                "points for a line cut.   row/col: click to plot that "
                "row/column (or press r / c at the cursor).   "
                "spectrum: click on a spectral trace to extract it.   "
                "blink: click to step through the stored buffers.")
            self.mode_combo.currentTextChanged.connect(self._on_mode)
            tb.addWidget(self.mode_combo)

            self.ext_label = QtWidgets.QLabel(" ext ")
            self.ext_label.setVisible(False)
            tb.addWidget(self.ext_label)
            self.ext_combo = QtWidgets.QComboBox()
            self.ext_combo.setVisible(False)
            self.ext_combo.currentIndexChanged.connect(self._on_ext_changed)
            tb.addWidget(self.ext_combo)

            tb.addSeparator()
            self._make_labels_button()
            self._make_rotate_button()
            self._make_blink_button()
            tb.addSeparator()

            tb.addWidget(QtWidgets.QLabel(" stretch "))
            self.scale_combo = QtWidgets.QComboBox()
            self.scale_combo.addItems(SCALINGS)
            self.scale_combo.setCurrentText(self.model.scaling)
            self.scale_combo.currentTextChanged.connect(self._on_scaling)
            tb.addWidget(self.scale_combo)

            # asinh softening parameter (enabled only for asinh stretch)
            self.beta_label = QtWidgets.QLabel(" \u03b2 ")
            tb.addWidget(self.beta_label)
            self.beta_edit = QtWidgets.QLineEdit()
            self.beta_edit.setFixedWidth(60)
            self.beta_edit.setToolTip("asinh softening parameter \u03b2")
            self.beta_edit.editingFinished.connect(self._on_beta_edit)
            tb.addWidget(self.beta_edit)

            tb.addWidget(QtWidgets.QLabel(" color "))
            self.cmap_combo = QtWidgets.QComboBox()
            self.cmap_combo.addItems(COLORMAPS)
            self.cmap_combo.currentTextChanged.connect(self._on_cmap)
            tb.addWidget(self.cmap_combo)

            self.invert_chk = QtWidgets.QCheckBox("invert")
            self.invert_chk.stateChanged.connect(self._on_cmap)
            tb.addWidget(self.invert_chk)

            tb.addSeparator()
            tb.addAction("Center", self._do_center)
            tb.addAction("AutoScale", self._do_autoscale)
            tb.addAction("ZScale", self._do_zscale)
            tb.addAction("Full", self._do_full)

            tb.addWidget(QtWidgets.QLabel(" min "))
            self.min_edit = QtWidgets.QLineEdit()
            self.min_edit.setFixedWidth(80)
            self.min_edit.editingFinished.connect(self._on_minmax_edit)
            tb.addWidget(self.min_edit)
            tb.addWidget(QtWidgets.QLabel(" max "))
            self.max_edit = QtWidgets.QLineEdit()
            self.max_edit.setFixedWidth(80)
            self.max_edit.editingFinished.connect(self._on_minmax_edit)
            tb.addWidget(self.max_edit)

        def _toolbar_menu(self, text):
            """A toolbar pulldown button backed by a QMenu."""
            btn = QtWidgets.QToolButton()
            btn.setText(text)
            btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
            menu = QtWidgets.QMenu(btn)
            btn.setMenu(menu)
            self._toolbar.addWidget(btn)
            return menu

        def _make_rotate_button(self):
            """Toolbar 'Rotate' pulldown: arbitrary angle, exact 90/180/270
            rotations, flips, and reset to the originally loaded image.
            """
            menu = self._toolbar_menu("Rotate")
            menu.addAction("Arbitrary angle\u2026", self._rotate_arbitrary)
            menu.addAction("90\u00b0 CCW", lambda: self._apply_geom("rot90"))
            menu.addAction("180\u00b0", lambda: self._apply_geom("rot180"))
            menu.addAction("270\u00b0 CCW", lambda: self._apply_geom("rot270"))
            menu.addSeparator()
            menu.addAction("Invert X", lambda: self._apply_geom("invertx"))
            menu.addAction("Invert Y", lambda: self._apply_geom("inverty"))
            menu.addAction("Invert XY", lambda: self._apply_geom("invertxy"))
            menu.addSeparator()
            menu.addAction("Reset (reload original)", self._reset_image)

        def _make_labels_button(self):
            """Toolbar 'Labels' pulldown opening the annotation dialogs (arrow,
            compass, scale bar, contours) plus erase actions.
            """
            menu = self._toolbar_menu("Labels")
            menu.addAction("Arrow\u2026", self._arrow_dialog)
            menu.addAction("Compass\u2026", self._compass_dialog)
            menu.addAction("Scale bar\u2026", self._scalebar_dialog)
            menu.addAction("Contours\u2026", self._contour_dialog)
            menu.addSeparator()
            menu.addAction("Erase arrows", self._erase_arrows)
            menu.addAction("Erase all annotations", self._erase_all)

        def _make_blink_button(self):
            """Toolbar 'Blink/RGB' pulldown: store/show blink buffers, auto-
            blink, clear, and the Make-RGB / Exit-RGB actions.
            """
            menu = self._toolbar_menu("Blink/RGB")
            for n in (1, 2, 3):
                menu.addAction(f"Store current \u2192 buffer {n}",
                               lambda _=False, k=n - 1: self._store_blink(k))
            menu.addSeparator()
            for n in (1, 2, 3):
                menu.addAction(f"Show buffer {n}  ({n})",
                               lambda _=False, k=n - 1: self._show_blink(k))
            menu.addSeparator()
            self._autoblink_action = menu.addAction("Auto-blink")
            self._autoblink_action.setCheckable(True)
            self._autoblink_action.toggled.connect(self._toggle_autoblink)
            menu.addAction("Clear buffers", self._clear_blink)
            menu.addSeparator()
            menu.addAction("Make RGB\u2026", self._makergb_dialog)
            menu.addAction("Exit RGB", self._exit_rgb)

        # ---- blink --------------------------------------------------
        def _capture_state(self):
            """Snapshot the current data + display state (stretch, range,
            colormap, brightness/contrast, label) for a blink buffer.
            """
            m = self.model
            if m.data is None:
                return None
            try:
                vr = self.vb.viewRange()
                view = [list(vr[0]), list(vr[1])]
            except Exception:
                view = None
            return dict(data=m.data.copy(), header=m.header, wcs=m.wcs,
                        scaling=m.scaling, min_value=m.min_value,
                        max_value=m.max_value, asinh_beta=m.asinh_beta,
                        image_min=m.image_min, image_max=m.image_max,
                        cmap=self.cmap_combo.currentText(),
                        invert=self.invert_chk.isChecked(),
                        bc=list(self._bc), label=self.file_label.text(),
                        view=view)

        def _restore_state(self, s, keep_view=True):
            """Restore a blink-buffer snapshot. If the buffer has the same
            shape as the image currently shown, the live view is kept so
            equal-size frames blink registered (you can zoom in and compare
            the same pixels). If it differs in size, the buffer's own saved
            view (how that image was last framed) is restored, so blinking
            between different-size frames is still useful.
            """
            self._rgb_mode = False
            m = self.model
            same_shape = (m.data is not None
                          and m.data.shape == s["data"].shape)
            try:
                live_view = [list(r) for r in self.vb.viewRange()]
            except Exception:
                live_view = None
            m.data, m.header, m.wcs = s["data"], s["header"], s["wcs"]
            m.scaling, m.asinh_beta = s["scaling"], s["asinh_beta"]
            m.min_value, m.max_value = s["min_value"], s["max_value"]
            m.image_min, m.image_max = s["image_min"], s["image_max"]
            self._bc = list(s["bc"])
            for w, fn, val in ((self.scale_combo, "setCurrentText", s["scaling"]),
                               (self.cmap_combo, "setCurrentText", s["cmap"])):
                w.blockSignals(True)
                getattr(w, fn)(val)
                w.blockSignals(False)
            self.invert_chk.blockSignals(True)
            self.invert_chk.setChecked(s["invert"])
            self.invert_chk.blockSignals(False)
            self.file_label.setText(s["label"])
            self._clear_apertures()      # don't let stale overlays set the view
            self._cursor_marker.setVisible(False)
            self.refresh(reset_view=False)
            if keep_view and same_shape and live_view is not None:
                self.vb.setRange(xRange=live_view[0], yRange=live_view[1],
                                 padding=0)
            elif s.get("view") is not None:
                self.vb.setRange(xRange=s["view"][0], yRange=s["view"][1],
                                 padding=0)
            else:
                self.vb.autoRange()

        def _store_blink(self, k):
            """Store the current image and display state into blink buffer k."""
            st = self._capture_state()
            if st is None:
                return
            self._blink[k] = st
            self.status.showMessage(f"Stored current image to blink buffer "
                                    f"{k + 1}.", 3000)

        def _show_blink(self, k):
            """Display blink buffer k (no-op with a message if empty)."""
            if self._blink[k] is None:
                self.status.showMessage(f"Blink buffer {k + 1} is empty.", 3000)
                return
            self._restore_state(self._blink[k], keep_view=True)
            self._current_buffer = k
            self._update_buffer_label()
            self.status.showMessage(f"Blink buffer {k + 1}", 1500)

        def _update_buffer_label(self):
            """Keep the persistent status-bar buffer indicator (far left) in
            sync with what is on screen: the live image, a blink buffer, or
            an RGB composite."""
            if getattr(self, "buffer_label", None) is None:
                return
            if getattr(self, "_rgb_mode", False):
                self.buffer_label.setText("buffer: RGB")
            elif self._current_buffer is None:
                self.buffer_label.setText("buffer: live")
            else:
                self.buffer_label.setText(f"buffer: {self._current_buffer + 1}")

        def _clear_blink(self):
            """Empty all three blink buffers and stop auto-blink."""
            self._blink = [None, None, None]
            if self._blink_timer is not None:
                self._autoblink_action.setChecked(False)
            self._current_buffer = None
            self._update_buffer_label()
            self.status.showMessage("Cleared blink buffers.", 3000)

        def _toggle_autoblink(self, on):
            """Start/stop the auto-blink timer cycling through the filled
            buffers (requires at least two).
            """
            if on:
                filled = [i for i, b in enumerate(self._blink) if b is not None]
                if len(filled) < 2:
                    self.status.showMessage(
                        "Need at least two filled buffers to blink.", 4000)
                    self._autoblink_action.setChecked(False)
                    return
                self._blink_timer = QtCore.QTimer(self)
                self._blink_timer.timeout.connect(self._blink_step)
                self._blink_timer.start(600)
            elif self._blink_timer is not None:
                self._blink_timer.stop()
                self._blink_timer = None

        def _blink_click_next(self):
            """blink mode: advance to the next filled buffer (1 -> 2 -> 3 ->
            1 ...) on each click."""
            filled = [i for i, b in enumerate(self._blink) if b is not None]
            if not filled:
                self.status.showMessage(
                    "No blink buffers stored (use Blink/RGB \u25b8 Store).",
                    4000)
                return
            nxt = getattr(self, "_blink_click_idx", -1) + 1
            self._blink_click_idx = nxt % len(filled)
            self._show_blink(filled[self._blink_click_idx])

        def _blink_step(self):
            """Advance auto-blink to the next filled buffer."""
            filled = [i for i, b in enumerate(self._blink) if b is not None]
            if not filled:
                return
            self._blink_idx = (self._blink_idx + 1) % len(filled)
            self._show_blink(filled[self._blink_idx])

        # ---- RGB ----------------------------------------------------
        def _rgb_sources(self):
            """Available channel sources: current + filled blink buffers."""
            srcs = {"current": self._capture_state()}
            for i, b in enumerate(self._blink):
                if b is not None:
                    srcs[f"buffer {i + 1}"] = b
            return srcs

        def _makergb_dialog(self):
            """Dialog assigning the R/G/B channels from the current image and
            filled blink buffers, with zscale or min/max channel scaling.
            """
            srcs = self._rgb_sources()
            names = [n for n in srcs if srcs[n] is not None]
            if len(names) < 1:
                return
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Make RGB")
            form = QtWidgets.QFormLayout(dlg)
            combos = {}
            defaults = {"R": "buffer 1", "G": "buffer 2", "B": "buffer 3"}
            for ch in ("R", "G", "B"):
                cb = QtWidgets.QComboBox()
                cb.addItems(names)
                if defaults[ch] in names:
                    cb.setCurrentText(defaults[ch])
                form.addRow(f"{ch} channel", cb)
                combos[ch] = cb
            scale = QtWidgets.QComboBox()
            scale.addItems(["zscale", "min/max"])
            form.addRow("scaling", scale)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                | QtWidgets.QDialogButtonBox.Apply)
            form.addRow(bb)

            def apply():
                """Validate matching shapes and display the RGB composite."""
                chans = []
                for ch in ("R", "G", "B"):
                    s = srcs[combos[ch].currentText()]
                    chans.append(np.asarray(s["data"], float))
                if len({c.shape for c in chans}) != 1:
                    QtWidgets.QMessageBox.warning(
                        self, "VTA", "RGB channels must have the same shape.")
                    return
                self._show_rgb(chans, scale.currentText())

            bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(apply)
            bb.accepted.connect(lambda: (apply(), dlg.accept()))
            bb.rejected.connect(dlg.reject)
            dlg.exec()

        def _show_rgb(self, channels, method):
            """Scale each channel to 0-255 (zscale or min/max) and display the
            HxWx3 composite; photometry and stretch are disabled in RGB
            mode.
            """
            rgb = np.zeros(channels[0].shape + (3,), dtype=np.ubyte)
            for i, ch in enumerate(channels):
                finite = ch[np.isfinite(ch)]
                if finite.size == 0:
                    lo, hi = 0.0, 1.0
                elif method == "zscale":
                    lo, hi = zscale_limits(ch)
                else:
                    lo, hi = float(np.nanmin(ch)), float(np.nanmax(ch))
                if hi <= lo:
                    hi = lo + 1.0
                scaled = np.clip((ch - lo) / (hi - lo), 0, 1)
                rgb[..., i] = (scaled * 255).astype(np.ubyte)
            self._rgb_mode = True
            self._rgb_channels = channels
            self.img_item.setImage(rgb, autoLevels=False)
            self._update_buffer_label()
            self.status.showMessage(
                "RGB composite shown (stretch/colormap/photometry disabled; "
                "use Blink/RGB \u2192 Exit RGB to return).", 6000)

        def _exit_rgb(self):
            """Leave RGB mode and restore the normal single-image display."""
            if not self._rgb_mode:
                return
            self._rgb_mode = False
            self._rgb_channels = None
            self.refresh(reset_view=False)
            self._update_buffer_label()

        # ---- data cube ----------------------------------------------
        def _build_cube_dock(self):
            """Bottom dock for data cubes: plane slider + spinbox, combine-N
            count, median/average method, and the plane/world-coordinate
            label.
            """
            dock = QtWidgets.QDockWidget("Cube", self)
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(6, 2, 6, 2)
            lay.addWidget(QtWidgets.QLabel("plane"))
            self.cube_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.cube_slider.valueChanged.connect(self._on_cube_slide)
            lay.addWidget(self.cube_slider, 1)
            self.cube_spin = QtWidgets.QSpinBox()
            self.cube_spin.valueChanged.connect(self._on_cube_spin)
            lay.addWidget(self.cube_spin)
            lay.addWidget(QtWidgets.QLabel("combine"))
            self.cube_combine = QtWidgets.QSpinBox()
            self.cube_combine.setRange(1, 999)
            self.cube_combine.setValue(1)
            self.cube_combine.valueChanged.connect(lambda *_:
                                                   self._show_slice(self._slice))
            lay.addWidget(self.cube_combine)
            self.cube_method = QtWidgets.QComboBox()
            self.cube_method.addItems(["median", "average"])
            self.cube_method.currentIndexChanged.connect(
                lambda *_: self._show_slice(self._slice))
            lay.addWidget(self.cube_method)
            self.cube_label = QtWidgets.QLabel("")
            self.cube_label.setStyleSheet("font-family: monospace;")
            lay.addWidget(self.cube_label)
            dock.setWidget(w)
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
            dock.setVisible(False)
            self.cube_dock = dock

        def _setup_cube(self, cube, header, wfull):
            """Install a 3-D cube: keep the full array and WCS, show plane 0
            with the celestial WCS, and reveal the cube dock.
            """
            self._cube = cube
            self._cube_wfull = wfull
            self._cube_header = header
            self._nslices = cube.shape[0]
            self._slice = 0
            wcs2d = None
            if wfull is not None:
                try:
                    wcs2d = wfull.celestial if wfull.has_celestial else None
                except Exception:
                    wcs2d = None
            self._cube_wcs2d = wcs2d
            for wdg in (self.cube_slider, self.cube_spin):
                wdg.blockSignals(True)
                wdg.setRange(0, self._nslices - 1)
                wdg.setValue(0)
                wdg.blockSignals(False)
            self.cube_combine.setMaximum(self._nslices)
            self.cube_dock.setVisible(True)
            self.model.set_data(cube[0].copy(), header, wcs2d)
            self.scale_combo.setCurrentText(self.model.scaling)
            self._reset_bc()
            self.refresh(reset_view=True)
            self._update_cube_label()

        def _clear_cube(self):
            """Forget any cube state and hide the cube dock."""
            self._cube = None
            self._nslices = 0
            if hasattr(self, "cube_dock"):
                self.cube_dock.setVisible(False)

        def _on_cube_slide(self, k):
            """Slider moved: sync the spinbox and show that plane."""
            self.cube_spin.blockSignals(True)
            self.cube_spin.setValue(k)
            self.cube_spin.blockSignals(False)
            self._show_slice(k)

        def _on_cube_spin(self, k):
            """Spinbox changed: sync the slider and show that plane."""
            self.cube_slider.blockSignals(True)
            self.cube_slider.setValue(k)
            self.cube_slider.blockSignals(False)
            self._show_slice(k)

        def _show_slice(self, k):
            """Display cube plane k, or the median/mean of planes k..k+N-1 when
            combine > 1, keeping the current stretch and view.
            """
            if self._cube is None:
                return
            k = int(np.clip(k, 0, self._nslices - 1))
            self._slice = k
            n = self.cube_combine.value()
            if n <= 1:
                plane = self._cube[k]
            else:
                hi = min(k + n, self._nslices)
                sub = self._cube[k:hi]
                plane = (np.median(sub, axis=0)
                         if self.cube_method.currentText() == "median"
                         else np.mean(sub, axis=0))
            self.model.data = np.asarray(plane, float)
            self.refresh(reset_view=False)      # keep stretch + view across planes
            self._update_cube_label()

        def _update_cube_label(self):
            """Show 'plane / nplanes' plus the third-axis world coordinate
            (e.g. wavelength) from the full WCS.
            """
            txt = f"{self._slice + 1} / {self._nslices}"
            w = getattr(self, "_cube_wfull", None)
            if w is not None and w.naxis >= 3:
                try:
                    world = w.all_pix2world([[0, 0, self._slice]], 0)[0]
                    ct = str(w.wcs.ctype[2])[:4].strip()
                    un = str(w.wcs.cunit[2]).strip()
                    txt += f"   {ct}={world[2]:.6g} {un}"
                except Exception:
                    pass
            self.cube_label.setText(txt)

        # ---- view navigation ---------------------------------------
        def _do_center(self):
            """Recenter / fit the whole frame in the view."""
            if self.model.data is not None:
                self.vb.autoRange()

        # ---- asinh beta --------------------------------------------
        def _on_beta_edit(self):
            """Apply a user-typed asinh beta (softening) and refresh without
            re-autoscaling.
            """
            if self.model.scaling != "asinh":
                return
            try:
                beta = float(self.beta_edit.text())
            except ValueError:
                self.beta_edit.setText(f"{self.model.asinh_beta:.4g}")
                return
            if beta <= 0:
                beta = self.model.asinh_beta
            self.model.asinh_beta = beta
            self.refresh()           # no re-autoscale: keep beta as set

        def _sync_beta_field(self):
            """Keep the beta field showing the active value and enabled only
            for the asinh stretch.
            """
            on = (self.model.scaling == "asinh")
            self.beta_edit.setEnabled(on)
            self.beta_label.setEnabled(on)
            self.beta_edit.setText(f"{self.model.asinh_beta:.4g}" if on else "")

        # ---- geometry: rotate / flip -------------------------------
        def _apply_geom(self, op):
            """Apply an exact geometric operation (90-degree rotations /
            flips): transform the data with numpy and the WCS with the
            matching affine, so coordinates stay exact.
            """
            if self.model.data is None:
                return
            d = self.model.data
            ny, nx = d.shape
            if op == "invertx":
                d2, A, b = d[:, ::-1].copy(), [[-1, 0], [0, 1]], [nx - 1, 0]
            elif op == "inverty":
                d2, A, b = d[::-1, :].copy(), [[1, 0], [0, -1]], [0, ny - 1]
            elif op == "invertxy":
                d2, A, b = (d[::-1, ::-1].copy(),
                            [[-1, 0], [0, -1]], [nx - 1, ny - 1])
            elif op == "rot90":
                d2, A, b = np.rot90(d, 1).copy(), [[0, -1], [1, 0]], [nx - 1, 0]
            elif op == "rot180":
                d2, A, b = (np.rot90(d, 2).copy(),
                            [[-1, 0], [0, -1]], [nx - 1, ny - 1])
            elif op == "rot270":
                d2, A, b = np.rot90(d, 3).copy(), [[0, 1], [-1, 0]], [0, ny - 1]
            else:
                return
            new_wcs = transform_wcs_affine(self.model.wcs, A, b)
            self._set_geom(d2, new_wcs)

        def _rotate_arbitrary(self):
            """Arbitrary-angle rotation dialog: resamples with
            scipy.ndimage.rotate and drops the celestial WCS (interpolated
            coordinates would lie).
            """
            if self.model.data is None:
                return
            angle, ok = QtWidgets.QInputDialog.getDouble(
                self, "Rotate image", "Angle (degrees, counter-clockwise):",
                0.0, -360.0, 360.0, 2)
            if not ok or angle == 0:
                return
            from scipy.ndimage import rotate as nd_rotate
            d2 = nd_rotate(self.model.data, angle, reshape=True, order=1,
                           mode="constant", cval=np.nan)
            # an arbitrary rotation drops the celestial WCS in this build
            self._set_geom(d2, None, wcs_note=True)

        def _set_geom(self, data, wcs, wcs_note=False):
            """Install transformed data/WCS, clear stale
            overlays/cursor/annotations, refresh, and notify if the WCS was
            dropped.
            """
            keep_scaling = self.model.scaling
            self.model.set_data(data, self.model.header, wcs)
            self.model.scaling = keep_scaling
            self.model.autoscale()
            # clear stale overlays / cursor
            for it in self._ap_items.values():
                it.setVisible(False)
            self._cen_marker.setVisible(False)
            self._cursor_marker.setVisible(False)
            self._cursor_xy = None
            self._stats_xy = None
            self._vec_line.setVisible(False)
            self._erase_arrows()          # data coords no longer valid
            self._clear_spec_overlays()
            self._clear_apertures()
            self._spec_trace = None
            self.refresh(reset_view=True)
            self._redraw_annotations()
            if wcs_note:
                self.status.showMessage(
                    "Arbitrary rotation applied \u2014 celestial WCS dropped "
                    "(90\u00b0/flip operations preserve it).", 6000)

        # ---- save current view as FITS -----------------------------
        def save_fits(self):
            """Write the currently displayed image with its current (possibly
            rotated/flipped) WCS to a new FITS file, preserving non-WCS
            header cards.
            """
            if self.model.data is None:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save image as FITS", "vta_image.fits",
                "FITS (*.fits *.fit)")
            if not path:
                return
            if self.model.header is not None:
                hdr = self.model.header.copy()
            else:
                hdr = fits.Header()
            # drop any stale WCS cards, then write the current WCS
            for k in list(hdr.keys()):
                if k and (k.startswith(("CRPIX", "CRVAL", "CDELT", "CTYPE",
                                        "CUNIT", "CD1_", "CD2_", "PC1_",
                                        "PC2_")) or k in ("CROTA1", "CROTA2",
                                        "LONPOLE", "LATPOLE", "RADESYS",
                                        "EQUINOX")):
                    del hdr[k]
            if self.model.wcs is not None:
                hdr.update(self.model.wcs.to_header())
            try:
                fits.PrimaryHDU(self.model.data.astype(np.float32),
                                header=hdr).writeto(path, overwrite=True)
                self.status.showMessage(f"Saved {path}", 5000)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "VTA",
                                              f"Could not save:\n{exc}")

        # ---- annotation helpers ------------------------------------
        def _color_hex(self, name):
            """Map an ATV color name to its hex value."""
            return LABEL_COLORS.get(name, "#ffe24d")

        def _arrow_points(self, tail, head, headfrac):
            """Polyline points for a real arrow: shaft + a V-shaped head at
            the tip (tail -> head -> headLeft -> head -> headRight)."""
            x0, y0 = tail
            x1, y1 = head
            dx, dy = x1 - x0, y1 - y0
            L = float(np.hypot(dx, dy))
            if L < 1e-6:
                return [x0, x1], [y0, y1]
            ux, uy = dx / L, dy / L
            hl = max(headfrac * L, 1e-6)
            ca, sa = np.cos(np.radians(26.0)), np.sin(np.radians(26.0))
            lx, ly = ux * ca - uy * sa, ux * sa + uy * ca
            rx, ry = ux * ca + uy * sa, -ux * sa + uy * ca
            return ([x0, x1, x1 - hl * lx, x1, x1 - hl * rx],
                    [y0, y1, y1 - hl * ly, y1, y1 - hl * ry])

        def _redraw_arrow(self, a):
            """Create or restyle the polyline item for one stored arrow."""
            xs, ys = self._arrow_points(a["tail"], a["head"], a["headfrac"])
            pen = pg.mkPen(self._color_hex(a["color"]), width=a["thick"])
            if a.get("item") is None:
                a["item"] = pg.PlotDataItem(xs, ys, pen=pen)
                a["item"].setZValue(25)
                self.vb.addItem(a["item"])
            else:
                a["item"].setData(xs, ys)
                a["item"].setPen(pen)

        def _arrow_drag(self, ev):
            """Live-preview an arrow while dragging; on release store it with
            the current arrow style.
            """
            p0 = self.vb.mapSceneToView(ev.buttonDownScenePos())
            p1 = self.vb.mapSceneToView(ev.scenePos())
            xs, ys = self._arrow_points((p0.x(), p0.y()), (p1.x(), p1.y()),
                                        self._arrow_style["headfrac"])
            self._arrow_preview.setData(xs, ys)
            self._arrow_preview.setVisible(True)
            if ev.isFinish():
                self._arrow_preview.setVisible(False)
                a = dict(tail=(p0.x(), p0.y()), head=(p1.x(), p1.y()),
                         color=self._arrow_style["color"],
                         thick=self._arrow_style["thick"],
                         headfrac=self._arrow_style["headfrac"], item=None)
                self._redraw_arrow(a)
                self._arrows.append(a)

        def _erase_arrows(self):
            """Remove all drawn arrows."""
            for a in self._arrows:
                if a.get("item") is not None:
                    self.vb.removeItem(a["item"])
            self._arrows = []

        def _erase_all(self):
            """Remove arrows and switch off compass, scale bar, and contours."""
            self._erase_arrows()
            self._clear_spec_overlays()
            self._clear_apertures()
            self._spec_trace = None
            self._compass["on"] = False
            self._scalebar["on"] = False
            self._contour["on"] = False
            self._draw_compass()
            self._draw_scalebar()
            self._draw_contours()

        def _pixscale_arcsec(self):
            """Mean pixel scale in arcsec/pixel from the celestial WCS, or None
            if there is no celestial WCS.
            """
            if self.model.wcs is None or not self.model.wcs.has_celestial:
                return None
            M = self.model.wcs.pixel_scale_matrix
            return float(np.sqrt(abs(np.linalg.det(M))) * 3600.0)

        def _label_text(self, lbl, color, size, anchor, pos):
            """Build a pyqtgraph TextItem with ATV-style charsize scaling at a
            data position.
            """
            t = pg.TextItem(lbl, color=color, anchor=anchor)
            if size:
                f = QtGui.QFont()
                f.setPointSizeF(max(6.0, 9.0 * float(size)))
                t.textItem.setFont(f)
            t.setPos(*pos)
            t.setZValue(25)
            return t

        # ---- compass -----------------------------------------------
        def _draw_compass(self):
            """(Re)draw the N/E compass at its stored position/style; needs a
            celestial WCS.
            """
            for it in self._compass_items:
                self.vb.removeItem(it)
            self._compass_items = []
            c = self._compass
            if not c["on"] or self.model.data is None:
                return
            ny, nx = self.model.data.shape
            x = c["x"] if c["x"] is not None else 0.15 * nx
            y = c["y"] if c["y"] is not None else 0.15 * ny
            vecs = compass_vectors(self.model.wcs, (ny, nx), x, y)
            if vecs is None:
                return            # no celestial WCS: silently draw nothing
            (nvx, nvy), (evx, evy) = vecs
            arm = c["arrowlen"] if c["arrowlen"] else 0.12 * nx
            col = self._color_hex(c["color"])
            vx, vy = x, y
            if not c["atvertex"]:
                vx = x - 0.5 * arm * (nvx + evx)
                vy = y - 0.5 * arm * (nvy + evy)
            for (ex, ey), lbl in (((nvx, nvy), "N"), ((evx, evy), "E")):
                xs, ys = self._arrow_points((vx, vy),
                                            (vx + arm * ex, vy + arm * ey), 0.22)
                ln = pg.PlotDataItem(xs, ys, pen=pg.mkPen(col, width=c["thick"]))
                ln.setZValue(25)
                self.vb.addItem(ln)
                self._compass_items.append(ln)
                t = self._label_text(lbl, col, c["size"], (0.5, 0.5),
                                     (vx + 1.2 * arm * ex, vy + 1.2 * arm * ey))
                self.vb.addItem(t)
                self._compass_items.append(t)

        # ---- scale bar ---------------------------------------------
        def _draw_scalebar(self):
            """(Re)draw the scale bar using the WCS pixel scale to convert the
            requested angular length to pixels.
            """
            for it in self._scalebar_items:
                self.vb.removeItem(it)
            self._scalebar_items = []
            s = self._scalebar
            if not s["on"] or self.model.data is None:
                return
            scale = self._pixscale_arcsec()
            if scale is None:
                self.status.showMessage(
                    "Scale bar needs a celestial WCS (pixel scale).", 4000)
                return
            ny, nx = self.model.data.shape
            x = s["x"] if s["x"] is not None else 0.7 * nx
            y = s["y"] if s["y"] is not None else 0.12 * ny
            arcsec = s["length"] * (60.0 if s["units"] == "arcmin" else 1.0)
            bar_pix = arcsec / scale
            col = self._color_hex(s["color"])
            ln = pg.PlotDataItem([x, x + bar_pix], [y, y],
                                 pen=pg.mkPen(col, width=s["thick"]))
            ln.setZValue(25)
            self.vb.addItem(ln)
            self._scalebar_items.append(ln)
            unit = "\u2033" if s["units"] == "arcsec" else "\u2032"
            t = self._label_text(f"{s['length']:g}{unit}", col, s["size"],
                                 (0.0, 1.2), (x, y))
            self.vb.addItem(t)
            self._scalebar_items.append(t)

        # ---- contours ----------------------------------------------
        def _draw_contours(self):
            """(Re)draw IsocurveItem contours at nlevels levels between minval
            and maxval in the chosen color/style.
            """
            for it in self._contour_items:
                self.vb.removeItem(it)
            self._contour_items = []
            c = self._contour
            if not c["on"] or self.model.data is None:
                return
            d = self.model.data
            lo = c["minval"] if c["minval"] is not None else self.model.min_value
            hi = c["maxval"] if c["maxval"] is not None else self.model.max_value
            n = max(int(c["nlevels"]), 1)
            levels = np.linspace(lo, hi, n + 1)[1:] if n > 1 else [float(hi)]
            style = LINE_STYLES.get(c["linestyle"], QtCore.Qt.SolidLine)
            col = self._color_hex(c["color"])
            for lev in levels:
                iso = pg.IsocurveItem(data=d.T, level=float(lev),
                                      pen=pg.mkPen(col, width=c["thick"],
                                                   style=style))
                iso.setZValue(20)
                self.vb.addItem(iso)
                self._contour_items.append(iso)

        def _redraw_annotations(self):
            """Redraw all annotations after a data/WCS change."""
            self._draw_compass()
            self._draw_scalebar()
            self._draw_contours()
            for a in self._arrows:
                self._redraw_arrow(a)

        # ---- annotation dialogs ------------------------------------
        def _color_combo(self, current):
            """Combo box of the ATV annotation color names."""
            cb = QtWidgets.QComboBox()
            cb.addItems(LABEL_COLOR_NAMES)
            cb.setCurrentText(current)
            return cb

        def _arrow_dialog(self):
            """Arrow dialog (OK/Apply/Cancel): arm/disarm mouse drawing, set
            color/thickness/head size, optionally restyle existing arrows.
            """
            st = self._arrow_style
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Arrow")
            form = QtWidgets.QFormLayout(dlg)
            arm = QtWidgets.QCheckBox("draw arrows with the mouse (drag on image)")
            arm.setChecked(self._arrow_armed)
            form.addRow(arm)
            col = self._color_combo(st["color"])
            form.addRow("color", col)
            thick = QtWidgets.QDoubleSpinBox()
            thick.setRange(0.2, 10.0)
            thick.setSingleStep(0.5)
            thick.setValue(st["thick"])
            form.addRow("line thickness", thick)
            head = QtWidgets.QSpinBox()
            head.setRange(2, 60)
            head.setValue(int(round(st["headfrac"] * 100)))
            head.setSuffix(" %")
            form.addRow("head size (% of length)", head)
            restyle = QtWidgets.QCheckBox("apply to existing arrows too")
            form.addRow(restyle)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                | QtWidgets.QDialogButtonBox.Apply)
            form.addRow(bb)

            def apply():
                """Apply the dialog's style (and optional restyle of existing
                arrows) without closing.
                """
                st.update(color=col.currentText(), thick=thick.value(),
                          headfrac=head.value() / 100.0)
                self._arrow_armed = arm.isChecked()
                if self._arrow_armed:
                    self.status.showMessage(
                        "Arrow: drag on the image to draw arrows.", 4000)
                if restyle.isChecked():
                    for a in self._arrows:
                        a.update(color=st["color"], thick=st["thick"],
                                 headfrac=st["headfrac"])
                        self._redraw_arrow(a)

            bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(apply)
            bb.accepted.connect(lambda: (apply(), dlg.accept()))
            bb.rejected.connect(dlg.reject)
            dlg.exec()

        def _compass_dialog(self):
            """Compass dialog (OK/Apply/Cancel): position, vertex/center
            anchoring, color, thickness, char size, arm length.
            """
            if self.model.data is None:
                return
            if self.model.wcs is None or not self.model.wcs.has_celestial:
                QtWidgets.QMessageBox.information(
                    self, "VTA", "Compass needs a celestial WCS.")
                return
            c = self._compass
            ny, nx = self.model.data.shape
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Compass")
            form = QtWidgets.QFormLayout(dlg)
            en = QtWidgets.QCheckBox("show compass")
            en.setChecked(c["on"])
            form.addRow(en)
            xs = QtWidgets.QSpinBox()
            xs.setRange(0, nx - 1)
            xs.setValue(int(c["x"] if c["x"] is not None else 0.15 * nx))
            ys = QtWidgets.QSpinBox()
            ys.setRange(0, ny - 1)
            ys.setValue(int(c["y"] if c["y"] is not None else 0.15 * ny))
            form.addRow("X center", xs)
            form.addRow("Y center", ys)
            spec = QtWidgets.QComboBox()
            spec.addItems(["vertex of compass", "center of compass"])
            spec.setCurrentIndex(0 if c["atvertex"] else 1)
            form.addRow("coords specify", spec)
            col = self._color_combo(c["color"])
            form.addRow("color", col)
            thick = QtWidgets.QDoubleSpinBox()
            thick.setRange(0.2, 10.0)
            thick.setSingleStep(0.5)
            thick.setValue(c["thick"])
            form.addRow("line thickness", thick)
            size = QtWidgets.QDoubleSpinBox()
            size.setRange(0.3, 4.0)
            size.setSingleStep(0.1)
            size.setValue(c["size"])
            form.addRow("char size", size)
            arm = QtWidgets.QDoubleSpinBox()
            arm.setRange(2.0, 5000.0)
            arm.setValue(c["arrowlen"] if c["arrowlen"] else round(0.12 * nx))
            form.addRow("arrow length (px)", arm)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                | QtWidgets.QDialogButtonBox.Apply)
            form.addRow(bb)

            def apply():
                """Apply the dialog settings and redraw the compass without
                closing.
                """
                c.update(on=en.isChecked(), x=xs.value(), y=ys.value(),
                         atvertex=(spec.currentIndex() == 0),
                         color=col.currentText(), thick=thick.value(),
                         size=size.value(), arrowlen=arm.value())
                self._draw_compass()

            bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(apply)
            bb.accepted.connect(lambda: (apply(), dlg.accept()))
            bb.rejected.connect(dlg.reject)
            dlg.exec()

        def _scalebar_dialog(self):
            """Scale-bar dialog (OK/Apply/Cancel): position, length,
            arcsec/arcmin units, color, thickness, char size.
            """
            if self.model.data is None:
                return
            if self.model.wcs is None or not self.model.wcs.has_celestial:
                QtWidgets.QMessageBox.information(
                    self, "VTA",
                    "Scale bar needs a celestial WCS (pixel scale).")
                return
            s = self._scalebar
            ny, nx = self.model.data.shape
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Scale bar")
            form = QtWidgets.QFormLayout(dlg)
            en = QtWidgets.QCheckBox("show scale bar")
            en.setChecked(s["on"])
            form.addRow(en)
            xs = QtWidgets.QSpinBox()
            xs.setRange(0, nx - 1)
            xs.setValue(int(s["x"] if s["x"] is not None else 0.7 * nx))
            ys = QtWidgets.QSpinBox()
            ys.setRange(0, ny - 1)
            ys.setValue(int(s["y"] if s["y"] is not None else 0.12 * ny))
            form.addRow("X (left end)", xs)
            form.addRow("Y (center)", ys)
            length = QtWidgets.QDoubleSpinBox()
            length.setRange(0.001, 100000.0)
            length.setDecimals(3)
            length.setValue(s["length"])
            form.addRow("bar length", length)
            units = QtWidgets.QComboBox()
            units.addItems(["arcsec", "arcmin"])
            units.setCurrentText(s["units"])
            form.addRow("units", units)
            col = self._color_combo(s["color"])
            form.addRow("color", col)
            thick = QtWidgets.QDoubleSpinBox()
            thick.setRange(0.2, 12.0)
            thick.setSingleStep(0.5)
            thick.setValue(s["thick"])
            form.addRow("line thickness", thick)
            size = QtWidgets.QDoubleSpinBox()
            size.setRange(0.3, 4.0)
            size.setSingleStep(0.1)
            size.setValue(s["size"])
            form.addRow("char size", size)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                | QtWidgets.QDialogButtonBox.Apply)
            form.addRow(bb)

            def apply():
                """Apply the dialog settings and redraw the scale bar without
                closing.
                """
                s.update(on=en.isChecked(), x=xs.value(), y=ys.value(),
                         length=length.value(), units=units.currentText(),
                         color=col.currentText(), thick=thick.value(),
                         size=size.value())
                self._draw_scalebar()

            bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(apply)
            bb.accepted.connect(lambda: (apply(), dlg.accept()))
            bb.rejected.connect(dlg.reject)
            dlg.exec()

        def _contour_dialog(self):
            """Contour dialog (OK/Apply/Cancel): color, line style, thickness,
            min/max level values, and number of levels.
            """
            if self.model.data is None:
                return
            c = self._contour
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Contours")
            form = QtWidgets.QFormLayout(dlg)
            en = QtWidgets.QCheckBox("show contours")
            en.setChecked(c["on"])
            form.addRow(en)
            col = self._color_combo(c["color"])
            form.addRow("color", col)
            ls = QtWidgets.QComboBox()
            ls.addItems(LINE_STYLE_NAMES)
            ls.setCurrentText(c["linestyle"])
            form.addRow("line style", ls)
            thick = QtWidgets.QDoubleSpinBox()
            thick.setRange(0.2, 8.0)
            thick.setSingleStep(0.3)
            thick.setValue(c["thick"])
            form.addRow("line thickness", thick)
            mn = QtWidgets.QDoubleSpinBox()
            mn.setRange(-1e9, 1e9)
            mn.setDecimals(3)
            mn.setValue(c["minval"] if c["minval"] is not None
                        else self.model.min_value)
            form.addRow("min value", mn)
            mx = QtWidgets.QDoubleSpinBox()
            mx.setRange(-1e9, 1e9)
            mx.setDecimals(3)
            mx.setValue(c["maxval"] if c["maxval"] is not None
                        else self.model.max_value)
            form.addRow("max value", mx)
            nl = QtWidgets.QSpinBox()
            nl.setRange(1, 50)
            nl.setValue(int(c["nlevels"]))
            form.addRow("n levels", nl)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                | QtWidgets.QDialogButtonBox.Apply)
            form.addRow(bb)

            def apply():
                """Apply the dialog settings and redraw the contours without
                closing.
                """
                c.update(on=en.isChecked(), color=col.currentText(),
                         linestyle=ls.currentText(), thick=thick.value(),
                         minval=mn.value(), maxval=mx.value(),
                         nlevels=nl.value())
                self._draw_contours()

            bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(apply)
            bb.accepted.connect(lambda: (apply(), dlg.accept()))
            bb.rejected.connect(dlg.reject)
            dlg.exec()

        # ---- core display update -----------------------------------
        def _build_menu(self):
            """Menu bar: File (open/save/header/quit), View (panel toggles),
            Help (help window / about).
            """
            m = self.menuBar().addMenu("File")
            m.addAction("Open\u2026", self.open_file)
            m.addAction("Save image as FITS\u2026", self.save_fits)
            m.addAction("View FITS Header", self.show_header)
            m.addSeparator()
            m.addAction("Quit", self.close)

            # photometry-at-cursor keyboard shortcut
            sc = QtGui.QShortcut(QtGui.QKeySequence("p"), self)
            sc.activated.connect(self.photometry_at_cursor)

            self._apply_cmap()

        def _add_panel_toggles(self):
            """View menu + Help menu, and the toolbar Help button."""
            vm = self.menuBar().addMenu("View")
            a_analysis = QtGui.QAction("Radial / Statistics", self,
                                       checkable=True)
            a_analysis.setChecked(True)
            a_analysis.toggled.connect(self.tabs.setVisible)
            vm.addAction(a_analysis)
            a_view = self.view_dock.toggleViewAction()
            a_view.setText("Magnifier")
            vm.addAction(a_view)

            hm = self.menuBar().addMenu("Help")
            a_help = hm.addAction("VTA help", self.show_help)
            a_help.setShortcut(QtGui.QKeySequence("F1"))
            hm.addAction("About VTA", self.show_about)
            # explicit Help button at the right end of the toolbar
            spacer = QtWidgets.QWidget()
            spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Preferred)
            self._toolbar.addWidget(spacer)
            self._toolbar.addAction("Help", self.show_help)

        def show_help(self):
            """Open the (non-modal) help window; F1 or Help menu/button."""
            if getattr(self, "_help_win", None) is not None \
                    and self._help_win.isVisible():
                self._help_win.raise_()
                return
            win = QtWidgets.QDialog(self)
            win.setWindowTitle("VTA help")
            win.resize(760, 640)
            lay = QtWidgets.QVBoxLayout(win)
            tb = QtWidgets.QTextBrowser()
            tb.setOpenExternalLinks(True)
            tb.setHtml(HELP_HTML + f"<hr><p><i>VTA version "
                       f"{__version__} &middot; David R. Ciardi &middot; "
                       f"heritage: ATV by Aaron Barth</i></p>")
            lay.addWidget(tb)
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            bb.rejected.connect(win.close)
            lay.addWidget(bb)
            win.show()
            self._help_win = win

        def show_about(self):
            """About box with version/heritage information."""
            QtWidgets.QMessageBox.about(
                self, "About VTA",
                "<b>Visualization Tool for Astronomy (VTA)</b> \u2014 an "
                "astronomical FITS image viewer for Python.<br><br>"
                f"Version {__version__}.<br><br>"
                "Built on PySide6, pyqtgraph, astropy, scipy, photutils, "
                "and matplotlib.<br><br>"
                "Heritage: ATV by Aaron Barth.<br>"
                "Author: David R. Ciardi.")

        def photometry_at_cursor(self):
            """Run aperture photometry at the keyboard cursor (the 'p' key)."""
            xy = self._need_cursor()
            if xy is not None:
                self.update_analysis(*xy)

        # ---- actions ------------------------------------------------
        def open_file(self):
            """File-open dialog: list image HDUs, populate the extension
            selector, and load the first image extension.
            """
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open FITS", "", "FITS (*.fits *.fit *.fts *.gz);;All (*)")
            if not path:
                return
            try:
                exts = list_fits_extensions(path)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "VTA", str(e))
                return
            images = [e for e in exts if e["is_image"]]
            if not images:
                QtWidgets.QMessageBox.warning(self, "VTA",
                                              "No image data found in file.")
                return
            self._path = path
            self._populate_ext_combo(images)
            self._load_ext(images[0]["index"])
            self.setWindowTitle(f"VTA  \u2014  {path}")
            self.file_label.setText(os.path.basename(path))
            self.file_label.setToolTip(path)

        def _populate_ext_combo(self, images):
            """Fill (and show/hide) the extension selector for multi-extension
            files.
            """
            self.ext_combo.blockSignals(True)
            self.ext_combo.clear()
            for e in images:
                shp = "x".join(str(s) for s in e["shape"]) if e["shape"] else ""
                self.ext_combo.addItem(f"[{e['index']}] {e['name']} ({shp})",
                                       e["index"])
            self.ext_combo.blockSignals(False)
            multi = len(images) > 1
            self.ext_label.setVisible(multi)
            self.ext_combo.setVisible(multi)

        def _on_ext_changed(self, i):
            """Load the HDU selected in the extension combo."""
            idx = self.ext_combo.itemData(i)
            if idx is not None and getattr(self, "_path", None):
                self._load_ext(int(idx))

        def _load_ext(self, idx):
            """Read HDU idx keeping native dimensionality: 3-D data opens the
            cube viewer, 2-D goes straight to the model; overlays are
            cleared either way.
            """
            try:
                data, header, wfull = read_fits_extension_full(self._path, idx)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "VTA", str(e))
                return
            # drop every overlay from the previous image BEFORE re-ranging,
            # so stale aperture/cursor/vector items can't stretch autoRange
            self._erase_arrows()
            self._clear_spec_overlays()
            self._clear_apertures()
            self._spec_trace = None
            self._cursor_marker.setVisible(False)
            self._cursor_xy = None
            self._stats_xy = None
            self._vec_line.setVisible(False)
            if data.ndim >= 3:
                cube = (data if data.ndim == 3
                        else data.reshape((-1,) + data.shape[-2:]))
                self._setup_cube(cube, header, wfull)
            else:
                self._clear_cube()
                wcs = None
                if wfull is not None:
                    try:
                        wcs = wfull.celestial if wfull.has_celestial else (
                            wfull if wfull.naxis == 2 else None)
                    except Exception:
                        wcs = None
                self.model.set_data(data, header, wcs)
                self.scale_combo.setCurrentText(self.model.scaling)
                self._reset_bc()
                self.refresh(reset_view=True)
            self._capture_original()
            self._update_radial_unit_enabled()
            self._redraw_annotations()
            # a freshly loaded image is the live image, not a blink buffer
            self._current_buffer = None
            self._update_buffer_label()

        def _capture_original(self):
            """Snapshot the freshly-loaded image so Rotate -> Reset can
            restore the original orientation / data / WCS."""
            d = self.model.data
            self._orig_data = d.copy() if d is not None else None
            self._orig_header = (self.model.header.copy()
                                 if self.model.header is not None else None)
            self._orig_wcs = (self.model.wcs.deepcopy()
                              if self.model.wcs is not None else None)

        def _reset_image(self):
            """Rotate -> Reset: restore the originally loaded data/header/WCS
            and clear overlays.
            """
            if getattr(self, "_orig_data", None) is None:
                return
            wcs = self._orig_wcs.deepcopy() if self._orig_wcs is not None else None
            hdr = self._orig_header.copy() if self._orig_header is not None else None
            self.model.set_data(self._orig_data.copy(), hdr, wcs)
            for it in self._ap_items.values():
                it.setVisible(False)
            self._cen_marker.setVisible(False)
            self._cursor_marker.setVisible(False)
            self._cursor_xy = None
            self._stats_xy = None
            self._vec_line.setVisible(False)
            self._erase_arrows()
            self._clear_spec_overlays()
            self._clear_apertures()
            self._spec_trace = None
            self._reset_bc()
            self.refresh(reset_view=True)
            self._redraw_annotations()

        def _reset_bc(self):
            """Reset mouse brightness/contrast to neutral."""
            self._bc = [0.5, 0.526]

        def _on_scaling(self, name):
            """Stretch combo changed: update the model scaling and refresh."""
            self.model.scaling = name
            self.model.autoscale()
            self._reset_bc()
            self.refresh()

        def _on_minmax_edit(self):
            """Apply user-typed display min/max."""
            try:
                self.model.min_value = float(self.min_edit.text())
                self.model.max_value = float(self.max_edit.text())
            except ValueError:
                pass
            self._reset_bc()
            self.refresh()

        def _do_autoscale(self):
            """AutoScale button: robust display range."""
            self.model.autoscale()
            self._reset_bc()
            self.refresh()

        def _do_zscale(self):
            """ZScale button: IRAF zscale display range."""
            self.model.zscale()
            self._reset_bc()
            self.refresh()

        def _do_full(self):
            """Full button: min/max display range."""
            self.model.full_range()
            self._reset_bc()
            self.refresh()

        def _on_cmap(self, *_):
            """Colormap combo / invert toggle changed: apply the colormap."""
            self._apply_cmap()

        def _apply_cmap(self):
            """Apply the selected colormap (built-in or custom ATV map) with
            optional inversion to the image and histogram LUT.
            """
            cmap = get_colormap(self.cmap_combo.currentText())
            lut = cmap.getLookupTable(0.0, 1.0, 256)
            if self.invert_chk.isChecked():
                lut = lut[::-1]
                self.img_item.setLookupTable(lut)
                self.hist.gradient.setColorMap(pg.ColorMap(
                    np.linspace(0, 1, 256), lut))
            else:
                self.img_item.setLookupTable(lut)
                self.hist.gradient.setColorMap(cmap)
            # mirror the LUT into the magnifier
            if getattr(self, "mag_img", None) is not None:
                self.mag_img.setLookupTable(lut)
            self._refresh_aux()

        def _mpl_cmap_name(self):
            """matplotlib colormap (name string, or a Colormap object for the
            custom maps) matching the current selection + invert."""
            name = self.cmap_combo.currentText()
            inv = self.invert_chk.isChecked()
            if name in CUSTOM_CMAPS:
                from matplotlib.colors import LinearSegmentedColormap
                colors = [(p, (r / 255.0, g / 255.0, b / 255.0))
                          for p, (r, g, b) in CUSTOM_CMAPS[name]]
                cm = LinearSegmentedColormap.from_list(name, colors)
                return cm.reversed() if inv else cm
            mpl = "gray" if name == "grey" else name
            return mpl + "_r" if inv else mpl

        def _refresh_aux(self):
            """Re-apply the current colormap / stretch / level window to the
            magnifier and the statistics subimage (so they track the main
            image's color table and brightness/contrast)."""
            if self._cursor_xy is not None:
                self._update_magnifier(*self._cursor_xy)
            if getattr(self, "stat_ax", None) is not None \
                    and self._stats_xy is not None:
                self.update_stats(*self._stats_xy)

        # ---- core display update -----------------------------------
        def _levels_from_bc(self, lo, hi):
            """Map brightness/contrast fractions onto a display level
            window over the natural range [lo, hi]. Left=brighter,
            down=higher contrast (ATV color-mode convention)."""
            bf, cf = self._bc
            span = (hi - lo) or 1.0
            width = span * (2.0 - 1.9 * cf)
            center = lo + bf * span
            return center - width / 2.0, center + width / 2.0

        def refresh(self, reset_view=False):
            """Central redraw: update the dimensions label, stretch the data,
            apply brightness/contrast, set the image, sync the histogram
            region and min/max fields, magnifier, and aux panels.
            """
            if self.model.data is None:
                return
            self._rgb_mode = False          # any normal redraw leaves RGB mode
            ny, nx = self.model.data.shape
            self.dims_label.setText(f"{ny} rows \u00d7 {nx} cols")
            disp, lo, hi = self.model.display()
            self._disp = disp
            self._nat_lohi = (lo, hi)
            vlo, vhi = self._levels_from_bc(lo, hi)
            self.img_item.setImage(disp, autoLevels=False)
            self.img_item.setLevels([vlo, vhi])
            self.hist.setLevels(vlo, vhi)
            self.hist.setHistogramRange(lo, hi)
            self.min_edit.setText(f"{self.model.min_value:.4g}")
            self.max_edit.setText(f"{self.model.max_value:.4g}")
            self._sync_beta_field()
            self._apply_cmap()
            if getattr(self, "perm_compass", None) is not None:
                self.perm_compass.set_wcs(self.model.wcs,
                                          self.model.data.shape)
            if reset_view:
                self.vb.autoRange()

        def _color_drag(self, scene_pos):
            """ATV color mode: horizontal drag sets brightness, vertical drag
            sets contrast.
            """
            rect = self.vb.sceneBoundingRect()
            if rect.width() <= 0 or rect.height() <= 0:
                return
            bf = min(max((scene_pos.x() - rect.left()) / rect.width(), 0.0), 1.0)
            cf = min(max((scene_pos.y() - rect.top()) / rect.height(), 0.0), 1.0)
            self._bc = [bf, cf]
            lo, hi = self._nat_lohi
            vlo, vhi = self._levels_from_bc(lo, hi)
            self.img_item.setLevels([vlo, vhi])
            self.hist.setLevels(vlo, vhi)
            self._refresh_aux()

        def _on_mouse_move(self, pos):
            """Track the pointer: update the cursor readout/magnifier in scan-
            like modes. The crosshair marker is hidden whenever the pointer
            is outside the image so it can't drag the auto-ranged view.
            """
            if self.model.data is None:
                return
            if not self.vb.sceneBoundingRect().contains(pos):
                self._cursor_marker.setVisible(False)
                return
            pt = self.vb.mapSceneToView(pos)
            x, y = int(np.floor(pt.x())), int(np.floor(pt.y()))
            ny, nx = self.model.data.shape
            if 0 <= x < nx and 0 <= y < ny:
                self._set_cursor(x, y)
            else:
                self._cursor_marker.setVisible(False)
                self.xy_label.setText("x= --  y= --  value= --")
                self.wcs_label.setText("")

        def _set_cursor(self, x, y):
            """Set the active cursor position (drives readout, magnifier,
            the cursor marker, and where imexam/photometry/cuts center)."""
            self._cursor_xy = (x, y)
            if getattr(self, "_rgb_mode", False) and self._rgb_channels:
                r, g, b = (c[y, x] for c in self._rgb_channels)
                self.xy_label.setText(
                    f"x= {x:<5d} y= {y:<5d} R={r:.4g} G={g:.4g} B={b:.4g}")
            else:
                val = self.model.data[y, x]
                self.xy_label.setText(
                    f"x= {x:<6d} y= {y:<6d} value= {val:.6g}")
            radec, sysname = format_coords(self.model.wcs, x, y, self._coordsys)
            self.wcs_label.setText(f"{radec}   {sysname}".strip())
            self._update_magnifier(x, y)
            self._cursor_marker.setData([x + 0.5], [y + 0.5])
            self._cursor_marker.setVisible(True)

        def _on_coordsys(self, name):
            """Coordinate-system selector changed: re-render the readout at the
            current cursor.
            """
            self._coordsys = name
            if self._cursor_xy is not None:
                self._set_cursor(*self._cursor_xy)

        def _move_cursor(self, dx, dy):
            """Nudge the cursor by (dx, dy) pixels (arrow keys)."""
            if self.model.data is None:
                return
            ny, nx = self.model.data.shape
            if self._cursor_xy is None:
                x, y = nx // 2, ny // 2
            else:
                x, y = self._cursor_xy
            x = int(np.clip(x + dx, 0, nx - 1))
            y = int(np.clip(y + dy, 0, ny - 1))
            self._set_cursor(x, y)

        def eventFilter(self, obj, ev):
            """Application-level key handling: arrow keys nudge the cursor
            (Shift = 10 px), digits 1-3 show blink buffers, r/c plot the
            row/column through the cursor (also when a plot window has
            focus).
            """
            if ev.type() != QtCore.QEvent.KeyPress:
                return super().eventFilter(obj, ev)
            k = ev.key()
            fw = QtWidgets.QApplication.focusWidget()
            typing = isinstance(fw, (QtWidgets.QLineEdit,
                                     QtWidgets.QAbstractSpinBox,
                                     QtWidgets.QComboBox))
            main_active = self.isActiveWindow()
            plot_active = any(
                pw["win"].isActiveWindow()
                for pw in getattr(self, "_plot_windows", {}).values())

            # r / c: plot the row / column through the cursor. Works whether
            # the main window or one of the cut windows is focused.
            if not typing and self._cursor_xy is not None \
                    and (main_active or plot_active):
                if k == QtCore.Qt.Key_R:
                    self.plot_row(self._cursor_xy[1])
                    return True
                if k == QtCore.Qt.Key_C:
                    self.plot_col(self._cursor_xy[0])
                    return True

            # arrow-key cursor nudge + blink digits act on the main window
            if main_active and not typing:
                arrows = {QtCore.Qt.Key_Up: (0, 1), QtCore.Qt.Key_Down: (0, -1),
                          QtCore.Qt.Key_Left: (-1, 0),
                          QtCore.Qt.Key_Right: (1, 0)}
                if k in arrows:
                    step = 10 if (ev.modifiers()
                                  & QtCore.Qt.ShiftModifier) else 1
                    dx, dy = arrows[k]
                    self._move_cursor(dx * step, dy * step)
                    return True
                blink = {QtCore.Qt.Key_1: 0, QtCore.Qt.Key_2: 1,
                         QtCore.Qt.Key_3: 2}
                if k in blink and any(self._blink):
                    self._show_blink(blink[k])
                    return True
            return super().eventFilter(obj, ev)

        # ---- view aids: magnifier, pixel table ----------------------
        def _build_view_dock(self):
            """Left dock: filename, image dimensions, magnifier, magnifier-size
            control.
            """
            dock = QtWidgets.QDockWidget("View", self)
            dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea
                                 | QtCore.Qt.RightDockWidgetArea)
            w = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(w)
            lay.setContentsMargins(3, 3, 3, 3)

            # file currently displayed
            self.file_label = QtWidgets.QLabel("(no file)")
            self.file_label.setWordWrap(True)
            self.file_label.setStyleSheet("font-weight: bold;")
            self.file_label.setToolTip("File currently displayed")
            lay.addWidget(self.file_label)

            self.dims_label = QtWidgets.QLabel("")
            self.dims_label.setStyleSheet("font-family: monospace;")
            self.dims_label.setToolTip("Image size (rows \u00d7 columns)")
            lay.addWidget(self.dims_label)

            lay.addWidget(QtWidgets.QLabel("magnifier"))
            self.mag_glw = pg.GraphicsLayoutWidget()
            self.mag_glw.setFixedHeight(230)
            self.mag_glw.setMinimumWidth(210)
            self.mag_vb = self.mag_glw.addViewBox(lockAspect=True, enableMouse=False)
            self.mag_img = pg.ImageItem()
            self.mag_vb.addItem(self.mag_img)
            self.mag_vline = pg.InfiniteLine(angle=90,
                                             pen=pg.mkPen("#37ff37", width=0.8))
            self.mag_hline = pg.InfiniteLine(angle=0,
                                             pen=pg.mkPen("#37ff37", width=0.8))
            self.mag_vb.addItem(self.mag_vline)
            self.mag_vb.addItem(self.mag_hline)
            lay.addWidget(self.mag_glw)

            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("mag size"))
            self.mag_spin = QtWidgets.QSpinBox()
            self.mag_spin.setRange(5, 101)
            self.mag_spin.setSingleStep(2)
            self.mag_spin.setValue(21)
            self.mag_spin.valueChanged.connect(self._on_mag_size)
            row.addWidget(self.mag_spin)
            row.addStretch(1)
            lay.addLayout(row)

            lay.addWidget(QtWidgets.QLabel("compass (N/E)"))
            self.perm_compass = CompassWidget()
            self.perm_compass.setToolTip(
                "Orientation of North and East from the image WCS")
            lay.addWidget(self.perm_compass)
            lay.addStretch(1)

            dock.setWidget(w)
            self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
            self.view_dock = dock
            self._mag_half = 10

        def _on_mag_size(self, n):
            """Magnifier size spinbox changed: redraw the magnifier."""
            self._mag_half = max(2, (int(n) - 1) // 2)
            if self._cursor_xy is not None:
                self._update_magnifier(*self._cursor_xy)

        def _update_magnifier(self, x, y):
            """Render the zoomed cutout around the cursor with the current
            stretch/colormap into the magnifier.
            """
            if self._disp is None or getattr(self, "mag_img", None) is None:
                return
            h = self._mag_half
            ny, nx = self._disp.shape
            x0, x1 = max(x - h, 0), min(x + h + 1, nx)
            y0, y1 = max(y - h, 0), min(y + h + 1, ny)
            cut = self._disp[y0:y1, x0:x1]
            self.mag_img.setImage(cut, autoLevels=False)
            self.mag_img.setLevels(self.img_item.levels)
            self.mag_img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
            self.mag_vline.setPos(x + 0.5)
            self.mag_hline.setPos(y + 0.5)
            self.mag_vb.setRange(xRange=(x - h, x + h + 1),
                                 yRange=(y - h, y + h + 1), padding=0)

        # ---- analysis dock -----------------------------------------
        def _mpl_canvas(self, w, h):
            """Create a dark-themed matplotlib FigureCanvas + axes."""
            fig = Figure(figsize=(w, h), facecolor="#15171c")
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111, facecolor="#0e1014")
            ax.tick_params(colors="#aaaaaa", labelsize=8)
            for s in ax.spines.values():
                s.set_color("#555555")
            fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)
            return canvas, fig, ax

        def _field(self, value, width=64):
            """One labeled QLineEdit in the photometry parameter grid."""
            e = QtWidgets.QLineEdit(str(value))
            e.setFixedWidth(width)
            e.setValidator(QtGui.QDoubleValidator())
            e.editingFinished.connect(self._recompute)
            return e

        def _build_analysis_dock(self):
            """Analysis panel on the right of the central splitter: Photometry
            tab (parameters, readout, radial plot, export) and Statistics tab
            (subimage, stats text, histogram). The splitter handle between the
            image and this panel is user-adjustable and both sides scale.
            """
            self.tabs = QtWidgets.QTabWidget()
            self.tabs.setMinimumWidth(240)
            self._main_splitter.addWidget(self.tabs)
            self._main_splitter.setStretchFactor(1, 0)
            self.analysis_panel = self.tabs

            # ---- Photometry tab ----
            phot = QtWidgets.QWidget()
            pv = QtWidgets.QVBoxLayout(phot)
            form = QtWidgets.QGridLayout()
            self.pf = {}
            defs = [("aprad", 5.0), ("innersky", 10.0), ("outersky", 20.0),
                    ("centerbox", 7), ("gain", 1.0), ("readnoise", 0.0),
                    ("zeropoint", 25.0), ("exptime", 1.0)]
            labels = {"aprad": "aperture r", "innersky": "inner sky",
                      "outersky": "outer sky", "centerbox": "center box",
                      "gain": "gain (e-/ADU)", "readnoise": "read noise",
                      "zeropoint": "zero point", "exptime": "exp time"}
            for i, (key, val) in enumerate(defs):
                r, c = divmod(i, 2)
                form.addWidget(QtWidgets.QLabel(labels[key]), r, c * 2)
                self.pf[key] = self._field(val)
                form.addWidget(self.pf[key], r, c * 2 + 1)
            pv.addLayout(form)

            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("sky"))
            self.sky_combo = QtWidgets.QComboBox()
            self.sky_combo.addItems(["DAOPHOT", "median", "none"])
            self.sky_combo.currentIndexChanged.connect(self._recompute)
            row.addWidget(self.sky_combo)
            row.addWidget(QtWidgets.QLabel("units"))
            self.units_combo = QtWidgets.QComboBox()
            self.units_combo.addItems(["counts", "magnitudes"])
            self.units_combo.setCurrentIndex(1)        # magnitudes by default
            self.units_combo.currentIndexChanged.connect(self._recompute)
            row.addWidget(self.units_combo)
            row.addStretch(1)
            pv.addLayout(row)

            self.phot_result = QtWidgets.QLabel(
                "imexam mode: click a source on the image.")
            self.phot_result.setStyleSheet(
                "font-family: monospace; font-size: 11px;")
            self.phot_result.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse)
            pv.addWidget(self.phot_result)

            self.rad_canvas, self.rad_fig, self.rad_ax = self._mpl_canvas(4.4, 4.4)
            self.rad_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                          QtWidgets.QSizePolicy.Expanding)
            pv.addWidget(self.rad_canvas, 1)
            self.show_ap_chk = QtWidgets.QCheckBox("Show apertures")
            self.show_ap_chk.setChecked(True)
            self.show_ap_chk.setToolTip(
                "Draw the aperture and sky-annulus circles on the image")
            self.show_ap_chk.toggled.connect(self._on_show_apertures)

            urow = QtWidgets.QHBoxLayout()
            urow.addWidget(self.show_ap_chk)
            urow.addStretch(1)
            # mirror the vector-plot control: a checkbox toggles the radial
            # axis from pixels to angular units, with a unit pulldown that is
            # live only while the box is checked
            self.radial_ang_chk = QtWidgets.QCheckBox("radius in angular units")
            self.radial_ang_chk.setToolTip(
                "Show the radial-profile x axis and FWHM in angular units "
                "(needs a celestial WCS); unchecked = pixels")
            self.radial_ang_chk.toggled.connect(self._on_radial_unit)
            self.radial_unit = QtWidgets.QComboBox()
            self.radial_unit.addItems(["arcsec", "arcmin", "degree"])
            self.radial_unit.setCurrentText("arcsec")
            self.radial_unit.setToolTip(
                "Angular unit for the radial-profile x axis and the FWHM")
            self.radial_unit.setEnabled(False)
            self.radial_unit.currentIndexChanged.connect(self._on_radial_unit)
            urow.addWidget(self.radial_ang_chk)
            urow.addWidget(self.radial_unit)
            pv.addLayout(urow)

            save_btn = QtWidgets.QPushButton("Save Radial plot (PNG)\u2026")
            save_btn.clicked.connect(self._save_radial_png)
            pv.addWidget(save_btn)

            # ---- export photometry log (append each measurement) ----
            self._phot_file = None
            erow = QtWidgets.QHBoxLayout()
            self.export_chk = QtWidgets.QCheckBox("Export photometry")
            self.export_chk.setChecked(False)
            self.export_chk.toggled.connect(self._on_export_toggled)
            erow.addWidget(self.export_chk)
            self.export_path = QtWidgets.QLineEdit("vtaphot.csv")
            self.export_path.setToolTip(
                "File to append photometry measurements to while the box "
                "is checked")
            erow.addWidget(self.export_path, 1)
            self.export_browse = QtWidgets.QPushButton("\u2026")
            self.export_browse.setMaximumWidth(32)
            self.export_browse.setToolTip("Choose file / folder\u2026")
            self.export_browse.clicked.connect(self._browse_export_path)
            erow.addWidget(self.export_browse)
            pv.addLayout(erow)
            self.tabs.addTab(phot, "Photometry")

            # ---- Statistics tab ----
            stat = QtWidgets.QWidget()
            sv = QtWidgets.QVBoxLayout(stat)
            srow = QtWidgets.QHBoxLayout()
            srow.addWidget(QtWidgets.QLabel("stats box size"))
            self.statbox_field = self._field(11)
            srow.addWidget(self.statbox_field)
            srow.addStretch(1)
            sv.addLayout(srow)
            self.stat_result = QtWidgets.QLabel(
                "imexam mode: click a point on the image.")
            self.stat_result.setStyleSheet(
                "font-family: monospace; font-size: 11px;")
            self.stat_result.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse)
            sv.addWidget(self.stat_result)
            # subimage display (square, centered)
            self.stat_canvas, self.stat_fig, self.stat_ax = self._mpl_canvas(3.2, 3.0)
            self.stat_fig.subplots_adjust(left=0.10, right=0.90,
                                          top=0.90, bottom=0.10)
            self.stat_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Expanding)
            sv.addWidget(self.stat_canvas, 3)
            # histogram of the subimage pixel values
            self.stat_hist_canvas, self.stat_hist_fig, self.stat_hist_ax = \
                self._mpl_canvas(3.2, 2.0)
            self.stat_hist_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                                QtWidgets.QSizePolicy.Expanding)
            sv.addWidget(self.stat_hist_canvas, 2)
            self.tabs.addTab(stat, "Statistics")

            # ---- Separations tab ----
            self._sep_ref = None      # locked reference measurement
            self._last_meas = None    # most recent imexam measurement
            sep = QtWidgets.QWidget()
            zv = QtWidgets.QVBoxLayout(sep)
            zv.addWidget(QtWidgets.QLabel(
                "imexam two point sources. Click one and press "
                "\u201cSet reference\u201d, then imexam the second."))
            brow = QtWidgets.QHBoxLayout()
            self.sep_setref_btn = QtWidgets.QPushButton("Set reference")
            self.sep_setref_btn.clicked.connect(self._sep_set_reference)
            brow.addWidget(self.sep_setref_btn)
            brow.addWidget(QtWidgets.QLabel("sky units"))
            self.sep_unit = QtWidgets.QComboBox()
            self.sep_unit.addItems(["arcsec", "arcmin", "degree"])
            self.sep_unit.currentIndexChanged.connect(self._update_separations)
            brow.addWidget(self.sep_unit)
            brow.addStretch(1)
            zv.addLayout(brow)
            self.sep_ref_label = QtWidgets.QLabel("reference: (none set)")
            self.sep_tgt_label = QtWidgets.QLabel("target:    (none yet)")
            for lb in (self.sep_ref_label, self.sep_tgt_label):
                lb.setStyleSheet("font-family: monospace; font-size: 11px;")
                lb.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                zv.addWidget(lb)
            self.sep_result = QtWidgets.QLabel(
                "Set a reference, then imexam a second source.")
            self.sep_result.setStyleSheet(
                "font-family: monospace; font-size: 12px;")
            self.sep_result.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse)
            zv.addWidget(self.sep_result)
            arow = QtWidgets.QHBoxLayout()
            self.sep_add_btn = QtWidgets.QPushButton("Add to table")
            self.sep_add_btn.clicked.connect(self._sep_add_row)
            arow.addWidget(self.sep_add_btn)
            self.sep_clear_btn = QtWidgets.QPushButton("Clear table")
            self.sep_clear_btn.clicked.connect(
                lambda: self.sep_table.setRowCount(0))
            arow.addWidget(self.sep_clear_btn)
            self.sep_export_btn = QtWidgets.QPushButton("Export CSV\u2026")
            self.sep_export_btn.clicked.connect(self._sep_export_csv)
            arow.addWidget(self.sep_export_btn)
            zv.addLayout(arow)
            self._sep_cols = ["dx_pix", "dy_pix", "sep_pix", "sep_pix_err",
                              "dRA", "dRA_err", "dDec", "dDec_err",
                              "sep_sky", "sep_sky_err", "unit",
                              "PA_EofN_deg", "PA_err_deg", "dmag", "dmag_err"]
            self.sep_table = QtWidgets.QTableWidget(0, len(self._sep_cols))
            self.sep_table.setHorizontalHeaderLabels(self._sep_cols)
            self.sep_table.horizontalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.Stretch)
            self.sep_table.setEditTriggers(
                QtWidgets.QAbstractItemView.NoEditTriggers)
            zv.addWidget(self.sep_table, 1)
            self.tabs.addTab(sep, "Separations")
            self._build_spectrum_tab()

        def _build_spectrum_tab(self):
            """Spectrum tab (next to Separations): an embedded extracted-
            spectrum plot plus the extraction parameters and Save button,
            laid out like the Photometry tab. Click a trace in 'spectrum'
            mode to extract into this tab (no pop-up windows).
            """
            spec = QtWidgets.QWidget()
            sv = QtWidgets.QVBoxLayout(spec)
            self.spec_canvas, self.spec_fig, self.spec_ax = \
                self._mpl_canvas(4.4, 3.0)
            self.spec_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Expanding)
            sv.addWidget(self.spec_canvas, 1)
            self.spec_status = QtWidgets.QLabel(
                "spectrum mode: click on a spectral trace to extract it.")
            self.spec_status.setStyleSheet("font-family: monospace; "
                                           "font-size: 11px;")
            sv.addWidget(self.spec_status)

            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(3)
            p = self._xpar

            def spin(key, lo, hi):
                s = QtWidgets.QSpinBox()
                s.setRange(lo, hi)
                s.setValue(int(p[key]))
                s.setMaximumWidth(80)
                s.valueChanged.connect(
                    lambda v, k=key: (p.__setitem__(k, int(v)),
                                      self._re_extract()))
                return s

            self._xpar_spins = {}
            specs = [("trace step", "tracestep", 3, 500),
                     ("trace height", "traceheight", 3, 500),
                     ("trace order", "traceorder", 0, 9),
                     ("extract start x", "xstart", 0, 999999),
                     ("extract end x", "xend", 0, 999999),
                     ("aperture lower", "lower", -500, 0),
                     ("aperture upper", "upper", 0, 500),
                     ("lower bg from", "back1", -999, 0),
                     ("lower bg to", "back2", -999, 0),
                     ("upper bg from", "back3", 0, 999),
                     ("upper bg to", "back4", 0, 999)]
            for i, (label, key, lo, hi) in enumerate(specs):
                row, col = divmod(i, 2)
                w = spin(key, lo, hi)
                self._xpar_spins[key] = w
                grid.addWidget(QtWidgets.QLabel(label), row, (col * 2))
                grid.addWidget(w, row, (col * 2) + 1)
            sv.addLayout(grid)

            crow = QtWidgets.QHBoxLayout()
            self.spec_backsub = QtWidgets.QCheckBox("Background subtraction")
            self.spec_backsub.setChecked(p["backsub"])
            self.spec_backsub.toggled.connect(
                lambda v: (p.__setitem__("backsub", bool(v)),
                           self._re_extract()))
            crow.addWidget(self.spec_backsub)
            self.spec_fixed = QtWidgets.QCheckBox("Hold trace fixed")
            self.spec_fixed.setChecked(p["fixed"])
            self.spec_fixed.toggled.connect(
                lambda v: p.__setitem__("fixed", bool(v)))
            crow.addWidget(self.spec_fixed)
            crow.addStretch(1)
            sv.addLayout(crow)

            brow = QtWidgets.QHBoxLayout()
            re_btn = QtWidgets.QPushButton("Re-extract")
            re_btn.clicked.connect(self._re_extract)
            brow.addWidget(re_btn)
            save_btn = QtWidgets.QPushButton("Save spectrum\u2026")
            save_btn.clicked.connect(self._save_spectrum)
            brow.addWidget(save_btn)
            sv.addLayout(brow)
            self.tabs.addTab(spec, "Spectrum")

        # ---- mode + click handling ----------------------------------
        def _on_mode(self, name):
            """Interaction mode changed: set pan/rect mouse mode, the pointer
            cursor, and cancel any half-drawn vector.
            """
            self.mode = name
            if name == "zoom":
                self.vb.setMouseMode(pg.ViewBox.RectMode)
            else:
                self.vb.setMouseMode(pg.ViewBox.PanMode)
            cursors = {"imexam": QtCore.Qt.CrossCursor,
                       "vector": QtCore.Qt.CrossCursor,
                       "row": QtCore.Qt.CrossCursor,
                       "col": QtCore.Qt.CrossCursor,
                       "spectrum": QtCore.Qt.CrossCursor,
                       "color": QtCore.Qt.SizeAllCursor,
                       "zoom": QtCore.Qt.PointingHandCursor,
                       "blink": QtCore.Qt.PointingHandCursor}
            self.glw.setCursor(cursors.get(name, QtCore.Qt.ArrowCursor))
            if name != "vector":          # cancel a half-finished vector
                self._vec_start = None
                self._vec_line.setVisible(False)

        def _on_click(self, ev):
            """Dispatch image clicks by mode: imexam measures, row/col plot
            that line, spectrum extracts, zoom zooms in/out.
            """
            if self.model.data is None:
                return
            pos = ev.scenePos()
            if not self.vb.sceneBoundingRect().contains(pos):
                return
            pt = self.vb.mapSceneToView(pos)
            x, y = pt.x(), pt.y()
            ny, nx = self.model.data.shape
            inside = (0 <= x < nx and 0 <= y < ny)

            if self.mode == "imexam" and ev.button() == QtCore.Qt.LeftButton \
                    and inside:
                self.update_analysis(x, y)
            elif self.mode == "row" and ev.button() == QtCore.Qt.LeftButton \
                    and inside:
                self.plot_row(int(round(y)))
            elif self.mode == "col" and ev.button() == QtCore.Qt.LeftButton \
                    and inside:
                self.plot_col(int(round(x)))
            elif self.mode == "spectrum" \
                    and ev.button() == QtCore.Qt.LeftButton and inside:
                self._extract_spectrum_at(x, y, newcoord=True)
            elif self.mode == "blink" \
                    and ev.button() == QtCore.Qt.LeftButton:
                self._blink_click_next()
            elif self.mode == "zoom":
                if ev.button() == QtCore.Qt.LeftButton:
                    self._zoom_at(x, y, 0.5)        # zoom in
                elif ev.button() == QtCore.Qt.RightButton:
                    self._zoom_at(x, y, 2.0)        # zoom out

        def _zoom_at(self, x, y, factor):
            """Zoom the view about (x, y) by the given factor."""
            (xr, yr) = self.vb.viewRange()
            w = (xr[1] - xr[0]) * factor / 2.0
            h = (yr[1] - yr[0]) * factor / 2.0
            self.vb.setRange(xRange=(x - w, x + w), yRange=(y - h, y + h),
                             padding=0)

        def _vector_drag(self, ev):
            """Click-drag-release a line in vector mode: draw it live, then
            plot the cut on release."""
            if self.model.data is None:
                return
            p0 = self.vb.mapSceneToView(ev.buttonDownScenePos())
            p1 = self.vb.mapSceneToView(ev.scenePos())
            self._vec_line.setData([p0.x(), p1.x()], [p0.y(), p1.y()])
            self._vec_line.setVisible(True)
            if ev.isFinish():
                self._plot_vector(p0.x(), p0.y(), p1.x(), p1.y())

        def _phot_params(self):
            """Collect the photometry parameters from the UI fields (with
            defaults on parse errors).
            """
            def f(key, default):
                """Float field value with fallback default."""
                try:
                    return float(self.pf[key].text())
                except ValueError:
                    return default
            return dict(
                aprad=f("aprad", 5.0), innersky=f("innersky", 10.0),
                outersky=f("outersky", 20.0), centerbox=int(f("centerbox", 7)),
                gain=f("gain", 1.0), readnoise=f("readnoise", 0.0),
                zeropoint=f("zeropoint", 25.0), exptime=f("exptime", 1.0),
                skytype=self.sky_combo.currentIndex(),
                magunits=(self.units_combo.currentIndex() == 1))

        def _recompute(self):
            """Photometry parameter edited: re-measure at the last position."""
            if getattr(self, "_last_click", None) is not None:
                self.update_analysis(*self._last_click)

        # ---- run + display the measurements -------------------------
        def update_analysis(self, x, y):
            """Measure at (x, y): centroid, ATV aperture photometry, radial
            profile + FWHM; update the readout, the radial plot, and the
            aperture overlays.
            """
            data = self.model.data
            if data is None:
                return
            if getattr(self, "_rgb_mode", False):
                self.status.showMessage(
                    "Photometry is not available in RGB mode.", 4000)
                return
            self._last_click = (x, y)
            p = self._phot_params()

            xc, yc, cwarn = centroid_com(data, x, y,
                                         centerbox=p["centerbox"],
                                         outersky=p["outersky"])
            ph = aperture_photometry_atv(
                data, xc, yc, aprad=p["aprad"], innersky=p["innersky"],
                outersky=p["outersky"], gain=p["gain"], readnoise=p["readnoise"],
                skytype=p["skytype"], magunits=p["magunits"],
                zeropoint=p["zeropoint"], exptime=p["exptime"])
            rp = radial_profile(data, xc, yc, outersky=p["outersky"])

            radec, sysname = format_coords(self.model.wcs, xc, yc,
                                           self._coordsys)
            fluxlbl = "magnitude " if ph["magunits"] else "object counts"
            warn = "; ".join(w for w in (cwarn, ph["warning"],
                                         rp["fwhm_warning"]) if w)
            lines = [
                f"cursor    x={int(round(x)):d}  y={int(round(y)):d}",
                f"centroid  x={xc:.2f}  y={yc:.2f}",
            ]
            if radec:
                lines.append(f"          {radec}  {sysname}")
            if ph["magunits"]:
                flux_line = (f"{fluxlbl:<13s} {ph['flux']:.4f}"
                             f"  +/- {ph['flux_err']:.4f}")
            else:
                flux_line = (f"{fluxlbl:<13s} {format_counts(ph['flux'])}"
                             f"  +/- {format_counts(ph['flux_err'])}")
            funit, fscale = self._radial_scale()
            if rp['fwhm'] > 0:
                fwhm_line = f"FWHM ({funit}){' ' * (7 - len(funit))}" \
                            f"{rp['fwhm'] * fscale:.3g}"
            else:
                fwhm_line = "FWHM (pix)    n/a"
            lines += [
                flux_line,
                f"sky level     {format_counts(ph['sky'])}   "
                f"({ph['nsky']} pix)",
                fwhm_line,
            ]
            if warn:
                lines.append(f"\u26a0 {warn}")
            self.phot_result.setText("\n".join(lines))

            self._log_photometry(xc, yc, ph, rp)
            self._plot_radial(rp, p)
            self._draw_apertures(xc, yc, p)
            self.update_stats(int(round(x)), int(round(y)))

            # record this measurement for the Separations tab
            ra_deg = dec_deg = None
            w = self.model.wcs
            if w is not None and getattr(w, "has_celestial", False):
                try:
                    sk = w.pixel_to_world(xc, yc)
                    if isinstance(sk, (list, tuple)):
                        sk = next(s for s in sk
                                  if hasattr(s, "ra") or hasattr(s, "icrs"))
                    icrs = sk.icrs
                    ra_deg, dec_deg = float(icrs.ra.deg), float(icrs.dec.deg)
                except Exception:
                    pass
            # centroid position uncertainty: sigma = FWHM / (S/N), with
            # S/N = counts / counts_err (sigma_x = sigma_y assumed).
            sigma_pos = None
            ce = ph["counts_err"]
            if ce and ce > 0 and rp["fwhm"] > 0 and ph["counts"] > 0:
                snr = ph["counts"] / ce
                if snr > 0:
                    sigma_pos = rp["fwhm"] / snr
            self._last_meas = dict(x=xc, y=yc, ra=ra_deg, dec=dec_deg,
                                   mag=ph["mag"], mag_err=ph["mag_err"],
                                   sigma=sigma_pos, fwhm=rp["fwhm"])
            self._update_separations()

        def _radial_scale(self):
            """Return (unit_label, multiplier) for radial-display units. The
            angular checkbox switches the radial axis (and FWHM) from pixels
            to the unit chosen in the combo, using the WCS pixel scale; falls
            back to pixels when unchecked or without a celestial WCS."""
            if (getattr(self, "radial_ang_chk", None) is not None
                    and self.radial_ang_chk.isChecked()):
                scale = self._pixscale_arcsec()
                if scale:
                    unit = self.radial_unit.currentText()
                    factor = scale / {"arcsec": 1.0, "arcmin": 60.0,
                                      "degree": 3600.0}[unit]
                    return unit, factor
            return "pix", 1.0

        def _update_radial_unit_enabled(self):
            """Enable the angular radial controls only when a celestial WCS is
            present; otherwise uncheck (force pixels) and disable them."""
            if getattr(self, "radial_ang_chk", None) is None:
                return
            has_wcs = bool(self._pixscale_arcsec())
            self.radial_ang_chk.blockSignals(True)
            self.radial_ang_chk.setEnabled(has_wcs)
            if not has_wcs:
                self.radial_ang_chk.setChecked(False)
            self.radial_ang_chk.blockSignals(False)
            self.radial_ang_chk.setToolTip(
                "Show the radial-profile x axis and FWHM in angular units; "
                "unchecked = pixels" if has_wcs else
                "No celestial WCS \u2014 radial units are in pixels")
            self.radial_unit.setEnabled(has_wcs
                                        and self.radial_ang_chk.isChecked())

        def _on_radial_unit(self, *_):
            """Angular checkbox / unit changed: keep the combo live only while
            the box is checked, then re-render the last measurement."""
            if getattr(self, "radial_ang_chk", None) is not None:
                self.radial_unit.setEnabled(self.radial_ang_chk.isEnabled()
                                            and self.radial_ang_chk.isChecked())
            if getattr(self, "_last_click", None) is not None:
                self.update_analysis(*self._last_click)

        def _plot_radial(self, rp, p):
            """Draw the radial profile: binned mean points + spline curve, raw
            pixel scatter, aperture/sky radii markers, FWHM title. The x axis
            and FWHM are shown in pixels or arcsec per the radial-units combo.
            """
            unit, sc = self._radial_scale()
            ax = self.rad_ax
            ax.clear()
            self.rad_fig.set_facecolor("white")
            ax.set_facecolor("white")
            # fill the available panel area (scales with the splitter/window)
            # faint cloud of individual pixel values
            if rp["r_pts"].size:
                ax.scatter(rp["r_pts"] * sc, rp["v_pts"], s=6, c="#bbbbbb",
                           alpha=0.6, linewidths=0, zorder=1)
            # binned radial profile: bigger black points + thin line
            if rp["r_prof"].size:
                ax.plot(rp["r_prof"] * sc, rp["prof"] + rp["sky"], "-",
                        color="black", lw=1.0, zorder=2)
                ax.plot(rp["r_prof"] * sc, rp["prof"] + rp["sky"], "o",
                        color="black", ms=6, zorder=3)
            ax.axhline(rp["sky"], color="#777777", ls=":", lw=1.2)
            ax.axvline(p["aprad"] * sc, color="#1a9d1a", ls="--", lw=1.4)
            ax.axvline(p["innersky"] * sc, color="#1f77b4", ls="--", lw=1.4)
            ax.axvline(p["outersky"] * sc, color="#9467bd", ls="--", lw=1.4)
            ax.set_xlabel(f"Radius ({unit})", color="black", fontsize=10)
            ax.set_ylabel("Counts", color="black", fontsize=10)
            ax.set_title(f"FWHM = {rp['fwhm'] * sc:.2f} {unit}"
                         if rp["fwhm"] > 0 else "FWHM: n/a",
                         color="black", fontsize=11)
            ax.tick_params(colors="black", labelsize=9)
            for s in ax.spines.values():
                s.set_color("black")
            self.rad_canvas.draw_idle()

        def _save_radial_png(self):
            """Save the radial plot to PNG/PDF/SVG via a file dialog."""
            if not self.rad_ax.has_data():
                QtWidgets.QMessageBox.information(
                    self, "VTA",
                    "Measure a source first (imexam mode), then save.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save radial plot", "radial.png",
                "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
            if path:
                self.rad_fig.savefig(path, dpi=150,
                                     facecolor=self.rad_fig.get_facecolor(),
                                     bbox_inches="tight")

        # ---- export photometry log ---------------------------------
        def _browse_export_path(self):
            """Pick the export file/location (allows navigating to another
            directory). Disabled while a log is open."""
            if self._phot_file is not None:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export photometry to", self.export_path.text(),
                "CSV (*.csv);;Text (*.txt *.dat);;All files (*)")
            if path:
                self.export_path.setText(path)

        def _on_export_toggled(self, on):
            """Check: open the file and start appending each measurement.
            Uncheck: flush and close the file. The filename/browse controls
            are locked while logging."""
            if on:
                path = self.export_path.text().strip() or "vtaphot.csv"
                try:
                    new = (not os.path.exists(path)
                           or os.path.getsize(path) == 0)
                    self._phot_file = open(path, "a")
                    if new:
                        self._phot_file.write(self._phot_header())
                    self._phot_file.flush()
                except OSError as e:
                    QtWidgets.QMessageBox.warning(self, "VTA",
                                                  f"Could not open {path}:\n{e}")
                    self.export_chk.setChecked(False)
                    return
                self.export_path.setEnabled(False)
                self.export_browse.setEnabled(False)
                self.status.showMessage(
                    f"Logging photometry to {path}", 4000)
            else:
                if self._phot_file is not None:
                    try:
                        self._phot_file.flush()
                        self._phot_file.close()
                        self.status.showMessage(
                            f"Saved photometry log: {self._phot_file.name}",
                            4000)
                    except OSError:
                        pass
                    self._phot_file = None
                self.export_path.setEnabled(True)
                self.export_browse.setEnabled(True)

        def _phot_header(self):
            """CSV column header written when a new log file is created."""
            return ("file,xcen,ycen,ra_deg,dec_deg,counts,counts_err,"
                    "mag,mag_err,sky,nsky,fwhm_pix\n")

        def _log_photometry(self, xc, yc, ph, rp):
            """Append one CSV row to the open export log (if any). Sky
            coordinates are ICRS (J2000) decimal degrees, taken straight
            from the WCS regardless of the readout's display system."""
            if self._phot_file is None:
                return
            fname = os.path.basename(getattr(self, "_path", "") or "-")
            ra_deg = dec_deg = ""
            w = self.model.wcs
            if w is not None and getattr(w, "has_celestial", False):
                try:
                    sk = w.pixel_to_world(xc, yc)
                    if isinstance(sk, (list, tuple)):
                        sk = next(s for s in sk
                                  if hasattr(s, "ra") or hasattr(s, "icrs"))
                    icrs = sk.icrs
                    ra_deg = f"{icrs.ra.deg:.7f}"
                    dec_deg = f"{icrs.dec.deg:+.7f}"
                except Exception:
                    pass
            row = (f"{fname},{xc:.3f},{yc:.3f},{ra_deg},{dec_deg},"
                   f"{ph['counts']:.4f},{ph['counts_err']:.4f},"
                   f"{ph['mag']:.4f},{ph['mag_err']:.4f},"
                   f"{ph['sky']:.4f},{ph['nsky']},{rp['fwhm']:.4f}\n")
            try:
                self._phot_file.write(row)
                self._phot_file.flush()
            except OSError as e:
                QtWidgets.QMessageBox.warning(self, "VTA",
                                              f"Write failed:\n{e}")
                self.export_chk.setChecked(False)

        def _draw_apertures(self, xc, yc, p):
            """Draw the aperture and sky-annulus circles at the measured
            centroid (only if 'show apertures' is enabled).
            """
            show = (not hasattr(self, "show_ap_chk")
                    or self.show_ap_chk.isChecked())
            th = np.linspace(0, 2 * np.pi, 120)
            for key in ("aprad", "innersky", "outersky"):
                r = p[key]
                self._ap_items[key].setData(xc + r * np.cos(th),
                                            yc + r * np.sin(th))
                self._ap_items[key].setVisible(show)
            self._cen_marker.setData([xc], [yc])
            self._cen_marker.setVisible(show)

        def _clear_apertures(self):
            """Hide the aperture/sky circles and the centroid marker (e.g.
            when a new image is loaded or the geometry changes)."""
            for it in self._ap_items.values():
                it.setVisible(False)
            self._cen_marker.setVisible(False)

        def _on_show_apertures(self, on):
            """Toggle visibility of the most recent aperture overlay."""
            has = (self._ap_items
                   and self._ap_items["aprad"].xData is not None
                   and len(self._ap_items["aprad"].xData) > 0)
            for it in self._ap_items.values():
                it.setVisible(bool(on) and has)
            self._cen_marker.setVisible(bool(on) and has)

        def update_stats(self, x, y):
            """Statistics tab: compute box statistics about the cursor, render
            the square subimage with the current display settings, and the
            pixel histogram.
            """
            data = self.model.data
            if data is None:
                return
            self._stats_xy = (x, y)
            try:
                box = int(float(self.statbox_field.text()))
            except ValueError:
                box = 11
            st = box_statistics(data, x, y, boxsize=box)
            self.stat_result.setText(
                f"box center  x={x:d}  y={y:d}\n"
                f"# pixels    {st['npix']:d}\n"
                f"total       {format_counts(st['total'])}\n"
                f"min         {st['min']:.6g}\n"
                f"max         {st['max']:.6g}\n"
                f"mean        {st['mean']:.6g}\n"
                f"median      {st['median']:.6g}\n"
                f"std dev     {st['std']:.6g}")
            ax = self.stat_ax
            ax.clear()
            # match the main image: same stretch, colormap, and level window
            disp_cut, lo, hi = transform_image(
                st["cut"], self.model.scaling, self.model.min_value,
                self.model.max_value, self.model.asinh_beta)
            vlo, vhi = self._levels_from_bc(lo, hi)
            ax.imshow(disp_cut, origin="lower", cmap=self._mpl_cmap_name(),
                      interpolation="nearest", aspect="equal",
                      vmin=vlo, vmax=vhi)
            ax.set_anchor("C")
            ax.set_title(f"{box}\u00d7{box} region", color="#dddddd", fontsize=9)
            ax.tick_params(colors="#aaaaaa", labelsize=8)
            self.stat_canvas.draw_idle()
            # histogram of the subimage pixel values
            hax = self.stat_hist_ax
            hax.clear()
            hax.set_facecolor("#0e1014")
            vals = st["cut"][np.isfinite(st["cut"])].ravel()
            if vals.size:
                hax.hist(vals, bins=50, color="#7fd4ff",
                         edgecolor="#0e1014", linewidth=0.4)
            hax.set_xlabel("Pixel value", color="#aaaaaa", fontsize=8)
            hax.set_ylabel("Number", color="#aaaaaa", fontsize=8)
            hax.tick_params(colors="#aaaaaa", labelsize=8)
            for s in hax.spines.values():
                s.set_color("#555555")
            self.stat_hist_canvas.draw_idle()

        # ---- separations between two imexam point sources -----------
        def _sep_fmt_meas(self, m):
            """One-line description of a stored measurement."""
            if m is None:
                return "(none)"
            s = f"x={m['x']:.2f} y={m['y']:.2f}"
            if m["ra"] is not None:
                s += f"   RA={m['ra']:.6f} Dec={m['dec']:+.6f} deg"
            if m["mag"] is not None and np.isfinite(m["mag"]):
                s += f"   mag={m['mag']:.3f}"
            return s

        def _sep_set_reference(self):
            """Lock the most recent imexam measurement as the reference."""
            if self._last_meas is None:
                self.status.showMessage(
                    "imexam a source first, then Set reference.", 4000)
                return
            self._sep_ref = dict(self._last_meas)
            self.sep_ref_label.setText("reference: "
                                       + self._sep_fmt_meas(self._sep_ref))
            self._update_separations()

        def _compute_separation(self, ref, tgt, unit):
            """Return a dict of separation quantities between ref and tgt.
            Sky quantities are None when either lacks RA/Dec."""
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            dx = tgt["x"] - ref["x"]
            dy = tgt["y"] - ref["y"]
            out = dict(dx=dx, dy=dy, sep_pix=float(np.hypot(dx, dy)),
                       dra=None, ddec=None, sep_sky=None, pa=None,
                       dmag=None, unit=unit,
                       esep_pix=None, edra=None, eddec=None, esep_sky=None,
                       epa=None, edmag=None)
            if (ref["mag"] is not None and tgt["mag"] is not None
                    and np.isfinite(ref["mag"]) and np.isfinite(tgt["mag"])):
                out["dmag"] = tgt["mag"] - ref["mag"]
                mer, met = ref.get("mag_err"), tgt.get("mag_err")
                if mer is not None and met is not None:
                    out["edmag"] = float(np.hypot(mer, met))

            # propagate the per-source centroid sigmas (sigma_x = sigma_y).
            # For isotropic position errors the combined per-axis sigma is
            # s = sqrt(sigma_ref^2 + sigma_tgt^2); then
            #   sigma(sep_pix) = s,
            #   sigma(PA)      = s / sep_pix  (radians),
            #   sigma(dRA) = sigma(dDec) = sigma(sep_sky) = s * pixel_scale.
            sr, st = ref.get("sigma"), tgt.get("sigma")
            s_comb = (float(np.hypot(sr, st))
                      if sr is not None and st is not None else None)
            if s_comb is not None:
                out["esep_pix"] = s_comb
                if out["sep_pix"] > 0:
                    out["epa"] = float(np.degrees(s_comb / out["sep_pix"]))

            if None not in (ref["ra"], ref["dec"], tgt["ra"], tgt["dec"]):
                uq = {"arcsec": u.arcsec, "arcmin": u.arcmin,
                      "degree": u.deg}[unit]
                r = SkyCoord(ref["ra"] * u.deg, ref["dec"] * u.deg)
                t = SkyCoord(tgt["ra"] * u.deg, tgt["dec"] * u.deg)
                dra, ddec = r.spherical_offsets_to(t)   # cos-dec corrected
                out["dra"] = dra.to_value(uq)
                out["ddec"] = ddec.to_value(uq)
                out["sep_sky"] = r.separation(t).to_value(uq)
                out["pa"] = r.position_angle(t).to_value(u.deg)  # East of North
                scale = self._pixscale_arcsec()           # arcsec / pixel
                if s_comb is not None and scale:
                    e = (s_comb * scale * u.arcsec).to_value(uq)
                    out["esep_sky"] = e
                    out["edra"] = e
                    out["eddec"] = e
            return out

        def _update_separations(self):
            """Refresh the live separations readout from reference + last."""
            if not hasattr(self, "sep_result"):
                return
            ref, tgt = self._sep_ref, self._last_meas
            self.sep_tgt_label.setText("target:    " + self._sep_fmt_meas(tgt))
            if ref is None or tgt is None:
                return
            unit = self.sep_unit.currentText()
            s = self._compute_separation(ref, tgt, unit)

            def pm(v, fmt="{:.4f}"):
                return "" if v is None else "  +/- " + fmt.format(v)
            lines = [f"delta x        {s['dx']:+.3f} pix",
                     f"delta y        {s['dy']:+.3f} pix",
                     f"separation     {s['sep_pix']:.3f}"
                     f"{pm(s['esep_pix'], '{:.3f}')} pix"]
            if s["sep_sky"] is not None:
                lines += [
                    f"delta RA       {s['dra']:+.4f}{pm(s['edra'])} {unit}",
                    f"delta Dec      {s['ddec']:+.4f}{pm(s['eddec'])} {unit}",
                    f"sky separation {s['sep_sky']:.4f}"
                    f"{pm(s['esep_sky'])} {unit}",
                    f"PA (E of N)    {s['pa']:.3f}"
                    f"{pm(s['epa'], '{:.3f}')} deg"]
            else:
                lines.append("(no celestial WCS \u2014 sky quantities n/a)")
            lines.append(f"delta mag      {s['dmag']:+.4f}{pm(s['edmag'])}"
                         if s["dmag"] is not None else "delta mag      n/a")
            if s["esep_pix"] is None:
                lines.append("(position errors need FWHM and S/N > 0)")
            self.sep_result.setText("\n".join(lines))

        def _sep_add_row(self):
            """Append the current reference/target separation to the table."""
            if self._sep_ref is None or self._last_meas is None:
                self.status.showMessage(
                    "Need a reference and a target measurement first.", 4000)
                return
            unit = self.sep_unit.currentText()
            s = self._compute_separation(self._sep_ref, self._last_meas, unit)

            def cell(v, fmt="{:.4f}"):
                return "" if v is None else fmt.format(v)
            vals = [cell(s["dx"], "{:.3f}"), cell(s["dy"], "{:.3f}"),
                    cell(s["sep_pix"], "{:.3f}"), cell(s["esep_pix"], "{:.3f}"),
                    cell(s["dra"]), cell(s["edra"]),
                    cell(s["ddec"]), cell(s["eddec"]),
                    cell(s["sep_sky"]), cell(s["esep_sky"]), unit,
                    cell(s["pa"], "{:.3f}"), cell(s["epa"], "{:.3f}"),
                    cell(s["dmag"]), cell(s["edmag"])]
            row = self.sep_table.rowCount()
            self.sep_table.insertRow(row)
            for c, v in enumerate(vals):
                self.sep_table.setItem(row, c,
                                       QtWidgets.QTableWidgetItem(str(v)))

        def _sep_export_csv(self):
            """Export the separations table to a CSV file."""
            if self.sep_table.rowCount() == 0:
                QtWidgets.QMessageBox.information(
                    self, "VTA", "No separations to export yet.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export separations to", "vtasep.csv",
                "CSV (*.csv);;All files (*)")
            if not path:
                return
            try:
                fname = os.path.basename(getattr(self, "_path", "") or "-")
                with open(path, "w") as f:
                    f.write(f"# file: {fname}\n")
                    f.write(",".join(self._sep_cols) + "\n")
                    for r in range(self.sep_table.rowCount()):
                        cells = [self.sep_table.item(r, c).text()
                                 if self.sep_table.item(r, c) else ""
                                 for c in range(self.sep_table.columnCount())]
                        f.write(",".join(cells) + "\n")
                self.status.showMessage(f"Saved separations to {path}", 5000)
            except OSError as e:
                QtWidgets.QMessageBox.warning(self, "VTA", str(e))

        # ---- header viewer ------------------------------------------
        def show_header(self):
            """FITS header viewer window with incremental find."""
            if self.model.header is None:
                QtWidgets.QMessageBox.information(self, "VTA",
                                                  "No header loaded.")
                return
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("FITS Header")
            dlg.resize(680, 580)
            lay = QtWidgets.QVBoxLayout(dlg)
            find = QtWidgets.QLineEdit()
            find.setPlaceholderText("find (press Enter for next)\u2026")
            lay.addWidget(find)
            txt = QtWidgets.QPlainTextEdit()
            txt.setReadOnly(True)
            txt.setStyleSheet("font-family: monospace; font-size: 11px;")
            try:
                cards = self.model.header.tostring(sep="\n", endcard=False,
                                                   padding=False)
            except Exception:
                cards = "\n".join(f"{k} = {v}"
                                  for k, v in self.model.header.items())
            txt.setPlainText(cards)
            lay.addWidget(txt)

            def do_find():
                """Find/highlight the next occurrence of the search text."""
                if not find.text():
                    return
                if not txt.find(find.text()):       # wrap to top
                    cur = txt.textCursor()
                    cur.movePosition(QtGui.QTextCursor.Start)
                    txt.setTextCursor(cur)
                    txt.find(find.text())
            find.returnPressed.connect(do_find)
            dlg.show()
            self._header_dlg = dlg                  # keep a reference alive

        # ---- shared plot window (row / col / vector / histogram) ----
        def _get_plot_window(self, kind):
            """Return the (lazily created) standalone plot window for one of
            'row', 'col', 'vector'. Each kind keeps its own window so a row,
            a column, and a vector cut can all be displayed at once."""
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
            if not hasattr(self, "_plot_windows"):
                self._plot_windows = {}
            pw = self._plot_windows.get(kind)
            if pw is not None and pw["win"].isVisible():
                return pw
            win = QtWidgets.QDialog(self)
            win.setWindowTitle({"row": "VTA \u2014 row plot",
                                "col": "VTA \u2014 column plot",
                                "vector": "VTA \u2014 vector plot"}[kind])
            win.resize(620, 460)
            lay = QtWidgets.QVBoxLayout(win)
            canvas, fig, ax = self._mpl_canvas(5.4, 3.6)
            ctrl = QtWidgets.QHBoxLayout()
            ctrl.setContentsMargins(2, 2, 2, 2)
            ctrl.addWidget(NavigationToolbar2QT(canvas, win))
            pw = {"win": win, "canvas": canvas, "fig": fig, "ax": ax}
            if kind in ("row", "col"):
                box = QtWidgets.QWidget()
                h = QtWidgets.QHBoxLayout(box)
                h.setContentsMargins(8, 0, 0, 0)
                h.addWidget(QtWidgets.QLabel("Row:" if kind == "row"
                                             else "Column:"))
                spin = QtWidgets.QSpinBox()
                spin.setMaximumWidth(110)
                spin.valueChanged.connect(
                    lambda v, k=kind: self._on_plot_spin(k, v))
                h.addWidget(spin)
                ctrl.addWidget(box)
                pw["spin"] = spin
            else:                              # vector: angular-distance axis
                box = QtWidgets.QWidget()
                h = QtWidgets.QHBoxLayout(box)
                h.setContentsMargins(8, 0, 0, 0)
                chk = QtWidgets.QCheckBox("x axis in angular distance")
                chk.toggled.connect(self._on_vec_units)
                combo = QtWidgets.QComboBox()
                combo.addItems(["arcsec", "arcmin", "degree"])
                combo.setCurrentText("arcsec")
                combo.currentIndexChanged.connect(self._on_vec_units)
                h.addWidget(chk)
                h.addWidget(combo)
                ctrl.addWidget(box)
                pw["ang_chk"] = chk
                pw["unit_combo"] = combo
            ctrl.addStretch(1)
            lay.addLayout(ctrl)
            canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Expanding)
            lay.addWidget(canvas, 1)
            self._plot_windows[kind] = pw
            win.show()
            win.raise_()
            return pw

        def _finish_plot(self, pw, title, xlabel, ylabel):
            """Common styling for the cut/spectrum plots: white background,
            dark lines/labels, title, redraw.
            """
            ax = pw["ax"]
            pw["fig"].set_facecolor("white")
            ax.set_facecolor("white")
            ax.set_title(title, color="black", fontsize=10)
            ax.set_xlabel(xlabel, color="black")
            ax.set_ylabel(ylabel, color="black")
            ax.tick_params(colors="black", labelsize=8)
            for s in ax.spines.values():
                s.set_color("black")
            pw["canvas"].draw_idle()

        def _need_cursor(self):
            """Return the keyboard-cursor position or prompt the user to set
            one.
            """
            if self.model.data is None:
                return None
            if self._cursor_xy is None:
                QtWidgets.QMessageBox.information(
                    self, "VTA", "Move the cursor over the image first.")
                return None
            return self._cursor_xy

        def plot_row(self, y0=None):
            """Open/update the row-plot window for row y0 (default: cursor
            row).
            """
            if self.model.data is None:
                return
            ny, nx = self.model.data.shape
            if y0 is None:
                y0 = self._cursor_xy[1] if self._cursor_xy else ny // 2
            y0 = int(np.clip(int(y0), 0, ny - 1))
            pw = self._get_plot_window("row")
            pw["spin"].blockSignals(True)
            pw["spin"].setRange(0, ny - 1)
            pw["spin"].setValue(y0)
            pw["spin"].blockSignals(False)
            self._render_row(y0)

        def plot_col(self, x0=None):
            """Open/update the column-plot window for column x0 (default:
            cursor column).
            """
            if self.model.data is None:
                return
            ny, nx = self.model.data.shape
            if x0 is None:
                x0 = self._cursor_xy[0] if self._cursor_xy else nx // 2
            x0 = int(np.clip(int(x0), 0, nx - 1))
            pw = self._get_plot_window("col")
            pw["spin"].blockSignals(True)
            pw["spin"].setRange(0, nx - 1)
            pw["spin"].setValue(x0)
            pw["spin"].blockSignals(False)
            self._render_col(x0)

        def _on_plot_spin(self, kind, v):
            """Row/column spinbox changed: re-render that window."""
            if kind == "row":
                self._render_row(v)
            else:
                self._render_col(v)

        def _render_row(self, y):
            """Plot pixel values along one row."""
            pw = self._get_plot_window("row")
            cols, vals = row_values(self.model.data, y)
            ax = pw["ax"]
            ax.clear()
            ax.set_facecolor("white")
            ax.step(cols, vals, where="mid", color="#1f4e79", lw=1.2)
            self._finish_plot(pw, f"Row {int(y)}", "Column", "Pixel value")

        def _render_col(self, x):
            """Plot pixel values along one column."""
            pw = self._get_plot_window("col")
            rows, vals = col_values(self.model.data, x)
            ax = pw["ax"]
            ax.clear()
            ax.set_facecolor("white")
            ax.step(rows, vals, where="mid", color="#1f4e79", lw=1.2)
            self._finish_plot(pw, f"Column {int(x)}", "Row", "Pixel value")

        def _plot_vector(self, x0, y0, x1, y1):
            """Open/update the vector-cut window; enable the angular-distance
            controls only when the WCS provides a pixel scale.
            """
            self._vec_pts = (x0, y0, x1, y1)
            pw = self._get_plot_window("vector")
            scale = self._pixscale_arcsec()
            have_scale = scale is not None
            chk = pw["ang_chk"]
            chk.blockSignals(True)
            chk.setEnabled(have_scale)
            if not have_scale:
                chk.setChecked(False)
            chk.blockSignals(False)
            chk.setToolTip("" if have_scale else "No pixel scale "
                           "(celestial WCS) available in the header.")
            pw["unit_combo"].setEnabled(have_scale and chk.isChecked())
            self._render_vector()

        def _on_vec_units(self, *_):
            """Angular checkbox/units changed: re-render the vector cut."""
            pw = getattr(self, "_plot_windows", {}).get("vector")
            if pw is None or not pw["win"].isVisible():
                return
            pw["unit_combo"].setEnabled(pw["ang_chk"].isEnabled()
                                        and pw["ang_chk"].isChecked())
            if getattr(self, "_vec_pts", None) is not None:
                self._render_vector()

        def _render_vector(self):
            """Plot the interpolated cut, converting the x axis to
            arcsec/arcmin/degree when requested.
            """
            if getattr(self, "_vec_pts", None) is None:
                return
            pw = self._get_plot_window("vector")
            x0, y0, x1, y1 = self._vec_pts
            dist, vals = vector_cut(self.model.data, x0, y0, x1, y1)
            npix = dist[-1]
            xlabel = "Distance (pixels)"
            if pw["ang_chk"].isChecked():
                scale = self._pixscale_arcsec()
                if scale is not None:
                    unit = pw["unit_combo"].currentText()
                    factor = scale / {"arcsec": 1.0, "arcmin": 60.0,
                                      "degree": 3600.0}[unit]
                    dist = dist * factor
                    xlabel = f"Distance ({unit})"
            ax = pw["ax"]
            ax.clear()
            ax.set_facecolor("white")
            ax.step(dist, vals, where="mid", color="#1f4e79", lw=1.2)
            self._finish_plot(
                pw, f"Vector [{x0:.0f},{y0:.0f}] \u2192 [{x1:.0f},{y1:.0f}]  "
                f"(len {npix:.1f} px)", xlabel, "Pixel value")

        # ---- spectral extraction (port of ATV atvextract) ------------
        def _clear_spec_overlays(self):
            """Remove the spectral-extraction overlays from the image."""
            for it in self._spec_items:
                self.vb.removeItem(it)
            self._spec_items = []

        def _extract_spectrum_at(self, x, y, newcoord=False):
            """Trace + extract a spectrum, ATV style. Called from a click in
            'spectrum' mode and from the parameter dialog (re-extract)."""
            d = self.model.data
            if d is None or getattr(self, "_rgb_mode", False):
                return
            ny, nx = d.shape
            if nx < 50 or ny < 20:
                self.status.showMessage(
                    "Image too small for spectral extraction.", 4000)
                return
            p = self._xpar
            if newcoord:
                self._spec_init = (float(x), float(y))
                p["xstart"], p["xend"] = 0, nx - 1
                # reflect the auto-filled x region in the tab's spinboxes
                if hasattr(self, "_xpar_spins"):
                    for k in ("xstart", "xend"):
                        w = self._xpar_spins[k]
                        w.blockSignals(True)
                        w.setRange(0, max(nx - 1, w.maximum()))
                        w.setValue(int(p[k]))
                        w.blockSignals(False)
            if self._spec_init is None:
                return
            p["traceheight"] = min(p["traceheight"], ny)
            x0, y0 = self._spec_init
            xregion = (int(np.clip(p["xstart"], 0, nx - 1)),
                       int(np.clip(p["xend"], 0, nx - 1)))
            if xregion[1] <= xregion[0]:
                xregion = (0, nx - 1)
            hdr = self.model.header
            if newcoord and hdr is not None and hdr.get("DISPAXIS") == 2:
                self.status.showMessage(
                    "Header has DISPAXIS=2 (vertical dispersion) \u2014 "
                    "extraction traces along x; use Rotate 90\u00b0 first.",
                    6000)
            try:
                if p["fixed"] and self._spec_trace is not None:
                    centers, points, xspec, ftrace = self._spec_trace
                else:
                    centers, points, xspec, ftrace = trace_spectrum(
                        d, x0, y0, tracestep=p["tracestep"],
                        traceheight=p["traceheight"],
                        traceorder=p["traceorder"], xregion=xregion)
                    self._spec_trace = (centers, points, xspec, ftrace)
                back = (p["back1"], p["back2"], p["back3"], p["back4"])
                self._spec = extract_spectrum(
                    d, xspec, ftrace, lower=p["lower"], upper=p["upper"],
                    backsub=p["backsub"], back=back)
            except Exception as e:
                self.status.showMessage(f"Extraction failed: {e}", 6000)
                return
            self._draw_spec_overlays(centers, points, xspec, ftrace)
            self._plot_spectrum(xspec, self._spec)
            rms = float(np.std(points - np.interp(centers, xspec, ftrace)))
            ngood = int(np.count_nonzero(self._spec))
            if hasattr(self, "spec_status"):
                self.spec_status.setText(
                    f"trace @ click ({x0:.0f},{y0:.0f})   x {xregion[0]}\u2013"
                    f"{xregion[1]}   trace rms {rms:.2f} px   "
                    f"{ngood} pts extracted")

        def _draw_spec_overlays(self, centers, points, xspec, ftrace):
            """ATV-style overlays: trace points (+), fitted trace (blue),
            aperture edges (yellow), background regions (magenta)."""
            self._clear_spec_overlays()
            p = self._xpar
            sc = pg.ScatterPlotItem(x=centers + 0.5, y=points + 0.5,
                                    symbol="+", size=9,
                                    pen=pg.mkPen("#7fff7f", width=1),
                                    brush=None)
            sc.setZValue(28)
            self.vb.addItem(sc)
            self._spec_items.append(sc)
            xs = xspec + 0.5
            specs = [(ftrace, "#5599ff", 1.6, QtCore.Qt.SolidLine),
                     (ftrace + p["lower"], "#ffe24d", 1.2, QtCore.Qt.SolidLine),
                     (ftrace + p["upper"], "#ffe24d", 1.2, QtCore.Qt.SolidLine)]
            if p["backsub"]:
                for b in ("back1", "back2", "back3", "back4"):
                    specs.append((ftrace + p[b], "#ff37ff", 1.0,
                                  QtCore.Qt.DashLine))
            for arr, col, w, style in specs:
                ln = pg.PlotDataItem(xs, arr + 0.5,
                                     pen=pg.mkPen(col, width=w, style=style))
                ln.setZValue(27)
                self.vb.addItem(ln)
                self._spec_items.append(ln)

        def _plot_spectrum(self, xspec, spec):
            """Plot the extracted spectrum (Column vs Counts) into the
            Spectrum tab and bring that tab to the front."""
            ax = self.spec_ax
            ax.clear()
            self.spec_fig.set_facecolor("white")
            ax.set_facecolor("white")
            ax.step(xspec, spec, where="mid", color="#1f4e79", lw=1.2)
            ax.set_title("Extracted Spectrum", color="black", fontsize=10)
            ax.set_xlabel("Column", color="black")
            ax.set_ylabel("Counts", color="black")
            ax.tick_params(colors="black", labelsize=8)
            for s in ax.spines.values():
                s.set_color("black")
            self.spec_canvas.draw_idle()
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "Spectrum":
                    self.tabs.setCurrentIndex(i)
                    break

        def _re_extract(self):
            """Re-run the extraction at the last clicked position with the
            current parameters.
            """
            if self._spec_init is not None:
                self._extract_spectrum_at(*self._spec_init, newcoord=False)

        def _save_spectrum(self):
            """Save the extracted spectrum as 1-D FITS (pixel WCS + inherited
            OBJECT/EXPTIME) or two-column text.
            """
            if self._spec is None or self._spec_trace is None:
                QtWidgets.QMessageBox.information(
                    self, "VTA", "Extract a spectrum first.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save extracted spectrum", "spectrum.fits",
                "FITS (*.fits);;Text (*.txt *.dat)")
            if not path:
                return
            xspec = self._spec_trace[2]
            try:
                if path.lower().endswith((".txt", ".dat")):
                    np.savetxt(path, np.column_stack([xspec, self._spec]),
                               fmt="%.6g", header="column  counts")
                else:
                    from astropy.io import fits
                    hdu = fits.PrimaryHDU(self._spec.astype(np.float32))
                    hdu.header["CRPIX1"] = 1
                    hdu.header["CRVAL1"] = float(xspec[0])
                    hdu.header["CDELT1"] = 1.0
                    hdu.header["CTYPE1"] = "PIXEL"
                    hdu.header["BUNIT"] = "counts"
                    if self.model.header is not None:
                        for k in ("OBJECT", "EXPTIME", "DATE-OBS"):
                            if k in self.model.header:
                                hdu.header[k] = self.model.header[k]
                    hdu.header.add_history("VTA spectral extraction")
                    hdu.writeto(path, overwrite=True)
                self.status.showMessage(f"Saved spectrum to {path}", 5000)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "VTA", str(e))

    return QtWidgets, ImageModel, Viewer


def main(argv=None):
    """Command-line entry point: parse arguments, build the GUI, load an
    optional FITS file (2-D image or 3-D cube), and start the Qt event loop.
    """
    parser = argparse.ArgumentParser(
        description="vta - Visualization Tool for Astronomy")
    parser.add_argument("image", nargs="?", help="FITS file to open")
    args = parser.parse_args(argv)

    _configure_qt_platform()
    QtWidgets, ImageModel, Viewer = build_gui()
    app = QtWidgets.QApplication(sys.argv)

    model = ImageModel()
    viewer = Viewer(model)
    if args.image:
        try:
            images = [e for e in list_fits_extensions(args.image)
                      if e["is_image"]]
        except Exception as e:
            images = []
            print(f"vta: {e}", file=sys.stderr)
        if images:
            viewer._path = args.image
            viewer._populate_ext_combo(images)
            viewer._load_ext(images[0]["index"])
            viewer.setWindowTitle(f"VTA  \u2014  {args.image}")
            viewer.file_label.setText(os.path.basename(args.image))
            viewer.file_label.setToolTip(args.image)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
