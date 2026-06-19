"""Unit tests for the Qt-free numerical core of VTA.

These exercise the pure-logic functions only (stretches, sky/scaling,
centroiding, photometry, statistics, profiles, cuts, tracing/extraction,
formatting) and import ``vta`` without instantiating any GUI -- so they run
headless and do not require PySide6 / pyqtgraph.

Run with:  pytest test_vta.py
"""

import numpy as np
import pytest

import vta


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

SKY = 100.0
PEAK = 1000.0
FWHM = 4.0
CX, CY = 40.0, 40.0


def gaussian_star(ny=81, nx=81, cx=CX, cy=CY, peak=PEAK, fwhm=FWHM, sky=SKY):
    """A clean 2-D Gaussian point source on a flat sky (no noise)."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    sigma = fwhm / 2.35482
    return peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                         / (2.0 * sigma ** 2)) + sky


@pytest.fixture
def star():
    return gaussian_star()


# --------------------------------------------------------------------------
# formatting / small helpers
# --------------------------------------------------------------------------

def test_format_counts_fixed_then_scientific():
    assert vta.format_counts(1234.56) == "1234.6"
    assert "e" in vta.format_counts(1.2e7)
    assert vta.format_counts(float("nan")) == "nan"


def test_angular_factor():
    # 0.5"/pix -> arcsec factor is the pixel scale itself
    assert vta.angular_factor(0.5, "arcsec") == pytest.approx(0.5)
    # arcmin/degree divide by 60 / 3600
    assert vta.angular_factor(60.0, "arcmin") == pytest.approx(1.0)
    assert vta.angular_factor(3600.0, "degree") == pytest.approx(1.0)
    # no pixel scale -> None
    assert vta.angular_factor(None, "arcsec") is None
    assert vta.angular_factor(0.0, "arcsec") is None


def test_arcsec_per_unit_table():
    assert vta.ARCSEC_PER_UNIT == {"arcsec": 1.0, "arcmin": 60.0,
                                   "degree": 3600.0}


# --------------------------------------------------------------------------
# sky / scaling
# --------------------------------------------------------------------------

def test_robust_sky_recovers_level():
    rng = np.random.default_rng(0)
    img = SKY + rng.normal(0.0, 3.0, size=(128, 128))
    sky, sig = vta.robust_sky(img)
    assert sky == pytest.approx(SKY, abs=0.5)
    assert sig == pytest.approx(3.0, rel=0.2)


def test_zscale_limits_ordered(star):
    z1, z2 = vta.zscale_limits(star)
    assert z1 < z2


def test_transform_image_linear_passthrough():
    img = np.linspace(0.0, 1000.0, 64 * 64).reshape(64, 64)
    disp, lo, hi = vta.transform_image(img, "linear", 0.0, 1000.0, 2.0)
    assert disp.shape == img.shape
    assert (lo, hi) == (0.0, 1000.0)
    np.testing.assert_allclose(disp, img)


def test_transform_image_nan_to_floor():
    img = np.array([[0.0, np.nan], [500.0, 1000.0]])
    disp, lo, hi = vta.transform_image(img, "linear", 0.0, 1000.0, 2.0)
    assert disp[0, 1] == pytest.approx(lo)        # NaN pushed to floor


# --------------------------------------------------------------------------
# centroiding / statistics
# --------------------------------------------------------------------------

def test_centroid_com_recovers_center(star):
    xc, yc, warn = vta.centroid_com(star, 41, 39)
    assert xc == pytest.approx(CX, abs=0.1)
    assert yc == pytest.approx(CY, abs=0.1)


def test_box_statistics(star):
    s = vta.box_statistics(star, int(CX), int(CY), boxsize=11)
    assert s["npix"] == 121
    assert s["max"] == pytest.approx(PEAK + SKY, rel=0.02)
    assert s["min"] >= SKY - 1.0
    assert s["median"] <= s["max"]


# --------------------------------------------------------------------------
# radial profile / FWHM
# --------------------------------------------------------------------------

def test_radial_profile_fwhm(star):
    rp = vta.radial_profile(star, int(CX), int(CY))
    assert rp["fwhm"] == pytest.approx(FWHM, rel=0.1)
    assert rp["sky"] == pytest.approx(SKY, abs=2.0)


def test_spline_fwhm_matches_profile(star):
    rp = vta.radial_profile(star, int(CX), int(CY))
    fwhm, warn = vta.spline_fwhm(rp["r_prof"], rp["prof"])
    assert fwhm == pytest.approx(rp["fwhm"], rel=1e-6)


# --------------------------------------------------------------------------
# aperture photometry
# --------------------------------------------------------------------------

def test_aperture_photometry(star):
    ph = vta.aperture_photometry_atv(star, int(CX), int(CY),
                                     aprad=8.0, innersky=12.0, outersky=20.0)
    assert ph["sky"] == pytest.approx(SKY, abs=2.0)
    assert ph["net"] > 0.0
    # net flux should be within a few percent of the analytic integral
    sigma = FWHM / 2.35482
    analytic = PEAK * 2.0 * np.pi * sigma ** 2
    assert ph["net"] == pytest.approx(analytic, rel=0.1)


def test_aperture_photometry_magnitudes(star):
    ph = vta.aperture_photometry_atv(star, int(CX), int(CY), aprad=8.0,
                                     magunits=True, zeropoint=25.0)
    assert np.isfinite(ph["mag"])
    assert ph["mag"] < 25.0           # source brighter than the zero point


# --------------------------------------------------------------------------
# cuts
# --------------------------------------------------------------------------

def test_vector_cut_peaks_at_center(star):
    dist, vals = vta.vector_cut(star, 10, int(CY), 70, int(CY))
    assert dist.shape == vals.shape
    # the horizontal cut through the star peaks at its center column
    assert dist[int(np.argmax(vals))] == pytest.approx(CX - 10, abs=1.0)


def test_row_and_col_values(star):
    cols, rvals = vta.row_values(star, int(CY))
    rows, cvals = vta.col_values(star, int(CX))
    assert cols.shape == rvals.shape == (star.shape[1],)
    assert rows.shape == cvals.shape == (star.shape[0],)
    assert int(np.argmax(rvals)) == int(CX)
    assert int(np.argmax(cvals)) == int(CY)


# --------------------------------------------------------------------------
# spectral tracing + extraction
# --------------------------------------------------------------------------

def make_spectrum(ny=80, nx=200, trace_row=40.0, profile_fwhm=3.0):
    """A horizontal spectrum: a Gaussian cross-dispersion profile centered
    on trace_row, modulated by a smooth flux envelope along x. The frame is
    tall enough (trace_row >= 25 from each edge) for extract_spectrum's
    default background strips (trace +/- 25) to stay on the image."""
    sigma = profile_fwhm / 2.35482
    prof = np.exp(-((np.arange(ny) - trace_row) ** 2) / (2.0 * sigma ** 2))
    flux = 300.0 * np.exp(-((np.arange(nx) - nx / 2) ** 2)
                          / (2.0 * 40.0 ** 2)) + 50.0
    return np.outer(prof, flux) + 5.0


def test_trace_spectrum_follows_trace():
    sp = make_spectrum()
    centers, points, xspec, ftrace = vta.trace_spectrum(
        sp, 100, 40, tracestep=21, traceheight=7, traceorder=3)
    assert xspec.shape == ftrace.shape == (200,)
    # the fitted trace should sit on the trace row across the whole range
    assert np.median(ftrace) == pytest.approx(40.0, abs=0.3)


def test_extract_spectrum_shape_and_envelope():
    sp = make_spectrum()
    _, _, xspec, ftrace = vta.trace_spectrum(sp, 100, 40)
    spec = vta.extract_spectrum(sp, xspec, ftrace, lower=-5, upper=5,
                                backsub=True)
    assert spec.shape == (200,)
    # background-subtracted flux peaks near the envelope center
    assert int(np.argmax(spec)) == pytest.approx(100, abs=10)
    assert np.all(np.isfinite(spec))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
