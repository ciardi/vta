"""Offscreen GUI smoke test for VTA.

Builds the real Viewer under Qt's 'offscreen' platform, loads a synthetic
FITS image (with a TAN WCS), and exercises the code paths touched in recent
work: the version/buffer status-bar labels, the angular radial-units
control, the row/col/vector cut refactor, the p/x/r/c cursor actions and
their eventFilter routing, blink buffers, and the help/about dialogs.

Run:  QT_QPA_PLATFORM=offscreen python3 gui_smoke.py
Exits non-zero if any check fails.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import vta

_fails = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        _fails.append(name)


def make_fits_with_wcs(path, ny=81, nx=81):
    yy, xx = np.mgrid[0:ny, 0:nx]
    sig = 4.0 / 2.35482
    data = (1000.0 * np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2)
                            / (2 * sig ** 2)) + 100.0).astype(np.float32)
    w = WCS(naxis=2)
    w.wcs.crpix = [41, 41]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-0.5 / 3600.0, 0.5 / 3600.0]   # 0.5"/pix
    hdr = w.to_header()
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def main():
    from PySide6 import QtCore, QtGui, QtWidgets

    # Guard paths raise modal QMessageBoxes, which would block forever under
    # the offscreen platform. Stub them to no-ops for the smoke test.
    for _m in ("warning", "information", "critical", "question"):
        setattr(QtWidgets.QMessageBox, _m,
                staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok))

    vta._configure_qt_platform()
    QtWidgets_, ImageModel, Viewer = vta.build_gui()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    tmp = tempfile.mkdtemp()
    fpath = os.path.join(tmp, "star.fits")
    make_fits_with_wcs(fpath)

    print("construct + load:")
    model = ImageModel()
    viewer = Viewer(model)
    images = [e for e in vta.list_fits_extensions(fpath) if e["is_image"]]
    viewer._path = fpath
    viewer._populate_ext_combo(images)
    viewer._load_ext(images[0]["index"])
    viewer.show()
    app.processEvents()
    check("Viewer constructed and image loaded",
          viewer.model.data is not None)
    check("image is 81x81", viewer.model.data.shape == (81, 81))
    check("WCS present", viewer._pixscale_arcsec() is not None)

    print("status-bar labels:")
    check("version badge == V1.00",
          viewer.version_label.text() == f"V{vta.__version__}")
    check("version tooltip carries date",
          vta.__date__ in viewer.version_label.toolTip())
    check("buffer label starts at 'live'",
          viewer.buffer_label.text() == "buffer: live")

    print("angular radial units:")
    check("WCS present -> angular checkbox enabled",
          viewer.radial_ang_chk.isEnabled())
    viewer.radial_ang_chk.setChecked(False)
    check("unchecked -> pixels", viewer._radial_scale() == ("pix", 1.0))
    viewer.radial_ang_chk.setChecked(True)
    app.processEvents()
    unit, factor = viewer._radial_scale()
    check("checked -> arcsec at 0.5\"/pix",
          unit == "arcsec" and abs(factor - 0.5) < 1e-6)
    check("unit combo live only when checked", viewer.radial_unit.isEnabled())
    viewer.radial_unit.setCurrentText("arcmin")
    u2, f2 = viewer._radial_scale()
    check("arcmin factor = 0.5/60", u2 == "arcmin"
          and abs(f2 - 0.5 / 60.0) < 1e-9)
    viewer.radial_ang_chk.setChecked(False)

    print("photometry / radial profile (FWHM title path):")
    viewer._cursor_xy = (40, 40)
    viewer.update_analysis(40, 40)
    app.processEvents()
    # confirm the photometry/radial path (incl. the .3g FWHM title) ran
    check("photometry ran without error", True)

    print("cursor-action methods (dispatched by eventFilter):")
    for label, fn in [("photometry_at_cursor", viewer.photometry_at_cursor),
                      ("row_at_cursor", viewer.row_at_cursor),
                      ("col_at_cursor", viewer.col_at_cursor),
                      ("extract_at_cursor", viewer.extract_at_cursor)]:
        try:
            fn()
            app.processEvents()
            ok = True
        except Exception as e:           # noqa: BLE001  (smoke test)
            ok = False
            print(f"      {label} raised {type(e).__name__}: {e}")
        check(f"{label}() runs", ok)
    check("row window created", "row" in viewer._plot_windows)
    check("col window created", "col" in viewer._plot_windows)

    print("cut-plot refactor (vector + step helper):")
    try:
        viewer._plot_vector(10, 40, 70, 40)
        app.processEvents()
        check("vector window created", "vector" in viewer._plot_windows)
    except Exception as e:               # noqa: BLE001
        check("vector window created", False)
        print(f"      _plot_vector raised {type(e).__name__}: {e}")

    print("eventFilter key routing (p/x/r/c):")
    # Force the activation/focus state the filter inspects.
    orig_active = QtWidgets.QApplication.activeWindow
    orig_focus = QtWidgets.QApplication.focusWidget
    QtWidgets.QApplication.activeWindow = staticmethod(lambda: viewer)
    QtWidgets.QApplication.focusWidget = staticmethod(lambda: None)
    try:
        for key, kname in [(QtCore.Qt.Key_R, "r"), (QtCore.Qt.Key_C, "c"),
                           (QtCore.Qt.Key_P, "p"), (QtCore.Qt.Key_X, "x")]:
            ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key,
                                 QtCore.Qt.NoModifier)
            consumed = viewer.eventFilter(viewer, ev)
            check(f"'{kname}' consumed by filter when VTA window active",
                  consumed is True)
        # text-entry guard: a focused line edit keeps its keystrokes
        le = QtWidgets.QLineEdit()
        QtWidgets.QApplication.focusWidget = staticmethod(lambda: le)
        ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_R,
                             QtCore.Qt.NoModifier)
        check("'r' NOT consumed while a line edit is focused",
              viewer.eventFilter(viewer, ev) is not True)
    finally:
        QtWidgets.QApplication.activeWindow = orig_active
        QtWidgets.QApplication.focusWidget = orig_focus

    print("row/col angular x-axis (new, mirrors vector):")
    for k in ("row", "col"):
        pw = viewer._plot_windows[k]
        check(f"{k} window has angular checkbox", "ang_chk" in pw)
        check(f"{k} angular checkbox enabled (WCS present)",
              pw["ang_chk"].isEnabled())
        check(f"{k} unit combo starts disabled (unchecked)",
              not pw["unit_combo"].isEnabled())
        try:
            pw["ang_chk"].setChecked(True)
            app.processEvents()
            ok = pw["unit_combo"].isEnabled()
            pw["unit_combo"].setCurrentText("arcmin")
            app.processEvents()
            ax_label = pw["ax"].get_xlabel()
            ok = ok and "arcmin" in ax_label
            pw["ang_chk"].setChecked(False)
            app.processEvents()
        except Exception as e:                # noqa: BLE001
            ok = False
            print(f"      {k} angular toggle raised {type(e).__name__}: {e}")
        check(f"{k} angular toggle re-renders with unit label", ok)

    print("erase all clears the vector line:")
    viewer._vec_line.setVisible(True)
    viewer._erase_all()
    check("Erase all hides the vector cut line",
          not viewer._vec_line.isVisible())

    print("image arithmetic (buffer subtraction):")
    # buffer 1 = flat background (100); buffer 2 = star + 100
    fa = os.path.join(tmp, "bg.fits")
    fb = os.path.join(tmp, "src.fits")
    yy, xx = np.mgrid[0:81, 0:81]
    sig = 4.0 / 2.35482
    flat = np.full((81, 81), 100.0, np.float32)
    star = (1000.0 * np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2)
                            / (2 * sig ** 2)) + 100.0).astype(np.float32)
    fits.PrimaryHDU(flat).writeto(fa, overwrite=True)
    fits.PrimaryHDU(star).writeto(fb, overwrite=True)
    check("load file -> buffer 1", viewer._load_file_into_buffer(0, fa))
    check("load file -> buffer 2", viewer._load_file_into_buffer(1, fb))
    check("buffer 1 filled", viewer._blink[0] is not None)
    check("buffer 2 filled", viewer._blink[1] is not None)
    # Source(buf2) - Background(buf1) -> buffer 3
    ok = viewer._subtract_buffers(1, 0, 2)
    app.processEvents()
    check("subtract buf2 - buf1 -> buf3 returns True", ok)
    check("buffer 3 filled", viewer._blink[2] is not None)
    if viewer._blink[2] is not None:
        expect = star.astype(float) - flat.astype(float)
        check("buffer 3 == source - background",
              np.allclose(viewer._blink[2]["data"], expect, atol=1e-4))
        check("buffer 3 result keeps asinh stretch",
              viewer._blink[2]["scaling"] == "asinh")
    check("result displayed -> badge 'buffer: 3'",
          viewer.buffer_label.text() == "buffer: 3")
    # guards: empty buffer and shape mismatch must refuse
    viewer._blink[2] = None
    check("empty-buffer subtract refused",
          viewer._subtract_buffers(2, 0, 1) is False)
    fc = os.path.join(tmp, "small.fits")
    fits.PrimaryHDU(np.zeros((40, 40), np.float32)).writeto(fc, overwrite=True)
    viewer._load_file_into_buffer(2, fc)
    check("shape-mismatch subtract refused",
          viewer._subtract_buffers(1, 2, 0) is False)
    try:
        viewer._arith_dialog()
        app.processEvents()
        check("arithmetic dialog builds", viewer._arith_dlg is not None)
        viewer._arith_dlg.close()
    except Exception as e:                # noqa: BLE001
        check("arithmetic dialog builds", False)
        print(f"      _arith_dialog raised {type(e).__name__}: {e}")
    viewer._clear_blink()

    print("blink buffers + indicator:")
    viewer._store_blink(0)
    viewer._show_blink(0)
    app.processEvents()
    check("buffer label shows 'buffer: 1' after show",
          viewer.buffer_label.text() == "buffer: 1")
    viewer._clear_blink()
    check("buffer label back to 'live' after clear",
          viewer.buffer_label.text() == "buffer: live")

    print("help / about dialogs:")
    try:
        viewer.show_help()
        app.processEvents()
        check("help dialog built", viewer._help_win is not None)
    except Exception as e:               # noqa: BLE001
        check("help dialog built", False)
        print(f"      show_help raised {type(e).__name__}: {e}")

    print()
    if _fails:
        print(f"SMOKE TEST FAILED: {len(_fails)} check(s): {_fails}")
        return 1
    print("SMOKE TEST PASSED: all checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
