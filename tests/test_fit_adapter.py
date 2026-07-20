from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from hapke_mcmc_package.etl.geometry_engine import _DetachedFitsImage, _find_label_value

STEM = "FC21B0000000_00000000000F1B"


def _write_synthetic_fit_pair(tmp_path: Path, stem: str = STEM) -> Path:
    """Write a minimal synthetic .FIT + detached .LBL pair mirroring the real
    Dawn FC2 QuickFITS-converted layout: single PrimaryHDU at index 0, float32.
    """
    fit_path = tmp_path / f"{stem}.FIT"
    lbl_path = tmp_path / f"{stem}.LBL"

    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    fits.PrimaryHDU(data=data).writeto(fit_path, overwrite=True)

    lbl_path.write_text(
        "PDS_VERSION_ID                = PDS3\n"
        f'FILE_NAME                     = "{stem}.FIT"\n'
        f'^HEADER                       = ("{stem}.FIT", 1)\n'
        f'^IMAGE                        = ("{stem}.FIT", 2)\n'
        'SPACECRAFT_CLOCK_START_COUNT  = "123456789:012"\n'
        "EXPOSURE_DURATION             = 9.000 <millisecond>\n"
        "OBJECT                        = IMAGE\n"
        "    LINE_SAMPLES              = 4\n"
        "    LINES                     = 4\n"
        "END_OBJECT                    = IMAGE\n"
        "END\n"
    )
    return fit_path


def test_detached_fits_image_exposes_correct_image_array(tmp_path: Path) -> None:
    fit_path = _write_synthetic_fit_pair(tmp_path)

    adapter = _DetachedFitsImage(fit_path)

    assert adapter.image.shape == (4, 4)
    np.testing.assert_array_equal(
        np.asarray(adapter.image, dtype=np.float32),
        np.arange(16, dtype=np.float32).reshape(4, 4),
    )


def test_detached_fits_image_label_supports_find_label_value(tmp_path: Path) -> None:
    """The exact interface compute_geometry() relies on: pds_img.label must be
    dict-like enough for the recursive _find_label_value() lookups it performs
    for SCLK and exposure duration.
    """
    fit_path = _write_synthetic_fit_pair(tmp_path)

    adapter = _DetachedFitsImage(fit_path)

    sclk = _find_label_value(adapter.label, "SPACECRAFT_CLOCK_START_COUNT")
    exposure = _find_label_value(adapter.label, "EXPOSURE_DURATION")

    assert sclk == "123456789:012"
    assert float(exposure) == pytest.approx(9.0)


def test_detached_fits_image_missing_label_raises(tmp_path: Path) -> None:
    fit_path = tmp_path / f"{STEM}.FIT"
    fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32)).writeto(fit_path, overwrite=True)
    # deliberately no .LBL written alongside it

    with pytest.raises(FileNotFoundError):
        _DetachedFitsImage(fit_path)


def test_detached_fits_image_lowercase_suffix_dispatch(tmp_path: Path) -> None:
    """compute_geometry() dispatches on image_path.suffix.lower() == '.fit',
    so the adapter itself must work regardless of the .LBL lookup being
    case-sensitive on the filesystem (real PDS releases use uppercase .LBL).
    """
    fit_path = tmp_path / f"{STEM}.fit"
    lbl_path = tmp_path / f"{STEM}.LBL"

    data = np.ones((4, 4), dtype=np.float32)
    fits.PrimaryHDU(data=data).writeto(fit_path, overwrite=True)
    lbl_path.write_text(
        'SPACECRAFT_CLOCK_START_COUNT  = "1:1"\n'
        "EXPOSURE_DURATION             = 1.0 <millisecond>\n"
        "END\n"
    )

    adapter = _DetachedFitsImage(fit_path)
    assert adapter.image.shape == (4, 4)
