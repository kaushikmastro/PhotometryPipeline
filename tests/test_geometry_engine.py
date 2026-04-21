import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hapke_mcmc_package.etl.geometry_engine import calibrate_iof_data  # noqa: E402


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (np.array([[0.0, 50.0, 100.0]], dtype=np.float32), np.array([[0.0, 0.5, 1.0]], dtype=np.float32)),
        (np.array([[25.0, 75.0]], dtype=np.float32), np.array([[0.25, 0.75]], dtype=np.float32)),
    ],
)
def test_calibrate_iof_divides_by_100(raw: np.ndarray, expected: np.ndarray) -> None:
    result = calibrate_iof_data(raw, image_id="UNIT_TEST")
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize(
    "raw",
    [
        np.array([[-1.1, 0.0, 100.0]], dtype=np.float32),  # min after /100 is -0.011
        np.array([[0.0, 106.0]], dtype=np.float32),  # max after /100 is 1.06
    ],
)
def test_calibrate_iof_guardrail_rejects_outside_tolerance(raw: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match="I/F guardrail violated"):
        calibrate_iof_data(raw, image_id="UNIT_TEST")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            np.array([[-0.5, 0.0, 50.0, 104.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.5, 1.0]], dtype=np.float32),
        ),
        (
            np.array([[100.0, 102.0]], dtype=np.float32),
            np.array([[1.0, 1.0]], dtype=np.float32),
        ),
    ],
)
def test_calibrate_iof_clip_within_tolerance(raw: np.ndarray, expected: np.ndarray) -> None:
    # -0.5/100=-0.005 and 104/100=1.04 are within tolerance, so they should be clipped.
    result = calibrate_iof_data(raw, image_id="UNIT_TEST")
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-7)
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0
