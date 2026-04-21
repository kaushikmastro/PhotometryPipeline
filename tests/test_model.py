import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hapke_mcmc_package.model.equations_2002 import HapkeModel  # noqa: E402


class TestHapkeModel(unittest.TestCase):
    def setUp(self):
        self.model = HapkeModel()

    def test_h_function_rejects_invalid_w(self):
        with self.assertRaises(ValueError):
            self.model.h_function(0.5, 1.2)

    def test_phase_function_rejects_invalid_g(self):
        with self.assertRaises(ValueError):
            self.model.phase_function(np.deg2rad(30), -1.2)

    def test_compute_reflectance_returns_array_shape(self):
        i = np.deg2rad(np.array([10.0, 20.0, 30.0]))
        e = np.deg2rad(np.array([10.0, 20.0, 30.0]))
        alpha = np.deg2rad(np.array([20.0, 30.0, 40.0]))

        result = self.model.compute_reflectance(
            i=i,
            e=e,
            alpha=alpha,
            w=0.5,
            g=0.2,
            theta_bar=np.deg2rad(15.0),
            B0=1.0,
            h=0.1,
        )

        self.assertEqual(result.shape, i.shape)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_compute_reflectance_night_side_returns_zero(self):
        i = np.deg2rad(np.array([95.0]))
        e = np.deg2rad(np.array([10.0]))
        alpha = np.deg2rad(np.array([30.0]))

        result = self.model.compute_reflectance(
            i=i,
            e=e,
            alpha=alpha,
            w=0.5,
            g=0.2,
            theta_bar=np.deg2rad(15.0),
            B0=1.0,
            h=0.1,
        )

        self.assertTrue(np.array_equal(result, np.array([0.0])))

    def test_compute_reflectance_masks_invalid_elements_only(self):
        i = np.deg2rad(np.array([10.0, 95.0]))
        e = np.deg2rad(np.array([10.0, 10.0]))
        alpha = np.deg2rad(np.array([20.0, 30.0]))

        result = self.model.compute_reflectance(
            i=i,
            e=e,
            alpha=alpha,
            w=0.5,
            g=0.2,
            theta_bar=np.deg2rad(15.0),
            B0=1.0,
            h=0.1,
        )

        self.assertEqual(result.shape, i.shape)
        self.assertGreater(result[0], 0.0)
        self.assertEqual(result[1], 0.0)


if __name__ == "__main__":
    unittest.main()
