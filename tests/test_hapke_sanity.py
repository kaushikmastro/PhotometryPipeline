from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.core.types import GeometryBatch  # noqa: E402
from photometry.models.hapke import HapkeModel  # noqa: E402

# Single-point sanity check, converted from the former repo-root run_hapke.py:
# predicted I/F at i=43°, e=21°, phase=29° against a measured value of 0.088.
MEASURED_IOF = 0.088
TOLERANCE = 0.20
INC_DEG, EMI_DEG, PHASE_DEG = 43.0, 21.0, 29.0


@pytest.mark.parametrize(
    ("label", "params", "enable_shoe", "enable_roughness"),
    [
        ("A", {"w": 0.2994, "g": -0.3879}, False, False),
        pytest.param(
            "B",
            {"w": 0.38, "g": -0.50, "theta_bar": 17.7, "B0": 1.7, "h": 0.07},
            True,
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason="SHOE+roughness case predicts I/F=0.1859, 111% above 0.088; "
                "the original run_hapke.py also reported it out of tolerance. "
                "Unresolved: wrong parameter set for this point, or a model issue.",
            ),
        ),
    ],
)
def test_hapke_iof_within_20pct_of_measured(
    label: str, params: dict[str, float], enable_shoe: bool, enable_roughness: bool
) -> None:
    geometry = GeometryBatch(
        incidence=np.array([np.radians(INC_DEG)]),
        emission=np.array([np.radians(EMI_DEG)]),
        phase=np.array([np.radians(PHASE_DEG)]),
    )
    model = HapkeModel(enable_shoe=enable_shoe, enable_roughness=enable_roughness)
    model.parameters = dict(params)

    iof = float(model.reflectance(geometry)[0])

    rel_diff = abs(iof - MEASURED_IOF) / MEASURED_IOF
    assert rel_diff <= TOLERANCE, f"Case {label}: I/F={iof:.4f}, {rel_diff:.1%} from {MEASURED_IOF}"
