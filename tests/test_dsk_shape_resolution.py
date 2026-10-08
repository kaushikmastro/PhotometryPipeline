"""Regression test for scripts/diagnostics/dsk_shape_resolution.py against the
committed Vesta DSK (vesta_gaskell_256_110825.bds) -- pins the numbers reported to
the user for the talk slide (Sep 18 2026) so a future SPICE/spiceypy upgrade or a
script refactor can't silently drift the shape-model resolution figures.

Real-kernel, real-compute (iterates all 786,432 plates) -- marked slow, run via
the project's normal sbatch/srun test invocation, not directly on the login node.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "diagnostics" / "dsk_shape_resolution.py"

_spec = importlib.util.spec_from_file_location("dsk_shape_resolution", SCRIPT_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

compute_dsk_resolution = _module.compute_dsk_resolution
DEFAULT_DSK = _module.DEFAULT_DSK

# Values confirmed via `scripts/diagnostics/dsk_shape_resolution.py --dsk
# data/spice_kernels/vesta_gaskell_256_110825.bds` (job under srun, Sep 18 2026),
# cross-checked against the published mean radius (~262 km).
EXPECTED_N_PLATES = 786_432
EXPECTED_N_VERTICES = 396_294
EXPECTED_EDGE_MEAN_KM = 1.686
EXPECTED_MEAN_VERTEX_RADIUS_KM = 262.12


@pytest.mark.slow
def test_vesta_gaskell_256_110825_known_resolution():
    if not Path(DEFAULT_DSK).exists():
        pytest.skip(f"DSK not present on this machine: {DEFAULT_DSK}")

    stats = compute_dsk_resolution(DEFAULT_DSK)

    assert stats["n_plates"] == EXPECTED_N_PLATES
    assert stats["n_vertices"] == EXPECTED_N_VERTICES
    assert stats["edge_mean_km"] == pytest.approx(EXPECTED_EDGE_MEAN_KM, abs=0.001)
    # Mean radius is a physical sanity check, not a bit-exact pin -- allow ~1 km.
    assert stats["mean_vertex_radius_km"] == pytest.approx(
        EXPECTED_MEAN_VERTEX_RADIUS_KM, abs=1.0
    )
