from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hapke_mcmc_package.etl.geometry_engine import GeometryEngine

ROUTE = GeometryEngine._phase_subdir_from_image_path


@pytest.mark.parametrize(
    "path,expected",
    [
        (Path("/data/calibrated_raw_images/hamo/FC21HAMO_0001.IMG"), "hamo"),
        (Path("/data/calibrated_raw_images/survey/FC21SURVEY_0001.IMG"), "survey"),
        (Path("/data/calibrated_raw_images/lamo/FC21LAMO_0001.IMG"), "lamo"),
        (Path("/data/calibrated_raw_images/rc/FC21RC_0001.IMG"), "rc"),
        # case-insensitivity
        (Path("/data/calibrated_raw_images/HAMO/FC21HAMO_0002.IMG"), "hamo"),
    ],
)
def test_phase_subdir_routes_correctly_when_phase_in_path(path: Path, expected: str) -> None:
    assert ROUTE(path) == expected


def test_phase_subdir_defaults_to_survey_when_no_phase_in_path() -> None:
    """Documents current (unfixed) behavior: a path with none of
    rc/survey/hamo/lamo in it silently routes to 'survey'. This is the exact
    mechanism that caused tonight's real contamination — user_roi_images/
    doesn't contain any of those four names, so its outputs landed in the
    committed survey/ baseline.
    """
    path = Path("/data/calibrated_raw_images/user_roi_images/FC21B0009981_11284200337F1G.IMG")

    assert ROUTE(path) == "survey"


def test_phase_subdir_function_itself_emits_no_warning_on_silent_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """KNOWN DESIGN GAP, not fixed tonight (deferred in favor of the required
    --output-subdir workaround at the run_geometry.py CLI level):
    _phase_subdir_from_image_path() itself does not log anything when it
    falls through to the 'survey' default. The warning that exists today
    ("File path context lacks structural phase designations...") lives in
    discover_worklist()'s caller loop, NOT in this function. Any other call
    site that invokes _phase_subdir_from_image_path() directly -- e.g.
    GeometryEngine.compute_geometry() itself, which is exactly how tonight's
    contaminating runs were invoked via srun, bypassing discover_worklist()
    entirely -- gets zero warning at all.

    This test asserts the CURRENT (gap) behavior so it fails loudly -- as a
    signal to fix it, not to hide it -- the moment someone adds logging
    directly inside _phase_subdir_from_image_path() and forgets to update
    this test to match the improved behavior.
    """
    with caplog.at_level(logging.WARNING):
        result = ROUTE(Path("/data/calibrated_raw_images/user_roi_images/FC21B0009981.IMG"))

    assert result == "survey"
    assert caplog.records == [], (
        "_phase_subdir_from_image_path() logged a warning on silent default -- "
        "if this function has since been fixed to warn on its own, update this "
        "test to assert the warning IS present, and remove this KNOWN GAP note."
    )
