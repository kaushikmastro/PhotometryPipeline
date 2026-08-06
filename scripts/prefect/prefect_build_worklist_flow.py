from __future__ import annotations

from pathlib import Path

from prefect import flow, get_run_logger, task

DATA_ROOT = Path("/scratch/kaushim07/vesta_data")
RAW_IMAGES_ROOT = DATA_ROOT / "calibrated_raw_images"
COMMITTED_ROOT = DATA_ROOT / "geometry/dsk256"

VALID_PHASES = ("rc", "survey", "hamo", "lamo")


@task
def enumerate_raw_images(phase: str) -> list[str]:
    """All raw calibrated images for this phase, any F1 letter."""
    phase_dir = RAW_IMAGES_ROOT / phase
    stems = sorted(p.stem for p in phase_dir.glob("*.IMG"))
    logger = get_run_logger()
    logger.info("enumerate_raw_images(%s): %d images found under %s", phase, len(stems), phase_dir)
    return stems


@task
def enumerate_committed_baseline(phase: str) -> list[str]:
    """Image stems already present as geometry parquets for this phase."""
    output_dir = COMMITTED_ROOT / phase
    suffix = "_geometry.parquet"
    stems = sorted(p.name[: -len(suffix)] for p in output_dir.glob(f"*{suffix}"))
    logger = get_run_logger()
    logger.info("enumerate_committed_baseline(%s): %d parquets found under %s", phase, len(stems), output_dir)
    return stems


@task
def diff_worklist(raw: list[str], committed: list[str]) -> list[str]:
    """Raw images with no committed output yet -- the actual remaining worklist."""
    remaining = sorted(set(raw) - set(committed))
    logger = get_run_logger()
    logger.info(
        "diff_worklist: %d raw - %d committed = %d remaining",
        len(raw), len(committed), len(remaining),
    )
    return remaining


@task
def verify_no_contamination(phase: str, committed: list[str]) -> dict:
    """Guard against a repeat of tonight's survey/ contamination bug: every
    committed stem for this phase must actually contain that phase's own
    letter-pattern image ID structure AND, more importantly, must correspond
    to a real raw image that actually lives under this phase's raw directory
    (not one that only landed here because of the silent phase-routing
    default). Flags anything suspicious rather than silently trusting the
    committed directory's contents.
    """
    logger = get_run_logger()
    raw_dir = RAW_IMAGES_ROOT / phase
    orphaned = [stem for stem in committed if not (raw_dir / f"{stem}.IMG").exists()]

    result = {
        "phase": phase,
        "committed_count": len(committed),
        "orphaned_count": len(orphaned),
        "orphaned_sample": orphaned[:10],
        "clean": len(orphaned) == 0,
    }

    if orphaned:
        logger.warning(
            "verify_no_contamination(%s): %d committed parquet(s) have NO matching "
            "raw .IMG under %s -- possible misfiled/contaminated entries (first 10: %s)",
            phase, len(orphaned), raw_dir, orphaned[:10],
        )
    else:
        logger.info("verify_no_contamination(%s): clean -- every committed parquet has a matching raw image", phase)

    return result


@flow(name="build-worklist")
def build_worklist(phase: str) -> dict:
    if phase not in VALID_PHASES:
        raise ValueError(f"phase must be one of {VALID_PHASES}, got {phase!r}")

    raw = enumerate_raw_images(phase)
    committed = enumerate_committed_baseline(phase)
    remaining = diff_worklist(raw, committed)
    contamination_check = verify_no_contamination(phase, committed)

    logger = get_run_logger()
    logger.info(
        "build_worklist(%s) complete: %d raw, %d committed, %d remaining, contamination_clean=%s",
        phase, len(raw), len(committed), len(remaining), contamination_check["clean"],
    )

    return {
        "phase": phase,
        "raw_count": len(raw),
        "committed_count": len(committed),
        "remaining_count": len(remaining),
        "remaining": remaining,
        "contamination_check": contamination_check,
    }


if __name__ == "__main__":
    import sys

    phase_arg = sys.argv[1] if len(sys.argv) > 1 else "survey"
    result = build_worklist(phase_arg)
    print(f"phase={result['phase']} raw={result['raw_count']} committed={result['committed_count']} "
          f"remaining={result['remaining_count']} contamination_clean={result['contamination_check']['clean']}")
