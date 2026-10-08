"""X-mirror deterministic classification (replaces the noisy correlation-based
detection from the prior, closed investigation session).

For each sampled image, loads its per-pixel geometry parquet
(data/geometry/gaskell_dsk256_110825/<phase>/<image_id>_geometry.parquet,
columns: image_id, pixel_x, pixel_y, iof, incidence, emission, phase, latitude,
longitude), reshapes iof/incidence/emission into (1024,1024) grids via pixel_x/
pixel_y (NOT assumed row-major order), and classifies MIRRORED / CORRECT /
UNDETERMINED by IoU of a physically-motivated mask under three hypotheses:
as-is, x-flip (np.fliplr), y-flip (np.flipud).

Limb frames (body occupies less than half the dense frame, real off-body/NaN
context present): IoU(iof < 0.005, geometry NaN), as specified.

Non-limb frames: IoU of same-percentile, size-matched masks -- darkest decile
of iof vs most-grazing decile of incidence -- NOT the literal
iof<0.2*median/incidence>88deg from the task spec. Checked concretely on both
positive controls: incidence>88deg matched only 7 and 758 of ~1.05M pixels
(0.0007%/0.07%), so IoU against the ~1M-pixel dark-iof mask is near zero by
construction regardless of alignment (union dominated by the large mask) --
the fixed threshold cannot detect a mirror on this data even when one is
present. Percentile-based masks keep the same physical idea (does dark/bright
structure line up with grazing-incidence structure) while staying sensitive to
misalignment. See classify_one()'s inline comment for the full reasoning.

Some products (confirmed: RC in this tree) store only a small on-body
bounding-box crop (~2.6k rows in e.g. x=[448,509], y=[468,522]) rather than
the full dense 1024x1024 raster -- checked directly via row count and
pixel_x/pixel_y span, not assumed. These have no off-body context left to
test the limb mask against and are recorded as their own category
(INSUFFICIENT_CONTEXT_SPARSE_CROP), not forced through either branch.

Secondary diagnostics (not used for the label, recorded for the cross-tab):
  - profile_corr_ls: correlation of the iof row/column profile against
    LS = cos(incidence)/(cos(incidence)+cos(emission)) -- the noisy method
    the prior session used, kept here only so its disagreement with the
    deterministic IoU label is visible, not as the classifier.
  - x_gradient_strength: mean |d(cos incidence)/dx| across the frame, so
    weak-gradient (unreliable-correlation) frames are flagged explicitly
    rather than silently trusted.

Does NOT modify geometry_engine.py or any production code. Does NOT regenerate
any committed product. Output: a per-image CSV only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

GEOM_ROOT = Path("/home/kaushim07/photometry_mcmc_env/data/geometry/gaskell_dsk256_110825")
RAW_ROOT = Path("/home/kaushim07/photometry_mcmc_env/data/calibrated_raw_images")
OUT_DIR = Path("/home/kaushim07/photometry_mcmc_env/outputs/diagnostics")

POSITIVE_CONTROLS = [
    ("survey", "FC21B0006203_11236015531F1B"),
    ("survey", "FC21B0006719_11241181536F1C"),
]

# (phase, n_per_phase) sampling targets per version letter -- chosen from the
# actual on-disk counts (checked via `ls` before writing this), so every entry
# below is drawable without falling back on anything.
SAMPLE_PLAN: dict[str, list[tuple[str, int]]] = {
    "B": [("survey", 5), ("hamo", 5), ("lamo", 5), ("rc", 5)],       # May 2026 baseline
    "C": [("survey", 7), ("hamo", 7), ("lamo", 7)],                  # Jul 2026 all-letter
    "D": [("survey", 7), ("hamo", 7), ("lamo", 7)],
    "E": [("hamo", 10), ("lamo", 11)],
    "F": [("survey", 7), ("hamo", 7), ("lamo", 7)],
    "G": [("hamo", 10), ("lamo", 11)],
    "H": [("lamo", 21)],
    "I": [("lamo", 21)],
}

GRIND_BATCH = {"B": "2026-05_baseline", **{L: "2026-07_allletter" for L in "CDEFGHI"}}


def list_images(phase: str, letter: str) -> list[str]:
    phase_dir = GEOM_ROOT / phase
    return sorted(
        p.stem.replace("_geometry", "")
        for p in phase_dir.glob(f"*F1{letter}_geometry.parquet")
    )


def build_sample() -> list[tuple[str, str]]:
    """Returns list of (phase, image_id), positive controls first."""
    chosen: list[tuple[str, str]] = list(POSITIVE_CONTROLS)
    chosen_ids = {iid for _, iid in chosen}
    rng = np.random.default_rng(1234)

    for letter, plan in SAMPLE_PLAN.items():
        for phase, n in plan:
            candidates = [iid for iid in list_images(phase, letter) if iid not in chosen_ids]
            if not candidates:
                continue
            take = min(n, len(candidates))
            picked = rng.choice(candidates, size=take, replace=False)
            for iid in picked:
                chosen.append((phase, iid))
                chosen_ids.add(iid)
    return chosen


FULL_FRAME = 1024


def load_grids(phase: str, image_id: str) -> tuple[dict[str, np.ndarray], bool] | None:
    """Returns (grids, is_sparse_crop). Some products (confirmed: RC in this tree)
    store only a small on-body bounding box (~2.6k rows spanning e.g. x in
    [448,509], y in [468,522]) rather than the full dense 1024x1024 raster --
    checked directly, not assumed. Reshaping those via (pixel_x.max()+1,
    pixel_y.max()+1) would silently build a mostly-fake small grid dominated by
    artificial NaN padding at the box edges, not a faithful full-frame
    reconstruction with real off-body context -- so that case is flagged and
    handled separately rather than forced through the dense-frame masks below.
    """
    path = GEOM_ROOT / phase / f"{image_id}_geometry.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["pixel_x", "pixel_y", "iof", "incidence", "emission"])

    # Fill fraction against the TRUE full 1024x1024 frame -- not span relative
    # to the sub-window's own bounding box. A body that doesn't reach one edge
    # (e.g. px in [226,1023], fully dense within that range -- confirmed on
    # FC21B0004222_11224172816F1C, 759,820 rows, 0% internal NaN) still
    # reconstructs correctly as a full (1024,1024) grid with genuine NaN
    # padding on the missing side, and fliplr/flipud on THAT full grid still
    # anchors the mirror at the true sensor center (511.5), which is what
    # matters. Checking span-within-bounding-box instead (the earlier,
    # wrong version of this check) would also mis-flag that case AND silently
    # produce an incorrectly-centered flip for any off-center sub-window that
    # slipped through -- fill fraction against the true frame avoids both.
    # RC, by contrast, really is sparse this way: 2,610 rows / 1,048,576 =
    # 0.25%.
    is_sparse_crop = len(df) < 0.05 * FULL_FRAME**2

    if is_sparse_crop:
        # Keep the raw (unpadded) columns for secondary diagnostics only; no grid.
        return {"raw": df}, True

    nx = int(df["pixel_x"].max()) + 1
    ny = int(df["pixel_y"].max()) + 1
    grids = {}
    for col in ("iof", "incidence", "emission"):
        grid = np.full((ny, nx), np.nan, dtype=np.float64)
        grid[df["pixel_y"].to_numpy(), df["pixel_x"].to_numpy()] = df[col].to_numpy()
        grids[col] = grid
    return grids, False


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def classify_one(grids: dict[str, np.ndarray]) -> dict:
    iof = grids["iof"]
    incidence = grids["incidence"]
    emission = grids["emission"]

    finite_frac = np.isfinite(incidence).sum() / incidence.size
    is_limb = finite_frac < 0.5

    results = {}
    for label, (iof_t, inc_t) in [
        ("asis", (iof, incidence)),
        ("xflip", (np.fliplr(iof), incidence)),
        ("yflip", (np.flipud(iof), incidence)),
    ]:
        if is_limb:
            # Dense frame where the body occupies only part of the chip (real
            # off-body/NaN context present, unlike the sparse-crop case) --
            # the task's literal mask applies as-is.
            mask_iof = iof_t < 0.005
            mask_geom = ~np.isfinite(inc_t)
        else:
            # DEVIATION FROM THE LITERAL TASK SPEC, logged here: fixed absolute
            # thresholds (iof<0.2*median, incidence>88 deg) produce wildly
            # size-mismatched masks for low-phase disk-resolved frames --
            # checked concretely on both positive controls, where incidence>88
            # deg matched only 7 and 758 pixels out of ~1.05M (0.0007% and
            # 0.07%). IoU of a ~1M-pixel dark-iof mask against a <0.1%-of-frame
            # incidence mask is near zero by construction regardless of
            # alignment (union is dominated by the large mask), so the fixed
            # 88 deg cutoff cannot detect a mirror on this data even when one
            # is present. Using same-percentile (decile), size-matched masks
            # instead operationalizes the identical physical idea (does the
            # dark/bright structure line up with the grazing-incidence
            # structure) in a way that is actually sensitive to misalignment.
            finite_iof = iof_t[np.isfinite(iof_t)]
            finite_inc = inc_t[np.isfinite(inc_t)]
            if finite_iof.size < 10 or finite_inc.size < 10:
                mask_iof = np.zeros_like(iof_t, dtype=bool)
                mask_geom = np.zeros_like(inc_t, dtype=bool)
            else:
                iof_thresh = np.percentile(finite_iof, 10)
                inc_thresh = np.percentile(finite_inc, 90)
                mask_iof = np.nan_to_num(iof_t, nan=np.inf) <= iof_thresh
                mask_geom = np.nan_to_num(inc_t, nan=-np.inf) >= inc_thresh
        results[label] = iou(mask_iof, mask_geom)

    iou_asis, iou_xflip, iou_yflip = results["asis"], results["xflip"], results["yflip"]
    finite_vals = sorted((v for v in results.values() if np.isfinite(v)), reverse=True)

    # BUG FIXED (was: delta = results[best_label] - iou_asis, which is
    # identically 0 whenever best_label=="asis" -- making CORRECT structurally
    # unreachable and silently forcing every well-oriented image into
    # UNDETERMINED. The margin must be the winner's lead over the RUNNER-UP,
    # not over "asis" specifically -- and the winner must show real signal at
    # all, not just nominally edge out two other near-zero values.
    if len(finite_vals) < 2 or finite_vals[0] < 0.05:
        label = "UNDETERMINED"
        delta = float("nan")
    else:
        best_label = max(results, key=lambda k: (results[k] if np.isfinite(results[k]) else -1.0))
        delta = finite_vals[0] - finite_vals[1]
        if delta < 0.05:
            label = "UNDETERMINED"
        elif best_label == "asis":
            label = "CORRECT"
        elif best_label == "xflip":
            label = "MIRRORED"
        else:
            label = "Y_FLIP_SUSPECT"  # not expected per "Y axis is clean everywhere tested"; flagged, not asserted

    # Secondary diagnostics (not used for the label).
    mu0 = np.cos(np.deg2rad(incidence))
    mu = np.cos(np.deg2rad(emission))
    with np.errstate(invalid="ignore", divide="ignore"):
        ls = mu0 / (mu0 + mu)
    valid = np.isfinite(iof) & np.isfinite(ls)
    if valid.sum() > 10:
        profile_corr_ls = float(np.corrcoef(iof[valid], ls[valid])[0, 1])
    else:
        profile_corr_ls = float("nan")

    d_mu0_dx = np.full_like(mu0, np.nan)
    d_mu0_dx[:, :-1] = np.diff(mu0, axis=1)
    x_gradient_strength = float(np.nanmean(np.abs(d_mu0_dx)))

    return {
        "is_limb": bool(is_limb),
        "finite_frac": float(finite_frac),
        "iou_asis": float(iou_asis),
        "iou_xflip": float(iou_xflip),
        "iou_yflip": float(iou_yflip),
        "delta_iou": float(delta),
        "classification": label,
        "profile_corr_ls": profile_corr_ls,
        "x_gradient_strength": x_gradient_strength,
    }


def reader_path_for(phase: str, image_id: str) -> dict:
    img_path = RAW_ROOT / phase / f"{image_id}.IMG"
    fit_path = RAW_ROOT / phase / f"{image_id}.FIT"
    if img_path.exists():
        return {"source_extension": ".IMG", "reader_path": "PDS3Image.open"}
    if fit_path.exists():
        return {"source_extension": ".FIT", "reader_path": "_DetachedFitsImage"}
    return {"source_extension": "UNKNOWN", "reader_path": "UNKNOWN"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = build_sample()
    print(f"Sample size: {len(sample)} images")

    rows = []
    for phase, image_id in sample:
        letter = image_id[-1]
        loaded = load_grids(phase, image_id)
        if loaded is None:
            print(f"  SKIP (no parquet): {phase}/{image_id}")
            continue
        grids, is_sparse_crop = loaded
        rp = reader_path_for(phase, image_id)
        base = {
            "image_id": image_id,
            "phase": phase,
            "version_letter": letter,
            "grind_batch": GRIND_BATCH.get(letter, "UNKNOWN"),
            "is_positive_control": (phase, image_id) in POSITIVE_CONTROLS,
            **rp,
        }

        if is_sparse_crop:
            # RC-style pre-cropped product: no off-body context survives to
            # test the limb-frame mask against, and the on-body patch is too
            # small/irregular for the percentile masks to be meaningful
            # either. Recorded honestly as its own category, not forced
            # through either branch -- see load_grids()'s docstring.
            raw = grids["raw"]
            row = {
                **base,
                "is_limb": None,
                "finite_frac": None,
                "iou_asis": None,
                "iou_xflip": None,
                "iou_yflip": None,
                "delta_iou": None,
                "classification": "INSUFFICIENT_CONTEXT_SPARSE_CROP",
                "profile_corr_ls": None,
                "x_gradient_strength": None,
                "n_rows": len(raw),
            }
            rows.append(row)
            print(f"  {phase}/{image_id}: INSUFFICIENT_CONTEXT_SPARSE_CROP (n_rows={len(raw)})")
            continue

        try:
            metrics = classify_one(grids)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR classifying {phase}/{image_id}: {exc}")
            continue
        row = {**base, **metrics}
        rows.append(row)
        print(
            f"  {phase}/{image_id}: {row['classification']}  "
            f"(iou_asis={row['iou_asis']:.3f} xflip={row['iou_xflip']:.3f} "
            f"yflip={row['iou_yflip']:.3f} limb={row['is_limb']})"
        )

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "xmirror_classification.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    print("\n=== classification counts ===")
    print(df["classification"].value_counts())

    print("\n=== cross-tab: version_letter x classification ===")
    print(pd.crosstab(df["version_letter"], df["classification"]))

    print("\n=== cross-tab: source_extension x classification ===")
    print(pd.crosstab(df["source_extension"], df["classification"]))

    print("\n=== cross-tab: phase x classification ===")
    print(pd.crosstab(df["phase"], df["classification"]))

    print("\n=== cross-tab: grind_batch x classification ===")
    print(pd.crosstab(df["grind_batch"], df["classification"]))

    print("\n=== positive controls ===")
    print(df[df["is_positive_control"]][["image_id", "classification", "iou_asis", "iou_xflip"]])


if __name__ == "__main__":
    main()
