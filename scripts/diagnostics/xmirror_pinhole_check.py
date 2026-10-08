"""Independent sub-spacecraft pixel prediction via direct pinhole projection --
spkpos + pxform + IK IFOV/CCD_CENTER -- deliberately bypassing
compute_pixel_rays() (the FOV-corner bilinear-interpolation grid that
geometry_engine.py actually uses) so this is a genuinely separate code path,
not a restatement of the same assumption.

Method, for each image:
  1. ET from the PDS label (SPACECRAFT_CLOCK_START_COUNT + 0.5*EXPOSURE_DURATION,
     same formula as GeometryEngine._extract_observation_et -- not the pipeline
     code itself, re-derived from the label by hand here).
  2. Direction from spacecraft to Vesta's center in J2000:
     spiceypy.spkpos("VESTA", et, "J2000", "LT+S", "DAWN").
  3. Rotate into the camera frame via spiceypy.pxform("J2000", "DAWN_FC2", et)
     -- explicit rotation step, not asking spkpos for DAWN_FC2 directly, so
     the frame transform is a visible, separate step as specified.
  4. Off-boresight angles angle_x = atan2(vx, vz), angle_y = atan2(vy, vz) in
     the camera frame (BORESIGHT is along +Z for DAWN_FC2 -- INS-203120_BORESIGHT
     = (0,0,150.07), confirmed from the IK).
  5. Pixel prediction from the IK's own IFOV/CCD_CENTER for NAIF ID -203120
     ('DAWN_FC2', the base/clear ID -- confirmed this, not -203121
     'DAWN_FC2_FILTER_1', is what GeometryEngine.cam_id actually resolves to
     via spiceypy.bodn2c("DAWN_FC2") == -203120, per dawn_v11.tf's
     NAIF_BODY_NAME/NAIF_BODY_CODE pairing):
       predicted_sample = CCD_CENTER[0] + angle_x / IFOV[0]
       predicted_line   = CCD_CENTER[1] + angle_y / IFOV[1]
     This is ONE fixed sign convention (+camera-frame-X -> +sample). The point
     of running it against both MIRRORED- and CORRECT-labeled images under
     the SAME fixed convention is to see whether it agrees with the pipeline's
     actual minimum-emission pixel for CORRECT images and disagrees (in x, by
     ~(1023 - 2*predicted_x) i.e. consistent with a mirror) for MIRRORED ones.

  6. Compare against "the pipeline's minimum-emission pixel": the (pixel_x,
     pixel_y) of the minimum-emission row in that image's own geometry
     parquet -- the sub-spacecraft point is, to good approximation for a
     body this close to spherical, where emission angle is smallest.

Does NOT modify geometry_engine.py or any production code. Output: a CSV only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvl
import spiceypy

ROOT = Path("/home/kaushim07/photometry_mcmc_env")
GEOM_ROOT = ROOT / "data" / "geometry" / "gaskell_dsk256_110825"
RAW_ROOT = ROOT / "data" / "calibrated_raw_images"
OUT_DIR = ROOT / "outputs" / "diagnostics"
METAKERNEL = ROOT / "data" / "spice_kernels" / "dawn_dynamic.tm"

CAM_ID = -203120  # DAWN_FC2 base/clear -- confirmed via bodn2c("DAWN_FC2") in dawn_v11.tf
IFOV = (0.000093242, 0.000093184)  # rad/pixel, (sample, line) -- INS-203120_IFOV, dawn_fc_v02.ti
CCD_CENTER = (511.5, 511.5)  # INS-203120_CCD_CENTER

# (phase, image_id, classification) -- from the corrected xmirror_classify.py run
# (job 26594188), chosen as the clearest-signal examples of each label.
CASES = [
    ("survey", "FC21B0006677_11241031036F1B", "MIRRORED"),
    ("survey", "FC21B0006653_11241004936F1B", "MIRRORED"),
    ("lamo", "FC21B0021912_12062152225F1F", "MIRRORED"),
    ("lamo", "FC21B0017892_12032171642F1E", "CORRECT"),
    ("lamo", "FC21B0017764_12031102151F1E", "CORRECT"),
    ("survey", "FC21B0004667_11227021924F1D", "CORRECT"),
]


def extract_et(label_path: Path) -> float:
    label = pvl.load(str(label_path))
    sclk_start = str(label["SPACECRAFT_CLOCK_START_COUNT"])
    exposure_ms = label["EXPOSURE_DURATION"]
    exposure_seconds = float(exposure_ms.value) / 1000.0 if hasattr(exposure_ms, "value") else float(exposure_ms) / 1000.0
    start_et = float(spiceypy.scs2e(-203, sclk_start))
    return start_et + 0.5 * exposure_seconds


def predict_pixel(et: float) -> tuple[float, float]:
    vesta_pos_j2000, _ = spiceypy.spkpos("VESTA", et, "J2000", "LT+S", "DAWN")
    rot = spiceypy.pxform("J2000", "DAWN_FC2", et)
    v_cam = np.asarray(rot) @ np.asarray(vesta_pos_j2000)
    vx, vy, vz = v_cam
    angle_x = np.arctan2(vx, vz)
    angle_y = np.arctan2(vy, vz)
    pred_sample = CCD_CENTER[0] + angle_x / IFOV[0]
    pred_line = CCD_CENTER[1] + angle_y / IFOV[1]
    return float(pred_sample), float(pred_line)


def pipeline_min_emission_pixel(phase: str, image_id: str) -> tuple[int, int, float]:
    path = GEOM_ROOT / phase / f"{image_id}_geometry.parquet"
    df = pd.read_parquet(path, columns=["pixel_x", "pixel_y", "emission"])
    row = df.loc[df["emission"].idxmin()]
    return int(row["pixel_x"]), int(row["pixel_y"]), float(row["emission"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spiceypy.furnsh(str(METAKERNEL))

    rows = []
    for phase, image_id, label in CASES:
        label_path = RAW_ROOT / phase / f"{image_id}.LBL"
        et = extract_et(label_path)
        pred_sample, pred_line = predict_pixel(et)
        actual_x, actual_y, min_emission = pipeline_min_emission_pixel(phase, image_id)
        mirrored_x_candidate = 1023.0 - pred_sample

        row = {
            "phase": phase,
            "image_id": image_id,
            "classification": label,
            "et": et,
            "predicted_sample_asis": pred_sample,
            "predicted_line": pred_line,
            "predicted_sample_if_xmirrored": mirrored_x_candidate,
            "pipeline_min_emission_pixel_x": actual_x,
            "pipeline_min_emission_pixel_y": actual_y,
            "pipeline_min_emission_deg": min_emission,
            "residual_asis_x": pred_sample - actual_x,
            "residual_xmirrored_x": mirrored_x_candidate - actual_x,
            "residual_line_y": pred_line - actual_y,
        }
        rows.append(row)
        print(
            f"{phase}/{image_id} [{label}]: predicted=({pred_sample:.1f},{pred_line:.1f})  "
            f"actual_min_emission_pixel=({actual_x},{actual_y}, emission={min_emission:.2f}deg)  "
            f"residual_asis_x={row['residual_asis_x']:+.1f}  "
            f"residual_if_xmirrored={row['residual_xmirrored_x']:+.1f}"
        )

    df_out = pd.DataFrame(rows)
    out_path = OUT_DIR / "xmirror_pinhole_check.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nWrote {len(df_out)} rows to {out_path}")

    print("\n=== summary: which residual is smaller, by classification ===")
    for _, r in df_out.iterrows():
        better = "asis" if abs(r["residual_asis_x"]) < abs(r["residual_xmirrored_x"]) else "xmirrored"
        print(f"  {r['image_id']} [{r['classification']}]: closer match = {better}  "
              f"(|asis|={abs(r['residual_asis_x']):.1f}, |xmirrored|={abs(r['residual_xmirrored_x']):.1f})")

    spiceypy.unload(str(METAKERNEL))


if __name__ == "__main__":
    main()
