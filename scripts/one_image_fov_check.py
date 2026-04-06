#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spiceypy


def load_planetaryimage():
    """Import planetaryimage with NumPy 2 compatibility shims."""
    if not hasattr(np, "product"):
        np.product = np.prod  # type: ignore[attr-defined]

    try:
        from planetaryimage import PDS3Image

        return PDS3Image
    except ValueError as exc:
        if "fromstring" not in str(exc):
            raise

        old_fromstring = np.fromstring

        def _fromstring_compat(string, dtype=float, count=-1, sep=''):
            if sep == '' and isinstance(string, (bytes, bytearray, memoryview)):
                return np.frombuffer(string, dtype=dtype, count=count)
            return old_fromstring(string, dtype=dtype, count=count, sep=sep)

        np.fromstring = _fromstring_compat  # type: ignore[assignment]
        from planetaryimage import PDS3Image

        return PDS3Image


def to_spice_utc_string(value: str) -> str:
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return text.replace(" ", "T")


def find_label_value(label: object, key: str):
    key_upper = key.upper()

    def _search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).upper() == key_upper:
                    return v
                nested = _search(v)
                if nested is not None:
                    return nested
        return None

    return _search(label)


def extract_observation_et(label: object) -> float:
    for key in [
        "START_TIME",
        "STOP_TIME",
        "IMAGE_TIME",
        "DAWN:ALT_START_TIME",
        "DAWN:ALT_STOP_TIME",
    ]:
        value = find_label_value(label, key)
        if value is None:
            continue
        try:
            return float(spiceypy.utc2et(to_spice_utc_string(value)))
        except Exception:
            continue
    raise RuntimeError("Could not extract observation time from image label")


def build_center_ray(bounds: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    corners = bounds.T if bounds.shape[0] == 3 else bounds
    c0, c1, c2, c3 = corners[:4]
    ny, nx = shape
    cx = nx // 2
    cy = ny // 2
    u = cx / max(1, nx - 1)
    v = cy / max(1, ny - 1)
    top = (1.0 - u) * c0 + u * c1
    bottom = (1.0 - u) * c3 + u * c2
    ray = (1.0 - v) * top + v * bottom
    return ray / np.linalg.norm(ray)


def main() -> int:
    PDS3Image = load_planetaryimage()

    data_root = Path("/home/kaushim07/photometry_mcmc_env/data")
    spice_dir = data_root / "02_spice_kernels"
    image_path = data_root / "01_calibrated_images" / "FC21B0003932_11223181943F1F.IMG"

    # Load kernels in fallback order from local folder.
    ordered_patterns = ["*.tls", "*.tsc", "*.tpc", "*.tf", "*.ti", "*.bsp", "*.bc"]
    for pat in ordered_patterns:
        for kernel in sorted(spice_dir.glob(pat)):
            try:
                spiceypy.furnsh(str(kernel))
            except Exception:
                continue

    pds = PDS3Image.open(str(image_path))
    et = extract_observation_et(pds.label)
    image_shape = pds.image.shape

    instrument = "DAWN_FC2"
    target = "VESTA"
    target_frame = "IAU_VESTA"
    observer = "DAWN"
    abcorr = "LT+S"

    cam_id = spiceypy.bodn2c(instrument)
    _, cam_frame, boresight, _, bounds = spiceypy.getfov(cam_id, 4)
    cam_frame = cam_frame.decode().strip() if isinstance(cam_frame, bytes) else str(cam_frame)

    center_ray = build_center_ray(np.asarray(bounds, dtype=np.float64), image_shape)

    print(f"image={image_path.name}")
    print(f"et={et:.6f}")
    print(f"frame_check instrument={instrument} cam_frame={cam_frame} target_frame={target_frame}")
    print("center_ray_cam", center_ray.tolist())
    print("boresight_cam", np.asarray(boresight, dtype=np.float64).tolist())

    try:
        in_fov = bool(
            spiceypy.fovtrg(
                instrument,
                target,
                "ELLIPSOID",
                target_frame,
                abcorr,
                observer,
                et,
            )
        )
        print(f"fovtrg={in_fov}")
    except Exception as exc:
        print(f"fovtrg_error={exc}")

    try:
        spoint, _, _ = spiceypy.sincpt(
            "ELLIPSOID",
            target,
            et,
            target_frame,
            abcorr,
            observer,
            cam_frame,
            center_ray,
        )
        spoint = np.asarray(spoint, dtype=np.float64)
        print("sincpt_center_hit=True")
        print("sincpt_spoint_km", spoint.tolist())
    except Exception as exc:
        print("sincpt_center_hit=False")
        print(f"sincpt_error={exc}")

    spiceypy.kclear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
