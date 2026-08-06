#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

KERNEL_DIR = Path("/scratch/kaushim07/vesta_data/spice_kernels")
METAKERNEL_PATH = KERNEL_DIR / "dawn_dynamic.tm"

# User-requested must-have references in PATH_VALUES.
REQ_SC_1108 = str(KERNEL_DIR / "dawn_sc_110802-110831_110922_v1.bsp")
REQ_SA_1108 = str(KERNEL_DIR / "dawn_sa_110808_110814.bc")

EXT_ORDER = [".tls", ".tsc", ".tpc", ".tf", ".ti", ".bsp", ".bc"]


def main() -> int:
    if not KERNEL_DIR.exists():
        raise FileNotFoundError(f"Kernel directory not found: {KERNEL_DIR}")

    kernel_files: list[Path] = []
    for ext in EXT_ORDER:
        kernel_files.extend(sorted(p.resolve() for p in KERNEL_DIR.glob(f"*{ext}")))

    ck_2012 = [kernel for kernel in kernel_files if kernel.suffix.lower() == ".bc" and "12" in kernel.name]
    if ck_2012:
        other_kernels = [kernel for kernel in kernel_files if kernel not in ck_2012]
        kernel_files = other_kernels + ck_2012

    if not kernel_files:
        raise RuntimeError(f"No kernels found in {KERNEL_DIR} for extensions: {EXT_ORDER}")

    lines: list[str] = [
        "KPL/MK",
        "",
        "\\begindata",
        "",
        "PATH_VALUES = (",
        f"  '{KERNEL_DIR}'",
        ")",
        "",
        "PATH_SYMBOLS = (",
        "  'SPICE'",
        ")",
        "",
        "KERNELS_TO_LOAD = (",
    ]

    for idx, kernel_path in enumerate(kernel_files):
        suffix = "," if idx < len(kernel_files) - 1 else ""
        # Use short symbolized paths to prevent SPICE text-kernel line truncation.
        lines.append(f"  '$SPICE/{kernel_path.name}'{suffix}")

    lines.extend(
        [
            ")",
            "",
            "\\begintext",
            "Manual-mode dynamic metakernel generated from local disk only.",
            "Includes absolute paths for all .bsp/.bc/.ti/.tf/.tpc/.tsc kernels in spice_kernels.",
        ]
    )

    METAKERNEL_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote metakernel: {METAKERNEL_PATH}")
    print(f"Kernel entries written: {len(kernel_files)}")
    print(f"Must-have PATH_VALUES SC exists: {Path(REQ_SC_1108).exists()} -> {REQ_SC_1108}")
    print(f"Must-have PATH_VALUES SA exists: {Path(REQ_SA_1108).exists()} -> {REQ_SA_1108}")
    print("--- First 20 lines of dawn_dynamic.tm ---")
    for line in METAKERNEL_PATH.read_text(encoding="utf-8").splitlines()[:20]:
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
