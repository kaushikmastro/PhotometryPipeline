"""Shape-model resolution report, read directly from a DSK file (not inferred).

Reports, for any DSK/type-2 (plate) shape kernel:
  - plate and vertex counts (dskz02)
  - plate edge length distribution (mean, min, max, std) in km
  - mean plate area in km^2, and the effective ground resolution that implies
  - volumetric mean radius (divergence-theorem mesh volume -> equivalent-sphere
    radius) and mean vertex-distance radius, as two independent estimates --
    useful as a sanity check against a body's published mean radius

Body-agnostic: takes a `--dsk` path, so the same script covers a future target
(e.g. a Psyche shape model) without modification.

No fitting, no per-pixel geometry -- a one-time DSK mesh read/reduction. Still
routed through srun per CLAUDE.md Rule 2 (it iterates the full plate list, which
can be several hundred thousand to a few million plates depending on Q):

  srun --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \\
       --partition=main --qos=standard --time=00:15:00 \\
       python3 scripts/diagnostics/dsk_shape_resolution.py --dsk <path>

Example citation-form output (vesta_gaskell_256_110825.bds, Q=256 Gaskell SPC):

    Vertices (NV): 396,294
    Plates  (NP):  786,432
    ...
    citation form: ~1.69 km/plate, 786,432 plates (Q=256 Gaskell SPC)
    ...
    volumetric mean radius      = 261.575 km
"""
from __future__ import annotations

import argparse

import numpy as np
import spiceypy as spice

DEFAULT_DSK = "/home/kaushim07/photometry_mcmc_env/data/spice_kernels/vesta_gaskell_256_110825.bds"

# SPICE dskv02/dskp02 per-call room limit; read in chunks of this size.
CHUNK = 20000


def compute_dsk_resolution(dsk_path: str) -> dict:
    """Read `dsk_path`'s plate/vertex mesh and return resolution statistics.

    Returns a dict with keys: n_vertices, n_plates, edge_mean_km, edge_min_km,
    edge_max_km, edge_std_km, plate_area_mean_km2, total_surface_area_km2,
    effective_resolution_km (sqrt(mean plate area)), mean_vertex_radius_km,
    mesh_volume_km3, volumetric_mean_radius_km.
    """
    spice.furnsh(dsk_path)
    handle = spice.dasopr(dsk_path)
    try:
        dladsc = spice.dlabfs(handle)
        nv, np_ = spice.dskz02(handle, dladsc)

        verts = np.empty((nv, 3), dtype=np.float64)
        i = 0
        while i < nv:
            n = min(CHUNK, nv - i)
            verts[i : i + n] = np.array(spice.dskv02(handle, dladsc, i + 1, n))
            i += n

        plates = np.empty((np_, 3), dtype=np.int64)
        i = 0
        while i < np_:
            n = min(CHUNK, np_ - i)
            plates[i : i + n] = np.array(spice.dskp02(handle, dladsc, i + 1, n))
            i += n
    finally:
        spice.dascls(handle)
        spice.kclear()

    v1 = verts[plates[:, 0] - 1]
    v2 = verts[plates[:, 1] - 1]
    v3 = verts[plates[:, 2] - 1]

    e12 = np.linalg.norm(v2 - v1, axis=1)
    e23 = np.linalg.norm(v3 - v2, axis=1)
    e31 = np.linalg.norm(v1 - v3, axis=1)
    all_edges = np.concatenate([e12, e23, e31])

    areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
    mean_plate_area = float(areas.mean())

    vertex_radii = np.linalg.norm(verts, axis=1)

    # Divergence theorem: V = (1/6) * sum( v1 . (v2 x v3) ) over origin-referenced
    # tetrahedra formed by each facet -- exact for a closed, outward-oriented mesh.
    signed_vol = float(np.sum(np.einsum("ij,ij->i", v1, np.cross(v2, v3)))) / 6.0
    mesh_volume = abs(signed_vol)
    volumetric_mean_radius = (3.0 * mesh_volume / (4.0 * np.pi)) ** (1.0 / 3.0)

    return {
        "n_vertices": int(nv),
        "n_plates": int(np_),
        "edge_mean_km": float(all_edges.mean()),
        "edge_min_km": float(all_edges.min()),
        "edge_max_km": float(all_edges.max()),
        "edge_std_km": float(all_edges.std()),
        "plate_area_mean_km2": mean_plate_area,
        "total_surface_area_km2": float(areas.sum()),
        "effective_resolution_km": float(np.sqrt(mean_plate_area)),
        "mean_vertex_radius_km": float(vertex_radii.mean()),
        "mesh_volume_km3": mesh_volume,
        "volumetric_mean_radius_km": volumetric_mean_radius,
    }


def format_report(dsk_path: str, stats: dict) -> str:
    lines = [
        f"DSK: {dsk_path}",
        f"Vertices (NV): {stats['n_vertices']:,}",
        f"Plates  (NP):  {stats['n_plates']:,}",
        "",
        "=== Plate edge length (km) ===",
        f"  mean = {stats['edge_mean_km']:.4f}",
        f"  min  = {stats['edge_min_km']:.4f}",
        f"  max  = {stats['edge_max_km']:.4f}",
        f"  std  = {stats['edge_std_km']:.4f}",
        "",
        "=== Plate area (km^2) ===",
        f"  mean = {stats['plate_area_mean_km2']:.5f}",
        f"  total surface area = {stats['total_surface_area_km2']:,.2f} km^2",
        "",
        "=== Effective ground resolution ===",
        f"  sqrt(mean plate area) = {stats['effective_resolution_km']:.4f} km/plate",
        f"  mean edge length      = {stats['edge_mean_km']:.4f} km/plate",
        f"  citation form: ~{stats['edge_mean_km']:.2f} km/plate, {stats['n_plates']:,} plates",
        "",
        "=== Mean radius (two independent estimates) ===",
        f"  mean vertex-distance radius = {stats['mean_vertex_radius_km']:.3f} km",
        f"  mesh volume                 = {stats['mesh_volume_km3']:,.1f} km^3",
        f"  volumetric mean radius      = {stats['volumetric_mean_radius_km']:.3f} km",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsk", default=DEFAULT_DSK, help="Path to a type-2 (plate) DSK file"
    )
    args = parser.parse_args()

    stats = compute_dsk_resolution(args.dsk)
    print(format_report(args.dsk, stats))
    print("\nDONE")


if __name__ == "__main__":
    main()
