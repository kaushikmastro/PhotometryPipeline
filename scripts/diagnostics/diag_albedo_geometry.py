"""
Albedo vs. geometry artifact diagnostic.

Confirms that the north-south residual structure (negative at -10 to -15 lat,
positive at +45 to +65 lat) from diag_albedo_heterogeneity.py is surface-locked
albedo, not a latitude-correlated viewing-geometry artifact.

Step 1: Emission-quartile split of the dark southern band (-15 to -10 lat).
        If residual is flat across emission quartiles → surface-locked (albedo).
        If residual tracks emission → geometry artifact.

Step 2: Longitude structure within latitude bands.
        Real terrain has coherent lat+lon structure (specific units/craters).
        A pure calibration artifact is longitude-flat. For each lat5 band,
        report the std of per-lon5-cell mean residuals vs mean residual.
        High lon-std at fixed lat = real albedo patchiness.

Step 3: Coordinates of the most negative residual cells for comparison with
        known Vesta dark material (Li et al., Reddy et al.).
        We report only our coordinates — cross-check against published map.

Model: Case 1, full-data committed parameters (w=0.46993, g=-0.33688,
       theta_bar=8.2662, B0=1.03, h=0.04). Hapke-2002 H-function.
"""
from __future__ import annotations
import importlib.util, os, sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import polars as pl

from photometry.models.hapke import HapkeModel
from photometry.core.types import GeometryBatch

DATA_5PCT = ROOT / "data" / "silver/dsk256" / "survey_5pct.parquet"

# Model parameters: Case 1 isotropic-H, 18m fit, phase<80
W, G, TB = 0.46993, -0.33688, 8.2662  # committed Case 1 full-data (Hapke-2002 H)

# ── Load and predict ──────────────────────────────────────────────────────────
print("Loading 5% Survey parquet …")
df_raw = (
    pl.scan_parquet(str(DATA_5PCT))
    .filter(pl.col("mission_phase") == "SURVEY")
    .select(["iof", "incidence", "emission", "phase", "latitude", "longitude", "image_id"])
    .collect()
)
print(f"  Rows: {len(df_raw):,}")

inc_px  = df_raw["incidence"].to_numpy().astype(np.float64)
emi_px  = df_raw["emission"].to_numpy().astype(np.float64)
pha_px  = df_raw["phase"].to_numpy().astype(np.float64)
obs_px  = df_raw["iof"].to_numpy().astype(np.float64)
lat_px  = df_raw["latitude"].to_numpy().astype(np.float64)
lon_px  = df_raw["longitude"].to_numpy().astype(np.float64)

print(f"  Evaluating model at {len(obs_px):,} pixels …")
model = HapkeModel(
    enable_shoe=True, enable_roughness=True,
    fixed_parameters={"B0": 1.03, "h": 0.04},
    parameters={"w": W, "g": G, "theta_bar": TB, "B0": 1.03, "h": 0.04},
)
geom_all = GeometryBatch(
    incidence=np.deg2rad(inc_px), emission=np.deg2rad(emi_px), phase=np.deg2rad(pha_px)
)
pred_px  = model._reflectance_numpy(geom_all)
resid_px = obs_px - pred_px

df = df_raw.with_columns([
    pl.Series("pred",  pred_px.astype(np.float32)),
    pl.Series("resid", resid_px.astype(np.float32)),
    (pl.col("latitude") .floordiv(5.0) * 5.0).alias("lat5"),
    (pl.col("longitude").floordiv(5.0) * 5.0).alias("lon5"),
])
print(f"  resid mean={resid_px.mean():+.5f}  std={resid_px.std():.5f}")

# ── STEP 1: Emission-quartile split of the dark southern band ─────────────────
print(f"\n{'='*68}")
print("STEP 1 — Emission-quartile split: lat -15 to -10 deg")
print("  Is the negative residual flat across emission angle (surface-locked)?")
print("  Or does it track emission angle (geometry artifact)?")
print("=" * 68)

south_band = df.filter(
    (pl.col("latitude") >= -15.0) & (pl.col("latitude") < -10.0)
)
n_south = len(south_band)
print(f"\n  Pixels in lat [-15, -10): {n_south:,}")

if n_south > 0:
    emi_south = south_band["emission"].to_numpy()
    q25, q50, q75 = np.percentile(emi_south, [25, 50, 75])
    print(f"  Emission quartile bounds: Q1={q25:.2f}°  Q2={q50:.2f}°  Q3={q75:.2f}°")

    def emit_quartile(emi):
        q = np.zeros(len(emi), dtype=int)
        q[emi <= q25] = 1
        q[(emi > q25) & (emi <= q50)] = 2
        q[(emi > q50) & (emi <= q75)] = 3
        q[emi > q75] = 4
        return q

    emi_q = emit_quartile(emi_south)
    resid_south = south_band["resid"].to_numpy()
    iof_south   = south_band["iof"].to_numpy()

    print(f"\n  {'Q':>2}  {'emit range':>16}  {'n':>7}  {'mean_resid':>11}  "
          f"{'std_resid':>10}  {'mean_emi':>9}  {'mean_iof':>9}")
    q_bounds = [(0, q25, 1), (q25, q50, 2), (q50, q75, 3), (q75, 90, 4)]
    means_q = []
    for lo, hi, q_idx in q_bounds:
        m = emi_q == q_idx
        n = int(m.sum())
        if n == 0:
            continue
        mr = float(resid_south[m].mean())
        sr = float(resid_south[m].std())
        me = float(emi_south[m].mean())
        mi = float(iof_south[m].mean())
        means_q.append(mr)
        print(f"  Q{q_idx}  {lo:5.1f}° – {hi:5.1f}°  {n:7d}  {mr:+11.5f}  "
              f"{sr:10.5f}  {me:9.2f}°  {mi:9.5f}")

    if len(means_q) >= 2:
        resid_range_q = max(means_q) - min(means_q)
        print(f"\n  Range of mean residuals across emission quartiles: {resid_range_q:.5f}")
        overall_resid = float(resid_south.mean())
        print(f"  Overall mean residual for this band:               {overall_resid:+.5f}")
        ratio = resid_range_q / abs(overall_resid) if abs(overall_resid) > 1e-9 else float("nan")
        print(f"  Quartile range / |overall mean| = {ratio:.3f}")
        if ratio < 0.3:
            verdict = "FLAT — residual is surface-locked. Consistent with real albedo."
        elif ratio < 0.6:
            verdict = "MODERATE — some emission dependence. Partially geometry-correlated."
        else:
            verdict = "VARIABLE — residual tracks emission angle. Geometry artifact likely."
        print(f"\n  VERDICT: {verdict}")

    # Also split by image_id to check temporal/illumination diversity
    n_images = south_band["image_id"].n_unique()
    print(f"\n  Unique images contributing to this lat band: {n_images}")
    print(f"  (High diversity → multiple illumination/viewing geometries sampled)")

# ── STEP 2: Longitude structure within latitude bands ─────────────────────────
print(f"\n{'='*68}")
print("STEP 2 — Longitude structure within latitude bands")
print("  High lon-std at fixed lat = patchwork terrain (real albedo).")
print("  Lon-flat at fixed lat = latitude-correlated artifact.")
print("=" * 68)

# Per (lat5, lon5) cell means; then aggregate per lat5 band
spatial = (
    df
    .group_by(["lat5", "lon5"])
    .agg([
        pl.len().alias("n"),
        pl.col("resid").mean().alias("mean_resid"),
        pl.col("iof").mean().alias("mean_iof"),
    ])
    .filter(pl.col("n") >= 30)
)

# For each lat5 band: how much do the lon5 cells scatter around the band mean?
lat_summary = (
    spatial
    .group_by("lat5")
    .agg([
        pl.len().alias("n_cells"),
        pl.col("n").sum().alias("n_pixels"),
        pl.col("mean_resid").mean().alias("lat_mean_resid"),
        pl.col("mean_resid").std().alias("lat_lon_std"),   # lon scatter within band
        pl.col("mean_resid").max().alias("lat_resid_max"),
        pl.col("mean_resid").min().alias("lat_resid_min"),
        pl.col("mean_iof").mean().alias("lat_mean_iof"),
    ])
    .filter(pl.col("n_cells") >= 3)  # need >=3 lon cells to measure scatter
    .sort("lat5")
)

print(f"\n  lat5 bands with >=3 lon cells (each with n>=30 pixels): {len(lat_summary)}")
print(f"\n  {'lat5':>6}  {'n_cells':>8}  {'n_pixels':>9}  {'lat_mean_resid':>15}  "
      f"{'lon_std':>8}  {'lon_range':>10}  {'mean_iof':>9}")
for row in lat_summary.iter_rows(named=True):
    lon_range = row["lat_resid_max"] - row["lat_resid_min"]
    print(f"  {row['lat5']:6.1f}  {row['n_cells']:8d}  {row['n_pixels']:9d}  "
          f"{row['lat_mean_resid']:+15.5f}  {row['lat_lon_std']:8.5f}  "
          f"{lon_range:10.5f}  {row['lat_mean_iof']:9.5f}")

# Summary: global lon-scatter vs lat-scatter
lon_std_arr  = lat_summary["lat_lon_std"].drop_nulls().to_numpy()
lat_mean_arr = lat_summary["lat_mean_resid"].to_numpy()
print(f"\n  Across all lat5 bands:")
print(f"    Lat-to-lat std (lat signal):   {lat_mean_arr.std():.5f}")
print(f"    Median within-lat lon-std:     {np.median(lon_std_arr):.5f}")
print(f"    Mean within-lat lon-std:       {lon_std_arr.mean():.5f}")
print(f"    Lon-std / lat-std ratio:       {lon_std_arr.mean() / max(lat_mean_arr.std(), 1e-9):.3f}")
print(f"    (ratio > 0.5 = strong 2D patchwork; < 0.2 = predominantly lat-striped)")

# Focus on the dark southern and bright northern bands
print(f"\n  Southern band residuals (lat5 = -15 to -10):")
south_cells = spatial.filter(
    (pl.col("lat5") >= -15.0) & (pl.col("lat5") < -5.0)
).sort(["lat5", "lon5"])
if len(south_cells) > 0:
    for row in south_cells.iter_rows(named=True):
        print(f"    lat5={row['lat5']:5.0f}  lon5={row['lon5']:6.0f}  "
              f"n={row['n']:6d}  mean_resid={row['mean_resid']:+.5f}  mean_iof={row['mean_iof']:.5f}")

print(f"\n  Northern band residuals (lat5 = +45 to +65):")
north_cells = spatial.filter(
    (pl.col("lat5") >= 45.0) & (pl.col("lat5") < 70.0)
).sort(["lat5", "lon5"])
if len(north_cells) > 0:
    for row in north_cells.iter_rows(named=True):
        print(f"    lat5={row['lat5']:5.0f}  lon5={row['lon5']:6.0f}  "
              f"n={row['n']:6d}  mean_resid={row['mean_resid']:+.5f}  mean_iof={row['mean_iof']:.5f}")

# ── STEP 3: Most extreme residual cells ──────────────────────────────────────
print(f"\n{'='*68}")
print("STEP 3 — Extreme residual cell coordinates")
print("  Report for cross-check against published Vesta albedo maps")
print("  (Li et al. 2012, Reddy et al. 2012, Longobardo et al. 2016)")
print("=" * 68)

spatial_full = (
    df
    .group_by(["lat5", "lon5"])
    .agg([
        pl.len().alias("n"),
        pl.col("resid").mean().alias("mean_resid"),
        pl.col("resid").std().alias("std_resid"),
        pl.col("iof").mean().alias("mean_iof"),
        pl.col("emission").mean().alias("mean_emi"),
        pl.col("incidence").mean().alias("mean_inc"),
    ])
    .filter(pl.col("n") >= 30)
)

n_show = 20
print(f"\n  Top {n_show} most NEGATIVE residual cells (model over-predicts → terrain darker than w=0.458 implies):")
print(f"  {'lat5':>6}  {'lon5':>6}  {'n':>6}  {'mean_resid':>11}  {'mean_iof':>9}  "
      f"{'mean_emi':>9}  {'mean_inc':>9}")
for row in spatial_full.sort("mean_resid").head(n_show).iter_rows(named=True):
    print(f"  {row['lat5']:6.1f}  {row['lon5']:6.1f}  {row['n']:6d}  "
          f"{row['mean_resid']:+11.5f}  {row['mean_iof']:9.5f}  "
          f"{row['mean_emi']:9.2f}°  {row['mean_inc']:9.2f}°")

print(f"\n  Top {n_show} most POSITIVE residual cells (model under-predicts → terrain brighter than w=0.458 implies):")
print(f"  {'lat5':>6}  {'lon5':>6}  {'n':>6}  {'mean_resid':>11}  {'mean_iof':>9}  "
      f"{'mean_emi':>9}  {'mean_inc':>9}")
for row in spatial_full.sort("mean_resid", descending=True).head(n_show).iter_rows(named=True):
    print(f"  {row['lat5']:6.1f}  {row['lon5']:6.1f}  {row['n']:6d}  "
          f"{row['mean_resid']:+11.5f}  {row['mean_iof']:9.5f}  "
          f"{row['mean_emi']:9.2f}°  {row['mean_inc']:9.2f}°")

# Check whether extreme residual cells have anomalous geometry
# (if dark cells were observed only at high emission, that's suspicious)
print(f"\n  Geometry check for the 20 darkest cells:")
bottom20 = spatial_full.sort("mean_resid").head(n_show)
top20    = spatial_full.sort("mean_resid", descending=True).head(n_show)
rest     = spatial_full.sort("mean_resid")[n_show:-n_show]

print(f"    {'group':>10}  {'mean_emi':>9}  {'std_emi':>8}  {'mean_inc':>9}")
for label, sub in [("darkest-20", bottom20), ("brightest-20", top20), ("rest", rest)]:
    print(f"    {label:>10}  {sub['mean_emi'].mean():9.2f}°  "
          f"{sub['mean_emi'].std():8.2f}°  {sub['mean_inc'].mean():9.2f}°")

print(f"\n  NOTE: Cross-check these coordinates against:")
print(f"    Li et al. 2012 (Icarus 219) — Vesta albedo/color map from Dawn FC")
print(f"    Reddy et al. 2012 (Icarus 221) — dark material distribution (impactor ejecta)")
print(f"    Longobardo et al. 2016 (Icarus 267) — Vesta photometric properties")
print(f"    Expected: dark material concentrated in equatorial/southern hemisphere")
print(f"    near Veneneia/Rheasilvia rim and Marcia crater region.")
