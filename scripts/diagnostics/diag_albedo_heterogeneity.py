"""
Albedo heterogeneity test: is CV_other real surface variation or instrumental?

Uses the 5% Survey parquet (33.5M pixels, broader spatial coverage) to test
whether the within-bin systematic residual (CV_other=8.20%, confirmed NOT
sampling noise in Test A) is caused by real Vesta albedo heterogeneity.

Step 1: Within-bin correlations. For 10 most-populated restricted bins,
        compute r(per-pixel residual, latitude) and r(residual, longitude).
        If |r| > 0.3 in many bins, residuals correlate with surface location
        at fixed geometry → albedo heterogeneity.

Step 2: Spatial residual map. Bin per-pixel (obs - pred) by 5-deg lat/lon
        surface cells. Systematic spatial patterns = global albedo structure
        that a single w cannot capture.

Step 3: Quantify. Correlate within-bin std_iof with within-bin I/F range
        (proxy for albedo diversity sampled). If r(std_iof, iof_range) > 0.7,
        within-bin scatter is proportional to albedo heterogeneity sampled.

Model: Case 1, full-data committed parameters (w=0.46993, g=-0.33688,
       theta_bar=8.2662, B0=1.03, h=0.04). Hapke-2002 H-function.
"""
from __future__ import annotations
import importlib.util, os, sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location("run_baseline_fit",
    ROOT / "scripts" / "run_baseline_fit.py")
rbf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbf)

from photometry.models.hapke import HapkeModel
from photometry.core.types import GeometryBatch

DATA_5PCT = ROOT / "data" / "silver/dsk256" / "survey_5pct.parquet"
GRID = 5.0
INC_MAX = rbf.INC_MAX   # 50.0
EMI_MAX = rbf.EMI_MAX   # 50.0
eps = 1e-12

# Model parameters: Case 1 isotropic-H, 18m data, phase<80
W, G, TB = 0.46993, -0.33688, 8.2662  # committed Case 1 full-data (Hapke-2002 H)

print(f"Loading 5% Survey parquet …")
df_raw = (
    pl.scan_parquet(str(DATA_5PCT))
    .filter(pl.col("mission_phase") == "SURVEY")
    .select(["iof","incidence","emission","phase","latitude","longitude"])
    .collect()
)
print(f"  Rows: {len(df_raw):,}")

# Per-pixel model prediction
inc_px  = df_raw["incidence"].to_numpy().astype(np.float64)
emi_px  = df_raw["emission"].to_numpy().astype(np.float64)
pha_px  = df_raw["phase"].to_numpy().astype(np.float64)
obs_px  = df_raw["iof"].to_numpy().astype(np.float64)
lat_px  = df_raw["latitude"].to_numpy()
lon_px  = df_raw["longitude"].to_numpy()

print(f"  Evaluating model at {len(obs_px):,} pixels …")
model = HapkeModel(
    enable_shoe=True, enable_roughness=True,
    fixed_parameters={"B0":1.03,"h":0.04},
    parameters={"w":W,"g":G,"theta_bar":TB,"B0":1.03,"h":0.04},
)
geom_all = GeometryBatch(
    incidence=np.deg2rad(inc_px), emission=np.deg2rad(emi_px), phase=np.deg2rad(pha_px)
)
pred_px = model._reflectance_numpy(geom_all)
resid_px = obs_px - pred_px  # positive = model under-predicts (too dark in model)

# Add grid columns and residual to dataframe
df_px = df_raw.with_columns([
    pl.Series("pred",  pred_px.astype(np.float32)),
    pl.Series("resid", resid_px.astype(np.float32)),
    ((pl.col("phase")     / GRID).round() * GRID).alias("alpha_grid"),
    ((pl.col("incidence") / GRID).round() * GRID).alias("i_grid"),
    ((pl.col("emission")  / GRID).round() * GRID).alias("e_grid"),
    (pl.col("latitude")  .floordiv(5.0) * 5.0).alias("lat5"),
    (pl.col("longitude") .floordiv(5.0) * 5.0).alias("lon5"),
])
print(f"  Per-pixel predictions done. resid mean={resid_px.mean():+.5f}  std={resid_px.std():.5f}")

# ── STEP 1: Within-bin correlations ──────────────────────────────────────────
print(f"\n{'='*68}")
print("STEP 1 — Within-bin r(residual, latitude/longitude)")
print(f"  Restricted bins: i<{INC_MAX}°, e<{EMI_MAX}°, phase<80°, n>=100 pixels")
print("=" * 68)

# Restrict to the same geometry subset as the main fit
df_restr = df_px.filter(
    (pl.col("incidence") < INC_MAX)
    & (pl.col("emission")  < EMI_MAX)
    & (pl.col("phase")     < 80.0)
)

# Compute within-bin Pearson r(residual, lat) and r(residual, lon)
bin_corr = (
    df_restr
    .group_by(["alpha_grid","i_grid","e_grid"])
    .agg([
        pl.len().alias("n"),
        pl.col("iof").mean().alias("mean_iof"),
        pl.col("iof").std().alias("std_iof"),
        pl.col("iof").max().alias("max_iof"),
        pl.col("iof").min().alias("min_iof"),
        pl.col("resid").mean().alias("mean_resid"),
        pl.col("resid").std().alias("std_resid"),
        pl.col("latitude").std().alias("lat_span"),
        pl.col("longitude").std().alias("lon_span"),
        pl.corr("resid","latitude").alias("r_lat"),
        pl.corr("resid","longitude").alias("r_lon"),
    ])
    .filter(pl.col("n") >= 100)
    .sort("n", descending=True)
)
print(f"\n  Bins with n>=100 pixels: {len(bin_corr)}")
print(f"  Showing top-10 most populated:")
print(f"\n  {'α':>5}  {'i':>5}  {'e':>5}  {'n':>7}  {'std_iof':>9}  "
      f"{'iof_range':>10}  {'r_lat':>7}  {'r_lon':>7}  {'|r|_max':>8}")

for row in bin_corr.head(10).iter_rows(named=True):
    iof_range = row["max_iof"] - row["min_iof"]
    r_abs = max(abs(row["r_lat"]), abs(row["r_lon"]))
    print(f"  {row['alpha_grid']:5.0f}  {row['i_grid']:5.0f}  {row['e_grid']:5.0f}  "
          f"{row['n']:7d}  {row['std_iof']:9.5f}  {iof_range:10.5f}  "
          f"{row['r_lat']:7.4f}  {row['r_lon']:7.4f}  {r_abs:8.4f}")

# Summary statistics of the within-bin correlations
r_lat_arr = bin_corr["r_lat"].drop_nulls().to_numpy()
r_lon_arr = bin_corr["r_lon"].drop_nulls().to_numpy()
print(f"\n  Summary across all {len(bin_corr)} bins (n>=100):")
print(f"    |r_lat|: mean={np.abs(r_lat_arr).mean():.4f}  median={np.median(np.abs(r_lat_arr)):.4f}  "
      f"p75={np.percentile(np.abs(r_lat_arr),75):.4f}  max={np.abs(r_lat_arr).max():.4f}")
print(f"    |r_lon|: mean={np.abs(r_lon_arr).mean():.4f}  median={np.median(np.abs(r_lon_arr)):.4f}  "
      f"p75={np.percentile(np.abs(r_lon_arr),75):.4f}  max={np.abs(r_lon_arr).max():.4f}")
frac_sig = np.mean((np.abs(r_lat_arr) > 0.3) | (np.abs(r_lon_arr) > 0.3))
print(f"    Fraction of bins with |r| > 0.3 (either lat or lon): {frac_sig:.3f} ({frac_sig*100:.1f}%)")

# ── STEP 2: Spatial residual map ──────────────────────────────────────────────
print(f"\n{'='*68}")
print("STEP 2 — Spatial residual map: mean(obs-pred) per 5-deg lat/lon cell")
print("=" * 68)

spatial = (
    df_px
    .group_by(["lat5","lon5"])
    .agg([
        pl.len().alias("n"),
        pl.col("resid").mean().alias("mean_resid"),
        pl.col("resid").std().alias("std_resid"),
        pl.col("iof").mean().alias("mean_iof"),
    ])
    .filter(pl.col("n") >= 50)
    .sort("mean_resid", descending=True)
)
print(f"\n  Surface cells with n>=50 pixels: {len(spatial)}")
print(f"  Residual range: {spatial['mean_resid'].min():.5f} – {spatial['mean_resid'].max():.5f}")
print(f"  Residual std across cells: {spatial['mean_resid'].std():.5f}")

# Report extremes
n_show = 12
print(f"\n  Top {n_show} POSITIVE residual cells (model under-predicts — terrain brighter than model):")
print(f"  {'lat':>6}  {'lon':>7}  {'n':>6}  {'mean_resid':>11}  {'mean_iof':>10}")
for row in spatial.head(n_show).iter_rows(named=True):
    print(f"  {row['lat5']:6.1f}  {row['lon5']:7.1f}  {row['n']:6d}  "
          f"{row['mean_resid']:+11.5f}  {row['mean_iof']:10.5f}")

print(f"\n  Top {n_show} NEGATIVE residual cells (model over-predicts — terrain darker than model):")
print(f"  {'lat':>6}  {'lon':>7}  {'n':>6}  {'mean_resid':>11}  {'mean_iof':>10}")
for row in spatial.sort("mean_resid").head(n_show).iter_rows(named=True):
    print(f"  {row['lat5']:6.1f}  {row['lon5']:7.1f}  {row['n']:6d}  "
          f"{row['mean_resid']:+11.5f}  {row['mean_iof']:10.5f}")

# Global spatial structure: variance in cell means vs pixel noise
var_cell_means = float(spatial["mean_resid"].var())
var_all_pixels = float(resid_px.var())
n_per_cell_med = float(spatial["n"].median())
expected_noise_var = var_all_pixels / n_per_cell_med
signal_to_noise = var_cell_means / expected_noise_var
print(f"\n  Spatial structure signal/noise = {signal_to_noise:.2f}")
print(f"  (>1 means spatial pattern is real, not just measurement noise per cell)")

# ── STEP 3: std_iof vs I/F range ──────────────────────────────────────────────
print(f"\n{'='*68}")
print("STEP 3 — Albedo heterogeneity proxy: r(std_iof, I/F range) per bin")
print("=" * 68)

iof_range_arr  = (bin_corr["max_iof"] - bin_corr["min_iof"]).to_numpy()
std_iof_arr    = bin_corr["std_iof"].to_numpy()
lat_span_arr   = bin_corr["lat_span"].to_numpy()
n_arr          = bin_corr["n"].to_numpy()

r_std_range = float(np.corrcoef(std_iof_arr, iof_range_arr)[0,1])
r_std_lat   = float(np.corrcoef(std_iof_arr, lat_span_arr)[0,1])
mean_iof_arr = bin_corr["mean_iof"].to_numpy()
r_std_meaniof = float(np.corrcoef(std_iof_arr, mean_iof_arr)[0,1])

print(f"\n  Across {len(bin_corr)} bins with n>=100:")
print(f"    r(std_iof, I/F range):   {r_std_range:+.4f}   [+1 = scatter ~ albedo spread]")
print(f"    r(std_iof, lat_span):    {r_std_lat:+.4f}   [+1 = scatter ~ geographic span]")
print(f"    r(std_iof, mean_iof):    {r_std_meaniof:+.4f}   [+1 = scatter ~ overall brightness]")

# CV (std/mean) to normalize for brightness dependence
cv_iof = std_iof_arr / np.maximum(mean_iof_arr, 1e-9)
iof_cv_range = iof_range_arr / np.maximum(mean_iof_arr, 1e-9)  # relative range
r_cv_relrange = float(np.corrcoef(cv_iof, iof_cv_range)[0,1])
print(f"    r(CV_iof, relative I/F range): {r_cv_relrange:+.4f}   [normalized for brightness]")

# Quartile split: high-CV vs low-CV bins
cv_median = float(np.median(cv_iof))
hi_cv = cv_iof > cv_median
lo_cv = ~hi_cv
print(f"\n  High-CV bins (n={hi_cv.sum()}) vs Low-CV bins (n={lo_cv.sum()}):")
print(f"    mean I/F range:  high={iof_range_arr[hi_cv].mean():.5f}  low={iof_range_arr[lo_cv].mean():.5f}  "
      f"ratio={iof_range_arr[hi_cv].mean()/max(iof_range_arr[lo_cv].mean(),1e-9):.2f}x")
print(f"    mean lat_span:   high={lat_span_arr[hi_cv].mean():.2f}°  low={lat_span_arr[lo_cv].mean():.2f}°  "
      f"ratio={lat_span_arr[hi_cv].mean()/max(lat_span_arr[lo_cv].mean(),1e-9):.2f}x")
print(f"    mean n_pixels:   high={n_arr[hi_cv].mean():.0f}  low={n_arr[lo_cv].mean():.0f}")

print(f"\n  CONCLUSION — STEP 3:")
if r_cv_relrange > 0.6:
    print(f"  STRONG: r(CV, relative_range)={r_cv_relrange:.3f}. Within-bin scatter is")
    print(f"  proportional to albedo heterogeneity sampled — strong evidence for real")
    print(f"  surface albedo variation as the dominant within-bin systematic.")
elif r_cv_relrange > 0.35:
    print(f"  MODERATE: r(CV, relative_range)={r_cv_relrange:.3f}. Partial albedo signal,")
    print(f"  consistent with real surface variation but also other contributions.")
else:
    print(f"  WEAK: r(CV, relative_range)={r_cv_relrange:.3f}. Within-bin scatter is NOT")
    print(f"  primarily driven by albedo heterogeneity within each geometric bin.")

# Summary
print(f"\n{'='*68}")
print("OVERALL CONCLUSION")
print("=" * 68)
print(f"\n  CV_other={8.199:.2f}% (within-bin, orthogonal to incidence trend)")
print(f"  Albedo heterogeneity evidence:")
print(f"    Step 1: {frac_sig*100:.0f}% of bins have |r(resid,lat/lon)|>0.3 (spatial correlation)")
print(f"    Step 2: spatial residual std={spatial['mean_resid'].std():.5f}  "
      f"signal/noise={signal_to_noise:.1f}")
print(f"    Step 3: r(CV, relative_iof_range)={r_cv_relrange:.3f}")
