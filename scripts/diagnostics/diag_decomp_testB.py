"""
Corrected decomposition + Test B.

Corrected Step 2: proper orthogonal decomposition.
  r_i = pred_i - obs_i
  r_trend_i = mean(r within 10-deg incidence bin)   [bin-mean component]
  r_other_i = r_i - r_trend_i                        [within-bin, orthogonal]
  Orthogonality: sum(r_trend * r_other) = 0 exactly
  → CV-RMSE² = CV_trend² + CV_other²  (clean quadrature, no cross terms)
  → ANOVA R² = SS_between / SS_total_centered = fraction uniquely from incidence

Test B: ratio pred/obs per 5-deg incidence bin on FULL Survey coverage (2891 bins).
  Slope in 0-50 regime vs 50-80 regime → one disk-function effect or two
  (disk-function trend + separate roughness/shadow breakdown at high-i)?
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

DATA_18M = ROOT / "data" / "silver/dsk256" / "combined_rc_survey_sample_corrected_dsk256.parquet"
DATA_7X7 = ROOT / "data" / "silver/dsk256" / "survey_7x7_geometry.parquet"
eps = 1e-12

# ── Load 7x7 restricted subset (906 bins, phase<80) ──────────────────────────
rbf.DATA_PATH = DATA_7X7
df_r = rbf.build_survey_loie()
mask80 = df_r["mean_phase"].to_numpy() < 80.0
df_r = df_r.filter(pl.Series(mask80))

obs = df_r["mean_iof"].to_numpy()
inc = df_r["mean_incidence"].to_numpy()
emi = df_r["mean_emission"].to_numpy()
pha = df_r["mean_phase"].to_numpy()

model = HapkeModel(enable_shoe=True, enable_roughness=True,
                   fixed_parameters={"B0":1.03,"h":0.04},
                   parameters={"w":0.46993,"g":-0.33688,"theta_bar":8.2662,"B0":1.03,"h":0.04})
pred = model._reflectance_numpy(GeometryBatch(
    incidence=np.deg2rad(inc), emission=np.deg2rad(emi), phase=np.deg2rad(pha)))

# ── CORRECTED STEP 2 ──────────────────────────────────────────────────────────
print("=" * 68)
print("CORRECTED STEP 2 — Orthogonal decomposition (906 bins, 7x7, phase<80)")
print("  r = pred - obs  partitioned by 10-deg incidence bin:")
print("    r_trend  = bin-mean(r)  [systematic incidence bias]")
print("    r_other  = r - r_trend  [within-bin residual, orthogonal to r_trend]")
print("=" * 68)

r = pred - obs

# Orthogonal partition
r_trend = np.zeros_like(r)
bins_info: dict[int, tuple[int, float, float]] = {}
for lo in range(0, 60, 10):
    hi = lo + 10
    m = (inc >= lo) & (inc < hi)
    if m.sum() > 0:
        bm = float(r[m].mean())
        r_trend[m] = bm
        bins_info[lo] = (int(m.sum()), bm, float(r[m].std()))

r_other = r - r_trend

# Verify
dot = float(np.dot(r_trend, r_other))
print(f"\n  Orthogonality check: r_trend · r_other = {dot:.3e}  (float round-off only)")

# Quadrature decomposition
cv_total = float(np.sqrt(np.mean(r**2)))       / obs.mean() * 100
cv_trend = float(np.sqrt(np.mean(r_trend**2))) / obs.mean() * 100
cv_other = float(np.sqrt(np.mean(r_other**2))) / obs.mean() * 100
check = np.sqrt(cv_trend**2 + cv_other**2)

print(f"\n  Quadrature decomposition (exact because orthogonal):")
print(f"    CV_total              = {cv_total:.4f}%")
print(f"    CV_trend (incidence)  = {cv_trend:.4f}%   variance share: {cv_trend**2/cv_total**2*100:.1f}%")
print(f"    CV_other (within-bin) = {cv_other:.4f}%   variance share: {cv_other**2/cv_total**2*100:.1f}%")
print(f"    sqrt(CV_trend²+CV_other²) = {check:.4f}%  [discrepancy: {abs(check-cv_total):.2e}%]")

# ANOVA R² on centered residuals
grand_mean = float(r.mean())
SS_total   = float(np.sum((r - grand_mean)**2))
SS_between = sum(n_b * (bm - grand_mean)**2 for _, (n_b, bm, _) in bins_info.items())
R2 = SS_between / SS_total

print(f"\n  ANOVA R² of incidence bin on (pred-obs):")
print(f"    R² = {R2:.4f}  ({R2*100:.1f}% of centered residual variance)")
print(f"    Fraction of variance uniquely attributable to incidence systematics.")

# Per-bin detail
print(f"\n  Per-bin breakdown:")
print(f"  {'inc bin':>9}  {'n':>5}  {'r_trend':>12}  {'r_trend/obs%':>13}  {'CV_other%':>11}")
for lo, (n_b, bm, sr) in sorted(bins_info.items()):
    hi = lo+10
    m = (inc>=lo)&(inc<hi)
    cv_o = float(np.sqrt(np.mean(r_other[m]**2))) / obs[m].mean() * 100
    print(f"  i={lo:2d}–{hi:2d}°  {n_b:5d}  {bm:+12.7f}  {bm/obs[m].mean()*100:+13.3f}%  {cv_o:11.3f}%")

print(f"\n  Corrected thesis statement:")
print(f"    Incidence-systematic: CV_trend={cv_trend:.2f}%, R²={R2:.3f} of centered variance.")
print(f"    Within-bin component: CV_other={cv_other:.2f}%, orthogonal to incidence trend.")
print(f"    These add in exact quadrature to CV_total={cv_total:.2f}%.")
print(f"    No comparison to Li et al.'s RMS is drawn — their incidence distribution")
print(f"    is not available, so the {cv_trend:.2f}% figure cannot be transferred to their data.")

# ── TEST B ─────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 68}")
print("TEST B — Ratio by 5-deg incidence bin, full Survey coverage (phase<90°)")
print("  Broad Case 1 (w=0.43482, g=-0.35198, theta_bar=1.0°, isotropic-H)")
print("  Q: single smooth monotone, or two regimes at i~50° inflection?")
print("=" * 68)

print("\nBuilding full Survey bins …")
df_b = (
    pl.scan_parquet(str(DATA_18M))
    .filter(pl.col("mission_phase") == "SURVEY")
    .with_columns([
        ((pl.col("phase")     / 5.0).round() * 5.0).alias("ag"),
        ((pl.col("incidence") / 5.0).round() * 5.0).alias("ig"),
        ((pl.col("emission")  / 5.0).round() * 5.0).alias("eg"),
    ])
    .group_by(["ag","ig","eg"])
    .agg([
        pl.col("incidence").mean().alias("mean_incidence"),
        pl.col("emission").mean().alias("mean_emission"),
        pl.col("phase").mean().alias("mean_phase"),
        pl.col("iof").mean().alias("mean_iof"),
        pl.col("iof").std().alias("std_iof"),
        pl.col("iof").count().alias("n_pixels"),
    ])
    .filter((pl.col("n_pixels") >= 10) & (pl.col("mean_phase") < 90.0))
    .drop_nulls(subset=["std_iof"])
    .with_columns(
        pl.when(pl.col("std_iof")==0).then(pl.col("mean_iof")*0.01)
        .otherwise(pl.col("std_iof")).alias("std_iof")
    )
    .sort(["ag","ig","eg"])
    .collect()
)
obs_b  = df_b["mean_iof"].to_numpy()
inc_b  = df_b["mean_incidence"].to_numpy()
emi_b  = df_b["mean_emission"].to_numpy()
pha_b  = df_b["mean_phase"].to_numpy()
print(f"  Broad bins: {len(df_b)}")

mb = HapkeModel(enable_shoe=True, enable_roughness=True, isotropic_h=True,
                fixed_parameters={"B0":1.03,"h":0.04},
                parameters={"w":0.43482,"g":-0.35198,"theta_bar":1.0,"B0":1.03,"h":0.04})
pred_b = mb._reflectance_numpy(GeometryBatch(
    incidence=np.deg2rad(inc_b), emission=np.deg2rad(emi_b), phase=np.deg2rad(pha_b)))
ratio_b = pred_b / np.maximum(obs_b, eps)

print(f"\n  {'inc bin':>9}  {'n':>5}  {'ratio':>8}  {'cv%':>7}")
bin5: list[tuple[int,int,int,float,float]] = []
for lo in range(0, 85, 5):
    hi = lo+5
    m = (inc_b>=lo)&(inc_b<hi)
    n = int(m.sum())
    if n == 0: continue
    rv = float(ratio_b[m].mean())
    cv = float(np.sqrt(np.mean((obs_b[m]-pred_b[m])**2)))/obs_b[m].mean()*100
    bin5.append((lo, hi, n, rv, cv))
    print(f"  i={lo:2d}–{hi:2d}°  {n:5d}  {rv:8.4f}  {cv:7.3f}%")

# Slope analysis
mild_i  = np.array([lo for lo,hi,n,rv,cv in bin5 if lo < 50], dtype=float)
mild_r  = np.array([rv  for lo,hi,n,rv,cv in bin5 if lo < 50])
steep_i = np.array([lo  for lo,hi,n,rv,cv in bin5 if lo >= 50], dtype=float)
steep_r = np.array([rv  for lo,hi,n,rv,cv in bin5 if lo >= 50])

slope_mild  = float(np.polyfit(mild_i,  mild_r,  1)[0])
slope_steep = float(np.polyfit(steep_i, steep_r, 1)[0])
rat_slopes  = slope_steep / slope_mild

# Continuity gap: extrapolate both lines to i=50
p_mild  = np.polyfit(mild_i,  mild_r,  1)
p_steep = np.polyfit(steep_i, steep_r, 1)
gap = abs(float(np.polyval(p_mild,50)) - float(np.polyval(p_steep,50)))

print(f"\n  Slope d(ratio)/d(i):")
print(f"    0–50°:  {slope_mild:+.5f} per degree  ({len(mild_i)} bins)")
print(f"    50–80°: {slope_steep:+.5f} per degree  ({len(steep_i)} bins)")
print(f"    Steep/mild: {rat_slopes:.2f}x")
print(f"    Continuity gap at i=50° (two-line extrapolation): {gap:.4f}")

print(f"\n  CONCLUSION — TEST B:")
if abs(rat_slopes) > 3.0:
    print(f"    TWO REGIMES (slope ratio {abs(rat_slopes):.1f}x).")
    print(f"    0–50°: mild Lommel-Seeliger disk-function trend ({slope_mild:+.5f}/deg)")
    print(f"    50–80°: steep terminator breakdown ({slope_steep:+.5f}/deg, {abs(rat_slopes):.1f}x steeper)")
    print(f"    These are distinct effects; the 50–80° region should be excluded")
    print(f"    from the disk-function characterization in the thesis.")
elif abs(rat_slopes) > 1.8:
    print(f"    GRADUAL STEEPENING (slope ratio {abs(rat_slopes):.1f}x).")
    print(f"    The trend accelerates above i~50° but without a sharp break.")
    print(f"    One effect (disk function) that becomes severe near the terminator.")
else:
    print(f"    SINGLE REGIME (slope ratio {abs(rat_slopes):.1f}x ≈ 1).")
    print(f"    Broadly uniform across 0–80°; one disk-function effect throughout.")
