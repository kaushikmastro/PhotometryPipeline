"""Residual-decomposition diagnostics: reproduce results A and C, and recompute A properly.

Run via sbatch only (CLAUDE.md RULE 4); ~30 s, <32G on one node:
    sbatch ... --wrap 'python scripts/diagnostics/residual_decomposition.py'
Verified 2026-10-21, job 26685858: every logged number below reproduces exactly.

Pinned inputs
-------------
data/silver/dsk256/survey_5pct.parquet
    33,500,645 rows, written 2026-05-31 19:17 by diag_sampling_noise.py (job 25752165):
    Bernoulli 5% (seed 42) of 04_geometry_tables_fast/survey/*.parquet (now
    geometry/gaskell_256/survey/, preliminary Gaskell DSK256), i<80, e<80, iof>0.01.
data/silver/dsk256/survey_7x7_geometry.parquet
    47,271 Survey rows (the "1%" baseline in C -- it is NOT the 6.7M-row 1% sample).
Binning: 5x5x5 deg (phase, incidence, emission), polars round() of value/5.
Trend partition: 10-deg incidence bins on the bin-mean incidence (diag_decomp_testB).

A  r = +0.8101   diag_albedo_heterogeneity.py Step 3, job 25757092 (2026-06-01)
   corr(std/mean, (max-min)/mean) of OBSERVED I/F across 990 bins with n>=100
   (pixel-level i<50, e<50, phase<80). No model, no CV_other: both axes share the
   1/mean_iof factor; absolute corr(std, max-min) is +0.3347.
A' corrected: per-bin |r_other| (committed Case 1: w=0.46993 g=-0.33688
   theta_bar=8.2662 B0=1.03 h=0.04, Hapke-2002 H) vs IQR, absolute and relative,
   Pearson/Spearman, and partial correlation controlling for log10(n_pixels).
C  noise fraction 0.22   diag_sampling_noise.py "Test A", job 25752165 (2026-05-31)
   baseline 7x7: iso-H Case 1 w=0.45940 g=-0.33649 theta_bar=4.0742
     (hapke_iso_h_25751773.out); 906 bins, n>=10, bin-mean i,e<50, phase<80.
   5% sample: iso-H w=0.45814 g=-0.33562 theta_bar=4.0056 (100-start fit logged
     in job 25752165; taken from the log here, not re-fitted); 926 bins, same cuts.
   expected drop = CV_other_7x7 * (1 - 1/sqrt(5)); noise fraction = actual/expected.

Not reproduced: CV_RMSE 9.223% = 5.125% (+) 7.669% (result B). Those numbers
appear in no job output, notebook or IPython history in this repo; their first
appearance is CLAUDE.md in commit 9d113ac (2026-08-03).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import polars as pl
from scipy import stats

from photometry.core.types import GeometryBatch
from photometry.models.hapke import HapkeModel

DATA_5PCT = ROOT / "data/silver/dsk256/survey_5pct.parquet"
DATA_7X7 = ROOT / "data/silver/dsk256/survey_7x7_geometry.parquet"
GRID = 5.0

spec = importlib.util.spec_from_file_location("rbf", ROOT / "scripts/utils/run_baseline_fit.py")
rbf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbf)


def hapke(params, inc, emi, pha, isotropic_h):
    m = HapkeModel(enable_shoe=True, enable_roughness=True, isotropic_h=isotropic_h,
                   fixed_parameters={"B0": 1.03, "h": 0.04},
                   parameters={**params, "B0": 1.03, "h": 0.04})
    return m._reflectance_numpy(GeometryBatch(
        incidence=np.deg2rad(inc), emission=np.deg2rad(emi), phase=np.deg2rad(pha)))


def orthogonal_decomp(obs, inc, pred):
    """Verbatim logic of diag_sampling_noise.orthogonal_decomp / diag_decomp_testB."""
    r = pred - obs
    r_trend = np.zeros_like(r)
    bins = []
    for lo in range(0, 60, 10):
        m = (inc >= lo) & (inc < lo + 10)
        if m.sum() > 0:
            bm = float(r[m].mean()); r_trend[m] = bm; bins.append((int(m.sum()), bm))
    r_other = r - r_trend
    cv = lambda x: float(np.sqrt(np.mean(x**2))) / obs.mean() * 100
    gm = float(r.mean())
    R2 = sum(n * (bm - gm) ** 2 for n, bm in bins) / float(np.sum((r - gm) ** 2))
    return cv(r), cv(r_trend), cv(r_other), R2, r_other


def partial(x, y, z):
    """Correlation of x and y after removing a linear dependence on z from each."""
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def corr_row(label, x, y, logn):
    pr = float(np.corrcoef(x, y)[0, 1])
    sr = float(stats.spearmanr(x, y).statistic)
    pp = partial(x, y, logn)
    sp = partial(stats.rankdata(x), stats.rankdata(y), stats.rankdata(logn))
    print(f"  {label:<44} {pr:+8.4f} {sr:+9.4f} {pp:+12.4f} {sp:+13.4f}")


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("A — REPRODUCE r(CV_iof, relative I/F range), diag_albedo_heterogeneity Step 3")
print("=" * 78)
px = (
    pl.scan_parquet(DATA_5PCT)
    .filter(pl.col("mission_phase") == "SURVEY")
    .select(["iof", "incidence", "emission", "phase"])
    .filter((pl.col("incidence") < 50.0) & (pl.col("emission") < 50.0) & (pl.col("phase") < 80.0))
    .with_columns([
        ((pl.col("phase") / GRID).round() * GRID).alias("alpha_grid"),
        ((pl.col("incidence") / GRID).round() * GRID).alias("i_grid"),
        ((pl.col("emission") / GRID).round() * GRID).alias("e_grid"),
    ])
)
bins = (
    px.group_by(["alpha_grid", "i_grid", "e_grid"])
    .agg([
        pl.len().alias("n"),
        pl.col("iof").mean().alias("mean_iof"),
        pl.col("iof").std().alias("std_iof"),
        pl.col("iof").max().alias("max_iof"),
        pl.col("iof").min().alias("min_iof"),
        pl.col("iof").median().alias("median_iof"),
        (pl.col("iof").quantile(0.75, "linear") - pl.col("iof").quantile(0.25, "linear")).alias("iqr_iof"),
        pl.col("incidence").mean().alias("mean_incidence"),
        pl.col("emission").mean().alias("mean_emission"),
        pl.col("phase").mean().alias("mean_phase"),
    ])
    .collect()
    .sort(["alpha_grid", "i_grid", "e_grid"])
)
b100 = bins.filter(pl.col("n") >= 100)
std = b100["std_iof"].to_numpy(); mean = b100["mean_iof"].to_numpy()
rng_ = (b100["max_iof"] - b100["min_iof"]).to_numpy(); n = b100["n"].to_numpy()
cv_iof = std / np.maximum(mean, 1e-9); rel_range = rng_ / np.maximum(mean, 1e-9)
print(f"  pixels in domain: {px.select(pl.len()).collect().item():,}   bins n>=100: {len(b100)}")
print(f"  r(std_iof, I/F range)          = {np.corrcoef(std, rng_)[0,1]:+.4f}   (log: +0.3347)")
print(f"  r(std_iof, mean_iof)           = {np.corrcoef(std, mean)[0,1]:+.4f}   (log: -0.4047)")
print(f"  r(CV_iof, relative I/F range)  = {np.corrcoef(cv_iof, rel_range)[0,1]:+.4f}   (log: +0.8101)")
print("  Shared-denominator check (both sides divided by mean_iof):")
print(f"    partial r(CV, rel_range | log mean_iof) = {partial(cv_iof, rel_range, np.log(mean)):+.4f}")
print(f"    r(1/mean_iof, CV) = {np.corrcoef(1/mean, cv_iof)[0,1]:+.4f}   "
      f"r(1/mean_iof, rel_range) = {np.corrcoef(1/mean, rel_range)[0,1]:+.4f}")

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("A' — CORRECTED: per-bin |r_other| vs within-bin I/F spread, same data")
print("  r = obs - pred at bin-mean geometry; r_other = r - mean(r) in 10° incidence bins")
print("  model: committed Case 1 (w=0.46993 g=-0.33688 θ̄=8.2662 B0=1.03 h=0.04, Hapke-2002 H)")
print("=" * 78)
CASE1 = {"w": 0.46993, "g": -0.33688, "theta_bar": 8.2662}
for label, sel in [("n>=100 (A's bin set)", b100), ("n>=10 (fit's MIN_PIXELS)", bins.filter(pl.col("n") >= 10))]:
    obs = sel["mean_iof"].to_numpy(); inc = sel["mean_incidence"].to_numpy()
    pred = hapke(CASE1, inc, sel["mean_emission"].to_numpy(), sel["mean_phase"].to_numpy(), False)
    *_, r_other = orthogonal_decomp(obs, inc, pred)
    a = np.abs(r_other); iqr = sel["iqr_iof"].to_numpy(); med = sel["median_iof"].to_numpy()
    logn = np.log10(sel["n"].to_numpy().astype(float))
    print(f"\n  [{label}]  N={len(sel)}  median n_pixels={int(np.median(sel['n'].to_numpy()))}")
    print(f"  {'':<44} {'Pearson':>8} {'Spearman':>9} {'partial_P|n':>12} {'partial_S|n':>13}")
    corr_row("abs: |r_other| vs IQR", a, iqr, logn)
    corr_row("rel: |r_other|/mean_iof vs IQR/median", a / obs, iqr / med, logn)
    corr_row("rel: |r_other|/mean_iof vs std/mean (CV)", a / obs, sel["std_iof"].to_numpy() / obs, logn)

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("C — REPRODUCE '5x resampling' Test A, diag_sampling_noise.py (job 25752165)")
print("=" * 78)
rbf.DATA_PATH = DATA_7X7
d1 = rbf.build_survey_loie()
d1 = d1.filter(pl.Series(d1["mean_phase"].to_numpy() < 80.0))
ISO1 = {"w": 0.45940, "g": -0.33649, "theta_bar": 4.0742}  # hapke_iso_h_25751773.out
o1 = d1["mean_iof"].to_numpy(); i1 = d1["mean_incidence"].to_numpy()
p1 = hapke(ISO1, i1, d1["mean_emission"].to_numpy(), d1["mean_phase"].to_numpy(), True)
c1 = orthogonal_decomp(o1, i1, p1)
n7 = pl.scan_parquet(DATA_7X7).filter(pl.col("mission_phase") == "SURVEY").select(pl.len()).collect().item()
print(f"  baseline '1%' = survey_7x7_geometry.parquet: {n7:,} Survey pixels, {len(d1)} bins, "
      f"median n_pixels={int(d1['n_pixels'].median())}")
print(f"    CV_total={c1[0]:.4f}%  CV_trend={c1[1]:.4f}%  CV_other={c1[2]:.4f}%  R²={c1[3]:.4f}"
      f"   (log: 9.7620 / 5.2988 / 8.1987 / 0.2883)")

# diag_sampling_noise bins ALL rows (i,e<80 from the extraction) and then filters on bin means
d5 = (
    pl.scan_parquet(DATA_5PCT)
    .with_columns([((pl.col(c) / GRID).round() * GRID).alias(a)
                   for c, a in [("phase", "alpha_grid"), ("incidence", "i_grid"), ("emission", "e_grid")]])
    .group_by(["alpha_grid", "i_grid", "e_grid"])
    .agg([pl.col("incidence").mean().alias("mean_incidence"),
          pl.col("emission").mean().alias("mean_emission"),
          pl.col("phase").mean().alias("mean_phase"),
          pl.col("iof").mean().alias("mean_iof"),
          pl.col("iof").std().alias("std_iof"),
          pl.col("iof").count().alias("n_pixels")])
    .filter((pl.col("n_pixels") >= 10) & (pl.col("mean_incidence") < 50.0)
            & (pl.col("mean_emission") < 50.0) & (pl.col("mean_phase") < 80.0))
    .drop_nulls(subset=["std_iof"])
    .sort(["alpha_grid", "i_grid", "e_grid"])
    .collect()
)
ISO5 = {"w": 0.45814, "g": -0.33562, "theta_bar": 4.0056}  # 5% fit, from job 25752165 log
o5 = d5["mean_iof"].to_numpy(); i5 = d5["mean_incidence"].to_numpy()
p5 = hapke(ISO5, i5, d5["mean_emission"].to_numpy(), d5["mean_phase"].to_numpy(), True)
c5 = orthogonal_decomp(o5, i5, p5)
print(f"  5% sample: {len(d5)} bins (log: 926), mean n_pixels={d5['n_pixels'].mean():.0f} (log: 18322)")
print(f"    CV_total={c5[0]:.4f}%  CV_trend={c5[1]:.4f}%  CV_other={c5[2]:.4f}%  R²={c5[3]:.4f}"
      f"   (log: 8.8665 / 5.1558 / 7.2134 / 0.3372)")
actual = c1[2] - c5[2]; expected = c1[2] - c1[2] / np.sqrt(5)
print(f"  CV_other drop {actual:.4f}%  expected {expected:.4f}%  noise fraction {actual/expected:.2f}  (log: 0.22)")

print("\n  Context — what per-bin-mean sampling noise actually contributes (std/sqrt(n)):")
for lbl, d in [("7x7 baseline", d1), ("5% sample", d5)]:
    se = d["std_iof"].to_numpy() / np.sqrt(d["n_pixels"].to_numpy())
    print(f"    {lbl:<13} RMS(SE of bin mean)/mean_obs = "
          f"{np.sqrt(np.mean(se**2)) / d['mean_iof'].to_numpy().mean() * 100:.4f}%")
print(f"    pixel-count ratio 5%/7x7 = {d5['n_pixels'].sum() / d1['n_pixels'].sum():.0f}x (test assumed 5x)")
