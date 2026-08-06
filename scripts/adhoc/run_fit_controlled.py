"""
Three-case Li et al. fit on both controlled shape-model inputs, plus Step 3 report.

Reads pre-binned parquets produced by run_controlled_comparison.py:
  - binned_prelim_1pct.parquet    (pre-Dawn DSK, 1% sample, banker's rounding)
  - binned_110825_1pct.parquet   (110825 DSK,   1% sample, banker's rounding)
  - scatter_characterisation_110825.parquet  (full data, for Step 3)

The ONLY difference between the two fits is the .bds file used during geometry.
"""
from __future__ import annotations
import importlib.util, sys, json
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SILVER = ROOT / "data" / "silver/dsk256"

PRELIM_BIN  = SILVER / "binned_prelim_1pct.parquet"
D110_BIN    = SILVER / "binned_110825_1pct.parquet"
SCATTER_BIN = SILVER / "scatter_characterisation_110825.parquet"

for f in [PRELIM_BIN, D110_BIN, SCATTER_BIN]:
    if not f.exists():
        print(f"ERROR: {f} missing — run run_controlled_comparison.py first.", file=sys.stderr)
        sys.exit(1)

# Load run_baseline_fit helpers
spec = importlib.util.spec_from_file_location("rbf", ROOT / "scripts" / "run_baseline_fit.py")
rbf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbf)

def load_binned(path: Path) -> pl.DataFrame:
    """Load pre-binned parquet and apply zero-std guard (matches build_survey_loie)."""
    df = pl.read_parquet(str(path))
    return df.with_columns(
        pl.when(pl.col("std_iof") == 0)
        .then(pl.col("mean_iof") * 0.01)
        .otherwise(pl.col("std_iof"))
        .alias("std_iof")
    )

def run_fit(path: Path, label: str) -> dict:
    df = load_binned(path)
    rbf.build_survey_loie = lambda: df
    rbf.DATA_PATH = path
    print(f"\n{'='*68}")
    print(f"FIT: {label}")
    print(f"  Source: {path.name}")
    print(f"  Bins: {len(df)}")
    iof_arr = df["mean_iof"].to_numpy()
    std_arr = df["std_iof"].to_numpy()
    cv_arr  = std_arr / np.maximum(iof_arr, 1e-9)
    print(f"  mean_iof: {iof_arr.mean():.5f}   std_iof mean: {std_arr.mean():.5f}")
    print(f"  CV>1.0: {(cv_arr>1.0).sum()} bins  CV>0.5: {(cv_arr>0.5).sum()} bins")
    print(f"{'='*68}")

    c1 = rbf.run_case(
        name=f"case1_{label}",
        fixed_parameters={"B0": 1.03, "h": 0.04},
        parameter_bounds={"w": (0.3, 0.7), "g": (-0.6, 0.0), "theta_bar": (1.0, 50.0)},
    )
    c2 = rbf.run_case(
        name=f"case2_{label}",
        fixed_parameters={"B0": 1.03},
        parameter_bounds={"w": (0.3, 0.7), "g": (-0.6, 0.0),
                          "theta_bar": (1.0, 50.0), "h": (0.001, 1.0)},
    )
    c3 = rbf.run_case(
        name=f"case3_{label}",
        fixed_parameters={},
        parameter_bounds={"w": (0.3, 0.7), "g": (-0.6, 0.0),
                          "theta_bar": (1.0, 50.0), "B0": (0.0, 2.0), "h": (0.001, 1.0)},
    )
    return {"label": label, "n_bins": len(df),
            "mean_iof": float(iof_arr.mean()), "n_cv_gt1": int((cv_arr>1.0).sum()),
            "case1": c1, "case2": c2, "case3": c3}

prelim_result = run_fit(PRELIM_BIN, "prelim_DSK")
d110_result   = run_fit(D110_BIN,   "110825_DSK")

# ── STEP 3: Scatter characterisation ─────────────────────────────────────────
print(f"\n{'='*68}")
print("STEP 3 — Within-bin scatter characterisation (full 110825, 285M pixels)")
print("="*68)

sc = pq.read_table(str(SCATTER_BIN)).to_pydict()
iof_sc   = np.array(sc["mean_iof"])
std_sc   = np.array(sc["std_iof"])
std_inc  = np.array(sc["std_incidence"])  # within-bin local-incidence spread
std_emi  = np.array(sc["std_emission"])
npx_sc   = np.array(sc["n_pixels"])
cv_sc    = std_sc / np.maximum(iof_sc, 1e-9)

total_px = npx_sc.sum()
cv1_mask = cv_sc > 1.0
pct_bins = 100.0 * cv1_mask.sum() / len(cv_sc)
pct_px   = 100.0 * npx_sc[cv1_mask].sum() / total_px

print(f"\n  Total bins (i<50, e<50, n>=10): {len(iof_sc)}")
print(f"  Total pixels:                    {total_px:,}")
print(f"  Bins with CV>1.0:               {cv1_mask.sum()} / {len(iof_sc)} ({pct_bins:.1f}% of bins)")
print(f"  Pixels in CV>1.0 bins:          {npx_sc[cv1_mask].sum():,} ({pct_px:.1f}% of pixels)")

print(f"\n  Within-bin std(incidence) distribution (degrees):")
for pctile, label in [(25,"p25"), (50,"median"), (75,"p75"), (90,"p90"), (95,"p95"), (99,"p99")]:
    v = float(np.percentile(std_inc, pctile))
    print(f"    {label}: {v:.3f}°")
print(f"    mean: {std_inc.mean():.3f}°   max: {std_inc.max():.3f}°")
print(f"  (A 5°×5° bin has total width 5°; std > ~1.5° means sub-bin slope scatter")
print(f"   is comparable to the bin step size)")

print(f"\n  std(incidence) vs CV correlation:")
r = float(np.corrcoef(std_inc, cv_sc)[0, 1])
print(f"    r(std_inc, CV_iof) = {r:.4f}  [1 = scatter driven by incidence heterogeneity]")

# Report CV>1 bins with their local-incidence spread
if cv1_mask.sum() > 0:
    print(f"\n  CV>1 bins: local-incidence spread summary:")
    print(f"    std_incidence in CV>1 bins: "
          f"mean={std_inc[cv1_mask].mean():.2f}°  max={std_inc[cv1_mask].max():.2f}°")
    print(f"    std_incidence in CV≤1 bins: "
          f"mean={std_inc[~cv1_mask].mean():.2f}°  max={std_inc[~cv1_mask].max():.2f}°")

# ── SIDE-BY-SIDE COMPARISON TABLE ─────────────────────────────────────────────
print(f"\n{'='*68}")
print("CONTROLLED COMPARISON — Case 1 (B0=1.03, h=0.04 fixed, 100 starts)")
print(f"  Only variable: shape model .bds file")
print("="*68)

def fp(r, key):
    return r["fitted_parameters"][key]
def sp(r, key):
    return r["multistart_spread"][key]["std"]

p1 = prelim_result["case1"]; d1 = d110_result["case1"]
print(f"\n  {'':20} {'Pre-Dawn DSK':>16} {'110825 DSK':>14} {'Δ':>10}")
print(f"  {'n_bins':20} {prelim_result['n_bins']:>16} {d110_result['n_bins']:>14} {d110_result['n_bins']-prelim_result['n_bins']:>10}")
print(f"  {'mean_iof (bins)':20} {prelim_result['mean_iof']:>16.5f} {d110_result['mean_iof']:>14.5f} {d110_result['mean_iof']-prelim_result['mean_iof']:>10.5f}")
print(f"  {'CV>1 bins':20} {prelim_result['n_cv_gt1']:>16} {d110_result['n_cv_gt1']:>14} {d110_result['n_cv_gt1']-prelim_result['n_cv_gt1']:>10}")
print(f"  {'w':20} {fp(p1,'w'):>16.5f} {fp(d1,'w'):>14.5f} {fp(d1,'w')-fp(p1,'w'):>+10.5f}")
print(f"  {'w spread (σ)':20} {sp(p1,'w'):>16.5f} {sp(d1,'w'):>14.5f}")
print(f"  {'g':20} {fp(p1,'g'):>16.5f} {fp(d1,'g'):>14.5f} {fp(d1,'g')-fp(p1,'g'):>+10.5f}")
print(f"  {'g spread (σ)':20} {sp(p1,'g'):>16.5f} {sp(d1,'g'):>14.5f}")
print(f"  {'theta_bar (deg)':20} {fp(p1,'theta_bar'):>16.5f} {fp(d1,'theta_bar'):>14.5f} {fp(d1,'theta_bar')-fp(p1,'theta_bar'):>+10.5f}")
print(f"  {'theta_bar spread (σ)':20} {sp(p1,'theta_bar'):>16.5f} {sp(d1,'theta_bar'):>14.5f}")
print(f"  {'fRMS (%)':20} {p1['fractional_rms_pct']:>16.3f} {d1['fractional_rms_pct']:>14.3f} {d1['fractional_rms_pct']-p1['fractional_rms_pct']:>+10.3f}")

print(f"\n{'='*68}")
print("FULL JSON OUTPUT")
print("="*68)
payload = {"controlled_comparison": True,
           "sampling": "1% Bernoulli seed=42",
           "binning": "DuckDB banker's rounding",
           "image_set": "Survey F1B (845 images common to both DSK geometry tables)",
           "prelim": prelim_result, "d110825": d110_result}
print(json.dumps(payload, indent=2, sort_keys=True))
