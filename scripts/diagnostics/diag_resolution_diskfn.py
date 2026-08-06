"""
Resolution and disk-function diagnostic.

Step 1: Compare incidence-binned pred/obs ratio between
        18m DSK256 geometry vs 7x7 averaged geometry.
        Both use isotropic-H Case 1 parameters on the
        phase<80 Survey-only low-i/low-e subset.

Step 2: Per-pixel incidence distribution — 18m single-ray vs
        7x7 averaged — for the same (image_id, pixel_x, pixel_y).

Step 3: Minnaert disk function diagnostic (DIAGNOSTIC ONLY —
        departs from Hapke IMSA framework, not a production model).
        Replace mu0/(mu0+mu) with mu0^k * mu^(k-1), fit w, g,
        theta_bar, k on 7x7 phase<80 subset. Report k and whether
        the incidence residual flattens.
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
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location("run_baseline_fit",
    ROOT / "scripts" / "run_baseline_fit.py")
rbf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbf)

from photometry.models.hapke import HapkeModel
from photometry.fitting.least_sq import LeastSquaresFitter
from photometry.core.types import GeometryBatch
from photometry import committed_params as CP

DATA_18M = ROOT / "data" / "silver/dsk256" / "combined_rc_survey_sample_corrected_dsk256.parquet"
DATA_7X7 = ROOT / "data" / "silver/dsk256" / "survey_7x7_geometry.parquet"
eps = 1e-12

def build_phase80(data_path):
    rbf.DATA_PATH = data_path
    df = rbf.build_survey_loie()
    m = df["mean_phase"].to_numpy() < 80.0
    return df.filter(pl.Series(m))

# ── STEP 1: 18m vs 7x7 incidence trend ───────────────────────────────────────
print("=" * 68)
print("STEP 1 — Incidence-binned ratio: 18m vs 7x7 geometry (Hapke-2002 H)")
print("=" * 68)

# --- 18m: fit Case 1 isotropic-H on phase<80 Survey-only subset ---
print("\nFitting Case 1 isotropic-H on 18m data (phase<80°, 100 starts) …")
df_18m = build_phase80(DATA_18M)
obs18  = df_18m["mean_iof"].to_numpy()
inc18  = df_18m["mean_incidence"].to_numpy()
emi18  = df_18m["mean_emission"].to_numpy()
pha18  = df_18m["mean_phase"].to_numpy()
std18  = df_18m["std_iof"].to_numpy()
geom18 = GeometryBatch(incidence=np.deg2rad(inc18),
                       emission=np.deg2rad(emi18),
                       phase=np.deg2rad(pha18))
wts18  = 1.0 / np.clip(std18, 1e-9, None)

rng = np.random.default_rng(42)
fitter = LeastSquaresFitter()
W_B, G_B, T_B = (0.3,0.7), (-0.6,0.0), (1.0,50.0)
best18_cost = np.inf; best18_fp = None

for _ in range(100):
    m = HapkeModel(enable_shoe=True, enable_roughness=True,
                   fixed_parameters={"B0":1.03,"h":0.04},
                   parameters={"w":float(rng.uniform(*W_B)),"g":float(rng.uniform(*G_B)),
                                "theta_bar":float(rng.uniform(*T_B)),"B0":1.03,"h":0.04})
    ob = m.parameter_bounds
    def _b(o=ob): b=o(); b.update({"w":W_B,"g":G_B,"theta_bar":T_B}); return b
    m.parameter_bounds = _b
    r = fitter.fit(m, geom18, obs18, weights=wts18)
    if r.objective_value < best18_cost:
        best18_cost = r.objective_value; best18_fp = dict(r.fitted_parameters)

w18,g18,tb18 = best18_fp["w"], best18_fp["g"], best18_fp["theta_bar"]
m18 = HapkeModel(enable_shoe=True, enable_roughness=True,
                 fixed_parameters={"B0":1.03,"h":0.04},
                 parameters={**best18_fp,"B0":1.03,"h":0.04})
pred18 = m18._reflectance_numpy(geom18)
cv18 = float(np.sqrt(np.mean((obs18-pred18)**2))) / obs18.mean() * 100
print(f"  18m: w={w18:.5f}  g={g18:.5f}  θ̄={tb18:.3f}°  CV-RMSE={cv18:.3f}%  bins={len(df_18m)}")

# --- 7x7: 100-start fit (same protocol as 18m above, Hapke-2002 H) ---
print("\nFitting Case 1 Hapke-2002 H on 7x7 data (phase<80°, 100 starts) …")
df_7x7 = build_phase80(DATA_7X7)
obs7   = df_7x7["mean_iof"].to_numpy()
inc7   = df_7x7["mean_incidence"].to_numpy()
emi7   = df_7x7["mean_emission"].to_numpy()
pha7   = df_7x7["mean_phase"].to_numpy()
std7   = df_7x7["std_iof"].to_numpy()
geom7  = GeometryBatch(incidence=np.deg2rad(inc7),
                       emission=np.deg2rad(emi7),
                       phase=np.deg2rad(pha7))
wts7   = 1.0 / np.clip(std7, 1e-9, None)

rng7 = np.random.default_rng(42)
best7_cost = np.inf; best7_fp = None

for _ in range(100):
    m = HapkeModel(enable_shoe=True, enable_roughness=True,
                   fixed_parameters=CP.FIXED,
                   parameters={"w": float(rng7.uniform(*W_B)),
                                "g": float(rng7.uniform(*G_B)),
                                "theta_bar": float(rng7.uniform(*T_B)),
                                **CP.FIXED})
    ob = m.parameter_bounds
    def _b7(o=ob): b=o(); b.update({"w":W_B,"g":G_B,"theta_bar":T_B}); return b
    m.parameter_bounds = _b7
    r = fitter.fit(m, geom7, obs7, weights=wts7)
    if r.objective_value < best7_cost:
        best7_cost = r.objective_value; best7_fp = dict(r.fitted_parameters)

w7, g7, tb7 = best7_fp["w"], best7_fp["g"], best7_fp["theta_bar"]
m7 = HapkeModel(enable_shoe=True, enable_roughness=True,
                fixed_parameters=CP.FIXED,
                parameters={**best7_fp, **CP.FIXED})
pred7 = m7._reflectance_numpy(geom7)
cv7   = float(np.sqrt(np.mean((obs7-pred7)**2))) / obs7.mean() * 100
print(f"  7x7: w={w7:.5f}  g={g7:.5f}  θ̄={tb7:.3f}°  CV-RMSE={cv7:.3f}%  bins={len(df_7x7)}")
print(f"  Δθ̄ (7x7 − 18m) = {tb7 - tb18:+.3f}°  — REAL independent fit (not hardcoded)")

print(f"\n  {'inc bin':>9}  {'n_18m':>6}  {'ratio_18m':>10}  {'CV_18m%':>9}  "
      f"  {'n_7x7':>6}  {'ratio_7x7':>10}  {'CV_7x7%':>9}  {'Δratio':>8}")
for lo in range(0, 60, 10):
    hi = lo + 10
    m18_ = (inc18>=lo)&(inc18<hi); m7_ = (inc7>=lo)&(inc7<hi)
    n18_ = int(m18_.sum()); n7_ = int(m7_.sum())
    if n18_==0 and n7_==0: continue
    r18_ = pred18[m18_].mean()/obs18[m18_].mean() if n18_>0 else float("nan")
    r7_  = pred7[m7_].mean()/obs7[m7_].mean()     if n7_>0  else float("nan")
    cv18_ = float(np.sqrt(np.mean((obs18[m18_]-pred18[m18_])**2)))/obs18[m18_].mean()*100 if n18_>0 else float("nan")
    cv7_  = float(np.sqrt(np.mean((obs7[m7_]-pred7[m7_])**2)))/obs7[m7_].mean()*100       if n7_>0  else float("nan")
    dr = r7_ - r18_
    print(f"  i={lo:2d}–{hi:2d}°  {n18_:6d}  {r18_:10.4f}  {cv18_:9.3f}%  "
          f"  {n7_:6d}  {r7_:10.4f}  {cv7_:9.3f}%  {dr:+8.4f}")

print(f"\n  Interpretation: if Δratio ≈ 0, the incidence trend persists after 7x7 smoothing")
print(f"  → fine-facet normal aliasing is NOT the cause.")

# ── STEP 2: per-pixel incidence scatter 18m vs 7x7 ───────────────────────────
print(f"\n{'=' * 68}")
print("STEP 2 — Per-pixel incidence: 18m single-ray vs 7x7 averaged")
print("         Matched on (image_id, pixel_x, pixel_y)")
print("=" * 68)

df7_px = pl.read_parquet(str(DATA_7X7)).select(
    ["image_id","pixel_x","pixel_y","incidence","emission"]).rename(
    {"incidence":"inc_7x7","emission":"emi_7x7"})
df18_px = (
    pl.scan_parquet(str(DATA_18M))
    .filter(pl.col("mission_phase")=="SURVEY")
    .select(["image_id","pixel_x","pixel_y","incidence","emission"])
    .collect()
    .rename({"incidence":"inc_18m","emission":"emi_18m"})
)
joined = df7_px.join(df18_px, on=["image_id","pixel_x","pixel_y"], how="inner")
print(f"  Matched pixels: {len(joined):,} / {len(df7_px):,}")

di = (joined["inc_7x7"] - joined["inc_18m"]).to_numpy()
de = (joined["emi_7x7"] - joined["emi_18m"]).to_numpy()
inc18_px = joined["inc_18m"].to_numpy()

print(f"\n  Incidence (7x7 − 18m) per pixel:")
print(f"    mean:    {di.mean():+.4f}°  (systematic bias from sub-pixel averaging)")
print(f"    std:     {di.std():.4f}°")
print(f"    |Δ| mean: {np.abs(di).mean():.4f}°")
print(f"    |Δ| p50:  {np.percentile(np.abs(di),50):.4f}°")
print(f"    |Δ| p95:  {np.percentile(np.abs(di),95):.4f}°")
print(f"    |Δ| max:  {np.abs(di).max():.4f}°")
print(f"  Emission (7x7 − 18m) per pixel:")
print(f"    mean:    {de.mean():+.4f}°  std={de.std():.4f}°  |Δ| mean={np.abs(de).mean():.4f}°")

print(f"\n  |Δinc| distribution:")
for lo_d, hi_d in [(0,1),(1,2),(2,5),(5,10),(10,20),(20,90)]:
    n = int(((np.abs(di)>=lo_d)&(np.abs(di)<hi_d)).sum())
    print(f"    Δ={lo_d:2d}–{hi_d:2d}°:  {n:6d}  ({100*n/len(di):5.1f}%)")

corr_di_i18 = float(np.corrcoef(inc18_px, np.abs(di))[0,1])
print(f"\n  r(|Δinc|, inc_18m) = {corr_di_i18:+.4f}")
print(f"  [if >0: larger sub-pixel scatter at higher incidence, i.e. near terminator]")

# ── STEP 3: Minnaert diagnostic ───────────────────────────────────────────────
print(f"\n{'=' * 68}")
print("STEP 3 — Minnaert disk function DIAGNOSTIC  [not production model]")
print("  I/F = (w/4) * mu0^k * mu^(k-1) * [(1+B)*P + H(mu0)*H(mu) - 1] * S")
print("  k=0.5 → Lommel-Seeliger;  k=1.0 → Lambert")
print("=" * 68)

print(f"\n  Dataset: 7x7 phase<80°, {len(df_7x7)} bins. Free: w, g, theta_bar, k.")

std7 = df_7x7["std_iof"].to_numpy()
wts7 = 1.0 / np.clip(std7, 1e-9, None)
mu0_m = np.cos(np.deg2rad(inc7))
mu_m  = np.cos(np.deg2rad(emi7))
pha_r = np.deg2rad(pha7)

def minnaert_iof(params, mu0, mu, phase):
    """Hapke IMSA, isotropic-H, Minnaert disk. DIAGNOSTIC ONLY."""
    w, g, theta_bar, k = float(params[0]),float(params[1]),float(params[2]),float(params[3])
    w = np.clip(w,0.01,0.99); theta_bar=np.clip(theta_bar,0.01,60.0); k=np.clip(k,0.01,1.5)
    inc_r = np.arccos(np.clip(mu0,eps,1.0)); emi_r = np.arccos(np.clip(mu,eps,1.0))
    cos_a = np.cos(phase)
    P = (1.0-g**2) / np.power(1.0+2.0*g*cos_a+g**2, 1.5)
    gm = np.sqrt(np.clip(1.0-w,0.0,1.0))
    H0 = (1.0+2.0*mu0)/(1.0+2.0*gm*mu0)
    Hm = (1.0+2.0*mu) /(1.0+2.0*gm*mu)
    B  = 1.03/(1.0+(1.0/0.04)*np.tan(phase/2.0))
    # Roughness S
    tr = np.deg2rad(theta_bar); tt = np.tan(tr)
    chi = 1.0/np.sqrt(1.0+np.pi*tt**2)
    cp = (cos_a-mu0*mu)/np.clip(np.sin(inc_r)*np.sin(emi_r),eps,None)
    psi = np.arccos(np.clip(cp,-1.0,1.0)); fp = np.exp(-2.0*np.tan(psi/2.0))
    ti = np.clip(np.tan(inc_r),eps,None); te = np.clip(np.tan(emi_r),eps,None)
    E1i=np.exp(-2.0/(np.pi*tt*ti)); E1e=np.exp(-2.0/(np.pi*tt*te))
    E2i=np.exp(-1.0/(np.pi*tt**2*ti**2)); E2e=np.exp(-1.0/(np.pi*tt**2*te**2))
    si=np.sin(inc_r); se=np.sin(emi_r); s2p=np.sin(psi/2.0)**2
    ile = inc_r<=emi_r
    mu0e=np.where(ile,chi*(mu0+si*tt*(E2e+s2p*E2i)),chi*(mu0+si*tt*(E2e-s2p*E2e)))
    mue =np.where(ile,chi*(mu +se*tt*(E2i-s2p*E2i)),chi*(mu +se*tt*(E2i+s2p*E2e)))
    sd  =np.where(ile,1.0-fp*E1i-(1.0-fp)*E1e,     1.0-fp*E1e-(1.0-fp)*E1i)
    S   = (mue/np.clip(mu,eps,None))*(mu0/np.clip(mu0e,eps,None))*chi/np.clip(sd,eps,None)
    # Minnaert disk
    disk = np.power(np.clip(mu0,eps,None),k)*np.power(np.clip(mu,eps,None),k-1.0)
    iof  = (w/4.0)*disk*((1.0+B)*P+H0*Hm-1.0)*S
    return np.where((mu0>0)&(mu>0)&np.isfinite(iof), iof, 0.0)

def resid_minn(params):
    return (minnaert_iof(params,mu0_m,mu_m,pha_r) - obs7) * wts7

best_mn_cost = np.inf; best_mn_p = None; all_k = []
for _ in range(100):
    p0 = [float(rng.uniform(0.3,0.7)), float(rng.uniform(-0.6,0.0)),
          float(rng.uniform(1.0,30.0)), float(rng.uniform(0.1,1.5))]
    try:
        res = least_squares(resid_minn, p0,
                            bounds=([0.3,-0.6,1.0,0.1],[0.7,0.0,50.0,1.5]),
                            method="trf", loss="linear", diff_step=1e-4)
        all_k.append(res.x[3])
        if res.cost < best_mn_cost:
            best_mn_cost = res.cost; best_mn_p = res.x.copy()
    except Exception:
        pass

if best_mn_p is not None:
    w_mn,g_mn,tb_mn,k_mn = best_mn_p
    pred_mn = minnaert_iof(best_mn_p, mu0_m, mu_m, pha_r)
    cv_mn = float(np.sqrt(np.mean((obs7-pred_mn)**2)))/obs7.mean()*100
    fr_mn = float(np.sqrt(np.mean(((obs7-pred_mn)/np.maximum(obs7,eps))**2)))*100
    k_arr = np.array(all_k)
    print(f"\n  Best-by-cost:")
    print(f"    w         = {w_mn:.5f}")
    print(f"    g         = {g_mn:.5f}")
    print(f"    theta_bar = {tb_mn:.4f}°")
    print(f"    k         = {k_mn:.5f}  [LS=0.5, Lambert=1.0]")
    print(f"    CV-RMSE   = {cv_mn:.3f}%   (vs iso-H Case1 LS: 9.762%)")
    print(f"    frac-RMS  = {fr_mn:.3f}%")
    print(f"  100-start k: mean={k_arr.mean():.4f}  std={k_arr.std():.4f}  "
          f"min={k_arr.min():.4f}  max={k_arr.max():.4f}")

    print(f"\n  Incidence-binned ratio: LS (7x7) vs Minnaert k={k_mn:.3f} (7x7):")
    print(f"  {'inc bin':>9}  {'n':>5}  {'ratio_LS':>10}  {'ratio_Minn':>12}  {'Δratio':>8}  {'CV_Minn%':>10}")
    ls_ref = {0:1.0821, 10:1.0698, 20:1.0469, 30:1.0025, 40:0.9208}
    for lo in range(0, 60, 10):
        hi = lo+10; mm = (inc7>=lo)&(inc7<hi); n=int(mm.sum())
        if n==0: continue
        r_mn = pred_mn[mm].mean()/obs7[mm].mean()
        cv_b = float(np.sqrt(np.mean((obs7[mm]-pred_mn[mm])**2)))/obs7[mm].mean()*100
        r_ls = ls_ref.get(lo, float("nan"))
        print(f"  i={lo:2d}–{hi:2d}°  {n:5d}  {r_ls:10.4f}  {r_mn:12.4f}  "
              f"{r_mn-r_ls:+8.4f}  {cv_b:10.3f}%")
else:
    print("  Minnaert fit failed.")
