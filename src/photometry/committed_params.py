"""
Single source of truth for committed Case 1 Hapke parameters.

Provenance
----------
- Source log  : prelim_physfilter_25799987.out, Config A
- Data        : full Survey F1B, 682M pixels, preliminary Gaskell DSK256
- Filter      : iof > 0.01 (validated shadow exclusion), i < 50°, e < 50°
- Geometry    : single-ray DSK/UNPRIORITIZED per pixel (04_geometry_tables_fast)
- Bins        : 950 (5° × 5° × 5° phase/inc/emi grid, banker's rounding, n >= 10)
- H-function  : Hapke-2002 approximation (isotropic_h=False)
- B0, h       : fixed (not free parameters)
- Multi-start : 100 starts, seed 42; spread std(θ̄) = 0.00006° → deterministic
"""

# ── Fitted free parameters ────────────────────────────────────────────────────
W          = 0.46993   # single-scattering albedo
G          = -0.33688  # Henyey-Greenstein asymmetry
THETA_BAR  = 8.2662    # macroscopic roughness (degrees)

# ── Fixed parameters ──────────────────────────────────────────────────────────
B0         = 1.03      # SHOE amplitude
H_SHOE     = 0.04      # SHOE angular half-width

# ── H-function flag ───────────────────────────────────────────────────────────
ISOTROPIC_H = False    # False = Hapke-2002; True = Li 2013 IMSA (deprecated)

# ── Fit quality (Config A) ────────────────────────────────────────────────────
FRMS        = 10.902   # fractional RMS (%)
REDUCED_CHI2 = 0.200   # reduced chi-squared
N_BINS      = 950

# ── Convenience dicts for HapkeModel ─────────────────────────────────────────
PARAMS = {"w": W, "g": G, "theta_bar": THETA_BAR, "B0": B0, "h": H_SHOE}
FIXED  = {"B0": B0, "h": H_SHOE}