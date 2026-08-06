#!/usr/bin/env bash
#SBATCH --job-name=lamo_validate
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=32G --time=01:00:00
#SBATCH --partition=main --qos=standard
#SBATCH --output=logs/lamo_validate_%j.out

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env
source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

LAMO="/scratch/kaushim07/vesta_data/04_geometry_tables_fast/lamo/*.parquet"
mkdir -p /scratch/kaushim07/duckdb_tmp

python3 - << 'PYEOF'
import duckdb, os

LAMO = "/scratch/kaushim07/vesta_data/04_geometry_tables_fast/lamo/*.parquet"
os.makedirs("/scratch/kaushim07/duckdb_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("SET memory_limit='28GB'; SET temp_directory='/scratch/kaushim07/duckdb_tmp'; SET threads=4;")

print("=== Query 1: all non-sort aggregates ===", flush=True)
r = con.execute(f"""
  SELECT
    COUNT(*)                 AS n_pixels,
    COUNT(DISTINCT image_id) AS n_images,
    AVG(iof)                 AS mean_iof,
    AVG(incidence)           AS mean_inc,
    AVG(emission)            AS mean_emi,
    AVG(phase)               AS mean_phase,
    MAX(iof)                 AS max_iof,
    MIN(iof)                 AS min_iof,
    STDDEV_SAMP(iof)         AS std_iof,
    COUNT(*) / COUNT(DISTINCT image_id) AS px_per_image,
    MAX(incidence)           AS max_inc,
    MAX(emission)            AS max_emi,
    MAX(phase)               AS max_phase,
    SUM(CASE WHEN iof > 0.55 THEN 1 ELSE 0 END)            AS n_above_055,
    SUM(CASE WHEN image_id NOT LIKE '%F1B%' THEN 1 ELSE 0 END) AS non_f1b_rows,
    SUM(CASE WHEN phase < ABS(incidence - emission)
              OR phase > incidence + emission + 0.1
             THEN 1 ELSE 0 END)                              AS tri_viol
  FROM read_parquet('{LAMO}')
""").fetchone()

(n_pixels, n_images, mean_iof, mean_inc, mean_emi, mean_phase,
 max_iof, min_iof, std_iof, px_per_image,
 max_inc, max_emi, max_phase,
 n_above_055, non_f1b, tri_viol) = r

print("=== STEP 3: CALIBRATION AND GEOMETRY ===")
print(f'{"n_pixels":26s}: {n_pixels:,}')
print(f'{"n_images":26s}: {n_images}')
print(f'{"mean_iof":26s}: {mean_iof:.6f}')
print(f'{"mean_inc (deg)":26s}: {mean_inc:.4f}')
print(f'{"mean_emi (deg)":26s}: {mean_emi:.4f}')
print(f'{"mean_phase (deg)":26s}: {mean_phase:.4f}')
print(f'{"max_iof":26s}: {max_iof:.6f}')
print(f'{"min_iof":26s}: {min_iof:.6f}')
print(f'{"std_iof":26s}: {std_iof:.6f}')
print(f'{"n_above_055":26s}: {n_above_055}')
print(f'{"non_f1b_rows":26s}: {non_f1b}')
print(f'{"triangle_violations":26s}: {tri_viol}')
print(f'{"frac>0.55":26s}: {n_above_055/n_pixels*100:.5f}%')
print(f'{"viol_frac":26s}: {tri_viol/n_pixels*100:.5f}%')

print("\n=== STEP 4: LAMO-SPECIFIC GEOMETRY ===")
print(f'{"px_per_image":26s}: {px_per_image:,.0f}')
print(f'{"global_std_iof":26s}: {std_iof:.6f}')
print(f'{"max_inc (deg)":26s}: {max_inc:.4f}')
print(f'{"max_emi (deg)":26s}: {max_emi:.4f}')
print(f'{"max_phase (deg)":26s}: {max_phase:.4f}')

print("\n=== Query 2: approximate percentiles (10M reservoir) ===", flush=True)
rp = con.execute(f"""
  SELECT
    PERCENTILE_CONT(0.99)  WITHIN GROUP (ORDER BY iof) AS p99_iof,
    PERCENTILE_CONT(0.999) WITHIN GROUP (ORDER BY iof) AS p999_iof
  FROM (SELECT iof FROM read_parquet('{LAMO}')
        USING SAMPLE reservoir(10000000 ROWS) REPEATABLE(42))
""").fetchone()
print(f'{"p99_iof  (sampled)":26s}: {rp[0]:.6f}')
print(f'{"p999_iof (sampled)":26s}: {rp[1]:.6f}')

print("\n=== PASS/FAIL ===")
print(f"non_f1b_rows == 0      : {'PASS' if non_f1b == 0 else 'FAIL'} ({non_f1b})")
print(f"mean_iof in [0.08,0.20]: {'PASS' if 0.08 <= mean_iof <= 0.20 else 'FAIL'} ({mean_iof:.4f})")
print(f"mean_iof >= 0.08 floor : {'PASS' if mean_iof >= 0.08 else 'FAIL -- check f_solar'}")
print(f"frac>0.55 < 0.01%      : {'PASS' if n_above_055/n_pixels*100 < 0.01 else 'FAIL'} ({n_above_055/n_pixels*100:.5f}%)")
print(f"tri_viol < 0.01%       : {'PASS' if tri_viol/n_pixels*100 < 0.01 else 'FAIL'} ({tri_viol/n_pixels*100:.5f}%)")
print(f"min_iof >= 0.0         : {'PASS' if min_iof >= 0 else 'FAIL'} ({min_iof:.6f})")
print(f"px/image vs HAMO(1047k): {'PASS lower' if px_per_image < 1047000 else 'FLAG higher'} ({px_per_image:,.0f})")
PYEOF