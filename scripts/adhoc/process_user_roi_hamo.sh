#!/bin/bash
FILES=(
  "FC21B0007083_11273004846F1G"
  "FC21B0007091_11273005418F1G"
  "FC21B0009692_11281165927F1E"
  "FC21B0009693_11281170527F1E"
  "FC21B0009979_11284195537F1G"
  "FC21B0009980_11284195937F1G"
  "FC21B0010702_11291230325F1D"
  "FC21B0011211_11297020515F1F"
  "FC21B0011212_11297020915F1F"
  "FC21B0012903_11302045058F1D"
  "FC21B0027433_12177203415F1B"
  "FC21B0028580_12189154216F1D"
  "FC21B0029043_12194184507F1B"
  "FC21B0029044_12194185107F1B"
  "FC21B0029229_12196185354F1E"
)

echo "Total files to process: ${#FILES[@]}"

for name in "${FILES[@]}"; do
  echo "=== Processing $name ==="
  python -c "
from photometry_etl.etl.geometry_engine import GeometryEngine
engine = GeometryEngine(
    data_root='/scratch/kaushim07/vesta_data',
    metakernel_path='/scratch/kaushim07/vesta_data/spice_kernels/dawn_dynamic.tm',
    surface_intercept_method='DSK/UNPRIORITIZED',
    output_subdir='geometry/dsk256',
    f_solar=892.0,
)
result = engine.compute_geometry(
    '/scratch/kaushim07/vesta_data/calibrated_raw_images/hamo_user_roi/${name}.FIT'
)
print(f'pixels={len(result)}')
"
done
echo "=== Done: 15 files processed ==="
