#!/bin/bash
set -x

rsync -av --checksum /scratch/kaushim07/vesta_data/calibrated_raw_images/ \
  /scratch/kaushim07/vesta_data/calibrated_raw_images/

rsync -av --checksum /scratch/kaushim07/vesta_data/spice_kernels/ \
  /scratch/kaushim07/vesta_data/spice_kernels/

rsync -av --checksum /scratch/kaushim07/vesta_data/geometry/dsk256/ \
  /scratch/kaushim07/vesta_data/geometry/dsk256/

rsync -av --checksum /scratch/kaushim07/vesta_data/geometry/ellipsoid/ \
  /scratch/kaushim07/vesta_data/geometry/ellipsoid/

rsync -av --checksum /scratch/kaushim07/vesta_data/silver/dsk256/ \
  /scratch/kaushim07/vesta_data/silver/dsk256/

rsync -av --checksum /scratch/kaushim07/vesta_data/golden/ \
  /scratch/kaushim07/vesta_data/golden/

echo "RENAME_RSYNC_ALL_DONE"
